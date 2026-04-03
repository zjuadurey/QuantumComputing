from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
NUMBA_CACHE_DIR = CACHE_ROOT / "numba"
MPLCONFIGDIR = CACHE_ROOT / "matplotlib"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import pandas as pd
from ldpc import BpLsdDecoder, BpOsdDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated BB code code-capacity evaluations.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "codes" / "bbcode_manifest.json",
        help="Manifest emitted by 01_build_bbcode.py. Used when --code-file is not provided.",
    )
    parser.add_argument(
        "--code-file",
        type=Path,
        action="append",
        default=None,
        help="Optional path to one or more specific code artifacts. Overrides --manifest when set.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of Monte Carlo shots per repeat.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="Number of independent repeats per (code, decoder, p).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for repeat generation.",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.05],
        help="Physical Z error rates to sweep.",
    )
    return parser.parse_args()


def wilson_interval(num_failures: int, num_shots: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if num_shots <= 0:
        raise ValueError("num_shots must be positive")

    phat = num_failures / num_shots
    z2_over_n = (z * z) / num_shots
    denom = 1.0 + z2_over_n
    center = (phat + 0.5 * z2_over_n) / denom
    margin = (
        z
        * np.sqrt((phat * (1.0 - phat) + 0.25 * z2_over_n) / num_shots)
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1))


def build_decoder(decoder_name: str, hx: np.ndarray, p: float):
    channel = [p] * hx.shape[1]
    common_kwargs = {
        "error_channel": channel,
        "max_iter": 30,
        "bp_method": "minimum_sum",
        "ms_scaling_factor": 0.625,
        "schedule": "parallel",
    }
    if decoder_name == "BP-OSD":
        return BpOsdDecoder(
            hx,
            osd_method="osd_0",
            osd_order=0,
            **common_kwargs,
        )
    if decoder_name == "BP-LSD":
        return BpLsdDecoder(
            hx,
            lsd_method="lsd_0",
            lsd_order=0,
            bits_per_step=1,
            **common_kwargs,
        )
    raise ValueError(f"Unknown decoder: {decoder_name}")


def is_logical_failure(
    error: np.ndarray,
    correction: np.ndarray,
    hx: np.ndarray,
    logical_x: np.ndarray,
) -> bool:
    residual = (error ^ correction).astype(np.uint8, copy=False)

    if np.any((hx @ residual) % 2):
        return True

    logical_syndrome = (logical_x @ residual) % 2
    return bool(np.any(logical_syndrome))


def load_code_jobs(args: argparse.Namespace) -> list[Path]:
    if args.code_file:
        return args.code_file

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        return [
            args.manifest.parent / code_metadata["artifact_file"]
            for code_metadata in manifest["codes"]
        ]

    return sorted((PROJECT_ROOT / "data" / "codes").glob("*.npz"))


def load_code_artifact(code_file: Path) -> dict[str, object]:
    code_data = np.load(code_file)
    hx = code_data["hx"].astype(np.uint8, copy=False)
    hz = code_data["hz"].astype(np.uint8, copy=False)
    logical_x = code_data["logical_x"].astype(np.uint8, copy=False)

    raw_d = float(code_data["d"][0])
    d = None if np.isnan(raw_d) else int(raw_d)

    return {
        "code_file": code_file,
        "code_name": str(code_data["code_name"].item()),
        "hx": hx,
        "hz": hz,
        "logical_x": logical_x,
        "n": int(code_data["n"][0]),
        "k": int(code_data["k"][0]),
        "d": d,
        "mx_rows": int(code_data["mx_rows"][0]),
        "mx_cols": int(code_data["mx_cols"][0]),
        "mz_rows": int(code_data["mz_rows"][0]),
        "mz_cols": int(code_data["mz_cols"][0]),
        "mx_density": float(np.count_nonzero(hx) / hx.size),
        "mz_density": float(np.count_nonzero(hz) / hz.size),
        "mx_row_weight_mean": float(hx.sum(axis=1).mean()),
        "mz_row_weight_mean": float(hz.sum(axis=1).mean()),
    }


def make_repeat_seed(base_seed: int, code_index: int, p_index: int, repeat_id: int) -> int:
    return int(base_seed + code_index * 100_000 + p_index * 1_000 + repeat_id)


def evaluate_one_repeat(
    *,
    code_job: dict[str, object],
    decoder_name: str,
    p: float,
    num_shots: int,
    repeat_id: int,
    seed: int,
    errors: np.ndarray,
    syndromes: np.ndarray,
    syndrome_weights: np.ndarray,
) -> dict[str, object]:
    hx = code_job["hx"]
    logical_x = code_job["logical_x"]
    decoder = build_decoder(decoder_name, hx, p)
    decoder.decode(np.zeros(hx.shape[0], dtype=np.uint8))

    num_failures = 0
    latencies_ms: list[float] = []
    bp_iterations: list[int] = []
    bp_converged = 0

    for error, syndrome in zip(errors, syndromes, strict=True):
        start = time.perf_counter()
        correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
        latencies_ms.append((time.perf_counter() - start) * 1_000.0)
        bp_iterations.append(int(decoder.iter))
        bp_converged += int(bool(decoder.converge))

        if is_logical_failure(error, correction, hx, logical_x):
            num_failures += 1

    logical_error_rate = num_failures / num_shots
    ler_ci_low, ler_ci_high = wilson_interval(num_failures, num_shots)

    return {
        "code_name": str(code_job["code_name"]),
        "p": p,
        "decoder_name": decoder_name,
        "repeat_id": repeat_id,
        "seed": seed,
        "num_shots": num_shots,
        "num_failures": num_failures,
        "logical_error_rate": logical_error_rate,
        "ler_ci_low": ler_ci_low,
        "ler_ci_high": ler_ci_high,
        "avg_latency_ms_per_shot": float(np.mean(latencies_ms)),
        "p50_latency_ms_per_shot": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms_per_shot": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms_per_shot": float(np.percentile(latencies_ms, 99)),
        "avg_bp_iterations_per_shot": float(np.mean(bp_iterations)),
        "p95_bp_iterations_per_shot": float(np.percentile(bp_iterations, 95)),
        "bp_converged_fraction": float(bp_converged / num_shots),
        "syndrome_avg_weight": float(np.mean(syndrome_weights)),
        "syndrome_p95_weight": float(np.percentile(syndrome_weights, 95)),
        "syndrome_avg_relative_weight": float(np.mean(syndrome_weights) / code_job["mx_rows"]),
        "syndrome_p95_relative_weight": float(np.percentile(syndrome_weights, 95) / code_job["mx_rows"]),
        "noise_model": "independent_z_only",
        "n": int(code_job["n"]),
        "k": int(code_job["k"]),
        "d": code_job["d"],
        "mx_rows": int(code_job["mx_rows"]),
        "mx_cols": int(code_job["mx_cols"]),
        "mz_rows": int(code_job["mz_rows"]),
        "mz_cols": int(code_job["mz_cols"]),
        "mx_density": float(code_job["mx_density"]),
        "mz_density": float(code_job["mz_density"]),
        "mx_row_weight_mean": float(code_job["mx_row_weight_mean"]),
        "mz_row_weight_mean": float(code_job["mz_row_weight_mean"]),
        "code_file": Path(code_job["code_file"]).name,
        "ci_method": "wilson_95",
    }


def summarize_repeated_runs(repeated_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    group_cols = ["code_name", "decoder_name", "p"]

    for _, subdf in repeated_df.groupby(group_cols, sort=False):
        first = subdf.iloc[0]
        num_failures_total = int(subdf["num_failures"].sum())
        total_shots = int(subdf["num_shots"].sum())
        ler_ci_low, ler_ci_high = wilson_interval(num_failures_total, total_shots)
        ler_mean = float(subdf["logical_error_rate"].mean())

        summary_rows.append(
            {
                "code_name": first["code_name"],
                "decoder_name": first["decoder_name"],
                "p": float(first["p"]),
                "num_repeats": int(len(subdf)),
                "shots_per_repeat": int(first["num_shots"]),
                "num_shots": total_shots,
                "num_failures": num_failures_total,
                "num_failures_total": num_failures_total,
                "logical_error_rate": ler_mean,
                "ler_mean": ler_mean,
                "ler_std": safe_std(subdf["logical_error_rate"]),
                "ler_ci_low": ler_ci_low,
                "ler_ci_high": ler_ci_high,
                "avg_latency_ms_per_shot": float(subdf["avg_latency_ms_per_shot"].mean()),
                "avg_latency_mean": float(subdf["avg_latency_ms_per_shot"].mean()),
                "avg_latency_std": safe_std(subdf["avg_latency_ms_per_shot"]),
                "p50_latency_ms_per_shot": float(subdf["p50_latency_ms_per_shot"].mean()),
                "p50_latency_mean": float(subdf["p50_latency_ms_per_shot"].mean()),
                "p50_latency_std": safe_std(subdf["p50_latency_ms_per_shot"]),
                "p95_latency_ms_per_shot": float(subdf["p95_latency_ms_per_shot"].mean()),
                "p95_latency_mean": float(subdf["p95_latency_ms_per_shot"].mean()),
                "p95_latency_std": safe_std(subdf["p95_latency_ms_per_shot"]),
                "p99_latency_ms_per_shot": float(subdf["p99_latency_ms_per_shot"].mean()),
                "p99_latency_mean": float(subdf["p99_latency_ms_per_shot"].mean()),
                "p99_latency_std": safe_std(subdf["p99_latency_ms_per_shot"]),
                "avg_bp_iterations_mean": float(subdf["avg_bp_iterations_per_shot"].mean()),
                "avg_bp_iterations_std": safe_std(subdf["avg_bp_iterations_per_shot"]),
                "p95_bp_iterations_mean": float(subdf["p95_bp_iterations_per_shot"].mean()),
                "p95_bp_iterations_std": safe_std(subdf["p95_bp_iterations_per_shot"]),
                "bp_converged_fraction_mean": float(subdf["bp_converged_fraction"].mean()),
                "bp_converged_fraction_std": safe_std(subdf["bp_converged_fraction"]),
                "syndrome_avg_weight_mean": float(subdf["syndrome_avg_weight"].mean()),
                "syndrome_avg_weight_std": safe_std(subdf["syndrome_avg_weight"]),
                "syndrome_p95_weight_mean": float(subdf["syndrome_p95_weight"].mean()),
                "syndrome_p95_weight_std": safe_std(subdf["syndrome_p95_weight"]),
                "syndrome_avg_relative_weight_mean": float(subdf["syndrome_avg_relative_weight"].mean()),
                "syndrome_avg_relative_weight_std": safe_std(subdf["syndrome_avg_relative_weight"]),
                "syndrome_p95_relative_weight_mean": float(subdf["syndrome_p95_relative_weight"].mean()),
                "syndrome_p95_relative_weight_std": safe_std(subdf["syndrome_p95_relative_weight"]),
                "noise_model": first["noise_model"],
                "n": int(first["n"]),
                "k": int(first["k"]),
                "d": first["d"],
                "mx_rows": int(first["mx_rows"]),
                "mx_cols": int(first["mx_cols"]),
                "mz_rows": int(first["mz_rows"]),
                "mz_cols": int(first["mz_cols"]),
                "mx_density": float(first["mx_density"]),
                "mz_density": float(first["mz_density"]),
                "mx_row_weight_mean": float(first["mx_row_weight_mean"]),
                "mz_row_weight_mean": float(first["mz_row_weight_mean"]),
                "seed_base": int(subdf["seed"].min()),
                "code_file": first["code_file"],
                "ci_method": first["ci_method"],
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    code_jobs = [load_code_artifact(code_file) for code_file in load_code_jobs(args)]
    repeated_rows = []

    print("Running repeated code-capacity experiment")
    print(
        f"  num_codes={len(code_jobs)} num_repeats={args.num_repeats} "
        f"shots_per_repeat={args.shots} p_values={args.p_values}"
    )
    print("  noise_model=independent Z-only data errors")

    for code_index, code_job in enumerate(code_jobs):
        code_name = str(code_job["code_name"])
        code_file = Path(code_job["code_file"])
        print(
            f"Code {code_name}: n={code_job['n']} k={code_job['k']} d={code_job['d']} "
            f"mx_density={code_job['mx_density']:.4f} file={code_file.name}"
        )

        for p_index, p in enumerate(args.p_values):
            for repeat_id in range(args.num_repeats):
                repeat_seed = make_repeat_seed(args.seed, code_index, p_index, repeat_id)
                rng = np.random.default_rng(repeat_seed)
                errors = (rng.random((args.shots, int(code_job["n"]))) < p).astype(np.uint8)
                syndromes = ((errors @ code_job["hx"].T) % 2).astype(np.uint8)
                syndrome_weights = syndromes.sum(axis=1)

                for decoder_name in ("BP-OSD", "BP-LSD"):
                    row = evaluate_one_repeat(
                        code_job=code_job,
                        decoder_name=decoder_name,
                        p=p,
                        num_shots=args.shots,
                        repeat_id=repeat_id,
                        seed=repeat_seed,
                        errors=errors,
                        syndromes=syndromes,
                        syndrome_weights=syndrome_weights,
                    )
                    repeated_rows.append(row)
                    print(
                        f"  p={p:.3f} repeat={repeat_id} seed={repeat_seed} decoder={decoder_name:<6} "
                        f"LER={row['logical_error_rate']:.4f} "
                        f"avg_ms={row['avg_latency_ms_per_shot']:.3f} "
                        f"p95_ms={row['p95_latency_ms_per_shot']:.3f} "
                        f"avg_bp_iter={row['avg_bp_iterations_per_shot']:.2f}"
                    )

    repeated_df = pd.DataFrame(repeated_rows)
    summary_df = summarize_repeated_runs(repeated_df)

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    repeated_path = results_dir / "results_repeated.csv"
    summary_path = results_dir / "results_repeated_summary.csv"
    legacy_results_path = results_dir / "results.csv"

    repeated_df.to_csv(repeated_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(legacy_results_path, index=False)

    print(f"Saved repeated results to {repeated_path}")
    print(f"Saved repeated summary to {summary_path}")
    print(f"Saved compatibility summary to {legacy_results_path}")


if __name__ == "__main__":
    main()
