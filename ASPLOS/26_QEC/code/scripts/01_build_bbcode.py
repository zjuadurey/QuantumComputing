from __future__ import annotations

import json
import os
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
import sympy as sp
from qldpc import codes
from qldpc.objects import Pauli


def get_code_specs() -> list[dict[str, object]]:
    x, y = sp.symbols("x y")
    return [
        {
            "code_name": "bb_n72_k12",
            "artifact_stem": "bbcode_n72_k12",
            "orders": (6, 6),
            "poly_a": x**3 + y + y**2,
            "poly_b": y**3 + x + x**2,
            "known_distance": 6,
            "distance_note": (
                "Distance 6 recorded from qldpc quantum_test.py comment for the "
                "[[72, 12, 6]] BB code example."
            ),
        },
        {
            "code_name": "bb_n96_k10",
            "artifact_stem": "bbcode_n96_k10",
            "orders": (12, 4),
            "poly_a": 1 + y + x * y + x**9,
            "poly_b": 1 + x**2 + x**7 + x**9 * y**2,
            "known_distance": None,
            "distance_note": "Exact distance not populated by qldpc for this instance in the current flow.",
        },
        {
            "code_name": "bb_n144_k12",
            "artifact_stem": "bbcode_n144_k12",
            "orders": (12, 6),
            "poly_a": x**3 + y + y**2,
            "poly_b": y**3 + x + x**2,
            "known_distance": 12,
            "distance_note": (
                "Distance 12 recorded from qldpc quantum_test.py comment for the "
                "[[144, 12, 12]] BB code example."
            ),
        },
        {
            "code_name": "bb_n64_k10",
            "artifact_stem": "bbcode_n64_k10",
            "orders": (8, 4),
            "poly_a": 1 + y + x * y + x**5,
            "poly_b": 1 + x**2 + x**3 + x**5 * y**2,
            "known_distance": None,
            "distance_note": (
                "Small BBCode in the same construction style as the qldpc Table-II-style example; "
                "exact distance not populated by qldpc in the current flow."
            ),
        },
        {
            "code_name": "bb_n80_k6",
            "artifact_stem": "bbcode_n80_k6",
            "orders": (8, 5),
            "poly_a": 1 + y + x * y + x**5,
            "poly_b": 1 + x**2 + x**3 + x**5 * y**2,
            "known_distance": None,
            "distance_note": (
                "Small BBCode in the same construction style as the qldpc Table-II-style example; "
                "exact distance not populated by qldpc in the current flow."
            ),
        },
        {
            "code_name": "bb_n108_k4",
            "artifact_stem": "bbcode_n108_k4",
            "orders": (9, 6),
            "poly_a": 1 + y + x * y + x**6,
            "poly_b": 1 + x**2 + x**4 + x**6 * y**2,
            "known_distance": None,
            "distance_note": (
                "Small BBCode in the same construction style as the qldpc Table-II-style example; "
                "exact distance not populated by qldpc in the current flow."
            ),
        },
    ]


def save_code_artifact(output_dir: Path, spec: dict[str, object]) -> dict[str, object]:
    code_name = str(spec["code_name"])
    artifact_stem = str(spec["artifact_stem"])
    orders = tuple(int(value) for value in spec["orders"])
    poly_a = spec["poly_a"]
    poly_b = spec["poly_b"]

    code = codes.BBCode(orders, poly_a, poly_b)

    hx = np.asarray(code.matrix_x, dtype=np.uint8)
    hz = np.asarray(code.matrix_z, dtype=np.uint8)
    logical_x = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    logical_z = np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8)

    n = int(code.num_qubits)
    k = int(code.dimension)
    mx_rows, mx_cols = hx.shape
    mz_rows, mz_cols = hz.shape
    check_weight = int(code.get_weight())
    runtime_distance = code.get_distance_if_known()
    distance = runtime_distance if runtime_distance is not None else spec["known_distance"]
    css_orthogonal = bool(np.all((hx @ hz.T) % 2 == 0))

    code_npz = output_dir / f"{artifact_stem}.npz"
    np.savez_compressed(
        code_npz,
        code_name=np.array(code_name),
        hx=hx,
        hz=hz,
        logical_x=logical_x,
        logical_z=logical_z,
        n=np.array([n], dtype=np.int64),
        k=np.array([k], dtype=np.int64),
        d=np.array([np.nan if distance is None else float(distance)], dtype=np.float64),
        check_weight=np.array([check_weight], dtype=np.int64),
        orders=np.array(orders, dtype=np.int64),
        mx_rows=np.array([mx_rows], dtype=np.int64),
        mx_cols=np.array([mx_cols], dtype=np.int64),
        mz_rows=np.array([mz_rows], dtype=np.int64),
        mz_cols=np.array([mz_cols], dtype=np.int64),
        poly_a=np.array(str(poly_a)),
        poly_b=np.array(str(poly_b)),
    )

    metadata = {
        "code_name": code_name,
        "code_type": "BBCode",
        "source": "qldpc.codes.BBCode parameters aligned with qldpc test examples",
        "orders": list(orders),
        "poly_a": str(poly_a),
        "poly_b": str(poly_b),
        "n": n,
        "k": k,
        "d": None if distance is None else int(distance),
        "distance_note": str(spec["distance_note"]),
        "matrix_x_shape": [mx_rows, mx_cols],
        "matrix_z_shape": [mz_rows, mz_cols],
        "logical_x_shape": list(logical_x.shape),
        "logical_z_shape": list(logical_z.shape),
        "max_check_weight": check_weight,
        "css_orthogonal": css_orthogonal,
        "artifact_file": code_npz.name,
        "metadata_file": f"{artifact_stem}.json",
    }

    metadata_path = output_dir / f"{artifact_stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "codes"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in get_code_specs():
        metadata = save_code_artifact(output_dir, spec)
        manifest.append(metadata)
        print(
            f"Built {metadata['code_name']}: "
            f"n={metadata['n']} k={metadata['k']} d={metadata['d']} "
            f"Hx={tuple(metadata['matrix_x_shape'])} Hz={tuple(metadata['matrix_z_shape'])}"
        )

    manifest_path = output_dir / "bbcode_manifest.json"
    manifest_path.write_text(json.dumps({"codes": manifest}, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
