#!/usr/bin/env python
"""Final policy selection experiment: per-condition break-even frontier + cross-condition transfer."""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from failureops.io_utils import ensure_parent_dir, fmt_float, write_csv_rows

CM_MU, CM_SIGMA = 4.654, 0.499
MULTIPLIERS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
DEADLINES = [4, 5, 6, 7, 8]

BREAKEVEN_FIELDS = [
    "condition", "mode", "basis", "cycles",
    "baseline_lfr", "intervened_lfr", "paired_delta_lfr",
    "break_even_D4", "break_even_D5", "break_even_D6", "break_even_D7", "break_even_D8",
]
TRANSFER_FIELDS = [
    "train_mode", "test_mode", "deadline_us",
    "break_even_multiplier", "tess_lfr_at_breakeven", "cm_lfr_at_breakeven",
]

def norm_cdf(x, mu, sigma):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def main():
    conditions = []
    for mode in ["reinforcement_learning", "traditional_calibration"]:
        for basis in ["X", "Z"]:
            for cycles in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                path = Path(f"data/results/p7_per_shot/surface_code_{mode}_{basis}_r{cycles:03d}_decoder_interventions.csv")
                if not path.exists():
                    continue
                with open(path) as f:
                    rows = list(csv.DictReader(f))
                bl = sum(1 for r in rows if str(r.get("baseline_logical_failure", "")).strip() == "True")
                iv = sum(1 for r in rows if str(r.get("intervened_logical_failure", "")).strip() == "True")
                n = len(rows)
                conditions.append(dict(name=f"surface_code_{mode}_{basis}_r{cycles:03d}", mode=mode, basis=basis, cycles=cycles,
                                       baseline_lfr=bl/n, intervened_lfr=iv/n, paired_delta_lfr=(bl-iv)/n, n=n))

    breakeven_rows = []
    for c in sorted(conditions, key=lambda c: (0 if c["mode"]=="reinforcement_learning" else 1, c["basis"], c["cycles"])):
        row = dict(condition=c["name"], mode=c["mode"], basis=c["basis"], cycles=str(c["cycles"]),
                   baseline_lfr=fmt_float(c["baseline_lfr"]), intervened_lfr=fmt_float(c["intervened_lfr"]),
                   paired_delta_lfr=fmt_float(c["paired_delta_lfr"]))
        for D in DEADLINES:
            cm_on = norm_cdf(D, CM_MU, CM_SIGMA)
            raw = c["baseline_lfr"] + 0.02
            best_M = 1.0
            for M in MULTIPLIERS:
                tess_mu = M * CM_MU; tess_sigma = M * CM_SIGMA
                tess_on = norm_cdf(D, tess_mu, tess_sigma)
                both_late = (1-tess_on)*(1-cm_on)
                tess_eff = tess_on*c["intervened_lfr"] + (1-tess_on)*cm_on*c["baseline_lfr"] + both_late*raw
                cm_eff = cm_on*c["baseline_lfr"] + (1-cm_on)*raw
                if tess_eff < cm_eff - 1e-6:
                    best_M = M
            row[f"break_even_D{D}"] = fmt_float(best_M)
        breakeven_rows.append(row)

    out = Path("data/results/p10_policy_breakeven.csv")
    ensure_parent_dir(out)
    write_csv_rows(str(out), breakeven_rows, BREAKEVEN_FIELDS)
    print(f"Wrote {len(breakeven_rows)} rows to {out}")

    transfer_rows = []
    for train_mode, test_mode in [("reinforcement_learning","traditional_calibration"),("traditional_calibration","reinforcement_learning")]:
        test_c = [c for c in conditions if c["mode"]==test_mode]
        for D in DEADLINES:
            cm_on = norm_cdf(D, CM_MU, CM_SIGMA)
            best_M, best_tess, best_cm = 1.0, 0.0, 0.0
            for M in MULTIPLIERS:
                tess_mu = M*CM_MU; tess_sigma = M*CM_SIGMA
                tess_on = norm_cdf(D, tess_mu, tess_sigma)
                both_late = (1-tess_on)*(1-cm_on)
                ts, cs, ns = 0.0, 0.0, 0.0
                for c in test_c:
                    raw = c["baseline_lfr"]+0.02
                    ts += (tess_on*c["intervened_lfr"]+(1-tess_on)*cm_on*c["baseline_lfr"]+both_late*raw)*c["n"]
                    cs += (cm_on*c["baseline_lfr"]+(1-cm_on)*raw)*c["n"]
                    ns += c["n"]
                if ts/ns < cs/ns - 1e-6:
                    best_M, best_tess, best_cm = M, ts/ns, cs/ns
            transfer_rows.append(dict(train_mode=train_mode, test_mode=test_mode, deadline_us=str(D),
                                      break_even_multiplier=fmt_float(best_M),
                                      tess_lfr_at_breakeven=fmt_float(best_tess),
                                      cm_lfr_at_breakeven=fmt_float(best_cm)))
    out2 = Path("data/results/p10_policy_transfer_summary.csv")
    write_csv_rows(str(out2), transfer_rows, TRANSFER_FIELDS)
    print(f"Wrote {len(transfer_rows)} rows to {out2}")

    print("\n=== Break-even summary ===")
    for D in DEADLINES:
        vals = [float(c[f"break_even_D{D}"]) for c in breakeven_rows]
        print(f"  D={D}us: min={min(vals):.1f}, median={sorted(vals)[len(vals)//2]:.1f}, max={max(vals):.1f}")

    print("\n=== Cross-condition transfer ===")
    for r in transfer_rows:
        print(f"  Train={r['train_mode']}, Test={r['test_mode']}, D={r['deadline_us']}us: "
              f"break-even M={r['break_even_multiplier']}")

if __name__ == "__main__":
    main()
