#!/usr/bin/env python
"""Extract all experiment data from paper CSV files into exp_fig/data/
as both .csv (for plotting) and .tex (for LaTeX input).

Generates one pair of files per figure.
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data/results"
OUT = REPO / "paper/exp_fig/data"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(filename, rows, fieldnames):
    with open(OUT / filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_tex_table(filename, caption, label, headers, rows, colfmts, notes=None):
    """Write a self-contained LaTeX table fragment."""
    lines = []
    lines.append("% Auto-extracted experiment data")
    lines.append("% Source: data/results/")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\footnotesize")
    lines.append(f"\\begin{{tabular}}{{{colfmts}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if notes:
        for n in notes:
            lines.append(n)
    lines.append("\\end{table}")
    lines.append("")

    with open(OUT / filename, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# Fig 01: Main per-condition paired delta LFR
# ============================================================
with open(RESULTS / "p7_google_rl_qec_decoder_effect_matrix.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = []
tex_rows = []
for r in sorted(rows, key=lambda r: (int(r["cycles"]), r["basis"], r["control_mode"])):
    cyc = int(r["cycles"])
    basis = r["basis"]
    ctrl = r["control_mode"].replace("reinforcement_learning", "RL").replace("traditional_calibration", "TC")
    bf = int(r["baseline_failure_count"])
    res = int(r["rescued_failure_count"])
    ind = int(r["induced_failure_count"])
    delta = float(r["paired_delta_lfr"])
    csv_rows.append({"cycles": cyc, "basis": basis, "control_mode": ctrl,
                     "baseline_failure_count": bf, "rescued": res, "induced": ind, "delta_lfr": delta})
    tex_rows.append([str(cyc), basis, ctrl, str(bf), str(res), str(ind), f"${delta:+.4f}$"])

write_csv("fig01_main_per_condition.csv", csv_rows,
          ["cycles", "basis", "control_mode", "baseline_failure_count", "rescued", "induced", "delta_lfr"])
write_tex_table("fig01_main_per_condition.tex",
    "Per-condition paired delta LFR for the main decoder-pathway intervention across 40 real-data conditions.",
    "tab:fig01_main_per_condition",
    ["Cyc", "Basis", "Ctrl", "BaseFail", "Rescued", "Induced", "$\\Delta$LFR"],
    tex_rows, "rrrrrrr")
print("Fig 01 done")

# ============================================================
# Fig 02: Paired vs unpaired bootstrap variance
# ============================================================
with open(RESULTS / "p7_5_paired_vs_unpaired_variance.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = []
tex_rows = []
for r in rows:
    cyc = int(r["cycles"])
    basis = r["basis"]
    ctrl = r["control_mode"].replace("reinforcement_learning", "RL").replace("traditional_calibration", "TC")
    paired = float(r["paired_bootstrap_std"])
    unpaired = float(r["unpaired_bootstrap_std"])
    ratio = float(r["std_ratio_unpaired_over_paired"])
    csv_rows.append({"cycles": cyc, "basis": basis, "control_mode": ctrl,
                     "paired_std": paired, "unpaired_std": unpaired, "ratio": ratio})
    tex_rows.append([str(cyc), basis, ctrl, f"{paired:.5f}", f"{unpaired:.5f}", f"{ratio:.2f}"])

write_csv("fig02_paired_vs_unpaired_std.csv", csv_rows,
          ["cycles", "basis", "control_mode", "paired_std", "unpaired_std", "ratio"])
write_tex_table("fig02_paired_vs_unpaired_std.tex",
    "Per-condition paired and unpaired bootstrap standard deviation across 40 conditions.",
    "tab:fig02_paired_vs_unpaired",
    ["Cyc", "Basis", "Ctrl", "Paired $\\sigma$", "Unpaired $\\sigma$", "Ratio"],
    tex_rows, "rrrrrr")
print("Fig 02 done")

# ============================================================
# Fig 03: Runtime deadline threshold
# ============================================================
with open(RESULTS / "p10_runtime_deadline_summary.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = []
tex_rows = []
for r in rows:
    deadline = float(r["deadline_us"])
    miss = float(r["deadline_miss_rate"])
    base_lfr = float(r["baseline_lfr"])
    intv_lfr = float(r["intervened_lfr"])
    delta = float(r["paired_delta_lfr"])
    rescued = int(r["rescued_failure_count"])
    induced = int(r["induced_failure_count"])
    csv_rows.append({"deadline_us": deadline, "miss_rate": miss, "baseline_lfr": base_lfr,
                     "intervened_lfr": intv_lfr, "paired_delta_lfr": delta,
                     "rescued": rescued, "induced": induced})
    tex_rows.append([f"{deadline:.0f}\\,$\\mu$s", f"{miss:.3f}", f"{base_lfr:.3f}",
                     f"{intv_lfr:.3f}", f"{delta:+.3f}", str(rescued), str(induced)])

write_csv("fig03_runtime_deadline.csv", csv_rows,
          ["deadline_us", "miss_rate", "baseline_lfr", "intervened_lfr", "paired_delta_lfr", "rescued", "induced"])
write_tex_table("fig03_runtime_deadline.tex",
    "Runtime deadline intervention: paired delta LFR, miss rate, and rescued/induced counts.",
    "tab:fig03_runtime",
    ["Deadline", "Miss rate", "Base LFR", "Intv LFR", "$\\Delta$LFR", "Rescued", "Induced"],
    tex_rows, "lrrrrrr")
print("Fig 03 done")

# ============================================================
# Fig 04: qec3v5 external replication by cycle depth
# ============================================================
with open(RESULTS / "p10_qec3v5_external_decoder_effect_matrix.csv") as f:
    rows = list(csv.DictReader(f))

from collections import defaultdict
by_cyc = defaultdict(lambda: {"deltas": [], "rescued": 0, "induced": 0, "base_fail": 0, "n": 0})
for r in rows:
    cyc = int(r["cycles"])
    by_cyc[cyc]["deltas"].append(float(r["paired_delta_lfr"]))
    by_cyc[cyc]["rescued"] += int(r["rescued_failure_count"])
    by_cyc[cyc]["induced"] += int(r["induced_failure_count"])
    by_cyc[cyc]["base_fail"] += int(r["baseline_failure_count"])
    by_cyc[cyc]["n"] += 1

import statistics
csv_rows = []
tex_rows = []
for cyc in sorted(by_cyc.keys()):
    d = by_cyc[cyc]
    md = statistics.mean(d["deltas"])
    rf = d["rescued"] / d["base_fail"] if d["base_fail"] > 0 else 0
    csv_rows.append({"cycles": cyc, "mean_delta_lfr": md, "rescued": d["rescued"],
                     "induced": d["induced"], "base_fail": d["base_fail"], "rescued_frac": rf, "n_conditions": d["n"]})
    tex_rows.append([str(cyc), f"${md:+.4f}$", str(d["rescued"]), str(d["induced"]),
                     str(d["base_fail"]), f"{rf:.3f}"])

write_csv("fig04_qec3v5_by_cycles.csv", csv_rows,
          ["cycles", "mean_delta_lfr", "rescued", "induced", "base_fail", "rescued_frac", "n_conditions"])
write_tex_table("fig04_qec3v5_by_cycles.tex",
    "qec3v5 external replication: per-cycle paired delta LFR, rescued/induced counts, and rescued fraction.",
    "tab:fig04_qec3v5",
    ["$r$", "Mean $\\Delta$LFR", "Rescued", "Induced", "BaseFail", "RescFrac"],
    tex_rows, "rrrrrr")
print("Fig 04 done")

# ============================================================
# Fig 05: v2 expanded evidence by subgroup
# ============================================================
with open(RESULTS / "p10_google_rl_qec_v2_decoder_effect_matrix.csv") as f:
    rows = list(csv.DictReader(f))

by_sg = defaultdict(lambda: {"deltas": [], "n": 0, "rescued": 0, "induced": 0})
for r in rows:
    sg = f"{r['code_family']}|{r['control_mode']}|{r['basis']}"
    by_sg[sg]["deltas"].append(float(r["paired_delta_lfr"]))
    by_sg[sg]["n"] += 1
    by_sg[sg]["rescued"] += int(r["rescued_failure_count"])
    by_sg[sg]["induced"] += int(r["induced_failure_count"])

code_map = {"surface_code_memory": "Surface (d3/d5/d7)", "color_code_memory": "Color (d5)"}
ctrl_map = {"traditional_calibration": "TC", "traditional_calibration_and_rl_fine_tuning": "TC+RL"}

csv_rows = []
tex_rows = []
for sg in sorted(by_sg.keys()):
    cf, cm, basis = sg.split("|")
    d = by_sg[sg]
    md = statistics.mean(d["deltas"])
    csv_rows.append({"code_family": cf, "control_mode": cm, "basis": basis,
                     "n_conditions": d["n"], "mean_delta_lfr": md,
                     "rescued": d["rescued"], "induced": d["induced"]})
    tex_rows.append([code_map.get(cf, cf), ctrl_map.get(cm, cm), basis,
                     str(d["n"]), f"${md:+.4f}$"])

# Overall
all_deltas = [float(r["paired_delta_lfr"]) for r in rows]
all_r = sum(int(r["rescued_failure_count"]) for r in rows)
all_i = sum(int(r["induced_failure_count"]) for r in rows)
csv_rows.append({"code_family": "All", "control_mode": "---", "basis": "---",
                 "n_conditions": len(all_deltas), "mean_delta_lfr": statistics.mean(all_deltas),
                 "rescued": all_r, "induced": all_i})
tex_rows.append(["All v2 conditions", "---", "---", str(len(all_deltas)), f"${statistics.mean(all_deltas):+.4f}$"])

write_csv("fig05_v2_subgroups.csv", csv_rows,
          ["code_family", "control_mode", "basis", "n_conditions", "mean_delta_lfr", "rescued", "induced"])
write_tex_table("fig05_v2_subgroups.tex",
    "Google RL QEC v2 expanded evidence: subgroup mean paired delta LFR.",
    "tab:fig05_v2_subgroups",
    ["Code Family", "Ctrl", "Basis", "Cond", "Mean $\\Delta$LFR"],
    tex_rows, "lllrr")
print("Fig 05 done")

# ============================================================
# Fig 06: Prior corpus aggregate
# ============================================================
with open(RESULTS / "p10_google_decoder_priors_prior_effects_aggregate.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = []
tex_rows = []
for r in rows:
    csv_rows.append({"intervened_prior": r["intervened_prior"],
                     "n_conditions": int(r["num_conditions"]),
                     "mean_delta_lfr": float(r["mean_paired_delta_lfr"]),
                     "min_delta": float(r["min_paired_delta_lfr"]),
                     "max_delta": float(r["max_paired_delta_lfr"]),
                     "neg_frac": float(r["negative_delta_fraction"])})
    tex_rows.append([r["intervened_prior"].replace("_", "\\_"), r["num_conditions"],
                     f"${float(r['mean_paired_delta_lfr']):+.4f}$",
                     f"{float(r['negative_delta_fraction']):.3f}",
                     f"${float(r['min_paired_delta_lfr']):+.4f}$",
                     f"${float(r['max_paired_delta_lfr']):+.4f}$"])

write_csv("fig06_prior_corpus.csv", csv_rows,
          ["intervened_prior", "n_conditions", "mean_delta_lfr", "min_delta", "max_delta", "neg_frac"])
write_tex_table("fig06_prior_corpus.tex",
    "Decoder-prior corpus study: aggregate across 5,724 prior variants.",
    "tab:fig06_prior_corpus",
    ["Intervened Prior", "Cond", "Mean $\\Delta$LFR", "NegFrac", "Min $\\Delta$", "Max $\\Delta$"],
    tex_rows, "lrrrrr")
print("Fig 06 done")

# ============================================================
# Fig 07: Post-selection (per-shot, surface Z r010)
# ============================================================
with open(RESULTS / "p10_postselection_per_shot.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = []
tex_rows = []
for r in rows:
    csv_rows.append({"threshold": r["threshold"],
                     "burden_cutoff": int(r["burden_cutoff"]),
                     "discarded_rescued_pct": float(r["discarded_rescued_pct"]),
                     "discarded_unchanged_pct": float(r["discarded_unchanged_pct"]),
                     "discarded_induced_pct": float(r["discarded_induced_pct"]),
                     "rescued_fraction_before": float(r["rescued_fraction_before"]),
                     "rescued_fraction_after": float(r["rescued_fraction_after"])})
    tex_rows.append([r["threshold"], r["burden_cutoff"],
                     f"{r['discarded_rescued_pct']}\\%",
                     f"{r['discarded_unchanged_pct']}\\%",
                     f"{r['discarded_induced_pct']}\\%",
                     r["rescued_fraction_before"],
                     r["rescued_fraction_after"]])

write_csv("fig07_postselection.csv", csv_rows,
          ["threshold", "burden_cutoff", "discarded_rescued_pct", "discarded_unchanged_pct",
           "discarded_induced_pct", "rescued_fraction_before", "rescued_fraction_after"])
write_tex_table("fig07_postselection.tex",
    "Post-selection via detector-burden threshold on surface\\_Z\\_r010 (10,000 paired shots).",
    "tab:fig07_postselection",
    ["Threshold", "Cutoff", "Disc. R", "Disc. UF", "Disc. I", "RF before", "RF after"],
    tex_rows, "lrrrrrr")
print("Fig 07 done")

# ============================================================
# Fig 08: PSM / unpaired comparison
# ============================================================
with open(RESULTS / "p10_psm_comparison.csv") as f:
    rows = list(csv.DictReader(f))

csv_rows = [{"metric": r["metric"], "value": r["value"]} for r in rows]
tex_rows = [[r["metric"], r["value"]] for r in rows]

write_csv("fig08_psm_comparison.csv", csv_rows, ["metric", "value"])
write_tex_table("fig08_psm_comparison.tex",
    "Unpaired versus FailureOps paired estimation on the 40-condition dataset.",
    "tab:fig08_psm",
    ["Property", "Value"],
    tex_rows, "ll")
print("Fig 08 done")

# ============================================================
# Fig 09: Heldout split verification
# ============================================================
with open(RESULTS / "p10_heldout_split_verification.csv") as f:
    rows = list(csv.DictReader(f))

# Summary table
n_total = len(rows)
n_sign = sum(1 for r in rows if r["sign_consistent"] == "True")
calib_deltas = [float(r["calibration_paired_delta_lfr"]) for r in rows]
eval_deltas = [float(r["evaluation_paired_delta_lfr"]) for r in rows]
full_deltas = [float(r["full_paired_delta_lfr"]) for r in rows]

csv_rows = [
    {"metric": "total_conditions", "value": str(n_total)},
    {"metric": "sign_consistent_50_50_split", "value": f"{n_sign}/{n_total}"},
    {"metric": "mean_full_delta_lfr", "value": f"{statistics.mean(full_deltas):.5f}"},
    {"metric": "mean_calibration_half_delta_lfr", "value": f"{statistics.mean(calib_deltas):.5f}"},
    {"metric": "mean_evaluation_half_delta_lfr", "value": f"{statistics.mean(eval_deltas):.5f}"},
]
tex_rows = [[r["metric"].replace("_", "\\_"), r["value"]] for r in csv_rows]

write_csv("fig09_heldout_split.csv", csv_rows, ["metric", "value"])
write_tex_table("fig09_heldout_split.tex",
    "Held-out split verification: 50/50 random split sign consistency across 40 conditions.",
    "tab:fig09_heldout",
    ["Metric", "Value"],
    tex_rows, "ll")
print("Fig 09 done")

print(f"\nAll files written to {OUT}")
