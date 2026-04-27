"""P3 intervention-rank stability metrics."""

from __future__ import annotations

import itertools
import json
import statistics
from collections import Counter, defaultdict

from failureops.io_utils import fmt_float
from failureops.metrics import summarize_attribution


def summarize_rank_stability(rows: list[dict[str, object]], *, top_k: int = 3) -> list[dict[str, object]]:
    per_run: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_run[(str(row["run_id"]), str(row["workload_id"]), str(row["stress_level"]))].append(row)

    rankings_by_group: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for (_run_id, workload_id, stress_level), group in per_run.items():
        summaries = summarize_attribution(group)
        ranking = [str(row["intervention"]) for row in sorted(summaries, key=lambda row: float(row["absolute_delta_lfr"]))]
        rankings_by_group[(workload_id, stress_level)].append(ranking)

    out = []
    for (workload_id, stress_level), rankings in rankings_by_group.items():
        top_items = [ranking[0] for ranking in rankings if ranking]
        mode, mode_count = Counter(top_items).most_common(1)[0] if top_items else ("none", 0)
        overlaps = []
        distances = []
        for left, right in itertools.combinations(rankings, 2):
            overlaps.append(top_k_overlap(left, right, top_k=top_k))
            distances.append(pairwise_rank_distance(left, right))
        consensus = consensus_ranking(rankings)
        out.append(
            {
                "workload_id": workload_id,
                "stress_level": stress_level,
                "num_groups": len(rankings),
                "mean_top3_overlap": fmt_float(statistics.fmean(overlaps) if overlaps else 1.0),
                "mean_pairwise_rank_distance": fmt_float(statistics.fmean(distances) if distances else 0.0),
                "top_intervention_mode": mode,
                "top_intervention_mode_fraction": fmt_float(mode_count / len(rankings) if rankings else 0.0),
                "ranked_interventions": json.dumps(consensus, separators=(",", ":")),
            }
        )
    out.sort(key=lambda row: (row["workload_id"], row["stress_level"]))
    return out


def top_k_overlap(left: list[str], right: list[str], *, top_k: int) -> float:
    left_set = set(left[:top_k])
    right_set = set(right[:top_k])
    return len(left_set & right_set) / max(1, top_k)


def pairwise_rank_distance(left: list[str], right: list[str]) -> float:
    items = list(dict.fromkeys([*left, *right]))
    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    max_rank = len(items)
    total = 0.0
    for item in items:
        total += abs(left_rank.get(item, max_rank) - right_rank.get(item, max_rank))
    return total / max(1, len(items))


def consensus_ranking(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, list[int]] = defaultdict(list)
    for ranking in rankings:
        for index, intervention in enumerate(ranking):
            scores[intervention].append(index)
    return [
        intervention
        for intervention, _score in sorted(
            ((intervention, statistics.fmean(values)) for intervention, values in scores.items()),
            key=lambda item: (item[1], item[0]),
        )
    ]

