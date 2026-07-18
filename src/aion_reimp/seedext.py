"""Combined base-run-plus-seed-extension summary statistics for Phase 3.

``configs/phase3_10k_seedext.yaml`` runs two more seeds (45, 57) on top of the
three a base run (``phase3_10k_v1``) already completed (13, 21, 33). The base
run's own summary and this run's own summary each only cover their own seeds.
:func:`build_combined_summary` pools the raw per-seed values from both into one
five-seed estimate, using :func:`aion_reimp.metrics.summary_statistics` exactly
once per metric over all five raw values, so the result is never an average of
two already-aggregated averages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .metrics import summary_statistics


# Each metric name maps to the key path used to reach its raw value inside a
# per-seed-per-condition gate dict (see scripts/run_phase3_10k_cluster.py and
# scripts/run_phase3_seedext_cluster.py, where these gates are built).
COMBINED_SUMMARY_METRICS: Dict[str, Sequence[str]] = {
    "validation_recall_at_10": ("validation_recall_at_10",),
    "spiral_ndcg_at_10": ("benchmark_metrics", "spiral", "ndcg@10"),
    "merger_ndcg_at_10": ("benchmark_metrics", "merger", "ndcg@10"),
    "lens_ndcg_at_10": ("benchmark_metrics", "lens", "ndcg@10"),
}


def _extract(gate: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = gate
    for key in path:
        value = value[key]
    return float(value)


def build_combined_summary(
    base_summary: Mapping[str, Any],
    base_run_id: str,
    reused_seeds: Sequence[int],
    extension_seed_condition_gates: Mapping[str, Mapping[str, Any]],
    extension_run_id: str,
    extension_seeds: Sequence[int],
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pool five seeds' raw per-seed metrics into one combined summary per condition.

    ``base_summary`` is the base run's already-loaded ``phase3_10k_summary.json``
    (its ``"condition_seed_gates"`` mapping, keyed ``"{condition}|seed={seed}"``,
    holds each reused seed's raw metrics). ``extension_seed_condition_gates`` is
    this run's own in-memory gates, built with the same key convention, for the
    seeds in ``extension_seeds``. Returns one entry per condition name with the
    five-seed ``seeds`` list, pooled ``summary_statistics`` per metric, and a
    ``per_seed`` table naming which run each seed's row came from.
    """
    base_gates = base_summary["condition_seed_gates"]
    combined: Dict[str, Any] = {}
    for condition in conditions:
        name = condition["name"]
        per_seed_rows: List[Dict[str, Any]] = []
        raw_values: Dict[str, List[float]] = {metric: [] for metric in COMBINED_SUMMARY_METRICS}

        def _collect(seed: int, gate: Mapping[str, Any], source_run_id: str) -> None:
            row: Dict[str, Any] = {"seed": seed, "source_run_id": source_run_id}
            for metric, path in COMBINED_SUMMARY_METRICS.items():
                value = _extract(gate, path)
                raw_values[metric].append(value)
                row[metric] = value
            per_seed_rows.append(row)

        for seed in reused_seeds:
            _collect(seed, base_gates[f"{name}|seed={seed}"], base_run_id)
        for seed in extension_seeds:
            _collect(seed, extension_seed_condition_gates[f"{name}|seed={seed}"], extension_run_id)

        combined[name] = {
            "seeds": [*reused_seeds, *extension_seeds],
            **{metric: summary_statistics(values) for metric, values in raw_values.items()},
            "per_seed": per_seed_rows,
        }
    return combined
