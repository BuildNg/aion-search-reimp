import pandas as pd

from aion_reimp.evaluate import check_reference_gate


def test_reference_gate_checks_paper_values_and_lens_count() -> None:
    metrics = {
        "spiral": {"ndcg@10": 0.9412},
        "merger": {"ndcg@10": 0.5537},
        "lens": {"ndcg@10": 0.1731},
    }
    ranked = pd.DataFrame(
        {
            "query_name": ["lens"] * 10,
            "rank": range(1, 11),
            "relevance": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    gate = {
        "published_rounded_targets": {"spiral": 0.941, "merger": 0.554, "lens": 0.173},
        "lens_top10_positives": 2,
    }

    checks, failures = check_reference_gate(metrics, ranked, gate)

    assert failures == []
    assert checks["lens_top10_confirmed"]["passed"] is True
