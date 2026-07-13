import json

import pandas as pd
import pytest

from aion_reimp.caption_audit import audit_metrics, audit_rows
from aion_reimp.captioning import ANSWER_VALUES, parse_caption_response


def _answers() -> dict:
    result = {key: "not-applicable" for key in ANSWER_VALUES}
    result.update(
        {
            "smooth-or-featured": "smooth",
            "how-rounded": "round",
            "merging": "none",
        }
    )
    return result


def test_caption_schema_and_direct_human_scoring() -> None:
    response = json.dumps({"summary": "A round smooth galaxy is visible.", "answers": _answers()})
    parsed = parse_caption_response("g1", response)
    captions = pd.DataFrame([parsed.as_record()])
    labels = pd.DataFrame(
        {
            "object_id": ["g1"],
            "decision_tree": [
                json.dumps(
                    [
                        {"question": "smooth-or-featured", "answer": "smooth"},
                        {"question": "how-rounded", "answer": "round"},
                        {"question": "merging", "answer": "none"},
                    ]
                )
            ],
        }
    )
    rows = audit_rows(captions, labels)
    metrics = audit_metrics(rows, bootstrap_samples=20, seed=1)
    assert metrics["accuracy"] == 1.0
    assert metrics["objects"] == 1


def test_caption_rejects_extra_keys() -> None:
    response = json.dumps({"summary": "A galaxy.", "answers": _answers(), "explanation": "extra"})
    with pytest.raises(ValueError, match="only summary and answers"):
        parse_caption_response("g1", response)


def test_merging_rejects_not_applicable() -> None:
    answers = _answers()
    answers["merging"] = "not-applicable"
    response = json.dumps({"summary": "A galaxy is visible.", "answers": answers})
    with pytest.raises(ValueError, match="Invalid merging answer"):
        parse_caption_response("g1", response)
