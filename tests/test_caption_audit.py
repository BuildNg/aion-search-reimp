import json

import pandas as pd
import pytest

from aion_reimp.caption_audit import audit_metrics, audit_rows, paired_accuracy_delta
from aion_reimp.captioning import (
    append_caption_results,
    parse_caption_response,
    truncate_caption_response,
)
from aion_reimp.morphology import ANSWER_VALUES, parse_morphology_response


def _extraction_payload(description: str) -> dict:
    answers = {
        key: {"value": "not-applicable", "evidence": None} for key in ANSWER_VALUES
    }
    answers.update(
        {
            "smooth-or-featured": {"value": "smooth", "evidence": "smooth"},
            "how-rounded": {"value": "round", "evidence": "round"},
            "merging": {"value": "none", "evidence": "no disturbance"},
        }
    )
    assert all(
        item["evidence"] is None or item["evidence"] in description
        for item in answers.values()
    )
    return {"answers": answers}


def test_text_extraction_and_direct_human_scoring() -> None:
    description = "The central galaxy is smooth and round with no disturbance."
    parsed = parse_morphology_response(
        "g1", description, json.dumps(_extraction_payload(description))
    )
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


def test_extractor_rejects_evidence_absent_from_caption() -> None:
    description = "The galaxy is smooth."
    payload = _extraction_payload("smooth round no disturbance")
    payload["answers"]["how-rounded"]["evidence"] = "round"
    with pytest.raises(ValueError, match="absent from the caption"):
        parse_morphology_response("g1", description, json.dumps(payload))


def test_not_stated_counts_as_abstention() -> None:
    labels = pd.DataFrame(
        {
            "object_id": ["g1"],
            "decision_tree": [
                json.dumps([{"question": "smooth-or-featured", "answer": "smooth"}])
            ],
        }
    )
    captions = pd.DataFrame(
        {
            "object_id": ["g1"],
            "answers_json": [json.dumps({"smooth-or-featured": "not-stated"})],
        }
    )
    rows = audit_rows(captions, labels)
    assert bool(rows.iloc[0]["abstained"]) is True


class _FakeCaptioner:
    def generate_response(self, image_path):
        if str(image_path).endswith("bad.png"):
            return ""
        return "A round smooth galaxy."


def test_scaled_captioning_logs_validation_errors_and_respects_cap(tmp_path) -> None:
    rows = [
        {"object_id": "good", "image_path": tmp_path / "good.png"},
        {"object_id": "bad", "image_path": tmp_path / "bad.png"},
    ]
    output = tmp_path / "captions.jsonl"
    errors = tmp_path / "errors.jsonl"
    metrics = append_caption_results(
        _FakeCaptioner(), rows, output, error_jsonl=errors, max_error_rate=0.5
    )
    assert metrics["completed_rows"] == 1
    assert metrics["error_rows"] == 1
    saved = json.loads(output.read_text(encoding="utf-8").strip())
    assert saved["description"] == "A round smooth galaxy."
    error = json.loads(errors.read_text(encoding="utf-8").strip())
    assert error["object_id"] == "bad"


def test_scaled_captioning_can_apply_preregistered_word_fallback(tmp_path) -> None:
    class _VerboseCaptioner:
        def generate_response(self, image_path):
            return "word " * 301

    output = tmp_path / "captions.jsonl"
    errors = tmp_path / "errors.jsonl"
    metrics = append_caption_results(
        _VerboseCaptioner(),
        [{"object_id": "verbose", "image_path": tmp_path / "image.png"}],
        output,
        error_jsonl=errors,
        max_error_rate=0.0,
        truncate_over_limit=True,
    )
    assert metrics["completed_rows"] == 1
    assert metrics["error_rows"] == 0
    saved = json.loads(output.read_text(encoding="utf-8").strip())
    assert saved["word_count"] == 300
    assert saved["compliance_failure"] is True
    error = json.loads(errors.read_text(encoding="utf-8").strip())
    assert error["compliance_failure"] is True


def test_freeform_caption_preserves_content() -> None:
    response = "A detailed galaxy description.\nA second sentence."
    parsed = parse_caption_response("g1", response)
    assert parsed.description == response
    assert parsed.word_count == 7


def test_freeform_caption_enforces_paper_word_limit() -> None:
    with pytest.raises(ValueError, match="exceeds 300 words"):
        parse_caption_response("g1", "word " * 301)


def test_preregistered_truncation_preserves_raw_response_and_marks_failure() -> None:
    response = " ".join(f"word-{index}" for index in range(304))
    parsed = truncate_caption_response("g1", response, max_words=300)
    record = parsed.as_record()
    assert len(record["description"].split()) == 300
    assert record["raw_response"] == response
    assert record["original_word_count"] == 304
    assert record["compliance_failure"] is True
    assert record["fallback"] == "truncate_original_to_300_words"


def test_paired_object_cluster_delta_uses_identical_rows() -> None:
    base = pd.DataFrame(
        {
            "object_id": ["a", "a", "b"],
            "question": ["q1", "q2", "q1"],
            "gold": ["x", "y", "x"],
        }
    )
    qwen = base.assign(correct=[False, False, True])
    gpt = base.assign(correct=[True, False, True])
    result = paired_accuracy_delta(qwen, gpt, bootstrap_samples=100, seed=4)
    assert result["point"] == pytest.approx(1 / 3)
    assert result["objects"] == 2
    assert result["answers"] == 3
    assert result["ci95"][0] <= result["point"] <= result["ci95"][1]


def test_paired_delta_rejects_mismatched_question_sets() -> None:
    qwen = pd.DataFrame(
        {"object_id": ["a"], "question": ["q1"], "gold": ["x"], "correct": [True]}
    )
    gpt = pd.DataFrame(
        {"object_id": ["a"], "question": ["q2"], "gold": ["x"], "correct": [True]}
    )
    with pytest.raises(ValueError, match="identical object-question"):
        paired_accuracy_delta(qwen, gpt, bootstrap_samples=10, seed=1)
