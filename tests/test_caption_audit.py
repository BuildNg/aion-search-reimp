import json

import pandas as pd
import pytest

from aion_reimp.caption_audit import (
    audit_metrics,
    audit_rows,
    paired_accuracy_delta,
    paired_path_score_delta,
    path_audit_metrics,
    path_audit_rows,
)
from aion_reimp.captioning import append_caption_results, parse_caption_response
from aion_reimp.morphology import (
    GalaxyDecisionTree,
    SCHEMA_JSON,
    build_decision_tree_path,
    calibration_metrics,
    parse_morphology_response,
)


def test_text_extraction_and_direct_human_scoring() -> None:
    parsed = parse_morphology_response(
        "g1",
        json.dumps(
            {
                "overall_shape": "smooth",
                "roundness": "round",
                "merging": "none",
            }
        ),
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


def test_released_schema_rejects_out_of_vocabulary_value() -> None:
    response = json.dumps(
        {
            "overall_shape": "featured-or-disk",
            "edge_on": "edge-on-no",
            "bulge_size": "prominent",
        }
    )
    with pytest.raises(ValueError, match="schema-constrained"):
        parse_morphology_response("g1", response)
    assert "prominent" not in SCHEMA_JSON


def test_extractor_does_not_broadly_repair_invalid_json() -> None:
    with pytest.raises(ValueError, match="schema-constrained"):
        parse_morphology_response("g1", "{'overall_shape': invalid}")


def test_decision_path_matches_released_branch_logic() -> None:
    tree = GalaxyDecisionTree(
        overall_shape="featured-or-disk",
        edge_on="edge-on-no",
        has_spiral_arms="has-spiral-arms-yes",
        spiral_winding="tight",
        spiral_arm_count="2",
        bar="strong",
        bulge_size="small",
        merging="none",
    )
    assert build_decision_tree_path(tree) == [
        "smooth-or-featured_featured-or-disk",
        "disk-edge-on_no",
        "has-spiral-arms_yes",
        "spiral-winding_tight",
        "spiral-arm-count_2",
        "bar_strong",
        "bulge-size_small",
        "merging_none",
    ]


def test_released_path_overlap_uses_human_path_denominator() -> None:
    captions = pd.DataFrame(
        {
            "object_id": ["g1"],
            "judge_path_json": [
                json.dumps(["smooth-or-featured_smooth", "how-rounded_round"])
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "object_id": ["g1"],
            "decision_tree": [
                json.dumps(
                    [
                        {"node": "smooth-or-featured_smooth"},
                        {"node": "how-rounded_round"},
                        {"node": "merging_none"},
                    ]
                )
            ],
        }
    )
    rows = path_audit_rows(captions, labels)
    assert rows.iloc[0]["score"] == pytest.approx(2 / 3)
    assert path_audit_metrics(rows, bootstrap_samples=20, seed=1)[
        "mean_score"
    ] == pytest.approx(2 / 3)


def test_calibration_persists_raw_response_before_parse_failure(tmp_path) -> None:
    class _InvalidExtractor:
        def generate_response_with_metadata(self, description):
            return "{'overall_shape': invalid}", {"decoded_with_special_tokens": "raw"}

    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        json.dumps(
            {
                "object_id": "cal-1",
                "description": "A smooth galaxy.",
                "expected": {"smooth-or-featured": "smooth"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    errors = tmp_path / "errors.jsonl"
    metrics = calibration_metrics(
        _InvalidExtractor(),
        calibration,
        response_jsonl=responses,
        error_jsonl=errors,
        continue_on_error=True,
    )
    assert metrics["parse_errors"] == 1
    assert json.loads(responses.read_text(encoding="utf-8"))["raw_response"] == "{'overall_shape': invalid}"
    saved_error = json.loads(errors.read_text(encoding="utf-8"))
    assert saved_error["object_id"] == "cal-1"
    assert saved_error["generation_metadata"]["decoded_with_special_tokens"] == "raw"


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


def test_scaled_captioning_preserves_verbose_responses(tmp_path) -> None:
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
    )
    assert metrics["completed_rows"] == 1
    assert metrics["error_rows"] == 0
    saved = json.loads(output.read_text(encoding="utf-8").strip())
    assert saved["word_count"] == 301
    assert len(saved["description"].split()) == 301
    assert not errors.exists() or not errors.read_text(encoding="utf-8").strip()


def test_freeform_caption_preserves_content() -> None:
    response = "A detailed galaxy description.\nA second sentence."
    parsed = parse_caption_response("g1", response)
    assert parsed.description == response
    assert parsed.word_count == 7


def test_freeform_caption_does_not_enforce_prompt_word_guidance() -> None:
    parsed = parse_caption_response("g1", "word " * 301)
    assert parsed.word_count == 301
    assert len(parsed.description.split()) == 301


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


def test_paired_path_delta_is_object_paired() -> None:
    human = json.dumps(["smooth-or-featured_smooth"])
    qwen = pd.DataFrame(
        {"object_id": ["a", "b"], "human_path_json": [human, human], "score": [0.0, 1.0]}
    )
    gpt = pd.DataFrame(
        {"object_id": ["a", "b"], "human_path_json": [human, human], "score": [1.0, 1.0]}
    )
    result = paired_path_score_delta(qwen, gpt, bootstrap_samples=100, seed=3)
    assert result["point"] == pytest.approx(0.5)
    assert result["objects"] == 2
