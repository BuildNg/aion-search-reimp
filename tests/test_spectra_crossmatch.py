from pathlib import Path

import pandas as pd
import pytest

from spectra_crossmatch.config import CrossmatchConfigError, load_config, validate_config
from spectra_crossmatch.crossmatch import (
    annotate_candidates,
    normalize_lsdb_matches,
    select_nearest_valid,
    source_fingerprint,
    summarize_matches,
)
from spectra_crossmatch.source import normalize_source_metadata, select_source_population


ROOT = Path(__file__).resolve().parents[1]


def test_crossmatch_config_is_strict_and_pinned() -> None:
    config = load_config(ROOT / "configs" / "phase6_crossmatch.yaml")
    assert config["source_population"]["sample_size"] == 18_000
    assert config["source_population"]["survey_value"] == "hsc_pdr3_wide"
    assert config["source_population"]["anchor"]["expected_survey_rows"] == 3_602
    assert config["desi_catalog"]["revision"] == "9fd88ba48233cb9857701ce802a7eade2d4c4a88"
    assert config["crossmatch"]["radii_arcsec"] == [0.5, 1.0, 2.0, 3.0]
    assert config["quality"]["zwarn_good_value"] is False
    broken = dict(config)
    broken["unused"] = True
    with pytest.raises(CrossmatchConfigError, match="Unknown top-level"):
        validate_config(broken)

    targeted = load_config(ROOT / "configs" / "phase6_crossmatch_morphology.yaml")
    assert targeted["source_population"]["sample_size"] == 25_000
    assert targeted["source_population"]["anchor"]["expected_survey_rows"] == 18_000
    assert (
        targeted["source_population"]["morphology_priority"][
            "reliable_fraction_threshold"
        ]
        == 0.7
    )


def test_source_selection_is_exact_deterministic_and_keeps_anchor() -> None:
    raw = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "survey": ["hsc", "hsc", "hsc", "hsc", "legacy"],
            "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
            "dec": [-1.0, 2.0, 3.0, 4.0, 5.0],
            "row": [7, 8, 9, 10, 11],
        }
    )
    metadata = normalize_source_metadata(
        raw,
        columns={
            "object_id": "id",
            "survey": "survey",
            "ra": "ra",
            "dec": "dec",
            "source_row_id": "row",
        },
    )
    selected = select_source_population(
        metadata,
        survey="hsc",
        sample_size=3,
        seed=17,
        salt="scale-v1",
        excluded_object_ids={"d"},
        anchor_object_ids={"b"},
    )
    repeated = select_source_population(
        metadata.iloc[::-1],
        survey="hsc",
        sample_size=3,
        seed=17,
        salt="scale-v1",
        excluded_object_ids={"d"},
        anchor_object_ids={"b"},
    )
    assert len(selected) == 3
    assert "b" in set(selected["source_object_id"])
    assert "d" not in set(selected["source_object_id"])
    assert set(selected["source_survey"]) == {"hsc"}
    assert source_fingerprint(selected) == source_fingerprint(repeated)


def test_source_selection_takes_ordered_priorities_before_hash_fill() -> None:
    raw = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "survey": ["hsc"] * 6,
            "ra": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "dec": [0.0] * 6,
            "row": list(range(6)),
        }
    )
    metadata = normalize_source_metadata(
        raw,
        columns={
            "object_id": "id",
            "survey": "survey",
            "ra": "ra",
            "dec": "dec",
            "source_row_id": "row",
        },
    )
    selected = select_source_population(
        metadata,
        survey="hsc",
        sample_size=4,
        seed=17,
        salt="targeted-v1",
        excluded_object_ids=set(),
        anchor_object_ids={"a"},
        priority_object_ids=["d", "c"],
        anchor_selection_reason="anchor_18k",
    )
    by_id = selected.set_index("source_object_id")
    assert {"a", "c", "d"}.issubset(set(selected["source_object_id"]))
    assert by_id.loc["a", "selection_reason"] == "anchor_18k"
    assert by_id.loc["c", "selection_reason"] == "morphology_priority"
    assert by_id.loc["d", "selection_reason"] == "morphology_priority"
    assert selected["selection_reason"].eq("deterministic_hsc_expansion").sum() == 1

    capped = select_source_population(
        metadata,
        survey="hsc",
        sample_size=2,
        seed=17,
        salt="targeted-v1",
        excluded_object_ids=set(),
        anchor_object_ids={"a"},
        priority_object_ids=["d", "c"],
    )
    assert set(capped["source_object_id"]) == {"a", "d"}


def test_normalize_lsdb_matches_requires_explicit_suffix_contract() -> None:
    class NestedLikeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return NestedLikeFrame

    raw = NestedLikeFrame(
        {
            "source_object_id_source": ["a"],
            "source_survey_source": ["north"],
            "source_ra_source": [10.0],
            "source_dec_source": [1.0],
            "source_row_id_source": [4],
            "selection_reason_source": ["test"],
            "selection_rank_source": [0],
            "object_id_desi": ["100"],
            "ra_desi": [10.0],
            "dec_desi": [1.0],
            "Z_desi": [0.2],
            "ZERR_desi": [0.001],
            "ZWARN_desi": [False],
            "_dist_arcsec": [0.05],
        }
    )
    normalized = normalize_lsdb_matches(
        raw,
        source_columns={
            "object_id": "source_object_id",
            "survey": "source_survey",
            "ra": "source_ra",
            "dec": "source_dec",
            "source_row_id": "source_row_id",
            "selection_reason": "selection_reason",
            "selection_rank": "selection_rank",
        },
        desi_columns={
            "object_id": "object_id",
            "ra": "ra",
            "dec": "dec",
            "redshift": "Z",
            "redshift_error": "ZERR",
            "zwarn": "ZWARN",
        },
    )
    assert type(normalized) is pd.DataFrame
    assert normalized.loc[0, "desi_object_id"] == "100"
    assert normalized.loc[0, "separation_arcsec"] == 0.05


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_object_id": ["a", "a", "a", "b", "c"],
            "source_survey": ["north", "north", "north", "south", "south"],
            "source_ra": [10.0, 10.0, 10.0, 20.0, 30.0],
            "source_dec": [0.0, 0.0, 0.0, 1.0, 2.0],
            "source_row_id": [1, 1, 1, 2, 3],
            "selection_reason": ["test"] * 5,
            "selection_rank": [0, 0, 0, 1, 2],
            "desi_object_id": ["bad", "200", "100", "300", "400"],
            "desi_ra": [10.0, 10.0, 10.0, 20.0, 30.0],
            "desi_dec": [0.0, 0.0, 0.0, 1.0, 2.0],
            "desi_z": [0.2, 0.3, 0.4, 0.5, -0.1],
            "desi_zerr": [0.01, 0.01, 0.01, 0.02, 0.01],
            "desi_zwarn": [True, False, False, False, False],
            "separation_arcsec": [0.2, 0.4, 0.4, 0.8, 2.5],
        }
    )


def test_quality_duplicate_ranking_and_radius_summaries() -> None:
    annotated = annotate_candidates(
        _candidate_frame(),
        zwarn_good_value=False,
        minimum_redshift=0.0,
        require_positive_redshift_error=True,
    )
    assert annotated.loc[annotated["desi_object_id"] == "bad", "quality_exclusion_reason"].item() == "zwarn_not_zero"
    assert annotated.loc[annotated["desi_object_id"] == "400", "quality_exclusion_reason"].item() == "redshift_below_minimum"

    selected = select_nearest_valid(annotated, 1.0)
    assert selected.set_index("source_object_id").loc["a", "desi_object_id"] == "100"
    assert set(selected["source_object_id"]) == {"a", "b"}

    source = pd.DataFrame(
        {
            "source_object_id": ["a", "b", "c"],
            "source_survey": ["north", "south", "south"],
            "source_ra": [10.0, 20.0, 30.0],
            "source_dec": [0.0, 1.0, 2.0],
            "source_row_id": [1, 2, 3],
            "selection_reason": ["test"] * 3,
            "selection_rank": [0, 1, 2],
        }
    )
    by_radius, by_survey, summary = summarize_matches(
        source, annotated, [0.5, 1.0, 3.0], primary_radius_arcsec=1.0
    )
    half_arcsec = by_radius.set_index("radius_arcsec").loc[0.5]
    assert half_arcsec["matched_any_objects"] == 1
    assert half_arcsec["matched_valid_objects"] == 1
    assert half_arcsec["ambiguous_valid_objects"] == 1
    one_arcsec = by_radius.set_index("radius_arcsec").loc[1.0]
    assert one_arcsec["matched_valid_objects"] == 2
    assert summary["valid_candidate_rows_within_max_radius"] == 3
    assert summary["primary_radius_arcsec"] == 1.0
    assert set(by_survey["source_survey"]) == {"north", "south"}
