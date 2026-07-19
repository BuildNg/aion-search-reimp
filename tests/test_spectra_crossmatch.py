from pathlib import Path

import pandas as pd
import pytest

from spectra_crossmatch.config import CrossmatchConfigError, load_config, validate_config
from spectra_crossmatch.crossmatch import (
    annotate_candidates,
    normalize_lsdb_matches,
    prepare_captioned_source,
    select_nearest_valid,
    source_fingerprint,
    summarize_matches,
)


ROOT = Path(__file__).resolve().parents[1]


def test_crossmatch_config_is_strict_and_pinned() -> None:
    config = load_config(ROOT / "configs" / "phase6_crossmatch.yaml")
    assert config["captioned_source"]["expected_rows"] == 10_000
    assert config["desi_catalog"]["revision"] == "9fd88ba48233cb9857701ce802a7eade2d4c4a88"
    assert config["crossmatch"]["radii_arcsec"] == [0.5, 1.0, 2.0, 3.0]
    broken = dict(config)
    broken["unused"] = True
    with pytest.raises(CrossmatchConfigError, match="Unknown top-level"):
        validate_config(broken)


def test_prepare_captioned_source_preserves_manifest_order_and_exact_population() -> None:
    manifest = pd.DataFrame({"object_id": ["b", "a"]})
    source = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "survey": ["south", "north"],
            "ra": [10.0, 20.0],
            "dec": [-1.0, 2.0],
            "source_row_id": [7, 8],
        }
    )
    prepared = prepare_captioned_source(
        manifest,
        source,
        expected_rows=2,
        object_id_column="object_id",
        survey_column="survey",
        ra_column="ra",
        dec_column="dec",
        source_row_id_column="source_row_id",
    )
    assert prepared["caption_object_id"].tolist() == ["b", "a"]
    assert prepared["caption_ra"].tolist() == [20.0, 10.0]
    assert source_fingerprint(prepared) == source_fingerprint(prepared.iloc[::-1])

    with pytest.raises(ValueError, match="manifest/source mismatch"):
        prepare_captioned_source(
            manifest,
            source.iloc[:1],
            expected_rows=2,
            object_id_column="object_id",
            survey_column="survey",
            ra_column="ra",
            dec_column="dec",
            source_row_id_column="source_row_id",
        )


def test_normalize_lsdb_matches_requires_explicit_suffix_contract() -> None:
    raw = pd.DataFrame(
        {
            "caption_object_id_caption": ["a"],
            "caption_survey_caption": ["north"],
            "caption_ra_caption": [10.0],
            "caption_dec_caption": [1.0],
            "caption_source_row_id_caption": [4],
            "object_id_desi": ["100"],
            "ra_desi": [10.0],
            "dec_desi": [1.0],
            "Z_desi": [0.2],
            "ZERR_desi": [0.001],
            "ZWARN_desi": [True],
            "_dist_arcsec": [0.05],
        }
    )
    normalized = normalize_lsdb_matches(
        raw,
        source_columns={
            "object_id": "caption_object_id",
            "survey": "caption_survey",
            "ra": "caption_ra",
            "dec": "caption_dec",
            "source_row_id": "caption_source_row_id",
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
    assert normalized.loc[0, "desi_object_id"] == "100"
    assert normalized.loc[0, "separation_arcsec"] == 0.05


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caption_object_id": ["a", "a", "a", "b", "c"],
            "caption_survey": ["north", "north", "north", "south", "south"],
            "caption_ra": [10.0, 10.0, 10.0, 20.0, 30.0],
            "caption_dec": [0.0, 0.0, 0.0, 1.0, 2.0],
            "caption_source_row_id": [1, 1, 1, 2, 3],
            "desi_object_id": ["bad", "200", "100", "300", "400"],
            "desi_ra": [10.0, 10.0, 10.0, 20.0, 30.0],
            "desi_dec": [0.0, 0.0, 0.0, 1.0, 2.0],
            "desi_z": [0.2, 0.3, 0.4, 0.5, -0.1],
            "desi_zerr": [0.01, 0.01, 0.01, 0.02, 0.01],
            "desi_zwarn_good": [False, True, True, True, True],
            "separation_arcsec": [0.2, 0.4, 0.4, 0.8, 2.5],
        }
    )


def test_quality_duplicate_ranking_and_radius_summaries() -> None:
    annotated = annotate_candidates(
        _candidate_frame(),
        zwarn_good_value=True,
        minimum_redshift=0.0,
        require_positive_redshift_error=True,
    )
    assert annotated.loc[annotated["desi_object_id"] == "bad", "quality_exclusion_reason"].item() == "zwarn_not_zero"
    assert annotated.loc[annotated["desi_object_id"] == "400", "quality_exclusion_reason"].item() == "redshift_below_minimum"

    selected = select_nearest_valid(annotated, 1.0)
    assert selected.set_index("caption_object_id").loc["a", "desi_object_id"] == "100"
    assert set(selected["caption_object_id"]) == {"a", "b"}

    source = pd.DataFrame(
        {
            "caption_object_id": ["a", "b", "c"],
            "caption_survey": ["north", "south", "south"],
            "caption_ra": [10.0, 20.0, 30.0],
            "caption_dec": [0.0, 1.0, 2.0],
            "caption_source_row_id": [1, 2, 3],
        }
    )
    by_radius, by_survey, summary = summarize_matches(source, annotated, [0.5, 1.0, 3.0])
    half_arcsec = by_radius.set_index("radius_arcsec").loc[0.5]
    assert half_arcsec["matched_any_objects"] == 1
    assert half_arcsec["matched_valid_objects"] == 1
    assert half_arcsec["ambiguous_valid_objects"] == 1
    one_arcsec = by_radius.set_index("radius_arcsec").loc[1.0]
    assert one_arcsec["matched_valid_objects"] == 2
    assert summary["valid_candidate_rows_within_max_radius"] == 3
    assert set(by_survey["caption_survey"]) == {"north", "south"}
