import pandas as pd
import pytest

from aion_reimp.manifest import (
    assert_no_benchmark_leakage,
    assert_exclusion_coverage,
    build_manifest,
    coordinate_exclusion_coverage,
    coordinate_exclusion_table,
    exact_exclusion_coverage,
    manifest_fingerprint,
    split_fraction,
)
from aion_reimp.smoke import select_smoke_rows


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "survey": ["legacy", "legacy", "hsc", "hsc"],
            "ra": [1.0, 2.0, 3.0, 4.0],
            "dec": [-1.0, -2.0, -3.0, -4.0],
            "source_row_id": [0, 1, 2, 3],
        }
    )


def test_sha_split_is_deterministic() -> None:
    assert split_fraction("galaxy-1", 42) == split_fraction("galaxy-1", 42)
    assert split_fraction("galaxy-1", 42) != split_fraction("galaxy-1", 43)


def test_exclusions_never_enter_train_or_validation() -> None:
    frame = build_manifest(
        _source(),
        {"caption_screen_64": ["b"], "retrieval_benchmark": ["d"]},
        seed=42,
    )
    excluded = frame.set_index("object_id").loc[["b", "d"]]
    assert set(excluded["split"]) == {"excluded"}
    assert_no_benchmark_leakage(frame)


def test_leakage_assertion_fails_loudly() -> None:
    frame = build_manifest(_source(), {"benchmark": ["b"]}, seed=42)
    frame.loc[frame["object_id"] == "b", "split"] = "train"
    with pytest.raises(AssertionError, match="Benchmark leakage"):
        assert_no_benchmark_leakage(frame)


def test_manifest_fingerprint_is_row_order_invariant() -> None:
    frame = build_manifest(_source(), {}, seed=42)
    shuffled = frame.sample(frac=1.0, random_state=1)
    assert manifest_fingerprint(frame) == manifest_fingerprint(shuffled)


def test_coordinate_exclusions_use_angular_radius() -> None:
    source = _source()
    benchmark = pd.DataFrame({"ra": [2.0 + 0.5 / 3600.0], "dec": [-2.0]})
    matches = coordinate_exclusion_table(source, {"caption_or_retrieval": benchmark}, radius_arcsec=1.0)
    assert matches["object_id"].tolist() == ["b"]
    assert matches.loc[0, "separation_arcsec"] < 1.0


def test_exclusion_coverage_names_matches_and_absences() -> None:
    benchmark = pd.DataFrame(
        {
            "benchmark_row": [10, 11],
            "ra": [2.0 + 0.5 / 3600.0, 100.0],
            "dec": [-2.0, 20.0],
        }
    )
    coordinate = coordinate_exclusion_coverage(
        _source(), {"retrieval": benchmark}, radius_arcsec=1.0
    )
    exact = exact_exclusion_coverage(_source()["object_id"], "caption_screen_64", ["a", "z"])
    coverage = pd.concat([coordinate, exact], ignore_index=True)
    assert_exclusion_coverage(coverage, expected_rows=4)
    assert coverage["status"].tolist() == ["matched", "absent", "matched", "absent"]
    assert coordinate.loc[0, "source_object_id"] == "b"


def test_smoke_selection_is_seeded_and_excludes_benchmarks() -> None:
    source = pd.concat([_source()] * 300, ignore_index=True)
    source["object_id"] = [f"object-{index}" for index in range(len(source))]
    source["source_row_id"] = range(len(source))
    manifest = build_manifest(source, {"benchmark": ["object-7"]}, seed=9)
    first = select_smoke_rows(manifest, sample_size=1000, seed=9)
    second = select_smoke_rows(manifest.sample(frac=1.0, random_state=2), 1000, seed=9)
    assert first["object_id"].tolist() == second["object_id"].tolist()
    assert "object-7" not in set(first["object_id"])
