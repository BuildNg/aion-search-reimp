import numpy as np
import pandas as pd

from spec_probes.morphology_coverage import (
    GALAXY_ZOO_COLUMNS,
    add_reliable_morphology_labels,
    match_galaxy_zoo,
    morphology_coverage_tables,
)


def _catalog_row(dr8_id, ra, *, smooth, featured, edge_yes, edge_no, spiral, strong, weak):
    return {
        "dr8_id": dr8_id,
        "ra": ra,
        "dec": 0.0,
        "smooth-or-featured_smooth_fraction": smooth,
        "smooth-or-featured_featured-or-disk_fraction": featured,
        "disk-edge-on_yes_fraction": edge_yes,
        "disk-edge-on_no_fraction": edge_no,
        "has-spiral-arms_yes_fraction": spiral,
        "bar_strong_fraction": strong,
        "bar_weak_fraction": weak,
    }


def test_galaxy_zoo_match_and_path_gated_labels():
    manifest = pd.DataFrame(
        {
            "object_id": ["smooth", "barred", "edge", "unmatched"],
            "spectrum_ra": [10.0, 20.0, 30.0, 40.0],
            "spectrum_dec": [0.0, 0.0, 0.0, 0.0],
            "z": [0.04, 0.12, 0.22, 0.4],
        }
    )
    catalog = pd.DataFrame(
        [
            _catalog_row(
                "gz-smooth", 10.0 + 0.2 / 3600,
                smooth=0.9, featured=0.1, edge_yes=np.nan, edge_no=np.nan,
                spiral=np.nan, strong=np.nan, weak=np.nan,
            ),
            _catalog_row(
                "gz-barred", 20.0 + 0.3 / 3600,
                smooth=0.05, featured=0.95, edge_yes=0.05, edge_no=0.95,
                spiral=0.9, strong=0.5, weak=0.4,
            ),
            _catalog_row(
                "gz-edge", 30.0 + 0.4 / 3600,
                smooth=0.05, featured=0.95, edge_yes=0.9, edge_no=0.1,
                spiral=0.99, strong=0.99, weak=0.0,
            ),
        ],
        columns=GALAXY_ZOO_COLUMNS,
    )
    matched = match_galaxy_zoo(manifest, catalog, radius_arcsec=1.0)
    labelled = add_reliable_morphology_labels(matched, fraction_threshold=0.8)

    assert labelled["object_id"].tolist() == ["barred", "edge", "smooth"]
    by_id = labelled.set_index("object_id")
    assert bool(by_id.loc["smooth", "reliable_smooth"])
    assert bool(by_id.loc["barred", "reliable_barred_spiral"])
    assert bool(by_id.loc["edge", "reliable_edge_on_disk"])
    # A high child vote is not enough when the face-on parent path fails.
    assert not bool(by_id.loc["edge", "reliable_spiral"])

    counts, crosstab, summary = morphology_coverage_tables(
        labelled,
        paired_redshifts=manifest["z"],
        fraction_threshold=0.8,
        redshift_bin_edges=[0.0, 0.1, 0.2, 0.3, 1.0],
    )
    positive = counts.set_index("morphology")["positive_objects"].to_dict()
    assert positive["smooth"] == 1
    assert positive["barred_spiral"] == 1
    assert positive["edge_on_disk"] == 1
    assert summary["matched_pairs"] == 3
    assert summary["coverage_fraction"] == 0.75
    first_bin = crosstab.loc[crosstab["redshift_bin"].eq("[0.0, 0.1)")].iloc[0]
    assert first_bin["total_paired_objects"] == 1
    assert first_bin["matched_objects"] == 1
    assert first_bin["match_fraction"] == 1.0
    last_bin = crosstab.loc[crosstab["redshift_bin"].eq("[0.3, 1.0)")].iloc[0]
    assert last_bin["total_paired_objects"] == 1
    assert last_bin["matched_objects"] == 0
    assert last_bin["match_fraction"] == 0.0
    assert crosstab.loc[
        crosstab["morphology"].eq("barred_spiral"), "positive_objects"
    ].sum() == 1


def test_reliable_labels_reject_missing_columns_and_weak_parent():
    row = pd.DataFrame(
        [
            _catalog_row(
                "gz", 1.0,
                smooth=0.1, featured=0.7, edge_yes=0.0, edge_no=0.99,
                spiral=0.99, strong=0.99, weak=0.0,
            )
        ]
    )
    labelled = add_reliable_morphology_labels(row, fraction_threshold=0.8)
    assert not bool(labelled.loc[0, "reliable_spiral"])
    assert not bool(labelled.loc[0, "reliable_barred_spiral"])


def test_coverage_rejects_duplicate_galaxy_zoo_assignments():
    matches = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "galaxy_zoo_dr8_id": ["same", "same"],
            "separation_arcsec": [0.1, 0.2],
            "z": [0.1, 0.2],
            **{
                f"reliable_{label}": [False, False]
                for label in ("smooth", "featured_or_disk", "spiral", "barred_spiral", "edge_on_disk")
            },
        }
    )
    with np.testing.assert_raises_regex(ValueError, "not one-to-one"):
        morphology_coverage_tables(
            matches,
            paired_redshifts=[0.1, 0.2],
            fraction_threshold=0.8,
            redshift_bin_edges=[0.0, 1.0],
        )
