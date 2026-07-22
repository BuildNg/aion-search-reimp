"""Galaxy Zoo DESI coverage checks for the locked paired population.

The public friendly catalog supplies predicted volunteer vote fractions.
This module only matches coordinates and counts decision-tree-aware,
high-confidence labels. It never reads model predictions or retrieval scores.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


GALAXY_ZOO_COLUMNS = (
    "dr8_id",
    "ra",
    "dec",
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "has-spiral-arms_yes_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
)

MORPHOLOGY_LABELS = (
    "smooth",
    "featured_or_disk",
    "spiral",
    "barred_spiral",
    "edge_on_disk",
)

MORPHOLOGY_RULES = {
    "smooth": "smooth-or-featured_smooth",
    "featured_or_disk": "smooth-or-featured_featured-or-disk",
    "spiral": "featured-or-disk AND disk-edge-on_no AND has-spiral-arms_yes",
    "barred_spiral": (
        "featured-or-disk AND disk-edge-on_no AND has-spiral-arms_yes "
        "AND (bar_strong + bar_weak)"
    ),
    "edge_on_disk": "featured-or-disk AND disk-edge-on_yes",
}


def _unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    ra_rad = np.deg2rad(np.asarray(ra, dtype=float))
    dec_rad = np.deg2rad(np.asarray(dec, dtype=float))
    cos_dec = np.cos(dec_rad)
    return np.column_stack(
        (cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad))
    )


def match_galaxy_zoo(
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Attach the nearest Galaxy Zoo row to each paired DESI coordinate."""
    required_manifest = {"object_id", "spectrum_ra", "spectrum_dec", "z"}
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise ValueError(f"Paired manifest missing columns: {sorted(missing_manifest)}")
    missing_catalog = set(GALAXY_ZOO_COLUMNS) - set(catalog)
    if missing_catalog:
        raise ValueError(f"Galaxy Zoo catalog missing columns: {sorted(missing_catalog)}")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    if manifest.empty or catalog.empty:
        raise ValueError("Coordinate matching requires non-empty inputs")

    manifest_coordinates = manifest[["spectrum_ra", "spectrum_dec"]].to_numpy(dtype=float)
    catalog_coordinates = catalog[["ra", "dec"]].to_numpy(dtype=float)
    if not np.isfinite(manifest_coordinates).all() or not np.isfinite(catalog_coordinates).all():
        raise ValueError("Coordinate matching requires finite coordinates")

    tree = cKDTree(_unit_vectors(catalog_coordinates[:, 0], catalog_coordinates[:, 1]))
    chord_distance, catalog_index = tree.query(
        _unit_vectors(manifest_coordinates[:, 0], manifest_coordinates[:, 1]), k=1
    )
    separation = np.rad2deg(
        2.0 * np.arcsin(np.clip(chord_distance / 2.0, 0.0, 1.0))
    ) * 3600.0
    matched = separation <= float(radius_arcsec)
    left = manifest.loc[matched, ["object_id", "z", "spectrum_ra", "spectrum_dec"]].reset_index(
        drop=True
    )
    right = catalog.iloc[catalog_index[matched]].reset_index(drop=True).copy()
    right = right.rename(
        columns={"dr8_id": "galaxy_zoo_dr8_id", "ra": "galaxy_zoo_ra", "dec": "galaxy_zoo_dec"}
    )
    joined = pd.concat([left, right], axis=1)
    joined.insert(4, "separation_arcsec", separation[matched])
    joined["object_id"] = joined["object_id"].astype(str)
    joined["galaxy_zoo_dr8_id"] = joined["galaxy_zoo_dr8_id"].astype(str)
    return joined.sort_values("object_id").reset_index(drop=True)


def add_reliable_morphology_labels(
    matches: pd.DataFrame,
    *,
    fraction_threshold: float,
) -> pd.DataFrame:
    """Add path-gated high-confidence morphology flags."""
    missing = set(GALAXY_ZOO_COLUMNS[3:]) - set(matches)
    if missing:
        raise ValueError(f"Matched labels missing vote fractions: {sorted(missing)}")
    if not 0.5 < fraction_threshold <= 1.0:
        raise ValueError("fraction_threshold must be in (0.5, 1.0]")

    frame = matches.copy()
    threshold = float(fraction_threshold)
    smooth = frame["smooth-or-featured_smooth_fraction"].ge(threshold)
    featured = frame["smooth-or-featured_featured-or-disk_fraction"].ge(threshold)
    face_on = featured & frame["disk-edge-on_no_fraction"].ge(threshold)
    spiral = face_on & frame["has-spiral-arms_yes_fraction"].ge(threshold)
    barred = frame[["bar_strong_fraction", "bar_weak_fraction"]].sum(
        axis=1, min_count=2
    ).ge(threshold)
    frame["reliable_smooth"] = smooth
    frame["reliable_featured_or_disk"] = featured
    frame["reliable_spiral"] = spiral
    frame["reliable_barred_spiral"] = spiral & barred
    frame["reliable_edge_on_disk"] = (
        featured & frame["disk-edge-on_yes_fraction"].ge(threshold)
    )
    return frame


def morphology_coverage_tables(
    labelled_matches: pd.DataFrame,
    *,
    total_pairs: int,
    fraction_threshold: float,
    redshift_bin_edges: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Summarize overall label support and support within redshift bins."""
    if total_pairs < len(labelled_matches):
        raise ValueError("total_pairs cannot be smaller than matched rows")
    edges = np.asarray(redshift_bin_edges, dtype=float)
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("redshift_bin_edges must be finite and strictly increasing")
    required = {"object_id", "galaxy_zoo_dr8_id", "separation_arcsec", "z"}
    required.update(f"reliable_{label}" for label in MORPHOLOGY_LABELS)
    missing = required - set(labelled_matches)
    if missing:
        raise ValueError(f"Labelled matches missing columns: {sorted(missing)}")

    count_rows = []
    for label in MORPHOLOGY_LABELS:
        positive = int(labelled_matches[f"reliable_{label}"].sum())
        count_rows.append(
            {
                "morphology": label,
                "positive_objects": positive,
                "fraction_of_matches": positive / len(labelled_matches) if len(labelled_matches) else 0.0,
                "fraction_of_all_pairs": positive / total_pairs if total_pairs else 0.0,
            }
        )
    counts = pd.DataFrame(count_rows)

    bins = pd.cut(
        labelled_matches["z"].astype(float),
        bins=edges,
        right=False,
        include_lowest=True,
    )
    crosstab_rows = []
    for interval in bins.cat.categories:
        in_bin = bins.eq(interval)
        for label in MORPHOLOGY_LABELS:
            crosstab_rows.append(
                {
                    "redshift_bin": str(interval),
                    "z_low": float(interval.left),
                    "z_high": float(interval.right),
                    "matched_objects": int(in_bin.sum()),
                    "morphology": label,
                    "positive_objects": int(
                        labelled_matches.loc[in_bin, f"reliable_{label}"].sum()
                    ),
                }
            )
    crosstab = pd.DataFrame(crosstab_rows)
    outside_bins = int(bins.isna().sum())
    duplicate_catalog_ids = int(labelled_matches["galaxy_zoo_dr8_id"].duplicated().sum())
    separations = labelled_matches["separation_arcsec"].to_numpy(dtype=float)
    summary = {
        "total_pairs": int(total_pairs),
        "matched_pairs": int(len(labelled_matches)),
        "coverage_fraction": float(len(labelled_matches) / total_pairs) if total_pairs else 0.0,
        "fraction_threshold": float(fraction_threshold),
        "duplicate_galaxy_zoo_assignments": duplicate_catalog_ids,
        "objects_outside_redshift_bins": outside_bins,
        "separation_arcsec": (
            {
                "median": float(np.median(separations)),
                "p95": float(np.quantile(separations, 0.95)),
                "max": float(np.max(separations)),
            }
            if len(separations)
            else None
        ),
    }
    return counts, crosstab, summary
