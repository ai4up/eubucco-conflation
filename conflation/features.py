from collections import defaultdict

import momepy
import numpy as np
import pandas as pd
import geopandas as gpd

from conflation.geoutil import align_centroids, buffer_area


def compute_ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Calculate the Two-Way Area Overlap (TWAO) between building pairs. Also referred to as symmetrical Intersection over Area (IoA).
    """
    return geoms1.intersection(geoms2).area / np.minimum(geoms1.area, geoms2.area)


def compute_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the Intersection over Union (IoU) between pairs of geometries.
    """
    intersection = geoms1.intersection(geoms2).area
    union = geoms1.union(geoms2).area
    iou = intersection / union
    iou = iou.fillna(0)  # mitigate devision by zero

    return iou


def compute_aligned_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the Intersection over Union (IoU) after aligning the centroids of geometry pairs.
    """
    aligned_geoms2 = gpd.GeoSeries([align_centroids(g1, g2) for g1, g2 in zip(geoms1, geoms2)], index=geoms1.index, crs=geoms1.crs)
    aligned_iou = compute_iou(geoms1, aligned_geoms2)

    return aligned_iou


def compute_aligned_ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the Two-Way Area Overlap (TWAO) after aligning the centroids of geometry pairs.
    """
    aligned_geoms2 = gpd.GeoSeries([align_centroids(g1, g2) for g1, g2 in zip(geoms1, geoms2)], index=geoms1.index, crs=geoms1.crs)
    aligned_ioa = compute_ioa(geoms1, aligned_geoms2)

    return aligned_ioa


def calculate_car_similarity(fts: pd.DataFrame) -> pd.Series:
    """
    Calculate a simple similarity score indicating how closely a building footprint resembles a reference car-like shape.
    """
    ref_elong = 0.5
    ref_area = 15
    ref_conv = 1

    return (((fts["footprint_area"] - ref_area).abs() / ref_area +
            (fts["elongation"] - ref_elong).abs() / ref_elong * 3 +
            (fts["convexity"] - ref_conv).abs() / ref_conv * 5) / 3).clip(upper=1)


def compute_wall_alignment(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.Series:
    """
    Estimate the wall alignment between pairs of geometries.

    Computes a normalized measure of how well the outer walls of two geometries align,
    based on buffered boundary overlap excluding interior area.
    """
    shared_border_area = buffer_area(geoms1) + buffer_area(geoms2) - buffer_area(geoms1.union(geoms2))
    intersection_area = geoms1.intersection(geoms2).area
    max_possible_shared_border = np.minimum(buffer_area(geoms1), buffer_area(geoms2))
    alignment = (shared_border_area - intersection_area) / max_possible_shared_border

    return alignment


def compute_shared_wall_length(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries, strict: bool = False, tolerance: float = 0.1
) -> pd.Series:
    """
    Calculate the shared wall length between pairs of geometries.

    Args:
        geoms1: Building geometries from the first dataset.
        geoms2: Building geometries from the second dataset.
        strict: If True, use exact geometries; if False, apply a buffer for fuzzy matching.
        tolerance: Buffer distance used when strict is False.

    Returns:
        Length of shared wall for each geometry pair.
    """
    if not strict:
        orig_geoms1_lengths = geoms1.length
        geoms1 = geoms1.buffer(tolerance)
        geoms2 = geoms2.buffer(tolerance)

    intersections = geoms1.intersection(geoms2).length

    if not strict:
        intersections = (intersections / 2) - (2 * tolerance)
        intersections = intersections.clip(lower=0, upper=orig_geoms1_lengths)

    return pd.Series(intersections, index=geoms1.index, name="shared_wall_length")


def compute_shared_wall_w_neighbors(
    geoms: gpd.GeoSeries, neighbors: gpd.GeoSeries, strict: bool = False, tolerance: float = 0.01
) -> pd.Series:
    """
    Calculate the proportion of a geometry's wall length shared with neighboring geometries.
    """
    shared_walls = pd.Series(index=geoms.index)
    intersecting = neighbors.sindex.query(geoms)

    # Convert [(A, B), (A, C), ...] pairs to {A: [B, C]}
    d = defaultdict(list)
    for geom_idx, neighbor_idx in zip(intersecting[0], intersecting[1]):
        d[geom_idx].append(neighbor_idx)

    for geom_idx, neighbor_idx in d.items():
        local_buildings = pd.concat([geoms.iloc[[geom_idx]], neighbors.iloc[neighbor_idx]])
        shared_wall = momepy.shared_walls(local_buildings, strict, tolerance)
        idx = geoms.index[geom_idx]
        shared_walls.at[idx] = shared_wall.loc[idx]

    shared_walls_norm = shared_walls / geoms.boundary.length

    return shared_walls_norm


def count_intersecting_geometries(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.Series:
    """
    For each geometry in geoms1, calculate the number of geometries in geoms2 that intersect it
    """
    idx1, _ = geoms2.sindex.query(geoms1, predicate="intersects")
    ids1 = geoms1.index[idx1]
    n_intersecting = pd.Series(ids1).value_counts()

    return n_intersecting


def calculate_area_intersected(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.Series:
    """
    For each geometry in geoms1, calculate the share of its footprint area that is intersected by geometries in geoms2.
    """
    idx1, idx2 = geoms2.sindex.query(geoms1, predicate="intersects")

    intersecting_geometries = geoms1.iloc[idx1].intersection(
        geoms2.iloc[idx2], align=False
    )
    intersection_area = intersecting_geometries.area.groupby(level=0).sum()

    rel_intersection_area = (
        intersection_area / geoms1.loc[intersection_area.index].area
    )

    return rel_intersection_area
