from collections import defaultdict

import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from conflation.geoutil import align_centroids, buffer_area, ioa, iou


def compute_ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Calculate the Two-Way Area Overlap (TWAO) between building pairs. Also referred to as symmetrical Intersection over Area (IoA).
    """
    return ioa(geoms1, geoms2)


def compute_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the Intersection over Union (IoU) between pairs of geometries.
    """
    return iou(geoms1, geoms2)


def compute_aligned_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the Intersection over Union (IoU) after aligning the centroids of geometry pairs.
    """
    centroids1 = geoms1.centroid
    centroids2 = geoms2.centroid

    dx = (centroids1.x - centroids2.x).to_numpy()
    dy = (centroids1.y - centroids2.y).to_numpy()

    geometries = geoms2.to_numpy().copy()
    coords, geom_idx = shapely.get_coordinates(geometries, return_index=True)

    coords[:, 0] += dx[geom_idx]
    coords[:, 1] += dy[geom_idx]

    aligned_geoms2 = gpd.GeoSeries(
        shapely.set_coordinates(geometries, coords),
        index=geoms1.index,
        crs=geoms1.crs,
    )
    aligned_iou = iou(geoms1, aligned_geoms2)

    return aligned_iou


def compute_wall_alignment(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.Series:
    """
    Estimate the wall alignment between pairs of geometries.

    Computes a normalized measure of how well the outer walls of two geometries align,
    based on buffered boundary overlap excluding interior area.
    """
    buffer_area_1 = buffer_area(geoms1)
    buffer_area_2 = buffer_area(geoms2)
    shared_border_area = buffer_area_1 + buffer_area_2 - buffer_area(geoms1.union(geoms2))
    intersection_area = geoms1.intersection(geoms2).area
    max_possible_shared_border = np.minimum(buffer_area_1, buffer_area_2)
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


def compute_area_intersected(
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

    return rel_intersection_area.reindex(geoms1.index, fill_value=0.0)
