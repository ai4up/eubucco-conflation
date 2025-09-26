import logging

from geopandas import GeoDataFrame
from pandas import DataFrame
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)
log = logger.info


def determine_candidate_pairs(
    gdf1: GeoDataFrame = None,
    gdf2: GeoDataFrame = None,
) -> DataFrame:
    """
    Identify pairs of potentially matching buildings from two datasets.
    """
    _verify_unique_index(gdf1, gdf2)
    pairs = _determine_candidate_pairs_knn(gdf1, gdf2, max_dist=10, k=3)

    return pairs


def _verify_unique_index(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame
) -> None:
    if not gdf1.index.is_unique or not gdf2.index.is_unique:
        raise ValueError("Unique index is required.")

    if _indices_overlap(gdf1, gdf2):
        raise ValueError("The indices of both datasets must not overlap.")


def _determine_candidate_pairs_knn(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame, max_dist: float, k: int
) -> DataFrame:
    """
    Symmetrical k-nearest neighbors search for polygon geometries with max distance threshold.
    Two step approach:
    1) Heuristic approximation using bounding box-based STRtree spatial index.
    2) Exact distance calculation and filtering of top-k neighbors within max_dist.
    """
    # Core logic: one-sided k-nearest neighbors search with max distance threshold
    def _neighbors_within(gdf_a, gdf_b, id_a, id_b):
        a_geoms = gdf_a.geometry.values
        b_geoms = gdf_b.geometry.values
        tree = STRtree(b_geoms)

        a_buf = a_geoms.buffer(max_dist)
        ai, bi = tree.query(a_buf)

        dists = np.array([a_geoms[i].distance(b_geoms[j]) for i, j in zip(ai, bi)])

        df = pd.DataFrame(
            {
                id_a: gdf_a.index[ai],
                id_b: gdf_b.index[bi],
                "distance": dists,
            }
        )

        # Keep pairs within max_dist (bbox preselection may return larger distances)
        df = df[df["distance"] <= max_dist]

        # Keep top-k per source geometry
        df = (
            df.sort_values([id_a, "distance"], kind="mergesort")
            .groupby(id_a)
            .head(k)
            .reset_index(drop=True)
        )
        return df[[id_a, id_b]]

    # Symmetrical nearest neighbors search
    pairs_1to2 = _neighbors_within(gdf1, gdf2, "id_existing", "id_new")
    pairs_2to1 = _neighbors_within(gdf2, gdf1, "id_new", "id_existing")

    # Drop duplicate pairs introduced by symmetrical search
    pairs = (
        pd.concat([pairs_1to2, pairs_2to1], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return pairs


def _indices_overlap(gdf1: GeoDataFrame, gdf2: GeoDataFrame) -> bool:
    return not gdf1.index.intersection(gdf2.index).empty
