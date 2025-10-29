from typing import Optional, List
import logging

import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from conflation.features import (
    compute_ioa,
    compute_iou,
    compute_aligned_iou,
    compute_wall_alignment,
    compute_shared_wall_length,
    compute_area_intersected,
)
from conflation.geoutil import (
    preprocess_geometry,
    generate_blocks,
    blocks_id_mapping,
    get_nearest_neighbors,
    groupby_apply,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def calculate_matching_features(
    existing_buildings: gpd.GeoDataFrame,
    new_buildings: gpd.GeoDataFrame,
    candidate_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    logger.info("Preprocessing geometry...")
    existing_buildings.geometry = preprocess_geometry(existing_buildings.geometry)
    new_buildings.geometry = preprocess_geometry(new_buildings.geometry)

    candidate_pairs = gpd.GeoDataFrame(candidate_pairs)
    candidate_pairs["geometry_existing"] = candidate_pairs["id_existing"].map(existing_buildings.geometry)
    candidate_pairs["geometry_existing"] = gpd.GeoSeries(candidate_pairs["geometry_existing"], crs=existing_buildings.crs)
    candidate_pairs["geometry_new"] = candidate_pairs["id_new"].map(new_buildings.geometry)
    candidate_pairs["geometry_new"] = gpd.GeoSeries(candidate_pairs["geometry_new"], crs=new_buildings.crs)

    logger.info("Determining blocks...")
    existing_blocks = groupby_apply(existing_buildings.copy(), "LAU_ID", generate_blocks, tolerance=0.25)
    new_blocks = groupby_apply(new_buildings.copy(), "LAU_ID", generate_blocks, tolerance=0.25)

    candidate_pairs["block_id_existing"] = candidate_pairs["id_existing"].map(blocks_id_mapping(existing_blocks))
    candidate_pairs["block_id_new"] = candidate_pairs["id_new"].map(blocks_id_mapping(new_blocks))
    candidate_pairs["block_geometry_existing"] = candidate_pairs["block_id_existing"].map(existing_blocks.geometry)
    candidate_pairs["block_geometry_new"] = candidate_pairs["block_id_new"].map(new_blocks.geometry)

    logger.info("Calculating building shape characteristics...")
    new_fts = calculate_shape_characteristics(candidate_pairs["geometry_new"])
    existing_fts = calculate_shape_characteristics(candidate_pairs["geometry_existing"])
    avg_fts = (new_fts + existing_fts) / 2

    logger.info("Calculating similarity characteristics...")
    bldg_similarities = calculate_similarity_features(candidate_pairs["geometry_new"], candidate_pairs["geometry_existing"])
    block_similarities = calculate_similarity_features(candidate_pairs["block_geometry_new"], candidate_pairs["block_geometry_existing"])
    bldg_diff = _percentage_diff(new_fts, existing_fts)

    logger.info("Joining characteristics...")
    bldg_diff = bldg_diff.add_prefix("diff_bldg_")
    avg_fts = avg_fts.add_prefix("bldg_")
    bldg_similarities = bldg_similarities.add_prefix("bldg_")
    block_similarities = block_similarities.add_prefix("block_")

    candidate_pairs = pd.concat([
        candidate_pairs,
        avg_fts,
        bldg_diff,
        bldg_similarities,
        block_similarities,
    ], axis=1).copy()

    logger.info("Calculating context features...")
    candidate_pairs = calculate_context_features(candidate_pairs, existing_buildings, new_buildings)

    return candidate_pairs


def calculate_shape_characteristics(geoms: gpd.GeoSeries) -> pd.DataFrame:
    fts = pd.DataFrame(index=geoms.index)
    fts["footprint_area"] = geoms.area
    fts["perimeter"] = geoms.length
    fts["longest_axis_length"] = momepy.longest_axis_length(geoms)
    fts["elongation"] = momepy.elongation(geoms)
    fts["orientation"] = momepy.orientation(geoms)
    fts["area_perimeter_ratio"] = fts["footprint_area"] / fts["perimeter"]
    fts["shape_index"] = momepy.shape_index(geoms, longest_axis_length=fts["longest_axis_length"])

    return fts


def calculate_similarity_features(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.DataFrame:
    similarities = pd.DataFrame(index=geoms1.index)
    similarities["distance"] = geoms1.distance(geoms2)
    similarities["distance_centroids"] = geoms1.centroid.distance(geoms2.centroid)
    similarities["intersection"] = geoms1.intersection(geoms2).area
    similarities["ioa"] = compute_ioa(geoms1, geoms2)
    similarities["iou"] = compute_iou(geoms1, geoms2)
    similarities["aligned_iou"] = compute_aligned_iou(geoms1, geoms2)
    similarities["wall_alignment"] = compute_wall_alignment(geoms1, geoms2)
    similarities["shared_walls"] = compute_shared_wall_length(geoms1, geoms2)

    return similarities


def calculate_context_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    existing_buildings["area_share_intersected"] = compute_area_intersected(existing_buildings.geometry, new_buildings.geometry)
    new_buildings["area_share_intersected"] = compute_area_intersected(new_buildings.geometry, existing_buildings.geometry)

    candidates = _calculate_area_share_intersected(candidates, existing_buildings, new_buildings)
    candidates = _calculate_neighborhood_alignment(candidates, existing_buildings, new_buildings)

    return candidates


def _percentage_diff(
    df1: pd.DataFrame, df2: pd.DataFrame, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if cols is None:
        return (df1 - df2) / df2

    return (df1[cols] - df2[cols]) / df2[cols]


def _candidate_pair_centroids(candidates: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Calculate the centroid of each candidate pair.
    """
    return gpd.GeoSeries([
        Point((a.x + b.x) / 2, (a.y + b.y) / 2)
        for a, b in zip(candidates["geometry_existing"].centroid, candidates["geometry_new"].centroid)
    ], crs=candidates["geometry_existing"].crs)


def _calculate_area_share_intersected(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    existing_area_share_intersected = candidates["id_existing"].map(existing_buildings["area_share_intersected"])
    existing_area_share_intersected_others = existing_area_share_intersected - (candidates["bldg_intersection"] / candidates["geometry_existing"].area)

    new_area_share_intersected = candidates["id_new"].map(new_buildings["area_share_intersected"])
    new_area_share_intersected_others = new_area_share_intersected - (candidates["bldg_intersection"] / candidates["geometry_new"].area)

    candidates["area_share_intersected"] = np.maximum(existing_area_share_intersected, new_area_share_intersected)
    candidates["area_share_intersected_others"] = np.maximum(existing_area_share_intersected_others, new_area_share_intersected_others)

    return candidates


def _calculate_neighborhood_alignment(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    centroids = _candidate_pair_centroids(candidates)

    indices, _ = get_nearest_neighbors(centroids, existing_buildings, k=5+1)
    existing_area_share_intersected_neighbors = np.array([existing_buildings.iloc[nearest_neighbors]["area_share_intersected"].mean() for nearest_neighbors in indices])

    indices, _ = get_nearest_neighbors(centroids, new_buildings, k=5+1)
    new_area_share_intersected_neighbors = np.array([new_buildings.iloc[nearest_neighbors]["area_share_intersected"].mean() for nearest_neighbors in indices])

    candidates["alignment"] = (existing_area_share_intersected_neighbors + new_area_share_intersected_neighbors) / 2

    return candidates
