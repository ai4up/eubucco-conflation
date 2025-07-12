from typing import Optional, List

import momepy
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from conflation.features import (
    compute_ioa,
    compute_iou,
    compute_aligned_ioa,
    compute_aligned_iou,
    compute_wall_alignment,
    compute_shared_wall_length,
    compute_shared_wall_w_neighbors,
    calculate_car_similarity,
    calculate_area_intersected,
    count_intersecting_geometries,
)
from conflation.geoutil import (
    preprocess_geometry,
    get_larger,
    generate_blocks,
    blocks_id_mapping,
    get_nearest_neighbors,
)


def calculate_matching_features(
    existing_buildings: gpd.GeoDataFrame,
    new_buildings: gpd.GeoDataFrame,
    candidate_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    print("Preprocessing geometry...")
    existing_buildings.geometry = preprocess_geometry(existing_buildings.geometry)
    new_buildings.geometry = preprocess_geometry(new_buildings.geometry)

    candidate_pairs = gpd.GeoDataFrame(candidate_pairs)
    candidate_pairs["geometry_existing"] = candidate_pairs["id_existing"].map(existing_buildings.geometry)
    candidate_pairs["geometry_existing"] = gpd.GeoSeries(candidate_pairs["geometry_existing"], crs=existing_buildings.crs)
    candidate_pairs["geometry_new"] = candidate_pairs["id_new"].map(new_buildings.geometry)
    candidate_pairs["geometry_new"] = gpd.GeoSeries(candidate_pairs["geometry_new"], crs=new_buildings.crs)

    print("Determining blocks...")
    existing_blocks = generate_blocks(existing_buildings.copy(), tolerance=0.25)
    new_blocks = generate_blocks(new_buildings.copy(), tolerance=0.25)

    candidate_pairs["block_id_existing"] = candidate_pairs["id_existing"].map(blocks_id_mapping(existing_blocks))
    candidate_pairs["block_id_new"] = candidate_pairs["id_new"].map(blocks_id_mapping(new_blocks))

    candidate_pairs["block_geometry_existing"] = candidate_pairs["block_id_existing"].map(existing_blocks.geometry)
    candidate_pairs["block_geometry_new"] = candidate_pairs["block_id_new"].map(new_blocks.geometry)

    candidate_pairs["existing_block_length"] = candidate_pairs["block_id_existing"].map(existing_blocks["size"])
    candidate_pairs["new_block_length"] = candidate_pairs["block_id_new"].map(new_blocks["size"])
    candidate_pairs["diff_block_length"] = (candidate_pairs["existing_block_length"] - candidate_pairs["new_block_length"]).abs() / candidate_pairs["existing_block_length"]
    candidate_pairs["block_length"] = (candidate_pairs["existing_block_length"] + candidate_pairs["new_block_length"]) / 2

    print("Determining hypothetical blocks...")
    candidate_pairs["block_geometry_reference"] = get_larger(candidate_pairs["block_geometry_new"], candidate_pairs["block_geometry_existing"])
    candidate_pairs["block_geometry_hypothetical"] = candidate_pairs.apply(
        lambda row: _get_union_of_intersecting_components(row, existing_buildings, new_buildings),
        axis=1
    ).convex_hull

    print("Calculating building shape characteristics...")
    new_fts = calculate_shape_characteristics(candidate_pairs["geometry_new"])
    existing_fts = calculate_shape_characteristics(candidate_pairs["geometry_existing"])
    spatial_fts = calculate_spatial_features(candidate_pairs["geometry_new"])
    new_fts["car_similarity"] = calculate_car_similarity(new_fts)
    existing_fts["car_similarity"] = calculate_car_similarity(existing_fts)
    avg_fts = (new_fts + existing_fts) / 2

    print("Calculating block shape characteristics...")
    new_block_fts = calculate_shape_characteristics(candidate_pairs["block_geometry_new"])
    existing_block_fts = calculate_shape_characteristics(candidate_pairs["block_geometry_existing"])
    avg_block_fts = (new_block_fts + existing_block_fts) / 2

    print("Calculating similarity characteristics...")
    bldg_similarities = calculate_similarity_features(candidate_pairs["geometry_new"], candidate_pairs["geometry_existing"])
    block_similarities = calculate_similarity_features(candidate_pairs["block_geometry_new"], candidate_pairs["block_geometry_existing"])
    hypothetical_block_similarities = calculate_similarity_features(candidate_pairs["block_geometry_reference"], candidate_pairs["block_geometry_hypothetical"])
    bldg_diff = _percentage_diff(new_fts, existing_fts)
    block_diff = _percentage_diff(new_block_fts, existing_block_fts)
    bldg_similarities["shape_difference"] = bldg_diff.abs().mean(axis=1)
    block_similarities["shape_difference"] = block_diff.abs().mean(axis=1)

    print("Joining characteristics...")
    block_diff = block_diff.add_prefix("diff_block_")
    bldg_diff = bldg_diff.add_prefix("diff_bldg_")
    avg_fts = avg_fts.add_prefix("bldg_")
    existing_fts = existing_fts.add_prefix("existing_bldg_")
    new_fts = new_fts.add_prefix("new_bldg_")
    avg_block_fts = avg_block_fts.add_prefix("block_")
    existing_block_fts = existing_block_fts.add_prefix("existing_block_")
    new_block_fts = new_block_fts.add_prefix("new_block_")
    bldg_similarities = bldg_similarities.add_prefix("bldg_")
    block_similarities = block_similarities.add_prefix("block_")
    hypothetical_block_similarities = hypothetical_block_similarities.add_prefix("hypothetical_block_")

    candidate_pairs = pd.concat([
        candidate_pairs,
        new_fts,
        existing_fts,
        avg_fts,
        new_block_fts,
        existing_block_fts,
        avg_block_fts,
        bldg_diff,
        block_diff,
        bldg_similarities,
        block_similarities,
        hypothetical_block_similarities,
        spatial_fts,
    ], axis=1).copy()

    candidate_pairs["bldg_car_similarity"] = candidate_pairs[["existing_bldg_car_similarity", "new_bldg_car_similarity"]].min(axis=1)
    candidate_pairs["min_bldg_shape_index"] = candidate_pairs[["existing_bldg_shape_index", "new_bldg_shape_index"]].min(axis=1)
    candidate_pairs["max_bldg_shape_index"] = candidate_pairs[["existing_bldg_shape_index", "new_bldg_shape_index"]].max(axis=1)

    print("Calculating context features...")
    candidate_pairs = calculate_buffer_features(candidate_pairs, existing_buildings, new_buildings)
    candidate_pairs = calculate_context_features(candidate_pairs, existing_buildings, new_buildings)

    return candidate_pairs


def calculate_shape_characteristics(geoms: gpd.GeoSeries) -> pd.DataFrame:
    fts = pd.DataFrame(index=geoms.index)
    fts["footprint_area"] = geoms.area
    fts["perimeter"] = geoms.length
    fts["corners"] = momepy.corners(geoms, eps=45)
    fts["longest_axis_length"] = momepy.longest_axis_length(geoms)
    fts["elongation"] = momepy.elongation(geoms)
    fts["orientation"] = momepy.orientation(geoms)
    fts["squareness"] = momepy.squareness(geoms)
    fts["convexity"] = momepy.convexity(geoms)
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
    similarities["aligned_ioa"] = compute_aligned_ioa(geoms1, geoms2)
    similarities["aligned_iou"] = compute_aligned_iou(geoms1, geoms2)
    similarities["wall_alignment"] = compute_wall_alignment(geoms1, geoms2)
    similarities["shared_walls"] = compute_shared_wall_length(geoms1, geoms2)

    return similarities


def calculate_spatial_interaction_features(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    gdf1["area_share_intersected"] = calculate_area_intersected(gdf1.geometry, gdf2.geometry)
    gdf1["n_intersecting"] = count_intersecting_geometries(gdf1.geometry, gdf2.geometry)

    gdf1["area_share_intersected"] = gdf1["area_share_intersected"].fillna(0)
    gdf1["n_intersecting"] = gdf1["n_intersecting"].fillna(0)

    return gdf1


def calculate_buffer_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    all_buildings = pd.concat([existing_buildings, new_buildings], ignore_index=True)
    pair_geom = candidates["geometry_existing"].union(candidates["geometry_new"])
    buffer_sizes = [5, 10, 25, 50]
    for size in buffer_sizes:
        buffer = pair_geom.buffer(size)
        candidates[f"n_intersecting_buffer_{size}"] = count_intersecting_geometries(buffer, all_buildings.geometry) - 2 # remove self-intersection

    return candidates


def calculate_context_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    existing_buildings = calculate_spatial_interaction_features(existing_buildings, new_buildings)
    new_buildings = calculate_spatial_interaction_features(new_buildings, existing_buildings)
    candidates = _map_to_candidate_pairs(candidates, existing_buildings, new_buildings)
    candidates = calculate_neighborhood_features(candidates, existing_buildings, new_buildings)

    return candidates


def calculate_neighborhood_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    centroids = _candidate_pair_centroids(candidates)

    indices, _ = get_nearest_neighbors(centroids, existing_buildings, k=5)
    candidates["_existing_n_intersecting_neighbors"] = [existing_buildings.iloc[nearest_neigbors]["n_intersecting"].mean() for nearest_neigbors in indices]
    candidates["_existing_area_share_intersected_neighbors"] = [existing_buildings.iloc[nearest_neigbors]["area_share_intersected"].mean() for nearest_neigbors in indices]
    candidates["_existing_n_intersecting_neighbors"] = candidates["_existing_n_intersecting_neighbors"].fillna(0)
    candidates["_existing_area_share_intersected_neighbors"] = candidates["_existing_area_share_intersected_neighbors"].fillna(0)

    indices, _ = get_nearest_neighbors(centroids, new_buildings, k=5)
    candidates["_new_n_intersecting_neighbors"] = [new_buildings.iloc[nearest_neigbors]["n_intersecting"].mean() for nearest_neigbors in indices]
    candidates["_new_area_share_intersected_neighbors"] = [new_buildings.iloc[nearest_neigbors]["area_share_intersected"].mean() for nearest_neigbors in indices]
    candidates["_new_n_intersecting_neighbors"] = candidates["_new_n_intersecting_neighbors"].fillna(0)
    candidates["_new_area_share_intersected_neighbors"] = candidates["_new_area_share_intersected_neighbors"].fillna(0)

    candidates["_existing_shared_walls_neighbors"] = compute_shared_wall_w_neighbors(candidates["geometry_existing"], new_buildings.geometry)
    candidates["_new_shared_walls_neighbors"] = compute_shared_wall_w_neighbors(candidates["geometry_new"], existing_buildings.geometry)

    candidates["alignment"] = candidates[["_existing_area_share_intersected_neighbors", "_new_area_share_intersected_neighbors"]].mean(axis=1)
    candidates["agreement"] = candidates[["_existing_n_intersecting_neighbors", "_new_n_intersecting_neighbors"]].mean(axis=1)
    candidates["agreement_diff"] = (candidates["_existing_n_intersecting_neighbors"] - candidates["_new_n_intersecting_neighbors"]).abs()
    candidates["shared_walls_neighbors"] = candidates[["_existing_shared_walls_neighbors", "_new_shared_walls_neighbors"]].max(axis=1)

    return candidates


def calculate_spatial_features(buildings: gpd.GeoSeries) -> pd.DataFrame:
    fts = pd.DataFrame(index=buildings.index)
    centroid = buildings.centroid.to_crs("EPSG:4326")
    fts["lat"] = centroid.y
    fts["lon"] = centroid.x

    return fts


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


def _map_to_candidate_pairs(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    candidates["_existing_n_intersecting"] = candidates["id_existing"].map(existing_buildings["n_intersecting"])
    candidates["_existing_area_share_intersected"] = candidates["id_existing"].map(existing_buildings["area_share_intersected"])
    candidates["_existing_area_share_intersected_others"] = candidates["_existing_area_share_intersected"] - (candidates["bldg_intersection"] / candidates["existing_bldg_footprint_area"])

    candidates["_new_n_intersecting"] = candidates["id_new"].map(new_buildings["n_intersecting"])
    candidates["_new_area_share_intersected"] = candidates["id_new"].map(new_buildings["area_share_intersected"])
    candidates["_new_area_share_intersected_others"] = candidates["_new_area_share_intersected"] - (candidates["bldg_intersection"] / candidates["new_bldg_footprint_area"])

    candidates["n_intersecting"] = candidates[["_existing_n_intersecting", "_new_n_intersecting"]].max(axis=1)
    candidates["area_share_intersected"] = candidates[["_existing_area_share_intersected", "_new_area_share_intersected"]].max(axis=1)
    candidates["area_share_intersected_others"] = candidates[["_existing_area_share_intersected_others", "_new_area_share_intersected_others"]].max(axis=1)

    return candidates


def _get_union_of_intersecting_components(
    pair: pd.Series, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoSeries:
    """
    For the larger block (existing or new), get the union of all intersecting buildings from the other dataset
    """
    existing_is_bigger = pair["block_geometry_existing"].area > pair["block_geometry_new"].area
    larger_block = pair["block_geometry_existing"] if existing_is_bigger else pair["block_geometry_new"]

    respective_buildings = new_buildings if existing_is_bigger else existing_buildings
    intersecting = respective_buildings.sindex.query(larger_block, predicate="intersects")

    if len(intersecting) > 0:
        return respective_buildings.geometry.iloc[intersecting].union_all()

    smaller_block = pair["block_geometry_new"] if existing_is_bigger else pair["block_geometry_existing"]

    return smaller_block
