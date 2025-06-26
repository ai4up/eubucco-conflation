import uuid

import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon, MultiPolygon

from shapely.affinity import translate
from typing import Optional, Tuple, List


def calculate_matching_features(
    existing_buildings: gpd.GeoDataFrame,
    new_buildings: gpd.GeoDataFrame,
    candidate_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    print("Preprocessing geometry...")
    existing_buildings = preprocess_geometry(existing_buildings)
    new_buildings = preprocess_geometry(new_buildings)

    candidate_pairs = gpd.GeoDataFrame(candidate_pairs)
    candidate_pairs["geometry_existing"] = candidate_pairs["id_existing"].map(existing_buildings.geometry)
    candidate_pairs["geometry_existing"] = gpd.GeoSeries(candidate_pairs["geometry_existing"], crs=existing_buildings.crs)
    candidate_pairs["geometry_new"] = candidate_pairs["id_new"].map(new_buildings.geometry)
    candidate_pairs["geometry_new"] = gpd.GeoSeries(candidate_pairs["geometry_new"], crs=new_buildings.crs)

    print("Determining blocks...")
    existing_blocks = generate_blocks(existing_buildings, tolerance=0.25)
    new_blocks = generate_blocks(new_buildings, tolerance=0.25)

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
        lambda row: get_union_of_intersecting_components(row, existing_buildings, new_buildings),
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
    candidate_pairs = calculate_shared_wall_w_neighbors(candidate_pairs, existing_buildings, new_buildings)

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


def calculate_wall_alignment(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> pd.Series:
    """
    Estimate the wall alignment between pairs of geometries.

    Computes a normalized measure of how well the outer walls of two geometries align,
    based on buffered boundary overlap excluding interior area.
    """
    shared_border_area = _buf_area(geoms1) + _buf_area(geoms2) - _buf_area(geoms1.union(geoms2))
    intersection_area = geoms1.intersection(geoms2).area
    max_possible_shared_border = np.minimum(_buf_area(geoms1), _buf_area(geoms2))
    alignment = (shared_border_area - intersection_area) / max_possible_shared_border

    return alignment


def compute_ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    return geoms1.intersection(geoms2).area / np.minimum(geoms1.area, geoms2.area)


def compute_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    intersection = geoms1.intersection(geoms2).area
    union = geoms1.union(geoms2).area
    iou = intersection / union
    iou = iou.fillna(0)  # mitigate devision by zero

    return iou


def compute_aligned_iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    aligned_geoms2 = gpd.GeoSeries([_align_centroids(g1, g2) for g1, g2 in zip(geoms1, geoms2)], index=geoms1.index, crs=geoms1.crs)
    aligned_iou = compute_iou(geoms1, aligned_geoms2)

    return aligned_iou


def compute_aligned_ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    aligned_geoms2 = gpd.GeoSeries([_align_centroids(g1, g2) for g1, g2 in zip(geoms1, geoms2)], index=geoms1.index, crs=geoms1.crs)
    aligned_ioa = compute_ioa(geoms1, aligned_geoms2)

    return aligned_ioa


def calculate_car_similarity(fts: pd.DataFrame) -> pd.Series:
    ref_elong = 0.5
    ref_area = 15
    ref_conv = 1

    return (((fts["footprint_area"] - ref_area).abs() / ref_area +
            (fts["elongation"] - ref_elong).abs() / ref_elong * 3 +
            (fts["convexity"] - ref_conv).abs() / ref_conv * 5) / 3).clip(upper=1)


def count_intersecting_buildings(
    existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> pd.Series:
    """For each existing building, calculate the number of new buildings that intersect it"""
    existing_building_idx, _ = new_buildings.sindex.query(existing_buildings.geometry, predicate="intersects")
    existing_building_ids = existing_buildings.index[existing_building_idx]
    n_intersecting = pd.Series(existing_building_ids).value_counts()

    return n_intersecting


def calculate_area_intersected(
    existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> pd.Series:
    """For each existing building, calculate the share of the footprint area that is intersected by new buildings"""
    new_buildings = new_buildings.to_crs(existing_buildings.crs)
    existing_building_idx, new_building_idx = new_buildings.sindex.query(existing_buildings.geometry, predicate="intersects")

    intersecting_geometries = existing_buildings.iloc[existing_building_idx].geometry.intersection(
        new_buildings.iloc[new_building_idx].geometry, align=False
    )
    intersection_area = intersecting_geometries.area.groupby(level=0).sum()

    rel_intersection_area = (
        intersection_area / existing_buildings.loc[intersection_area.index].area
    )

    return rel_intersection_area


def generate_blocks(
    buildings: gpd.GeoDataFrame, tolerance: Optional[float] = None
) -> gpd.GeoDataFrame:
    if buildings.empty:
        blocks_gdf = gpd.GeoDataFrame(columns=["building_ids"], geometry=[], crs=buildings.crs)
        blocks_gdf.index.name = "block_id"
        return blocks_gdf

    if tolerance:
        buildings.geometry = _simplified_rectangular_buffer(buildings, tolerance)

    # Determine touching buildings
    left_idx, right_idx = buildings.sindex.query(buildings.geometry, predicate="intersects")
    left_idx = buildings.index[left_idx]
    right_idx = buildings.index[right_idx]
    
    # Exclude intersections with itself
    mask = left_idx != right_idx
    left_idx = left_idx[mask]
    right_idx = right_idx[mask]

    graph = nx.Graph()
    graph.add_nodes_from(buildings.index)
    graph.add_edges_from(zip(left_idx, right_idx))
    connected_components = list(nx.connected_components(graph))

    blocks = []
    for component in connected_components:
        block_buildings = buildings.loc[list(component), "geometry"]
        block_geometry = block_buildings.union_all()
        blocks.append({
            "geometry": block_geometry,
            "building_ids": list(component),
            "block_id": uuid.uuid4().hex[:16],
            "size": len(component),
        })

    blocks_gdf = gpd.GeoDataFrame(blocks, geometry="geometry", crs=buildings.crs).set_index("block_id")
    blocks_gdf.geometry = _simplified_rectangular_buffer(blocks_gdf, 0.01)  # ensure all geometries are Polygons and valid        
    print(f"Generated {len(blocks_gdf)} blocks with on average {blocks_gdf['size'].mean():.1f} buildings.")

    return blocks_gdf


def blocks_id_mapping(blocks: gpd.GeoDataFrame) -> pd.Series:
    s = blocks["building_ids"].explode()

    return pd.Series(s.index, index=s.values)


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
    similarities["wall_alignment"] = calculate_wall_alignment(geoms1, geoms2)
    similarities["shared_walls"] = shared_wall_pairs(geoms1, geoms2, strict=False)    

    return similarities


def shared_wall_pairs(
    left: gpd.GeoSeries, right: gpd.GeoSeries, strict: bool = True, tolerance: float = 0.1
) -> pd.Series:
    """
    Calculate the shared wall length between aligned pairs of geometries.

    Parameters
    ----------
    left : GeoSeries
        Left geometries in the pair (e.g., first building in a pair).
    right : GeoSeries
        Right geometries in the pair (e.g., second building in a pair).
    strict : bool
        Use exact geometries (True) or buffered geometries for fuzzy matching (False).
    tolerance : float
        Buffer distance to use if strict=False.

    Returns
    -------
    Series
        Length of shared wall for each pair.
    """
    if not strict:
        orig_left_lengths = left.length
        left = left.buffer(tolerance)
        right = right.buffer(tolerance)

    intersections = left.intersection(right).length

    if not strict:
        intersections = (intersections / 2) - (2 * tolerance)
        intersections = intersections.clip(lower=0, upper=orig_left_lengths)

    return pd.Series(intersections, index=left.index, name="shared_wall_length")


def calculate_spatial_interaction_features(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    gdf1["area_share_intersected"] = calculate_area_intersected(gdf1, gdf2)
    gdf1["n_intersecting"] = count_intersecting_buildings(gdf1, gdf2)

    gdf1["area_share_intersected"] = gdf1["area_share_intersected"].fillna(0)
    gdf1["n_intersecting"] = gdf1["n_intersecting"].fillna(0)

    return gdf1


def get_nearest_neighbors_kd(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    coordinates1 = np.array([[point.x, point.y] for point in gdf1.centroid])
    coordinates2 = np.array([[point.x, point.y] for point in gdf2.centroid])

    tree = KDTree(coordinates2)

    # Find 5 nearest neighbors for each point
    distances, indices = tree.query(coordinates1, k=k+1)

    return indices, distances


def get_larger(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> gpd.GeoSeries:
    
    return gpd.GeoSeries(
        np.where(
            geoms1.area > geoms2.area,
            geoms1,
            geoms2,
        ), index=geoms1.index)


def candidate_pair_centroids(candidates: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Calculate the centroid of each candidate pair.
    """
    return gpd.GeoSeries([
        Point((a.x + b.x) / 2, (a.y + b.y) / 2)
        for a, b in zip(candidates["geometry_existing"].centroid, candidates["geometry_new"].centroid)
    ], crs=candidates["geometry_existing"].crs)


def get_union_of_intersecting_components(
    pair: pd.Series, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoSeries:
    """
    For the larger block (existing or new), get the union of all intersecting buildings grom the other dataset
    """
    existing_is_bigger = pair["block_geometry_existing"].area > pair["block_geometry_new"].area
    larger_block = pair["block_geometry_existing"] if existing_is_bigger else pair["block_geometry_new"]

    respective_buildings = new_buildings if existing_is_bigger else existing_buildings
    intersecting = respective_buildings.sindex.query(larger_block, predicate="intersects")

    if len(intersecting) > 0:
        return respective_buildings.geometry.iloc[intersecting].union_all()

    smaller_block = pair["block_geometry_new"] if existing_is_bigger else pair["block_geometry_existing"]

    return smaller_block


def calculate_buffer_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    all_buildings = pd.concat([existing_buildings, new_buildings], ignore_index=True)
    pair_geom = candidates["geometry_existing"].union(candidates["geometry_new"])
    buffer_sizes = [5, 10, 25, 50]
    for size in buffer_sizes:
        buffer = pair_geom.buffer(size)
        candidates[f"n_intersecting_buffer_{size}"] = count_intersecting_buildings(buffer, all_buildings) - 2 # remove self-intersection

    return candidates


def map_to_candidate_pairs(
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


def calculate_context_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    existing_buildings = calculate_spatial_interaction_features(existing_buildings, new_buildings)
    new_buildings = calculate_spatial_interaction_features(new_buildings, existing_buildings)
    candidates = map_to_candidate_pairs(candidates, existing_buildings, new_buildings)
    candidates = calculate_neighborhood_features(candidates, existing_buildings, new_buildings)

    return candidates


def calculate_shared_wall_w_neighbors(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    candidates["_existing_shared_walls_neighbors"] = momepy.shared_walls(pd.concat([candidates["geometry_existing"], new_buildings.geometry]), strict=False)
    candidates["_existing_shared_walls_neighbors"] = candidates["_existing_shared_walls_neighbors"] / candidates["geometry_existing"].boundary.length
        
    candidates["_new_shared_walls_neighbors"] = momepy.shared_walls(pd.concat([candidates["geometry_new"], existing_buildings.geometry]), strict=False)
    candidates["_new_shared_walls_neighbors"] = candidates["_new_shared_walls_neighbors"] / candidates["geometry_new"].boundary.length
        
    candidates["shared_walls_neighbors"] = candidates[["_existing_shared_walls_neighbors", "_new_shared_walls_neighbors"]].max(axis=1)

    return candidates


def calculate_neighborhood_features(
    candidates: gpd.GeoDataFrame, existing_buildings: gpd.GeoDataFrame, new_buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    centroids = candidate_pair_centroids(candidates)

    indices, _ = get_nearest_neighbors_kd(centroids, existing_buildings, k=5)
    candidates["_existing_n_intersecting_neighbors"] = [existing_buildings.iloc[nearest_neigbors]["n_intersecting"].mean() for nearest_neigbors in indices]
    candidates["_existing_area_share_intersected_neighbors"] = [existing_buildings.iloc[nearest_neigbors]["area_share_intersected"].mean() for nearest_neigbors in indices]
    candidates["_existing_n_intersecting_neighbors"] = candidates["_existing_n_intersecting_neighbors"].fillna(0)
    candidates["_existing_area_share_intersected_neighbors"] = candidates["_existing_area_share_intersected_neighbors"].fillna(0)

    indices, _ = get_nearest_neighbors_kd(centroids, new_buildings, k=5)
    candidates["_new_n_intersecting_neighbors"] = [new_buildings.iloc[nearest_neigbors]["n_intersecting"].mean() for nearest_neigbors in indices]
    candidates["_new_area_share_intersected_neighbors"] = [new_buildings.iloc[nearest_neigbors]["area_share_intersected"].mean() for nearest_neigbors in indices]
    candidates["_new_n_intersecting_neighbors"] = candidates["_new_n_intersecting_neighbors"].fillna(0)
    candidates["_new_area_share_intersected_neighbors"] = candidates["_new_area_share_intersected_neighbors"].fillna(0)

    candidates["alignment"] = candidates[["_existing_area_share_intersected_neighbors", "_new_area_share_intersected_neighbors"]].mean(axis=1)
    candidates["agreement"] = candidates[["_existing_n_intersecting_neighbors", "_new_n_intersecting_neighbors"]].mean(axis=1)
    candidates["agreement_diff"] = (candidates["_existing_n_intersecting_neighbors"] - candidates["_new_n_intersecting_neighbors"]).abs()

    return candidates


def calculate_spatial_features(buildings: gpd.GeoSeries) -> pd.DataFrame:
    fts = pd.DataFrame(index=buildings.index)
    centroid = buildings.centroid.to_crs("EPSG:4326")
    fts["lat"] = centroid.y
    fts["lon"] = centroid.x

    return fts


def preprocess_geometry(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def extract_largest_polygon_from_multipolygon(geom):
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda a: a.area)
        return geom

    buildings["geometry"] = buildings.buffer(0).simplify(0.5)
    buildings["geometry"] = buildings.geometry.apply(extract_largest_polygon_from_multipolygon)

    return buildings


def _simplified_rectangular_buffer(geoms: gpd.GeoSeries, size: float) -> gpd.GeoSeries:
    return geoms.simplify(0.1).buffer(size, join_style="mitre")


def _percentage_diff(
    gdf1: pd.DataFrame, gdf2: pd.DataFrame, cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if cols is None:
        return (gdf1 - gdf2) / gdf2

    return (gdf1[cols] - gdf2[cols]) / gdf2[cols]


def _align_centroids(geom1: Polygon, geom2: Polygon) -> Polygon:
    c1 = geom1.centroid
    c2 = geom2.centroid
    
    # Compute translation distances
    dx = c1.x - c2.x
    dy = c1.y - c2.y
    
    # Apply translation to geom2
    return translate(geom2, xoff=dx, yoff=dy)


def _buf_area(gdf: gpd.GeoSeries, size: float = 1) -> pd.Series:
    return gdf.buffer(size, join_style="mitre").difference(gdf).area
