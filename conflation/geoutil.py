import uuid
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import h3
from scipy.spatial import KDTree
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate


def overlapping(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> Tuple[pd.Index, pd.Index]:
    """
    Find all overlapping building pairs between two GeoDataFrames.
    """
    idx2, idx1 = gdf1.sindex.query(gdf2.geometry, predicate="intersects")

    return gdf1.index[idx1], gdf2.index[idx2]


def generate_blocks(
    buildings: gpd.GeoDataFrame, tolerance: Optional[float] = None
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Generate blocks from building footprints by grouping touching buildings.
    """
    if buildings.empty:
        blocks_gdf = gpd.GeoDataFrame(columns=["building_ids"], geometry=[], crs=buildings.crs)
        blocks_gdf.index.name = "block_id"
        return blocks_gdf

    if tolerance:
        buildings.geometry = _simplified_rectangular_buffer(buildings, tolerance)

    # Determine touching buildings
    left_idx, right_idx = overlapping(buildings, buildings)

    # Exclude intersections with itself
    mask = left_idx != right_idx
    left_idx = left_idx[mask]
    right_idx = right_idx[mask]

    graph = nx.Graph()
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

    if blocks:
        blocks_gdf = gpd.GeoDataFrame(blocks, geometry="geometry", crs=buildings.crs).set_index("block_id")
        blocks_gdf.geometry = _simplified_rectangular_buffer(blocks_gdf, 0.01)  # ensure all geometries are Polygons and valid
    else:
        blocks_gdf = gpd.GeoDataFrame(columns=["geometry", "building_ids", "block_id", "size"], crs=buildings.crs).set_index("block_id")

    print(f"Generated {len(blocks_gdf)} blocks with on average {blocks_gdf['size'].mean():.1f} buildings.")

    blocks_gdf = _add_standalone_buildings_to_blocks(blocks_gdf, buildings)

    return blocks_gdf


def generate_blocks_from_ids(
    buildings: gpd.GeoDataFrame, tolerance: Optional[float] = None, block_id_col: str = "block_id", building_id_col: str = "id", geom_col: str = "geometry"
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Generate blocks from building's predefined block ids.
    """
    buildings = buildings.set_geometry(geom_col).rename(columns={building_id_col: "building_ids", block_id_col: "block_id"})

    if tolerance:
        buildings.geometry = _simplified_rectangular_buffer(buildings, tolerance)

    blocks_gdf = buildings.dissolve(
        by="block_id",
        aggfunc={"building_ids": list},
    )

    blocks_gdf["size"] = blocks_gdf["building_ids"].str.len()
    blocks_gdf.geometry = _simplified_rectangular_buffer(blocks_gdf, 0.01)  # ensure all geometries are Polygons and valid

    return blocks_gdf


def blocks_id_mapping(blocks: gpd.GeoDataFrame) -> pd.Series:
    """
    Create a map to link building IDs with their corresponding block IDs.
    """
    s = blocks["building_ids"].explode()

    return pd.Series(s.index, index=s.values)


def preprocess_geometry(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Preprocess geometries by simplifying and extracting the largest polygon of MultiPolygon geometries.
    """
    def extract_largest_polygon_from_multipolygon(geom):
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda a: a.area)
        return geom

    geoms = geoms.buffer(0).simplify(0.5).make_valid()
    geoms = geoms.apply(extract_largest_polygon_from_multipolygon)

    return geoms


def h3_index(gdf: gpd.GeoDataFrame, res: int) -> List[str]:
    """
    Generate H3 indexes for the geometries in a GeoDataFrame.
    """
    # H3 operations require a lat/lon point geometry
    centroids = gdf.centroid.to_crs("EPSG:4326")
    lngs = centroids.x
    lats = centroids.y
    h3_idx = [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]

    return h3_idx


def get_nearest_neighbors(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the k-nearest neighbors for each geometry in gdf1 from gdf2.
    Distances are calculated based on the centroids of the geometries.
    """
    coordinates1 = np.array([[point.x, point.y] for point in gdf1.centroid])
    coordinates2 = np.array([[point.x, point.y] for point in gdf2.centroid])

    tree = KDTree(coordinates2)

    # Find 5 nearest neighbors for each point
    distances, indices = tree.query(coordinates1, k=k)

    return indices, distances


def get_larger(
    geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries
) -> gpd.GeoSeries:
    """
    Get the larger geometry between pairs of geometries.
    """

    return gpd.GeoSeries(
        np.where(
            geoms1.area > geoms2.area,
            geoms1,
            geoms2,
        ), index=geoms1.index)


def align_centroids(geom1: Polygon, geom2: Polygon) -> Polygon:
    """
    Align the centroid of geom2 to the centroid of geom1.
    """
    c1 = geom1.centroid
    c2 = geom2.centroid

    # Compute translation distances
    dx = c1.x - c2.x
    dy = c1.y - c2.y

    # Apply translation to geom2
    return translate(geom2, xoff=dx, yoff=dy)


def buffer_area(geoms: gpd.GeoSeries, size: float = 1) -> pd.Series:
    """
    Calculate the area of the buffer zone around each geometry (excluding the geometry itself).
    """
    return geoms.buffer(size, join_style="mitre").difference(geoms).area


def ioa(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Calculate the Two-Way Area Overlap (TWAO) between building pairs.
    Also referred to as symmetrical Intersection over Area (IoA).
    """
    return geoms1.intersection(geoms2, align=False).area / np.minimum(geoms1.area, geoms2.area).values


def iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the pairwise Intersection over Union (IoU) for two sets of geometries."""
    intersection = geoms1.intersection(geoms2, align=False).area
    union = geoms1.union(geoms2, align=False).area
    iou = intersection / union
    iou = iou.fillna(0)

    return iou


def deduplicate(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """
    Remove buildings that significantly overlap with other buildings in the same dataset,
    retaining only the larger building.
    """
    idx1, idx2 = overlapping(gdf, gdf)

    # Exclude intersections with itself
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Filter overlapping buildings
    overlap = ioa(gdf.loc[idx1].geometry.reset_index(), gdf.loc[idx2].geometry.reset_index())
    mask = overlap > tolerance
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Keep only the larger building
    smaller = gdf.loc[idx1].area.values < gdf.loc[idx2].area.values
    idx = np.where(smaller, idx1, idx2)
    gdf = gdf.drop(idx)

    print(f"{len(idx)} buildings removing with an overlap of more than {tolerance:.2f} during deduplication.")

    return gdf


def dissolve_geometries_of_m_n_matches(
    matching_pairs: pd.DataFrame,
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Identify connected components and dissolve their geometries to aggregate matching pairs into m:n matches.
    """
    components = _aggregate_to_m_n_matches(matching_pairs)
    rows = []
    for ids_1, ids_2 in components:
        geoms_1 = gdf1.loc[list(ids_1), 'geometry']
        geoms_2 = gdf2.loc[list(ids_2), 'geometry']

        rows.append({
            "component_id": uuid.uuid4().hex[:16],
            "ids_1": ids_1,
            "ids_2": ids_2,
            "geometry_1": geoms_1.union_all(),
            "geometry_2": geoms_2.union_all(),
        })

    gdf = gpd.GeoDataFrame(rows)
    gdf["geometry_1"] = gpd.GeoSeries(gdf["geometry_1"], crs=gdf1.crs)
    gdf["geometry_2"] = gpd.GeoSeries(gdf["geometry_2"], crs=gdf2.crs)

    return gdf


def groupby_apply(gdf: gpd.GeoDataFrame, by: str, func: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame], *args, **kwargs) -> gpd.GeoDataFrame:
    """
    Apply func to each group, keeping geometry awareness.
    """
    results = []
    for _, df in gdf.groupby(by):
        sub = gpd.GeoDataFrame(df, geometry=gdf.geometry.name, crs=gdf.crs)
        results.append(func(sub, *args, **kwargs))

    out = pd.concat(results, ignore_index=False)

    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)


def _aggregate_to_m_n_matches(
    matching_pairs: pd.DataFrame,
    col1: str = "building_id_1",
    col2: str = "building_id_2"
) -> List[Tuple[set, set]]:
    """
    Aggregate matching pairs into m:n matches by identifying connected components in a bipartite graph.
    """
    G = nx.Graph()
    G.add_edges_from(zip(matching_pairs[col1], matching_pairs[col2]))

    components = []
    for nodes in nx.connected_components(G):
        sub_df = matching_pairs[
            matching_pairs[col1].isin(nodes) | matching_pairs[col2].isin(nodes)
        ]
        set_A = set(sub_df[col1])
        set_B = set(sub_df[col2])
        components.append((set_A, set_B))

    return components


def _simplified_rectangular_buffer(geoms: gpd.GeoSeries, size: float) -> gpd.GeoSeries:
    """
    Create a simplified rectangular buffer around each geometry.
    """
    return geoms.simplify(0.1).buffer(size, join_style="mitre")


def _add_standalone_buildings_to_blocks(
    blocks: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Add standalone buildings (not part of any block) as individual blocks.
    """
    singles = buildings[~buildings.index.isin(blocks["building_ids"].explode())].copy()
    if not singles.empty:
        singles["size"] = 1
        singles["block_id"] = [uuid.uuid4().hex[:16] for _ in range(len(singles))]
        singles["building_ids"] = singles.index.map(lambda x: [x])
        singles = singles.set_index("block_id")[["geometry", "building_ids", "size"]]

        blocks = pd.concat([blocks, singles])

    return blocks
