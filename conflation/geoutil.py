import uuid

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import KDTree
from shapely.geometry import Polygon, MultiPolygon

from shapely.affinity import translate
from typing import Optional, Tuple


def generate_blocks(
    buildings: gpd.GeoDataFrame, tolerance: Optional[float] = None
) -> gpd.GeoDataFrame:
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

    geoms = geoms.buffer(0).simplify(0.5)
    geoms = geoms.apply(extract_largest_polygon_from_multipolygon)

    return geoms


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
    distances, indices = tree.query(coordinates1, k=k+1)

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


def iou(geoms1: gpd.GeoSeries, geoms2: gpd.GeoSeries) -> pd.Series:
    """
    Compute the pairwise Intersection over Union (IoU) for two sets of geometries."""
    intersection = geoms1.intersection(geoms2, align=False).area
    union = geoms1.union(geoms2, align=False).area
    iou = intersection / union
    iou = iou.fillna(0)

    return iou


def _simplified_rectangular_buffer(geoms: gpd.GeoSeries, size: float) -> gpd.GeoSeries:
    """
    Create a simplified rectangular buffer around each geometry.
    """
    return geoms.simplify(0.1).buffer(size, join_style="mitre")
