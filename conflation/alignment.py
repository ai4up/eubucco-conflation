from pathlib import Path
from typing import Union

import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.affinity import translate

from geo_matcher import spatial


def spatially_align_building_datasets(
    nuts_region: str,
    h3_res: int,
    data_dir_existing_buildings: Union[str, Path],
    data_dir_new_buildings: Union[str, Path],
    data_dir_results: Union[str, Path],
) -> None:
    """
    Spatially aligns new buildings with existing buildings.
    """
    existing_buildings_path = Path(data_dir_existing_buildings, f"{nuts_region}.parquet")
    new_buildings_path = Path(data_dir_new_buildings, f"{nuts_region}.parquet")
    results_path = Path(data_dir_results, f"{nuts_region}.parquet")

    if results_path.exists():
        print(f"Aligned data already exists for {nuts_region}, skipping alignment.")
        return

    existing_buildings = gpd.read_parquet(existing_buildings_path)
    new_buildings = gpd.read_parquet(new_buildings_path)

    new_buildings["geometry"] = correct_local_shift(existing_buildings, new_buildings.copy(), h3_res)

    new_buildings.to_parquet(results_path)


def correct_local_shift(
    existing_buildings: gpd.GeoDataFrame,
    new_buildings: gpd.GeoDataFrame,
    h3_res: int
) -> gpd.GeoSeries:
    """
    Spatially aligns new buildings with existing buildings by correcting any consistent shifts within an H3 grid cell using highly similar landmark pairs.
    """
    existing_x = existing_buildings.centroid.x.values
    existing_y = existing_buildings.centroid.y.values
    existing_centroids = np.column_stack((existing_x, existing_y))

    new_x = new_buildings.centroid.x.values
    new_y = new_buildings.centroid.y.values
    new_centroids = np.column_stack((new_x, new_y))

    tree_existing = KDTree(existing_centroids)
    tree_new = KDTree(new_centroids)

    # Find closest existing building for each new one
    _, idx_n2e = tree_existing.query(new_centroids, distance_upper_bound=25)

    # Find closest new building for each existing one
    _, idx_e2n = tree_new.query(existing_centroids, distance_upper_bound=25)

    # Vectorized symmetric match check
    valid_n2e = idx_n2e < len(existing_centroids)
    new_idxs = np.flatnonzero(valid_n2e)
    existing_idxs = idx_n2e[valid_n2e]

    # Check symmetric condition: new[i] → existing[j] and existing[j] → new[i]
    reciprocal = idx_e2n[existing_idxs] == new_idxs
    new_idxs = new_idxs[reciprocal]
    existing_idxs = existing_idxs[reciprocal]

    # Area filtering
    gov_areas = existing_buildings.area.values[existing_idxs]
    osm_areas = new_buildings.area.values[new_idxs]
    area_ratio = np.minimum(osm_areas / gov_areas, gov_areas / osm_areas)
    area_mask = area_ratio >= 0.5

    new_idxs = new_idxs[area_mask]
    existing_idxs = existing_idxs[area_mask]

    # Compute offset vectors
    new_buildings["dx"] = np.nan
    new_buildings["dy"] = np.nan
    new_buildings.loc[new_buildings.index[new_idxs], "dx"] = existing_x[existing_idxs] - new_x[new_idxs]
    new_buildings.loc[new_buildings.index[new_idxs], "dy"] = existing_y[existing_idxs] - new_y[new_idxs]

    # Determine neighborhoods
    new_buildings["neighborhood"] = spatial.h3_index(new_buildings, h3_res)

    # Count matched buildings per neighborhood
    matched_buildings = new_buildings.iloc[new_idxs].copy()
    matched_counts = matched_buildings["neighborhood"].value_counts()
    total_counts = new_buildings["neighborhood"].value_counts()

    # Only keep neighborhoods with ≥5 matches and ≥10% coverage
    sufficient_coverage = (matched_counts >= 5) & (matched_counts / total_counts >= 0.10)
    valid_neighborhoods = sufficient_coverage[sufficient_coverage].index

    print(f"Correcting misaligned for {len(valid_neighborhoods) / len(total_counts) * 100:.1f}% ")

    # Filter matched buildings accordingly
    matched_buildings = matched_buildings[matched_buildings["neighborhood"].isin(valid_neighborhoods)]

    # Regional average offsets
    region_offsets = matched_buildings.groupby("neighborhood")[["dx", "dy"]].mean()
    new_buildings["dx"] = new_buildings["neighborhood"].map(region_offsets["dx"])
    new_buildings["dy"] = new_buildings["neighborhood"].map(region_offsets["dy"])

    # Ensure geometries are valid before translation
    new_buildings["geometry"] = new_buildings.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    new_buildings["old_geometry"] = new_buildings["geometry"]
    aligned_geometry = new_buildings.apply(
        lambda row: translate(row.geometry, xoff=row.dx, yoff=row.dy) 
        if row.geometry and row.geometry.is_valid and not np.isnan(row.dx) and not np.isnan(row.dy)
        else row.geometry, axis=1
    )

    return aligned_geometry
