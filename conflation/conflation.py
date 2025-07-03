from pathlib import Path
import itertools

import geopandas as gpd
import pandas as pd

from geo_matcher.dataset import create_candidate_pairs_dataset
from conflation.alignment import correct_local_shift
from conflation.feateng import calculate_matching_features
from conflation.prediction import predict_match
from conflation.merge import block_wise_merge


def conflate(
    datasets: list[str],
    region_id: str,
    data_dirs_input: list[str],
    data_dirs_matching: str,
    data_dir_results: str,
    attribute_mapping: list[bool],
    h3_res: int,
    model_path: str,
    db_version: str,
) -> None:
    model_path = Path(model_path)
    data_dir_results = Path(data_dir_results)
    data_dir_results.mkdir(parents=True, exist_ok=True)
    results_path = data_dir_results / f"{region_id}.parquet"

    if results_path.exists():
        print(f"Conflated data already exists for {region_id}, skipping.")
        return

    ref_index, reference_data = _get_first_existing_parquet(region_id, data_dirs_input)
    reference_data["dataset"] = datasets[ref_index]

    for data_dir, matching_dir, mapping, name in zip(data_dirs_input[ref_index+1:], data_dirs_matching[ref_index:], attribute_mapping[ref_index:], datasets[ref_index+1:]):
        new_data_path = Path(data_dir, f"{region_id}.parquet")
        matching_path = Path(matching_dir, f"{region_id}.parquet")
        matching_path.parent.mkdir(parents=True, exist_ok=True)

        if not new_data_path.exists():
            print(f"No data found for {region_id} in {data_dir}, skipping.")
            continue

        new_data = gpd.read_parquet(new_data_path).set_index("id")
        new_data["dataset"] = name
        reference_data = conflate_pair(reference_data, new_data, h3_res, model_path, matching_path, mapping)

        if mapping and 'filled_height' in reference_data.columns:
            reference_data["filled_height"] = reference_data["filled_height"].replace({True: name})
            reference_data["filled_age"] = reference_data["filled_age"].replace({True: name})
            reference_data["filled_type"] = reference_data["filled_type"].replace({True: name})
            reference_data["filled_residential_type"] = reference_data["filled_residential_type"].replace({True: name})

        if mapping and 'height_mapped' in reference_data.columns:
            attr = ["height", "age", "type", "residential_type"]
            suffixes = ["source_ids", "mapped", "confidence_iou", "confidence_ioa"]
            reference_data = reference_data.rename(columns={f"{attr}_{suffix}": f"{attr}_{suffix}_{name}" for attr, suffix in itertools.product(attr, suffixes)})
            reference_data = reference_data.rename(columns={"matching_confidence": f"matching_confidence_{name}"})

    reference_data = _generate_unique_id(reference_data, db_version)
    reference_data.to_parquet(results_path)


def conflate_gov_osm_msft(
    region_id: str,
    gov_dir: str,
    osm_dir: str,
    msft_dir: str,
    matching_dir: str,
    out_dir: str,
    h3_res: int,
    model_path: str,
    db_version: str,
) -> None:
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    matching_dir = Path(matching_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{region_id}.parquet"

    if out_path.is_file():
        print(f"Conflated data already exists for {region_id}.")
        return

    matching_dir_osm = (matching_dir / "osm")
    matching_dir_msft = (matching_dir / "msft")
    matching_dir_osm.mkdir(parents=True, exist_ok=True)
    matching_dir_msft.mkdir(parents=True, exist_ok=True)
    matching_path_osm = matching_dir_osm / f"{region_id}.parquet"
    matching_path_msft = matching_dir_msft / f"{region_id}.parquet"

    gov_path = Path(gov_dir, f"{region_id}.parquet")
    osm_path = Path(osm_dir, f"{region_id}.parquet")
    msft_path = Path(msft_dir, f"{region_id}.parquet")

    osm = gpd.read_parquet(osm_path).set_index("id")
    msft = gpd.read_parquet(msft_path).set_index("id")

    osm["dataset"] = "osm"
    msft["dataset"] = "msft"

    if gov_path.exists():
        gov = gpd.read_parquet(gov_path).set_index("id")
        gov["dataset"] = "gov"

        print(f"Conflating Gov, OSM and MSFT for {region_id}.")
        conflated = conflate_pair(gov, osm, h3_res, model_path, matching_path_osm, attribute_mapping=True)
        conflated = conflate_pair(conflated, msft, h3_res, model_path, matching_path_msft, attribute_mapping=False)
    else:
        print(f"No government data found for {region_id}, conflating only OSM and MSFT.")
        conflated = conflate_pair(osm, msft, h3_res, model_path, matching_path_msft, attribute_mapping=False)

    conflated = _generate_unique_id(conflated, db_version)
    conflated.to_parquet(out_path)


def conflate_pair(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    h3_res: int,
    model_path: Path,
    matching_results_path: Path,
    attribute_mapping: bool,
) -> gpd.GeoDataFrame:
    print("(1) Spatially aligning datasets...")
    gdf2["geometry"] = correct_local_shift(gdf1, gdf2.copy(), h3_res)

    if matching_results_path.exists():
        print("Loading matching results. Skipping (2)-(4).")
        pairs_w_pred = pd.read_parquet(matching_results_path)
    else:
        print("(2) Creating candidate pairs...")
        pairs = create_candidate_pairs_dataset(gdf1, gdf2).pairs
        print(f"Number of candidate pairs: {len(pairs)}")

        print("(3) Calculating matching features...")
        pairs_w_fts = calculate_matching_features(gdf1, gdf2, pairs)

        print("(4) Estimating matching relationships...")
        pairs_w_pred = predict_match(model_path, pairs_w_fts)
        pairs_w_pred.to_parquet(matching_results_path, index=False)
        print(f"Share of matching pairs: {pairs_w_pred['match'].mean():.2%}")


    print("(5) Merging data...")
    conflated_buildings = block_wise_merge(gdf1, gdf2, pairs_w_pred, attribute_mapping)
    n_added_buildings = len(conflated_buildings) - len(gdf1)
    print(f"Added buildings during conflation stage: +{n_added_buildings} ({n_added_buildings / len(gdf1):.2%})")

    if attribute_mapping and 'height_mapped' in conflated_buildings.columns:
        print(f"Added height information during conflation stage: +{conflated_buildings['height_mapped'].notna().sum()} ({conflated_buildings['height_mapped'].notna().mean():.2%})")
        print(f"Added age information during conflation stage: +{conflated_buildings['age_mapped'].notna().sum()} ({conflated_buildings['age_mapped'].notna().mean():.2%})")
        print(f"Added type information during conflation stage: +{conflated_buildings['type_mapped'].notna().sum()} ({conflated_buildings['type_mapped'].notna().mean():.2%})")
        print(f"Added residential type information during conflation stage: +{conflated_buildings['residential_type_mapped'].notna().sum()} ({conflated_buildings['residential_type_mapped'].notna().mean():.2%})")

    if attribute_mapping and 'filled_height' in conflated_buildings.columns:
        print(f"Added height information during conflation stage: +{conflated_buildings['filled_height'].eq(True).sum()} ({conflated_buildings['filled_height'].eq(True).mean():.2%})")
        print(f"Added age information during conflation stage: +{conflated_buildings['filled_age'].eq(True).sum()} ({conflated_buildings['filled_age'].eq(True).mean():.2%})")
        print(f"Added type information during conflation stage: +{conflated_buildings['filled_type'].eq(True).sum()} ({conflated_buildings['filled_type'].eq(True).mean():.2%})")

    return conflated_buildings


def _generate_unique_id(gdf: gpd.GeoDataFrame, db_version: str) -> gpd.GeoDataFrame:
    gdf = gdf.reset_index(names="id_source")
    gdf["id"] = (
        "v" + str(db_version) + "-" +
        gdf["LAU_ID"] + "-" +
        gdf.groupby("LAU_ID").cumcount().astype(str)
    )

    return gdf


def _get_first_existing_parquet(region_id: str, data_dirs: list[str]) -> tuple[int, gpd.GeoDataFrame]:
    for i, data_dir in enumerate(data_dirs):
        path = Path(data_dir, f"{region_id}.parquet")
        if path.exists():
            gdf = gpd.read_parquet(path).set_index("id")
            return i, gdf

    return None, None
