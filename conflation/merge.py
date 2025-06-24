from conflation.features import generate_blocks, blocks_id_mapping
import geopandas as gpd
import pandas as pd


def block_wise_merge(
    existing_buildings: gpd.GeoDataFrame,
    new_buildings: gpd.GeoDataFrame,
    candidate_pairs: pd.DataFrame,
    attribute_mapping: bool,
) -> gpd.GeoDataFrame:
    """
    Merges existing and new buildings at the block level based on matching results.
    """
    existing_blocks = generate_blocks(existing_buildings, tolerance=0.25)
    new_blocks = generate_blocks(new_buildings, tolerance=0.25)
    existing_block_mapping = blocks_id_mapping(existing_blocks)
    new_block_mapping = blocks_id_mapping(new_blocks)

    existing_buildings["block_id"] = existing_buildings.index.map(existing_block_mapping)
    new_buildings["block_id"] = new_buildings.index.map(new_block_mapping)
    candidate_pairs["block_id_existing"] = candidate_pairs["id_existing"].map(existing_block_mapping)
    candidate_pairs["block_id_new"] = candidate_pairs["id_new"].map(new_block_mapping)

    # Block-wise attribute mapping
    if attribute_mapping:
        mapping = candidate_pairs[candidate_pairs["match"]][["block_id_existing", "block_id_new"]].drop_duplicates()
        existing_buildings = _fill_missing_attributes_blockwise(existing_buildings, new_buildings, mapping)

    # Block-wise spatial merge
    matching_blocks = candidate_pairs[candidate_pairs["match"]]["block_id_new"]
    non_matching = new_buildings[~new_buildings["block_id"].isin(matching_blocks)]
    conflated = pd.concat([existing_buildings, non_matching])

    return conflated


def _fill_missing_attributes_blockwise(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, mapping: pd.DataFrame) -> gpd.GeoDataFrame:
    type_missings = gdf1["type"].isna()
    height_missings = gdf1["height"].isna()
    age_missings = gdf1["age"].isna()

    matching = gdf2[gdf2["block_id"].isin(mapping["block_id_new"])]
    block_attributes = matching.groupby("block_id").agg({
        "height": "mean",
        "age": "mean",
        "type": _most_frequent_category,
    })

    matched_block_attributes = mapping.merge(
        block_attributes,
        left_on="block_id_new",
        right_index=True,
    ).groupby("block_id_existing").agg({
        "height": "mean",
        "age": "mean",
        "type": _most_frequent_category,
    })

    gdf1["height"] = gdf1["height"].fillna(gdf1["block_id"].map(matched_block_attributes["height"]))
    gdf1["age"] = gdf1["age"].fillna(gdf1["block_id"].map(matched_block_attributes["age"]))
    gdf1["type"] = gdf1["type"].fillna(gdf1["block_id"].map(matched_block_attributes["type"]))

    gdf1["filled_type"] = type_missings & gdf1["type"].notna()
    gdf1["filled_height"] = height_missings & gdf1["height"].notna()
    gdf1["filled_age"] = age_missings & gdf1["age"].notna()

    return gdf1


def _most_frequent_category(s: pd.Series) -> str:
    value_counts = s.value_counts()
    max_count = value_counts.max()
    most_frequent = value_counts[value_counts == max_count].index[0]

    return most_frequent
