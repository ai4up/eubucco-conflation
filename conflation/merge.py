import geopandas as gpd
import pandas as pd

from conflation.geoutil import generate_blocks, blocks_id_mapping


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
        matching_block_pairs = candidate_pairs[candidate_pairs["match"]][["block_id_existing", "block_id_new"]].drop_duplicates()
        # existing_buildings = _fill_missing_attributes_blockwise(existing_buildings, new_buildings, matching_block_pairs)
        existing_buildings = _fill_missing_attributes_by_intersection(existing_buildings, new_buildings, matching_block_pairs)

    # Block-wise spatial merge
    matched_blocks = candidate_pairs[candidate_pairs["match"]]["block_id_new"]
    non_matching = new_buildings[~new_buildings["block_id"].isin(matched_blocks)]
    conflated = pd.concat([existing_buildings, non_matching])

    return conflated


def _fill_missing_attributes_blockwise(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, matching_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Fills missing attributes of existing buildings with averages of matching new building blocks.
    """
    type_missings = gdf1["type"].isna()
    height_missings = gdf1["height"].isna()
    age_missings = gdf1["age"].isna()

    matching = gdf2[gdf2["block_id"].isin(matching_pairs["block_id_new"])]
    block_attributes = matching.groupby("block_id").agg({
        "height": "mean",
        "age": "mean",
        "type": _most_frequent_category,
    })

    matched_block_attributes = matching_pairs.merge(
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


def _fill_missing_attributes_by_intersection(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    matching_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Fills missing attributes of existing buildings with weighted averages from intersecting, matching new buildings.

    Averages are weighted by intersection area.
    For categorical attributes, the category with the largest cumulative intersecting area is chosen.
    """
    type_missings = gdf1["type"].isna()
    height_missings = gdf1["height"].isna()
    age_missings = gdf1["age"].isna()

    # Focus only on buildings with missing attributes
    gdf1_missing = gdf1[type_missings | height_missings | age_missings]
    gdf1_missing = gdf1_missing[["block_id", "geometry"]].reset_index(names="building_id")

    # Determine intersecting building pairs
    intersections = gpd.overlay(gdf1_missing, gdf2, how="intersection")
    intersections["area"] = intersections.geometry.area

    # Only merge attributes between matching blocks
    intersections_matching = intersections.merge(
        matching_pairs,
        how="inner",
        left_on=["block_id_1", "block_id_2"],
        right_on=["block_id_existing", "block_id_new"]
    )

    # Caclulate average age and height weighted by intersection area
    def weighted_avg(group, column):
        return (group[column] * group["area"]).sum() / group["area"].sum()

    # Choose building type with largest cumulative intersecting area
    def type_by_max_area(group):
        return group.groupby("type")["area"].sum().idxmax()

    # Aggregate attributes of all matches
    height_avg = intersections_matching[~intersections_matching["height"].isna()].groupby("building_id").apply(lambda g: weighted_avg(g, "height"))
    age_avg = intersections_matching[~intersections_matching["age"].isna()].groupby("building_id").apply(lambda g: weighted_avg(g, "age"))
    type_dominant = intersections_matching[~intersections_matching["type"].isna()].groupby("building_id").apply(type_by_max_area)

    # Merge attributes
    if not height_avg.empty:
        gdf1["height"] = gdf1["height"].fillna(height_avg)
    if not age_avg.empty:
        gdf1["age"] = gdf1["age"].fillna(age_avg)
    if not type_dominant.empty:
        gdf1["type"] = gdf1["type"].fillna(type_dominant)

    # Track for which buildings attributes were merged
    gdf1["filled_type"] = type_missings & gdf1["type"].notna()
    gdf1["filled_height"] = height_missings & gdf1["height"].notna()
    gdf1["filled_age"] = age_missings & gdf1["age"].notna()

    return gdf1


def _most_frequent_category(s: pd.Series) -> str:
    value_counts = s.value_counts()
    max_count = value_counts.max()
    most_frequent = value_counts[value_counts == max_count].index[0]

    return most_frequent
