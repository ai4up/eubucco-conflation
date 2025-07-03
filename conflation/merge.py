import geopandas as gpd
import pandas as pd
import shapely

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
        # existing_buildings = _fill_missing_attributes_by_intersection(existing_buildings, new_buildings, matching_block_pairs)
        existing_buildings = _merge_attributes_by_intersection(existing_buildings, new_buildings, matching_block_pairs)

    # Block-wise spatial merge
    matched_blocks = candidate_pairs[candidate_pairs["match"]]["block_id_new"]
    non_matching = new_buildings[~new_buildings["block_id"].isin(matched_blocks)]
    conflated = pd.concat([existing_buildings, non_matching])

    return conflated


def _fill_missing_attributes_blockwise(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, matching_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Fills missing attributes of existing buildings with averages of matching new building blocks.
    """
    height_missings = gdf1["height"].isna()
    age_missings = gdf1["age"].isna()
    type_missings = gdf1["type"].isna()
    res_type_missing = gdf1["residential_type"].isna()

    # Ensure consistent schema of input datasets
    if "residential_type" not in gdf2.columns:
        gdf2["residential_type"] = pd.NA

    matching = gdf2[gdf2["block_id"].isin(matching_pairs["block_id_new"])]
    block_attributes = matching.groupby("block_id").agg({
        "height": "mean",
        "age": "mean",
        "type": _most_frequent_category,
        "residential_type": _most_frequent_category,
    })

    matched_block_attributes = matching_pairs.merge(
        block_attributes,
        left_on="block_id_new",
        right_index=True,
    ).groupby("block_id_existing").agg({
        "height": "mean",
        "age": "mean",
        "type": _most_frequent_category,
        "residential_type": _most_frequent_category,
    })

    gdf1["height"] = gdf1["height"].fillna(gdf1["block_id"].map(matched_block_attributes["height"]))
    gdf1["age"] = gdf1["age"].fillna(gdf1["block_id"].map(matched_block_attributes["age"]))
    gdf1["type"] = gdf1["type"].fillna(gdf1["block_id"].map(matched_block_attributes["type"]))
    gdf1["residential_type"] = gdf1["residential_type"].fillna(gdf1["block_id"].map(matched_block_attributes["residential_type"]))

    gdf1["filled_type"] = type_missings & gdf1["type"].notna()
    gdf1["filled_height"] = height_missings & gdf1["height"].notna()
    gdf1["filled_age"] = age_missings & gdf1["age"].notna()
    gdf1["filled_residential_type"] = res_type_missing & gdf1["residential_type"].notna()

    return gdf1


def _fill_missing_attributes_by_intersection(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    matching_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Fills missing attributes of existing buildings with weighted averages from intersecting, matching new buildings.

    For numerical attributes, averages are weighted by intersection area.
    For categorical attributes, the category with the largest cumulative intersecting area is chosen.
    """
    height_missings = gdf1["height"].isna()
    age_missings = gdf1["age"].isna()
    type_missings = gdf1["type"].isna()
    res_type_missing = gdf1["residential_type"].isna()

    # Focus only on buildings with missing attributes
    gdf1_missing = gdf1[height_missings | age_missings | type_missings | res_type_missing]
    gdf1_missing = gdf1_missing[["block_id", "geometry"]]

    # Ensure consistent schema of input datasets
    if "residential_type" not in gdf2.columns:
        gdf2["residential_type"] = pd.NA
    
    # Determine intersecting building pairs
    intersections = gpd.overlay(gdf1_missing.reset_index(names="building_id"), gdf2, how="intersection")
    intersections["area"] = intersections.geometry.area

    # Only merge attributes between matching blocks
    intersections_matching = intersections.merge(
        matching_pairs,
        how="inner",
        left_on=["block_id_1", "block_id_2"],
        right_on=["block_id_existing", "block_id_new"]
    )

    # Only fill attributes for buildings ≥50% intersected
    inter_area = intersections_matching.groupby("building_id")["area"].sum()
    valid_ids = inter_area[inter_area >= 0.5 * gdf1_missing.loc[inter_area.index]["geometry"].area].index
    intersections_matching = intersections_matching[intersections_matching["building_id"].isin(valid_ids)]

    # Caclulate average age and height weighted by intersection area
    def weighted_avg(group, column):
        return (group[column] * group["area"]).sum() / group["area"].sum()

    # Choose building type with largest cumulative intersecting area
    def most_dominant_category_by_area(group, column):
        return group.groupby(column)["area"].sum().idxmax()

    # Aggregate attributes of all matches
    height_avg = intersections_matching[~intersections_matching["height"].isna()].groupby("building_id").apply(lambda g: weighted_avg(g, "height"))
    age_avg = intersections_matching[~intersections_matching["age"].isna()].groupby("building_id").apply(lambda g: weighted_avg(g, "age"))
    type_dominant = intersections_matching[~intersections_matching["type"].isna()].groupby("building_id").apply(most_dominant_category_by_area, column="type")
    res_type_dominant = intersections_matching[~intersections_matching["residential_type"].isna()].groupby("building_id").apply(most_dominant_category_by_area, column="residential_type")

    # Merge attributes
    if not height_avg.empty:
        gdf1["height"] = gdf1["height"].fillna(height_avg)
    if not age_avg.empty:
        gdf1["age"] = gdf1["age"].fillna(age_avg)
    if not type_dominant.empty:
        gdf1["type"] = gdf1["type"].fillna(type_dominant)
    if not res_type_dominant.empty:
        gdf1["residential_type"] = gdf1["residential_type"].fillna(res_type_dominant)

    # Track filled attributes
    if "filled_height" not in gdf1.columns:
        gdf1["filled_height"] = pd.NA
    if "filled_age" not in gdf1.columns:
        gdf1["filled_age"] = pd.NA
    if "filled_type" not in gdf1.columns:
        gdf1["filled_type"] = pd.NA
    if "filled_residential_type" not in gdf1.columns:
        gdf1["filled_residential_type"] = pd.NA

    gdf1.loc[height_missings & gdf1["height"].notna(), "filled_height"] = True
    gdf1.loc[age_missings & gdf1["age"].notna(), "filled_age"] = True
    gdf1.loc[type_missings & gdf1["type"].notna(), "filled_type"] = True
    gdf1.loc[res_type_missing & gdf1["residential_type"].notna(), "filled_residential_type"] = True

    return gdf1


def _merge_attributes_by_intersection(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    matching_pairs: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Merges attributes from new buildings (gdf2) into existing buildings (gdf1)
    based on geometric intersection, considering only building pairs within
    matching block pairs.

    For each attribute, only the subset of intersecting new buildings that
    maximizes the intersection-over-union (IoU) with the existing building is used.
    This improves robustness to spatial noise and overlapping geometries.

    Numerical attributes (e.g., height, age) are merged using intersection-area-weighted averages.
    Categorical attributes (e.g., type) are assigned based on the most dominant category
    by cumulative intersecting area.

    For each attribute, the function also records:
      - Confidence scores based on IoU and intersection-over-area (IoA)
      - A list of contributing source building IDs
    """
    gdf1_tmp = gdf1[["block_id", "geometry"]].reset_index(names="building_id")
    gdf1_tmp["area"] = gdf1_tmp.area

    # Ensure consistent schema of input datasets
    if "residential_type" not in gdf2.columns:
        gdf2["residential_type"] = pd.NA

    gdf2_tmp = gdf2[["block_id", "geometry", "height", "age", "type", "residential_type"]].reset_index(names="building_id")
    gdf2_tmp["area"] = gdf2_tmp.area

    # Determine intersecting building pairs
    intersections = gpd.overlay(gdf1_tmp, gdf2_tmp, how="intersection")
    intersections["area_int"] = intersections.geometry.area
    intersections["int_ratio"] = intersections["area_int"] / (intersections["area_2"] - intersections["area_int"]).clip(lower=0)  # Prevent division by negative area

    # Only merge attributes between matching blocks
    intersections_matching = intersections.merge(
        matching_pairs,
        how="inner",
        left_on=["block_id_1", "block_id_2"],
        right_on=["block_id_existing", "block_id_new"]
    )

    def weighted_avg(group, column):
        """Calculate average age and height weighted by intersection area"""
        return (group[column] * group["area_int"]).sum() / group["area_int"].sum()

    def most_dominant_category_by_area(group, column):
        """Choose building type with largest cumulative intersecting area"""
        return group.groupby(column)["area_int"].sum().idxmax()

    gdf1 = _merge_attribute(gdf1, gdf2, intersections_matching, "height", weighted_avg)
    gdf1 = _merge_attribute(gdf1, gdf2, intersections_matching, "age", weighted_avg)
    gdf1 = _merge_attribute(gdf1, gdf2, intersections_matching, "type", most_dominant_category_by_area)

    return gdf1


def _merge_attribute(
    gdf1, gdf2, mapping, attr, agg_func
):
    def argmax_iou(group):
        """
        Returns the subset of the group that maximizes a heuristic estimate of the intersection-over-union (IoU).
        The geometry is assumed to be topologically correct. Overlapping polygons will lead to incorrect IoU estimates.
        """
        if len(group) == 1:
            return group

        group = group.sort_values(by="int_ratio", ascending=False)
        iou_max = 0
        for i in range(len(group)):
            g = group.iloc[:i + 1]
            union_area = g["area_1"].iloc[0] + (g["area_2"] - g["area_int"]).sum()
            intersection_area = g["area_int"].sum()
            new_iou = intersection_area / union_area
            if new_iou > iou_max:
                iou_max = new_iou
            else:
                return group.iloc[:i]

        return group

    def iou(group):
        """
        Returns the intersection-over-union (IoU) of the group. Robust to overlapping polygons.
        """
        intersection_area = group.geometry.union_all().area
        union_area = shapely.unary_union([gdf1.loc[group.name].geometry, *gdf2.loc[group["building_id_2"]].geometry]).area
        return intersection_area / union_area

    def ioa(group):
        """
        Returns the intersection-over-area (IoA) of the group. Robust to overlapping polygons.
        """
        intersection_area = group.geometry.union_all().area
        area = gdf1.loc[group.name].geometry.area
        return intersection_area / area

    mapping = mapping[~mapping[attr].isna()]

    if mapping.empty:
        for suffix in ["source_ids", "mapped", "confidence_iou", "confidence_ioa"]:
            gdf1[f"{attr}_{suffix}"] = pd.NA
        return gdf1

    # Only consider intersecting buildings that increase the IoU
    mapping = mapping.groupby("building_id_1", group_keys=False).apply(argmax_iou)

    # (1) Aggregate attributes, (2) track source IDs, and (3) calculate confidence scores
    gdf1[f"{attr}_source_ids"] = mapping.groupby("building_id_1")["building_id_2"].apply(list)
    gdf1[f"{attr}_mapped"] = mapping.groupby("building_id_1").apply(lambda g: agg_func(g, attr))
    gdf1[f"{attr}_confidence_iou"] = mapping.groupby("building_id_1").apply(iou)
    gdf1[f"{attr}_confidence_ioa"] = mapping.groupby("building_id_1").apply(ioa)

    return gdf1


def _most_frequent_category(s: pd.Series) -> str:
    value_counts = s.value_counts()
    max_count = value_counts.max()
    most_frequent = value_counts[value_counts == max_count].index[0]

    return most_frequent
