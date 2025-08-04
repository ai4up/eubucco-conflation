import pandas as pd
import geopandas as gpd

from geo_matcher import spatial


sample_nuts_ids = [
    'AT212', 'BE335', 'BG325', 'CH061', 'CY000', 'CZ031', 'DE211', 'DK012', 'EE004', 'EL631',
    'ES422', 'FI1D1', 'FRG01', 'HR04C', 'HU213', 'IE051', 'ITF14', 'LT011', 'LU000', 'LV006',
    'MT001', 'NL213', 'NO041', 'PL218', 'PT16D', 'RO114', 'SE321', 'SI037', 'SK023', 'UKK41',
    'RO315', 'DE218', 'DE719', 'FRG02', 'DE94E', 'FRK11', 'FRJ27', 'DEA1A', 'CZ042', 'ES511',
    'FI200', 'BG422', 'ITG12', 'FR106', 'BE213', 'DE232', 'FRE12', 'ES417', 'UKH14', 'DE271'
]
neighborhoods = [
    '891e161660fffff', '891fa01630fffff', '891f83212cbffff',
    '892da451a53ffff', '891e303b29bffff', '891f8d98973ffff',
    '891f0591613ffff', '89089ec9c2fffff', '893f2dcc563ffff',
    '89390e2269bffff', '8911200014bffff', '89186e4b503ffff',
    '891e1aac577ffff', '891e1cc4c27ffff', '891e81109afffff',
    '891f4004a03ffff', '891fa3cd013ffff', '891f60728c3ffff',
    '893f3040b67ffff', '891f16c1d23ffff', '89099eadcc3ffff',
    '891e2a0186fffff', '893931a6bd3ffff', '8908a83b41bffff',
    '891eac94cc7ffff', '891e03c909bffff', '891874ac6a3ffff',
    '891eed70417ffff', '891f8d7087bffff', '891fa8d2d1bffff',
    '89186eb3467ffff', '891f105b207ffff', '891f946a0cfffff',
    '893960161cbffff', '891fa52eb07ffff', '891e3414d53ffff',
    '8939446742bffff', '8908838d253ffff', '891ec104017ffff',
    '891e9a75d67ffff', '891fb42893bffff', '891fa43a0a3ffff',
    '891e364c673ffff', '891948a9b6bffff', '8939769153bffff',
    '89194ed680fffff', '891f8c3482bffff' # missing neighborhoods for ['RO114', 'IE051', 'BG325'] due to missing data 
]

for dataset in ['gov', 'osm', 'msft']:
    data = []
    for nuts_id in sample_nuts_ids:

        filepath = f'/p/projects/eubucco/data/3-attrib-cleaning-v1-{dataset}/{nuts_id}.parquet'

        try:
            if dataset != 'msft':
                gdf = gpd.read_parquet(filepath, columns=['id', 'height', 'age', 'type', 'residential_type', 'geometry'])
            else:
                gdf = gpd.read_parquet(filepath)
                gdf["residential_type"] = pd.NA
                gdf["height"] = gdf["height_source"]
                gdf = gdf[['id', 'height', 'age', 'type', 'residential_type', 'geometry']]


        except FileNotFoundError:
            print(f'File not found for {dataset} in {nuts_id}')
            continue

        gdf = gdf.reset_index(drop=True).add_prefix(f'{nuts_id}_{dataset}_', axis=0)
        gdf["neighborhood"] = spatial.h3_index(gdf, 9)
        gdf = gdf[gdf["neighborhood"].isin(neighborhoods)]
        df = gdf.drop(columns=['geometry', 'neighborhood'])

        data.append(df)

    df = pd.concat(data, ignore_index=True)
    df.to_parquet(f'/p/projects/eubucco/data/conflation-training-data/candidate-attributes-{dataset}.parquet', index=False)
