import pandas as pd
import geopandas as gpd

from geo_matcher.dataset import create_candidate_pairs_dataset
from geo_matcher.candidate_pairs import CandidatePairs


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

for dataset_a, dataset_b in [('gov', 'osm'), ('gov', 'msft'), ('osm', 'msft')]:
    datasets = []
    for nuts_id in sample_nuts_ids:

        print(f'Processing: {dataset_a} <- {dataset_b} - {len(datasets)+1}/{len(sample_nuts_ids)} - {nuts_id}')
        filepath_a = f'/p/projects/eubucco/data/3-attrib-cleaning-v1-{dataset_a}/{nuts_id}.parquet'
        filepath_b = f'/p/projects/eubucco/data/3-attrib-cleaning-v1-{dataset_b}/{nuts_id}.parquet'

        try:
            gdf_a = gpd.read_parquet(filepath_a, columns=['id', 'geometry'])
            gdf_b = gpd.read_parquet(filepath_b, columns=['id', 'geometry'])

        except FileNotFoundError:
            print(f'File not found for {dataset_a} and {dataset_b} in {nuts_id}')
            continue

        gdf_a = gdf_a.reset_index(drop=True).add_prefix(f'{nuts_id}_{dataset_a}_', axis=0)
        gdf_b = gdf_b.reset_index(drop=True).add_prefix(f'{nuts_id}_{dataset_b}_', axis=0)

        cp = create_candidate_pairs_dataset(
            gdf1=gdf_a,
            gdf2=gdf_b,
            neighborhoods=neighborhoods,
            h3_res=9,
        )

        # # Create new index (format: <NUTS-ID>_<DATASET-TYPE>_<INT>) and update the pairs respectively
        # cp.dataset_a = cp.dataset_a.reset_index(drop=False)
        # cp.dataset_b = cp.dataset_b.reset_index(drop=False)
        # cp.dataset_a = cp.dataset_a.add_prefix(f'{nuts_id}_{dataset_a}_', axis=0)
        # cp.dataset_b = cp.dataset_b.add_prefix(f'{nuts_id}_{dataset_b}_', axis=0)

        # old_new_index_mapping_a = cp.dataset_a.reset_index().set_index('id')['index']
        # old_new_index_mapping_b = cp.dataset_b.reset_index().set_index('id')['index']
        # cp.pairs['id_existing'] = cp.pairs['id_existing'].map(old_new_index_mapping_a)
        # cp.pairs['id_new'] = cp.pairs['id_new'].map(old_new_index_mapping_b)

        datasets.append(cp)

    cp = CandidatePairs(
            dataset_a=pd.concat([cp.dataset_a for cp in datasets]),
            dataset_b=pd.concat([cp.dataset_b for cp in datasets]),
            pairs=pd.concat([cp.pairs for cp in datasets], ignore_index=True),
        )
    cp.save(
            f'/p/projects/eubucco/data/conflation-training-data/neighborhood-candidate-pairs-{dataset_a}-{dataset_b}.pickle'
        )
