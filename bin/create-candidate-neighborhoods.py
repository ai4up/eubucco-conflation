import pandas as pd
import geopandas as gpd

from geo_matcher.dataset import create_candidate_pairs_dataset
from geo_matcher.candidate_pairs import CandidatePairs


sample_size = 50
path_lau = '/p/projects/eubucco/data/0-raw-data/lau/lau_nuts.gpkg'
lau = gpd.read_file(path_lau)
datasets_all = []

lau = lau.drop_duplicates(subset=['NUTS_ID'])
lau['country'] = lau['NUTS_ID'].str[:2]
sample_nuts_ids = list(lau.groupby('country')['NUTS_ID'].sample(n=1, random_state=42)) + list(lau['NUTS_ID'].sample(n=sample_size - len(lau['country'].unique()), random_state=42))
print(f"Sample regions: {', '.join(sample_nuts_ids)}")

# sum(os.path.isfile(f'/p/projects/eubucco/data/3-attrib-cleaning-v1-gov/{nuts_id}.parquet') for nuts_id in sample_nuts_ids)

for dataset_a, dataset_b in [('gov', 'osm'), ('gov', 'msft'), ('osm', 'msft')]:
    datasets = []
    for nuts_id in sample_nuts_ids:

        print(f'Processing: {dataset_a} <- {dataset_b} - {len(datasets)+1}/{sample_size} - {nuts_id}')
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
            n_neighborhoods=1,
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
            pairs=pd.concat([cp.pairs for cp in datasets]),
        )
    cp.save(
            f'/p/projects/eubucco/data/conflation-training-data/neighborhood-candidate-pairs-{dataset_a}-{dataset_b}.pickle'
        )
