import random

import pandas as pd
import geopandas as gpd

from geo_matcher.dataset import create_candidate_pairs_dataset
from geo_matcher.candidate_pairs import CandidatePairs


sample_size = 20
path_lau = '/p/projects/eubucco/data/0-raw-data/lau/lau_nuts.gpkg'
lau = gpd.read_file(path_lau)
datasets_all = []

nuts_ids = set(lau['NUTS_ID'])
included_nuts_ids = set(CandidatePairs.load(
    '/p/projects/eubucco/data/conflation-training-data/candidate-pairs.pickle'
).dataset_b.index.str.split('_').str[0].unique())

for dataset_a, dataset_b in [('gov', 'osm'), ('gov', 'msft'), ('osm', 'msft')]:
    datasets = []
    while len(datasets) < sample_size:
        nuts_id = random.sample(list(nuts_ids - included_nuts_ids), 1)[0]

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
            overlap_range=(0, 0.01),
            similarity_range=None,
            max_distance=10,
            max_overlap_others=0.25,
            n=25,
            h3_res=9,
        )

        datasets.append(cp)
        included_nuts_ids.add(nuts_id)

    cp = CandidatePairs(
            dataset_a=pd.concat([cp.dataset_a for cp in datasets]),
            dataset_b=pd.concat([cp.dataset_b for cp in datasets]),
            pairs=pd.concat([cp.pairs for cp in datasets]).reset_index(drop=True),
        )
    cp.save(
            f'/p/projects/eubucco/data/conflation-training-data/candidate-pairs-{dataset_a}-{dataset_b}-non-overlapping.pickle'
        )
    
    datasets_all.append(cp)

candidate_pairs = CandidatePairs(
        dataset_a=pd.concat([cp.dataset_a for cp in datasets_all]),
        dataset_b=pd.concat([cp.dataset_b for cp in datasets_all]),
        pairs=pd.concat([cp.pairs for cp in datasets_all]).reset_index(drop=True),
    )
candidate_pairs.save('/p/projects/eubucco/data/conflation-training-data/candidate-pairs-non-overlapping.pickle')
