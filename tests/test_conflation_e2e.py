from pathlib import Path
import os
import glob

from conflation.conflation import conflate, conflate_gov_osm_msft

def test_conflation_e2e():
    """
    Test the end-to-end conflation process.
    """
    region_id = 'ES630-88391aa985fffff'
    model_path = Path('data', 'train', 'xgboost-model-eubucco-v1.json')
    test_dir = Path('data', 'conflation-test-regions')
    gov_dir = test_dir / 'gov'
    osm_dir = test_dir / 'osm'
    msft_dir = test_dir / 'msft'
    matching_dir = test_dir / 'matching'
    out_dir = test_dir / 'merged'

    # Remove matching_dir and result files for region
    for dir_path in [out_dir]:
        files = glob.glob(os.path.join(dir_path, "**", f"{region_id}.parquet"), recursive=True)
        for file in files:
            os.remove(file)
    
    conflate_gov_osm_msft(
        region_id=region_id,
        gov_dir=gov_dir,
        osm_dir=osm_dir,
        msft_dir=msft_dir,
        matching_dir=matching_dir,
        out_dir=out_dir,
        h3_res=9,
        model_path=model_path,
        db_version='1.0,'
    )

    # Remove matching_dir and result files for region
    for dir_path in [out_dir]:
        files = glob.glob(os.path.join(dir_path, "**", f"{region_id}.parquet"), recursive=True)
        for file in files:
            os.remove(file)

    conflate(
        datasets=['gov', 'osm', 'msft'],
        region_id=region_id,
        data_dirs_input=[gov_dir, osm_dir, msft_dir],
        data_dirs_matching=[matching_dir / 'osm', matching_dir / 'msft'],
        data_dir_results=out_dir,
        attribute_mapping=[True, True],
        h3_res=9,
        model_path=model_path,
        db_version='1.0,'
    )


if __name__ == "__main__":
    test_conflation_e2e()
