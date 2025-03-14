# Building Footprint Conflation Tool

A web-based tool for conflating building footprint datasets, visually identifying duplicates, and labeling them. Runs locally as a Flask app, displaying Folium-generated maps of potential duplicates.


## Install
```bash
pip install git+https://github.com/ai4up/eubucco-conflation@main
```

## Usage
Create a dataset of potential duplicate buildings for manual inspection and labeling from two footprint datasets:
```bash
conflator create-labeling-dataset dataset1.parquet dataset2.parquet
```

Initiate browser-based labeling:
```bash
conflator label
```

## Demo
Create a dataset of potential duplicates of [government buildings](https://eubucco.com/data/) and [Microsoft buildings](https://github.com/microsoft/GlobalMLBuildingFootprints) for a small region in France using the [demo data](data/) in the repository. Include only buildings which overlap slightly (0-20%).
```bash
conflator create-labeling-dataset \
    --min-intersection=0.0 \ # Minimum relative overlap for new buildings to be considered for duplicate labeling [0,1)
    --max-intersection=0.2 \ # Maximum relative overlap for new buildings to be considered for duplicate labeling (0,1]
    --distance=100 \ # Distance threshold for displaying neighboring buildings [meters]
    data/demo-gov.parquet data/demo-microsoft.parquet
```
The resulting dataset is locally stored as `candidates.parquet`. To initiate the browser-based labeling, run:
```bash
conflator label
```
![Example of Building Footprint Conflation Tool](example.png)


## Development

Install dev dependencies using [poetry](https://python-poetry.org/):
```bash
poetry install --only dev
```

Install git pre-commit hooks:
```bash
pre-commit install
```

Build from source:
```bash
poetry build
pip install dist/eubucco_conflator-*.whl
```
