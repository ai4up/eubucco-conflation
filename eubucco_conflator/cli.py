import click

from eubucco_conflator import app, dataset
from eubucco_conflator.state import CANDIDATES_FILE, State


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("filepath", default="candidates.parquet", type=click.Path(exists=True))
def label(filepath: str) -> None:
    """
    Start labeling of duplicated buildings.

    FILEPATH to GeoParquet file containing the potential duplicates to label.
    """
    State.init(filepath, logger=click.echo)
    click.echo(f"Loaded {len(State.gdf)} buildings")
    click.echo(
        f"Loaded latest labeling state: {len(State.results)} buildings already labeled"
    )
    click.echo(f"Starting labeling of {len(State.candidates)} buildings...")

    click.echo("Starting browser app...")
    app.start()


@cli.command()
@click.argument("filepath1", required=True, type=click.Path(exists=True))
@click.argument("filepath2", required=True, type=click.Path(exists=True))
@click.option(
    "--id-col",
    default=None,
    help="Name of the column containing unique building identifiers (default index).",
)
@click.option(
    "--min-intersection",
    "-l",
    default=0.0,
    help="Minimum relative overlap for new buildings to be considered for duplicate labeling [0,1).",  # noqa: E501
)
@click.option(
    "--max-intersection",
    "-u",
    default=1.0,
    help="Maximum relative overlap for new buildings to be considered for duplicate labeling (0,1].",  # noqa: E501
)
@click.option(
    "--distance",
    "-d",
    default=100,
    help="Distance threshold for displaying neighboring buildings [meters].",
)
def create_labeling_dataset(
    filepath1: str,
    filepath2: str,
    id_col: str,
    min_intersection: float,
    max_intersection: float,
    distance: int,
) -> None:
    """
    Create a dataset of potential duplicate buildings.

    FILEPATH1 to GeoParquet file containing the reference (existing) buildings.
    FILEPATH2 to GeoParquet file containing the new (to be added) buildings.
    """
    click.echo("Loading geodata...")
    dataset.create_duplicate_candidates_dataset(
        filepath1,
        filepath2,
        id_col,
        (min_intersection, max_intersection),
        distance,
        logger=click.echo,
    )
    click.echo(
        f"Dataset of duplicate candidates created and stored in {CANDIDATES_FILE}"
    )


if __name__ == "__main__":
    cli()
