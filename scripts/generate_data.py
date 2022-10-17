import trajectory
import divergence
import gtr_modeling
import bootstrap_BH
import click
import shutil
import os

WH_DATA_FOLDER = "data/WH/"
BH_DATA_FOLDER = "data/BH/"
MODELING_DATA_FOLDER = "data/modeling/"


@click.group()
def cli():
    pass


@cli.command()
def make_data():
    """
    Creates all the intermediate data necessary for the figures. One needs to run the snakemake analysis
    beforehand (snakemake figure_data).
    """
    print()
    print("--- Generating BH bootstraps data ---")
    bootstrap_BH.make_intermediate_data(BH_DATA_FOLDER)

    if not os.path.exists(WH_DATA_FOLDER):
        os.mkdir(WH_DATA_FOLDER)
    print()
    print("--- Generating trajectory data ---")
    trajectory.make_intermediate_data(WH_DATA_FOLDER)
    print()
    print("--- Generating divergence data ---")
    divergence.make_intermediate_data(WH_DATA_FOLDER)

    if not os.path.exists(MODELING_DATA_FOLDER):
        os.mkdir(MODELING_DATA_FOLDER)
    print()
    print("--- Generating modeling data ---")
    gtr_modeling.make_intermediate_data(MODELING_DATA_FOLDER)


@cli.command()
def clean_data():
    "Removes all the intermediate data generated for the WH analysis."
    shutil.rmtree(WH_DATA_FOLDER)


if __name__ == '__main__':
    cli()
