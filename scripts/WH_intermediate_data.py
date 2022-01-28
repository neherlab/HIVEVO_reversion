import trajectory
import divergence
import gtr_modeling
import click
import shutil

WH_DATA_FOLDER = "data/WH/"


@click.group()
def cli():
    pass


@cli.command()
def make_data(folder_path=WH_DATA_FOLDER):
    "Creates all the intermediate data necessary for the WH analysis."
    print("--- Generating trajectory data ---")
    trajectory.make_intermediate_data(folder_path)
    print("--- Generating divergence data ---")
    divergence.make_intermediate_data(folder_path)
    print("--- Generating modeling data ---")
    gtr_modeling.make_intermediate_data()


@cli.command()
def clean_data(folder_path=WH_DATA_FOLDER):
    "Removes all the intermediate data generated for the WH analysis."
    shutil.rmtree(WH_DATA_FOLDER)


if __name__ == '__main__':
    cli()
