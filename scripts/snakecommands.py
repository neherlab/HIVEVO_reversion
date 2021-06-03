# Scripts used for the snakefile commands
import click
import sys
import pandas as pd
from Bio import SeqIO


@click.group()
def cli():
    pass


@cli.command()
@click.argument("sequences", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def metadata_from_names(sequences, output):
    """
    Creates an OUTPUT metadata tsv file from the SEQUENCES file using the sequences names.
    """

    sequences = list(SeqIO.parse(sequences, "fasta"))
    columns = ["strain", "virus", "date", "subtype", "country"]
    raw_names = [seq.name for seq in sequences]
    df = pd.DataFrame(data=None, index=None, columns=columns)
    for raw_name in raw_names:
        raw_name_split = raw_name.split(".")
        date = raw_name_split[2] + "-XX-XX"
        subtype = raw_name_split[0]
        country = raw_name_split[1]

        tmp_col = [raw_name]
        tmp_col.append("HIV")
        tmp_col.append(date)
        tmp_col.append(subtype)
        tmp_col.append(country)
        tmp = pd.DataFrame(data=[tmp_col], columns=columns)
        df = df.append(tmp)

    df.to_csv(output, index=False, sep="\t")


if __name__ == '__main__':
    cli()
