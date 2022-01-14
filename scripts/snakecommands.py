"""
Scripts used for the snakefile commands
"""
import click
import json
import pandas as pd
import numpy as np
from Bio import SeqIO, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import SingleLetterAlphabet
from Bio.Align import MultipleSeqAlignment
from treetime import TreeAnc, TreeTime
from treetime.utils import parse_dates


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


@cli.command()
@click.argument("sequences", type=click.Path(exists=True))
@click.argument("metadata", type=click.Path(exists=True))
@click.argument("number", type=int)
@click.argument("output_sequences", type=click.Path(exists=False))
@click.argument("output_metadata", type=click.Path(exists=False))
@click.option("--remove_subtype_o", is_flag=True, help="Removes subtype O from the sequences.")
@click.option("--remove_subtype_n", is_flag=True, help="Removes subtype N from the sequences.")
def subsample(sequences, metadata, number, output_sequences, output_metadata,
              remove_subtype_o, remove_subtype_n):
    """
    Subsamples the data in SEQUENCES and METADATA homegeneously in time and returns OUTPUT_SEQUENCES and
    OUTPUT_METADATA.
    """

    df = pd.read_csv(metadata, sep='\t')
    if remove_subtype_o:
        df = df[df["subtype"] != "O"]  # removing subtype O
    if remove_subtype_n:
        df = df[df["subtype"] != "N"]  # removing subtype N
    df["date"] = [int(date[:4]) for date in df["date"]]
    df = df.sort_values("date")

    # Computing number of sequences for each year
    hist, bins = np.histogram(df["date"], bins=np.arange(1976, 2022))
    hist = hist[:-1]
    bins = bins[:-1]
    average_per_bin = float(number) / len(bins)
    nb_per_bin = np.zeros_like(hist)
    nb_per_bin[hist < average_per_bin] = hist[hist < average_per_bin]
    nb_per_bin[hist > average_per_bin] = average_per_bin

    while np.sum(nb_per_bin) < int(number):
        nb_per_bin[hist >= nb_per_bin + 1] += 1

    # adjust to have exactly the desired number of sequences
    while np.sum(nb_per_bin) > int(number):
        ii = np.argmax(nb_per_bin)
        nb_per_bin[ii] -= 1

    # Getting the sequences names
    seq_names = []
    for ii, year in enumerate(bins[:-1]):
        if nb_per_bin[ii] == hist[ii]:
            seq_names = seq_names + df[df["date"] == year]["strain"].to_list()
        else:
            permut = np.random.permutation(hist[ii])
            names = df[df["date"] == year]["strain"].to_numpy()
            names = names[permut[:nb_per_bin[ii]]].tolist()
            seq_names = seq_names + names

    # Selecting the given sequences
    for ii, seq_name in enumerate(seq_names):
        if ii == 0:
            output_df = df[df["strain"] == seq_name]
        else:
            output_df = output_df.append(df[df["strain"] == seq_name])

    # Reformating the date field for TreeTime
    output_df["date"] = [str(date) + "-XX-XX" for date in output_df["date"]]

    # Creating the output sequences
    sequences = list(SeqIO.parse(sequences, "fasta"))
    sequences = [seq for seq in sequences if seq.name in seq_names]

    # Cleaning the sequences (some characters are non ATGC-N sometimes)
    for ii in range(len(sequences)):
        seq = np.array(sequences[ii])
        tmp1 = np.logical_and(seq != "a", seq != "t")
        tmp2 = np.logical_and(seq != "g", seq != "c")
        tmp3 = np.logical_and(seq != "-", seq != "n")
        tmp4 = np.logical_and(tmp1, tmp2)
        tmp5 = np.logical_and(tmp4, tmp3)
        seq[tmp5] = "n"
        sequences[ii].seq = Seq("".join(seq))

    # Creating the output files
    output_df.to_csv(output_metadata, index=False, sep="\t")
    SeqIO.write(sequences, output_sequences, "fasta")


@cli.command()
@click.argument("alignment", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def consensus(alignment, output):
    """
    Computes the OUTPUT consensus sequences from the ALIGNMENT.
    """
    alignment_file = alignment
    alignment = AlignIO.read(alignment, "fasta")
    alignment_array = np.array(alignment)

    # Consensus computation
    consensus_sequence = []
    for ii in range(alignment_array.shape[-1]):
        values, counts = np.unique(alignment_array[:, ii], return_counts=True)
        consensus_nucleotide = values[np.argmax(counts)]
        consensus_sequence = consensus_sequence + [consensus_nucleotide]

    consensus_sequence = np.array(consensus_sequence)
    consensus_sequence = Seq("".join(consensus_sequence))
    consensus_sequence = SeqRecord(
        seq=consensus_sequence, id=f"Consensus_{alignment_file}", name="", description="")

    with open(output, "w") as handle:
        SeqIO.write([consensus_sequence], handle, "fasta")


@cli.command()
@click.argument("alignment", type=click.Path(exists=True))
def split_positions(alignment):
    """
    Subsamples the given ALIGNMENT into 1st 2nd and 3rd positions
    """
    alignment_file = alignment
    alignment = AlignIO.read(alignment, "fasta")
    alignment_array = np.array(alignment)

    basename = alignment_file.replace(".fasta", "")

    for position, name in zip([0, 1, 2], ["1st", "2nd", "3rd"]):
        sub_alignment = alignment_array[:, position::3]
        seq_list = []
        for ii in range(alignment_array.shape[0]):
            seq = "".join(sub_alignment[ii, :])
            seq_list += [SeqRecord(Seq(seq, SingleLetterAlphabet()), id=alignment[ii].id,
                                   name=alignment[ii].name, description="")]

        sub_alignment = MultipleSeqAlignment(seq_list)
        AlignIO.write([sub_alignment], basename + "_" + name + ".fasta", "fasta")


@cli.command()
@click.argument("tree", type=click.Path(exists=True))
@click.argument("alignment", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def gtr(tree, alignment, output):
    """
    Infers a the OUTPUT GTR model from the TREE and ALIGNMENT.
    """
    alignment_file = alignment
    alignment = AlignIO.read(alignment, "fasta")
    tt = TreeAnc(tree=tree, aln=alignment_file)
    gtr = tt.infer_gtr(marginal=True, normalized_rate=False)
    output_JSON(gtr, output)


@cli.command()
@click.argument("refine", type=click.Path(exists=True))
@click.argument("gtr", type=click.Path(exists=True))
@click.argument("gtr1", type=click.Path(exists=True))
@click.argument("gtr2", type=click.Path(exists=True))
@click.argument("gtr3", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def mutation_rate(refine, gtr, gtr1, gtr2, gtr3, output):
    """
    Returns the OUTPUT mutation rate (per site per year) from the TreeTime REFINE file and the GTR files.
    """
    rate_dict = {}
    rate_dict["all"] = get_mutation_rate(gtr, refine)
    rate_dict["first"] = get_mutation_rate(gtr1, refine)
    rate_dict["second"] = get_mutation_rate(gtr2, refine)
    rate_dict["third"] = get_mutation_rate(gtr3, refine)

    with open(output, "w") as output:
        json.dump(rate_dict, output)

    return mutation_rate


@cli.command()
@click.argument("refine", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def mean_branch_length(refine, output):
    """
    Computes the OUTPUT mean branch length for the given REFINE file.
    """
    with open(refine, "r") as file:
        refine = json.load(file)

    total_length = 0
    for key in refine["nodes"].keys():
        total_length += refine["nodes"][key]["branch_length"]
    mean_length = total_length / len(refine["nodes"])
    mean_length = mean_length / refine["clock"]["rate"]

    mean_length = {"mean_branch_length": mean_length}

    with open(output, "w") as o:
        json.dump(mean_length, o)


@cli.command()
@click.argument("alignment", type=click.Path(exists=True))
def split_subtypes(alignment):
    """
    Split the original ALIGNMENT into subtype B and C and saves the 2 sub alignments for these subtypes.
    """
    alignment_file = alignment
    alignment = AlignIO.read(alignment, "fasta")

    basename = alignment_file.replace(".fasta", "")

    for subtype in ["B", "C"]:
        seq_list = [seq for seq in alignment if seq.id[0] == subtype]
        sub_alignment = MultipleSeqAlignment(seq_list)
        AlignIO.write([sub_alignment], basename + "_" + subtype + ".fasta", "fasta")


@cli.command()
@click.argument("tree", type=click.Path(exists=True))
@click.argument("alignment", type=click.Path(exists=True))
@click.argument("metadata", type=click.Path(exists=True))
def reroot_tree(tree, alignment, metadata):
    """
    Reroot the TREE to best root. TREE must be in newick format.
    """
    tt = Phylo.read(tree, "newick")
    dates = parse_dates(metadata)
    ttree = TreeTime(gtr='Jukes-Cantor', tree=tt, precision=1, aln=alignment, verbose=2, dates=dates)
    ttree.reroot()
    Phylo.write(ttree._tree, tree, "newick")


def get_mutation_rate(gtr_file, refine_file):
    """
    Returns the mutation rate in mutation per site per year from the GTR model and its corresponding
    tree file from augur refine output.
    """
    with open(gtr_file, "r") as file:
        gtr = json.load(file)
    with open(refine_file, "r") as file:
        refine = json.load(file)

    mutation_rate = gtr["mu"] * refine["clock"]["rate"]

    return mutation_rate


def output_JSON(gtr_model, output_file):
    gtr = gtr_model.__dict__
    save_dict = {}
    for key in ["_mu", "_Pi", "_W"]:
        if isinstance(gtr[key], np.ndarray):
            save_dict[key.replace("_", "")] = gtr[key].tolist()
        else:
            save_dict[key.replace("_", "")] = gtr[key]

    with open(output_file, 'w') as output:
        json.dump(save_dict, output)


if __name__ == '__main__':
    cli()
