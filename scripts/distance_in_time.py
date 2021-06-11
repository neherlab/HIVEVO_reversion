import json
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO, Phylo


def get_reference_sequence(filename):
    """
    Loads the reference sequence from the given file and returns it. Can either be the consensus sequence
    (from fasta file), or the root sequence (from the nt_muts.json file from TreeTime).
    """
    # Root sequence from tree file
    if ".json" in filename:
        with open(filename) as f:
            data = json.load(f)
            reference_sequence = list(data["nodes"]["NODE_0000000"]["sequence"])
        reference_sequence = np.array(reference_sequence)
    # The consensus sequence from fasta file
    elif ".fasta" in filename:
        consensus = AlignIO.read(filename, "fasta")
        reference_sequence = np.array(consensus)[0]
    return reference_sequence


def get_gap_mask(alignment_array, threshold=0.1):
    """
    Return a vector were true are the sites seen with less than threshold fraction of N.
    """
    gaps = alignment_array == "N"
    gap_proportion = np.sum(gaps, axis=0, dtype=int) / gaps.shape[0]
    return gap_proportion < threshold


def get_mean_distance_in_time(alignment_file, reference_sequence, subtype="B"):
    """
    Returns the time, average distance and standard deviation of the average distance to the reference
    sequence.
    """
    # Data loading
    alignment = AlignIO.read(alignment_file, "fasta")
    alignment_array = np.array(alignment)
    names = [seq.id for seq in alignment]
    dates = np.array([int(name.split(".")[2]) for name in names])
    subtypes = np.array([name.split(".")[0] for name in names])

    # Selecting subtype
    alignment_array = alignment_array[subtypes == subtype]
    dates = dates[subtypes == subtype]

    # Distance to consensus sequence
    gap_mask = get_gap_mask(alignment_array)
    distance_matrix = (alignment_array != reference_sequence)[:, gap_mask]
    distance = np.sum(distance_matrix, axis=1, dtype=int) / distance_matrix.shape[-1]

    # Distance average per year
    average_distance = []
    std_distance = []
    years = np.unique(dates)
    for year in years:
        average_distance += [np.mean(distance[dates == year])]
        std_distance += [np.std(distance[dates == year])]

    average_distance_in_time = np.array(average_distance)
    std_distance_in_time = np.array(std_distance)

    return years, average_distance_in_time, std_distance_in_time


def get_root_to_tip_distance(tree_file, branch_length_file):
    """
    Computes the mean root to tipe distance for each year.
    Returns a list of mean root_to_tip distance and a list of corresponding years.
    """

    # Loading the tree and branch_length file
    tree = Phylo.read(tree_file, "newick")
    tips = tree.get_terminals()

    with open(branch_length_file) as f:
        file = json.load(f)
        nodes = file["nodes"]

    # Replacing the branch length by the mutation_length entry of the branch_length_file
    for tip in tips:
        tip.branch_length = nodes[tip.name]["mutation_length"]

    # Getting the years of the tips
    dates = []
    for tip in tips:
        date = tip.name.split(".")[2]
        dates += [int(date)]
    dates = np.unique(dates)

    mean_lengths = []
    for date in dates:
        lengths = [tree.distance(tip) for tip in tips if int(tip.name.split(".")[2]) == date]
        mean_lengths += [np.mean(lengths)]

    return dates, mean_lengths


def plot_mean_distance_in_time(consensus="global", savefig=False):
    """
    Plots the figure for the mean  hamiltonian distance in time.
    Consensus = True to compare to the consensus sequence from the alignment. Set to False to compare to root
    of the tree instead.
    """

    assert consensus in ["global", "root",
                         "subtypes"], "Reference sequence can only be 'global' 'root' or 'subtypes'"

    alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    if consensus == "global":
        reference_file = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"
    elif consensus == "root":
        reference_file = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    elif consensus == "subtypes":
        reference_file = {}
        reference_file["B"] = "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta"
        reference_file["C"] = "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta"

    plt.figure()
    subtypes = ["B", "C"]
    colors = ["C0", "C1"]
    fontsize = 16
    c = 0
    for subtype in subtypes:
        if consensus != "subtypes":
            reference_sequence = get_reference_sequence(reference_file)
        else:
            reference_sequence = get_reference_sequence(reference_file[subtype])

        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        fit = np.polyfit(years, dist, deg=1)
        # fit = np.polyfit(years[std != 0], dist[std != 0], deg=1, w=(1 / std[std != 0]))
        plt.errorbar(years, dist, yerr=std, fmt=".", label=subtype, color=colors[c])
        plt.plot(years, np.polyval(fit, years), "--",
                 color=colors[c], label=f"{round(fit[0],5)}x + {round(fit[1],5)}")
        c += 1

    plt.grid()
    plt.xlabel("Time [years]", fontsize=fontsize)
    plt.ylabel("Average fraction difference", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if consensus == "global":
        plt.title("Distance to global consensus", fontsize=fontsize)
        if savefig:
            plt.savefig("figures/Distance_to_consensus.png", format="png")
    elif consensus == "root":
        plt.title("Distance to root", fontsize=fontsize)
        if savefig:
            plt.savefig("figures/Distance_to_root.png", format="png")
    elif consensus == "subtypes":
        plt.title("Distance to subtype consensus", fontsize=fontsize)
        if savefig:
            plt.savefig("figures/Distance_to_subtype_consensus.png", format="png")


def plot_root_to_tip(savefig):
    """
    Plots the figure for the root to tip distance in time.
    """
    tree_file = "data/BH/intermediate_files/timetree_pol_1000.nwk"
    branch_length_file = "data/BH/intermediate_files/branch_lengths_pol_1000.json"
    fontsize = 16

    dates, lengths = get_root_to_tip_distance(tree_file, branch_length_file)

    plt.figure()
    plt.plot(dates, lengths, '.', label="Data")
    fit = np.polyfit(dates, lengths, deg=1)
    plt.plot(dates, np.polyval(fit, dates), "--", label=f"{round(fit[0],5)}x + {round(fit[1],5)}")
    plt.xlabel("Time [years]", fontsize=fontsize)
    plt.ylabel("Mean root-tip length", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.title("Root to tip distance", fontsize=fontsize)
    if savefig:
        plt.savefig("figures/Distance_to_root.png", format="png")


if __name__ == '__main__':
    savefig = False
    plot_mean_distance_in_time("global", savefig)
    plot_mean_distance_in_time("root", savefig)
    plot_mean_distance_in_time("subtypes", savefig)
    plot_root_to_tip(savefig)
    plt.show()
