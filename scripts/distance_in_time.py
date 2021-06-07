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


def get_root_to_tip_distance(tree_file, clock_file):
    """
    Computes the mean root to tipe distance for each year.
    Returns a list of mean root_to_tip distance and a list of corresponding years.
    """
    tree = Phylo.read(tree_file, "newick")
    with open(clock_file, "r") as file:
        clock = json.load(file)
        clock = clock["clock"]["rate"]

    # Loading the tips of the tree
    tips = tree.get_terminals()

    # Getting the years of the tips
    dates = []
    for tip in tips:
        date = tip.name.split(".")[2]
        dates += [int(date)]
    dates = np.unique(dates)

    mean_lengths = []
    for date in dates:
        tmp = [tip for tip in tips if int(tip.name.split(".")[2]) == date]
        lengths = [tree.distance(tip) for tip in tmp]
        mean_lengths += [np.mean(lengths) / clock]

    return dates, mean_lengths


def plot_mean_distance_in_time(consensus=True):
    """
    Plots the figure for the mean  hamiltonian distance in time.
    Consensus = True to compare to the consensus sequence from the alignment. Set to False to compare to root
    of the tree instead.
    """
    alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    if consensus:
        reference_file = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"
    else:
        reference_file = "data/BH/intermediate_files/pol_1000_nt_muts.json"

    plt.figure()
    subtypes = ["B", "C"]
    colors = ["C0", "C1"]
    fontsize = 16
    c = 0
    for subtype in subtypes:
        reference_sequence = get_reference_sequence(reference_file)
        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        fit = np.polyfit(years[std != 0], dist[std != 0], deg=1, w=(1 / std[std != 0]))
        plt.errorbar(years, dist, yerr=std, fmt=".", label=subtype, color=colors[c])
        plt.plot(years, np.polyval(fit, years), "--",
                 color=colors[c], label=f"{round(fit[0],5)}x + {round(fit[1],5)}")
        c += 1

    plt.grid()
    plt.xlabel("Time [years]", fontsize=fontsize)
    plt.ylabel("Average fraction difference", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if consensus:
        plt.title("Distance to consensus", fontsize=fontsize)
        plt.savefig("figures/Distance_to_consensus.png", format="png")
    else:
        plt.title("Distance to root", fontsize=fontsize)
        plt.savefig("figures/Distance_to_root.png", format="png")


def plot_root_to_tip():
    """
    Plots the figure for the root to tip distance in time.
    """
    tree_file = "data/BH/intermediate_files/timetree_pol_1000.nwk"
    clock_file = "data/BH/intermediate_files/branch_lengths_pol_1000.json"
    fontsize = 16

    dates, lengths = get_root_to_tip_distance(tree_file, clock_file)

    plt.figure()
    plt.plot(dates, lengths, '.', label="Data")
    fit = np.polyfit(dates, lengths, deg=1)
    plt.plot(dates, np.polyval(fit, dates), "--", label=f"{round(fit[0],5)}x + {round(fit[1],5)}")
    plt.xlabel("Time [years]", fontsize=fontsize)
    plt.ylabel("Mean root-tip length [years]", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()


if __name__ == '__main__':
    # plot_mean_distance_in_time(True)
    # plot_mean_distance_in_time(False)
    plot_root_to_tip()
    plt.show()
