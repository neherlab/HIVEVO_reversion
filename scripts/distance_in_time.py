import json
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO


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
    else:
        plt.title("Distance to root", fontsize=fontsize)


if __name__ == '__main__':
    plot_mean_distance_in_time(True)
    plot_mean_distance_in_time(False)
    plt.show()
