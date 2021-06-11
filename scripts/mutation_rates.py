import json
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO, Phylo
from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance


def compute_mutation_rates():
    """
    Returns the mutation rates from the distance to root, distance to subtype consensus and root to tip
    distance.
    """
    # The files used to compute the rates
    reference_file = {}
    reference_file["root"] = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    reference_file["B"] = "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta"
    reference_file["C"] = "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta"
    alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    tree_file = "data/BH/intermediate_files/timetree_pol_1000.nwk"
    branch_length_file = "data/BH/intermediate_files/branch_lengths_pol_1000.json"

    # Rates from hamming distance
    rates = {"root": {}, "subtypes": {}}

    reference_sequence = get_reference_sequence(reference_file["root"])
    for subtype in ["B", "C"]:
        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        fit = np.polyfit(years, dist, deg=1)
        rates["root"][subtype] = fit[0]

    for subtype in ["B", "C"]:
        reference_sequence = get_reference_sequence(reference_file[subtype])
        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        fit = np.polyfit(years, dist, deg=1)
        rates["subtypes"][subtype] = fit[0]

    # Rates from root to tip distance
    dates, lengths = get_root_to_tip_distance(tree_file, branch_length_file)
    fit = np.polyfit(dates, lengths, deg=1)
    rates["rtt"] = fit[0]

    return rates


def plot_mutation_rates():
    """
    Plot for the mutation rates.
    """
    rates = compute_mutation_rates()
    labels = ["Hamming root", "Hamming subtype", "RTT", "WH"]

    fontsize = 16
    markersize = 16

    plt.figure()
    for ii, key in enumerate(["root", "subtypes"]):
        plt.plot(ii, rates[key]["B"]*1e4, '.', color="C0", markersize=markersize)
        plt.plot(ii, rates[key]["C"]*1e4, '.', color="C1", markersize=markersize)
    plt.plot(2, rates["rtt"]*1e4, '.', markersize=markersize)
    plt.plot(3, 19, '.', color="C0", markersize=markersize)
    plt.xticks(range(4), labels, fontsize=fontsize, rotation=8)
    plt.ylabel("Mutation rates (per year) * e-4", fontsize=fontsize)
    plt.legend(["B", "C"], fontsize=fontsize)
    plt.grid()
    plt.savefig("figures/mutation_rates.png", format="png")
    plt.show()


if __name__ == '__main__':
    plot_mutation_rates()
