import json
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO, Phylo
from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance


def compute_mutation_rates(nb_sequences = [1000, 500, 250, 125, 60, 30]):
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
    tree_file = {}
    branch_length_file = {}
    for nb in nb_sequences:
        tree_file[str(nb)] = "data/BH/intermediate_files/timetree_pol_" + str(nb) + ".nwk"
        branch_length_file[str(nb)] = "data/BH/intermediate_files/branch_lengths_pol_" + str(nb) + ".json"

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
    dates = {}
    lengths = {}
    rates["rtt"] = {}

    for key in tree_file.keys():
        dates[key], lengths[key] = get_root_to_tip_distance(tree_file[key], branch_length_file[key])
        fit = np.polyfit(dates[key], lengths[key], deg=1)
        rates["rtt"][key] = fit[0]

    return rates


def plot_mutation_rates():
    """
    Plot for the mutation rates.
    """
    rates = compute_mutation_rates()
    labels = ["Hamming root", "Hamming subtype", "RTT", "WH"]

    fontsize = 16
    markersize = 16
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    plt.figure()
    for ii, key in enumerate(["root", "subtypes"]):
        if ii:
            plt.plot(ii, rates[key]["B"]*1e4, '.', color="C0", markersize=markersize, label="B")
            plt.plot(ii, rates[key]["C"]*1e4, '.', color="C1", markersize=markersize, label="C")
        else:
            plt.plot(ii, rates[key]["B"]*1e4, '.', color="C0", markersize=markersize)
            plt.plot(ii, rates[key]["C"]*1e4, '.', color="C1", markersize=markersize)

    for ii, key in enumerate(rates["rtt"].keys()):
        plt.plot(2, rates["rtt"][key]*1e4, '.', color=colors[ii+2], markersize=markersize, label=key)
    plt.plot(3, 19, '.', color="C0", markersize=markersize)
    plt.xticks(range(4), labels, fontsize=fontsize, rotation=8)
    plt.ylabel("Mutation rates (per year) * e-4", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.ylim([0, 20])
    plt.savefig("figures/mutation_rates.png", format="png")
    plt.show()


if __name__ == '__main__':
    plot_mutation_rates()
    # gitkrakentest
