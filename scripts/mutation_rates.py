import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO, Phylo
from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance
import divergence
plt.style.use("tex")


def compute_mutation_rates(region):
    """
    Returns the mutation rates from the distance to root, distance to subtype consensus and root to tip
    distance.
    """
    # BH rate files
    reference_file = {}
    reference_file["root"] = f"data/BH/intermediate_files/{region}_1000_nt_muts.json"
    reference_file["B"] = f"data/BH/alignments/to_HXB2/{region}_1000_B_consensus.fasta"
    reference_file["C"] = f"data/BH/alignments/to_HXB2/{region}_1000_C_consensus.fasta"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    tree_file = f"data/BH/intermediate_files/timetree_{region}_1000.nwk"
    branch_length_file = f"data/BH/intermediate_files/branch_lengths_{region}_1000.json"

    # BH GTR files
    gtr_file = f"data/BH/mutation_rates/{region}_1000.json"

    # Rates from hamming distance
    rates = {"root": {}, "subtypes": {}}

    reference_sequence = get_reference_sequence(reference_file["root"])
    for subtype in ["B", "C"]:
        rates["root"][subtype] = {}
        years, dist, std, _ = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        for key in ["first", "second", "third", "all"]:
            fit = np.polyfit(years, dist[key], deg=1)
            rates["root"][subtype][key] = fit[0]

    for subtype in ["B", "C"]:
        reference_sequence = get_reference_sequence(reference_file[subtype])
        rates["subtypes"][subtype] = {}
        years, dist, std, _ = get_mean_distance_in_time(alignment_file, reference_sequence, subtype)
        for key in ["first", "second", "third", "all"]:
            fit = np.polyfit(years, dist[key], deg=1)
            rates["subtypes"][subtype][key] = fit[0]

    # Rates from root to tip distance
    dates, lengths = get_root_to_tip_distance(tree_file, branch_length_file)
    fit = np.polyfit(dates, lengths, deg=1)
    rates["rtt"] = fit[0]

    # BH rates from GTR estimates
    with open(gtr_file) as f:
        rates["GTR"] = json.load(f)

    # WH rates
    WH_file = "data/WH/avg_rate_dict.json"
    WH_rate_dict = divergence.load_avg_rate_dict(WH_file)
    rates["WH"] = WH_rate_dict["pol"]

    return rates


def plot_mutation_rates():
    """
    Plot for the mutation rates.
    """
    rates = compute_mutation_rates("pol")
    labels = ["H-root", "H-subtype", "RTT", "GTR", "WH_root", "WH_subtypes", "WH_founder"]

    markersize = 5
    colors = ["k", "C0", "C1", "C2", "C3", "C4", "C5", "C6"]

    plt.figure()
    # BH stuff
    for ii, key in enumerate(["root", "subtypes"]):
        # for jj, key2 in enumerate(["all", "first", "second", "third"]):
        for jj, key2 in enumerate(["all"]):
            if ii == 0 and jj == 0:  # For labelling
                plt.plot(ii, rates[key]["B"][key2], 's',
                         color=colors[jj], markersize=markersize, label="B")
                plt.plot(ii, rates[key]["C"][key2], 'X',
                         color=colors[jj], markersize=markersize, label="C")
            else:
                plt.plot(ii, rates[key]["B"][key2], 's', color=colors[jj])
                plt.plot(ii, rates[key]["C"][key2], 'X', color=colors[jj], markersize=markersize)

    plt.plot(2, rates["rtt"], 'o', color=colors[0], markersize=markersize)

    for ii, key in enumerate(rates["GTR"].keys()):
        plt.plot(3, rates["GTR"][key], "o", color=colors[ii], markersize=markersize, label=key)

    # WH stuff
    for ii, key in enumerate(["all", "first", "second", "third"]):
        plt.plot(4, rates["WH"]["root"]["global"]["all"][key], 'o',
                 color=colors[ii], markersize=markersize)
        plt.plot(5, rates["WH"]["subtypes"]["global"]["all"][key], 'o',
                 color=colors[ii], markersize=markersize)
        plt.plot(6, rates["WH"]["founder"]["global"]["all"][key], 'o',
                 color=colors[ii], markersize=markersize)

    plt.xticks(range(len(labels)), labels, rotation=14)
    plt.ylabel("Mutation rates")
    plt.legend()
    plt.savefig("figures/Rates.pdf")
    plt.show()


if __name__ == '__main__':
    plot_mutation_rates()
