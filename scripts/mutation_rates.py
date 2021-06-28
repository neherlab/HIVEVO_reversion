import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO, Phylo
from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance
import divergence


def compute_mutation_rates(nb_sequences=[1000, 500, 250, 125]):
    """
    Returns the mutation rates from the distance to root, distance to subtype consensus and root to tip
    distance.
    """
    # BH rate files
    reference_file = {}
    reference_file["root"] = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    reference_file["B"] = "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta"
    reference_file["C"] = "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta"
    alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    tree_file = {}
    branch_length_file = {}
    for nb in nb_sequences:
        tree_file[str(nb)] = f"data/BH/intermediate_files/timetree_pol_{nb}.nwk"
        branch_length_file[str(nb)] = f"data/BH/intermediate_files/branch_lengths_pol_{nb}.json"

    # BH GTR files
    gtr_file = "data/BH/mutation_rates/pol_1000.json"

    # WH rates
    WH_file = "data/WH/avg_rate_dict.json"
    WH_rate_dict = divergence.load_avg_rate_dict(WH_file)

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

    # BH rates from GTR estimates
    with open(gtr_file) as f:
        rates["GTR"] = json.load(f)

    # WH rates from divergence
    rates["WH"] = WH_rate_dict["pol"]

    return rates


def plot_mutation_rates():
    """
    Plot for the mutation rates.
    """
    rates = compute_mutation_rates()
    labels = ["H-root", "H-subtype", "RTT", "GTR", "WH_global", "WH_subtypes", "WH_founder"]

    fontsize = 16
    markersize = 16
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap_colors = [cmap(x) for x in np.linspace(0, 1, len(rates["rtt"].keys()))]

    plt.figure(figsize=(10, 7))
    # BH stuff
    for ii, key in enumerate(["root", "subtypes"]):
        if ii:
            plt.plot(ii, rates[key]["B"] * 1e4, '.', color="C0", markersize=markersize, label="B")
            plt.plot(ii, rates[key]["C"] * 1e4, '.', color="C1", markersize=markersize, label="C")
        else:
            plt.plot(ii, rates[key]["B"] * 1e4, '.', color="C0", markersize=markersize)
            plt.plot(ii, rates[key]["C"] * 1e4, '.', color="C1", markersize=markersize)

    for ii, key in enumerate(rates["rtt"].keys()):
        plt.plot(2, rates["rtt"][key] * 1e4, '.', color=cmap_colors[ii], markersize=markersize, label=key)

    for ii, key in enumerate(rates["GTR"].keys()):
        plt.plot(3, rates["GTR"][key] * 1e4, ".", color=colors[ii + 2], markersize=markersize, label=key)

    # WH stuff
    for ii, key in enumerate(["all", "first", "second", "third"]):
        plt.plot(4, rates["WH"]["any"]["global"]["all"][key] * 1e4, ".",
                 color=colors[ii + 2], markersize=markersize)
        plt.plot(5, rates["WH"]["subtypes"]["global"]["all"][key] * 1e4, ".",
                 color=colors[ii + 2], markersize=markersize)
        plt.plot(6, rates["WH"]["founder"]["global"]["all"][key] * 1e4, ".",
                 color=colors[ii + 2], markersize=markersize)

    plt.xticks(range(len(labels)), labels, fontsize=fontsize, rotation=14)
    plt.ylabel("Mutation rates (per year) * e-4", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.savefig("figures/mutation_rates.png", format="png")
    plt.show()


if __name__ == '__main__':
    plot_mutation_rates()
