"""
Script used to test the bootstrapping estimates for the rates of fig 1 BH. It is used to estimate the errors
on the rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import filenames
import os
from Bio import AlignIO, Phylo
import tools
from distance_in_time import get_reference_sequence


def iter_tree(tree_json, node_names, cutoff_date=1980):
    """
    Returns the name of the first clades after the cutoff_date. It does it by iteratively searching the
    the children of the given tree_json, and stopping as soon as one children is after the cutoff_date.
    One has to pass an empty list as node_names otherwise it stays and memory and things are appended to it
    if several calls of iter_tree are done (not sure why).
    """
    if tree_json["node_attrs"]["num_date"]["value"] > cutoff_date:
        node_names += [get_tips(tree_json)]
    else:
        if "children" in tree_json.keys():
            for child in tree_json["children"]:
                iter_tree(child, node_names, cutoff_date)

    return node_names


def get_tips(tree):
    """
    Returns a list of names of the tips of the tree. Does this recursively.
    """
    tips = []
    if "children" in tree.keys():  # If this node has children
        for child in tree["children"]:
            tips += get_tips(child)
    else:  # If it does not have children
        tips += [tree["name"]]

    return tips


def bootstrap_clades(subtree_list):
    """
    Returns a list of bootstrapped subtrees from the orginal list of subtrees given by iter_tree.
    """
    choice = np.random.choice(range(len(subtree_list)), len(subtree_list))
    sequence_names = []
    for ii in choice:
        sequence_names += subtree_list[ii]
    return sequence_names


def subtree_length_statistics(tree_json_path, cutoff_dates=[1975, 1980, 1985]):
    "Plot a histogram of the subtree length for the chosen cutoff dates."
    with open(tree_json_path, "r") as f:
        tree_json = json.load(f)
    tree_json = tree_json["tree"]

    list_lengths = []
    for cutoff_date in cutoff_dates:
        tmp = iter_tree(tree_json, [], cutoff_date)
        lengths = [len(l) for l in tmp]
        list_lengths += [lengths]

    plt.figure()
    for ii in range(3):
        values, bins = np.histogram(list_lengths[ii], bins=np.linspace(1, 100, 100))
        bins = 0.5*(bins[:-1] + bins[1:])
        plt.plot(bins, values, '.-', label=cutoff_dates[ii])
    plt.yscale("log")
    plt.legend()
    plt.ylabel("# occurence")
    plt.xlabel("nb tips in subtree")
    # plt.ylim([-1, 200])
    plt.show()


def get_gap_mask(alignment_array, threshold=0.1):
    """
    Return a vector were true are the sites seen with less than threshold fraction of N.
    """
    gaps = alignment_array == "N"
    gap_proportion = np.sum(gaps, axis=0, dtype=int) / gaps.shape[0]
    return gap_proportion < threshold


def bootstrap_mean_distance_in_time(sequences_names, alignment_file, reference_sequence, subtype=""):
    """
    Computes the hamming distance of all sequences in sequences_names to the reference_sequence. 
    Does this for all sites together.
    """
    from scipy.stats import scoreatpercentile
    # Checks
    assert os.path.exists(alignment_file), f"{alignment_file} doesn't exist."
    assert type(reference_sequence) == np.ndarray, f"reference sequence must be a numpy array."

    # Data loading
    alignment = AlignIO.read(alignment_file, "fasta")
    alignment_array = np.array(alignment)
    names = np.array([seq.id for seq in alignment])
    dates = np.array([int(name.split(".")[2]) for name in names])
    subtypes = np.array([name.split(".")[0] for name in names])

    # Selecting subtype if needed
    if subtype != "":
        alignment_array = alignment_array[subtypes == subtype]
        dates = dates[subtypes == subtype]
        names = names[subtypes == subtype]

    gap_mask = get_gap_mask(alignment_array)
    distance_matrix = (alignment_array != reference_sequence)[:, gap_mask]
    distance_vector = np.sum(distance_matrix, axis=1, dtype=int) / distance_matrix.shape[-1]

    years = []
    distances = []
    for name in sequences_names:
        tmp = np.where(names == name)[0]
        if tmp.shape != (0,):
            idx = np.where(names == name)[0][0]
            years += [dates[idx]]
            distances += [distance_vector[idx]]
    years = np.array(years)
    distances = np.array(distances)

    time = np.unique(dates)
    average_distance = []
    nb_seq = []
    for date in time:
        average_distance += [np.mean(distances[years == date])]
        nb_seq += [distances[years == date].shape[0]]
    average_distance = np.array(average_distance)

    return time, average_distance, nb_seq


def make_root_bootstrap(region, output_folder, nb_bootstrap=100, cutoff_date=1980):
    """
    Creates the file for the root bootstrap.
    """
    file_path = f"visualisation/{region}.json"  # Uses the output of augur to relate clades
    root_path = f"data/BH/intermediate_files/{region}_nt_muts.json"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}.fasta"

    # Load the tree and ref sequence
    with open(file_path, "r") as f:
        tree_json = json.load(f)
    tree_json = tree_json["tree"]
    reference_sequence = get_reference_sequence(root_path)

    # Make a list of subtree from the cut at cutoff_date
    subtree_list = iter_tree(tree_json, [], cutoff_date)

    # Compute the distances to root from bootstrapping with replacement
    distances = []
    number_sequences = []
    for ii in range(nb_bootstrap):
        sequences_names = bootstrap_clades(subtree_list)
        years, dist, nb_seq = bootstrap_mean_distance_in_time(
            sequences_names, alignment_file, reference_sequence)
        distances += [dist]
        number_sequences += [nb_seq]
    distances = np.array(distances)

    # Compute the evolution rate of these bootstrap
    rates = []
    for distance in distances:
        fit = np.polyfit(years[~np.isnan(distance)], distance[~np.isnan(distance)], deg=1)
        rates += [fit[0]]

    # Saving all the files
    save_dict = {"distances": distances.tolist(), "rates": rates, "years": years.tolist()}
    with open(output_folder + region + "_root_bootstrap.json", "w") as f:
        json.dump(save_dict, f, indent=4)


def make_subtypes_bootstrap(region, output_folder, nb_bootstrap=100, cutoff_date=1980):
    """
    Creates the file for the subtype bootstrap.
    """
    file_path = f"visualisation/{region}.json"  # Uses the output of augur to relate clades
    subtype_B_path = f"data/BH/alignments/to_HXB2/{region}_B_consensus.fasta"
    subtype_C_path = f"data/BH/alignments/to_HXB2/{region}_C_consensus.fasta"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}.fasta"

    # Load the tree and ref sequence
    with open(file_path, "r") as f:
        tree_json = json.load(f)
    tree_json = tree_json["tree"]

    # Make a list of subtree from the cut at cutoff_date
    subtree_list = iter_tree(tree_json, [], cutoff_date)

    # Compute distances to subtype B
    reference_sequence = get_reference_sequence(subtype_B_path)
    distances_B = []
    nb_seq_B = []
    for ii in range(nb_bootstrap):
        sequences_names = bootstrap_clades(subtree_list)
        years_B, dist, nb_seq = bootstrap_mean_distance_in_time(
            sequences_names, alignment_file, reference_sequence, subtype="B")
        distances_B += [dist]
        nb_seq_B += [nb_seq]
    distances_B = np.array(distances_B)
    nb_seq_B = np.array(nb_seq_B)

    # Compute distances to subtype C
    reference_sequence = get_reference_sequence(subtype_C_path)
    distances_C = []
    nb_seq_C = []
    for ii in range(nb_bootstrap):
        sequences_names = bootstrap_clades(subtree_list)
        years_C, dist, nb_seq = bootstrap_mean_distance_in_time(
            sequences_names, alignment_file, reference_sequence, subtype="C")
        distances_C += [dist]
        nb_seq_C += [nb_seq]
    distances_C = np.array(distances_C)
    nb_seq_C = np.array(nb_seq_C)

    # Mixing bootstraps from B and C, starting from B and adding contribution of C
    distances = distances_B
    years = years_B
    for ii in range(nb_bootstrap):
        idxs = np.isin(years_B, years_C)  # This is because subtype B is seen in more years than subtype C
        distances[ii][idxs] = (nb_seq_B[ii][idxs] * distances_B[ii][idxs] + nb_seq_C[ii] *
                               distances_C[ii]) / (nb_seq_B[ii][idxs] + nb_seq_C[ii])

    # Compute the evolution rate of these bootstrap
    rates = []
    for distance in distances:
        fit = np.polyfit(years[~np.isnan(distance)], distance[~np.isnan(distance)], deg=1)
        rates += [fit[0]]

    # Saving all the files
    save_dict = {"distances": distances.tolist(), "rates": rates, "years": years.tolist()}
    with open(output_folder + region + "_subtypes_bootstrap.json", "w") as f:
        json.dump(save_dict, f, indent=4)


def make_RTT_bootstrap(region, output_folder, nb_bootstrap=100, cutoff_date=1980):
    """
    Creates the file for the RTT bootstrap.
    """
    file_path = f"visualisation/{region}.json"  # Uses the output of augur to relate clades
    tree_path = f"data/BH/intermediate_files/tree_{region}.nwk"

    # Load the tree.json
    with open(file_path, "r") as f:
        tree_json = json.load(f)
    tree_json = tree_json["tree"]

    # Make a list of subtree from the cut at cutoff_date
    subtree_list = iter_tree(tree_json, [], cutoff_date)

    # Load the tree.nwk
    tree = Phylo.read(tree_path, "newick")

    # Computes the RTT from bootstrapping with replacement and the associated rate
    rates = []
    distances = []
    for ii in range(nb_bootstrap):
        sequences_names = bootstrap_clades(subtree_list)
        years, distance = bootstrap_RTT(sequences_names, tree)
        distances += [distance.tolist()]

        fit = np.polyfit(years[~np.isnan(distance)], distance[~np.isnan(distance)], deg=1)
        rates += [fit[0]]

    # Saving all the files
    save_dict = {"distances": distances, "rates": rates, "years": years.tolist()}
    with open(output_folder + region + "_RTT_bootstrap.json", "w") as f:
        json.dump(save_dict, f, indent=4)


def bootstrap_RTT(sequences_names, tree):
    """
    Computes the RTT average of all sequences in sequences_names using the given tree.
    """
    dates = []
    for name in sequences_names:
        date = name.split(".")[2]
        dates += [int(date)]
    dates = np.array(dates)

    rtt_all = np.array([tree.distance(next(tree.find_clades(name))) for name in sequences_names])
    rtt = []

    for date in np.arange(1982, 2020):
        rtt += [np.mean(rtt_all[dates == date])]

    return np.arange(1982, 2020), np.array(rtt)


def load_bootstrap(file_path):
    """
    Loads the bootstrap.json file and returns it. These bootstraps contains lots of NANs in the years where
    no sequences was present during the bootstraps.
    """
    with open(file_path, "r") as f:
        bootstrap_dict = json.load(f)

    for key in bootstrap_dict.keys():
        bootstrap_dict[key] = np.array(bootstrap_dict[key])

    return bootstrap_dict


if __name__ == "__main__":
    from scipy.stats import scoreatpercentile
    region = "gag"
    make_root_bootstrap(region, "data/BH/bootstraps/", nb_bootstrap=100)
    make_subtypes_bootstrap(region, "data/BH/bootstraps/", nb_bootstrap=100)
    make_RTT_bootstrap(region, "data/BH/bootstraps/", nb_bootstrap=100)

    # root_bootstrap = load_bootstrap("data/BH/bootstraps/pol_root_bootstrap.json")
    # subtypes_bootstrap = load_bootstrap("data/BH/bootstraps/pol_subtypes_bootstrap.json")
    # RTT_bootstrap = load_bootstrap("data/BH/bootstraps/pol_RTT_bootstrap.json")

    # plt.figure()
    # for bdict, name in zip([RTT_bootstrap, root_bootstrap, subtypes_bootstrap], ["RTT", "root", "subtypes"]):
    #     years = bdict["years"]
    #     distances = bdict["distances"]
    #     rate = bdict["rates"]
    #     mean = np.nanmean(distances, axis=0)
    #     std = np.nanstd(distances, axis=0)
    #     lower, higher = [], []
    #     for ii in range(len(years)):
    #         lower += [scoreatpercentile(distances[:, ii][~np.isnan(distances[:, ii])], 10)]
    #         higher += [scoreatpercentile(distances[:, ii][~np.isnan(distances[:, ii])], 90)]
    #     plt.plot(years, mean,
    #         label=f"{name} ({round(np.mean(rate)*10**4,1)} +- {round(np.std(rate)*10**4,1)})*10^-4")
    #     plt.fill_between(years, lower, higher, alpha=0.4)
    #     # plt.fill_between(years, mean-std, mean+std, alpha = 0.4)
    # plt.legend()
    # plt.xlabel("Years")
    # plt.ylabel("Distance")
    # plt.show()

    # file_path = f"visualisation/pol.json"
    # subtree_length_statistics(file_path)
