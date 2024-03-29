"""
Main file related to the divergence over time intermediate data.
"""
import numpy as np
import filenames
import tools
import json
from scipy.optimize import curve_fit

from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference


def divergence_matrix(aft, raw_mask, opposite_mask):
    """
    Returns the divergence matrix over time (3D). raw_mask (resp opposite_mask) are 1D boolean vectors where
    True positions contribute their frequency (resp 1-frequency) at each time point.
    """
    div = np.zeros_like(aft)
    for ii in range(aft.shape[0]):
        div[ii, :, :] = raw_mask * aft[ii, :, :] + opposite_mask * (1 - aft[ii, :, :])

    return div


def get_rate_from_divergence(time, div, first_idx=2, last_idx=20):
    """
    Computes the evolution rate from the divergence (slope of the divergence) by 1st order fit between 
    first_idx and last_idx.
    """
    def f(x, a, b): return a * x + b  # linear fit function

    # Fit the divergence
    p, _ = curve_fit(f, time[first_idx:last_idx+1], div[first_idx:last_idx+1], p0=[1e-3, 1e-3])

    # Return the slope only (parameter a)
    return p[0]


def divergence_in_time(patient, region, aft, div_ref):
    """
    Returns the divergence in time matrix (2D) for each genome position at each time point.
    div_ref specifies the reference to which the divergence needs to be computed.
    """
    # founder is founder sequence, any is global consensus
    assert div_ref in ["founder", "any", "B",
                       "C", "root"], f"Reference must be 'founder' 'any' 'B' 'C' or 'root', got {div_ref}"

    initial_idx = patient.get_initial_indices(region)
    if div_ref == "founder":
        aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                          initial_idx, np.arange(aft.shape[-1])]
        aft_initial = np.squeeze(aft_initial)
        raw_mask = np.zeros(aft.shape[-1])
        opposite_mask = np.ones(aft.shape[-1])
        div_3D = divergence_matrix(aft, raw_mask, opposite_mask)
    elif div_ref == "root":
        ref = HIVreference(subtype="any")  # just for the mapping
        root_file = f"data/BH/intermediate_files/{region}_nt_muts.json"
        raw_mask = tools.non_root_mask(patient, region, aft, root_file, ref)
        opposite_mask = tools.root_mask(patient, region, aft, root_file, ref)
        div_3D = divergence_matrix(aft, raw_mask, opposite_mask)
    else:
        ref = HIVreference(subtype=div_ref)
        raw_mask = tools.non_reference_mask(patient, region, aft, ref)
        opposite_mask = tools.reference_mask(patient, region, aft, ref)
        div_3D = divergence_matrix(aft, raw_mask, opposite_mask)

    div = div_3D[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    return np.squeeze(div)


def mean_divergence_in_time(patient, region, aft, div_ref, consensus):
    """
    Returns the average over subset of position of divergence_in_time(patient, region, aft, div_ref).
    Does it for all the sites, for consensus sites and for non_consensus sites. These are defined by the
    consensus used.
    """
    consensus_mask = tools.reference_mask(patient, region, aft, consensus)
    non_consensus_mask = tools.non_reference_mask(patient, region, aft, consensus)
    divergence = divergence_in_time(patient, region, aft, div_ref)
    first_mask = tools.site_mask(aft, 1)
    second_mask = tools.site_mask(aft, 2)
    third_mask = tools.site_mask(aft, 3)
    mask_20 = tools.diversity_mask(patient, region, aft, 0, 20)
    mask_40 = tools.diversity_mask(patient, region, aft, 20, 40)
    mask_60 = tools.diversity_mask(patient, region, aft, 40, 60)
    mask_80 = tools.diversity_mask(patient, region, aft, 60, 80)
    mask_100 = tools.diversity_mask(patient, region, aft, 80, 100)

    div_dict = {}
    div_dict["all"] = {
        "all": np.mean(divergence, axis=1),
        "first": np.mean(divergence[:, first_mask], axis=1),
        "second": np.mean(divergence[:, second_mask], axis=1),
        "third": np.mean(divergence[:, third_mask], axis=1),
        "0-20%": np.mean(divergence[:, mask_20], axis=1),
        "20-40%": np.mean(divergence[:, mask_40], axis=1),
        "40-60%": np.mean(divergence[:, mask_60], axis=1),
        "60-80%": np.mean(divergence[:, mask_80], axis=1),
        "80-100%": np.mean(divergence[:, mask_100], axis=1)
    }
    div_dict["consensus"] = {
        "all": np.mean(divergence[:, consensus_mask], axis=1),
        "first": np.mean(divergence[:, np.logical_and(consensus_mask, first_mask)], axis=1),
        "second": np.mean(divergence[:, np.logical_and(consensus_mask, second_mask)], axis=1),
        "third": np.mean(divergence[:, np.logical_and(consensus_mask, third_mask)], axis=1),
        "0-20%": np.mean(divergence[:, np.logical_and(consensus_mask, mask_20)], axis=1),
        "20-40%": np.mean(divergence[:, np.logical_and(consensus_mask, mask_40)], axis=1),
        "40-60%": np.mean(divergence[:, np.logical_and(consensus_mask, mask_60)], axis=1),
        "60-80%": np.mean(divergence[:, np.logical_and(consensus_mask, mask_80)], axis=1),
        "80-100%": np.mean(divergence[:, np.logical_and(consensus_mask, mask_100)], axis=1)
    }
    div_dict["non_consensus"] = {
        "all": np.mean(divergence[:, non_consensus_mask], axis=1),
        "first": np.mean(divergence[:, np.logical_and(non_consensus_mask, first_mask)], axis=1),
        "second": np.mean(divergence[:, np.logical_and(non_consensus_mask, second_mask)], axis=1),
        "third": np.mean(divergence[:, np.logical_and(non_consensus_mask, third_mask)], axis=1),
        "0-20%": np.mean(divergence[:, np.logical_and(non_consensus_mask, mask_20)], axis=1),
        "20-40%": np.mean(divergence[:, np.logical_and(non_consensus_mask, mask_40)], axis=1),
        "40-60%": np.mean(divergence[:, np.logical_and(non_consensus_mask, mask_60)], axis=1),
        "60-80%": np.mean(divergence[:, np.logical_and(non_consensus_mask, mask_80)], axis=1),
        "80-100%": np.mean(divergence[:, np.logical_and(non_consensus_mask, mask_100)], axis=1)
    }

    return div_dict


def make_intermediate_data(folder_path):
    """
    Creates the bootstrapped divergence in time dictionaries and saves them in the defined folder as
    intermediate data for the Within Host analysis.
    """
    import bootstrap
    import os

    if os.path.exists(folder_path + "bootstrap_div_dict.json"):
        print(folder_path + "bootstrap_div_dict.json already exists, skipping computation.")
        div_dict = load_div_dict(folder_path + "bootstrap_div_dict.json")
    else:
        print("Computing " + folder_path + "boostrap_div_dict.json")
        div_dict = bootstrap.make_bootstrap_div_dict(nb_bootstrap=100)
        div_dict["time"] = (div_dict["time"] / 365).tolist()
        for key in ["env", "pol", "gag"]:  # Region
            for key2 in div_dict[key].keys():  # Reference to which compute the divergence
                for key3 in div_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                    for key4 in div_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                        for key5 in div_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                            # Converting numpy to list for .json compatibility
                            div_dict[key][key2][key3][key4][key5]["mean"] = \
                                div_dict[key][key2][key3][key4][key5]["mean"].tolist()
                            div_dict[key][key2][key3][key4][key5]["std"] = \
                                div_dict[key][key2][key3][key4][key5]["std"].tolist()

        with open(folder_path + "bootstrap_div_dict.json", "w") as f:
            json.dump(div_dict, f, indent=4)

    print("Computing " + folder_path + "rate_dict.json")
    rate_dict = make_rate_dict(div_dict)
    if type(rate_dict["time"]) != list:
        rate_dict["time"] = rate_dict["time"].tolist()

    with open(folder_path + "rate_dict" + ".json", "w") as f:
        json.dump(rate_dict, f, indent=4)


def load_div_dict(filename):
    """Loads the divergence dictionary and returns it.

    Args:
        filename (str): Path to the savefile.

    Returns:
        div_dict(dict): Dictionary containing the divergence in time for the different categories.

    """
    with open(filename, "r") as f:
        div_dict = json.load(f)

    div_dict["time"] = np.array(div_dict["time"])

    for key in ["env", "pol", "gag"]:  # Region
        for key2 in div_dict[key].keys():  # Reference to which compute the divergence
            for key3 in div_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in div_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in div_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        div_dict[key][key2][key3][key4][key5]["mean"] = np.array(
                            div_dict[key][key2][key3][key4][key5]["mean"])
                        div_dict[key][key2][key3][key4][key5]["std"] = np.array(
                            div_dict[key][key2][key3][key4][key5]["std"])
    return div_dict


def make_rate_dict(div_dict):
    """
    Compute the evolution rates and its standard deviation from the bootstrapped divergence values.
    """
    import copy

    fit_dict = copy.deepcopy(div_dict)
    time = div_dict["time"]
    def f(x, a, b): return a * x + b  # function to fit
    for key in ["env", "pol", "gag"]:  # Region
        for key2 in div_dict[key].keys():  # Reference to which compute the divergence
            for key3 in div_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in div_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in div_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        mean = div_dict[key][key2][key3][key4][key5]["mean"]
                        std = np.std(np.array(div_dict[key][key2][key3][key4]
                                              [key5]["rates"])*365)  # originally in days
                        rate = get_rate_from_divergence(time, mean)
                        fit_dict[key][key2][key3][key4][key5] = {}
                        fit_dict[key][key2][key3][key4][key5]["rate"] = rate
                        fit_dict[key][key2][key3][key4][key5]["std"] = std

    return fit_dict


# def load_rate_dict(filename):
#     """
#     Loads the dictionary containing the rates over time (as vectors). Same format as divergence dictionary.
#     """
#     with open(filename, "r") as f:
#         rate_dict = json.load(f)

#     rate_dict["time"] = np.array(rate_dict["time"])

#     for key in ["env", "pol", "gag"]:  # Region
#         for key2 in rate_dict[key].keys():  # Reference to which compute the divergence
#             for key3 in rate_dict[key][key2].keys():  # Reference to define consensus and non-consensus
#                 for key4 in rate_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
#                     for key5 in rate_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
#                         rate_dict[key][key2][key3][key4][key5] = np.array(
#                             rate_dict[key][key2][key3][key4][key5])
#     return rate_dict


def load_avg_rate_dict(filename):
    """
    Loads the dictionary containing the averaged mutation rates between 200 and 2000 days.
    """
    with open(filename, "r") as f:
        avg_rate_dict = json.load(f)
    return avg_rate_dict


if __name__ == '__main__':
    # region = "pol"
    # patient = Patient.load("p1")
    # aft = patient.get_allele_frequency_trajectories(region)
    # div = mean_divergence_in_time(patient, region, aft, "root", HIVreference(subtype="any"))

    make_intermediate_data("data/WH/")
    # div_dict = load_div_dict("data/WH/bootstrap_div_dict.json")
    # rate_dict = average_rate_dict(div_dict)

    # Mutation rate plot
    # div_dict = load_div_dict("data/WH/bootstrap_div_dict.json")
    # rate_dict = load_rate_dict("data/WH/rate_dict.json")
    # avg_rate_dict = load_avg_rate_dict("data/WH/avg_rate_dict.json")

    # average_rate_dict(div_dict)

    # import matplotlib.pyplot as plt
    # plt.style.use("tex")
    # plt.figure()
    # lines = ["-", "--", ":"]
    # colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    # region = "pol"
    # reference = "global"
    # time = div_dict["time"]
    # idxs = time < 5.3  # Time around which some patients stop being followed
    # time = time[idxs]
    # for ii, key in enumerate(["consensus", "non_consensus"]):
    #     for jj, key2 in enumerate(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]):
    #         data = div_dict[region]["founder"][reference][key][key2]["mean"][idxs]
    #         std = div_dict[region]["founder"][reference][key][key2]["std"][idxs]
    #         plt.plot(time, data, lines[ii], color=colors[jj])
    #         plt.fill_between(time, data + std, data - std, color=colors[jj], alpha=0.15)

    # for ii, label in enumerate(["consensus", "non-consensus"]):
    #     plt.plot([0], [0], lines[ii], color="k", label=label)
    # for jj, label in enumerate(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]):
    #     plt.plot([0], [0], lines[0], color=colors[jj], label=label)
    # plt.legend()
    # plt.show()

    # import matplotlib.pyplot as plt
    # patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    # region = "pol"
    # div_ref = "founder"
    # plt.figure()
    # for patient_name in patient_names:
    #     patient = Patient.load(patient_name)
    #     aft = patient.get_allele_frequency_trajectories(region)
    #     div_dict = mean_divergence_in_time(patient, region, aft, div_ref, HIVreference(subtype="any"))
    #     plt.plot(patient.ysi ,div_dict["all"]["all"])
    # plt.ylim([0,0.14])
    # plt.xlim([0,5.5])
    # plt.show()
