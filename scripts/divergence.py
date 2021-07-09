# import filenames
import tools
import json
import numpy as np

from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient


def divergence_matrix(aft, raw_mask, opposite_mask):
    """
    Returns the divergence matrix over time (3D). raw_mask (resp opposite_mask) are 1D boolean vectors where
    True positions contribute their frequency (resp 1-frequency) at each time point.
    """
    div = np.zeros_like(aft)
    for ii in range(aft.shape[0]):
        div[ii, :, :] = raw_mask * aft[ii, :, :] + opposite_mask * (1 - aft[ii, :, :])

    return div


def divergence_in_time(patient, region, aft, div_ref):
    """
    Returns the divergence in time matrix (2D) for each genome position at each time point.
    div_ref specifies the reference to which the divergence needs to be computed.
    """
    # founder is founder sequence, any is global consensus
    assert div_ref in ["founder", "any", "B",
                       "C"], f"Reference must be 'founder' 'any' 'B' or 'C', got {div_ref}"

    initial_idx = patient.get_initial_indices(region)
    if div_ref == "founder":
        aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                          initial_idx, np.arange(aft.shape[-1])]
        aft_initial = np.squeeze(aft_initial)
        raw_mask = np.zeros(aft.shape[-1])
        opposite_mask = aft_initial[0]
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

    div_dict = {}
    div_dict["all"] = {
        "all": np.mean(divergence, axis=1),
        "first": np.mean(divergence[:, first_mask], axis=1),
        "second": np.mean(divergence[:, second_mask], axis=1),
        "third": np.mean(divergence[:, third_mask], axis=1)
    }
    div_dict["consensus"] = {
        "all": np.mean(divergence[:, consensus_mask], axis=1),
        "first": np.mean(divergence[:, np.logical_and(consensus_mask, first_mask)], axis=1),
        "second": np.mean(divergence[:, np.logical_and(consensus_mask, second_mask)], axis=1),
        "third": np.mean(divergence[:, np.logical_and(consensus_mask, third_mask)], axis=1)
    }
    div_dict["non_consensus"] = {
        "all": np.mean(divergence[:, non_consensus_mask], axis=1),
        "first": np.mean(divergence[:, np.logical_and(non_consensus_mask, first_mask)], axis=1),
        "second": np.mean(divergence[:, np.logical_and(non_consensus_mask, second_mask)], axis=1),
        "third": np.mean(divergence[:, np.logical_and(non_consensus_mask, third_mask)], axis=1)
    }

    return div_dict


def make_intermediate_data(folder_path):
    """
    Creates the bootstrapped divergence in time dictionaries and saves them in the defined folder.
    """
    import bootstrap

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

    with open(folder_path + "bootstrap_div_dict" + ".json", "w") as f:
        json.dump(div_dict, f, indent=4)

    rate_dict = make_rate_dict(div_dict)
    avg_rate_dict = average_rate_dict(rate_dict)
    rate_dict["time"] = rate_dict["time"].tolist()
    for key in ["env", "pol", "gag"]:  # Region
        for key2 in rate_dict[key].keys():  # Reference to which compute the divergence
            for key3 in rate_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in rate_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in rate_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        # Converting numpy to list for .json compatibility
                        rate_dict[key][key2][key3][key4][key5] = \
                            rate_dict[key][key2][key3][key4][key5].tolist()

    with open(folder_path + "rate_dict" + ".json", "w") as f:
        json.dump(rate_dict, f, indent=4)

    with open(folder_path + "avg_rate_dict" + ".json", "w") as f:
        json.dump(avg_rate_dict, f, indent=4)


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
    Creates a dictionary with the rates computed from the derivative of the divergence dictionary.
    """
    import copy

    rate_dict = copy.deepcopy(div_dict)
    for key in ["env", "pol", "gag"]:  # Region
        for key2 in div_dict[key].keys():  # Reference to which compute the divergence
            for key3 in div_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in div_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in div_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        rate_dict[key][key2][key3][key4][key5] = np.gradient(
                            rate_dict[key][key2][key3][key4][key5]["mean"],
                            rate_dict["time"][1] - rate_dict["time"][0])
    return rate_dict


def average_rate_dict(rate_dict, first_idx=2, last_idx=20):
    """
    Average the rates for the rate dictionary between 200 and 2000 days (by default). Returns a dictionary
    with the same structure but with scalars instead of vectors at the leafs.
    """
    import copy

    avg_dict = copy.deepcopy(rate_dict)
    del avg_dict["time"]
    for key in ["env", "pol", "gag"]:  # Region
        for key2 in avg_dict[key].keys():  # Reference to which compute the divergence
            for key3 in avg_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in avg_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in avg_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        tmp = avg_dict[key][key2][key3][key4][key5]
                        rate = np.mean(tmp[first_idx:last_idx])  # between 200 and 2000 days by default
                        avg_dict[key][key2][key3][key4][key5] = rate
    return avg_dict


def load_rate_dict(filename):
    """
    Loads the dictionary containing the rates over time (as vectors). Same format as divergence dictionary.
    """
    with open(filename, "r") as f:
        rate_dict = json.load(f)

    rate_dict["time"] = np.array(rate_dict["time"])

    for key in ["env", "pol", "gag"]:  # Region
        for key2 in rate_dict[key].keys():  # Reference to which compute the divergence
            for key3 in rate_dict[key][key2].keys():  # Reference to define consensus and non-consensus
                for key4 in rate_dict[key][key2][key3].keys():  # all, consensus or non_consensus sites
                    for key5 in rate_dict[key][key2][key3][key4].keys():  # all, first, second, third sites
                        rate_dict[key][key2][key3][key4][key5] = np.array(
                            rate_dict[key][key2][key3][key4][key5])
    return rate_dict


def load_avg_rate_dict(filename):
    """
    Loads the dictionary containing the averaged mutation rates between 200 and 2000 days.
    """
    with open(filename, "r") as f:
        avg_rate_dict = json.load(f)
    return avg_rate_dict


if __name__ == '__main__':
    # region = "env"
    # patient = Patient.load("p2")
    # aft = patient.get_allele_frequency_trajectories(region)
    # div = mean_divergence_in_time(patient, region, aft, "founder", HIVreference(subtype="any"))

    # make_intermediate_data("data/WH/")

    # Mutation rate plot
    div_dict = load_div_dict("data/WH/bootstrap_div_dict.json")
    rate_dict = load_rate_dict("data/WH/rate_dict.json")
    avg_rate_dict = load_avg_rate_dict("data/WH/avg_rate_dict.json")

    lines = ["-", "--", ":"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    region = "pol"
    reference = "founder"

    import matplotlib.pyplot as plt
    plt.style.use("tex")

    plt.figure(figsize=(6.5315, 4.0367))
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        for jj, key2 in enumerate(["all", "first", "second", "third"]):
            tmp = div_dict[region][reference]["global"][key][key2]["mean"]
            estimate = (tmp[20] - tmp[0]) / (2000 / 365)
            estimate = round(estimate, 4)
            estimate2 = round(avg_rate_dict[region][reference]["global"][key][key2], 4)
            plt.plot(rate_dict["time"], rate_dict[region][reference]["global"][key][key2], lines[ii],
                     color=colors[jj], label=f"{key} {key2} {estimate} {estimate2}")
    plt.legend()
    plt.xlabel("test")
    plt.grid()
    plt.savefig("figures/mutation_rate_divergence.pdf", format='pdf')
    plt.show()
