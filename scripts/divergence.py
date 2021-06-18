import filenames
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


def mean_divergence_in_time(patient, region, aft, div_ref):
    """
    Returns the average over all positions of divergence_in_time(patient, region, aft, div_ref)
    """
    # TODO
    reference_mask = tools.reference_mask(patient, region, aft, ref)
    non_reference_mask = tools.non_reference_mask(patient, region, aft, ref)
    return np.mean(divergence_in_time(patient, region, aft, div_ref), axis=1)


def make_intermediate_data(folder_path):
    """
    Creates the bootstrapped divergence in time dictionaries and saves them in the defined folder.
    """
    import bootstrap

    div_dict = bootstrap.make_bootstrap_div_dict(nb_bootstrap=5)
    for key in div_dict.keys():
        for key2 in div_dict[key].keys():
            # Converting numpy to list for .json compatibility
            div_dict[key][key2]["mean"] = div_dict[key][key2]["mean"].tolist()
            div_dict[key][key2]["std"] = div_dict[key][key2]["std"].tolist()
            div_dict[key][key2]["time"] = div_dict[key][key2]["time"].tolist()

    with open(folder_path + "bootstrap_div_dict" + ".json", "w") as f:
        json.dump(div_dict, f, indent=4)


def load_div_dict(filename):
    """Loads the divergence dictionary and returns it.

    Args:
        filename (str): Path to the savefile.

    Returns:
        div_dict(dict): Dictionary containing the divergence in time for the different categories.

    """
    with open(filename, "r") as f:
        div_dict = json.load(f)

    for key1 in div_dict.keys():
        for key2 in div_dict[key1].keys():
            div_dict[key1][key2]["mean"] = np.array(div_dict[key1][key2]["mean"])
            div_dict[key1][key2]["std"] = np.array(div_dict[key1][key2]["std"])
            div_dict[key1][key2]["time"] = np.array(div_dict[key1][key2]["time"])
    return div_dict


if __name__ == '__main__':
    # region = "env"
    # patient = Patient.load("p2")
    # aft = patient.get_allele_frequency_trajectories(region)
    # div = mean_divergence_in_time(patient, region, aft, "founder")
    # div = mean_divergence_in_time(patient, region, aft, "any")
    # div = mean_divergence_in_time(patient, region, aft, "B")

    # div_dict = make_intermediate_data("data/WH/")
    div_dict = load_div_dict("data/WH/bootstrap_div_dict.json")
