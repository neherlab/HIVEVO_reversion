import copy
import numpy as np

import trajectory
import divergence
from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient


def bootstrap_patient_names(patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    "Returns a list of patient name bootstrap with replacement (patient can appear more than once)."
    choice = np.random.choice(range(len(patient_names)), len(patient_names))
    names = []
    for ii in choice:
        names += [patient_names[ii]]
    return names


def bootstrap_mean_in_time(trajectories, freq_range, nb_bootstrap=10):
    """
    Computes the mean_in_time for the given list of trajectories nb_boostrap time by bootstrapping patients
    and returns the average and standard deviation vectors.
    """

    means = []
    for ii in range(nb_bootstrap):
        # Bootstrapping trajectories
        bootstrap_names = bootstrap_patient_names()
        bootstrap_trajectories = []
        for name in bootstrap_names:
            bootstrap_trajectories += [traj for traj in trajectories if traj.patient == name]

        # Computing the mean in time for each boostrap
        time, mean, _, _ = trajectory.get_mean_in_time(bootstrap_trajectories, freq_range=freq_range)
        means += [[mean]]

    means = np.array(means)
    average = np.nanmean(means, axis=0)[0, :]
    std = np.nanstd(means, axis=0)[0, :]

    return time, average, std


def make_bootstrap_mean_dict(trajectory_list, nb_bootstrap=10):
    """
    Generates the dictionary for bootstrapped mean frequency in time. Does it for all regions, the 3 frequency
    windows and rev / non_rev mutations. The trajectory list should contain the trajectories from all patients
    Keys are the following : dict["rev"/"non_rev"]["[0.2,0.4]","[0.4,0.6]","[0.6,0.8]"]
    """
    bootstrap_dict = {"rev": {}, "non_rev": {}}
    for mut_type in ["rev", "non_rev"]:
        for freq_range in [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]:
            if mut_type == "rev":
                trajectories = [traj for traj in trajectory_list if traj.reversion]
            else:
                trajectories = [traj for traj in trajectory_list if ~traj.reversion]

            times, mean, std = bootstrap_mean_in_time(trajectories, freq_range, nb_bootstrap)
            bootstrap_dict[mut_type][str(freq_range)] = {"mean": mean, "std": std}

    return bootstrap_dict, times


def bootstrap_divergence_in_time(region, reference, consensus, nb_bootstrap=10, time=np.arange(0, 3100, 100),
                                 patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    """
    Computes the mean divergence in time for all patient, and then computes a bootstrapped value and std from
    that. Returns the times, average and std.
    Reference designate the sequence to which compute the divergence (founder, the global consensus or subtype
    consensus).
    Consensus designate the sequence from which we categorize the sites into 'consensus' and 'non_consensus'.
    """
    # founder is founder sequence, any is global consensus
    assert reference in ["founder", "any", "subtypes"], "Reference must be 'founder' 'any' 'subtypes'"
    assert consensus in ["global", "subtype"], "Consensus must be 'global' or 'subtype'"
    patient_names = copy.copy(patient_names)

    if reference == "subtypes" or consensus == "subtype":
        patient_names.remove("p1")  # p1 is subtype AE
        if patient_names == ["p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]:
            subtypes = ["B", "B", "B", "B", "C", "B", "B", "B"]
        else:
            raise ValueError("Must be the regular patients to compute divergence to subtype")

    # Computes divergence for each patient
    patient_div_dict = {k: {} for k in patient_names}
    for ii, patient_name in enumerate(patient_names):
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)

        # Loading reference for categorisation into consensus and non consensus sites
        if consensus == "global":
            hivreference = HIVreference(subtype="any")
        else:
            hivreference = HIVreference(subtype=subtypes[ii])

        if reference == "subtypes":
            tmp_div_dict = divergence.mean_divergence_in_time(
                patient, region, aft, subtypes[ii], hivreference)
        else:
            tmp_div_dict = divergence.mean_divergence_in_time(
                patient, region, aft, reference, hivreference)

        # Interpolation of divergence value as samples are not homogeneous in time. Fine because monotonic
        for key in ["all", "consensus", "non_consensus"]:
            patient_div_dict[patient_name][key] = np.interp(time, patient.dsi, tmp_div_dict[key])

    means = {"all": [], "consensus": [], "non_consensus": []}
    for ii in range(nb_bootstrap):
        bootstrap_names = bootstrap_patient_names(patient_names)

        for key in ["all", "consensus", "non_consensus"]:
            divergences = np.array([patient_div_dict[name][key] for name in bootstrap_names])
            means[key] += [np.mean(divergences, axis=0)]

    bootstrapped_dict = {"all": {}, "consensus": {}, "non_consensus": {}}
    for key in ["all", "consensus", "non_consensus"]:
        bootstrapped_dict[key]["mean"] = np.mean(means[key], axis=0)
        bootstrapped_dict[key]["std"] = np.std(means[key], axis=0)

    return time, bootstrapped_dict


def make_bootstrap_div_dict(nb_bootstrap=100):
    """
    Computes the average divergence in time over patients.
    Returns a dictionary of the format:
        divergence[env/pol/gag][founder/any/subtypes][global/subtype][all/consensus/non_consensus][mean/std]
    There is also the entry for the time vector : divergence[time]
    """
    div_dict = {}

    for region in ["env", "pol", "gag"]:
        div_dict[region] = {}
        for reference in ["founder", "any", "subtypes"]:
            div_dict[region][reference] = {}
            for consensus in ["global", "subtype"]:
                print(f"Computing bootstrapped divergence for {region} {reference} {consensus}")
                time, bootstrap_dict = bootstrap_divergence_in_time(
                    region, reference, consensus, nb_bootstrap)
                div_dict[region][reference][consensus] = bootstrap_dict
                div_dict["time"] = time

    return div_dict


if __name__ == '__main__':
    # trajectories = trajectory.load_trajectory_list("data/Trajectory_list_any.json")
    # bootstrap_dict, times = make_bootstrap_mean_dict(trajectories)

    div_dict = make_bootstrap_div_dict(5)
