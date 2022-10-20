"""
Script to handle the bootstrapping for WH data generation.
"""

import copy
import numpy as np
import trajectory
import divergence
import distance_in_time
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
    # print(average, std)

    return time, average, std


def make_bootstrap_mean_dict(trajectory_list, nb_bootstrap=10):
    """
    Generates the dictionary for bootstrapped mean frequency in time. Does it for all regions, the 3 frequency
    windows and rev / non_rev mutations. The trajectory list should contain the trajectories from all patients
    Keys are the following : dict["rev"/"non_rev/all"]["syn"/"non_syn"/"all"]["[0.2,0.4]","[0.4,0.6]","[0.6,0.8]"]
    """
    bootstrap_dict = {"rev": {"syn": {}, "non_syn": {}, "all": {}},
                      "non_rev": {"syn": {}, "non_syn": {}, "all": {}},
                      "all": {"syn": {}, "non_syn": {}, "all": {}}}

    for key in bootstrap_dict.keys():
        if key == "rev":
            trajectories = [traj for traj in trajectory_list if traj.reversion]
        elif key == "non_rev":
            trajectories = [traj for traj in trajectory_list if  not traj.reversion]
        elif key == "all":
            trajectories = trajectory_list

        for key2 in bootstrap_dict[key].keys():
            if key2 == "syn":
                tmp = [traj for traj in trajectories if traj.synonymous]
            elif key2 == "non_syn":
                tmp = [traj for traj in trajectories if not traj.synonymous]
            elif key2 == "all":
                tmp = trajectories

            for freq_range in [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]:
                times, mean, std = bootstrap_mean_in_time(tmp, freq_range, nb_bootstrap)
                bootstrap_dict[key][key2][str(freq_range)] = {"mean": mean, "std": std}

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
    assert reference in ["founder", "any", "subtypes",
                         "root"], "Reference must be 'founder' 'any' 'subtypes' 'root'"
    assert consensus in ["global", "subtype", "root"], "Consensus must be 'global' 'root' or 'subtype'"
    patient_names = copy.copy(patient_names)

    if reference == "subtypes" or consensus == "subtype":
        patient_names.remove("p1")  # p1 is subtype AE
        if patient_names == ["p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]:
            subtypes = ["B", "B",   "B", "B",  "C", "B", "B", "B"]
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
        elif consensus == "root":
            hivreference = HIVreference(subtype="any")
            # Small hacks to load the root sequence in this object
            map = patient.map_to_external_reference(region)[:, 0]
            root_sequence = distance_in_time.get_reference_sequence(
                f"data/BH/intermediate_files/{region}_nt_muts.json")
            hivreference.consensus[map] = root_sequence.astype("S1")[map - map[0]]
            tmp = np.zeros_like(root_sequence)
            tmp[root_sequence == "A"] = 0
            tmp[root_sequence == "C"] = 1
            tmp[root_sequence == "G"] = 2
            tmp[root_sequence == "T"] = 3
            tmp[root_sequence == "-"] = 4
            tmp[root_sequence == "N"] = 5
            hivreference.consensus_indices[map] = tmp[map - map[0]]
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
            patient_div_dict[patient_name][key] = {}
            for key2 in tmp_div_dict[key].keys():
                patient_div_dict[patient_name][key][key2] = np.interp(
                    time, patient.dsi, tmp_div_dict[key][key2])

    # Initializing dictionary to store the different bootstrap values
    means = {"all": {}, "consensus": {}, "non_consensus": {}}
    for key in means.keys():
        for key2 in tmp_div_dict[key].keys():
            means[key][key2] = []

    for ii in range(nb_bootstrap):
        bootstrap_names = bootstrap_patient_names(patient_names)
        for key in ["all", "consensus", "non_consensus"]:
            for key2 in means[key].keys():
                divergences = np.array([patient_div_dict[name][key][key2] for name in bootstrap_names])
                means[key][key2] += [np.mean(divergences, axis=0)]

    bootstrapped_dict = {"all": {}, "consensus": {}, "non_consensus": {}}
    for key in ["all", "consensus", "non_consensus"]:
        for key2 in means[key].keys():
            bootstrapped_dict[key][key2] = {}
            bootstrapped_dict[key][key2]["mean"] = np.mean(means[key][key2], axis=0)
            bootstrapped_dict[key][key2]["std"] = np.std(means[key][key2], axis=0)
            bootstrapped_dict[key][key2]["rates"] = [
                divergence.get_rate_from_divergence(time, mean) for mean in means[key][key2]]

    return time, bootstrapped_dict


def make_bootstrap_div_dict(nb_bootstrap=100):
    """
    Computes the average divergence in time over patients.
    Returns a dictionary of the format:
        divergence[env/pol/gag][founder/any/subtypes/root][global/subtype/root][all/consensus/non_consensus][all/first/second/third][mean/std/rates]
    There is also the entry for the time vector : divergence[time]
    """
    div_dict = {}

    for region in ["env", "pol", "gag"]:
        div_dict[region] = {}
        for reference in ["founder", "any", "subtypes", "root"]:
            div_dict[region][reference] = {}
            for consensus in ["global", "subtype", "root"]:
                print(f"Computing bootstrapped divergence for {region} {reference} {consensus}")
                time, bootstrap_dict = bootstrap_divergence_in_time(
                    region, reference, consensus, nb_bootstrap)
                div_dict["time"] = time
                div_dict[region][reference][consensus] = bootstrap_dict

    return div_dict


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # region = "pol"
    # trajectories = trajectory.load_trajectory_list("data/WH/Trajectory_list_any.json")
    # trajectories = [traj for traj in trajectories if traj.region == region]
    # bootstrap_dict, times = make_bootstrap_mean_dict(trajectories, nb_bootstrap=10)
    #
    # colors = ["C0", "C1", "C2"]
    # plt.figure()
    # for ii, key in enumerate(bootstrap_dict["rev"].keys()):
    #     plt.plot(times, bootstrap_dict["rev"][key]["mean"], '-', color=colors[ii], label=key)
    #     plt.plot(times, bootstrap_dict["non_rev"][key]["mean"], '--', color=colors[ii], label=key)
    #
    # plt.show()

    # div_dict = make_bootstrap_div_dict(5)

    # time, bootstrapped_dict = bootstrap_divergence_in_time("pol", "founder", "global")

    # plt.figure()
    # plt.title("all")
    # for key in bootstrapped_dict["all"].keys():
    #     plt.plot(time, bootstrapped_dict["all"][key]["mean"], label=key)
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    # plt.title("consensus")
    # for key in bootstrapped_dict["consensus"].keys():
    #     plt.plot(time, bootstrapped_dict["consensus"][key]["mean"], label=key)
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    # plt.title('non_consensus')
    # for key in bootstrapped_dict["non_consensus"].keys():
    #     plt.plot(time, bootstrapped_dict["non_consensus"][key]["mean"], label=key)
    # plt.legend()
    # plt.grid()
    # plt.show()

    # t, div_dict = bootstrap_divergence_in_time("pol", "founder", "global")

    # trajectories = trajectory.load_trajectory_list("data/WH_test/Trajectory_list_any.json")
    # mean_dict, time = make_bootstrap_mean_dict(trajectories, 10)
    mean_dict = trajectory.load_mean_in_time_dict("data/WH_test/bootstrap_mean_dict_any.json")
    time = trajectory.create_time_bins(400)
    time = 0.5 * (time[:-1] + time[1:]) / 365  # In years

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    colors = ["C0", "C1", "C2", "C4"]

    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["rev", "non_rev"], ["-", "--"]):
            for jj, key2 in enumerate(["syn", "non_syn"]):
                mean = mean_dict[key][key2][str(freq_range)]["mean"]
                std = mean_dict[key][key2][str(freq_range)]["std"]
                axs[jj].plot(time, mean, line, color=colors[ii])
                axs[jj].fill_between(time, mean - std, mean + std, color=colors[ii], alpha=0.3)

    plt.show()
