import numpy as np

import trajectory


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


if __name__ == '__main__':
    trajectories = trajectory.load_trajectory_list("data/Trajectory_list_any.json")
    bootstrap_dict, times = make_bootstrap_mean_dict(trajectories)
