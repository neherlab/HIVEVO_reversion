import filenames
import numpy as np
import tools


def create_time_bins(bin_size=400):
    """
    Create time bins for the mean in time analysis. It does homogeneous bins, except for the one at t=0 that
    only takes point where t=0. Bin_size is in days.
    """
    time_bins = [-5, 5]
    interval = [-600, 3000]
    while time_bins[0] > interval[0]:
        time_bins = [time_bins[0] - bin_size] + time_bins
    while time_bins[-1] < interval[1]:
        time_bins = time_bins + [time_bins[-1] + bin_size]

    return np.array(time_bins)


def get_mean_in_time(trajectories, bin_size=400, freq_range=[0.4, 0.6]):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    trajectories = copy.deepcopy(trajectories)

    # Create bins and select trajectories going through the freq_range
    time_bins = create_time_bins(bin_size)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    t_traj = np.array([])
    f_traj = np.array([])
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        t_traj = np.concatenate((t_traj, traj.t))
        f_traj = np.concatenate((f_traj, traj.frequencies))

    # Binning of all the data in the time bins
    filtered_fixed = [traj for traj in trajectories if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in trajectories if traj.fixation == "lost"]
    freqs, fixed, lost = [], [], []
    for ii in range(len(time_bins) - 1):
        freqs = freqs + [f_traj[np.logical_and(t_traj >= time_bins[ii], t_traj < time_bins[ii + 1])]]
        fixed = fixed + [len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])]
        lost = lost + [len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])]

    # Computation of the mean in each bin, active trajectories contribute their current frequency,
    # fixed contribute 1 and lost contribute 0
    mean = []
    for ii in range(len(freqs)):
        mean = mean + [np.sum(freqs[ii]) + fixed[ii]]
        if len(freqs[ii]) + fixed[ii] + lost[ii] != 0:
            mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])
        else:
            mean[-1] = np.nan
    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


def make_mean_in_time_dict(trajectories):
    regions = ["env", "pol", "gag", "all"]
    means = {}
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    times = []

    for freq_range in freq_ranges:
        means[str(freq_range)] = {}
        for region in regions:
            means[str(freq_range)][region] = {}
            for key in trajectories[region].keys():
                times, means[str(freq_range)][region][key], _, _ = get_mean_in_time(
                    trajectories[region][key], freq_range=freq_range)
    return times, means, freq_ranges


def mean_in_time_plot(fontsize=16, fill_alpha=0.15, grid_alpha=0.5, ticksize=14):
    trajectories = trajectory.load_trajectory_dict("trajectory_dict")
    # times, means, freq_ranges = make_mean_in_time_dict(trajectories)
    # bootstrap_dict, times = make_bootstrap_mean_dict(trajectories, 100)
    # save(bootstrap_dict, "bootstrap_dict")
    times = create_time_bins()
    times = 0.5 * (times[:-1] + times[1:]) / 365
    bootstrap_dict = load_dict()
    trajectories_scheme = get_trajectories_offset(trajectories["all"]["rev"], [0.4, 0.6])

    colors = ["C0", "C1", "C2", "C4"]
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7), sharey=True)

    # Plot left

    for traj in trajectories_scheme:
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.1, linewidth=1)

    mean = bootstrap_dict["rev"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["rev"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]", fontsize=fontsize)
    axs[0].set_ylabel("Frequency", fontsize=fontsize)
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].grid(grid_alpha)
    axs[0].set_xlim([-677 / 365, 3000 / 365])
    axs[0].tick_params(axis="x", labelsize=ticksize)
    axs[0].tick_params(axis="y", labelsize=ticksize)

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["Individual trajectories", "Average"],
                  fontsize=fontsize, loc="lower right")

    # Plot right
    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["rev", "non_rev"], ["-", "--"]):
            mean = bootstrap_dict[key][str(freq_range)]["mean"]
            std = bootstrap_dict[key][str(freq_range)]["std"]
            axs[1].plot(times, mean, line, color=colors[ii])
            axs[1].fill_between(times, mean - std, mean + std, color=colors[ii], alpha=fill_alpha)

    line1, = axs[1].plot([0], [0], "k-")
    line2, = axs[1].plot([0], [0], "k--")
    line3, = axs[1].plot([0], [0], "-", color=colors[0])
    line4, = axs[1].plot([0], [0], "-", color=colors[1])
    line5, = axs[1].plot([0], [0], "-", color=colors[2])

    axs[1].set_xlabel("Time [years]", fontsize=fontsize)
    # axs[1].set_ylabel("Frequency", fontsize=fontsize)
    axs[1].set_ylim([-0.03, 1.03])
    axs[1].grid(grid_alpha)
    axs[1].tick_params(axis="x", labelsize=ticksize)
    axs[1].legend([line3, line4, line5, line1, line2],
                  ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]", "reversion", "non-reversion"],
                  fontsize=fontsize, ncol=2, loc="lower right")

    plt.show()


if __name__ == "__main__":
