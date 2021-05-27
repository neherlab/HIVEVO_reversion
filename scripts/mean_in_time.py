import matplotlib.pyplot as plt
import filenames
import trajectory
import bootstrap

if __name__ == "__main__":
    fontsize = 16
    fill_alpha = 0.15
    grid_alpha = 0.5
    ticksize = 14

    trajectories = trajectory.load_trajectory_dict("trajectory_dict")
    times = trajectory.create_time_bins()
    times = 0.5 * (times[:-1] + times[1:]) / 365
    bootstrap_dict = load_dict()
    trajectories_scheme = trajectory.get_trajectories_offset(trajectories["all"]["rev"], [0.4, 0.6])

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
