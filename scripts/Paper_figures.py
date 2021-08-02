from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance
import divergence
import numpy as np
import matplotlib.pyplot as plt
import trajectory
import json
plt.style.use("tex")


def make_figure_1(region, text_pos, ylim, sharey, cutoff=1977, savefig=False):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    fill_alpha = 0.15
    figsize = (6.7315, 3)

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    time = div_dict["time"]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=sharey)

    # BH plot pol
    ref_files = {"root": f"data/BH/intermediate_files/{region}_1000_nt_muts.json",
                 "B": f"data/BH/alignments/to_HXB2/{region}_1000_B_consensus.fasta",
                 "C": f"data/BH/alignments/to_HXB2/{region}_1000_C_consensus.fasta"}

    tree_file = f"data/BH/intermediate_files/timetree_{region}_1000.nwk"
    branch_length_file = f"data/BH/intermediate_files/branch_lengths_{region}_1000.json"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"

    ii = 0
    dates, lengths = get_root_to_tip_distance(tree_file, branch_length_file)
    lengths = np.array(lengths)[dates >= cutoff]
    dates = dates[dates >= cutoff]
    fit = np.polyfit(dates, lengths, deg=1)
    axs[ii].plot(dates, lengths, '.', label="RTT", color=colors[ii])
    axs[ii].plot(dates, np.polyval(fit, dates), "-", linewidth=1, color=colors[ii])
    axs[ii].text(text_pos[0][0], text_pos[0][1],
                 f"$\\propto {round(fit[0]*1e4,1)}\\cdot 10^{{-4}} t$", color=colors[ii])
    ii += 1

    for key in ["root", "subtypes"]:
        if key == "subtypes":
            ref_sequence = get_reference_sequence(ref_files["B"])
            years, dist, std, nb = get_mean_distance_in_time(alignment_file, ref_sequence, subtype="B")
            ref_sequence = get_reference_sequence(ref_files["C"])
            years2, dist2, std2, nb2 = get_mean_distance_in_time(alignment_file, ref_sequence, subtype="C")

            # Averaging the subtypes distance
            for key2 in dist.keys():
                idxs = np.isin(years, years2)
                dist[key2][idxs] = (nb[idxs] * dist[key2][idxs] + nb2 *
                                    dist2[key2]) / (nb[idxs] + nb2)
        else:
            reference_sequence = get_reference_sequence(ref_files[key])
            years, dist, std, _ = get_mean_distance_in_time(alignment_file, reference_sequence)

        dist["all"] = dist["all"][years >= cutoff]
        years = years[years >= cutoff]
        fit = np.polyfit(years, dist["all"], deg=1)
        axs[0].plot(years, dist["all"], '.', color=colors[ii], label=key)
        axs[0].plot(years, np.polyval(fit, years), "-", color=colors[ii])
        axs[0].text(text_pos[ii][0], text_pos[ii][1],
                    f"$\\propto {round(fit[0]*1e4,1)}\\cdot 10^{{-4}} t$", color=colors[ii])
        ii += 1

    axs[0].set_xlabel("Sample date")
    axs[0].set_ylabel("Distance")
    axs[0].set_ylim(ylim)
    if region == "env":
        axs[0].legend(loc="center left")
    else:
        axs[0].legend()
    axs[0].ticklabel_format(axis="x", style="plain")

    # WH plot references
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]
    keys = ["root", "subtypes", "founder"]
    for key in keys:
        data = div_dict[region][key]["global"]["all"]["all"]["mean"][idxs]
        std = div_dict[region][key]["global"]["all"]["all"]["std"][idxs]
        axs[1].plot(time, data, "-", color=colors[ii - 2], label=key)
        axs[1].fill_between(time, data + std, data - std, color=colors[ii - 2], alpha=fill_alpha)
        fit = np.polyfit(time, data, deg=1)
        axs[1].text(text_pos[ii][0], text_pos[ii][1],
                    f"$\\propto{round(fit[0]*1e4,1)}\\cdot 10^{{-4}} t$", color=colors[ii - 2])
        ii += 1

    axs[1].set_xlabel("Time [years]")
    if not sharey:
        axs[1].set_ylabel("Divergence")
    axs[1].legend()
    axs[1].set_xlim([-0.3, 5.5])

    if savefig:
        fig.savefig(f"figures/Distance_{region}.pdf")


def make_figure_2(region, text, savefig=False):
    fill_alpha = 0.15
    colors = ["C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    lines = ["-", "-", "--"]
    figsize = (6.7315, 3)

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    time = div_dict["time"]
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True, sharex=True)
    # Left plot
    labels = ["all", "consensus", "non-consensus"]
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        data = div_dict[region]["founder"]["global"][key]["all"]["mean"][idxs]
        std = div_dict[region]["founder"]["global"][key]["all"]["std"][idxs]

        axs[0].plot(time, data, lines[ii], color=colors[ii], label=labels[ii])
        axs[0].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    axs[0].text(text[0][1][0], text[0][1][1], text[0][0], color=colors[1])
    axs[0].text(text[1][1][0], text[1][1][1], text[1][0], color=colors[2])

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Divergence from founder")
    axs[0].legend()
    axs[0].set_xlim([-0.3, 5.5])

    # Right plot
    for ii, key in enumerate(["consensus", "non_consensus"]):
        for jj, key2 in enumerate(["first", "second", "third"]):
            data = div_dict["pol"]["founder"]["global"][key][key2]["mean"][idxs]
            std = div_dict[region]["founder"]["global"][key][key2]["std"][idxs]
            axs[1].plot(time, data, lines[ii + 1], color=colors[jj + 3])
            axs[1].fill_between(time, data + std, data - std, color=colors[jj + 3], alpha=fill_alpha)
    axs[1].text(text[2][1][0], text[2][1][1], text[2][0], color=colors[3])
    axs[1].text(text[3][1][0], text[3][1][1], text[3][0], color=colors[4])
    axs[1].text(text[4][1][0], text[4][1][1], text[4][0], color=colors[5])

    axs[1].set_xlabel("Time [years]")

    # Legend
    for ii, label in enumerate(["consensus", "non-consensus"]):
        axs[1].plot([0], [0], lines[ii + 1], color="k", label=label)
    for jj, label in enumerate(["1st", "2nd", "3rd"]):
        axs[1].plot([0], [0], lines[0], color=colors[jj + 3], label=label)

    axs[1].legend()

    if savefig:
        plt.savefig(f"figures/Divergence_details_{region}.pdf")


def make_figure_3(savegif=False):
    reference = "any"  # "any" or "subtypes"
    fill_alpha = 0.15
    figsize = (6.7315, 3)
    colors = ["C0", "C1", "C2", "C4"]

    trajectory_file = f"data/WH/Trajectory_list_{reference}.json"
    mean_in_time_file = f"data/WH/bootstrap_mean_dict_{reference}.json"

    # Data loading
    trajectories = trajectory.load_trajectory_list(trajectory_file)
    times = trajectory.create_time_bins(400)
    times = 0.5 * (times[:-1] + times[1:]) / 365  # In years
    bootstrap_dict = trajectory.load_mean_in_time_dict(mean_in_time_file)

    # Selecting reversion trajectories in [0.4, 0.6] for left pannel
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    trajectories_scheme = [traj for traj in trajectories if traj.reversion]
    trajectories_scheme = trajectory.offset_trajectories(trajectories_scheme, [0.4, 0.6])

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True, sharex=True)

    # Plot left
    for traj in trajectories_scheme:
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.2, linewidth=0.5)

    mean = bootstrap_dict["rev"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["rev"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].set_xlim([-2, 8.5])

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["Individual trajectories", "Average"], loc="lower right")

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

    axs[1].set_xlabel("Time [years]")
    axs[1].set_ylim([-0.03, 1.03])
    axs[1].legend([line3, line4, line5, line1, line2],
                  ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]", "reversion", "non-reversion"],
                  ncol=2, loc="lower right")

    if savefig:
        plt.savefig(f"figures/mean_in_time_{reference}.pdf")


def compute_rates(region):
    """
    Returns the mutation rates from the distance to root, distance to subtype consensus and root to tip
    distance.
    """
    # BH rate files
    reference_file = {}
    reference_file["root"] = f"data/BH/intermediate_files/{region}_1000_nt_muts.json"
    reference_file["B"] = f"data/BH/alignments/to_HXB2/{region}_1000_B_consensus.fasta"
    reference_file["C"] = f"data/BH/alignments/to_HXB2/{region}_1000_C_consensus.fasta"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    tree_file = f"data/BH/intermediate_files/timetree_{region}_1000.nwk"
    branch_length_file = f"data/BH/intermediate_files/branch_lengths_{region}_1000.json"

    # BH GTR files
    gtr_file = f"data/BH/mutation_rates/{region}_1000.json"

    # Rates from hamming distance
    rates = {"root": {}, "subtypes": {}}

    # BH to root
    reference_sequence = get_reference_sequence(reference_file["root"])
    years, dist, std, _ = get_mean_distance_in_time(alignment_file, reference_sequence)
    for key in dist.keys():
        fit = np.polyfit(years, dist[key], deg=1)
        rates["root"][key] = fit[0]

    ref_sequence = get_reference_sequence(reference_file["B"])
    years, dist, std, nb = get_mean_distance_in_time(alignment_file, ref_sequence, subtype="B")
    ref_sequence = get_reference_sequence(reference_file["C"])
    years2, dist2, std2, nb2 = get_mean_distance_in_time(alignment_file, ref_sequence, subtype="C")

    # Averaging the subtypes distance
    for key in dist.keys():
        idxs = np.isin(years, years2)
        dist[key][idxs] = (nb[idxs] * dist[key][idxs] + nb2 *
                           dist2[key]) / (nb[idxs] + nb2)
        fit = np.polyfit(years, dist[key], deg=1)
        rates["subtypes"][key] = fit[0]

    # Rates from root to tip distance
    dates, lengths = get_root_to_tip_distance(tree_file, branch_length_file)
    fit = np.polyfit(dates, lengths, deg=1)
    rates["rtt"] = fit[0]

    # BH rates from GTR estimates
    with open(gtr_file) as f:
        rates["GTR"] = json.load(f)

    # WH rates
    WH_file = "data/WH/avg_rate_dict.json"
    WH_rate_dict = divergence.load_avg_rate_dict(WH_file)
    rates["WH"] = WH_rate_dict["pol"]

    return rates


def make_figure_4(region, savefig=False):
    """
    Plot for the mutation rates.
    """
    markersize = 5
    colors = {"all": "k", "first": "C0", "second": "C1", "third": "C2"}
    labels = ["H-root", "H-subtype", "RTT", "GTR", "WH_root", "WH_subtypes", "WH_founder"]

    rates = compute_rates(region)

    plt.figure()
    # BH stuff
    for ii, key in enumerate(["root", "subtypes"]):
        for key2 in ["all", "first", "second", "third"]:
            plt.plot(ii, rates[key][key2], 'o', color=colors[key2])

    plt.plot(2, rates["rtt"], 'o', color=colors["all"], markersize=markersize)

    for key in rates["GTR"].keys():
        plt.plot(3, rates["GTR"][key], "o", color=colors[key], markersize=markersize, label=key)

    # WH stuff
    for key in ["all", "first", "second", "third"]:
        plt.plot(4, rates["WH"]["root"]["global"]["all"][key],
                 'o', color=colors[key], markersize=markersize)
        plt.plot(5, rates["WH"]["subtypes"]["global"]["all"][key],
                 'o', color=colors[key], markersize=markersize)
        plt.plot(6, rates["WH"]["founder"]["global"]["all"][key],
                 'o', color=colors[key], markersize=markersize)

    plt.xticks(range(len(labels)), labels, rotation=14)
    plt.ylabel("Mutation rates")
    plt.legend()
    if savefig:
        plt.savefig(f"figures/Rates_{region}.pdf")
    plt.show()


if __name__ == '__main__':
    fig1 = False
    fig2 = False
    fig3 = False
    fig4 = True
    savefig = True

    if fig1:
        text = {"env": [(2000, 0.36), (2000, 0.16), (2000, 0.03), (1.2, 0.079), (1.2, 0.045), (1.2, 0.024)],
                "pol": [(2000, 0.107), (2000, 0.0585), (2000, 0.024), (1.2, 0.072), (1.2, 0.042), (1.2, 0.01)],
                "gag": [(2000, 0.109), (2000, 0.068), (2000, 0.03), (1.2, 0.077), (1.2, 0.047), (1.2, 0.0165)]}

        ylim = {"env": [0, 0.45], "pol": [0, 0.12], "gag": [0, 0.145]}
        sharey = {"env": False, "pol": True, "gag": True}

        for region in ["env", "pol", "gag"]:
            make_figure_1(region, text[region], ylim[region], sharey[region], savefig=savefig)

    if fig2:
        # from the fraction_consensus.py file
        text = {"env": [("92%", [4.1, 0.003]), ("8%", [4.1, 0.062]), ("7%", [4.1, 0.018]),
                        ("6%", [4.1, 0.09]), ("12%", [4.1, 0.058])],
                "pol": [("94%", [4.1, -0.002]), ("6%", [4.1, 0.054]), ("4%", [4.1, 0.048]),
                        ("2%", [4.1, 0.111]), ("11%", [4.1, 0.023])],
                "gag": [("93%", [4.1, 0.001]), ("7%", [4.1, 0.078]), ("5%", [4.1, 0.051]),
                        ("4%", [4.1, 0.09]), ("11%", [4.1, 0.029])]}
        for region in ["env", "pol", "gag"]:
            make_figure_2(region, text[region], savefig)

    if fig3:
        make_figure_3(savefig)
    if fig4:
        make_figure_4("pol", savefig)
    plt.show()
