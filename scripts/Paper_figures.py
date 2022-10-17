"""
Script to create the figures of the paper
"""
from distance_in_time import get_reference_sequence, get_mean_distance_in_time
import divergence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import trajectory
import glob
plt.style.use("tex.mplstyle")
figsize_wide = (6.7315, 3)
figsize_narrow = (5.3852, 3)
fill_alpha = 0.15
fontsize_panel_label = 12


def make_figure_1CD(region, text_pos, ylim, sharey, cutoff=1977, savefig=False):
    from gtr_modeling import get_RTT
    from Bio import Phylo
    from bootstrap_BH import load_bootstrap
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    figsize = figsize_narrow if region == "pol" else figsize_wide

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    time = div_dict["time"]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=sharey)

    # BH plot
    ref_files = {"root": f"data/BH/intermediate_files/{region}_nt_muts.json",
                 "B": f"data/BH/alignments/to_HXB2/{region}_B_consensus.fasta",
                 "C": f"data/BH/alignments/to_HXB2/{region}_C_consensus.fasta"}

    bootstrap_files = {"root": f"data/BH/bootstraps/{region}_root_bootstrap.json",
                       "subtypes": f"data/BH/bootstraps/{region}_subtypes_bootstrap.json",
                       "RTT": f"data/BH/bootstraps/{region}_RTT_bootstrap.json"}

    tree_file = f"data/BH/intermediate_files/tree_{region}.nwk"
    alignment_file = f"data/BH/alignments/to_HXB2/{region}.fasta"
    fit_err = np.std(load_bootstrap(bootstrap_files["RTT"])["rates"])

    ii = 0
    lengths, dates = get_RTT(Phylo.read(tree_file, "newick"))
    lengths = np.array(lengths)[dates >= cutoff]
    dates = dates[dates >= cutoff]
    lengths, dates, ranges = average_rtt(lengths, dates)
    fit = np.polyfit(dates, lengths, deg=1)
    axs[0].plot(dates, lengths, '.', label="RTT", color=colors[ii])
    axs[0].fill_between(dates, ranges[:, 0], ranges[:, 1], color=colors[ii], alpha=fill_alpha)
    axs[0].plot(dates, np.polyval(fit, dates), "-", linewidth=1, color=colors[ii])
    axs[0].text(
        text_pos[0][0],
        text_pos[0][1],
        f"$\\propto {round(fit[0]*1e4,1)} \pm {round(fit_err*1e4,1)}\\cdot 10^{{-4}}$", color=colors[ii])
    if region == "pol":
        axs[0].annotate("C", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    else:
        axs[0].annotate("A", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    ii += 1

    for key in ["root", "subtypes"]:
        if key == "subtypes":
            ref_sequence = get_reference_sequence(ref_files["B"])
            years, dist, std, nb, ranges = get_mean_distance_in_time(
                alignment_file, ref_sequence, subtype="B")
            ref_sequence = get_reference_sequence(ref_files["C"])
            years2, dist2, std2, nb2, ranges2 = get_mean_distance_in_time(
                alignment_file, ref_sequence, subtype="C")

            # Averaging the subtypes distance
            for key2 in dist.keys():
                idxs = np.isin(years, years2)
                idxs2 = np.isin(years2, years)
                dist[key2][idxs] = (nb[idxs] * dist[key2][idxs] + nb2[idxs2] *
                                    dist2[key2][idxs2]) / (nb[idxs] + nb2[idxs2])
                ranges[key2][idxs] = ((nb[idxs] * ranges[key2][idxs].T + nb2[idxs2] *
                                       ranges2[key2][idxs2].T) / (nb[idxs] + nb2[idxs2])).T

        else:
            reference_sequence = get_reference_sequence(ref_files[key])
            years, dist, std, _, ranges = get_mean_distance_in_time(
                alignment_file, reference_sequence, subtype="")
            # years, dist, std, _ = get_mean_distance_in_time(alignment_file, reference_sequence)

        ind = years >= cutoff
        dist["all"] = dist["all"][ind]
        years = years[ind]
        fit = np.polyfit(years, dist["all"], deg=1)
        fit_err = np.std(load_bootstrap(bootstrap_files[key])["rates"])
        axs[0].plot(years, dist["all"], '.', color=colors[ii], label=key)
        # axs[0].fill_between(years, ranges["all"][ind, 0], ranges["all"][ind, 1], color=colors[ii], alpha=0.1)
        axs[0].fill_between(
            years, dist["all"] - std["all"][ind],
            dist["all"] + std["all"][ind],
            color=colors[ii],
            alpha=0.1)
        axs[0].plot(years, np.polyval(fit, years), "-", color=colors[ii])
        axs[0].text(
            text_pos[ii][0],
            text_pos[ii][1],
            f"$\\propto {round(fit[0]*1e4,1)} \pm {round(fit_err*1e4,1)} \\cdot 10^{{-4}}$", color=colors
            [ii])
        ii += 1

    axs[0].set_xlabel("Sample date")
    axs[0].set_ylabel("Distance")
    axs[0].set_ylim(ylim)
    # axs[0].legend()
    axs[0].ticklabel_format(axis="x", style="plain")

    # WH plot references
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]
    rate_dict = divergence.load_avg_rate_dict("data/WH/rate_dict.json")
    keys = ["root", "subtypes", "founder"]
    axs[1].plot([0], [0], "-", color=colors[0], label="RTT")
    for key in keys:
        data = div_dict[region][key]["global"]["all"]["all"]["mean"][idxs]
        std = div_dict[region][key]["global"]["all"]["all"]["std"][idxs]
        axs[1].plot(time, data, "-", color=colors[ii - 2], label=key)
        axs[1].fill_between(time, data + std, data - std, color=colors[ii - 2], alpha=fill_alpha)
        fit2 = rate_dict[region][key]["global"]["all"]["all"]["rate"]
        fit2_std = rate_dict[region][key]["global"]["all"]["all"]["std"]
        axs[1].text(text_pos[ii][0], text_pos[ii][1],
                    f"$\\propto{round(fit2*1e4,1)} \pm {round(fit2_std*1e4,1)} \\cdot 10^{{-4}}$",
                    color=colors[ii - 2])
        ii += 1

    axs[1].set_xlabel("Time [years]")
    if not sharey:
        axs[1].set_ylabel("Divergence")

    if region == "env":
        axs[1].legend(loc="lower right")
    else:
        axs[1].legend(loc="upper right")

    axs[1].set_xlim([-0.3, 5.5])
    if region == "pol":
        axs[1].annotate("D", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    else:
        axs[1].annotate("B", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    plt.tight_layout()
    if savefig:
        fig.savefig(f"figures/Distance_{region}.pdf")


def make_figure_S3S4(region, text, savefig=False, reference="global"):
    colors = ["C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    lines = ["-", "-", "--"]
    figsize = figsize_wide

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    time = div_dict["time"]
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True, sharex=True)
    # Left plot
    labels = ["all", "consensus", "non-consensus"]
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        data = div_dict[region]["founder"][reference][key]["all"]["mean"][idxs]
        std = div_dict[region]["founder"][reference][key]["all"]["std"][idxs]

        axs[0].plot(time, data, lines[ii], color=colors[ii], label=labels[ii])
        axs[0].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    axs[0].text(text[0][1][0], text[0][1][1], text[0][0], color=colors[1])
    axs[0].text(text[1][1][0], text[1][1][1], text[1][0], color=colors[2])
    axs[0].annotate("A", xy=(0, 1.05), xycoords="axes fraction")

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Divergence from founder")
    axs[0].legend()
    axs[0].set_xlim([-0.3, 5.5])

    # Right plot
    for ii, key in enumerate(["consensus", "non_consensus"]):
        for jj, key2 in enumerate(["first", "second", "third"]):
            data = div_dict[region]["founder"][reference][key][key2]["mean"][idxs]
            std = div_dict[region]["founder"][reference][key][key2]["std"][idxs]
            axs[1].plot(time, data, lines[ii + 1], color=colors[jj + 3])
            axs[1].fill_between(time, data + std, data - std, color=colors[jj + 3], alpha=fill_alpha)
    axs[1].text(text[2][1][0], text[2][1][1], text[2][0], color=colors[3])
    axs[1].text(text[3][1][0], text[3][1][1], text[3][0], color=colors[4])
    axs[1].text(text[4][1][0], text[4][1][1], text[4][0], color=colors[5])
    axs[1].annotate("B", xy=(0, 1.05), xycoords="axes fraction")

    axs[1].set_xlabel("Time [years]")

    # Legend
    for ii, label in enumerate(["consensus", "non-consensus"]):
        axs[1].plot([0], [0], lines[ii + 1], color="k", label=label)
    for jj, label in enumerate(["1st", "2nd", "3rd"]):
        axs[1].plot([0], [0], lines[0], color=colors[jj + 3], label=label)

    axs[1].legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(f"figures/Divergence_details_{region}.pdf")


def make_figure_3(savefig=False):
    reference = "any"  # "any" or "subtypes"
    figsize = figsize_wide
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
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.15, linewidth=0.5)

    mean = bootstrap_dict["rev"]["all"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["rev"]["all"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].set_xlim([-2, 8.5])

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["iSNV trajectory", "Average"], loc="lower right")
    axs[0].annotate("A", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)

    # Plot right
    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["rev", "non_rev"], ["-", "--"]):
            mean = bootstrap_dict[key]["all"][str(freq_range)]["mean"]
            std = bootstrap_dict[key]["all"][str(freq_range)]["std"]
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
    axs[1].annotate("B", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    plt.tight_layout()

    if savefig:
        plt.savefig(f"figures/mean_in_time_{reference}.pdf")


def average_rtt(rtt, dates, cutoff=1977):
    "Average rtt per years"
    from scipy.stats import scoreatpercentile
    years = np.unique(dates)
    lengths = []
    length_ranges = []
    years = years[years >= cutoff]
    for year in years:
        values_in_year = rtt[dates == year]
        lengths += [np.mean(values_in_year)]
        length_ranges += [[scoreatpercentile(values_in_year, 10), scoreatpercentile(values_in_year, 90)]]
    lengths = np.array(lengths)
    return lengths, years, np.array(length_ranges)


def make_figure_4(region, text, limits, savefig, colors=["C0", "C1", "C2", "C3"]):
    "GTR modeling figure"
    from gtr_modeling import get_RTT, get_ATGC_content, get_hamming_distance
    from Bio import Phylo, AlignIO

    rate_variation = 0  # 0 for no rate variation. Or 1, 2 for parameter of rate gamma distribution
    figsize = figsize_wide
    MSA_or = f"data/BH/alignments/to_HXB2/{region}.fasta"
    MSA_naive = glob.glob(f"data/modeling/generated_MSA/{region}_control_rv_{rate_variation}_*.fasta")[0]
    MSA_biased = glob.glob(
        f"data/modeling/generated_MSA/{region}_3class_binary_rv_{rate_variation}_*.fasta")[0]
    tree_or = f"data/BH/intermediate_files/tree_{region}.nwk"
    tree_naive = glob.glob(f"data/modeling/generated_trees/{region}_control_rv_{rate_variation}_*.nwk")[0]
    tree_biased = glob.glob(
        f"data/modeling/generated_trees/{region}_3class_binary_rv_{rate_variation}_*.nwk")[0]
    root_path = f"data/BH/intermediate_files/{region}_nt_muts.json"

    MSA = {}
    for key, path in zip(["original", "naive", "biased"], [MSA_or, MSA_naive, MSA_biased]):
        MSA[key] = AlignIO.read(path, "fasta")
        MSA[key] = np.array(MSA[key])
    tree_or = Phylo.read(tree_or, "newick")
    tree_naive = Phylo.read(tree_naive, "newick")
    tree_biased = Phylo.read(tree_biased, "newick")
    nucleotides = ["A", "T", "G", "C"]
    root_seq = get_reference_sequence(root_path)

    plt.figure(figsize=figsize)

    # Top-Left plot
    nb_bins = 500
    ax1 = plt.subplot(221)

    ATGC = {}
    for key in MSA.keys():
        ATGC[key] = get_ATGC_content(MSA[key])

    tmp = [0, -0.3, 0.3]
    for ii, key in enumerate(ATGC.keys()):
        data = []
        pos = []
        for jj in range(4):
            data += [ATGC[key][:, jj]]
            pos += [jj + tmp[ii]]
        ax1.violinplot(data, pos, showmeans=True, showextrema=True, showmedians=False, widths=0.3)

    labels = []
    for ii, key in enumerate(ATGC.keys()):
        labels.append((mpl.patches.Patch(color=colors[ii]), ["BH data", "WH naive", "WH reversion"][ii]))

    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(nucleotides)
    ax1.set_ylabel("ATGC content")
    ax1.annotate("A", xy=(0, 1.1), xycoords="axes fraction", fontsize=fontsize_panel_label)

    # Bottom-Left plot
    nb_bins = 200
    ax2 = plt.subplot(223)

    distances = {}
    for key in MSA.keys():
        distances[key] = get_hamming_distance(MSA[key], root_seq)

    for ii, key in enumerate(distances.keys()):
        hist, bins = np.histogram(distances[key], bins=nb_bins, range=[0, 0.5])
        hist = hist / 1000
        bins = 0.5 * (bins[:-1] + bins[1:])
        ax2.plot(bins, hist, "-", color=colors[ii], label=["BH data", "WH naive", "WH reversion"][ii])

    ax2.set_xlabel("Distance to root sequence")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(limits)
    ax2.annotate("B", xy=(0, 1.1), xycoords="axes fraction", fontsize=fontsize_panel_label)

    # Right plot
    trees = {"original": tree_or, "naive": tree_naive, "biased": tree_biased}
    rtts, dates, ranges, fits = {}, {}, {}, {}
    labels = ["BH data", "WH naive", "WH reversion"]

    ax3 = plt.subplot(122)
    for key in trees.keys():
        rtts[key], dates[key] = get_RTT(trees[key])
        rtts[key], dates[key], ranges[key] = average_rtt(rtts[key], dates[key])
        fits[key] = np.polyfit(dates[key], rtts[key], deg=1)

    for ii, key in enumerate(rtts.keys()):
        ax3.plot(dates[key], rtts[key], '.', label=f"{labels[ii]}", color=colors[ii])
        ax3.fill_between(dates[key], ranges[key][:, 0], ranges[key][:, 1], color=colors[ii], alpha=fill_alpha)
        ax3.plot(dates[key], np.polyval(fits[key], dates[key]), "-", color=colors[ii])
        ax3.text(text[ii][0], text[ii][1],
                 f"$\\propto {round(fits[key][0]*1e4,1)}\\cdot 10^{{-4}}$", color=colors[ii])

    ax3.annotate("C", xy=(0, 1.035), xycoords="axes fraction", fontsize=fontsize_panel_label)
    ax3.set_xlabel("Years")
    ax3.ticklabel_format(axis="x", style="plain")
    ax3.set_ylim(0)
    ax3.set_ylabel("RTT")
    ax3.legend(loc="upper left")

    plt.tight_layout()
    if savefig:
        plt.savefig(f"figures/RTT_modeling_{region}.pdf")


def make_figure_S7(savefig=False):
    """
    Plot for equivalant of the reversion in time figure but using synonymous / non-synonymous in this case.
    """
    reference = "any"  # "any" or "subtypes"
    figsize = figsize_wide
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
    trajectories_scheme = [traj for traj in trajectories if traj.synonymous]
    trajectories_scheme = trajectory.offset_trajectories(trajectories_scheme, [0.4, 0.6])

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True, sharex=True)

    # Plot left
    for traj in trajectories_scheme:
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.15, linewidth=0.5)

    mean = bootstrap_dict["all"]["syn"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["all"]["syn"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].set_xlim([-2, 8.5])

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["Individual trajectories", "Average"], loc="lower right")
    axs[0].annotate("A", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)

    # Plot right
    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["syn", "non_syn"], ["-", "--"]):
            mean = bootstrap_dict["all"][key][str(freq_range)]["mean"]
            std = bootstrap_dict["all"][key][str(freq_range)]["std"]
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
                  ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]", "synonymous", "non-synonymous"],
                  ncol=2, loc="upper left")
    axs[1].annotate("B", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    plt.tight_layout()

    if savefig:
        plt.savefig(f"figures/mean_in_time_syn_{reference}.pdf")


def make_figure_S6(savefig):
    """
    Plot for the details about mean in time
    """
    regions = ["env", "pol", "gag"]
    colors = ["C0", "C1", "C2"]
    figsize = figsize_wide
    lines = ["-", "--"]

    dict_names = {}
    for region in regions:
        dict_names[region] = f"data/WH/bootstrap_mean_dict_{region}.json"

    bootstrap_dicts = {}
    for region in regions:
        bootstrap_dicts[region] = trajectory.load_mean_in_time_dict(dict_names[region])

    times = trajectory.create_time_bins(400)
    times = 0.5 * (times[:-1] + times[1:]) / 365  # In years

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=figsize, sharey=True, sharex=True)
    for yy, region in enumerate(regions):
        for ii, key in enumerate(bootstrap_dicts["pol"]["rev"]["all"].keys()):
            for jj, key2 in enumerate(["rev", "non_rev"]):
                mean = bootstrap_dicts[region][key2]["all"][key]["mean"]
                std = bootstrap_dicts[region][key2]["all"][key]["std"]
                axs[yy].plot(times, mean, lines[jj], color=colors[ii])
                axs[yy].fill_between(times, mean - std, mean + std, color=colors[ii], alpha=fill_alpha)

        axs[yy].set_xlabel("Time [years]")
        axs[yy].annotate(region, xy=(0.44, 1.03), xycoords="axes fraction", fontsize=fontsize_panel_label)

    line1, = axs[2].plot([0], [0], "k-")
    line2, = axs[2].plot([0], [0], "k--")
    line3, = axs[2].plot([0], [0], "-", color=colors[0])
    line4, = axs[2].plot([0], [0], "-", color=colors[1])
    line5, = axs[2].plot([0], [0], "-", color=colors[2])

    axs[2].legend([line3, line4, line5, line1, line2],
                  ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]", "rev", "non-rev"],
                  ncol=2, loc="lower right")
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].set_ylabel("Frequency")

    if savefig:
        plt.savefig(f"figures/Mean_in_time_details.pdf")


def make_figure_2(region, text, savefig=False, reference="global"):
    colors = ["C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    lines = ["-", "-", "--"]
    markers = ["o", "<", "s"]
    figsize = figsize_wide
    positions = ["first", "second", "third"]
    regions = ["pol", "env", "gag"]

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    rate_dict = divergence.load_avg_rate_dict("data/WH/rate_dict.json")
    time = div_dict["time"]
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]
    mpl.rcParams['figure.autolayout'] = False

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig)
    # Left plot
    labels = ["all", "consensus", "non-consensus"]
    ax1 = fig.add_subplot(gs[0])
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        data = div_dict[region]["founder"][reference][key]["all"]["mean"][idxs]
        std = div_dict[region]["founder"][reference][key]["all"]["std"][idxs]

        ax1.plot(time, data, lines[ii], color=colors[ii], label=labels[ii])
        ax1.fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    ax1.text(text[0][1][0], text[0][1][1], text[0][0], color=colors[1])
    ax1.text(text[1][1][0], text[1][1][1], text[1][0], color=colors[2])
    ax1.annotate("A", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)

    ax1.set_xlabel("Time [years]")
    ax1.set_ylabel("Divergence from founder")
    ax1.legend()
    ax1.set_xlim([-0.3, 5.5])

    # Middle plot
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    for ii, key in enumerate(["consensus", "non_consensus"]):
        for jj, key2 in enumerate(positions):
            data = div_dict[region]["founder"][reference][key][key2]["mean"][idxs]
            std = div_dict[region]["founder"][reference][key][key2]["std"][idxs]
            ax2.plot(time, data, lines[ii + 1], color=colors[jj + 3])
            ax2.fill_between(time, data + std, data - std, color=colors[jj + 3], alpha=fill_alpha)
    ax2.text(text[2][1][0], text[2][1][1], text[2][0], color=colors[3])
    ax2.text(text[3][1][0], text[3][1][1], text[3][0], color=colors[4])
    ax2.text(text[4][1][0], text[4][1][1], text[4][0], color=colors[5])
    ax2.annotate("B", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)

    ax2.set_xlabel("Time [years]")

    for ii, label in enumerate(["consensus", "non-consensus"]):
        ax2.plot([0], [0], lines[ii + 1], color="k", label=label)
    for jj, label in enumerate(["1st", "2nd", "3rd"]):
        ax2.plot([0], [0], lines[0], color=colors[jj + 3], label=label)

    ax2.legend()
    ax2.tick_params(labelleft=False)

    # Right plot
    ax3 = fig.add_subplot(gs[2])
    for ii, reg in enumerate(regions):
        for jj, pos in enumerate(["first", "second", "third"]):
            rate_consensus = rate_dict[reg]["founder"][reference]["consensus"][pos]["rate"]
            std_consensus = rate_dict[reg]["founder"][reference]["consensus"][pos]["std"]
            rate_non_consensus = rate_dict[reg]["founder"][reference]["non_consensus"][pos]["rate"]
            std_non_consensus = rate_dict[reg]["founder"][reference]["non_consensus"][pos]["std"]

            ratio = rate_non_consensus / rate_consensus
            std = ratio*np.sqrt((std_consensus/rate_consensus)**2 + (std_non_consensus/rate_non_consensus)**2)
            if ii == 0:
                ax3.plot(ii, ratio, markers[jj], color=colors[jj+3], label=["1st", "2nd", "3rd"][jj])
                ax3.errorbar(ii, ratio, yerr=std, color=colors[jj+3], capsize=3)
            else:
                ax3.plot(ii, ratio, markers[jj], color=colors[jj+3])
                ax3.errorbar(ii, ratio, yerr=std, color=colors[jj+3], capsize=3)

    ax3.set_ylabel("Rate non-consensus / rate consensus")
    ax3.set_xticks([0, 1, 2])
    ax3.set_xlim([-0.7, 2.7])
    ax3.set_xticklabels(regions, fontsize=9)
    ax3.annotate("C", xy=(0, 1.05), xycoords="axes fraction", fontsize=fontsize_panel_label)
    ax3.set_yscale("log")
    # ax3.set_ylim([0, 40])
    ax3.legend()

    if savefig:
        plt.savefig("figures/Fig_2.pdf")


def make_poster_figures(savefig=False):
    # Reversion figure
    reference = "any"  # "any" or "subtypes"
    figsize = figsize_narrow
    colors = ["C2", "C5", "C4"]

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
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.1, linewidth=0.5)

    mean = bootstrap_dict["rev"]["all"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["rev"]["all"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].set_xlim([-2, 8.5])

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["iSNV trajectory", "Average"], loc="lower right")

    # Plot right
    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["rev", "non_rev"], ["-", "--"]):
            mean = bootstrap_dict[key]["all"][str(freq_range)]["mean"]
            std = bootstrap_dict[key]["all"][str(freq_range)]["std"]
            axs[1].plot(times, mean, line, color=colors[ii])
            axs[1].fill_between(times, mean - std, mean + std, color=colors[ii], alpha=fill_alpha)

    line1, = axs[1].plot([0], [0], "k-")
    line2, = axs[1].plot([0], [0], "k--")

    axs[1].set_xlabel("Time [years]")
    axs[1].set_ylim([-0.03, 1.03])
    axs[1].legend([line1, line2],
                  ["reversion", "non-reversion"],
                  ncol=2, loc="lower right")
    if savefig:
        plt.savefig(f"figures/Poster_mean_in_time.pdf")

    # Divergence figure
    colors = ["C3", "C0", "C1"]
    lines = ["-", "-", "--"]
    figsize = figsize_narrow
    region = "pol"
    reference = "global"
    text = [
        ("94%", [4.1, 0.002]),
        ("6%", [4.1, 0.031])]

    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    time = div_dict["time"]
    idxs = time < 5.3  # Time around which some patients stop being followed
    time = time[idxs]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True, sharex=True)
    # Left plot
    labels = ["all", "consensus", "non-consensus"]
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        data = div_dict[region]["founder"][reference][key]["all"]["mean"][idxs]
        std = div_dict[region]["founder"][reference][key]["all"]["std"][idxs]

        axs[0].plot(time, data, lines[ii], color=colors[ii], label=labels[ii])
        axs[0].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    axs[0].text(text[0][1][0], text[0][1][1], text[0][0], color=colors[1])
    axs[0].text(text[1][1][0], text[1][1][1], text[1][0], color=colors[2])
    axs[0].set_xlabel("Time [years]")
    axs[0].set_ylabel("Divergence from founder")
    axs[0].legend()
    axs[0].set_xlim([-0.3, 5.5])
    axs[0].set_ylim([-0.002, 0.062])

    if savefig:
        plt.savefig(f"figures/Poster_divergence.pdf")


def make_figure_1A(real_mu=10.3e-4, savefig=False):
    """
    Function to generate panel A of fig 1. Uses treetime to compute the outcome we would have from a Jukes
    Cantor model with rates that are gamma distributed with parameter 2. 
    """
    from treetime import GTR_site_specific
    from scipy.stats import gamma

    L = 1000
    t_plot = np.arange(1980, 2021)
    mu = gamma.rvs(2, size=L, random_state=123) / 2  # Gamma distribution with parameter 2
    W = np.ones((4, 4))
    # W = np.array([[0., 0.763, 2.902, 0.391], # Same as I use for fig4 modeling, doesn't make a big difference
    #               [0.763, 0., 0.294, 3.551],
    #               [2.902, 0.294, 0., 0.317],
    #               [0.391, 3.551, 0.317, 0.]])
    p = np.ones((4, L))*0.25  # no bias for reversions.

    myGTR = GTR_site_specific.custom(mu, p, W, alphabet="nuc_nogap", approximate=False)
    myGTR.mu *= real_mu / myGTR.average_rate().mean()

    # dates = {"root": 1914, "subtypes": 1970,"founder": 1980}  # estimates from tree
    dates = {"root": 1925, "subtypes": 1965, "founder": 1980}  # estimates to match t = 1980
    distances = {"root": [], "subtypes": [], "founder": []}

    for key in dates:
        for t in t_plot:
            tmp = 1 - myGTR.expQt(t-dates[key])[0, 0, :]
            distances[key] += [np.mean(tmp)]

    plt.figure(figsize=(2, 1.5))
    plt.plot(t_plot, (t_plot - dates["root"])*real_mu, label="RTT")
    for key in dates:
        plt.plot(t_plot, distances[key], label=key)

    plt.xlabel("Year")
    plt.ylabel("Distance")
    plt.xlim([1978, 2022])
    plt.ylim([0, 0.14])
    plt.ticklabel_format(axis="x", style="plain")

    if savefig:
        plt.savefig(f"figures/JC_schematic.pdf")


def make_figure_S8(savefig):
    "Mean in time figure for syn+rev syn+non-rev, non-syn+rev non-syn+non-rev."

    mean_dict = trajectory.load_mean_in_time_dict("data/WH/bootstrap_mean_dict_any.json")
    time = trajectory.create_time_bins(400)
    time = 0.5 * (time[:-1] + time[1:]) / 365  # In years

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True, figsize=figsize_wide)
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    colors = ["C0", "C1", "C2", "C4"]

    for jj, key2 in enumerate(["syn", "non_syn"]):
        for ii, freq_range in enumerate(freq_ranges):
            for key, line in zip(["rev", "non_rev"], ["-", "--"]):
                mean = mean_dict[key][key2][str(freq_range)]["mean"]
                std = mean_dict[key][key2][str(freq_range)]["std"]
                axs[jj].plot(time, mean, line, color=colors[ii])
                axs[jj].fill_between(time, mean - std, mean + std, color=colors[ii], alpha=fill_alpha)
        axs[jj].set_xlabel("Time [years]")

    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Synonymous")
    axs[1].set_title("Non-synonymous")

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
        plt.savefig(f"figures/Mean_in_time_syn_rev.pdf")


if __name__ == '__main__':
    fig1A = False
    fig1 = False
    fig2 = False
    fig3 = False
    fig4 = False
    figS3S4 = False
    figS6 = False
    figS7 = False
    figS8 = False
    figPoster = False

    savefig = False

    if fig1A:
        mu = 10.3e-4
        make_figure_1A(mu, savefig)

    if fig1:
        text = {
            "env": [(2000, 0.192), (2000, 0.14), (2000, 0.045), (1.2, 0.072), (1.2, 0.058), (1.2, 0.028)],
            "pol": [(1995, 0.087), (1995, 0.056), (1995, 0.02), (1.2, 0.072), (1.2, 0.042), (1.2, 0.01)],
            "gag": [(2000, 0.125), (2000, 0.072), (2000, 0.028), (1.2, 0.085), (1.2, 0.047), (1.2, 0.0165)]}

        ylim = {"env": [0, 0.28], "pol": [0, 0.142], "gag": [0, 0.18]}
        sharey = {"env": False, "pol": True, "gag": True}

        for region in ["env", "pol", "gag"]:
            make_figure_1CD(region, text[region], ylim[region], sharey[region], savefig=savefig)

    if fig2:
        text = {"env": [("92%", [4.1, 0.001]), ("8%", [4.1, 0.062]), ("7%", [4.1, 0.145]),
                        ("6%", [4.1, 0.085]), ("12%", [4.1, 0.042])],
                "pol": [("94%", [4.1, -0.002]), ("6%", [4.1, 0.054]), ("4%", [4.1, 0.048]),
                        ("2%", [4.1, 0.111]), ("11%", [4.1, 0.023])],
                "gag": [("93%", [4.1, -0.002]), ("7%", [4.1, 0.078]), ("5%", [4.1, 0.122]),
                        ("4%", [4.1, 0.065]), ("11%", [4.1, 0.026])]}
        region = "pol"
        make_figure_2(region, text[region], savefig, reference="global")

    if fig3:
        make_figure_3(savefig)

    if fig4:
        text = {"pol": [(2003, 0.07), (2003, 0.24), (2003, 0.145)],
                "gag": [(2003, 0.1), (2003, 0.37), (2003, 0.24)],
                "env": [(2003, 0.3), (2003, 0.52), (2003, 0.1)]}
        limits = {"pol": (0.03, 0.23),
                  "gag": (0.06, 0.3),
                  "env": (0.08, 0.45)}
        for region in ["env", "pol", "gag"]:
            make_figure_4(region, text[region], limits[region], savefig)

    if figS3S4:
        text = {"env": [("92%", [4.1, 0.001]), ("8%", [4.1, 0.062]), ("7%", [4.1, 0.145]),
                        ("6%", [4.1, 0.085]), ("12%", [4.1, 0.042])],
                "pol": [("94%", [4.1, -0.002]), ("6%", [4.1, 0.054]), ("4%", [4.1, 0.048]),
                        ("2%", [4.1, 0.111]), ("11%", [4.1, 0.023])],
                "gag": [("93%", [4.1, -0.002]), ("7%", [4.1, 0.078]), ("5%", [4.1, 0.122]),
                        ("4%", [4.1, 0.065]), ("11%", [4.1, 0.026])]}
        for region in ["env", "pol", "gag"]:
            make_figure_S3S4(region, text[region], savefig, reference="global")

    if figS6:
        make_figure_S6(savefig)

    if figS7:
        make_figure_S7(savefig)

    if figS8:
        make_figure_S8(savefig)

    if figPoster:
        make_poster_figures(savefig)

    plt.show()
