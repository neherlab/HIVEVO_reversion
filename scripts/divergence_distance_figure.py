import matplotlib.pyplot as plt
import numpy as np
import divergence
from distance_in_time import get_reference_sequence, get_mean_distance_in_time


if __name__ == '__main__':
    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    lines = [":", "--", "-"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    fontsize = 14
    fill_alpha = 0.15
    grid_alpha = 0.5
    tick_size = 12
    equation_pos = [(2005, 0.13), (2005, 0.102), (2005, 0.062)]

    regions = ["env", "gag", "pol"]
    time = div_dict["time"]

    lines_label = ["all", "consensus", "non-consensus"]

    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(14, 3.5), sharey=True)

    # BH plot region
    for ii, region in enumerate(regions):
        ref_file = f"data/BH/intermediate_files/{region}_1000_nt_muts.json"
        reference_sequence = get_reference_sequence(ref_file)
        alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, "")
        fit = np.polyfit(years, dist["all"], deg=1)
        axs[0].plot(years, dist["all"], '.', color=colors[ii], label=region)
        axs[0].plot(years, np.polyval(fit, years), "-", linewidth=1, color=colors[ii])
        # axs[0].text(equation_pos[ii][0], equation_pos[ii][1],
        #             f"$\sim {fit[0]:.1E}\cdot t$", fontsize=fontsize, color=colors[ii])

    axs[0].grid(grid_alpha)
    axs[0].set_xlabel("Sample date", fontsize=fontsize)
    axs[0].set_ylabel("Fraction difference", fontsize=fontsize)
    axs[0].legend(fontsize=fontsize)
    axs[0].tick_params(axis="x", labelsize=tick_size)
    axs[0].tick_params(axis="y", labelsize=tick_size)

    # BH plot pol
    ref_files = {"root": "data/BH/intermediate_files/pol_1000_nt_muts.json",
                 "global": "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta",
                 "B": "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta",
                 "C": "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta"}

    alignment_file = f"data/BH/alignments/to_HXB2/pol_1000.fasta"
    for ii, key in enumerate(["root", "global", "B", "C"]):
        reference_sequence = get_reference_sequence(ref_files[key])
        years, dist, std = get_mean_distance_in_time(
            alignment_file, reference_sequence, key if key in ["B", "C"] else "")

        fit = np.polyfit(years, dist["all"], deg=1)
        axs[1].plot(years, dist["all"], '.', color=colors[ii + 2], label=key)
        axs[1].plot(years, np.polyval(fit, years), "-", linewidth=1, color=colors[ii + 2])

    axs[1].grid(grid_alpha)
    axs[1].set_xlabel("Sample date", fontsize=fontsize)
    # axs[1].set_ylabel("Fraction difference", fontsize=fontsize)
    axs[1].legend(fontsize=fontsize)
    axs[1].tick_params(axis="x", labelsize=tick_size)
    axs[1].tick_params(axis="y", labelsize=tick_size)

    # WH plot references
    for ii, key in enumerate(div_dict["pol"].keys()):
        data = div_dict["pol"][key]["global"]["all"]["all"]["mean"]
        std = div_dict["pol"][key]["global"]["all"]["all"]["std"]
        axs[2].plot(time, data, "-", color=colors[ii+5], label=key)
        axs[2].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    axs[2].grid(grid_alpha)
    axs[2].set_xlabel("Time [years]", fontsize=fontsize)
    axs[2].set_ylabel("Divergence", fontsize=fontsize)
    axs[2].legend(fontsize=fontsize)
    axs[2].tick_params(axis="x", labelsize=tick_size)
    axs[2].tick_params(axis="y", labelsize=tick_size)
    axs[2].set_xlim([0, 6])

    # WH plot positions
    for ii, key in enumerate(div_dict["pol"]["founder"]["global"]["all"].keys()):
        data = div_dict["pol"]["founder"]["global"]["all"][key]["mean"]
        std = div_dict["pol"]["founder"]["global"]["all"][key]["std"]
        axs[3].plot(time, data, "-", color=colors[ii], label=key)
        axs[3].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    axs[3].grid(grid_alpha)
    axs[3].set_xlabel("Time [years]", fontsize=fontsize)
    axs[3].legend(fontsize=fontsize)
    axs[3].tick_params(axis="x", labelsize=tick_size)
    axs[3].tick_params(axis="y", labelsize=tick_size)
    axs[3].set_xlim([0, 6])

    plt.tight_layout()
    # plt.savefig("figures/Paper_fig_1.png", format="png")
    plt.show()
