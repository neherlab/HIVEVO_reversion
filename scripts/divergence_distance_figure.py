import matplotlib.pyplot as plt
import numpy as np
import divergence
from distance_in_time import get_reference_sequence, get_mean_distance_in_time


if __name__ == '__main__':
    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    lines = [":", "--", "-"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    fontsize = 16
    fill_alpha = 0.15
    grid_alpha = 0.5
    tick_size = 14
    equation_pos = [(2005, 0.13), (2005, 0.102), (2005, 0.062)]

    regions = ["env", "gag", "pol"]
    time = div_dict["time"]

    lines_label = ["all", "consensus", "non-consensus"]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7), sharey=True)

    # WH plot (left)
    for ii, region in enumerate(regions):
        for jj, key2 in enumerate(div_dict[region]["founder"]["global"].keys()):
            data = div_dict[region]["founder"]["global"][key2]["all"]["mean"]
            std = div_dict[region]["founder"]["global"][key2]["all"]["std"]
            axs[0].plot(time, data, lines[jj], color=colors[ii])
            axs[0].fill_between(time, data + std, data - std, color=colors[ii], alpha=fill_alpha)

    # For legend
    for jj in range(len(lines)):
        axs[0].plot([0], [0], lines[jj], color="k", label=lines_label[jj])
    for ii in range(len(regions)):
        axs[0].plot([0], [0], "-", color=colors[ii], label=regions[ii])

    axs[0].grid(grid_alpha)
    axs[0].set_xlabel("Time [years]", fontsize=fontsize)
    axs[0].set_ylabel("Divergence", fontsize=fontsize)
    axs[0].legend(fontsize=fontsize)
    axs[0].tick_params(axis="x", labelsize=tick_size)
    axs[0].tick_params(axis="y", labelsize=tick_size)
    axs[0].set_xlim([0, 6])

    # BH plot (right)
    for ii, region in enumerate(regions):
        ref_file = f"data/BH/intermediate_files/{region}_1000_nt_muts.json"
        reference_sequence = get_reference_sequence(ref_file)
        alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
        years, dist, std = get_mean_distance_in_time(alignment_file, reference_sequence, "")
        fit = np.polyfit(years, dist["all"], deg=1)
        axs[1].plot(years, dist["all"], '.', color=colors[ii], label=region)
        axs[1].plot(years, np.polyval(fit, years), "-", linewidth=1, color=colors[ii])
        axs[1].text(equation_pos[ii][0], equation_pos[ii][1],
                    f"{fit[0]:.1E}x + {round(fit[1], 1)}", fontsize=fontsize, color=colors[ii])

    axs[1].grid(grid_alpha)
    axs[1].set_xlabel("Sample date", fontsize=fontsize)
    axs[1].set_ylabel("Average fraction difference", fontsize=fontsize)
    axs[1].legend(fontsize=fontsize, loc="upper left")
    axs[1].tick_params(axis="x", labelsize=tick_size)
    axs[1].tick_params(axis="y", labelsize=tick_size)

    plt.tight_layout()
    plt.savefig("figures/Paper_fig_1.png", format="png")
    plt.show()
