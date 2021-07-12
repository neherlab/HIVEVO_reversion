from distance_in_time import get_reference_sequence, get_mean_distance_in_time
import divergence
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("tex")


if __name__ == '__main__':
    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    lines = [":", "--", "-"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    fill_alpha = 0.15
    figsize = (6.7315, 3)

    regions = ["env", "gag", "pol"]
    time = div_dict["time"]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    # BH plot pol
    ref_files = {"root": "data/BH/intermediate_files/pol_1000_nt_muts.json",
                 "B": "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta",
                 "C": "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta"}

    alignment_file = f"data/BH/alignments/to_HXB2/pol_1000.fasta"
    for ii, key in enumerate(ref_files.keys()):
        reference_sequence = get_reference_sequence(ref_files[key])
        years, dist, std = get_mean_distance_in_time(
            alignment_file, reference_sequence, key if key in ["B", "C"] else "")

        fit = np.polyfit(years, dist["all"], deg=1)
        axs[0].plot(years, dist["all"], '.', color=colors[ii], label=key)
        axs[0].plot(years, np.polyval(fit, years), "-", linewidth=1, color=colors[ii])

    axs[0].set_xlabel("Sample date")
    axs[0].set_ylabel("Fraction difference")
    axs[0].legend()
    axs[0].ticklabel_format(axis="x", style="plain")

    # WH plot references
    for ii, key in enumerate(div_dict["pol"].keys()):
        data = div_dict["pol"][key]["global"]["all"]["all"]["mean"]
        std = div_dict["pol"][key]["global"]["all"]["all"]["std"]
        axs[1].plot(time, data, "-", color=colors[ii + 4], label=key)
        axs[1].fill_between(time, data + std, data - std, color=colors[ii + 4], alpha=fill_alpha)

    axs[1].set_xlabel("Time [years]")
    axs[1].set_ylabel("Divergence")
    axs[1].legend()

    fig.savefig("figures/Paper_fig_1.pdf")
    plt.show()
