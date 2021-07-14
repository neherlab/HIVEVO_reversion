from distance_in_time import get_reference_sequence, get_mean_distance_in_time, get_root_to_tip_distance
import divergence
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("tex")


def make_figure(region, text_pos, ylim, sharey, savefig=False):
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
    lengths = np.array(lengths)[dates >= 1977]
    dates = dates[dates >= 1977]
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

        fit = np.polyfit(years, dist["all"], deg=1)
        axs[0].plot(years, dist["all"], '.', color=colors[ii], label=key)
        axs[0].plot(years, np.polyval(fit, years), "-", linewidth=1, color=colors[ii])
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
    idxs = time < 5.3
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
    if sharey:
        axs[1].set_ylabel("Divergence")
    axs[1].legend()
    axs[1].set_xlim([-0.3, 5.5])

    if savefig:
        fig.savefig(f"figures/Distance_{region}.pdf")


if __name__ == '__main__':
    text = {"env": [(2000, 0.36), (2000, 0.16), (2000, 0.03), (1.2, 0.079), (1.2, 0.045), (1.2, 0.024)],
            "pol": [(2000, 0.107), (2000, 0.0585), (2000, 0.024), (1.2, 0.072), (1.2, 0.042), (1.2, 0.01)],
            "gag": [(2000, 0.109), (2000, 0.068), (2000, 0.03), (1.2, 0.077), (1.2, 0.047), (1.2, 0.0165)]}

    ylim = {"env": [0, 0.45], "pol": [0, 0.12], "gag": [0, 0.145]}
    sharey = {"env": False, "pol": True, "gag": True}

    for region in ["env", "pol", "gag"]:
        make_figure(region, text[region], ylim[region], sharey[region])
    plt.show()
