import matplotlib.pyplot as plt
import divergence


def divergence_region_plot(divergence_dict, figsize=(14, 10), fontsize=20, tick_fontsize=18,
                           colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                           fill_alpha=0.15):
    time_average = np.arange(0, 2001, 40) / 365
    gene_annotation = [" (surface proteins)", "  (viral enzymes)", " (capsid proteins)"]

    plt.figure(figsize=(14, 10))
    for ii, region in enumerate(["env", "pol", "gag"]):
        mean = divergence_dict[region]["all"]["all"]["mean"]
        std = divergence_dict[region]["all"]["all"]["std"]
        time = divergence_dict[region]["all"]["all"]["time"]
        plt.plot(time_average, mean, '-', color=colors[ii], label=region + gene_annotation[ii])
        plt.fill_between(time_average, mean + std, mean - std, color=colors[ii], alpha=fill_alpha)
    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Mean divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_region.png", format="png")
    plt.show()


def divergence_consensus_plot(divergence_dict, figsize=(14, 10), fontsize=20, tick_fontsize=18,
                              fill_alpha=0.15):
    region = "pol"
    time_average = np.arange(0, 2001, 40) / 365

    plt.figure(figsize=(14, 10))

    mean = divergence_dict[region]["all"]["all"]["mean"]
    std = divergence_dict[region]["all"]["all"]["std"]
    plt.plot(time_average, mean, color="C1", label="pol all")
    plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color="C1")

    mean = divergence_dict[region]["consensus"]["all"]["mean"]
    std = divergence_dict[region]["consensus"]["all"]["std"]
    plt.plot(time_average, mean, '-', color="tab:red", label="pol consensus")
    plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color="tab:red")

    mean = divergence_dict[region]["non_consensus"]["all"]["mean"]
    std = divergence_dict[region]["non_consensus"]["all"]["std"]
    plt.plot(time_average, mean, '--', color="tab:green", label="pol non-consensus")
    plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color="tab:green")

    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Mean divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_consensus.png", format="png")
    plt.show()


def divergence_site_plot(divergence_dict, figsize=(14, 10), fontsize=20, tick_fontsize=18,
                         colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                         fill_alpha=0.15):
    time_average = np.arange(0, 2001, 40) / 365

    plt.figure(figsize=figsize)
    region = "pol"
    for ii, site in enumerate(["first", "second", "third"]):
        mean = divergence_dict[region]["consensus"][site]["mean"]
        std = divergence_dict[region]["consensus"][site]["std"]
        plt.plot(time_average, mean, '-', color=colors[ii])
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

        mean = divergence_dict[region]["non_consensus"][site]["mean"]
        std = divergence_dict[region]["non_consensus"][site]["std"]
        plt.plot(time_average, mean, '--', color=colors[ii])
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

    # plt.plot(time_average, divergence_dict[region]["non_consensus"]["all"]["mean"], 'k--')
    # plt.plot(time_average, divergence_dict[region]["consensus"]["all"]["mean"], 'k-')

    plt.plot([0], [0], "k-", label="consensus")
    plt.plot([0], [0], "k--", label="non-consensus")
    plt.plot([0], [0], "-", label="1st  codon position", color=colors[0])
    plt.plot([0], [0], "-", label="2nd codon position", color=colors[1])
    plt.plot([0], [0], "-", label="3rd  codon position", color=colors[2])
    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Mean divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_sites.png", format="png")
    plt.show()


if __name__ == '__main__':
    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    lines = ["-", "--", ":"]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fontsize = 16

    plt.figure(figsize=(14, 10))
    for ii, key in enumerate(div_dict.keys()):
        for jj, key2 in enumerate(div_dict[key].keys()):
            plt.plot(div_dict[key][key2]["time"]/365, div_dict[key][key2]["mean"], lines[jj],
                     label=f"{key} {key2}", color=colors[ii])
    plt.grid()
    plt.xlabel("Time [years]", fontsize=fontsize)
    plt.ylabel("Divergence", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlim([0, 2000/365])
    plt.plot()
