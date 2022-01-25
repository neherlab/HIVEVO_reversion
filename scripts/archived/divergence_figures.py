# Detailed analysis of the divergence in time for WH data. Must be run from the script/ folder

import matplotlib.pyplot as plt
import divergence
plt.style.use("tex.mplstyle")

if __name__ == '__main__':
    div_dict = divergence.load_div_dict("data/WH/bootstrap_div_dict.json")
    lines = ["-", "--", ":", "-."]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    fontsize = 16

    regions = ["env", "pol", "gag"]
    time = div_dict["time"]

    # General picture
    plt.figure()
    for ii, region in enumerate(regions):
        for jj, key2 in enumerate(div_dict[region].keys()):
            data = div_dict[region][key2]["global"]["all"]["all"]["mean"]
            data = data - data[0]
            plt.plot(time, data, lines[jj],
                     label=f"{region} {key2}", color=colors[ii])
    plt.grid()
    plt.xlabel("Time [years]")
    plt.ylabel("Divergence")
    plt.legend()
    plt.xlim([0, 6])
    plt.savefig("figures/Divergence_region.png", format="png")

    # pol relative to global consensus
    plt.figure()
    plt.title("Pol global consensus")
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        for jj, key2 in enumerate(div_dict["pol"].keys()):
            data = div_dict["pol"][key2]["global"][key]["all"]["mean"]
            data = data - data[0]
            plt.plot(time, data, lines[jj],
                     label=f"pol {key2} {key}", color=colors[ii])
    plt.grid()
    plt.xlabel("Time [years]")
    plt.ylabel("Divergence")
    plt.legend()
    plt.xlim([0, 6])
    plt.savefig("figures/Divergence_pol_global.png", format="png")

    # pol relative to subtype consensus
    plt.figure()
    plt.title("Pol subtype consensus")
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        for jj, key2 in enumerate(div_dict["pol"].keys()):
            data = div_dict["pol"][key2]["subtype"][key]["all"]["mean"]
            data = data - data[0]
            plt.plot(time, data, lines[jj],
                     label=f"pol {key2} {key}", color=colors[ii])
    plt.grid()
    plt.xlabel("Time [years]")
    plt.ylabel("Divergence")
    plt.legend()
    plt.xlim([0, 6])
    plt.savefig("figures/Divergence_pol_subtype.png", format="png")

    # By Positions
    plt.figure()
    plt.title("Pol positions")
    for ii, key in enumerate(["all", "consensus", "non_consensus"]):
        for jj, key2 in enumerate(["first", "second", "third"]):
            data = div_dict["pol"]["founder"]["global"][key][key2]["mean"]
            data = data - data[0]
            plt.plot(time, data, lines[jj],
                     label=f"pol {key2} {key}", color=colors[ii])
    plt.grid()
    plt.xlabel("Time [years]")
    plt.ylabel("Divergence")
    plt.legend()
    plt.xlim([0, 6])
    plt.savefig("figures/Divergence_pol_positions.png", format="png")

    plt.show()
