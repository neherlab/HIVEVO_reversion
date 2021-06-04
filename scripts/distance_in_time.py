import json
import matplotlib.pyplot as plt
import numpy as np
from Bio import AlignIO


def get_reference_sequence(filename):
    """
    Loads the reference sequence from the given file and returns it. Can either be the consensus sequence
    (from fasta file), or the root sequence (from the nt_muts.json file from TreeTime).
    """
    # Root sequence from tree file
    if ".json" in filename:
        with open(filename) as f:
            data = json.load(f)
            reference_sequence = list(data["nodes"]["NODE_0000000"]["sequence"])
        reference_sequence = np.array(reference_sequence)
    # The consensus sequence from fasta file
    elif ".fasta" in filename:
        consensus = AlignIO.read(filename, "fasta")
        reference_sequence = np.array(consensus)[0]
    return reference_sequence


if __name__ == '__main__':
    alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    consensus_file = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"
    # consensus_file = "intermediate_files/pol_1000_nt_muts.json"

    plt.figure()
    subtypes = ["B", "C"]
    colors = ["C0", "C1"]
    c = 0
    for subtype in subtypes:
        consensus_sequence = get_consensus_sequence(consensus_file)
        years, dist, std = get_mean_distance_in_time(alignment_file, consensus_sequence, subtype)
        fit = np.polyfit(years[std != 0], dist[std != 0], deg=1, w=(1 / std[std != 0]))
        plt.errorbar(years, dist, yerr=std, fmt=".", label=subtype, color=colors[c])
        plt.plot(years, np.polyval(fit, years), "--",
                 color=colors[c], label=f"{round(fit[0],5)}x + {round(fit[1],5)}")
        c += 1

    plt.grid()
    plt.xlabel("Time [years]")
    plt.ylabel("Average fraction difference")
    plt.legend()
    plt.show()
