"""
Histogram of diversity for all sites and the 3 regions. Must be run from the script folder
"""
import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
from hivevo.patients import Patient
from Bio import AlignIO


def get_diversity_histo(diversity, nb_bins=50):
    "Compute and returns the histogram distribution of diversity for each site of the alignment."
    counts, bins = np.histogram(diversity, bins=nb_bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    frequencies = counts / np.sum(counts)
    return bins, frequencies


if __name__ == "__main__":
    plt.figure()
    for region in ["env", "pol", "gag"]:
        alignment_file = f"data/BH/alignments/to_HXB2/{region}.fasta"
        diversity = tools.get_diversity(alignment_file)
        bins, frequencies = get_diversity_histo(diversity)
        mask = tools.mask_diversity_percentile(diversity, 90, 100)
        tmp = diversity[mask]
        plt.plot(bins, frequencies, label=region)

    plt.yscale("log")
    plt.xlabel("Diversity")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
    plt.show()
