import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
from hivevo.patients import Patient
from Bio import AlignIO


def get_diversity_histo(alignment_file, nb_bins=50):
    "Compute and returns the histogram distribution of diversity for each site of the alignment."
    alignment = AlignIO.read(alignment_file, "fasta")
    alignment = np.array(alignment)

    probabilities = []
    for nuc in ["A", "T", "G", "C", "-"]:
        probabilities += [(alignment == nuc).sum(axis=0) / alignment.shape[0]]
    probabilities = np.array(probabilities)

    eps = 1e-8
    entropy = np.sum(probabilities * np.log(probabilities + eps), axis=0) / np.log(1 / 5)
    entropy[entropy < eps] = 0
    counts, bins = np.histogram(entropy, bins=nb_bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    frequencies = counts / np.sum(counts)
    return bins, frequencies


if __name__ == "__main__":
    plt.figure()
    for region in ["env", "pol", "gag"]:
        alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
        bins, frequencies = get_diversity_histo(alignment_file)
        plt.plot(bins, frequencies, label=region)

    plt.yscale("log")
    plt.xlabel("Diversity")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
    plt.show()
