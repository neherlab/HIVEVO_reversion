import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
from hivevo.patients import Patient
from Bio import AlignIO


def get_diversity(alignment_file):
    "Compute the diversity at each site of the alignment"
    alignment = AlignIO.read(alignment_file, "fasta")
    alignment = np.array(alignment)

    probabilities = []
    for nuc in ["A", "T", "G", "C", "-"]:
        probabilities += [(alignment == nuc).sum(axis=0) / alignment.shape[0]]
    probabilities = np.array(probabilities)

    eps = 1e-8
    entropy = np.sum(probabilities * np.log(probabilities + eps), axis=0) / np.log(1 / 5)
    entropy[entropy < eps] = 0
    return entropy


def get_diversity_histo(diversity, nb_bins=50):
    "Compute and returns the histogram distribution of diversity for each site of the alignment."
    counts, bins = np.histogram(diversity, bins=nb_bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    frequencies = counts / np.sum(counts)
    return bins, frequencies


def mask_entropy_percentile(diversity, p_low, p_high):
    """
    Returns a 1D boolean vector where True are the position corresponding to diversity in [p_low, p_high].
    Sites close to 0 are the lowest diversity, while sites close to 1 are the highest.
    """
    from scipy.stats import scoreatpercentile
    thresholds = [scoreatpercentile(diversity, p) for p in [p_low, p_high]]
    return np.logical_and(diversity > thresholds[0], diversity <= thresholds[1])


if __name__ == "__main__":
    plt.figure()
    for region in ["env", "pol", "gag"]:
        alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
        diversity = get_diversity(alignment_file)
        bins, frequencies = get_diversity_histo(diversity)
        mask = mask_entropy_percentile(diversity, 90, 100)
        tmp = diversity[mask]
        breakpoint()
        plt.plot(bins, frequencies, label=region)

    plt.yscale("log")
    plt.xlabel("Diversity")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
    plt.show()
