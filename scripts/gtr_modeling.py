import numpy as np
import matplotlib.pyplot as plt
from treetime import TreeTime
from treetime.gtr_site_specific import GTR_site_specific
from treetime.seqgen import SeqGen
from treetime.utils import parse_dates
from Bio import AlignIO
from Bio.Seq import Seq
from Bio import Phylo
from distance_in_time import get_reference_sequence
import filenames
import tools


def homogeneous_p(consensus_seq, r_minus, r_plus):
    """
    Returns a p matrix of shape (4*len(consensus_seq)) where the equilibrium frequencies are set
    according to the global reversion and non-reversion mutation rate.
    """
    p = np.zeros((4, len(consensus_seq)))
    consensus_idxs = tools.sequence_to_indices(consensus_seq)
    for ii in range(len(consensus_seq)):
        tmp = r_minus / (3 * r_plus + r_minus)
        p[consensus_idxs[ii], ii] = tmp
        p[np.delete([0, 1, 2, 3], consensus_idxs[ii]), ii] = (1 - tmp) / 3.0
    return p


def SeqGen_homogeneous():
    """
    Generates an MSA based on a homogeneous (same for all site) model for the reversion and non-reversion rates.
    """
    tree_path = "data/BH/intermediate_files/tree_pol_1000.nwk"
    root_path = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    consensus_path = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"

    root_seq = get_reference_sequence(root_path)
    root_seq = Seq("".join(root_seq))
    tree = Phylo.read(tree_path, "newick")
    consensus_seq = get_reference_sequence(consensus_path)
    # Replaces N by A, it's not correct but I have a 4 letter aphabet for now
    consensus_seq[consensus_seq == "N"] = "A"

    non_reversion_rate = 0.0011872256804794856
    reversion_rate = 0.009270115831081912

    L = len(root_seq)  # sequence length
    N = 4  # alphabet size
    mu = np.ones(L)
    W = np.array([[0., 0.763, 2.902, 0.391],
                  [0.763, 0., 0.294, 3.551],
                  [2.902, 0.294, 0., 0.317],
                  [0.391, 3.551, 0.317, 0.]])
    p = homogeneous_p(consensus_seq, reversion_rate, non_reversion_rate)

    myGTR = GTR_site_specific.custom(mu, p, W, alphabet="nuc_nogap")
    MySeq = SeqGen(3012, gtr=myGTR, tree=tree)
    MySeq.evolve(root_seq=root_seq)
    with open("data/modeling/homogeneous/generated_MSA/homogeneous.fasta", "wt") as f:
        AlignIO.write(MySeq.get_aln(), f, "fasta")


def get_ATGC_content(alignment):
    """
    Returns a nb_sequence*4 vector that gives the proportion of ATGC in each sequence.
    """
    prop = np.zeros((alignment.shape[0], 4))
    for ii, nuc in enumerate(["A", "T", "G", "C"]):
        prop[:, ii] = np.sum(alignment == nuc, axis=1)
    for ii in range(prop.shape[0]):
        prop[ii] /= np.sum(prop[ii])
    return prop


def compare_ATGC_distributions(original_MSA, generated_MSA, nucleotides=["A", "T", "G", "C"],
                               colors=["C0", "C1", "C2", "C3"]):
    """
    Plots the distribution for the original and generated proportion of ATGC from the MSA.
    """
    ATGC_original = get_ATGC_content(original_MSA)
    ATGC_generated = get_ATGC_content(generated_MSA)

    plt.figure()
    for ii in range(4):
        hist_or, bins = np.histogram(ATGC_original[:, ii], bins=500, range=[0, 0.5])
        hist_gen, _ = np.histogram(ATGC_generated[:, ii], bins=500, range=[0, 0.5])
        bins = 0.5 * (bins[:-1] + bins[1:])

        plt.plot(bins, hist_or, '-', color=colors[ii], label=f"orginal {nucleotides[ii]}")
        plt.plot(bins, hist_gen, '--', color=colors[ii], label=f"generated {nucleotides[ii]}")
    plt.legend()
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Counts")
    plt.show()


def get_hamming_distance(MSA, reference_sequence):
    """
    Computes the hamming distance of each sequence in the MSA to the reference sequence provided.
    """
    distance = np.zeros(MSA.shape[0])
    for ii in range(MSA.shape[0]):
        distance[ii] = np.sum(MSA[ii] != reference_sequence, dtype=np.float32) / MSA.shape[1]
    return distance


def compare_hamming_distributions(original_MSA, generated_MSA, ref_seq, title=""):
    """
    Computes and plots the distribution of hamming distances to the reference sequence.
    """
    distance_original = get_hamming_distance(original_MSA, ref_seq)
    distance_generated = get_hamming_distance(generated_MSA, ref_seq)

    plt.figure()
    plt.title(title)
    hist_or, bins = np.histogram(distance_original, bins=500, range=[0, 0.5])
    hist_gen, _ = np.histogram(distance_generated, bins=500, range=[0, 0.5])
    bins = 0.5 * (bins[:-1] + bins[1:])

    plt.plot(bins, hist_or, '-', label=f"orginal")
    plt.plot(bins, hist_gen, '-', label=f"generated")
    plt.legend()
    plt.grid()
    plt.xlabel("Percentage difference")
    plt.ylabel("Counts")
    plt.show()


def get_RTT(tree):
    """
    Return the RTT distance and dates for each tip in the tree.
    """
    tips = tree.get_terminals()
    dates = []
    for tip in tips:
        date = tip.name.split(".")[2]
        dates += [int(date)]

    rtt = [tree.distance(tip) for tip in tips]

    return np.array(rtt), np.array(dates)


if __name__ == "__main__":
    original_MSA_path = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    generated_MSA_path = "data/modeling/generated_MSA/homogeneous.fasta"
    original_tree_path = "data/BH/intermediate_files/tree_pol_1000.nwk"
    generated_tree_path = "data/modeling/generated_trees/homogeneous.nwk"
    root_path = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    consensus_path = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"
    original_metadata_path = "data/BH/raw/pol_1000_subsampled_metadata.tsv"

    original_MSA = AlignIO.read(original_MSA_path, "fasta")
    original_MSA = np.array(original_MSA_path)
    generated_MSA = AlignIO.read(generated_MSA_path, "fasta")
    generated_MSA = np.array(generated_MSA_path)

    # compare_ATGC_distributions(original_MSA, generated_MSA)
    #
    # ref_seq = get_reference_sequence(root_path)
    # compare_hamming_distributions(original_MSA, generated_MSA, ref_seq, "to root")
    # ref_seq = get_reference_sequence(consensus_path)
    # compare_hamming_distributions(original_MSA, generated_MSA, ref_seq, "to consensus")

    # tree_or = Phylo.read(original_tree_path, "newick")
    # tree_gen = Phylo.read(generated_tree_path, "newick")

    dates = parse_dates(original_metadata_path)
    ttree = TreeTime(gtr='Jukes-Cantor', tree=original_tree_path,
                     precision=1, aln=original_MSA_path, verbose=2, dates=dates)
    ttree.reroot()
    tree_or = ttree._tree
    ttree = TreeTime(gtr='Jukes-Cantor', tree=generated_tree_path,
                     precision=1, aln=generated_MSA_path, verbose=2, dates=dates)
    ttree.reroot()
    tree_gen = ttree._tree

    rtt_or, dates_or = get_RTT(tree_or)
    rtt_gen, dates_gen = get_RTT(tree_gen)

    plt.figure()
    plt.plot(dates_or, rtt_or, '.', label="original")
    # plt.plot(dates_gen, rtt_gen, '.', label="generated")
    plt.legend()
    plt.grid()
    plt.xlabel("Years")
    plt.ylabel("RTT")
