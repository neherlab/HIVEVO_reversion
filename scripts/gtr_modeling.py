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
import os
import divergence


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


def transition_idx(idx):
    "Returns the indice corresponding to the transition mutation from the given idx."
    transi_list = [2, 3, 0, 1, 5, 6]
    return transi_list[idx]


def binary_p(consensus_seq, r_minus, r_plus):
    """
    Returns a p matrix of shape (4*len(consensus_seq)) where the equilibrium frequencies are set
    according to the global reversion and non-reversion mutation rate.
    """
    cons_fraction = r_minus / (r_plus + r_minus)
    p = np.ones((4, len(consensus_seq))) * (1 - cons_fraction) * 0.1
    consensus_idxs = tools.sequence_to_indices(consensus_seq)
    for ii in range(len(consensus_seq)):
        p[consensus_idxs[ii], ii] = cons_fraction
        p[transition_idx(consensus_idxs[ii]), ii] = (1 - cons_fraction) * 0.8
    return p


def site_mask(consensus_seq, position):
    """
    Returns a bolean of shape len(consensus_seq) where True are the sites corresponding to the given
    position (first, second or third)
    """
    assert position in [1, 2, 3], "Position must be 1 2 or 3."
    position_mask = np.zeros(consensus_seq.shape[-1], dtype=bool)
    position_mask[position - 1::3] = True
    return position_mask


def consensus_mask(consensus_seq):
    """
    Returns a bolean of shape 4*len(consensus_seq) where True are the positions that correspond to the
    consensus sequence.
    """
    mask = np.zeros((4, len(consensus_seq)), dtype=bool)
    consensus_idxs = tools.sequence_to_indices(consensus_seq)

    for ii in range(len(consensus_seq)):
        mask[consensus_idxs[ii], ii] = True
    return mask


def transition_mask(consensus_seq):
    """
    Returns a bolean of shape 4*len(consensus_seq) where True are the positions that correspond to the
    transition nucleotide from the consensus.
    """
    mask = np.zeros((4, len(consensus_seq)), dtype=bool)
    consensus_idxs = tools.sequence_to_indices(consensus_seq)

    for ii in range(len(consensus_seq)):
        mask[transition_idx(consensus_idxs[ii]), ii] = True
    return mask


def p_3class_homogeneous(consensus_seq, rates):
    """
    Returns the p matrix for a 3 class homogeneous model.
    """
    def p_consensus(rate_consensus, rate_non_consensus):
        return rate_non_consensus / (3 * rate_consensus + rate_non_consensus)

    def p_non_consensus(rate_consensus, rate_non_consensus):
        return (1 - p_consensus(rate_consensus, rate_non_consensus)) / 3

    mask_first = site_mask(consensus_seq, 1)
    mask_second = site_mask(consensus_seq, 2)
    mask_third = site_mask(consensus_seq, 3)
    mask_consensus = consensus_mask(consensus_seq)

    p = np.zeros((4, len(consensus_seq)))
    p[:, mask_first] = p_non_consensus(rates["consensus"]["first"]["rate"],
                                       rates["non_consensus"]["first"]["rate"])
    p[:, mask_second] = p_non_consensus(rates["consensus"]["second"]["rate"],
                                        rates["non_consensus"]["second"]["rate"])
    p[:, mask_third] = p_non_consensus(rates["consensus"]["third"]["rate"],
                                       rates["non_consensus"]["third"]["rate"])
    p[np.logical_and(mask_first, mask_consensus)] = p_consensus(rates["consensus"]["first"]["rate"],
                                                                rates["non_consensus"]["first"]["rate"])
    p[np.logical_and(mask_second, mask_consensus)] = p_consensus(rates["consensus"]["second"]["rate"],
                                                                 rates["non_consensus"]["second"]["rate"])
    p[np.logical_and(mask_third, mask_consensus)] = p_consensus(rates["consensus"]["third"]["rate"],
                                                                rates["non_consensus"]["third"]["rate"])
    return p


def p_3class_binary(consensus_seq, rates):
    """
    Returns the p matrix for a 3 class homogeneous model.
    """
    def p_consensus(rate_consensus, rate_non_consensus):
        return rate_non_consensus / (rate_consensus + rate_non_consensus)

    def p_non_consensus(rate_consensus, rate_non_consensus):
        return (1 - p_consensus(rate_consensus, rate_non_consensus))

    mask_first = site_mask(consensus_seq, 1)
    mask_second = site_mask(consensus_seq, 2)
    mask_third = site_mask(consensus_seq, 3)
    mask_consensus = consensus_mask(consensus_seq)
    mask_transition = transition_mask(consensus_seq)

    # Non consensus transversion
    p = np.zeros((4, len(consensus_seq)))
    p[:, mask_first] = p_non_consensus(rates["consensus"]["first"]["rate"],
                                       rates["non_consensus"]["first"]["rate"]) * 0.1
    p[:, mask_second] = p_non_consensus(rates["consensus"]["second"]["rate"],
                                        rates["non_consensus"]["second"]["rate"]) * 0.1
    p[:, mask_third] = p_non_consensus(rates["consensus"]["third"]["rate"],
                                       rates["non_consensus"]["third"]["rate"]) * 0.1
    # Non consensus transitions
    p[np.logical_and(mask_first, mask_transition)] = p_non_consensus(rates["consensus"]["first"]["rate"],
                                                                     rates["non_consensus"]["first"]["rate"]) * 0.8
    p[np.logical_and(mask_second, mask_transition)] = p_non_consensus(rates["consensus"]["second"]["rate"],
                                                                      rates["non_consensus"]["second"]["rate"]) * 0.8
    p[np.logical_and(mask_third, mask_transition)] = p_non_consensus(rates["consensus"]["third"]["rate"],
                                                                     rates["non_consensus"]["third"]["rate"]) * 0.8
    # Consensus
    p[np.logical_and(mask_first, mask_consensus)] = p_consensus(rates["consensus"]["first"]["rate"],
                                                                rates["non_consensus"]["first"]["rate"])
    p[np.logical_and(mask_second, mask_consensus)] = p_consensus(rates["consensus"]["second"]["rate"],
                                                                 rates["non_consensus"]["second"]["rate"])
    p[np.logical_and(mask_third, mask_consensus)] = p_consensus(rates["consensus"]["third"]["rate"],
                                                                rates["non_consensus"]["third"]["rate"])
    return p


def generate_MSA(tree_path, root_path, consensus_path, MSA, metadata, save_path, rates,
                 p_type="homogeneous", scaling=1.3):
    """
    Generates an MSA based on a homogeneous (same for all site) model for the reversion and non-reversion
    rates.
    """
    assert p_type in ["homogeneous", "binary", "3class_homogeneous",
                      "3class_binary", "control"], f"p_type must be 'homogeneous' 'binary' '3class_homogeneous' 'control' or '3class_binary', got {p_type}"

    root_seq = get_reference_sequence(root_path)
    root_seq = Seq("".join(root_seq))

    tree = Phylo.read(tree_path, "newick")
    dates = parse_dates(metadata)
    ttree = TreeTime(gtr='Jukes-Cantor', tree=tree, precision=1, aln=MSA, verbose=2, dates=dates)
    ttree.reroot()
    tree = ttree._tree

    consensus_seq = get_reference_sequence(consensus_path)
    # Replaces N by A, it's not correct but I have a 4 letter aphabet for now
    consensus_seq[consensus_seq == "N"] = "A"

    non_reversion_rate = rates["consensus"]["all"]["rate"]
    reversion_rate = rates["non_consensus"]["all"]["rate"]

    L = len(root_seq)  # sequence length
    mu = np.ones(L)
    W = np.array([[0., 0.763, 2.902, 0.391],
                  [0.763, 0., 0.294, 3.551],
                  [2.902, 0.294, 0., 0.317],
                  [0.391, 3.551, 0.317, 0.]])

    if p_type == "homogeneous":
        p = homogeneous_p(consensus_seq, reversion_rate, non_reversion_rate)
    elif p_type == "binary":
        p = binary_p(consensus_seq, reversion_rate, non_reversion_rate)
    elif p_type == "3class_homogeneous":
        p = p_3class_homogeneous(consensus_seq, rates)
    elif p_type == "3class_binary":
        p = p_3class_binary(consensus_seq, rates)
    elif p_type == "control":
        p = np.ones((4, len(consensus_seq))) * 0.25

    myGTR = GTR_site_specific.custom(mu, p, W, alphabet="nuc_nogap")
    myGTR.mu /= myGTR.average_rate().mean()
    myGTR.mu *= scaling
    MySeq = SeqGen(len(consensus_seq), gtr=myGTR, tree=tree)
    MySeq.evolve(root_seq=root_seq)
    with open(save_path, "wt") as f:
        AlignIO.write(MySeq.get_aln(), f, "fasta")


def generate_tree(MSA_path, output_path, builder_args="-m GTR+F+R10 -czb"):
    """
    Generates a tree from the given MSA using augur tree command.
    """
    cmd_str = f"""augur tree --method iqtree --tree-builder-args='{builder_args}' --alignment {MSA_path} --output {output_path} --nthreads 4"""
    print("Executing command: " + cmd_str)
    os.system(cmd_str)


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


def compare_RTT(tree_or, MSA_or, tree_gen, MSA_gen, metadata):
    """
    Returns a p matrix of shape (4*len(consensus_seq)) where the equilibrium frequencies are set
    according to the global reversion and non-reversion mutation rate.
    """
    dates = parse_dates(metadata)
    ttree = TreeTime(gtr='Jukes-Cantor', tree=tree_or,
                     precision=1, aln=MSA_or, verbose=2, dates=dates)
    ttree.reroot()
    tree_or = ttree._tree
    ttree = TreeTime(gtr='Jukes-Cantor', tree=tree_gen,
                     precision=1, aln=MSA_gen, verbose=2, dates=dates)
    ttree.reroot()
    tree_gen = ttree._tree

    rtt_or, dates_or = get_RTT(tree_or)
    rtt_gen, dates_gen = get_RTT(tree_gen)
    fit_or = np.polyfit(dates_or, rtt_or, deg=1)
    fit_gen = np.polyfit(dates_gen, rtt_gen, deg=1)

    plt.figure()
    plt.plot(dates_or, rtt_or, '.', label="original", color="C0")
    plt.plot(dates_or, np.polyval(fit_or, dates_or), "-", color="C0",
             label=f"Clock: ${round(fit_or[0]*1e4,1)}\\cdot 10^{{-4}}$  Root: {round(-fit_or[1]/fit_or[0])}")
    plt.plot(dates_gen, rtt_gen, '.', label="generated", color="C1")
    plt.plot(dates_gen, np.polyval(fit_gen, dates_gen), "-", color="C1",
             label=f"Clock: ${round(fit_gen[0]*1e4,1)}\\cdot 10^{{-4}}$  Root: {round(-fit_gen[1]/fit_gen[0])}")
    plt.legend()
    plt.grid()
    plt.xlabel("Years")
    plt.ylabel("RTT")


if __name__ == "__main__":
    region = "gag"
    original_MSA_path = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    original_tree_path = f"data/BH/intermediate_files/tree_{region}_1000.nwk"
    root_path = f"data/BH/intermediate_files/{region}_1000_nt_muts.json"
    consensus_path = f"data/BH/alignments/to_HXB2/{region}_1000_consensus.fasta"
    original_metadata_path = f"data/BH/raw/{region}_1000_subsampled_metadata.tsv"
    generated_MSA_folder = f"data/modeling/generated_MSA/{region}_"
    generated_tree_folder = f"data/modeling/generated_trees/{region}_"
    rate_dict_path = "data/WH/avg_rate_dict.json"
    rates = divergence.load_avg_rate_dict(rate_dict_path)
    rates = rates[region]["founder"]["global"]

    # p_type = "homogeneous"
    # p_type = "binary"
    # p_type = "3class_homogeneous"
    # p_type = "3class_binary"
    p_type = "control"
    regenerate = True
    analysis = True

    generated_MSA_path = generated_MSA_folder + p_type + ".fasta"
    generated_tree_path = generated_tree_folder + p_type + ".nwk"

    if regenerate:
        if p_type == "control":
            generate_MSA(original_tree_path, root_path, consensus_path,
                         original_MSA_path, original_metadata_path, generated_MSA_path, rates, p_type, scaling=1)
        else:
            generate_MSA(original_tree_path, root_path, consensus_path,
                         original_MSA_path, original_metadata_path, generated_MSA_path, rates, p_type, scaling=1.3)

        generate_tree(generated_MSA_path, generated_tree_path)

    # --- Data analysis ---
    if analysis:
        original_MSA = AlignIO.read(original_MSA_path, "fasta")
        original_MSA = np.array(original_MSA)
        generated_MSA = AlignIO.read(generated_MSA_path, "fasta")
        generated_MSA = np.array(generated_MSA)

        compare_ATGC_distributions(original_MSA, generated_MSA)

        ref_seq = get_reference_sequence(root_path)
        compare_hamming_distributions(original_MSA, generated_MSA, ref_seq, "to root")
        ref_seq = get_reference_sequence(consensus_path)
        compare_hamming_distributions(original_MSA, generated_MSA, ref_seq, "to consensus")

        compare_RTT(original_tree_path, original_MSA_path,
                    generated_tree_path, generated_MSA_path, original_metadata_path)

        plt.show()
