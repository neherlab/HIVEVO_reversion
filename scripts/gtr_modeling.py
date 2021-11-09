import numpy as np
from treetime.gtr_site_specific import GTR_site_specific
from treetime.seqgen import SeqGen
from treetime.treeanc import TreeAnc
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


if __name__ == "__main__":
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
