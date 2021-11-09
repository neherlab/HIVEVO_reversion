from treetime.gtr_site_specific import GTR_site_specific
from treetime.seqgen import SeqGen
from treetime.treeanc import TreeAnc
from Bio import AlignIO
from Bio.Seq import Seq
from Bio import Phylo
from distance_in_time import get_reference_sequence

if __name__ == "__main__":
    alpha = 1.0
    rate_alpha = 1.5
    W_dirichlet_alpha = 2.0
    myGTR = GTR_site_specific.random(3012, alphabet="nuc_nogap", pi_dirichlet_alpha=alpha,
                                     mu_gamma_alpha=rate_alpha, W_dirichlet_alpha=W_dirichlet_alpha)

    tree_path = "data/BH/intermediate_files/tree_pol_1000.nwk"
    root_path = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    root_seq = get_reference_sequence(root_path)
    root_seq = Seq("".join(root_seq))
    tree = Phylo.read(tree_path, "newick")

    MySeq = SeqGen(3012, gtr=myGTR, tree=tree)
    MySeq.evolve(root_seq=root_seq)
    with open("test.fasta", "wt") as f:
        AlignIO.write(MySeq.get_aln(), f, "fasta")
