import os
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from treetime import TreeTime
from treetime.utils import parse_dates

if __name__ == "__main__":
    region = "pol"
    tree_path = f"data/BH/intermediate_files/timetree_{region}_1000.nwk"
    metadata_path = f"data/BH/raw/{region}_1000_subsampled_metadata.tsv"
    alignment_path = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    dates = parse_dates(metadata_path)

    tt = TreeTime(gtr='Jukes-Cantor', tree=tree_path, precision=1, aln=alignment_path, verbose=2, dates=dates)

    result = tt.run(root='best', infer_gtr=True, relaxed_clock=False, max_iter=2,
                    branch_length_mode='input', n_iqd=3, resolve_polytomies=True, fixed_rate=10.8e-4,
                    Tc='skyline', time_marginal="assign", vary_rate=2e-4, use_covariation=True)

    tt.plot_root_to_tip(add_internal=True)
