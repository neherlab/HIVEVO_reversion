import os
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from treetime import TreeTime
from treetime.utils import parse_dates
import json

if __name__ == "__main__":
    region = "pol"
    tree_path = f"data/BH/intermediate_files/tree_{region}_1000.nwk"
    metadata_path = f"data/BH/raw/{region}_1000_subsampled_metadata.tsv"
    alignment_path = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    dates = parse_dates(metadata_path)

    rates = np.linspace(5e-4, 15e-4, 10)
    likelihood = []
    for rate in rates:
        tt = TreeTime(gtr='Jukes-Cantor', tree=tree_path, precision=1,
                      aln=alignment_path, verbose=2, dates=dates)
        result = tt.run(root='best', infer_gtr=True, relaxed_clock=False, max_iter=2,
                        branch_length_mode='input', n_iqd=3, resolve_polytomies=True, fixed_clock_rate=rate,
                        Tc='skyline', time_marginal="assign")
        likelihood += [tt.timetree_likelihood()]
    tmp = {"rates": rates, "likelihood": likelihood}

    with open("likelihood.json", "w") as output:
        json.dump(tmp, output, indent=4)
