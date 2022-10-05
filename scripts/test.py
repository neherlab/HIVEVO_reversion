"""
Script used to test the bootstrapping estimates for the rates of fig 1 BH
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import filenames
from Bio import Phylo
from distance_in_time import get_reference_sequence



def iter_tree(tree_json, node_names = [], cutoff_date=1980):
    """
    Returns the name of the first clades after the cutoff_date. It does it by iteratively searching the
    the children of the given tree_json, and stopping as soon as one children is after the cutoff_date.
    """
    if tree_json["node_attrs"]["num_date"]["value"] > cutoff_date:
        node_names += [get_tips(tree_json)]
    else:
        if "children" in tree_json.keys():
            for child in tree_json["children"]:
                iter_tree(child, node_names, cutoff_date)
    
    return node_names

def get_tips(tree):
    """
    Returns a list of names of the tips of the tree. Does this recursively.
    """
    tips = []
    if "children" in tree.keys(): # If this node has children
        for child in tree["children"]:
            tips += get_tips(child)
    else: # If it does not have children
        tips += [tree["name"]]

    return tips



if __name__ == "__main__":
    region = "pol"
    mutation_rate = 0.0010347399249381831 # from branch length file
    cutoff_date = 1980
    file_path = f"visualisation/{region}.json"
    tree_path = f"data/BH/intermediate_files/timetree_{region}.nwk"
    root_path = f"data/BH/intermediate_files/{region}_nt_muts.json"
    consensus_path = f"data/BH/alignments/to_HXB2/{region}_consensus.fasta"
    original_metadata_path = f"data/BH/raw/{region}_subsampled_metadata.tsv"

    with open(file_path, "r") as f:
        tree_json = json.load(f)
    tree_json = tree_json["tree"]

    tmp = iter_tree(tree_json)
    