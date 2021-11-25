import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
import json
from Bio import Phylo
from predictions import predict_average_error, compute_diversity_divergence_fit, get_diversity_divergence
import divergence
from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient
import tools
from treetime.gtr_site_specific import GTR_site_specific
from treetime.seqgen import SeqGen
from treetime.treeanc import TreeAnc
from Bio import AlignIO
from Bio.Seq import Seq


def non_consensus_percentages():
    # Loading data
    MSA_path = "data/BH/alignments/to_HXB2/pol_1000.fasta"
    consensus_path = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta"
    msa = AlignIO.read(MSA_path, "fasta")
    msa = np.array(msa)
    consensus = AlignIO.read(consensus_path, "fasta")
    consensus = np.array(consensus).squeeze()

    # Creating frequency array
    af = np.zeros((5, consensus.shape[0]), dtype=int)
    for ii, nuc in enumerate(["A", "T", "G", "C", "N"]):
        af[ii, :] = np.sum(msa == nuc, axis=0)
    af = af / msa.shape[0]

    means = {}
    for ii, nuc in enumerate(["A", "T", "G", "C", "N"]):
        mask = consensus == nuc
        tmp = af[:, mask]
        means[nuc] = np.mean(tmp, axis=1)
    print(means)


def non_consensus_to_non_consensus():
    "Quick and dirty analysis to get an idea of how much reversion VS non-consensus to non-consensus happen"
    region = "pol"
    ref = HIVreference(subtype="any")

    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    full_change = []
    non_reversion_change = []
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        non_consensus_mask = tools.non_reference_mask(patient, region, aft, ref)
        reversion_map = tools.reversion_map(patient, region, aft, ref)
        initial_idx = patient.get_initial_indices(region)

        # Set initial nucleotide to 0 frequency
        aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])] = 0
        tmp = aft[:, :, non_consensus_mask]
        tmp = np.sum(tmp, axis=1)
        full_change += [np.sum(tmp[-1, :])]

        aft[:, reversion_map] = 0
        tmp = aft[:, :, non_consensus_mask]
        tmp = np.sum(tmp, axis=1)
        non_reversion_change += [np.sum(tmp[-1, :])]

    print(full_change)
    print(non_reversion_change)


if __name__ == "__main__":
    non_consensus_percentages()
    # non_consensus_to_non_consensus()
