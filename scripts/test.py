import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
from hivevo.patients import Patient


def get_sweeps(patient, aft, region):
    "Returns the position and nucleotide of sweeps hapening in the aft."
    # Masking low depth
    depth = tools.depth_mask(patient, region)
    aft.mask = np.logical_or(aft.mask, ~depth)

    # Set all initial idx to 0
    initial_idx = patient.get_initial_indices(region)
    aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])] = 0

    # Find the nucleotide and position of sweeps
    sweeps = aft > 0.5
    sweeps = np.sum(sweeps, axis=0, dtype=bool)
    sweeps_idxs = np.where(sweeps)
    p = sweeps_idxs[1].argsort()
    positions = sweeps_idxs[1][p]
    nucleotides = sweeps_idxs[0][p]

    return nucleotides, positions


    # Map that to the synonymous mask
    # syn_mask = patient.get_syn_mutations(region)
    # synonymous = syn_mask[nucleotides, positions]
    # breakpoint()


if __name__ == "__main__":
    region = "env"
    fontsize = 16
    patient = Patient.load("p1")
    aft = patient.get_allele_frequency_trajectories(region)
    nucleotides, positions = get_sweeps(patient, aft, region)
