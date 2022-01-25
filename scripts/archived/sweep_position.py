"""
Script to plot the position of sweeps on the genome (from the BH data)
"""
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


def get_all_patient_sweeps(region):
    """
    Returns the sweeps positions for all patients, by mapping to the reference (HXB2).
    Also returns a mask for wether they are synonymous or non-synonymous mutations.
    """
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

    all_sweeps = []
    all_synonymous = []
    region_length = 0
    for patient_name in patient_names:
        # Getting info from the patient
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        nucleotides, positions = get_sweeps(patient, aft, region)
        syn_mask = patient.get_syn_mutations(region)
        synonymous = syn_mask[nucleotides, positions]

        # Get the position relative to reference
        map_to_ref = patient.map_to_external_reference(region)
        idxs = np.searchsorted(map_to_ref[:, 2], positions)
        ref_positions = map_to_ref[idxs, 0] - map_to_ref[0, 0]
        region_length = max(max(map_to_ref[-1, 0] - map_to_ref[0, 0], map_to_ref.shape[0]), region_length)

        # Add to list for all patients
        all_sweeps += ref_positions.tolist()
        all_synonymous += synonymous.tolist()

    return np.array(all_sweeps), np.array(all_synonymous), region_length


if __name__ == "__main__":
    region = "gag"
    sweeps, synonymous, region_length = get_all_patient_sweeps(region)

    syns = np.zeros(region_length)
    syns[sweeps[synonymous]] += 1
    non_syns = np.zeros(region_length)
    non_syns[sweeps[~synonymous]] += 1

    fontsize = 16
    linewidth = 0.5
    plt.figure(figsize=(14, 7))
    plt.plot(syns, '-', linewidth=linewidth, label="Synonymous")
    plt.plot(non_syns, '-', linewidth=linewidth, label="Non-synonymous")
    plt.xlabel("Position", fontsize=fontsize)
    plt.ylabel("Sweeping sites", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/Sweep_position_{region}.png", format="png")
    plt.show()
