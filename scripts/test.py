# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import get_non_consensus_mask, get_consensus_mask
import trajectory
import functools
sys.path.append("../scripts/")


def get_sweep_mask(patient, aft, region, threshold_low=0.05):
    # Masking low depth
    depth = trajectory.get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = aft_initial[:, 0, :]

    mask = aft_initial <= threshold_low
    mask = np.sum(mask, axis=0, dtype=bool)
    return mask


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_sweep_sites_sum(region, patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    "Returns a 1D vector with the sum of sweep sites over all patients"
    sites = []
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        sweep_mask = get_sweep_mask(patient, aft, region, threshold_low=0.5)
        ref = HIVreference(subtype="any")
        reference_mask = trajectory.get_reference_filter(patient, region, aft, ref)
        sweep_mask = sweep_mask[reference_mask]
        sites = sites + [list(sweep_mask[:2964])]

    sites = np.array(sites)
    sites = np.sum(sites, axis=0, dtype=int)
    return sites


if __name__ == "__main__":
    region = "pol"
    fontsize = 16

    sites = get_sweep_sites_sum(region)
    # sites = smooth(sites, 5)

    plt.figure(figsize=(10, 7))
    plt.plot(sites)
    plt.xlabel("Position", fontsize=fontsize)
    plt.ylabel("Sweeping sites", fontsize=fontsize)
    plt.grid()
    # plt.savefig("Sweep_positions.png", format="png")
    plt.show()
