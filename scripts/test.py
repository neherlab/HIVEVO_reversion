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

if __name__ == "__main__":
    region = "gag"
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    for patient_name in patient_names:
        # patient_name = "p1"
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        ref = HIVreference(subtype="any")

        consensus_mask = tools.reference_mask(patient, region, aft, ref)
        non_consensus_mask = tools.non_reference_mask(patient, region, aft, ref)
        first_mask = tools.site_mask(aft, 1)
        second_mask = tools.site_mask(aft, 2)
        third_mask = tools.site_mask(aft, 3)

        map_to_ref = patient.map_to_external_reference(region)
        mapped_mask = np.in1d(np.arange(aft.shape[-1]), map_to_ref[:, 2])
        mask = np.logical_and(consensus_mask, first_mask)[mapped_mask]
        test = np.remainder(map_to_ref[:, 0][mask], 3)
        print(test[test != 0])

    # div = divergence.mean_divergence_in_time(patient, region, aft, "founder", ref)
    # time = patient.ysi
    # idxs = time < 5.3
    # print(f"last time: {np.round(patient.ysi,3)[idxs][-1]}")
    # print(f"1st: {np.round(div['non_consensus']['first'][idxs],3)}")
    # print(f"2st: {np.round(div['non_consensus']['second'][idxs],3)}")
