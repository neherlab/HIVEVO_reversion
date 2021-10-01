import numpy as np
import matplotlib.pyplot as plt
import filenames
import tools
from hivevo.patients import Patient
import divergence
from hivevo.HIVreference import HIVreference
from scipy.interpolate import interp1d


if __name__ == "__main__":
    region = "pol"
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p11", "p9"]
    time_interpolation = np.arange(0, 5.1, 0.1)
    ref = HIVreference(subtype="any")
    # This is just to load the correct length of the region in HXB2, patient independant
    patient = Patient.load("p1")
    map_to_HXB2 = patient.map_to_external_reference(region)
    dim = map_to_HXB2[-1, 0] - map_to_HXB2[0, 0] + 1
    nb_consensus = np.zeros(dim)
    nb_non_consensus = np.zeros(dim)
    div_consensus = np.zeros((time_interpolation.shape[0], dim))
    div_non_consensus = np.zeros((time_interpolation.shape[0], dim))

    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        map_to_HXB2 = patient.map_to_external_reference(region)
        aft = patient.get_allele_frequency_trajectories(region)

        div = divergence.divergence_in_time(patient, region, aft, "founder")
        f = interp1d(patient.ysi, div, axis=0, bounds_error=False, fill_value=0)
        div_interpolated = f(time_interpolation)

        consensus_mask = tools.reference_mask(patient, region, aft, ref)
        non_consensus_mask = tools.non_reference_mask(patient, region, aft, ref)
        consensus_div = div_interpolated * consensus_mask
        non_consensus_div = div_interpolated * non_consensus_mask
        div_non_consensus[:, map_to_HXB2[:, 0] - map_to_HXB2[0, 0]] += non_consensus_div[:, map_to_HXB2[:, 2]]
        div_consensus[:, map_to_HXB2[:, 0] - map_to_HXB2[0, 0]] += consensus_div[:, map_to_HXB2[:, 2]]
        nb_consensus[map_to_HXB2[:, 0] - map_to_HXB2[0, 0]] += consensus_mask[map_to_HXB2[:, 2]]
        nb_non_consensus[map_to_HXB2[:, 0] - map_to_HXB2[0, 0]] += non_consensus_mask[map_to_HXB2[:, 2]]

    div_consensus[:, nb_consensus != 0] /= nb_consensus[nb_consensus != 0]
    div_non_consensus[:, nb_non_consensus != 0] /= nb_non_consensus[nb_non_consensus != 0]
    alignment_file = f"data/BH/alignments/to_HXB2/{region}_1000.fasta"
    diversity = tools.get_diversity(alignment_file)

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    lsmooth = 20

    plt.figure()
    ds = diversity[nb_consensus != 0]
    dcs = div_consensus[-1, nb_consensus != 0]
    sort_idxs = np.argsort(ds)
    plt.plot(ds[sort_idxs], dcs[sort_idxs], ".", label="consensus", color="C0")
    plt.plot(ds[sort_idxs], smooth(dcs[sort_idxs], lsmooth), "-", label="smoothed", color="C0")

    ds = diversity[nb_non_consensus != 0]
    dncs = div_non_consensus[-1, nb_non_consensus != 0]
    sort_idxs = np.argsort(ds)
    plt.plot(ds[sort_idxs], dncs[sort_idxs], ".", label="non_consensus", color="C1")
    plt.plot(ds[sort_idxs], smooth(dncs[sort_idxs], lsmooth), "-", label="smoothed", color="C1")

    plt.legend()
    plt.ylabel("Divergence at 5y")
    plt.xlabel("Diversity")
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
