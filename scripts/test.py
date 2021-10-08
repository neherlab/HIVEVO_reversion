import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import filenames
import tools
from hivevo.patients import Patient
import divergence
from hivevo.HIVreference import HIVreference
from scipy.interpolate import interp1d


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":
    region = "pol"
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p11", "p9"]
    time_interpolation = np.arange(0, 5.1, 0.1)
    ref = HIVreference(subtype="any")

    diversity_consensus = []
    diversity_non_consensus = []
    divergence_consensus = []
    divergence_non_consensus = []

    plt.figure()
    for ii, patient_name in enumerate(patient_names):
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        div = divergence.divergence_in_time(patient, region, aft, "founder")
        map_to_HXB2 = patient.map_to_external_reference(region)
        diversity = tools.diversity_per_site(patient, region, aft)

        f = interp1d(patient.ysi, div, axis=0, bounds_error=False, fill_value=0)
        div_interpolated = f(time_interpolation)

        consensus_mask = tools.reference_mask(patient, region, aft, ref)
        non_consensus_mask = tools.non_reference_mask(patient, region, aft, ref)
        div_consensus = div_interpolated[-1, :] * consensus_mask
        div_non_consensus = div_interpolated[-1, :] * non_consensus_mask

        mask = (diversity != np.nan)
        mask2 = np.logical_and(mask, consensus_mask)
        mask3 = np.logical_and(mask, non_consensus_mask)

        diversity_consensus += diversity[mask2].tolist()
        diversity_non_consensus += diversity[mask3].tolist()
        divergence_consensus += div_consensus[mask2].tolist()
        divergence_non_consensus += div_non_consensus[mask3].tolist()

    idxs = np.argsort(diversity_consensus)
    diversity_consensus = np.array(diversity_consensus)[idxs]
    divergence_consensus = np.array(divergence_consensus)[idxs]
    idxs = np.argsort(diversity_non_consensus)
    diversity_non_consensus = np.array(diversity_non_consensus)[idxs]
    divergence_non_consensus = np.array(divergence_non_consensus)[idxs]

    plt.plot(diversity_consensus, divergence_consensus, ".", label="consensus", color="C0")
    plt.plot(diversity_non_consensus, divergence_non_consensus, ".", label="non_consensus", color="C1")
    fit = np.polyfit(diversity_consensus, divergence_consensus, deg=1)
    plt.plot(diversity_consensus, np.polyval(fit, diversity_consensus), "-",
             color="C0", label=f"{round(fit[0],3)}x + {round(fit[1],3)}")
    fit = np.polyfit(diversity_non_consensus, divergence_non_consensus, deg=1)
    plt.plot(diversity_non_consensus, np.polyval(fit, diversity_non_consensus),
             "-", color="C1", label=f"{round(fit[0],3)}x + {round(fit[1],3)}")

    plt.legend()
    plt.ylabel("Divergence at 5y")
    plt.xlabel("Diversity")
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")

    # from scipy import stats
    # X, Y = np.mgrid[0:1:100j, 0:1:100j]
    # data = [diversity_non_consensus, divergence_non_consensus]
    # data = np.array(data)
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # kernel = stats.gaussian_kde(data)
    # Z = np.reshape(kernel(positions).T, X.shape)
    #
    # plt.figure()
    # plt.title("Unormalized")
    # plt.imshow(np.rot90(Z), extent=[0, 1, 0, 1])
    # plt.ylabel("Divergence at 5y")
    # plt.xlabel("Diversity")
    # plt.colorbar(label="Probability")
    #
    # Z = Z / np.sum(Z, axis=1)[:, np.newaxis]
    # plt.figure()
    # plt.title("Normalized")
    # plt.imshow(np.rot90(Z), extent=[0, 1, 0, 1])
    # plt.ylabel("Divergence at 5y")
    # plt.xlabel("Diversity")
    # plt.colorbar(label="Probability")
    #
    # plt.figure()
    # cmap = matplotlib.cm.get_cmap('plasma')
    # for ii in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
    #     plt.plot(Z[ii, :], '-', color=cmap(ii / 100), label=f"Diversity {ii/100}")
    # plt.grid()
    # plt.xlabel("Divergence at 5y")
    # plt.ylabel("Probability")
    # plt.legend()
    plt.show()
