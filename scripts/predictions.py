import numpy as np
import matplotlib.pyplot as plt
import tools
from hivevo.patients import Patient
import divergence
from hivevo.HIVreference import HIVreference
from scipy.interpolate import interp1d


def get_diversity_divergence(region, ref=HIVreference(subtype="any"),
                             time_interpolation=np.arange(0, 5.1, 0.1),
                             patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p11", "p9"]):
    """
    Computes the divergence for all patients at 5 years, concatenate all values and return them with the
    corresponding diversity values.
    """
    diversity_consensus = []
    diversity_non_consensus = []
    divergence_consensus = []
    divergence_non_consensus = []

    for ii, patient_name in enumerate(patient_names):
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        div = divergence.divergence_in_time(patient, region, aft, "founder")
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

    return diversity_consensus, divergence_consensus, diversity_non_consensus, divergence_non_consensus


def plot_diversity_divergence(diversity_consensus, divergence_consensus, diversity_non_consensus,
                              divergence_non_consensus, smooth_window=50):
    plt.figure()
    x, y = smooth(diversity_consensus, divergence_consensus, smooth_window)
    plt.plot(x, y, ".", label="consensus", color="C0")
    x, y = smooth(diversity_non_consensus, divergence_non_consensus, smooth_window)
    plt.plot(x, y, ".", label="non_consensus", color="C1")
    fit = compute_diversity_divergence_fit(diversity_consensus, divergence_consensus)
    plt.plot(diversity_consensus, np.polyval(fit, diversity_consensus), "-",
             color="C0", label=f"{round(fit[0],3)}x + {round(fit[1],3)}")
    fit = compute_diversity_divergence_fit(diversity_non_consensus, divergence_non_consensus)
    plt.plot(diversity_non_consensus, np.polyval(fit, diversity_non_consensus),
             "-", color="C1", label=f"{round(fit[0],3)}x + {round(fit[1],3)}")

    plt.legend()
    plt.ylabel("Divergence at 5y")
    plt.xlabel("Diversity")
    plt.grid()
    # plt.yscale("log")
    # plt.xscale("log")


def compute_diversity_divergence_fit(diversity, divergence):
    "Computes the linear regression of divergence vs diversity."
    return np.polyfit(diversity, divergence, deg=1)


def model_prediction(diversity, fit_consensus, fit_non_consensus, t):
    mu_plus = diversity * fit_consensus[0] / 5 + fit_consensus[1] / 5
    mu_minus = diversity * fit_non_consensus[0] / 5 + fit_non_consensus[1] / 5

    saturation_time = 1 / (mu_plus + mu_minus)
    d_err_20y = (mu_plus + mu_minus) * t / (1 - np.exp(-(mu_plus + mu_minus) * t)) - 1

    return saturation_time, d_err_20y


def smooth(x, y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    x_smooth = np.convolve(x, box, mode="valid")
    return x_smooth, y_smooth


if __name__ == "__main__":
    region = "pol"
    tmp = get_diversity_divergence(region)
    diversity_consensus = tmp[0]
    divergence_consensus = tmp[1]
    diversity_non_consensus = tmp[2]
    divergence_non_consensus = tmp[3]

    plot_diversity_divergence(diversity_consensus, divergence_consensus, diversity_non_consensus,
                              divergence_non_consensus)

    plt.figure()
    hist_consensus, bins = np.histogram(diversity_consensus, bins=40)
    bins = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bins, hist_consensus, ".-", label="consensus")
    hist_non_consensus, bins = np.histogram(diversity_non_consensus, bins=40)
    bins = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bins, hist_non_consensus, ".-", label="non-consensus")
    plt.ylabel("Counts")
    plt.xlabel("Diversity")
    plt.legend()
    plt.yscale("log")
    plt.grid()

    consensus_fit = compute_diversity_divergence_fit(diversity_consensus, divergence_consensus)
    non_consensus_fit = compute_diversity_divergence_fit(diversity_non_consensus, divergence_non_consensus)

    x = np.linspace(0, 0.8, 20)
    t = 20  # years
    saturation_time, d_err_20y = model_prediction(x, consensus_fit, non_consensus_fit, t)
    plt.figure()
    plt.plot(x, d_err_20y)
    plt.xlabel("Diversity")
    plt.ylabel("Relative distance error after 20y")
    plt.grid()

    times = np.linspace(1, 50, 20)
    errors = np.array([])
    diversity = tools.get_diversity(f"data/BH/alignments/to_HXB2/{region}_1000.fasta")
    for t in times:
        tau, err = model_prediction(diversity, consensus_fit, non_consensus_fit, t)
        error = np.mean(err)
        errors = np.concatenate((errors, np.array([error])))

    plt.figure()
    plt.plot(times, errors, ".")
    plt.xlabel("Time [years]")
    plt.ylabel("Average predicted error over all sites")
    plt.grid()

    plt.show()
