# Adds link to the scripts folder
import filenames
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from numba import jit
sys.path.append("../scripts/")


@jit(nopython=True)  # makes it ~10 times faster
def simulation_third(x, dt, rate_consensus, rate_non_consensus):
    """
    Returns the boolean vector x(t+dt) from x(t). Time unit is day, rates are per day and per nucleotide.
    Used to simulate one third of the sites (either 1st, 2nd or 3rd position sites).
    """
    nb_consensus = len(x[x])
    nb_non_consensus = len(x) - nb_consensus
    nb_rev = np.random.poisson(nb_non_consensus * rate_non_consensus * dt)
    nb_non_rev = np.random.poisson(nb_consensus * rate_consensus * dt)

    idxs_consensus = np.where(x)[0]
    idxs_non_consensus = np.where(~x)[0]

    mut_rev = np.random.choice(idxs_non_consensus, nb_rev)
    mut_non_rev = np.random.choice(idxs_consensus, nb_non_rev)

    idxs_mutation = np.concatenate((mut_rev, mut_non_rev))
    x[idxs_mutation] = ~x[idxs_mutation]
    return x


def simulation_step(x, dt, rates):
    """
    Returns the boolean vector x(t+dt) from x(t). Time unit is day, rates are per day and per nucleotide.
    Rates is a dict containing keys ["consensus", "non_consensus"].
    rates["consensus"] and rates["non_consensus"] are dict containing the rates for the keys ["first", "second", "third"]
    """
    nb_site = x.shape[0] // 3
    x[:nb_site] = simulation_third(x[:nb_site], dt, rates["consensus"]
                                   ["first"], rates["non_consensus"]["first"])
    x[nb_site:2 * nb_site] = simulation_third(x[nb_site:2 * nb_site],
                                              dt, rates["consensus"]["second"], rates["non_consensus"]["second"])
    x[2 * nb_site:] = simulation_third(x[2 * nb_site:], dt, rates["consensus"]
                                       ["third"], rates["non_consensus"]["third"])

    return x


def initialize_fixed_point(sequence_length, frequencies):
    """
    Return initialize_seq with frequence according to the fixed point for 1st, 2nd and 3rd nucleotides.
    Frequencies is a dictionary containing the frequency of consensus sites for the keys ["first", "second", "third"].
    """
    x = np.ones(sequence_length, dtype=bool)
    nb_site = x.shape[0] // 3

    for ii, key in enumerate(["first", "second", "third"]):
        nb_non_consensus = round(nb_site * (1 - frequencies[key]))
        idxs = np.random.choice(nb_site, nb_non_consensus)
        idxs += ii * nb_site
        x[idxs] = False
    return x


def run_simulation(x, simulation_time, dt, rates, sampling_time):
    """
    Runs a simulation and stores the sampled sequences the matrix sequences (nb_nucleotide * nb_sequences).
    x is modified during the simulation. The original sequence is included in the sequences matrix, in the first row.
    """
    ii = 0
    time = np.arange(0, simulation_time + 1, dt)
    nb_samples = simulation_time // sampling_time
    sequences = np.zeros(shape=(len(x), nb_samples + 1), dtype=bool)

    for t in time:
        if (t % sampling_time == 0):
            sequences[:, ii] = x
            ii += 1

        x = simulation_step(x, dt, rates)
    return sequences


def run_simulation_group(x_0, simulation_time, dt, rates, sampling_time, nb_sim):
    """
    Runs several simulation starting from the same x_0, and returns a 3D matrix containing the sequences
    (nb_nucleotide * nb_sequences * nb_simulation)
    """
    nb_samples = simulation_time // sampling_time
    sequences = np.zeros(shape=(len(x_0), nb_samples + 1, nb_sim), dtype=bool)

    for ii in range(nb_sim):
        x = np.copy(x_0)
        sim_matrix = run_simulation(x, simulation_time, dt, rates, sampling_time)
        sequences[:, :, ii] = sim_matrix

    return sequences


@jit(nopython=True)
def hamming_distance(a, b):
    """
    Returns the hamming distance between sequence a and b. Sequences must be 1D and have the same length.
    """
    return np.count_nonzero(a != b)


def distance_to_initial(sequences):
    """
    Returns a 2D matrix (timepoint*nb_sim) of hamming distance to the initial sequence.
    """
    result = np.zeros((sequences.shape[1], sequences.shape[2]))
    for ii in range(sequences.shape[1]):
        for jj in range(sequences.shape[2]):
            result[ii, jj] = hamming_distance(sequences[:, 0, jj], sequences[:, ii, jj])
    return result


def distance_to_pairs(sequences):
    """
    Returns a 2D matrix (timepoint*nb_pair_combination) of distance between sequences at each time point.
    """
    result = np.zeros((sequences.shape[1], math.comb(sequences.shape[2], 2)))
    for ii in range(sequences.shape[1]):
        counter = 0
        for jj in range(sequences.shape[2]):
            for kk in range(jj + 1, sequences.shape[2]):
                result[ii, counter] = hamming_distance(sequences[:, ii, jj], sequences[:, ii, kk])
                counter += 1
    return result


if __name__ == '__main__':
    evo_rates = {
        "pol": {"consensus": {"first": 1.98e-6, "second": 1.18e-6, "third": 5.96e-6},
                "non_consensus": {"first": 2.88e-5, "second": 4.549e-5, "third": 2.06e-5}}
    }

    # These are the equilibrium frequency of consensus sites extracted from intra host data
    equilibrium_frequency = {"pol": {"first": 0.952, "second": 0.975, "third": 0.860}}

    # These are per nucleotide per year, need to change it for per day to match the simulation
    BH_rates = {"all": 0.0009372268087945193, "first": 0.0006754649449205438,
                "second": 0.000407792658976286, "third": 0.0017656284793794623}
    for key in BH_rates.keys():
        BH_rates[key] /= 365

    nb_simulation = 10
    simulation_time = 36500  # in days
    dt = 10
    time = np.arange(0, simulation_time + 1, dt)
    sampling_time = 10 * dt
    sequence_length = 3000

    # True is consensus, False is non consensus
    # First position are the first third of sites, second position are 2nd third, 3rd are last third
    x_0 = initialize_fixed_point(sequence_length, equilibrium_frequency["pol"])
    sequences = run_simulation_group(x_0, simulation_time, dt, evo_rates["pol"], sampling_time, nb_simulation)

    plt.figure()
    plt.plot(x, mean_distance_initial, label="Mean distance to initial")
    plt.plot(time, theory, "k--", label="x")
    plt.xlabel("Time [years]")
    plt.ylabel("Hamming distance")
    plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.show()
