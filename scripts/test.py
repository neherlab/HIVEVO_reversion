"""
Script used to test the bootstrapping estimates for the rates of fig 1 BH
"""

import numpy as np
import matplotlib.pyplot as plt
from treetime import GTR, GTR_site_specific
import filenames
from Bio import AlignIO, Phylo
from Bio.Seq import Seq
from scipy.stats import gamma
from treetime.seqgen import SeqGen
from treetime.utils import parse_dates
from treetime import TreeTime, TreeAnc
from distance_in_time import get_reference_sequence, get_mean_distance_in_time


def standard():
    "Predictions with standard JK model."
    mu = 10.3e-4
    myGTR = GTR.standard(model="jc69", mu=mu, alphabet="nuc_nogap")

    t_plot = np.arange(1980, 2021)

    #  dates = {"root": np.arange(1941, 2020), "subtypes": np.arange(1970, 2020),
    #           "founder": np.arange(1980, 2020)}  # estimates from tree
    dates = {"root": 1925, "subtypes": 1965, "founder": 1980}  # to match t = 1980
    distances = {"root": [], "subtypes": [], "founder": []}

    for key in dates:
        for t in t_plot:
            distances[key] += [1 - myGTR.expQt(t-dates[key])[0, 0]]

    plt.figure()
    plt.plot(t_plot, (t_plot - dates["root"])*mu, label="RTT")
    for key in dates:
        plt.plot(t_plot, distances[key], label=key)

    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Distance")
    plt.xlim([1978, 2022])
    plt.ylim([0, 0.14])
    plt.show()


if __name__ == "__main__":
    L = 1000
    real_mu = 10.3e-4
    # mu = gamma.rvs(2, size=L, random_state=123) / 2
    mu = np.ones(L) * real_mu
    W = np.ones((4, 4))
    W = np.array([[0., 0.763, 2.902, 0.391],
                  [0.763, 0., 0.294, 3.551],
                  [2.902, 0.294, 0., 0.317],
                  [0.391, 3.551, 0.317, 0.]])
    p = np.ones((4, L))*0.25

    myGTR = GTR_site_specific.custom(mu, p, W, alphabet="nuc_nogap", approximate=False)
    print(myGTR.average_rate().mean())
    myGTR.mu *= real_mu / myGTR.average_rate().mean()
    print(myGTR.average_rate().mean())

    t_plot = np.arange(1980, 2021)
    dates = {"root": 1914, "subtypes": 1970, "founder": 1980}  # estimates from tree
    # dates = {"root": 1925, "subtypes": 1965, "founder": 1980}  # to match t = 1980
    distances = {"root": [], "subtypes": [], "founder": []}

    for key in dates:
        for t in t_plot:
            tmp = 1 - myGTR.expQt(t-dates[key])[0, 0, :]
            distances[key] += [np.mean(tmp)]

    plt.figure(figsize=(2, 1.5))
    plt.plot(t_plot, (t_plot - dates["root"])*real_mu, label="RTT")
    for key in dates:
        plt.plot(t_plot, distances[key], label=key)

    plt.xlabel("Year")
    plt.ylabel("Distance")
    plt.xlim([1978, 2022])
    plt.ylim([0, 0.14])
    plt.ticklabel_format(axis="x", style="plain")
    plt.show()
