import json
import copy
import numpy as np

import filenames
import tools
from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient


class Trajectory():
    def __init__(self, frequencies, t, date, t_last_sample, t_previous_sample, fixation, threshold_low,
                 threshold_high, patient, region, position, nucleotide, synonymous, reversion, fitness_cost):

        self.frequencies = frequencies          # Numpy 1D vector
        self.t = t                              # Numpy 1D vector (in days)
        self.date = date                        # Date at t=0 (int, in days)
        # Number of days at which last sample was taken for the patient (relative to t[0] = 0)
        self.t_last_sample = t_last_sample
        # Number of days between the first point of the trajectory and the previous sample that was taken
        self.t_previous_sample = t_previous_sample
        self.fixation = fixation                # "fixed", "active", "lost" (at the next time point)
        self.threshold_low = threshold_low      # Value of threshold_low used for extraction
        self.threshold_high = threshold_high    # Value of threshold_high used for extraction
        self.patient = patient                  # Patient name (string)
        self.region = region                    # Region name, string
        self.position = position                # Position on the region (int)
        # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha
        self.nucleotide = nucleotide
        self.synonymous = synonymous            # True if this trajectory is part of synonymous mutation
        self.reversion = reversion              # True if the trajectory is a reversion to consensus sequence
        self.fitness_cost = fitness_cost        # Associated fitness_cost from the HIV_fitness pooled files

    def __repr__(self):
        return str(self.__dict__)

    def to_json(self):
        """
        Transform the array to a dictonary that is compatible with json dump.
        """
        tmp = copy.deepcopy(self.__dict__)
        tmp["frequencies_mask"] = tmp["frequencies"].mask.tolist()
        tmp["frequencies"] = tmp["frequencies"].data.tolist()
        tmp["t"] = tmp["t"].tolist()

        for key in tmp.keys():  # To replace numpy types by python standard types
            if isinstance(tmp[key], np.generic):
                tmp[key] = tmp[key].item()
        return tmp


def _make_coordinates(aft_shape, mutation_mask):
    """
    Returns the coordinates of the mutations. coordinates[t,ii,:] gives the [nucleotide, genome_position] of
    the mutations selected by mutation_mask for any t.
    """
    i_idx, j_idx = np.meshgrid(range(aft_shape[1]), range(aft_shape[2]), indexing="ij")
    coordinates = np.array([i_idx, j_idx])
    coordinates = np.tile(coordinates, (aft_shape[0], 1, 1, 1))
    coordinates = np.swapaxes(coordinates, 1, 3)
    coordinates = np.swapaxes(coordinates, 1, 2)
    coordinates = coordinates[mutation_mask]
    coordinates = np.reshape(coordinates, (aft_shape[0], -1, 2))
    return coordinates


def create_trajectory_list(patient, region, ref_subtype, threshold_low=0.01, threshold_high=0.99,
                           syn_constrained=False, gap_threshold=0.1):
    """
    Creates a list of trajectories from a patient allele frequency trajectory (aft).
    Select the maximal amount of trajectories:
        - trajectories are extinct before the first time point
        - trajectories are either active, extinct or fixed after the last time point, which is specified
        - a trajectory can be as small as 1 point (extinct->active->fixed, or extinct->active->exctinct)
        - several trajectories can come from a single aft (for ex. extinct->active->extinct->active->fixed)
        - masked datapoints (low depth / coverage) are included only if in the middle of a trajectory (ie. [0.2, --, 0.6] is kept, but [--, 0.2, 0] gives [0.2] and [0.5, --, 1] gives [0.5])
    """
    trajectories = []
    aft = patient.get_allele_frequency_trajectories(region)
    ref = HIVreference(subtype=ref_subtype)

    # Adding masking for low depth fragments
    depth = tools.depth_mask(patient, region)
    aft.mask = np.logical_or(aft.mask, ~depth)

    # Exctract the full time series of af for mutations and place them in a 2D matrix as columns
    # Mask to select positions where mutations are seen
    mutation_positions_mask = tools.mutation_positions_mask(patient, region, aft, threshold_low)
    # Mask to filter the aft in positions where there is no reference or seen to often gapped
    reference_mask = tools.reference_filter_mask(patient, region, aft, ref, gap_threshold)
    mutation_mask = np.logical_and(mutation_positions_mask, reference_mask)
    mutation_mask = np.tile(mutation_mask, (aft.shape[0], aft.shape[1], 1))
    mut_frequencies = aft[mutation_mask]
    mut_frequencies = np.reshape(mut_frequencies, (aft.shape[0], -1))  # each column is a different mutation

    # Map the original position and nucleotide
    coordinates = _make_coordinates(aft.shape, mutation_mask)

    # Removing "mutation" at first time point because we don't know their history, ie rising or falling
    mask2 = np.where(mut_frequencies[0, :] > threshold_low)
    mut_freq_mask = mut_frequencies.mask  # keep the mask aside as np.delete removes it
    mut_freq_mask = np.delete(mut_freq_mask, mask2, axis=1)
    mut_frequencies = np.array(np.delete(mut_frequencies, mask2, axis=1))
    coordinates = np.delete(coordinates, mask2, axis=1)

    filter1 = mut_frequencies > threshold_low
    filter2 = mut_frequencies < threshold_high

    # true for the rest of time points once it hits fixation
    filter_fixation = np.cumsum(~filter2, axis=0, dtype=bool)
    trajectory_filter = np.logical_and(~filter_fixation, filter1)
    new_trajectory_filter = np.logical_and(~trajectory_filter[:-1, :], trajectory_filter[1:, :])
    new_trajectory_filter = np.insert(new_trajectory_filter, 0, trajectory_filter[0, :], axis=0)
    trajectory_stop_filter = np.logical_and(trajectory_filter[:-1, :], ~trajectory_filter[1:, :])
    trajectory_stop_filter = np.insert(trajectory_stop_filter, 0, np.zeros(
        trajectory_stop_filter.shape[1], dtype=bool), axis=0)

    # Include the masked points in middle of trajectories (ex [0, 0.2, 0.6, --, 0.8, 1])
    stop_at_masked_filter = np.logical_and(trajectory_stop_filter, mut_freq_mask)
    stop_at_masked_shifted = np.roll(stop_at_masked_filter, 1, axis=0)
    stop_at_masked_shifted[0, :] = False
    stop_at_masked_restart = np.logical_and(stop_at_masked_shifted, new_trajectory_filter)

    new_trajectory_filter[stop_at_masked_restart] = False
    trajectory_stop_filter[np.roll(stop_at_masked_restart, -1, 0)] = False

    # Get boolean matrix to label trajectories as synonymous and/or reversion
    syn_mutations = patient.get_syn_mutations(region, mask_constrained=syn_constrained)
    reversion_map = tools.reversion_map(patient, region, aft, ref)
    seq_fitness = tools.get_fitness_cost(patient, region, aft, subtype=ref_subtype)

    date = patient.dsi[0]
    time = patient.dsi - date
    # iterate though all columns (<=> mutations trajectories)
    for ii in range(mut_frequencies.shape[1]):
        # iterate for all trajectories inside this column
        for jj, idx_start in enumerate(np.where(new_trajectory_filter[:, ii])[0]):

            if True not in (trajectory_stop_filter[idx_start:, ii]):  # still active
                idx_end = None
            else:
                idx_end = np.where(trajectory_stop_filter[:, ii])[0][jj]  # fixed or lost

            if idx_end is None:
                freqs = np.ma.array(mut_frequencies[idx_start:, ii])
                freqs.mask = mut_freq_mask[idx_start:, ii]
                t = time[idx_start:]
            else:
                freqs = np.ma.array(mut_frequencies[idx_start:idx_end, ii])
                freqs.mask = mut_freq_mask[idx_start:idx_end, ii]
                t = time[idx_start:idx_end]

            t_prev_sample = 0
            if idx_start != 0:
                t_prev_sample = time[idx_start - 1]
            t_prev_sample -= t[0]  # offset so that trajectory starts at t=0

            if idx_end is None:
                fixation = "active"
            elif filter_fixation[idx_end, ii]:
                fixation = "fixed"
            else:
                fixation = "lost"

            position = coordinates[0, ii, 1]
            nucleotide = coordinates[0, ii, 0]
            traj = Trajectory(freqs, t - t[0], date + t[0], time[-1] - t[0], t_prev_sample,
                              fixation, threshold_low, threshold_high, patient.name, region,
                              position=position, nucleotide=nucleotide,
                              synonymous=syn_mutations[nucleotide, position],
                              reversion=reversion_map[nucleotide, position],
                              fitness_cost=seq_fitness[position])
            trajectories = trajectories + [traj]

    # Quick and dirty fix to correct for trajectories that contain only masked data
    trajectories = [traj for traj in trajectories if False in traj.frequencies.mask]
    return trajectories


def create_all_patient_trajectories(region, ref_subtype="any", patient_names=[]):
    assert ref_subtype in ["any", "subtypes"], "ref_subtype should be 'any' or 'subtypes'"

    if patient_names == []:
        patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
        if ref_subtype == "subtypes":
            subtypes = ["AE", "B", "B", "B", "B", "B", "C", "B", "B", "B"]

    trajectories = []
    for ii, patient_name in enumerate(patient_names):
        patient = Patient.load(patient_name)
        subtype = ""
        if ref_subtype == "any":
            subtype = "any"
        else:
            subtype = subtypes[ii]
        trajectories = trajectories + create_trajectory_list(patient, region, subtype)

    return trajectories


def make_intermediate_data():
    """
    Creates the intermediate data files.
    """
    import bootstrap
    traj_list_names = ["Trajectory_list_any", "Trajectory_list_subtypes"]
    ref_subtypes = ["any", "subtypes"]

    for name, ref_subtype in zip(traj_list_names, ref_subtypes):
        trajectories = create_all_patient_trajectories("env", ref_subtype) + \
            create_all_patient_trajectories("pol", ref_subtype) + \
            create_all_patient_trajectories("gag", ref_subtype)

        with open("data/WH/" + name + ".json", "w") as f:
            json.dump([traj.to_json() for traj in trajectories], f, indent=4)

        bootstrap_dict, _ = bootstrap.make_bootstrap_mean_dict(trajectories, nb_bootstrap=100)
        # json formating of numpy arrays
        for key1 in bootstrap_dict.keys():
            for key2 in bootstrap_dict[key1].keys():
                bootstrap_dict[key1][key2]["mean"] = bootstrap_dict[key1][key2]["mean"].tolist()
                bootstrap_dict[key1][key2]["std"] = bootstrap_dict[key1][key2]["std"].tolist()

        with open("data/WH/bootstrap_mean_dict_" + ref_subtype + ".json", "w") as f:
            json.dump(bootstrap_dict, f, indent=4)


def instanciate_trajectory(traj_dict):
    """
    Instanciate Trajectory object from the dictionary style used for .json save.
    """
    kwargs = {}
    for key in traj_dict.keys():
        if key not in ["frequencies", "frequencies_mask"]:
            kwargs[key] = traj_dict[key]
    kwargs["frequencies"] = np.ma.array(traj_dict["frequencies"], mask=traj_dict["frequencies_mask"])
    kwargs["t"] = np.array(traj_dict["t"])

    return Trajectory(**kwargs)


def load_trajectory_list(filename):
    """
    Loads a trajectory list from the .json file.
    """
    with open(filename, "r") as f:
        trajectories = json.load(f)

    return [instanciate_trajectory(traj) for traj in trajectories]


def create_time_bins(bin_size=400):
    """
    Create time bins for the mean in time computation. It does homogeneous bins, except for the one at t=0 that
    only takes point where t=0. Bin_size is in days.
    """
    time_bins = [-5, 5]
    interval = [-600, 3000]
    while time_bins[0] > interval[0]:
        time_bins = [time_bins[0] - bin_size] + time_bins
    while time_bins[-1] < interval[1]:
        time_bins = time_bins + [time_bins[-1] + bin_size]

    return np.array(time_bins)


def get_mean_in_time(trajectories, bin_size=400, freq_range=[0.4, 0.6]):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the
    freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    trajectories = copy.deepcopy(trajectories)

    # Create bins and select trajectories going through the freq_range
    time_bins = create_time_bins(bin_size)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the
    # frequencies / times
    # to arrays for later computation of mean
    t_traj = np.array([])
    f_traj = np.array([])
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        t_traj = np.concatenate((t_traj, traj.t))
        f_traj = np.concatenate((f_traj, traj.frequencies))

    # Binning of all the data in the time bins
    filtered_fixed = [traj for traj in trajectories if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in trajectories if traj.fixation == "lost"]
    freqs, fixed, lost = [], [], []
    for ii in range(len(time_bins) - 1):
        freqs = freqs + [f_traj[np.logical_and(t_traj >= time_bins[ii], t_traj < time_bins[ii + 1])]]
        fixed = fixed + [len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])]
        lost = lost + [len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])]

    # Computation of the mean in each bin, active trajectories contribute their current frequency,
    # fixed contribute 1 and lost contribute 0
    mean = []
    for ii in range(len(freqs)):
        mean = mean + [np.sum(freqs[ii]) + fixed[ii]]
        if len(freqs[ii]) + fixed[ii] + lost[ii] != 0:
            mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])
        else:
            mean[-1] = np.nan
    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


def load_mean_in_time_dict(filename):
    """
    Loads the mean_in_time dictionary and returns it.
    """
    with open(filename, "r") as f:
        mean_in_time = json.load(f)

    for key1 in mean_in_time.keys():
        for key2 in mean_in_time[key1].keys():
            mean_in_time[key1][key2]["mean"] = np.array(mean_in_time[key1][key2]["mean"])
            mean_in_time[key1][key2]["std"] = np.array(mean_in_time[key1][key2]["std"])
    return mean_in_time


def offset_trajectories(trajectories, freq_range):
    """
    Offset trajectories to set t=0 at the point they are seen in the freq_range.
    """
    trajectories = copy.deepcopy(trajectories)
    # Selecting trajectories in frequency window
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]

        # Adds a last point for better visual if we now fixation or loss
        time = np.random.rand() * 400 + 100

        if traj.fixation == "fixed":
            traj.t = np.append(traj.t, [traj.t[-1] + time, 3000])
            traj.frequencies = np.append(traj.frequencies, [1, 1])
        elif traj.fixation == "lost":
            traj.t = np.append(traj.t, [traj.t[-1] + time, 3000])
            traj.frequencies = np.append(traj.frequencies, [0, 0])

    return trajectories


if __name__ == "__main__":
    # make_intermediate_data()

    trajectories = create_all_patient_trajectories("env", "any")
    print(len(trajectories))
    trajectories = create_all_patient_trajectories("env", "subtypes")
    print(len(trajectories))
