import numpy as np
import pandas as pd
import copy
import filenames
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference
import tools
import pickle


class Trajectory():
    def __init__(self, frequencies, t, date, t_last_sample, t_previous_sample, fixation, threshold_low, threshold_high,
                 patient, region, position, nucleotide, synonymous, reversion, fitness_cost):

        self.frequencies = frequencies              # Numpy 1D vector
        self.t = t                                  # Numpy 1D vector (in days)
        self.date = date                            # Date at t=0 (int, in days)
        # Number of days at which last sample was taken for the patient (relative to t[0] = 0)
        self.t_last_sample = t_last_sample
        # Number of days between the first point of the trajectory and the previous sample that was taken
        self.t_previous_sample = t_previous_sample
        self.fixation = fixation                    # "fixed", "active", "lost" (at the next time point)
        self.threshold_low = threshold_low          # Value of threshold_low used for extraction
        self.threshold_high = threshold_high        # Value of threshold_high used for extraction
        self.patient = patient                      # Patient name (string)
        self.region = region                        # Region name, string
        self.position = position                    # Position on the region (int)
        self.nucleotide = nucleotide                # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha
        self.synonymous = synonymous                # True if this trajectory is part of synonymous mutation
        self.reversion = reversion                  # True if the trajectory is a reversion to consensus sequence
        self.fitness_cost = fitness_cost            # Associated fitness_cost from the HIV_fitness pooled files

    def __repr__(self):
        return str(self.__dict__)


def create_trajectory_list(patient, region, aft, ref, threshold_low=0.01, threshold_high=0.99,
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
    # Adding masking for low depth fragments
    depth = get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    # Exctract the full time series of af for mutations and place them in a 2D matrix as columns
    mutation_positions = tools.get_mutation_positions(patient, region, aft, threshold_low)
    region_mut_pos = np.sum(mutation_positions, axis=0, dtype=bool)
    # Mask to select positions where mutations are seen
    mask = np.tile(region_mut_pos, (aft.shape[0], 1, 1))
    # Mask to filter the aft in positions where there is no reference or seen to often gapped
    reference_mask = np.tile(get_reference_filter(patient, region, aft, ref,
                                                  gap_threshold), (aft.shape[0], aft.shape[1], 1))
    mask = np.logical_and(mask, reference_mask)
    mut_frequencies = aft[mask]
    mut_frequencies = np.reshape(mut_frequencies, (aft.shape[0], -1))  # each column is a different mutation

    # Map the original position and nucleotide
    i_idx, j_idx = np.meshgrid(range(mutation_positions.shape[1]), range(
        mutation_positions.shape[2]), indexing="ij")
    coordinates = np.array([i_idx, j_idx])
    coordinates = np.tile(coordinates, (aft.shape[0], 1, 1, 1))
    coordinates = np.swapaxes(coordinates, 1, 3)
    coordinates = np.swapaxes(coordinates, 1, 2)
    coordinates = coordinates[mask]
    coordinates = np.reshape(coordinates, (aft.shape[0], -1, 2))
    # coordinates[t,ii,:] gives the [nucleotide, genome_position] of the mut_frequencies[t,ii] for any t

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
    reversion_map = get_reversion_map(patient, region, aft, ref)
    seq_fitness = get_fitness_cost(patient, region, aft)  # to the "any" subtype by default

    date = patient.dsi[0]
    time = patient.dsi - date
    # iterate though all columns (<=> mutations trajectories)
    for ii in range(mut_frequencies.shape[1]):
        # iterate for all trajectories inside this column
        for jj, idx_start in enumerate(np.where(new_trajectory_filter[:, ii] == True)[0]):

            if not True in (trajectory_stop_filter[idx_start:, ii] == True):  # still active
                idx_end = None
            else:
                idx_end = np.where(trajectory_stop_filter[:, ii] == True)[0][jj]  # fixed or lost

            if idx_end == None:
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

            if idx_end == None:
                fixation = "active"
            elif filter_fixation[idx_end, ii] == True:
                fixation = "fixed"
            else:
                fixation = "lost"

            position = coordinates[0, ii, 1]
            nucleotide = coordinates[0, ii, 0]
            traj = Trajectory(np.ma.array(freqs), t - t[0], date + t[0], time[-1] - t[0], t_prev_sample, fixation, threshold_low,
                              threshold_high, patient.name, region, position=position, nucleotide=nucleotide,
                              synonymous=syn_mutations[nucleotide,
                                                       position], reversion=reversion_map[nucleotide, position],
                              fitness_cost=seq_fitness[position])
            trajectories = trajectories + [traj]

    # Quick and dirty fix to correct for trajectories that contain only masked data
    trajectories = [traj for traj in trajectories if False in traj.frequencies.mask]
    return trajectories


def create_all_patient_trajectories(region, patient_names=[]):
    if patient_names == []:
        patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

    trajectories = []
    ref = HIVreference(subtype="any")
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectories + create_trajectory_list(patient, region, aft, ref)

    return trajectories


def get_fragment_per_site(patient, region):
    """
    Returns a list of fragment associated to each position in the region.
    """
    fragment_list = [[]] * len(patient._region_to_indices(region))
    frag = patient._annotation_to_fragment_indices(region)
    fragment_names = [*frag][2:]

    for ii in range(len(fragment_list)):
        for frag_name in fragment_names:
            if ii in frag[frag_name][0]:
                fragment_list[ii] = fragment_list[ii] + [frag_name]

    return fragment_list, fragment_names


def get_fragment_depth(patient, fragment):
    "Returns the depth of the fragment for each time point."
    return [s[fragment] for s in patient.samples]


def associate_depth(fragments, fragment_depths, fragment_names):
    "Associate a bolean array (true where coverage is ok) to each positions of the region."
    bool_frag_depths = np.array(fragment_depths) == "ok"
    depths = []
    for ii in range(len(fragments)):
        if len(fragments[ii]) == 1:  # Site only belongs to 1 fragment
            depths += [bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][0])[0][0]]]
        elif len(fragments[ii]) == 2:  # Site belongs to 2 fragments => take the best coverage
            depth1 = bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][0])[0][0]]
            depth2 = bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][1])[0][0]]
            depths += [np.logical_or(depth1, depth2)]
        else:
            raise(ValueError("Number of fragments for each site must be either 1 or 2."))

    return np.swapaxes(np.array(depths), 0, 1)


def get_depth(patient, region):
    """
    Returns nb_timepoint*nb_site boolean matrix where True are samples where the depth was labeled "ok" in the tsv files.
    """
    fragments, fragment_names = get_fragment_per_site(patient, region)
    fragment_depths = [get_fragment_depth(patient, frag) for frag in fragment_names]
    return associate_depth(fragments, fragment_depths, fragment_names)


def get_reference_filter(patient, region, aft, ref, gap_threshold=0.1):
    """
    Returns a 1D boolean vector where False are the positions (in aft.shape[-1]) that are unmapped to reference or too often gapped.
    """
    map_to_ref = patient.map_to_external_reference(region)
    ungapped_genomewide = ref.get_ungapped(gap_threshold)
    ungapped_region = ungapped_genomewide[map_to_ref[:, 0]]

    # excludes the positions that are not mapped to the reference (i.e. insertions as alignement is unreliable)
    mask1 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[:, 2])

    # excludes positions that are often gapped in the reference (i.e. where the alignement is unreliable)
    mask2 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[ungapped_region, 2])

    return np.logical_and(mask1, mask2)


def get_reversion_map(patient, region, aft, ref):
    """
    Returns a 2D boolean matrix (nucleotide*patient_sequence_length) where True are the positions that
    correspond to the reference nucleotide.
    """
    reversion_map = np.zeros((aft.shape[1], aft.shape[2]), dtype="bool")
    map_to_ref = patient.map_to_external_reference(region)
    ref_idx = ref.get_consensus_indices_in_patient_region(map_to_ref)

    reversion_map[ref_idx, map_to_ref[:, 2]] = True
    return reversion_map


def get_fitness_cost(patient, region, aft, subtype="any"):
    """
    Returns a 1D vector (patient_sequence_length) with the fitness coefficient for each sites. Sites missing
    from the consensus sequence or without fitness_cost associated are nans.
    """
    filename = filenames.get_fitness_filename(region, subtype)
    data = pd.read_csv(filename, skiprows=[0], sep="\t")
    fitness_consensus = data["median"]
    map_to_ref = patient.map_to_external_reference(region)
    fitness = np.empty(aft.shape[2])
    fitness[:] = np.nan
    fitness[map_to_ref[:, 2]] = fitness_consensus[map_to_ref[:, 0] - map_to_ref[0][0]]
    return fitness


def make_trajectory_dict(remove_one_point=False):
    """
    Returns a dictionary of the form trajectories[region][type]. Trajectories[region][type] is a list of all
    the trajectories in the given region and with the given type.
    Possible regions are ["env", "pol", "gag", "all"].
    Possible types are ["syn", "non_syn", "rev", "non_rev"].
    """

    regions = ["env", "pol", "gag", "all"]
    trajectories = {}

    for region in regions:
        # Create the dictionary with the different regions
        if region != "all":
            tmp_trajectories = create_all_patient_trajectories(region)
        else:
            tmp_trajectories = create_all_patient_trajectories(
                "env") + create_all_patient_trajectories("pol") + create_all_patient_trajectories("gag")
        if remove_one_point:
            tmp_trajectories = [traj for traj in tmp_trajectories if traj.t[-1] != 0]
        trajectories[region] = tmp_trajectories

        # Split into sub dictionnaries (rev, non_rev and all)
        rev = [traj for traj in trajectories[region] if traj.reversion == True]
        non_rev = [traj for traj in trajectories[region] if traj.reversion == False]
        syn = [traj for traj in trajectories[region] if traj.synonymous == True]
        non_syn = [traj for traj in trajectories[region] if traj.synonymous == False]
        trajectories[region] = {"rev": rev, "non_rev": non_rev,
                                "syn": syn, "non_syn": non_syn, "all": trajectories[region]}

    return trajectories


def load_trajectory_dict(path="trajectory_dict"):
    trajectories = {}
    with open(path, 'rb') as file:
        trajectories = pickle.load(file)

    return trajectories


if __name__ == "__main__":
    region = "pol"
    patient = Patient.load("p1")
    ref = HIVreference(subtype="any")
    aft = patient.get_allele_frequency_trajectories(region)
    trajectories = create_trajectory_list(patient, region, aft, ref)
