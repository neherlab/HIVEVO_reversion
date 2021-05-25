import numpy as np
import pandas as pd

import filenames
from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient


def mutation_positions_mask(patient, region, aft, eps=0.01):
    """
    Return a 1D boolean matrix where True are positions where new mutations are seen at some time with more
    than eps frequency.
    """
    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = np.ma.squeeze(aft_initial)
    aft_initial = np.ma.sum(aft_initial < 1 - eps, axis=0, dtype=bool)
    return aft_initial


# def get_fixation_positions(patient, region, aft, eps=0.01, timepoint="any"):
#     """
#     Return a 2D (letter*genome_position) boolean matrix where True are the mutations with more than 1-eps
#     frequency at some timepoint / last time point.
#     timepoint = ["any", "last"]
#     """
#     fixation_positions = get_mutation_positions(patient, region, aft, 1 - eps)
#
#     if timepoint == "any":
#         return np.sum(fixation_positions, axis=0, dtype=bool)
#     elif timepoint == "last":
#         return fixation_positions[-1, :, :]
#     else:
#         raise ValueError("Condition of fixation is not understood.")


def reference_filter_mask(patient, region, aft, ref, gap_threshold=0.1):
    """
    Returns a 1D boolean vector where False are the positions (in aft.shape[-1]) that are unmapped to
    reference or too often gapped.
    """
    map_to_ref = patient.map_to_external_reference(region)
    ungapped_genomewide = ref.get_ungapped(gap_threshold)
    ungapped_region = ungapped_genomewide[map_to_ref[:, 0]]

    # excludes positions that are not mapped to the reference (i.e. insertions as alignement is unreliable)
    mask1 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[:, 2])

    # excludes positions that are often gapped in the reference (i.e. where the alignement is unreliable)
    mask2 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[ungapped_region, 2])

    return np.logical_and(mask1, mask2)


def depth_mask(patient, region):
    """
    Returns boolean matrix (same shape as aft) where True are samples where the depth was labeled "ok" in the
    tsv files.
    """
    fragments, fragment_names = get_fragment_per_site(patient, region)
    fragment_depths = [get_fragment_depth(patient, frag) for frag in fragment_names]
    depth = associate_depth(fragments, fragment_depths, fragment_names)
    depth = np.tile(depth, (6, 1, 1))  # For the 6 nucleotides (ATGC-N)
    depth = np.swapaxes(depth, 0, 1)
    return depth


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


def reversion_map(patient, region, aft, ref):
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


if __name__ == "__main__":
    region = "env"
    patient = Patient.load("p1")
    ref = HIVreference(subtype="any")
    aft = patient.get_allele_frequency_trajectories(region)
    aft_initial = mutation_positions_mask(patient, region, aft)
    depth = depth_mask(patient, region)
