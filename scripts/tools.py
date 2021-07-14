import numpy as np
import pandas as pd

import filenames
from hivevo.HIVreference import HIVreference
from hivevo.patients import Patient


def mutation_positions_mask(patient, region, aft, eps=0.01):
    """Returns a 1D boolean matrix where True are positions where new mutations are seen at some time with more
    than eps frequency.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].
        aft (np.array): Allele frequency trajectories (3D).
        eps (float): Threshold for mutation detection.

    Returns:
        (np.array): Mask for mutation positions (1D).

    """
    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = np.ma.squeeze(aft_initial)
    mask = np.ma.sum(aft_initial < 1 - eps, axis=0, dtype=bool)
    return mask


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
    """Returns a 1D boolean vector where False are the genome positions that are unmapped to reference or too
    often gapped.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].
        aft (np.array): Allele frequency trajectories (3D).
        ref (HIVreference): HIVreference object from hivevo package.
        gap_threshold (float): Max frequency of gaps at a given position.

    Returns:
        (np.array): Mask for reference / ungapped genome positions (1D).

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
    """Returns boolean matrix where True are samples where the depth was labeled "ok" in the tsv files.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].

    Returns:
        (np.array): Mask for non problematic depth (same shape as aft).

    """
    fragments, fragment_names = get_fragment_per_site(patient, region)
    fragment_depths = [get_fragment_depth(patient, frag) for frag in fragment_names]
    depth = associate_depth(fragments, fragment_depths, fragment_names)
    depth = np.tile(depth, (6, 1, 1))  # For the 6 nucleotides (ATGC-N)
    depth = np.swapaxes(depth, 0, 1)
    return depth


def associate_depth(fragments, fragment_depths, fragment_names):
    """Associate a bolean array (true where coverage is ok) to each positions of the region.

    Args:
        fragments (list(list(string))): List containing a list of fragments of each genome positions.
        fragment_depths (list(list(string))): List containing a list of depth label for each fragment at all
                                              timepoints.
        fragment_names (list(string)): The fragment names in the region.

    Returns:
        (np.array): Bool 2D matrix where True are the positions/time where depth is ok.

    """
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
    """Returns a list of fragment associated to each position in the region and a list of the fragment names.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].

    Returns:
        (list(list(string))): List containing a list of depth label for each fragment at all timepoints.
        (list(string)): List containing the fragment names in the region.

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
    """Returns the depth of the fragment for each time point.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].

    Returns:
        list(string): List of the depth for each timepoint in the given fragment.

    """
    return [s[fragment] for s in patient.samples]


def reversion_map(patient, region, aft, ref):
    """Returns a 2D boolean matrix where True are the positions that correspond to the reference nucleotide.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].
        aft (np.array): Allele frequency trajectories (3D).
        ref (HIVreference): HIVreference object from hivevo package.

    Returns:
        (np.array): Mask where True are the reference nucleotide (2D: aft.shape[1]*aft.shape[2]).

    """
    reversion_map = np.zeros((aft.shape[1], aft.shape[2]), dtype="bool")
    map_to_ref = patient.map_to_external_reference(region)
    ref_idx = ref.get_consensus_indices_in_patient_region(map_to_ref)

    reversion_map[ref_idx, map_to_ref[:, 2]] = True
    return reversion_map


def get_fitness_cost(patient, region, aft, subtype="any"):
    """Returns a 1D vector of fitness coefficient for each sites. Sites missing from the consensus sequence or
       without fitness_cost associated are nans.

    Args:
        patient (Patient): The patient to analyse.
        region (string): Region to analyse ['env', 'pol', 'gag'].
        aft (np.array): Allele frequency trajectories (3D).
        subtype (string): either 'any' or 'B'.

    Returns:
        np.array: Fitness coefficient of each genome position.

    """
    if subtype not in ["any", "B"]:  # Fitness was computed only for global consensus and subtype B consensus
        tmp = np.empty(aft.shape[-1])
        tmp[:] = np.nan
        return tmp
    else:
        filename = filenames.get_fitness_filename(region, subtype)
        data = pd.read_csv(filename, skiprows=[0], sep="\t")
        fitness_consensus = data["median"]
        map_to_ref = patient.map_to_external_reference(region)
        fitness = np.empty(aft.shape[2])
        fitness[:] = np.nan
        fitness[map_to_ref[:, 2]] = fitness_consensus[map_to_ref[:, 0] - map_to_ref[0][0]]
        return fitness


def reference_mask(patient, region, aft, ref):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that correspond to the reference
    sequence. Position that are not mapped to reference or seen too often gapped are always False.
    """
    ref_filter = reference_filter_mask(patient, region, aft, ref)
    consensus_mask = reversion_map(patient, region, aft, ref)
    initial_idx = patient.get_initial_indices(region)
    # gives reversion mask at initial majority nucleotide
    consensus_mask = consensus_mask[initial_idx, np.arange(aft.shape[-1])]

    return np.logical_and(ref_filter, consensus_mask)


def non_reference_mask(patient, region, aft, ref):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that do not correspond to the
    reference sequences. Position that are not mapped to reference or seen too often gapped are always False.
    """
    ref_filter = reference_filter_mask(patient, region, aft, ref)
    consensus_mask = reversion_map(patient, region, aft, ref)
    initial_idx = patient.get_initial_indices(region)
    # gives reversion mask at initial majority nucleotide
    consensus_mask = consensus_mask[initial_idx, np.arange(aft.shape[-1])]

    return np.logical_and(ref_filter, ~consensus_mask)


def site_mask(aft, position):
    """
    Returns a 1D boolean vector of size aft.shape[-1] where True are the positions corresponding to 1st 2nd or
    3rd.
    """
    assert position in [1, 2, 3], "Position must be 1 2 or 3."
    position_mask = np.zeros(aft.shape[-1], dtype=bool)
    position_mask[position - 1::3] = True
    return position_mask


def load_root_sequence(root_file):
    """
    Loads the root sequence from the .json tree file given as argument.
    """
    import json
    with open(root_file) as f:
        data = json.load(f)
        root_sequence = list(data["nodes"]["NODE_0000000"]["sequence"])
    root_sequence = np.array(root_sequence)

    return root_sequence


def sequence_to_indices(sequence):
    """
    Returns the index version of the sequence. Translate nucleotides to numbers as in hivevo_access. A=0, C=1,
    G=2, T=3, -=4, N=5
    """
    seq = np.copy(sequence)
    for nuc, idx in zip(["A", "C", "G", "T", "-", "N"], range(6)):
        seq[sequence == nuc] = idx

    return np.array(seq, dtype=int)


def root_mask(patient, region, aft, root_file, ref):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that correspond to the root
    sequence. Position that are seen too often gapped are always False.
    """
    import os
    assert os.path.exists(root_file), f"File {root_file} doesn't exist."
    root_sequence = load_root_sequence(root_file)
    root_idxs = sequence_to_indices(root_sequence)
    initial_idxs = patient.get_initial_indices(region)

    ref_filter = reference_filter_mask(patient, region, aft, ref)
    root_mask = np.zeros(aft.shape[-1], dtype=bool)
    map_to_ref = patient.map_to_external_reference(region)  # Map to HXB2 sequence
    root_idxs_mapped = root_idxs[map_to_ref[:, 0] - map_to_ref[0, 0]]
    initial_idxs_mapped = initial_idxs[map_to_ref[:, 2]]
    mask = root_idxs_mapped == initial_idxs_mapped
    root_mask[map_to_ref[:, 2][mask]] = True

    return np.logical_and(root_mask, ref_filter)


def non_root_mask(patient, region, aft, root_file, ref):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that correspond to the root
    sequence. Position that are seen too often gapped are always False.
    """
    import os
    assert os.path.exists(root_file), f"File {root_file} doesn't exist."
    root_sequence = load_root_sequence(root_file)
    root_idxs = sequence_to_indices(root_sequence)
    initial_idxs = patient.get_initial_indices(region)

    ref_filter = reference_filter_mask(patient, region, aft, ref)
    root_mask = np.zeros(aft.shape[-1], dtype=bool)
    map_to_ref = patient.map_to_external_reference(region)  # Map to HXB2 sequence
    root_idxs_mapped = root_idxs[map_to_ref[:, 0] - map_to_ref[0, 0]]
    initial_idxs_mapped = initial_idxs[map_to_ref[:, 2]]
    mask = root_idxs_mapped == initial_idxs_mapped
    root_mask[map_to_ref[:, 2][mask]] = True

    return np.logical_and(~root_mask, ref_filter)


if __name__ == "__main__":
    region = "pol"
    patient = Patient.load("p1")
    ref = HIVreference(subtype="any")
    aft = patient.get_allele_frequency_trajectories(region)
    root_file = "data/BH/intermediate_files/pol_1000_nt_muts.json"
    root_mask = root_mask(patient, region, aft, root_file, ref)
    non_root_mask = non_root_mask(patient, region, aft, root_file, ref)

    for p in ["p1", "p2", "p3", "p4", "p5"]:
        print(f"Patient {p}")
        patient = Patient.load(p)
        map_to_ref = patient.map_to_external_reference(region)
        consensus_sequence = ref.get_consensus_in_patient_region(map_to_ref)
        consensus_sequence = np.array(consensus_sequence, dtype="<U1")
        root_sequence = load_root_sequence(root_file)
        root_sequence = root_sequence[map_to_ref[:, 0] - map_to_ref[0, 0]]
        patient_sequence = np.array(patient.get_initial_sequence(region), dtype="<U1")
        patient_sequence = patient_sequence[map_to_ref[:, 2]]

        d_root_consensus = len(np.where(root_sequence != consensus_sequence)[0])
        d_root_patient = len(np.where(root_sequence != patient_sequence)[0])
        d_patient_consensus = len(np.where(patient_sequence != consensus_sequence)[0])
        str = f"  root to consensus: {d_root_consensus}   root to patient {d_root_patient}   patient to consensus {d_patient_consensus}"
        print(str)
