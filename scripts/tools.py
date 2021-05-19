import numpy as np

def initial_idx_mask(patient, region, aft):
    """
    Return a 3D (time*letter*genome_position) boolean matrix where True are all the position that correspond
    to the initial sequence.
    """
    initial_idx = patient.get_initial_indices(region)
    #TODO: optimize this, it is really slow
    mask = np.zeros(aft.shape, dtype=bool)
    for ii in range(aft.shape[2]):
        mask[:, initial_idx[ii], ii] = np.ones(aft.shape[0], dtype=bool)
    return mask

def get_mutation_positions(patient, region, aft, eps=0.01):
    """
    Return a 3D (time*letter*genome_position) boolean matrix where True are mutations positions with more
    than eps frequency.
    Original nucleotides are not considered as mutations.
    """
    mutation_positions = aft > eps
    mutation_positions[initial_idx_mask(patient, region, aft)] = False
    return mutation_positions

def get_fixation_positions(patient, region, aft, eps=0.01, timepoint="any"):
    """
    Return a 2D (letter*genome_position) boolean matrix where True are the mutations with more than 1-eps
    frequency at some timepoint / last time point.
    timepoint = ["any", "last"]
    """
    fixation_positions = get_mutation_positions(patient, region, aft, 1-eps)

    if timepoint == "any":
        return np.sum(fixation_positions, axis=0, dtype=bool)
    elif timepoint == "last":
        return fixation_positions[-1,:,:]
    else:
        raise ValueError("Condition of fixation is not understood.")
