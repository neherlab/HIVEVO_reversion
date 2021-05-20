import numpy as np


# def get_mutation_positions(patient, region, aft, eps=0.01):
#     """
#     Return a 3D (time*letter*genome_position) boolean matrix where True are mutations positions with more
#     than eps frequency.
#     Original nucleotides are not considered as mutations.
#     """
#     mutation_positions = aft > eps
#     mutation_positions[initial_idx_mask(patient, region, aft)] = False
#     return mutation_positions


def mutation_positions_mask(patient, region, aft, eps=0.01):
    """
    Return a 1D boolean matrix where True are positions where mutations are seen at some time with more than
    eps frequency.
    """
    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    aft_initial = np.squeeze(aft_initial)

    # mutation_positions = aft > eps
    # mutation_positions[initial_idx_mask(patient, region, aft)] = False
    # return mutation_positions


def get_fixation_positions(patient, region, aft, eps=0.01, timepoint="any"):
    """
    Return a 2D (letter*genome_position) boolean matrix where True are the mutations with more than 1-eps
    frequency at some timepoint / last time point.
    timepoint = ["any", "last"]
    """
    fixation_positions = get_mutation_positions(patient, region, aft, 1 - eps)

    if timepoint == "any":
        return np.sum(fixation_positions, axis=0, dtype=bool)
    elif timepoint == "last":
        return fixation_positions[-1, :, :]
    else:
        raise ValueError("Condition of fixation is not understood.")


if __name__ == "__main__":
    from hivevo.patients import Patient
    from hivevo.HIVreference import HIVreference
    import filenames

    region = "pol"
    patient = Patient.load("p1")
    ref = HIVreference(subtype="any")
    aft = patient.get_allele_frequency_trajectories(region)
    aft_initial = mutation_positions_mask(patient, region, aft)
    
