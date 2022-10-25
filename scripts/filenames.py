"""
Defines the paths to relevant data from the HIVEevo_access analysis
"""
import os
import sys

# HIVEVO_access repository location
HIVEVO_PATH = "/scicore/home/neher/druell0000/PhD/HIVEVO_access"
# HIVEVO_PATH = "/home/valentin/Desktop/richardLab/scicore_local/PhD/HIVEVO_access"
sys.path.append(HIVEVO_PATH)

# HIVEVO data folder
HIVEVO_ROOT_DATA_PATH = "/scicore/home/neher/druell0000/PhD/MiSeq_HIV_Karolinska/"
# HIVEVO_ROOT_DATA_PATH = "/home/valentin/Desktop/richardLab/scicore_local/PhD/MiSeq_HIV_Karolinska/"
os.environ["HIVEVO_ROOT_DATA_FOLDER"] = HIVEVO_ROOT_DATA_PATH

# HIV_fitness_landscape fitness_pooled data folder
HIV_FITNESS_PATH = "/scicore/home/neher/druell0000/PhD/HIV_fitness_landscape/data/fitness_pooled"
# HIV_FITNESS_PATH = "/home/valentin/Desktop/richardLab/scicore_local/PhD/HIV_fitness_landscape/data/fitness_pooled"
sys.path.append(HIV_FITNESS_PATH)


def get_fitness_filename(region, subtype):
    "Returns the path to the selection_coefficient file."
    return HIV_FITNESS_PATH + "/nuc_" + region + "_selection_coefficients_unsensored_" + subtype + ".tsv"
