#!/bin/sh

#SBATCH --output=/scicore/home/neher/druell0000/PhD/HIVEVO_reversion/log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/scicore/home/neher/druell0000/PhD/HIVEVO_reversion/log/%j.err                  # where to store error messages

# activate conda environment
source /scicore/home/neher/druell0000/miniconda3/condabin/conda
conda activate hivevo

{exec_job}
