#!/bin/sh

#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages

# activate conda environment
source /scicore/home/neher/druell0000/miniconda3/condabin/conda
conda activate hivevo

{exec_job}
