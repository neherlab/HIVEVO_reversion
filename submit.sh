#!/bin/sh

#SBATCH --output=log/$SLURM_JOB_NAME.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%SLURM_JOB_NAME.err                  # where to store error messages

#Run .bashrc to initialize conda
source $HOME/.bashrc

# Activate conda env
conda activate hivevo

{exec_job}
