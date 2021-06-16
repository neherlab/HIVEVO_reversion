#!/bin/sh

#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages

#Run .bashrc to initialize conda
source $HOME/.bashrc

# Activate conda env
conda activate hivevo

{exec_job}
