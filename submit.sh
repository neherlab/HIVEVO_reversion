#!/bin/sh

#SBATCH --output=log/%x.%j.out                 # where to store the output ( %j is the jobID )
#SBATCH --error=log/%x.%j.err                  # where to store error messages (%x is the jobname)

#Run .bashrc to initialize conda
source $HOME/.bashrc

# Activate conda env
conda activate hivevo

{exec_job}
