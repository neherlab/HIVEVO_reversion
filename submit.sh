#!/bin/sh

#SBATCH --output=log/%x.out                 # where to store the output ( %j is the jobname )
#SBATCH --error=log/%x.err                  # where to store error messages

#Run .bashrc to initialize conda
source $HOME/.bashrc

# Activate conda env
conda activate hivevo

{exec_job}
