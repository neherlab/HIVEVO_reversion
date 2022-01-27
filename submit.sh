#!/bin/sh

#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=05:59:00
#SBATCH --qos=6hours
#SBATCH --output=log/%x.%j.out                 # where to store the output ( %j is the jobID )
#SBATCH --error=log/%x.%j.err                  # where to store error messages (%x is the jobname)

#Run .bashrc to initialize conda
source $HOME/.bashrc

# Activate conda env
conda activate hivevo

# echo $@
python scripts/WH_intermediate_data.py make-data
