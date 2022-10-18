# HIVEVO_reversion
Repository for the analysis and figures of the paper "Reversion to consensus are positively selected in HIV-1 and bias substitution rate estimates" by Valentin Druelle and Richard Neher.
This repository contains a compressed version of the intermediate data used to create the figures along with the scripts needed for that. 
It also contains the scripts to generate the intermediate data yourself if needed.
The raw files being too heavy for a github repository, you will have to download the full dataset yourself.

## Figure plots
All the figures are plotted using the `Paper_figures.py` script.
You will have to uncompress the intermediate data as explained in the following seciton.
## Working with the intermediate files
This section is intended for people who which to work with the intermediate data used for the paper.
Due to size issues, the intermediate data is saved in a compressed format in `data_mini/`.
One can use the `unpack_data.sh` script to uncompress it, which will generate the `data/` folder.
This can be done with the command `./unpack_data.sh` on Linux distributions.

This data contains:
- The subsampled sequences for the BH analysis for the different regions and their metadata in `data/BH/raw/`
- The alignments for these sequences in `data/BH/alignments/to_HXB2/`
- The trees and TreeTime files in `data/BH/intermediate_files/`
- The reference HXB2 sequence in `data/BH/reference/`
- The dictionnaries used for the WH analysis in `data/WH/`
- The MSAs generated for the modeling part in `data/modeling/generated_MSA/`
- The trees generated for the modeling part in `data/modeling/generate_trees/`

## Working with the raw files
This section is intended for people that which to regenerate the intermediate files. This is computationaly intensive and will take several hours on a regular laptop.

The full dataset folder can be found here: TODO. 
It contains both the raw data and the intermediate files needed for the analysis.
One can remove the intermediate files using `snakemake clean --cores 1` and regenerate them as explained in the following secitons.
### Generate between host data
For the between host analysis, make sure the raw data is in the `data/BH/raw` folder, then use snakemake to execute the rule `all`. 
This can be done with the command `snakemake all --cores 8` to run it on 8 cores for example.
This will compute a bunch of files for the 3 HIV-1 genes studied, which can take a lot of time. 
Details on how to do this in section "Sidenote: use on the cluster".

### Generate within host and modeling data
The WH intermediate data is generated by using the HIVevo_access repo: https://github.com/neherlab/HIVEVO_access
It also requires the files from the between host analysis, so one has to generate them first.
Generate the intermediate data by using `python scripts/generate_data.py make-data`.
Note that to run properly, one needs to set the correct paths to the HIVevo access folder in the `scripts/filenames.py` file.
It will generate all the intermediate data needed for the within host analysis.
One can use `python scripts/WH_intermediate_data.py clean-data` to remove the intermediate data.

## Sidenote : use on the cluster
Generating the intermediate files (both for the BH and WH analysis) is computationnaly intensive. The whole thing will take several hours if run on a laptop.
The generation of the BH data can be speed up using a cluster to run the snakemake rules.
Command to launch the jobs on the cluster:
`snakemake all --jobs=16 --cluster "sbatch --time={cluster.time} --mem={cluster.mem} --cpus-per-task={cluster.n} --qos={cluster.qos}" --jobscript snake_submit.sh --cluster-config cluster.json --jobname "{rulename}_{jobid}" `
