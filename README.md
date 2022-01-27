# HIVEVO_reversion
Repository for the analysis and figures of Valentin Druelle's paper on HIV-1 reversion. This is a cleaned up version of the analysis done in HIVEVO_recombination. It uses some of the code present in Fabio's repository HIVEVO_access.

## Data folder
`data_mini` contains the minimal data necessary to run the figures in compressed format. One can use the `unpack_data.sh` script to uncompress it. Alternatively, one can do it by hand with `gzip -d -r data_mini` and rename the folder to `data` instead of `data_mini` once it's done (`mv data_mini data`). With this you should be able to redo the figures.

\TODO

<!-- The full dataset folder can be found here: https://drive.switch.ch/index.php/s/9GRtcq2UrPHytzI. It contains both the raw data and the intermediate files needed for the analysis.

### Generate between host data
For the between host analysis, make sure the raw data is in the `data/BH/raw` folder, then use snakemake to execute the rule `figure_data`. This will compute a bunch of files for the 3 HIV-1 genes studied, which can take a lot of time. For use in the University Basel it is recommended to do this on the cluster instead. One can use `snakemake clean` to remove the intermediate files created.

### Generate within host data
The WH intermediate data is generated by using the HIVevo_access repo: https://github.com/neherlab/HIVEVO_access
Generate the intermediate data by using `python scripts/WH_intermediate_data.py make-data`. Note that to run properly, one needs to set the correct paths to the HIVevo access folder in the `scripts/filenames.py` file. It will generate all the intermediate data needed for the within host analysis. One can use `python scripts/WH_intermediate_data.py clean-data` to remove the intermediate data.

### Generate the modelling data
Generation of data from the modelling part can be done using the `scripts\gtr_modeling.py` file. This requires the intermediate files from the BH and WH analysis to run properly, so one has to generate those first.

## Figure plots
All the figures are plotted using the `Paper_figures.py` script.

## Use on the cluster
Command to launch the jobs on the cluster:
`snakemake figure_data --jobs=16 --cluster "sbatch --time={cluster.time} --mem={cluster.mem} --cpus-per-task={cluster.n} --qos={cluster.qos}" --jobscript submit.sh --cluster-config cluster.json --jobname "{rulename}_{jobid}" ` -->
