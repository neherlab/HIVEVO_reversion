# HIVEVO_reversion
Repository for the analysis and figures of Valentin Druelle's paper on HIV-1 reversion. This is a cleaned up version of the analysis done in HIVEVO_recombination. It uses some of the code present in Fabio's repository HIVEVO_access.

\TODO

<!-- ## Intermediate data
Intermediate data for the between host and within host analysis can be found in a compress format in the repository. They need to be uncompressed before usage. TODO
The intermediate data can be generated in the following way:
- for the within host analysis using `python scripts/WH_intermediate_data.py make-data`. It will generate all the intermediate data needed for the within host analysis. One can use `python scripts/WH_intermediate_data.py clean-data` to remove the intermediate data.
- for the between host analysis using snakemake to execute the rule `figure data`. This will compute a bunch of files for the 3 HIV-1 genes studied, which can take a lot of time. For use in the University Basel it is recommended to do this on the cluster instead. One can use `snakemake clean` to remove the intermediate files created.

Command to launch the jobs on the cluster:
`snakemake --jobs=16 --cluster "sbatch --time={cluster.time} --mem={cluster.mem} --cpus-per-task={cluster.n} --qos={cluster.qos}" --jobscript submit.sh --cluster-config cluster.json --jobname "{rulename}_{jobid}" ` -->
