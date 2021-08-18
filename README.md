# HIVEVO_reversion
Repository for the analysis and figures of Valentin Druelle's paper on HIV-1 reversion. This is a cleaned up version of the analysis done in HIVEVO_recombination. It uses some of the code present in Fabio's repository HIVEVO_access.

Command to launch the jobs on the cluster:
`snakemake --jobs=16 --cluster "sbatch --time={cluster.time} --mem={cluster.mem} --cpus-per-task={cluster.n} --qos={cluster.qos}" --jobscript submit.sh --cluster-config cluster.json --jobname "{rulename}_{jobid}" `
