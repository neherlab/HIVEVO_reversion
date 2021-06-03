rule all:
    input:
        auspice_json = "visualisation/pol_1000.json",
        rates0 = "mutation_rates/pol_1000.json",
        tree = "intermediate_files/timetree_pol_1000.nwk",
        branch = "branch_lengths/pol_1000.json"


rule lanl_metadata:
    message:
        """
        Creating metadata for the original LANL data.
        """
    input:
        lanl_data = "data/BH/raw/{region}.fasta"
    output:
        lanl_metadata = "data/BH/raw/{region}_metadata.tsv"
    shell:
        """
        python scripts/snakecommands.py metadata-from-names {input.lanl_data} {output.lanl_metadata}
        """


rule sub_sample:
    message:
        """
        Subsampling the original lanl data homogeneously in time and creating subsample metadata.
        """
    input:
        lanl_data = "data/BH/raw/{region}.fasta",
        lanl_metadata = "data/BH/raw/{region}_metadata.tsv"
    output:
        sequences = "data/BH/raw/{region}_{nb_sequences}_subsampled.fasta",
        metadata = "data/BH/raw/{region}_{nb_sequences}_subsampled_metadata.tsv"
    shell:
        """
        python scripts/snakecommands.py subsample {input.lanl_data} {input.lanl_metadata} \
        {wildcards.nb_sequences} {output.sequences} {output.metadata} \
        --remove_subtype_o
        """


rule align:
    message:
        """
        Aligning sequences to {input.reference} using Augur. Add reference to alignment and strips gaps
        relative to reference.
        """
    input:
        sequences = rules.sub_sample.output.sequences,
        reference = "data/BH/reference/HXB2_{region}.fasta"
    output:
        alignment = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}.fasta"
    threads: 8
    shell:
        """
        augur align \
            --sequences {input.sequences} \
            --reference-sequence {input.reference} \
            --output {output.alignment} \
            --fill-gaps \
            --nthreads {threads}
        """

rule consensus:
    message:
        """
        Computing the consensus sequence of the {wildcards.region}_{wildcards.nb_sequences} alignment.
        """
    input:
        alignment = rules.align.output.alignment
    output:
        consensus_sequence = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_consensus.fasta"
    shell:
        """
        python scripts/snakecommands.py consensus {input.alignment} {output.consensus_sequence}
        """


rule split_positions:
    message:
        "Splitting {input.alignment} into 1st, 2nd and 3rd position alignments."
    input:
        alignment = rules.align.output.alignment
    output:
        alignment_first = "data/alignments/to_HXB2/{region}_{nb_sequences}_1st.fasta",
        alignment_second = "data/alignments/to_HXB2/{region}_{nb_sequences}_2nd.fasta",
        alignment_third = "data/alignments/to_HXB2/{region}_{nb_sequences}_3rd.fasta"
    shell:
        """
        python scripts/split_positions.py {input.alignment}
        """

rule tree:
    message:
        "Building tree using augur"
    input:
        alignment = rules.align.output.alignment
    output:
        tree = "intermediate_files/tree_{region}_{nb_sequences}.nwk"
    threads: 8
    shell:
        """
        augur tree \
            --alignment {input.alignment} \
            --output {output.tree} \
            --nthreads {threads}
        """

rule refine:
    message:
        """
        Refining tree using augur refine.
        """
    input:
        tree = rules.tree.output.tree,
        alignment = rules.align.output.alignment,
        metadata = rules.sub_sample.output.metadata
    output:
        tree = "intermediate_files/timetree_{region}_{nb_sequences}.nwk",
        node_data = "intermediate_files/branch_lengths_{region}_{nb_sequences}.json"
    params:
        coalescent = "opt",
        date_inference = "marginal",
        clock_filter_iqd = 4
    shell:
        """
        augur refine \
            --tree {input.tree} \
            --alignment {input.alignment} \
            --metadata {input.metadata} \
            --output-tree {output.tree} \
            --output-node-data {output.node_data} \
            --timetree \
            --coalescent {params.coalescent} \
            --date-confidence \
            --date-inference {params.date_inference} \
            --clock-filter-iqd {params.clock_filter_iqd}
        """

rule ancestral:
    message: "Reconstructing ancestral sequences and mutations"
    input:
        tree = rules.refine.output.tree,
        alignment = rules.align.output.alignment
    output:
        node_data = "intermediate_files/{region}_{nb_sequences}_nt_muts.json"
    params:
        inference = "joint"
    shell:
        """
        augur ancestral \
            --tree {input.tree} \
            --alignment {input.alignment} \
            --output-node-data {output.node_data} \
            --inference {params.inference}
        """

rule export:
    message: "Exporting data files for visualisation in auspice"
    input:
        tree = rules.refine.output.tree,
        metadata = rules.sub_sample.output.metadata,
        branch_lengths = rules.refine.output.node_data,
        nt_muts = rules.ancestral.output.node_data
    output:
        auspice_json = "visualisation/{region}_{nb_sequences}.json"
    shell:
        """
        augur export v2 \
            --tree {input.tree} \
            --metadata {input.metadata} \
            --node-data {input.branch_lengths} {input.nt_muts} \
            --output {output.auspice_json} \
            --title HIV-1_{wildcards.region}
        """


rule gtr:
    message: "Inferring GTR model for alignment {wildcards.region} using TreeTime."
    input:
        tree = rules.refine.output.tree,
        align = "data/alignments/to_HXB2/{region}_{nb_sequences}.fasta"
    output:
        gtr_json = "gtr/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/infer_gtr.py {input.align} {input.tree} {output.gtr_json}
        """

rule subalign_gtr:
    message: "Inferring gtr model for subalignment {wildcards.region}_{wildcards.position} using TreeTime."
    input:
        tree = rules.refine.output.tree,
        align = "data/alignments/to_HXB2/{region}_{nb_sequences}_{position}.fasta"
    output:
        gtr_json = "gtr/{region}_{nb_sequences}_{position}.json"
    shell:
        """
        python scripts/infer_gtr.py {input.align} {input.tree} {output.gtr_json}
        """

rule mutation_rates:
    message: "Computing the mutation_rates for region {wildcards.region}."
    input:
        refine_file = rules.refine.output.node_data,
        gtr_all = rules.gtr.output.gtr_json,
        gtr_first = "gtr/{region}_{nb_sequences}_1st.json",
        gtr_second = "gtr/{region}_{nb_sequences}_2nd.json",
        gtr_third = "gtr/{region}_{nb_sequences}_3rd.json"
    output:
        mutation_rates = "mutation_rates/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/extract_mut_rate.py {input.refine_file} {input.gtr_all} {input.gtr_first} \
            {input.gtr_second} {input.gtr_third} {output.mutation_rates}
        """

rule mean_branch_length:
    message: "Computing mean branch length for {wildcards.region}."
    input:
        refine_file = rules.refine.output.node_data,
    output:
        mean_branch_length = "branch_lengths/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/extract_mean_branch_length.py {input.refine_file} {output.mean_branch_length}
        """

rule clean:
    message: "Removing generated files."
    shell:
        """
        rm data/raw/*subsampled* -f
        rm data/alignments/to_HXB2/* -f
        rm intermediate_files/* -f
        rm visualisation/* -f
        rm gtr/* -f
        rm mutation_rates/* -f
        rm branch_lengths/* -f
        """
