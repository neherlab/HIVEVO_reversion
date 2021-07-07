NB_SEQUENCES = [1000, 500, 250, 125]
REGIONS = ["env", "pol", "gag"]
SUBTYPES = ["B", "C"]

wildcard_constraints:
    region = "(env|pol|gag)",
    nb_sequences = "(4000|2000|1000|500|250|125)",
    subtype = "(B|C)",
    position = "(1st|2nd|3rd)"


rule all:
    input:
        consensus = "data/BH/alignments/to_HXB2/pol_1000_consensus.fasta",
        auspice_json = "visualisation/pol_1000.json",
        rates = "data/BH/mutation_rates/pol_1000.json",
        tree = "data/BH/intermediate_files/timetree_pol_1000.nwk",
        branch = "data/BH/branch_lengths/pol_1000.json",
        subtype = "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta"


rule figure_mut_rate:
    message:
        """
        Creating the files for the mutation rate figure.
        """
    input:
        reference_file_B = "data/BH/alignments/to_HXB2/pol_1000_B_consensus.fasta",
        reference_file_C = "data/BH/alignments/to_HXB2/pol_1000_C_consensus.fasta",
        alignment_file = "data/BH/alignments/to_HXB2/pol_1000.fasta",
        tree_file = "data/BH/intermediate_files/timetree_pol_1000.nwk",
        reference_files_root = expand("data/BH/intermediate_files/pol_{nb}_nt_muts.json", nb=NB_SEQUENCES),
        branch_length_file = expand(
            "data/BH/intermediate_files/branch_lengths_pol_{nb}.json", nb=NB_SEQUENCES),
        mutation_rates_file = expand("data/BH/mutation_rates/pol_{nb}.json", nb=NB_SEQUENCES)

rule figure_distance:
    message:
        """
        Creating the files for the BH distance in time figure.
        """
    input:
        reference_files = expand("data/BH/alignments/to_HXB2/{region}_1000_consensus.fasta", region=REGIONS),
        reference_files2 = expand("data/BH/intermediate_files/{region}_1000_nt_muts.json", region=REGIONS)

rule lanl_metadata:
    message:
        """
        Creating metadata for the original LANL data of region {wildcards.region}.
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
        Subsampling the original lanl data for region {wildcards.region} homogeneously in time and creating
        subsampled {wildcards.nb_sequences} sequences + their metadata.
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
        Aligning sequences {input.sequences} to {input.reference} using Augur. Add HXB2 reference to
        alignment and strips gaps relative to reference.
        """
    input:
        sequences = rules.sub_sample.output.sequences,
        reference = "data/BH/reference/HXB2_{region}.fasta"
    output:
        alignment = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}.fasta"
    threads: 4
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
        Computing the consensus sequence of {input.alignment}.
        """
    input:
        alignment = rules.align.output.alignment
    output:
        consensus_sequence = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_consensus.fasta"
    shell:
        """
        python scripts/snakecommands.py consensus {input.alignment} {output.consensus_sequence}
        """

rule split_subtypes:
    message:
        """
        Splitting {input.alignment} into subtype B and C.
        """
    input:
        alignment = rules.align.output.alignment
    output:
        alignment_B = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_B.fasta",
        alignment_C = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_C.fasta"
    shell:
        """
        python scripts/snakecommands.py split-subtypes {input.alignment}
        """

rule split_positions:
    message:
        "Splitting {input.alignment} into 1st, 2nd and 3rd position alignments."
    input:
        alignment = rules.align.output.alignment
    output:
        alignment_first = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_1st.fasta",
        alignment_second = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_2nd.fasta",
        alignment_third = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_3rd.fasta"
    shell:
        """
        python scripts/snakecommands.py split-positions {input.alignment}
        """

rule subtype_consensus:
    message:
        """
        Computing the consensus sequence of {input.alignment}.
        """
    input:
        alignment = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_{subtype}.fasta"
    output:
        consensus = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_{subtype}_consensus.fasta",
    shell:
        """
        python scripts/snakecommands.py consensus {input.alignment} {output.consensus}
        """


rule tree:
    message:
        "Building tree for {input.alignment} using augur and IQtree GTR-F-R10 model."
    input:
        alignment = rules.align.output.alignment
    output:
        tree = "data/BH/intermediate_files/tree_{region}_{nb_sequences}.nwk"
    threads: 4
    shell:
        """
        augur tree \
            --method iqtree \
            --tree-builder-args="-czb -m GTR+F+R10" \
            --alignment {input.alignment} \
            --output {output.tree} \
            --nthreads {threads}
        """

rule refine:
    message:
        """
        Computing  TimeTree from {input.tree} using augur refine.
        """
    input:
        tree = rules.tree.output.tree,
        alignment = rules.align.output.alignment,
        metadata = rules.sub_sample.output.metadata
    output:
        tree = "data/BH/intermediate_files/timetree_{region}_{nb_sequences}.nwk",
        node_data = "data/BH/intermediate_files/branch_lengths_{region}_{nb_sequences}.json"
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
            --clock-filter-iqd {params.clock_filter_iqd} \
            --branch-length-inference input
        """

rule ancestral:
    message: "Reconstructing ancestral sequences and mutations from {input.tree}"
    input:
        tree = rules.refine.output.tree,
        alignment = rules.align.output.alignment
    output:
        node_data = "data/BH/intermediate_files/{region}_{nb_sequences}_nt_muts.json"
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
    message:
        """
        Exporting data for {wildcards.region}_{wildcards.nb_sequences} files for visualisation in
        auspice.
        """
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
    message: "Inferring GTR model from {input.tree} using TreeTime."
    input:
        tree = rules.refine.output.tree,
        align = rules.align.output.alignment
    output:
        gtr_json = "data/BH/gtr/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/snakecommands.py gtr {input.tree} {input.align} {output.gtr_json}
        """

rule subalign_gtr:
    message:
        """
        Inferring gtr model for subalignment {wildcards.region}_{wildcards.nb_sequences}_{wildcards.position}
        using TreeTime.
        """
    input:
        tree = rules.refine.output.tree,
        align = "data/BH/alignments/to_HXB2/{region}_{nb_sequences}_{position}.fasta"
    output:
        gtr_json = "data/BH/gtr/{region}_{nb_sequences}_{position}.json"
    shell:
        """
        python scripts/snakecommands.py gtr {input.tree} {input.align} {output.gtr_json}
        """

rule mutation_rates:
    message: "Computing the mutation_rates for {wildcards.region}_{wildcards.nb_sequences}."
    input:
        refine_file = rules.refine.output.node_data,
        gtr_all = rules.gtr.output.gtr_json,
        gtr_first = "data/BH/gtr/{region}_{nb_sequences}_1st.json",
        gtr_second = "data/BH/gtr/{region}_{nb_sequences}_2nd.json",
        gtr_third = "data/BH/gtr/{region}_{nb_sequences}_3rd.json"
    output:
        mutation_rates = "data/BH/mutation_rates/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/snakecommands.py mutation-rate {input.refine_file} {input.gtr_all} {input.gtr_first} \
            {input.gtr_second} {input.gtr_third} {output.mutation_rates}
        """

rule mean_branch_length:
    message: "Computing mean branch length for {wildcards.region}_{wildcards.nb_sequences}."
    input:
        refine_file = rules.refine.output.node_data,
    output:
        mean_branch_length = "data/BH/branch_lengths/{region}_{nb_sequences}.json"
    shell:
        """
        python scripts/snakecommands.py mean-branch-length {input.refine_file} {output.mean_branch_length}
        """

rule clean:
    message: "Removing generated files."
    shell:
        """
        rm data/BH/raw/*subsampled* -f
        rm data/BH/alignments/to_HXB2/* -f
        rm data/BH/intermediate_files/* -f
        rm visualisation/* -f
        rm data/BH/gtr/* -f
        rm data/BH/mutation_rates/* -f
        rm data/BH/branch_lengths/* -f
        rm log/* -f
        """
