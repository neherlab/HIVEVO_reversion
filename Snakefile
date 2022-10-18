REGIONS = ["env", "pol", "gag"]
SUBTYPES = ["B", "C"]

wildcard_constraints:
    region = "(env|pol|gag)",
    subtype = "(B|C)",
    position = "(1st|2nd|3rd)"


rule all:
    message:
        """
        Creating the files for the BH distance in time figure left panel (figure 1 5 and 6) and the modeling
        (fig 4 S9 and S10).
        """
    input:
        consensus_sequences = expand("data/BH/alignments/to_HXB2/{region}_consensus.fasta", region=REGIONS),
        subtype_consensus_sequences = expand("data/BH/alignments/to_HXB2/{region}_{subtype}_consensus.fasta",
                                             region=REGIONS, subtype=SUBTYPES),
        root_files = expand("data/BH/intermediate_files/{region}_nt_muts.json", region=REGIONS),
        tree_files = expand("data/BH/intermediate_files/tree_{region}.nwk", region=REGIONS),
        alignment_files = expand("data/BH/alignments/to_HXB2/{region}.fasta", region=REGIONS),
        visualisation_files = expand("data/BH/visualisation/{region}.json", region=REGIONS),


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
        subsampled 1000 sequences + their metadata.
        """
    input:
        lanl_data = "data/BH/raw/{region}.fasta",
        lanl_metadata = "data/BH/raw/{region}_metadata.tsv"
    output:
        sequences = "data/BH/raw/{region}_subsampled.fasta",
        metadata = "data/BH/raw/{region}_subsampled_metadata.tsv"
    params:
        nb_sequences = 1000
    shell:
        """
        python scripts/snakecommands.py subsample {input.lanl_data} {input.lanl_metadata} \
        {params.nb_sequences} {output.sequences} {output.metadata} \
        --remove_subtype_o --remove_subtype_n
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
        alignment = "data/BH/alignments/to_HXB2/{region}.fasta"
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
        consensus_sequence = "data/BH/alignments/to_HXB2/{region}_consensus.fasta"
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
        alignment_B = "data/BH/alignments/to_HXB2/{region}_B.fasta",
        alignment_C = "data/BH/alignments/to_HXB2/{region}_C.fasta"
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
        alignment_first = "data/BH/alignments/to_HXB2/{region}_1st.fasta",
        alignment_second = "data/BH/alignments/to_HXB2/{region}_2nd.fasta",
        alignment_third = "data/BH/alignments/to_HXB2/{region}_3rd.fasta"
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
        alignment = "data/BH/alignments/to_HXB2/{region}_{subtype}.fasta"
    output:
        consensus = "data/BH/alignments/to_HXB2/{region}_{subtype}_consensus.fasta",
    shell:
        """
        python scripts/snakecommands.py consensus {input.alignment} {output.consensus}
        """


rule tree:
    message:
        "Building tree for {input.alignment} using augur and IQtree GTR-F-R10 model."
    input:
        alignment = rules.align.output.alignment,
        metadata = rules.sub_sample.output.metadata
    output:
        tree = "data/BH/intermediate_files/tree_{region}.nwk"
    threads: 4
    shell:
        """
        augur tree \
            --method iqtree \
            --tree-builder-args="-czb -m GTR+F+R10" \
            --alignment {input.alignment} \
            --output {output.tree} \
            --nthreads {threads}

        python scripts/snakecommands.py reroot-tree {output.tree} {input.alignment} {input.metadata}
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
        tree = "data/BH/intermediate_files/timetree_{region}.nwk",
        node_data = "data/BH/intermediate_files/branch_lengths_{region}.json"
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
        node_data = "data/BH/intermediate_files/{region}_nt_muts.json"
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
        Exporting data for {wildcards.region} files for visualisation in auspice.
        """
    input:
        tree = rules.refine.output.tree,
        metadata = rules.sub_sample.output.metadata,
        branch_lengths = rules.refine.output.node_data,
        nt_muts = rules.ancestral.output.node_data
    output:
        auspice_json = "data/BH/visualisation/{region}.json"
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
        gtr_json = "data/BH/gtr/{region}.json"
    shell:
        """
        python scripts/snakecommands.py gtr {input.tree} {input.align} {output.gtr_json}
        """

rule subalign_gtr:
    message:
        """
        Inferring gtr model for subalignment {wildcards.region}_{wildcards.position}
        using TreeTime.
        """
    input:
        tree = rules.refine.output.tree,
        align = "data/BH/alignments/to_HXB2/{region}_{position}.fasta"
    output:
        gtr_json = "data/BH/gtr/{region}_{position}.json"
    shell:
        """
        python scripts/snakecommands.py gtr {input.tree} {input.align} {output.gtr_json}
        """

rule mutation_rates:
    message: "Computing the mutation_rates for {wildcards.region}."
    input:
        refine_file = rules.refine.output.node_data,
        gtr_all = rules.gtr.output.gtr_json,
        gtr_first = "data/BH/gtr/{region}_1st.json",
        gtr_second = "data/BH/gtr/{region}_2nd.json",
        gtr_third = "data/BH/gtr/{region}_3rd.json"
    output:
        mutation_rates = "data/BH/mutation_rates/{region}.json"
    shell:
        """
        python scripts/snakecommands.py mutation-rate {input.refine_file} {input.gtr_all} {input.gtr_first} \
            {input.gtr_second} {input.gtr_third} {output.mutation_rates}
        """

rule mean_branch_length:
    message: "Computing mean branch length for {wildcards.region}."
    input:
        refine_file = rules.refine.output.node_data,
    output:
        mean_branch_length = "data/BH/branch_lengths/{region}.json"
    shell:
        """
        python scripts/snakecommands.py mean-branch-length {input.refine_file} {output.mean_branch_length}
        """

rule HXB2_regions:
    message: "Cutting HXB2 sequence in data/BH/reference/HXB2.fasta to the env pol and gag regions."
    input:
        HXB2_original = "data/BH/reference/HXB2.fasta"
    output:
        HXB2_env = "data/BH/reference/HXB2_env.fasta",
        HXB2_pol = "data/BH/reference/HXB2_pol.fasta",
        HXB2_gag = "data/BH/reference/HXB2_gag.fasta"
    shell:
        """
        python scripts/snakecommands.py hxb2-regions {input.HXB2_original}
        """

rule clean:
    message: "Removing generated files."
    shell:
        """
        rm data/BH/raw/*subsampled* -f
        rm data/BH/alignments/to_HXB2/* -f
        rm data/BH/intermediate_files/* -f
        rm data/BH/visualisation/* -f
        rm data/BH/gtr/* -f
        rm data/BH/mutation_rates/* -f
        rm data/BH/branch_lengths/* -f
        rm log/* -f
        rm data/BH/raw/*metadata* -f
        """
