# Dirty script to generate metadata from names

import sys
import pandas as pd
from Bio import SeqIO

def metadata_from_names(sequences):
    """
    Creates a metadata tsv file from the sequences names.
    """
    columns = ["strain", "virus", "date", "subtype", "country"]
    raw_names = [seq.name for seq in sequences]
    df = pd.DataFrame(data=None, index=None, columns=columns)
    for raw_name in raw_names:
        raw_name_split = raw_name.split(".")
        date = raw_name_split[2] + "-XX-XX"
        subtype = raw_name_split[0]
        country = raw_name_split[1]

        tmp_col = [raw_name]
        tmp_col.append("HIV")
        tmp_col.append(date)
        tmp_col.append(subtype)
        tmp_col.append(country)
        tmp = pd.DataFrame(data=[tmp_col], columns=columns)
        df = df.append(tmp)
    return df

sequences_file = sys.argv[1]
output = sys.argv[2]

sequences = list(SeqIO.parse(sequences_file, "fasta"))
metadata = metadata_from_names(sequences)
metadata.to_csv(output, index=False, sep="\t")
