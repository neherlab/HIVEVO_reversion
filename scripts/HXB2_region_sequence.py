# Script to create the HXB2 sequences for each region (env pol and gag) from the HIVevo references
import filenames  # link to the hivevo_access folder
from hivevo.patients import Patient
from Bio import SeqIO

if __name__ == '__main__':
    ref_sequence = SeqIO.read("data/BH/reference/HXB2.fasta", format="fasta")
    for region in ["env", "pol", "gag"]:
        patient = Patient.load("p1")  # choice of patient is irrelevant for this
        map = patient.map_to_external_reference(region)
        region_seq = ref_sequence[map[0, 0]:map[-1, 0]+1]
        with open(f"data/BH/reference/HXB2_{region}.fasta", "w") as f:
            SeqIO.write(region_seq, f, "fasta")
