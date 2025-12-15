
import csv
import sys
from Bio import SeqIO

def extract_benchmark_sequences(ground_truth_csv, database_fasta, output_fasta, limit=50):
    # 1. Get List of Proteins from Ground Truth
    print(f"Reading protein IDs from {ground_truth_csv}...")
    protein_ids = set()
    with open(ground_truth_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Protein']:
                protein_ids.add(row['Protein'])
    
    print(f"Found {len(protein_ids)} unique proteins in ground truth.")
    
    # 2. Scan FASTA and extract
    print(f"Scanning {database_fasta} for matches...")
    count = 0
    found_ids = set()
    
    with open(output_fasta, 'w') as out_f:
        # Using SeqIO.parse instead of index to save memory on large file
        for record in SeqIO.parse(database_fasta, "fasta"):
            # Check ID
            # SwissProt header format: sp|ID|Name ...
            # Biopython record.id usually is the first part "sp|ID|Name"
            # We need to extract the ID.
            
            # Example: sp|P26439|3BHS2_HUMAN
            parts = record.id.split('|')
            if len(parts) >= 2:
                pid = parts[1]
                if pid in protein_ids:
                    if pid not in found_ids:
                        SeqIO.write(record, out_f, "fasta")
                        found_ids.add(pid)
                        count += 1
                        if count % 100 == 0:
                            print(f"Extracted {count} sequences...")
                        if count >= limit:
                            break
    
    print(f"Done. Extracted {count} sequences to {output_fasta}")

if __name__ == "__main__":
    extract_benchmark_sequences(
        "sift4g_python/humvar_GroundTruth.csv",
        "uniprot_sprot.fasta/uniprot_sprot.fasta", 
        "sift4g_python/benchmark_query_2000.fasta", 
        limit=2000
    )
