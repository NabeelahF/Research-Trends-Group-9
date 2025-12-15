import gzip
import shutil
import os
import csv
from Bio import SeqIO

def uncompress_file(gz_file, out_file):
    if not os.path.exists(out_file):
        print(f"Unzipping {gz_file}...")
        with gzip.open(gz_file, 'rt') as f_in:
             with open(out_file, 'w') as f_out:
                 shutil.copyfileobj(f_in, f_out)
        print("Unzip complete.")
    else:
        print(f"{out_file} already exists. Skipping unzip.")

def create_subset(seq_file, truth_csv, out_fasta, num_proteins=5):
    print(f"Selecting {num_proteins} proteins for benchmarking...")
    
    # Get target proteins from Truth file
    targets = set()
    with open(truth_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.add(row['Protein'])
            if len(targets) >= num_proteins * 2: # Get a few spares
                break
    
    print(f"Target proteins: {list(targets)}")
    
    # Extract from FASTA
    count = 0
    selected_records = []
    
    # UniProt Fasta headers usually look like: >sp|P12345|ID_SPECIES ...
    # Our CSV has "ID_SPECIES" (e.g. P53_HUMAN) or Accession?
    # HumVar CSV earlier output showed: "P26439" (Accession) or "Q9H2F3".
    # Let's check the FASTA format. PolyPhen seqs usually use Accession or ID.
    # We will check if ID is in header.
    
    with open(seq_file, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # header example: "sp|P12345|ID..." or just "P12345"
            # match check
            found = False
            for t in targets:
                if t in record.id or t in record.description:
                    found = True
                    # Update target to match record ID for consistency if needed?
                    break
            
            if found:
                selected_records.append(record)
                count += 1
                if count >= num_proteins:
                    break
                    
    if selected_records:
        SeqIO.write(selected_records, out_fasta, "fasta")
        print(f"Wrote {len(selected_records)} sequences to {out_fasta}")
    else:
        print("No matching proteins found! Check ID formats.")

if __name__ == "__main__":
    gz_path = "human-2011_12.seq.gz"
    seq_path = "human-2011_12.seq"
    truth_path = "humvar_GroundTruth.csv"
    bench_path = "benchmark_query.fasta"
    
    if os.path.exists(gz_path):
        uncompress_file(gz_path, seq_path)
        create_subset(seq_path, truth_path, bench_path)
    else:
        print(f"Error: {gz_path} not found.")
