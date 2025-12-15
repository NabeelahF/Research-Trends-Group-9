import csv
import sys
import os

def convert_csv_to_fasta(csv_path, fasta_path):
    print(f"Converting {csv_path} to {fasta_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile, open(fasta_path, 'w', encoding='utf-8') as fastafile:
            reader = csv.reader(csvfile)
            headers = next(reader) # Skip header (sequence_window, label)
            
            count = 0
            for i, row in enumerate(reader):
                if not row:
                    continue
                
                # Assuming row[0] is sequence
                # We can store the label in the header for tracking: >index_label
                sequence = row[0].strip()
                label = row[1].strip() if len(row) > 1 else "?"
                
                # SIFT4G uses the header ID for filenames. 
                # Needs to be safe. "seq_{i}_{label}"
                header = f"seq_{i}_{label}"
                
                fastafile.write(f">{header}\n{sequence}\n")
                count += 1
                
        print(f"Successfully converted {count} sequences.")
        
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_fasta.py <csv_file> [fasta_file]")
        sys.exit(1)
        
    csv_input = sys.argv[1]
    if len(sys.argv) >= 3:
        fasta_output = sys.argv[2]
    else:
        # Default output name
        base, _ = os.path.splitext(csv_input)
        fasta_output = base + ".fasta"
        
    convert_csv_to_fasta(csv_input, fasta_output)
