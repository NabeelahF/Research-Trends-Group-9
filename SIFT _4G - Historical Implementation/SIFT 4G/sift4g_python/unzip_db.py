
import gzip
import shutil
import os

def unzip_file(gz_path, out_path):
    if not os.path.exists(gz_path):
        print(f"File not found: {gz_path}")
        return

    if os.path.exists(out_path):
        print(f"Output file already exists: {out_path}")
        return

    print(f"Unzipping {gz_path} to {out_path}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Done.")
    except Exception as e:
        print(f"Error unzipping: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gz_file = os.path.join(base_dir, "uniprot_sprot.fasta.gz")
    out_file = os.path.join(base_dir, "uniprot_sprot.fasta")
    
    unzip_file(gz_file, out_file)
