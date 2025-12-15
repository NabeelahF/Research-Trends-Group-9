import os
import csv
import glob

def convert_dataset(data_dir, output_dir):
    datasets = ["humvar", "humdiv"]
    
    for ds in datasets:
        output_file = os.path.join(output_dir, f"{ds}_GroundTruth.csv")
        print(f"Creating {output_file}...")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Protein", "Position", "Ref", "Alt", "Label"])
            
            # Deleterious
            del_files = glob.glob(os.path.join(data_dir, f"{ds}*.deleterious.pph.input"))
            for fpath in del_files:
                print(f"  Reading {os.path.basename(fpath)}...")
                with open(fpath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            # P26439 10 A E
                            writer.writerow([parts[0], parts[1], parts[2], parts[3], "Deleterious"])

            # Neutral
            neu_files = glob.glob(os.path.join(data_dir, f"{ds}*.neutral.pph.input"))
            for fpath in neu_files:
                print(f"  Reading {os.path.basename(fpath)}...")
                with open(fpath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                             writer.writerow([parts[0], parts[1], parts[2], parts[3], "Tolerated"])
        
        print(f"Finished {ds}.")

if __name__ == "__main__":
    convert_dataset("training-2.2.2", ".")
