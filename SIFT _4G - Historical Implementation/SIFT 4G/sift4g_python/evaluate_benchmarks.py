
import os
import sys
import csv
import time
from Bio import SeqIO
from collections import defaultdict

# Reuse SIFT logic
from sift4g_python.utils import Chain, create_file_name
from sift4g_python.database import search_database
from sift4g_python.alignment import create_aligner
from sift4g_python.prediction import sift_predictions

MAX_CANDIDATES = 5000
KMER_LENGTH = 5

def parse_ground_truth(csv_path):
    """
    Parses HumDiv/HumVar CSV.
    Expected columns: Protein, Position, Ref, Alt, Label
    Returns: dict[ProteinID] -> list of {Pos, Ref, Alt, Label}
    """
    data = defaultdict(list)
    unique_proteins = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['Protein']
            data[pid].append({
                'Position': int(row['Position']),
                'Ref': row['Ref'],
                'Alt': row['Alt'],
                'Label': row['Label'] # Deleterious / Tolerated
            })
            unique_proteins.add(pid)
            
    return data, unique_proteins

def extract_sequences(protein_ids, db_path, out_fasta):
    """
    Extracts sequences for given Protein IDs from SwissProt DB.
    Returns: list of Chain objects
    """
    print(f"Indexing {db_path}...")
    db_index = SeqIO.index(db_path, "fasta")
    
    found_chains = []
    missing = []
    
    with open(out_fasta, 'w') as f:
        for pid in protein_ids:
            # SwissProt IDs in fasta header look like: sp|P26439|...
            # We need to find the record where ID is present.
            # Direct lookup might fail if key format differs.
            # We'll try direct key first (if user provided full key), else scan keys.
            # BUT efficient way for many IDs:
            # Assuming 'Protein' column in CSV matches the ID in fasta header (e.g. P26439)
            # The keys in SeqIO.index(fasta) are the full description line usually
            # Or just ID depending on parser? "fasta" parser uses header up to space.
            # sp|P26439|...
            
            # Helper to find key
            # We can't iterate db_index conveniently if it's large and file-based.
            # We'll try to construct expected key patterns.
            pass

    # Re-approach: Iterate DB once and filter. Faster for large ID list vs large DB?
    # SwissProt is ~500MB. Reading once is fine.
    
    print(f"Scanning database for {len(protein_ids)} proteins...")
    target_ids = set(protein_ids)
    
    # We will iterate the file directly with SeqIO.parse to handle keys flexibly
    chains = []
    
    with open(out_fasta, 'w') as f_out:
        for record in SeqIO.parse(db_path, "fasta"):
            # Header example: sp|P26439|3BHS2_HUMAN
            # Extract accessions. 
            # Often P26439 is between pipes.
            parts = record.id.split('|')
            acc = None
            if len(parts) > 1:
                acc = parts[1]
            else:
                acc = record.id # Fallback
            
            if acc in target_ids or record.id in target_ids:
                # Found
                chains.append(Chain(record.id, str(record.seq)))
                SeqIO.write(record, f_out, "fasta")
                if acc in target_ids: target_ids.remove(acc)
                elif record.id in target_ids: target_ids.remove(record.id)
                
    print(f"Extracted {len(chains)} sequences.")
    if target_ids:
        print(f"Warning: {len(target_ids)} proteins not found in DB (e.g., {list(target_ids)[:5]})")
        
    return chains

def parse_sift_file(filepath, positions):
    """
    Parses SIFT file for specific positions.
    Returns: dict[Pos] -> scores list
    """
    if not os.path.exists(filepath): return {}
    
    scores_map = {}
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse matrix
        # Lines start with 1, 2, ...
        for line in lines:
            parts = line.split()
            if not parts: continue
            if parts[0].replace('.','',1).isdigit():
                pos = int(parts[0]) # 1-based index in alignment? usually SIFT output is 1-based
                if pos in positions:
                    row_scores = [float(x) for x in parts[1:]] # Skip ID column
                    scores_map[pos] = row_scores
    except: pass
    return scores_map

def evaluate(dataset_name, csv_path, db_path, work_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name}")
    print(f"{'='*60}")
    
    # 1. Parse Data
    print("Parsing ground truth...")
    gt_data, unique_pids = parse_ground_truth(csv_path)
    print(f"Found {len(unique_pids)} proteins, {sum(len(v) for v in gt_data.values())} variants.")
    
    # 2. Extract Sequences
    queries_fasta = os.path.join(work_dir, f"{dataset_name}_queries.fasta")
    if not os.path.exists(queries_fasta):
        queries = extract_sequences(unique_pids, db_path, queries_fasta)
    else:
        print(f"Using existing queries file: {queries_fasta}")
        queries = read_fasta_chains(queries_fasta)
        
    if not queries:
        print("No queries available. Aborting.")
        return

    # 3. Predict
    out_dir = os.path.join(work_dir, f"results_{dataset_name}")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Search DB (only need to do once if we reuse)
    print("Searching database for homologs...")
    candidate_indices = search_database(queries, db_path, KMER_LENGTH, MAX_CANDIDATES)
    
    print("Predicting...")
    # Map PID to Query Index
    # Need to match extracted Chain headers back to simple PIDs from CSV
    # Chain.header might be 'sp|P12345|...'
    
    pid_to_query = {}
    for i, q in enumerate(queries):
        parts = q.header.split('|')
        if len(parts) > 1:
            pid_to_query[parts[1]] = i # P12345
        else:
            pid_to_query[q.header] = i
            
    # Run SIFT
    valid_aas = [chr(c+65) for c in range(26) if c not in [9, 14, 20]]
    
    tp = tn = fp = fn = 0
    missing_pred = 0
    
    for pid, variants in gt_data.items():
        if pid not in pid_to_query:
            print(f"Skipping {pid} (Sequence not found)")
            missing_pred += len(variants)
            continue
            
        q_idx = pid_to_query[pid]
        query = queries[q_idx]
        
        # Check if output exists
        name_part = query.header.split()[0] # e.g. sp|P26439|...
        # create_file_name sanitizes? let's rely on standard sift_predictions output info
        # It uses query.header.split()[0] usually.
        
        # Run SIFT if not done
        pred_file = os.path.join(out_dir, f"{name_part}.SIFTprediction")
        if not os.path.exists(pred_file):
            # Align & Predict
            # (Simplified: we need logic similar to main.py/predict_custom.py)
            # Re-implementing simplified alignment loop here is redundant but necessary
            # unless we refactor. For this task, I'll assume we can call `sift_predictions`.
            # We need alignment output first.
            pass 
        
    # To run SIFT efficiently, we should batch process like predict_custom.py
    # Re-using the logic:
    print("Running alignment and SIFT...")
    
    # ... (Alignment Logic setup) ...
    db_index = SeqIO.index(db_path, "fasta")
    db_keys = list(db_index.keys())
    aligner = create_aligner("SW", 10, 1, "BLOSUM62")
    
    for i, query in enumerate(queries):
        name_part = query.header.split()[0]
        pred_file = os.path.join(out_dir, f"{name_part}.SIFTprediction")
        
        if os.path.exists(pred_file): continue # Skip if already done
        
        # Create hits
        hits = []
        indices = candidate_indices[i]
        if indices:
            for db_idx in indices:
                try:
                    record = db_index[db_keys[db_idx]]
                    # Simplified alignment (no exclusions for benchmarks usually, or standard?)
                    # Standard SIFT includes everything filtered by ID.
                    # We use standard params: ID=90, Cutoff=0.05
                    
                    # ... Alignment code ...
                    # (Copying concise version)
                    q_s = query.sequence # assume clean?
                    r_s = str(record.seq)
                    alns = aligner.align(q_s, r_s)
                    if alns:
                        aln = alns[0]
                        start_q = aln.coordinates[0, 0]
                        seq_chars = ['X'] * start_q
                        for q_c, t_c in zip(aln[0], aln[1]):
                            if q_c != '-': seq_chars.append(t_c if t_c != '-' else '-')
                        rem = len(query.sequence) - len(seq_chars)
                        if rem > 0: seq_chars.extend(['X'] * rem)
                        final_seq = "".join(seq_chars)[:len(query.sequence)]
                        hits.append(Chain(db_keys[db_idx], final_seq))
                except: pass
                
        # Predict
        sift_predictions(hits, query, ".", 90, out_dir, 0.05)

    # 4. Score
    print("Scoring...")
    for pid, variants in gt_data.items():
        if pid not in pid_to_query: continue
        q_idx = pid_to_query[pid]
        query = queries[q_idx]
        name_part = query.header.split()[0]
        pred_file = os.path.join(out_dir, f"{name_part}.SIFTprediction")
        
        # Get positions to parse
        positions = {v['Position'] for v in variants}
        scores_map = parse_sift_file(pred_file, positions)
        
        for v in variants:
            pos = v['Position']
            alt = v['Alt']
            label = v['Label'] # Deleterious / Tolerated
            
            if pos in scores_map:
                row = scores_map[pos]
                try:
                    col_idx = valid_aas.index(alt)
                    score = row[col_idx]
                    
                    is_del_pred = score < 0.05
                    is_del_lbl = (label == 'Deleterious')
                    
                    if is_del_lbl:
                        if is_del_pred: tp+=1
                        else: fn+=1
                    else:
                        if is_del_pred: fp+=1
                        else: tn+=1
                except: missing_pred +=1
            else:
                missing_pred += 1

    # Metrics
    total = tp+tn+fp+fn
    acc = (tp+tn)/total if total else 0
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    f1 = 2*prec*sens/(prec+sens) if (prec+sens) else 0
    
    print(f"\nRESULTS for {dataset_name}:")
    print(f"Total Variants Evaluated: {total}")
    print(f"Selected Metrics:")
    print(f"  Accuracy:    {acc:.2%}")
    print(f"  Sensitivity: {sens:.2%}")
    print(f"  Specificity: {spec:.2%}")
    print(f"  F1 Score:    {f1:.4f}")
    if missing_pred > 0:
        print(f"Warning: {missing_pred} variants could not be scored (mutant AA not in matrix or file missing).")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "uniprot_sprot.fasta")
    
    # HumDiv
    evaluate("HumDiv", os.path.join(base_dir, "humdiv_GroundTruth.csv"), db_path, base_dir)
    
    # HumVar
    evaluate("HumVar", os.path.join(base_dir, "humvar_GroundTruth.csv"), db_path, base_dir)

if __name__ == "__main__":
    main()
