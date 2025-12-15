import pandas as pd
import requests
import json
import re
import os
import time
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
INPUT_FILE = "humsavar_extracted.csv"
OUTPUT_FILE = "humsavar_cleaned.csv"
CACHE_FILE = "uniprot_id_cache.json"

AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*' # Stop codon
}

# =============================================================================
# Batch API Fetcher
# =============================================================================
def fetch_sequences_batch(accessions_list):
    """
    Fetches sequences for a list of accessions.
    Returns a dict {id: sequence}
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    # We can use the 'accession:ID OR accession:ID' query
    # Chunks of 50 avoid URL length limits
    
    id_map = {}
    
    chunk_size = 50
    for i in tqdm(range(0, len(accessions_list), chunk_size), desc="Fetching batches"):
        chunk = accessions_list[i:i+chunk_size]
        
        # Build Query: (accession:P1 OR accession:P2 ...)
        query = " OR ".join([f"accession:{acc}" for acc in chunk])
        query = f"({query})"
        
        params = {
            "query": query,
            "fields": "accession,sequence",
            "format": "json",
            "size": 500 # Should cover the chunk
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            for res in results:
                primary_acc = res['primaryAccession']
                seq = res['sequence']['value']
                id_map[primary_acc] = seq
                
        except Exception as e:
            print(f"Error fetching batch {i}: {e}")
            
        time.sleep(0.2) # Politeness
        
    return id_map

def parse_aa_change(change_str):
    pattern = r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"
    match = re.search(pattern, change_str)
    if match:
        wt_3 = match.group(1)
        pos = int(match.group(2))
        mut_3 = match.group(3)
        if wt_3 in AA_MAP and mut_3 in AA_MAP:
            return pos, AA_MAP[wt_3], AA_MAP[mut_3]
    return None

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found!")
        return

    # Filter Labels
    valid_categories = ['LB/B', 'LP/P']
    df = df[df['Variant category'].isin(valid_categories)].copy()
    df['label'] = df['Variant category'].apply(lambda x: 0 if x == 'LB/B' else 1)
    
    print(f"Filtered Row Count: {len(df)}")
    
    # 1. Prepare Cache
    unique_ids = df['Swiss-Prot AC'].unique().tolist()
    print(f"Found {len(unique_ids)} unique Proteins.")
    
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            
    # Find missing IDs
    missing_ids = [uid for uid in unique_ids if uid not in cache]
    print(f"Need to fetch {len(missing_ids)} new sequences...")
    
    if missing_ids:
        new_sequences = fetch_sequences_batch(missing_ids)
        cache.update(new_sequences)
        
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
            
    print("All sequences ready. Processing variants...")
    
    processed_data = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        uniprot_id = row['Swiss-Prot AC']
        aa_change = row['AA change']
        label = row['label']
        gene_name = row['Main gene name']

        parsed = parse_aa_change(aa_change)
        if not parsed: continue
        pos, wt, mut = parsed
        
        seq = cache.get(uniprot_id)
        if not seq: continue
        
        # Validation
        if pos > len(seq): continue
        if seq[pos-1] != wt: continue
        
        processed_data.append({
            'gene': gene_name,
            'sequence': seq,
            'label': label,
            'name': aa_change,
            'position': pos,
            'wt': wt,
            'mut': mut
        })
        
    result_df = pd.DataFrame(processed_data)
    print("\n--- Processing Complete ---")
    print(result_df['label'].value_counts())
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result_df)} clean variants to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
