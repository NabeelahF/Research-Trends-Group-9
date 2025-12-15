import torch
import numpy as np
import matplotlib.pyplot as plt
from my_esm2 import MyESM2
from load_weights import load_official_weights
from transformers import AutoTokenizer
import pandas as pd

# =============================================================================
# Setup
# =============================================================================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing Experiment Runner on {DEVICE}...")

# 1. Initialize Model
# 650M Config: 33 layers, 1280 dim, 20 heads
model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
model = load_official_weights(model, MODEL_NAME)
model.to(DEVICE)
model.eval()

# 2. Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================================================================
# Experiment 1: Context Sensitivity (Global vs Local)
# =============================================================================
def run_context_experiment(sequence, position, window_size=20):
    """
    Checks if the model's prediction changes when we crop the sequence.
    Hypothesis: A good model needs long-range context.
    """
    print("\n--- Experiment 1: Context Sensitivity ---")
    
    # Full Sequence Prediction
    tokens_full = tokenizer(sequence, return_tensors="pt")['input_ids'].to(DEVICE)
    with torch.no_grad():
        logits_full = model(tokens_full)
    
    # Get probability of the correct amino acid at 'position'
    # Note: Tokenizer adds <cls> at start, so index is position+1
    target_idx = position + 1
    target_token = tokens_full[0, target_idx]
    prob_full = torch.softmax(logits_full[0, target_idx], dim=0)[target_token].item()
    
    # Cropped Sequence Prediction
    start = max(0, position - window_size)
    end = min(len(sequence), position + window_size)
    seq_cropped = sequence[start:end]
    
    tokens_cropped = tokenizer(seq_cropped, return_tensors="pt")['input_ids'].to(DEVICE)
    with torch.no_grad():
        logits_cropped = model(tokens_cropped)
        
    # In cropped sequence, the target is roughly in the middle
    # We need to find the new index relative to the crop
    new_target_idx = (position - start) + 1
    prob_cropped = torch.softmax(logits_cropped[0, new_target_idx], dim=0)[target_token].item()
    
    print(f"Full Context Probability:    {prob_full:.4f}")
    print(f"Local Context Probability:   {prob_cropped:.4f}")
    print(f"Dependency on Long Range:    {abs(prob_full - prob_cropped):.4f}")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Full Context', 'Local Context'], [prob_full, prob_cropped], color=['#3498db', '#e74c3c'])
    plt.ylabel('Probability of Correct AA')
    plt.title(f'Impact of Long-Range Context (Position {position})')
    plt.ylim(0, 1.0)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
                
    plt.savefig("experiment_context_sensitivity.png")
    print("Saved plot to experiment_context_sensitivity.png")
    
    return prob_full, prob_cropped

# =============================================================================
# Experiment 2: Masking Robustness
# =============================================================================
def run_masking_experiment(sequence, position):
    """
    Checks if the model can fill in the blank.
    """
    print("\n--- Experiment 2: Masking Robustness ---")
    
    tokens = tokenizer(sequence, return_tensors="pt")['input_ids'].to(DEVICE)
    target_idx = position + 1
    original_token = tokens[0, target_idx].item()
    
    # Mask the position
    tokens[0, target_idx] = tokenizer.mask_token_id
    
    with torch.no_grad():
        logits = model(tokens)
        
    probs = torch.softmax(logits[0, target_idx], dim=0)
    top_5 = torch.topk(probs, 5)
    
    print(f"Original AA: {tokenizer.decode([original_token])}")
    print("Top 5 Predictions:")
    for i in range(5):
        aa = tokenizer.decode([top_5.indices[i].item()])
        p = top_5.values[i].item()
        print(f"  {aa}: {p:.4f}")
        
    rank = (probs > probs[original_token]).sum().item() + 1
    print(f"Rank of Truth: {rank}")
    return rank

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Load Real Data
    # Load Real Data
    csv_path = "humsavar_cleaned.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} variants from {csv_path}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        print("Please run 'preprocess_humsavar.py' to generate the data.")
        exit()

    # Run on the first 5 samples for demonstration
    # (Running on all 20k samples takes time, usually we show a few examples for the report)
    print("\n--- Running Experiments on Real Data (First 5 samples) ---")
    
    results = []
    
    for idx, row in df.head(5).iterrows():
        seq = row['sequence']
        pos_1_indexed = int(row['position']) # Ensure integer
        wt = row['wt']
        mut = row['mut']
        gene = row.get('gene', f"Variant_{idx}")
        
        print(f"\nProcessing {gene}: {wt}{pos_1_indexed}{mut}")
        
        # Experiment 1: Context Sensitivity
        # We need to ensure position is valid
        if pos_1_indexed > len(seq):
            print(f"Skipping: Position {pos_1_indexed} out of bounds for len {len(seq)}")
            continue
            
        # Note: run_context_experiment expects 0-indexed position internally? 
        # No, my function takes 1-indexed (human) and converts inside?
        # Let's check `run_context_experiment`:
        # "tokens_full[0, position+1]" -> It assumes input 'position' is 0-indexed relative to sequence?
        # Let's fix the call: Pass 0-indexed position to the function to be safe/consistent with Python.
        
        pos_0_indexed = pos_1_indexed - 1
        
        try:
            p_full, p_local = run_context_experiment(seq, pos_0_indexed)
            rank = run_masking_experiment(seq, pos_0_indexed)
            
            results.append({
                "gene": gene,
                "variant": f"{wt}{pos_1_indexed}{mut}",
                "p_full": p_full,
                "p_local": p_local,
                "mask_rank": rank
            })
        except Exception as e:
            print(f"Error processing {gene}: {e}")

    # Save summary results
    results_df = pd.DataFrame(results)
    print("\n--- Summary Results ---")
    print(results_df)
    results_df.to_csv("experiment_results_summary.csv", index=False)
    print("Saved summary to experiment_results_summary.csv")
