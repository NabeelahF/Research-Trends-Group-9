import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer

from my_esm2 import MyESM2
from load_weights import load_official_weights

# =============================================================================
# Setup
# =============================================================================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 5
LEARNING_RATE = 1e-4

print(f"Initializing Fine-Tuning on {DEVICE}...")

# =============================================================================
# 1. The Classifier Head
# =============================================================================
class ESM2Classifier(nn.Module):
    def __init__(self, esm2_model, hidden_dim=1280):
        super().__init__()
        self.esm2 = esm2_model
        # Freeze the ESM-2 backbone to save memory and prevent overfitting
        for param in self.esm2.parameters():
            param.requires_grad = False
            
        # The Trainable Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
            # Removed Sigmoid! BCEWithLogitsLoss expects raw logits.
        )

    def forward(self, tokens):
        # Get embeddings from ESM-2
        x = self.esm2.embed_tokens(tokens)
        for layer in self.esm2.layers:
            x = layer(x)
        x = self.esm2.emb_layer_norm_after(x)
        
        # Mean Pooling
        mask = (tokens != 1).unsqueeze(-1).float() # 1 is padding idx
        x = x * mask
        embeddings = x.sum(dim=1) / mask.sum(dim=1)
        
        return self.classifier(embeddings)

# =============================================================================
# 2. Dataset Loader
# =============================================================================
class HumSaVarDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=1022, debug_sample=False):
        try:
            self.df = pd.read_csv(csv_file)
            # size=self.df.shape[0]
            size = 20000
            if debug_sample:
                print(f"⚠️ DEBUG MODE: Using only first {size} samples for speed!")
                self.df = self.df.head(size)
        except FileNotFoundError:
            self.df = pd.DataFrame({
                'sequence': ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"] * 100,
                'label': np.random.randint(0, 2, 100)
            })
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = self.df['label'].values # Expose locally for counting

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence']
        label = float(row['label'])
        
        tokens = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=self.max_len, padding="max_length")
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


# =============================================================================
# 3. Training Loop
# =============================================================================
def train():
    # Load Model
    base_model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
    base_model = load_official_weights(base_model, MODEL_NAME)
    
    model = ESM2Classifier(base_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Data (DEBUG MODE ENABLED)
    dataset = HumSaVarDataset("humsavar_cleaned.csv", tokenizer, debug_sample=True)
    
    # Calculate Class Weights
    labels = dataset.labels
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    print(f"Dataset Stats: Pathogenic (1): {num_pos}, Benign (0): {num_neg}")
    
    # We want to penalize errors on the Minority Class (0) more?
    # No, usually we upweight the Positive class if it's minority.
    # Here Positive (1) is Majority.
    # So we want to DOWN weight Positive class.
    # pos_weight = num_neg / num_pos
    pos_weight_val = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    print(f"Using Weighted Loss with pos_weight={pos_weight_val:.4f} (Down-weighting Majority Class 1)")
    
    # Split Train/Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # Critical Change: BCEWithLogitsLoss accepts pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create checkpoints dir if not exists
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n--- Starting Fine-Tuning ---")
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs = batch['input_ids'].to(DEVICE)
            labels = batch['label'].to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Returns logits now
            loss = criterion(outputs, labels) # BCEWithLogitsLoss computes Sigmoid internally
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        
        # Evaluation
        val_auc = evaluate(model, test_loader)
        
        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"🔥 New Best AUC! Saving model to 'checkpoints/best_model.pth'...")
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

    # Save Final Model
    print("Saving final model to 'checkpoints/final_model.pth'...")
    torch.save(model.state_dict(), "checkpoints/final_model.pth")

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            logits = model(inputs).squeeze()
            probs = torch.sigmoid(logits)    
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics Analysis
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Prediction Stats
    avg_prob = np.mean(all_preds)
    ratio_pos = np.mean(all_preds > 0.5)
    print(f"Validation Stats: Avg Prob={avg_prob:.4f} | Predicted Pathogenic (>0.5)={ratio_pos*100:.1f}%")
    
    # 2. Find Optimal Threshold
    best_acc = 0
    best_thresh = 0.5
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    for t in thresholds:
        preds_bin = (all_preds > t).astype(int)
        acc = accuracy_score(all_labels, preds_bin)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5 
        
    print(f"Validation AUC: {auc:.4f}")
    print(f"Max Accuracy: {best_acc:.4f} at Threshold {best_thresh:.2f}")
    
    # Standard 0.5 Accuracy for comparison
    std_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    print(f"Standard Acc (0.5): {std_acc:.4f}\n")
    
    return auc

if __name__ == "__main__":
    train()
