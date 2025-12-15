import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from my_esm2 import MyESM2
from load_weights import load_official_weights
from fine_tune import ESM2Classifier, HumSaVarDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# =============================================================================
# Configuration
# =============================================================================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "final_checkpoints/final_model.pth"
DATA_FILE = "humsavar_cleaned.csv"
BATCH_SIZE = 16

print(f"🔬 Comprehensive Model Evaluation")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}\n")

# =============================================================================
# Load Model
# =============================================================================
print("Loading model...")
base_model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
model = ESM2Classifier(base_model).to(DEVICE)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("✅ Checkpoint loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("Using untrained model (results will be meaningless)\n")

model.eval()

# =============================================================================
# Load Test Data
# =============================================================================
print("Loading test dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load only 20,000 samples (matching training size)
dataset = HumSaVarDataset(DATA_FILE, tokenizer, debug_sample=20000)

# Use 20% as test set (same split as fine_tune.py)
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
_, test_dataset = torch.utils.data.random_split(
    dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Total dataset size: {len(dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

# =============================================================================
# Run Inference
# =============================================================================
print("Running inference on test set...")
all_preds = []
all_probs = []
all_labels = []

from tqdm import tqdm

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = batch['input_ids'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        logits = model(inputs).squeeze()
        probs = torch.sigmoid(logits)
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = (all_probs > 0.5).astype(int)

print("✅ Inference complete!\n")

# =============================================================================
# Generate Metrics
# =============================================================================
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(
    all_labels, 
    all_preds, 
    target_names=['Benign', 'Pathogenic'],
    digits=4
)
print(report)

# Save report to file
with open('evaluation_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("COMPREHENSIVE EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    f.write(f"Test Samples: {len(test_dataset)}\n\n")
    f.write("="*70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(report)
    f.write("\n")

# =============================================================================
# Create Visualizations
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ESM-2 Model Evaluation Metrics', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Benign', 'Pathogenic'],
            yticklabels=['Benign', 'Pathogenic'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# Add percentages
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        axes[0, 0].text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(alpha=0.3)

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
avg_precision = average_precision_score(all_labels, all_probs)
axes[1, 0].plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].legend(loc="lower left")
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])

# 4. Prediction Distribution
axes[1, 1].hist(all_probs[all_labels == 0], bins=50, alpha=0.6, label='Benign', color='green')
axes[1, 1].hist(all_probs[all_labels == 1], bins=50, alpha=0.6, label='Pathogenic', color='red')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Saved evaluation_metrics.png")

# =============================================================================
# Additional Metrics
# =============================================================================
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score

mcc = matthews_corrcoef(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)

print("\n" + "="*70)
print("ADDITIONAL METRICS")
print("="*70)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Append to report file
with open('evaluation_report.txt', 'a') as f:
    f.write("="*70 + "\n")
    f.write("ADDITIONAL METRICS\n")
    f.write("="*70 + "\n")
    f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")

print("\n✅ Evaluation complete!")
print(f"📊 Metrics visualization saved to: evaluation_metrics.png")
print(f"📄 Detailed report saved to: evaluation_report.txt")
