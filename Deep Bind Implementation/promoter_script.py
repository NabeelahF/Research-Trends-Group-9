import torch
import torch.nn as nn

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE (Must match training)
# ==========================================
class DeepPromoterModel(nn.Module):
    def __init__(self):
        super(DeepPromoterModel, self).__init__()
        # Matches your training: Embedding(5, 16)
        self.embedding = nn.Embedding(5, 16, padding_idx=0)
        self.conv = nn.Conv1d(16, 128, kernel_size=15)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) # Swap dimensions for Conv1d
        x = self.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_promoter(sequence, model_path='deeppromoter_dna_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Load Model
    model = DeepPromoterModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        return "Error: Model file not found. Please save 'deeppromoter_dna_model.pth' first."

    # B. Preprocess Input (Integer Encoding)
    vocab = "ACGT"
    dna_to_int = {n: i+1 for i, n in enumerate(vocab)}

    clean_seq = sequence.upper().strip()
    indices = [dna_to_int.get(n, 0) for n in clean_seq]

    # Pad to 300 (Matches your training MAX_LEN)
    MAX_LEN = 300
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]

    # Convert to Tensor and add Batch dimension: (1, 300)
    tensor_input = torch.tensor(indices).long().unsqueeze(0).to(device)

    # C. Predict
    with torch.no_grad():
        logit = model(tensor_input).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    # D. Result
    label = "Promoter" if prob > 0.5 else "Non-Promoter"
    return f"Prediction: {label} (Probability: {prob:.4f})"

# ==========================================
# 3. RUN IT
# ==========================================
if __name__ == "__main__":
    print("--- DeepPromoter DNA Predictor ---")
    print("Type 'exit' to stop the program.")

    while True:
        # Get input from user
        user_input = input("\nEnter a DNA Sequence (ACGT): ").strip()

        # Check if user wants to quit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Run Prediction
        try:
            # We call the specific function for this script: predict_promoter
            result = predict_promoter(user_input)
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")