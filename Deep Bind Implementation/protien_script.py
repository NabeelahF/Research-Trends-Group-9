import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE (Must match training)
# ==========================================
class DeepBindFinal(nn.Module):
    def __init__(self, num_filters=128, kernel_size=12, hidden_units=32):
        super(DeepBindFinal, self).__init__()
        self.conv = nn.Conv1d(20, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_filters, hidden_units)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_protein(sequence, model_path='deepbind_protein_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Load Model
    model = DeepBindFinal().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        return "Error: Model file not found. Please save 'deepbind_protein_model.pth' first."

    # B. Preprocess Input (One-Hot Encoding)
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    char_to_int = {c: i for i, c in enumerate(alphabet)}

    # Create empty matrix (20, Length)
    # Note: We use the length of the input sequence dynamically
    seq_len = len(sequence)
    one_hot = np.zeros((20, seq_len), dtype=np.float32)

    for i, char in enumerate(sequence):
        if char in char_to_int:
            idx = char_to_int[char]
            one_hot[idx, i] = 1.0

    # Convert to Tensor and add Batch dimension: (1, 20, Length)
    tensor_input = torch.tensor(one_hot).unsqueeze(0).to(device)

    # C. Predict
    with torch.no_grad():
        logit = model(tensor_input).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    # D. Result
    label = "Pathogenic" if prob > 0.5 else "Benign"
    return f"Prediction: {label} (Probability: {prob:.4f})"

# ==========================================
# 3. RUN IT
# ==========================================
if __name__ == "__main__":
    print("--- Protein DeepBind Predictor ---")
    print("Type 'exit' to stop the program.")
    
    while True:
        # Get input from user
        user_input = input("\nEnter a Protein Sequence: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue

        # Run Prediction
        try:
            result = predict_protein(user_input)
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")