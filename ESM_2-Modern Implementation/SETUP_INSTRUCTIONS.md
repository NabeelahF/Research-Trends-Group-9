# Setup & Execution Guide (From Scratch)


Follow these steps exactly to run the **Modern Approach (ESM-2)** implementation.

## 1. Create a Fresh Python Environment
It is best to use **Conda** to manage dependencies and valid Python versions.

**Open your Terminal / Command Prompt and run:**

```bash
# 1. Create a new environment named 'esm_env' with Python 3.9
conda create -n esm_env python=3.9 -y

# 2. Activate the environment
conda activate esm_env
```
## 2. Install PyTorch with GPU Support
You need a version of PyTorch that matches your GPU's CUDA version.
*   **For most Cloud GPUs (H100/A100):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(If you are on CPU only, just run `pip install torch`)*

## 3. Install Project Dependencies
Navigate to the project folder (`Modern_Approach`) used in this project.

```bash
cd "d:\IIT\4th year\RT\CW_implementations\Modern_Approach"

# Install the rest of the libraries
pip install -r requirements.txt
```

---

## 4. Prepare Data
1. Ensure `humsavar_extracted.csv` is in the directory (dataset).
2. Run preprocessing:
   ```bash
   python preprocess_humsavar.py
   ```
   - This fetches sequences from UniProt (cached) and creates `humsavar_cleaned.csv`.

---

## 4. Verify Installation
Run the architecture script to make sure PyTorch is working and the code is valid.

```bash
python my_esm2.py
```
**Success:** You should see `Success! The architecture is valid.` printed at the end.

---

## 5. Run the Experiments (The "Results")

### A. Zero-Shot Experiments (Context & Masking)
This downloads the model (2.5GB) and generates the data for your report.
```bash
python run_experiments.py
```
**Output:** Look for `experiment_context_sensitivity.png` in the folder.

### B. 3D Structure Visualization
This generates the PDB file.
```bash
python run_folding.py
```
**Output:** Look for `hbb_structure.pdb`.

### C. Fine-Tuning (Optional but Recommended)
*   **Prerequisite:** Ensure you have `humsavar_cleaned.csv` in the folder.
*   *If you don't have it, the script runs on dummy data (bad accuracy).*
```bash
python fine_tune.py
```
**Output:** Displays Accuracy/AUC per epoch.

---

## Troubleshooting
*   **OOM (Out Of Memory):** If `run_experiments.py` crashes on the GPU, open the file and change `MODEL_NAME` to a smaller model like `"facebook/esm2_t30_150M_UR50D"`.
*   **Missing Biotite:** If folding fails, run `pip install biotite`.
