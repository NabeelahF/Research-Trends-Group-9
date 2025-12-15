import torch
import torch.nn as nn
from my_esm2 import MyESM2
from load_weights import load_official_weights
from fine_tune import ESM2Classifier
from transformers import AutoTokenizer

# =============================================================================
# Setup
# =============================================================================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "final_checkpoints\\final_model.pth"

print(f"Initializing Inference Pipeline on {DEVICE}...")

def update_html_viewer(seq_length, prediction):
    """Update the HTML viewer with current prediction results."""
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Protein Structure Viewer - ESMFold</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        #viewer {{
            width: 100%;
            height: 600px;
            position: relative;
            background: #000;
        }}
        .controls {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }}
        .controls h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .button-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        button:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .info {{
            padding: 20px 30px;
            background: #fff;
            border-top: 1px solid #dee2e6;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .info-item strong {{
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Protein Structure Viewer</h1>
            <p>ESMFold Prediction - Custom Sequence</p>
        </div>
        
        <div id="viewer"></div>
        
        <div class="controls">
            <h3>Visualization Controls</h3>
            <div class="button-group">
                <button onclick="setStyle('cartoon')">Cartoon</button>
                <button onclick="setStyle('stick')">Stick</button>
                <button onclick="setStyle('sphere')">Sphere</button>
                <button onclick="setStyle('line')">Line</button>
                <button onclick="setStyle('cross')">Cross</button>
            </div>
            <div class="button-group">
                <button onclick="setColor('spectrum')">Spectrum</button>
                <button onclick="setColor('chain')">By Chain</button>
                <button onclick="setColor('ss')">Secondary Structure</button>
                <button onclick="setColor('white')">White</button>
            </div>
            <div class="button-group">
                <button onclick="viewer.zoomTo()">Reset View</button>
                <button onclick="viewer.spin(true)">Spin</button>
                <button onclick="viewer.spin(false)">Stop Spin</button>
            </div>
        </div>
        
        <div class="info">
            <h3>Structure Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Model</strong>
                    ESMFold v1
                </div>
                <div class="info-item">
                    <strong>Source</strong>
                    ESM Atlas API
                </div>
                <div class="info-item">
                    <strong>Sequence Length</strong>
                    {seq_length} residues
                </div>
                <div class="info-item">
                    <strong>Prediction</strong>
                    {prediction}
                </div>
            </div>
        </div>
    </div>

    <script>
        let viewer = null;
        let currentStyle = 'cartoon';
        let currentColor = 'spectrum';

        function initViewer() {{
            viewer = $3Dmol.createViewer('viewer', {{
                backgroundColor: 'black'
            }});
            
            fetch('predicted_structure.pdb')
                .then(response => response.text())
                .then(pdbData => {{
                    viewer.addModel(pdbData, 'pdb');
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    viewer.zoomTo();
                    viewer.render();
                }})
                .catch(error => {{
                    console.error('Error loading PDB:', error);
                    alert('Error loading structure file. Make sure predicted_structure.pdb is in the same directory.');
                }});
        }}

        function setStyle(style) {{
            currentStyle = style;
            let styleObj = {{}};
            styleObj[style] = {{color: currentColor}};
            viewer.setStyle({{}}, styleObj);
            viewer.render();
        }}

        function setColor(color) {{
            currentColor = color;
            let styleObj = {{}};
            styleObj[currentStyle] = {{color: color}};
            viewer.setStyle({{}}, styleObj);
            viewer.render();
        }}

        window.onload = initViewer;
    </script>
</body>
</html>'''
    
    with open('visualize_structure.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    print("✅ Updated visualization HTML")

def run_inference(sequence, save_pdb=True):
    # 1. Initialize & Load Fine-Tuned Model
    print("\n--- 1. Pathogenicity Prediction ---")
    
    # Init Backbone
    base_model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
    # Note: We technically don't need to download official weights again if the checkpoint has them.
    # But usually checkpoints only save state_dict. 
    # If the checkpoint is full model, we just load it.
    # Let's assume checkpoint contains everything (backbone + head).
    
    model = ESM2Classifier(base_model).to(DEVICE)
    
    try:
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Checkpoint '{CHECKPOINT_PATH}' not found! Using initialized weights (Garbage output warning).")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        # Fallback: Load official backbone at least
        print("Falling back to official backbone weights...")
        base_model = load_official_weights(base_model, MODEL_NAME)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Predict
    tokens = tokenizer(sequence, return_tensors="pt")['input_ids'].to(DEVICE)
    with torch.no_grad():
        logits = model(tokens).squeeze()
        prob = torch.sigmoid(logits).item()
        
    print(f"Sequence Length: {len(sequence)}")
    print(f"Pathogenicity Probability: {prob:.4f}")
    if prob > 0.5:
        print("Result: 🔴 PATHOGENIC")
        prediction_label = f"PATHOGENIC ({prob*100:.1f}%)"
    else:
        print("Result: 🟢 BENIGN")
        prediction_label = f"BENIGN ({prob*100:.1f}%)"

    # Update HTML visualization file
    update_html_viewer(len(sequence), prediction_label)

    # 2. Folding (Optional)
    if save_pdb:
        print("\n--- 2. Structure Prediction (ESMFold) ---")
        
        # Try local ESMFold first
        local_success = False
        try:
            import esm
            print("Attempting local ESMFold...")
            folding_model = esm.pretrained.esmfold_v1()
            folding_model = folding_model.eval().to(DEVICE)
            
            print("Folding sequence locally...")
            with torch.no_grad():
                output = folding_model.infer(sequence)
                
            pdb_filename = "predicted_structure.pdb"
            with open(pdb_filename, "w") as f:
                f.write(output)
            print(f"✅ Saved 3D structure to {pdb_filename}")
            local_success = True
            
        except ImportError as e:
            if "openfold" in str(e):
                # print(f"⚠️ Local ESMFold unavailable: 'openfold' not installed.")
                print(" Loading ESMFold API...")
            else:
                print(f"⚠️ ImportError: {e}")
                print(" Loading ESMFold API...")
        except Exception as e:
            print(f"⚠️ Local folding failed: {e}")
            print(" Loading ESMFold API...")
        
        # Fallback to ESMFold API
        if not local_success:
            try:
                import requests
                print("Querying ESMFold API (this may take 30-60 seconds)...")
                
                # Use ESMFold API endpoint
                api_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
                
                response = requests.post(api_url, data=sequence, headers={'Content-Type': 'text/plain'})
                
                if response.status_code == 200:
                    pdb_filename = "predicted_structure.pdb"
                    with open(pdb_filename, "w") as f:
                        f.write(response.text)
                    print(f"✅ Saved 3D structure from API to {pdb_filename}")
                else:
                    print(f"❌ API request failed with status {response.status_code}")
                    print(f"   Response: {response.text[:200]}")
                    
            except Exception as e:
                print(f"❌ API fallback also failed: {e}")
                print("   Structure prediction unavailable.")

if __name__ == "__main__":
    # Example: Hemoglobin Beta (HBB)
    # hbb_seq = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    hbb_seq = "WAESESLLKPLANVTLTCQARLETPDFQLFKNGVAQEPVHL"
    
    run_inference(hbb_seq)
