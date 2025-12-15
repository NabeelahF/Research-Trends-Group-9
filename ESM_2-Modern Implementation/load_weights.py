import torch
from my_esm2 import MyESM2
from transformers import AutoModelForMaskedLM

def load_official_weights(my_model, model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Downloads official weights from HuggingFace and maps them 
    to our custom architecture.
    """
    print(f"Downloading official weights for {model_name}...")
    model_hf = AutoModelForMaskedLM.from_pretrained(model_name)
    hf_state_dict = model_hf.state_dict()
    
    my_state_dict = my_model.state_dict()
    
    print("Mapping weights to custom architecture (Step 2)...")
    
    for hf_key, param in hf_state_dict.items():
        my_key = hf_key
        
        # 1. Global Embeddings
        if "esm.embeddings.word_embeddings" in my_key:
            my_key = my_key.replace("esm.embeddings.word_embeddings", "embed_tokens")
        elif "esm.embeddings.position_embeddings" in my_key:
            continue
        elif "esm.encoder.emb_layer_norm_after" in my_key:
            my_key = my_key.replace("esm.encoder.emb_layer_norm_after", "emb_layer_norm_after")
            
        # 2. Layers
        if "esm.encoder.layer." in my_key:
            my_key = my_key.replace("esm.encoder.layer.", "layers.")
            
            # Attention
            if "attention.self.query" in my_key:
                my_key = my_key.replace("attention.self.query", "self_attn.q_proj")
            elif "attention.self.key" in my_key:
                my_key = my_key.replace("attention.self.key", "self_attn.k_proj")
            elif "attention.self.value" in my_key:
                my_key = my_key.replace("attention.self.value", "self_attn.v_proj")
            elif "attention.output.dense" in my_key:
                my_key = my_key.replace("attention.output.dense", "self_attn.out_proj")
            elif "attention.LayerNorm" in my_key:
                my_key = my_key.replace("attention.LayerNorm", "self_attn_layer_norm")
                
            # FeedForward (GeLU)
            if "intermediate.dense" in my_key:
                my_key = my_key.replace("intermediate.dense", "fc.fc1")
            elif "output.dense" in my_key:
                my_key = my_key.replace("output.dense", "fc.fc2")
            elif "LayerNorm" in my_key and "attention" not in my_key:
                # Catches 'esm.encoder.layer.X.LayerNorm' which is the FC norm
                my_key = my_key.replace("LayerNorm", "fc_layer_norm")
                
        # 3. Heads
        if "lm_head.dense" in my_key:
             pass # 1:1 mapping now
        elif "lm_head.layer_norm" in my_key:
             pass # 1:1 mapping now
        elif "lm_head.decoder" in my_key:
             pass # 1:1 mapping now
        elif "lm_head.bias" in my_key:
             pass 

        # LOAD
        if my_key in my_state_dict:
            target_shape = my_state_dict[my_key].shape
            loaded_shape = param.shape
            
            if target_shape == loaded_shape:
                my_state_dict[my_key].copy_(param)
            else:
                print(f"Shape mismatch for {my_key}: Target {target_shape} vs Loaded {loaded_shape}")
        else:
            # print(f"Key {my_key} not found in custom model.")
            pass
            
    my_model.load_state_dict(my_state_dict)
    print("Weights loaded successfully!")
    return my_model

if __name__ == "__main__":
    # Test loading
    model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
    load_official_weights(model)
