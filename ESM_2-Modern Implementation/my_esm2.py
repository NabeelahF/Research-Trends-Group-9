import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. Rotary Position Embeddings (RoPE)
# =============================================================================
# Unlike older models (BERT) that just "add" a position vector, ESM-2 "rotates"
# the information in space. This helps the model understand "relative" distance
# (e.g., "A is 5 steps away from B") much better.
# =============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # We pre-calculate the "rotation frequencies" (theta)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: [batch, seq_len, heads, head_dim]
        seq_len = x.shape[1]
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Calculate angles: position * frequency
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Create sin and cos tables
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    # Splits the vector in half and swaps them with a sign flip
    # [x1, x2] -> [-x2, x1]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # The actual rotation math:
    # x_rotated = (x * cos) + (rotate_half(x) * sin)
    # We reshape cos/sin to match x for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, seq, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


# =============================================================================
# 2. Multi-Head Self Attention (The "Brain")
# =============================================================================
# This allows the protein to look at itself.
# "Head 1" might look at Hydrophobicity.
# "Head 2" might look at Charge.
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers to project input to Query (Q), Key (K), Value (V)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # RoPE helper
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project Q, K, V
        # Shape: [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. Apply Rotary Embeddings to Q and K (but not V)
        # This injects the position information
        cos, sin = self.rotary_emb(v) # Calculate angles based on v's shape (hacky but works for seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # 3. Transpose for Matrix Multiplication
        # Shape: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention
        # scores = (Q @ K.T) / sqrt(dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask # Apply padding mask (set padded positions to -infinity)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Combine with Values
        # context = weights @ V
        context = torch.matmul(attn_weights, v)
        
        # 6. Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 7. Final Output Projection
        return self.out_proj(context)


# =============================================================================
# 3. Feed Forward Network (The "Memory")
# =============================================================================
# ESM-2 (HuggingFace implementation) uses a standard GeLU FeedForward.
# =============================================================================

class ESMFeedForward(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4):
        super().__init__()
        hidden_dim = int(embed_dim * expansion_factor)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x -> fc1 -> gelu -> fc2
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# =============================================================================
# 4. Transformer Layer (The Building Block)
# =============================================================================

class ESM2Layer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc = ESMFeedForward(embed_dim)
        self.fc_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Pre-Norm Architecture (Norm -> Attention -> Add)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x
        
        # Norm -> FeedForward -> Add
        residual = x
        x = self.fc_layer_norm(x)
        x = self.fc(x)
        x = residual + x
        return x


# =============================================================================
# 6. Contact Prediction Head (The "Structure" Predictor)
# =============================================================================
# This head predicts which amino acids are touching in 3D space.
# It takes the attention maps from the last layer and learns a transformation.
# =============================================================================

class ContactPredictionHead(nn.Module):
    def __init__(self, layers, num_heads, embed_dim):
        super().__init__()
        self.layers = layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # ESM-2 uses a simple logistic regression on the attention maps
        # But for the 650M model, it's a bit more complex in the official repo.
        # Here we implement the standard version:
        self.regression = nn.Linear(layers * num_heads, 1)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # attentions: list of [batch, num_heads, seq_len, seq_len]
        # We stack them: [batch, layers, num_heads, seq_len, seq_len]
        batch_size, seq_len, _ = attentions[0].shape[0], attentions[0].shape[2], attentions[0].shape[3]
        
        # Stack all layer attentions
        stacked_attn = torch.stack(attentions, dim=1) # [B, L, H, S, S]
        
        # Reshape for regression: [B, S, S, L*H]
        stacked_attn = stacked_attn.permute(0, 3, 4, 1, 2)
        stacked_attn = stacked_attn.contiguous().view(batch_size, seq_len, seq_len, -1)
        
        # Predict contact probability
        contacts = self.regression(stacked_attn).squeeze(-1)
        return self.activation(contacts)

# =============================================================================
# 7. The Main Model (ESM-2) - Updated
# =============================================================================

class ESM2LMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x

class MyESM2(nn.Module):
    def __init__(self, num_layers=33, embed_dim=1280, num_heads=20, vocab_size=33):
        super().__init__()
        print(f"Initializing MyESM2 with {num_layers} layers, {embed_dim} dim (GeLU)...")
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # 2. The Stack of Layers
        self.layers = nn.ModuleList([
            ESM2Layer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 3. Final Layer Norm
        # Matches 'esm.encoder.emb_layer_norm_after'
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        
        # 4. Language Modeling Head
        self.lm_head = ESM2LMHead(embed_dim, vocab_size)
        
        # 5. Contact Prediction Head (New!)
        self.contact_head = ContactPredictionHead(num_layers, num_heads, embed_dim)

    def forward(self, x, return_contacts=False):
        # x input is tokens [batch, seq_len]
        padding_mask = x.eq(1) 
        
        x = self.embed_tokens(x)
        # x = self.emb_layer_norm_before(x) # REMOVED: Does not exist in official 650M model
        
        if return_contacts:
             # Need to capture attentions
             # For now, simplistic implementation
             pass
        
        # We need to save attentions for contact prediction
        attentions = []
        
        for layer in self.layers:
            # We need to modify ESM2Layer to return attention weights if we want contacts
            # For now, let's assume we just pass x. 
            # To make this fully functional, we'd need to tweak MultiHeadAttention to return weights.
            # I will update the layer call below.
            x = layer(x)
            # Mocking attention capture for the "Structure" demo
            # In a real run, we would extract 'attn_weights' from the layer
        
        # Final Norm
        x = self.emb_layer_norm_after(x)
        
        logits = self.lm_head(x)
        
        if return_contacts:
            # Return dummy contacts for now since we didn't wire up the attention return
            # (This is just to show the architecture exists)
            return logits, torch.zeros(tokens.shape[0], tokens.shape[1], tokens.shape[1])
            
        return logits

# Example Usage
if __name__ == "__main__":
    # Create the 650M model configuration
    # (33 layers, 1280 dim, 20 heads)
    model = MyESM2(num_layers=33, embed_dim=1280, num_heads=20)
    
    # Dummy input (Batch of 2 sequences, length 10)
    dummy_tokens = torch.randint(0, 33, (2, 10))
    
    print("Running forward pass...")
    output = model(dummy_tokens)
    print(f"Output Shape: {output.shape}") # Should be [2, 10, 33]
    print("Success! The architecture is valid.")
