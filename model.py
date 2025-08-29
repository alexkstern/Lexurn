import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap 

def create_causal_mask(seq_len, device=None):
    # torch.triu creates upper triangular matrix with 1s above diagonal
    # diagonal=1 means start masking from diagonal+1 (exclude diagonal itself)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  # Convert to boolean: True = masked, False = allowed

# use swiglu and no dropout ( no one used it anymore )
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inner_dim = 4 * d_model
        self.up_proj = nn.Linear(d_model, inner_dim)
        self.gate_proj = nn.Linear(d_model, inner_dim)
        self.gelu = nn.GELU(approximate="none")
        self.down_proj = nn.Linear(inner_dim, d_model)

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        x = self.gelu(up) * gate
        x = self.down_proj(x)
        return x


class myattention(nn.Module):
    def __init__(self, d_model):
        inner_dim = d_model * 4
        self.query = nn.Linear(d_model, inner_dim)
        self.key = nn.Linear(d_model, inner_dim)
        self.value = nn.Linear(d_model, inner_dim)

    def forward(self, x):
        self.query(x)
        self.key(x)
        pass


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )  # would remove dropout
        self.dropout = nn.Dropout(0.1)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x, mask):
        # Attention block with residual
        normed = self.ln1(x)
        attn_output, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_output)  # would remove dropout

        # MLP block with residual
        normed = self.ln2(x)
        mlp_output = self.mlp(normed)
        x = x + mlp_output
        return x


class UrnTransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size=4, d_model=128, n_layers=4, n_heads=4, context_len=8, lex=False
    ):
        super().__init__()

        self.lex = lex
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_len = context_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if self.lex:
            # Freeze token embeddings for lexical invariance experiments
            self.token_embedding.weight.requires_grad = False  # not using this anyways in lex mode, remove this line
            

        self.pos_embedding = nn.Embedding(context_len, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads) for i in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        #

    def forward(self, x):
        batch_size, seq_len = x.shape

        if self.lex:
            # 1) sample per-sequence tables
            embeddings = torch.randn(
                batch_size, self.vocab_size, self.d_model, device=x.device
            )
            # 2) batch-embed with vmap
            x = vmap(
                lambda emb, idx: F.embedding(idx, emb),
                in_dims=(0, 0)
            )(embeddings, x)
        else:
            x = self.token_embedding(x)

        pos_ids = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(pos_ids)

        # Add positional embeddings
        x = x + pos_emb

        # Create causal mask for current sequence length
        causal_mask = create_causal_mask(seq_len, device=x.device)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.ln_final(x)
        
        if self.lex:
            logits = torch.matmul(x, embeddings.transpose(-2, -1))
        else:
            logits = self.output_proj(x)
        
        return logits


if __name__ == "__main__":
    from build_dataset import generate_dataset
    
    print("=== Testing UrnTransformerDecoder ===")
    
    # Generate test data
    sequences, urns, task_ids = generate_dataset(
        context_len=8,
        n_tasks=2,
        n_colors=4,
        n_steps=3,
        seed=42
    )
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Urns shape: {urns.shape}")
    
    # Test 1: Normal model
    print(f"\n=== Test 1: Normal Model ===")
    model_normal = UrnTransformerDecoder(
        vocab_size=4,
        d_model=64,
        n_layers=2,
        n_heads=2,
        context_len=8,
        lex=False
    )
    
    logits_normal = model_normal(sequences)
    print(f"Output shape: {logits_normal.shape}")
    
    # Test 2: Lexical model
    print(f"\n=== Test 2: Lexical Model ===")
    model_lex = UrnTransformerDecoder(
        vocab_size=4,
        d_model=64,
        n_layers=2,
        n_heads=2,
        context_len=8,
        lex=True
    )
    
    logits_lex = model_lex(sequences)
    print(f"Lexical output shape: {logits_lex.shape}")
    
    print(f"\n=== All tests completed successfully! ===")