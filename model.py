import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import count_parameters, load_model_config
from torch.func import vmap 

def create_causal_mask(seq_len, device=None):
    # torch.triu creates upper triangular matrix with 1s above diagonal
    # diagonal=1 means start masking from diagonal+1 (exclude diagonal itself)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  # Convert to boolean: True = masked, False = allowed


# class MLP(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         inner_dim=4*d_model
#         self.up_proj=nn.Linear(d_model,inner_dim)
#         self.gelu=nn.GELU(approximate='none')
#         self.dropout = nn.Dropout(0.05)
#         self.down_proj=nn.Linear(inner_dim,d_model)

#     def forward(self, x):

#         x= self.up_proj(x)
#         x=self.gelu(x)
#         x=self.dropout(x)
#         x=self.down_proj(x)
#         return x
#


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

    # def forward(self, x):
    #     x = self.up_proj(x)
    #     gate = self.gate_proj(x)
    #     x = self.gelu(x) * gate
    #     x = self.down_proj(x)
    #     return x


# will work on this after
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


"""
#will work on this after
class mymultiheadattention(nn.Module):
    def __init__(self,d_model,n_heads):

        pass
    def forward(self,x):
        pass
"""


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
            self.token_embedding.weight.requires_grad = False  # you're not using this anyways in lex mode, yo ucan remove this line

        self.pos_embedding = nn.Embedding(context_len, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads) for i in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # # Lexical invariance: randomize token embeddings per sequence to force generalization
        # if self.lex:  # and self.training:
        #     # much simpler, the same (otherwise you can do permutation imo even better)
        #     embeddings = torch.randn(
        #         batch_size, self.vocab_size, self.d_model, device=x.device
        #     )
        #     x = F.embedding(x, embeddings)
        # else:
        #     # Normal mode: use learned token embeddings
        #     x = self.token_embedding(x)


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

        # Concatenate
        x = x + pos_emb

        # Create causal mask for current sequence length
        # This prevents each token from attending to future positions
        causal_mask = create_causal_mask(seq_len, device=x.device)

        # x=x+self.layer(x)
        for layer in (
            self.layers
        ):  # nn.ModuleList is a container, one has to manually iterate through it
            x = layer(x, causal_mask)  # Residual connection per layer
        x = self.ln_final(x)
        if self.lex:
            x = torch.matmul(x, embeddings.transpose(-2, -1))
        else:
            x = self.output_proj(
                x
            )  # cross entropy will later expect the raw un normalized logits
        return x


if __name__ == "__main__":
    # Test with config loading
    config_path = "configs/dummy.config"
    configs = load_model_config(config_path)

    print("Loaded configs:")
    for section, params in configs.items():
        print(f"  {section}: {params}")

    # Test regular model from config
    model_params = configs["model"].copy()
    model_params["lex"] = False  # Override for normal model test
    model = UrnTransformerDecoder(**model_params)

    test_input = torch.randint(low=0, high=4, size=(1, 8))
    test_input = test_input.repeat(2, 1)
    print(f"\nTest input: {test_input}")
    output = model(test_input)
    print(f"Output shape: {output.shape}")

    # Test with lexical invariance from config
    print("\nTesting with lexical invariance:")
    lex_model_params = configs["model"].copy()
    lex_model_params["lex"] = True  # Override for lex model test
    lex_model = UrnTransformerDecoder(**lex_model_params)
    lex_model.train()
    lex_output = lex_model(test_input)
    print(f"Lexical invariance output shape: {lex_output.shape}")

    # Parameter counts
    print("\n" + "=" * 50)
    print("Parameter counts:")

    total_n, trainable_n, frozen_n = count_parameters(model)
    print(
        f"Normal Model: {total_n:,} total, {trainable_n:,} trainable, {frozen_n:,} frozen"
    )

    total_l, trainable_l, frozen_l = count_parameters(lex_model)
    print(
        f"Lex Model: {total_l:,} total, {trainable_l:,} trainable, {frozen_l:,} frozen"
    )
