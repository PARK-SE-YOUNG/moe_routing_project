import torch
import torch.nn as nn

from src.models.moe import MoELayer


class SparseTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=100,
        hidden_dim=64,
        num_heads=4,
        num_classes=2,
        num_experts=4,
        top_k=2,
        use_adapter=True,
        temperature=0.7
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.moe = MoELayer(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k,
            use_adapter=use_adapter,
            temperature=temperature
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        # ---------------------------------
        # Embedding
        # ---------------------------------

        x = self.embedding(x)

        # ---------------------------------
        # Attention
        # ---------------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # ---------------------------------
        # MoE
        # ---------------------------------

        moe_outputs = self.moe(
            attention_output
        )

        moe_output = moe_outputs[
            "moe_output"
        ]

        router_outputs = moe_outputs[
            "router_outputs"
        ]

        # ---------------------------------
        # Pooling
        # ---------------------------------

        pooled = moe_output.mean(dim=1)

        # ---------------------------------
        # Classification
        # ---------------------------------

        logits = self.classifier(
            pooled
        )

        # ---------------------------------
        # Return
        # ---------------------------------

        return {
            "logits": logits,
            "router_outputs": router_outputs
        }