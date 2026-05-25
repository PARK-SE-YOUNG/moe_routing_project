import torch
import torch.nn as nn


class DenseTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=100,
        hidden_dim=64,
        num_heads=4,
        num_classes=2
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

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        ffn_output = self.ffn(attention_output)

        pooled = ffn_output.mean(dim=1)

        logits = self.classifier(pooled)

        return logits