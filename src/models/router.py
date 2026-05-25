import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Router Adapter
# =====================================================

class RouterAdapter(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_experts,
        adapter_dim=32
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(hidden_dim, adapter_dim),

            nn.ReLU(),

            nn.Linear(adapter_dim, num_experts)
        )

        # ---------------------------------
        # Zero-init final layer
        # ---------------------------------

        nn.init.zeros_(self.net[-1].weight)

        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):

        return self.net(x)


# =====================================================
# Main Router
# =====================================================

class Router(nn.Module):

    def __init__(
        self,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        temperature=0.7,
        use_adapter=True
    ):

        super().__init__()

        self.num_experts = num_experts

        self.top_k = top_k

        self.temperature = temperature

        self.use_adapter = use_adapter

        # ---------------------------------
        # Original router
        # ---------------------------------

        self.router = nn.Linear(
            hidden_dim,
            num_experts
        )

        # ---------------------------------
        # Adapter
        # ---------------------------------

        if self.use_adapter:

            self.adapter = RouterAdapter(
                hidden_dim=hidden_dim,
                num_experts=num_experts
            )

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        """
        x:
        [batch_size, seq_len, hidden_dim]
        """

        # ---------------------------------
        # Original router logits
        # ---------------------------------

        original_logits = self.router(x)

        # ---------------------------------
        # Original probs
        # ---------------------------------

        with torch.no_grad():

            original_router_probs = F.softmax(
                original_logits,
                dim=-1
            )

        # ---------------------------------
        # Adapter delta logits
        # ---------------------------------

        if self.use_adapter:

            delta_logits = self.adapter(x)

        else:

            delta_logits = torch.zeros_like(
                original_logits
            )

        # ---------------------------------
        # Final router logits
        # ---------------------------------

        router_logits = (
            original_logits + delta_logits
        )

        # ---------------------------------
        # Temperature scaling
        # ---------------------------------

        router_logits = (
            router_logits / self.temperature
        )

        # ---------------------------------
        # Router probs
        # ---------------------------------

        router_probs = F.softmax(
            router_logits,
            dim=-1
        )

        # ---------------------------------
        # Top-k routing
        # ---------------------------------

        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1
        )

        # ---------------------------------
        # Expert usage
        # ---------------------------------

        flat_indices = topk_indices.reshape(-1)

        expert_counts = torch.bincount(
            flat_indices,
            minlength=self.num_experts
        ).float()

        expert_usage = (
            expert_counts / expert_counts.sum()
        )

        # ---------------------------------
        # Entropy
        # ---------------------------------

        entropy = -(
            router_probs *
            torch.log(router_probs + 1e-9)
        ).sum(dim=-1).mean()

        # ---------------------------------
        # KL divergence
        # ---------------------------------

        kl_to_original = (
            router_probs *
            (
                torch.log(router_probs + 1e-9)
                -
                torch.log(
                    original_router_probs + 1e-9
)
            )
        ).sum(dim=-1).mean()

        # ---------------------------------
        # RL log-prob
        # ---------------------------------

        selected_log_probs = torch.log(
            topk_probs + 1e-9
        )

        # ---------------------------------
        # Return
        # ---------------------------------

        return {
            # ---------------------------------
            # # Router outputs
            # ---------------------------------

            "router_logits": router_logits,

            "router_probs": router_probs,

            "original_router_probs": original_router_probs,

            # ---------------------------------
            # Top-k routing
            # ---------------------------------

            "topk_probs": topk_probs,

            "topk_indices": topk_indices,

            # ---------------------------------
            # Metrics
            # ---------------------------------

            "expert_usage": expert_usage,

            "entropy": entropy,

            "kl_to_original": kl_to_original,

            # ---------------------------------
            # RL-ready logging
            # ---------------------------------

            "selected_log_probs": selected_log_probs
        }