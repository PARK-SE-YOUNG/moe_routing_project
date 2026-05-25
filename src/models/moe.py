import torch
import torch.nn as nn

from src.models.router import Router


class Expert(nn.Module):

    def __init__(
        self,
        hidden_dim=64,
        ffn_dim=128
    ):

        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim)
        )

    def forward(self, x):

        return self.ffn(x)


class MoELayer(nn.Module):

    def __init__(
        self,
        num_experts=4,
        hidden_dim=64,
        ffn_dim=128,
        top_k=2,
        use_adapter=True,
        temperature=0.7
    ):

        super().__init__()

        self.num_experts = num_experts

        self.top_k = top_k

        # ---------------------------------
        # Router
        # ---------------------------------

        self.router = Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            use_adapter=use_adapter
        )

        # ---------------------------------
        # Experts
        # ---------------------------------

        self.experts = nn.ModuleList([

            Expert(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim
            )

            for _ in range(num_experts)

        ])

    def forward(self, x):

        # =================================
        # Router Forward
        # =================================

        router_outputs = self.router(x)

        router_probs = router_outputs[
            "router_probs"
        ]

        topk_probs = router_outputs[
            "topk_probs"
        ]

        topk_indices = router_outputs[
            "topk_indices"
        ]

        entropy = router_outputs[
            "entropy"
        ]

        expert_usage = router_outputs[
            "expert_usage"
        ]

        # =================================
        # Log probabilities
        # (for future RL training)
        # =================================

        log_router_probs = torch.log(
            router_probs + 1e-9
        )

        selected_log_probs = torch.gather(

            log_router_probs,

            dim=-1,

            index=topk_indices

        )

        # =================================
        # Final output tensor
        # =================================

        final_output = torch.zeros_like(x)

        # =================================
        # Weighted Expert Aggregation
        # =================================

        for k_idx in range(self.top_k):

            # ---------------------------------
            # kth selected expert
            # ---------------------------------

            selected_experts = topk_indices[
                ..., k_idx
            ]

            selected_probs = topk_probs[
                ..., k_idx
            ]

            # ---------------------------------
            # Loop over experts
            # ---------------------------------

            for expert_idx, expert in enumerate(
                self.experts
            ):

                mask = (
                    selected_experts
                    == expert_idx
                )

                if mask.sum() == 0:
                    continue

                # -----------------------------
                # Token selection
                # -----------------------------

                expert_input = x[mask]

                # -----------------------------
                # Expert forward
                # -----------------------------

                expert_output = expert(
                    expert_input
                )

                # -----------------------------
                # Gate weighting
                # -----------------------------

                gate_weight = selected_probs[
                    mask
                ].unsqueeze(-1)

                weighted_output = (
                    gate_weight
                    * expert_output
                )

                # -----------------------------
                # Aggregation
                # -----------------------------

                final_output[mask] += (
                    weighted_output
                )

        # =================================
        # Return
        # =================================

        return {
            "moe_output": final_output,
            "router_outputs": router_outputs
        }