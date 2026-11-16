"""LoRA on OLMoE's fused experts (Model 4).

In transformers >=5.x OLMoE stores all experts of a layer as fused 3-D tensors
inside a single ``OlmoeExperts`` module (``gate_up_proj`` with shape
``[num_experts, 2*intermediate, hidden]`` and ``down_proj`` with shape
``[num_experts, hidden, intermediate]``). They are ``nn.Parameter`` tensors, not
``nn.Linear`` sub-modules, so the standard PEFT ``LoraConfig(target_modules=...)``
cannot attach to them.

This module adds a per-expert low-rank adapter directly to the fused weights and
replaces the experts module with one whose ``forward`` we control (so the result
is identical on CPU and GPU and never silently dispatches to a fused kernel that
would skip the adapter). The frozen base weights keep their original parameter
names, and the new LoRA parameters are plain named tensors, so the existing
"trainable-params-only" checkpoint save/load works unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExperts(nn.Module):
    """Drop-in replacement for ``OlmoeExperts``: frozen fused base weights plus a
    per-expert low-rank update on ``gate_up_proj`` and ``down_proj``.

    The forward mirrors the eager ``OlmoeExperts.forward`` exactly and adds the
    standard LoRA path ``scaling * B @ (A @ x)`` for each expert. LoRA ``B`` is
    initialised to zero, so at step 0 the model is bit-for-bit the base model.
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_experts = base.num_experts
        self.hidden_dim = base.hidden_dim
        self.intermediate_dim = base.intermediate_dim
        self.act_fn = base.act_fn

        # Reuse the base fused weights (names preserved) and freeze them.
        self.gate_up_proj = base.gate_up_proj
        self.down_proj = base.down_proj
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        E, H, I = self.num_experts, self.hidden_dim, self.intermediate_dim
        dev, dt = self.gate_up_proj.device, self.gate_up_proj.dtype
        # gate_up weight is (2I, H): LoRA B(2I, r) @ A(r, H)
        self.lora_A_gu = nn.Parameter(torch.empty(E, r, H, device=dev, dtype=dt))
        self.lora_B_gu = nn.Parameter(torch.zeros(E, 2 * I, r, device=dev, dtype=dt))
        # down weight is (H, I): LoRA B(H, r) @ A(r, I)
        self.lora_A_dn = nn.Parameter(torch.empty(E, r, I, device=dev, dtype=dt))
        self.lora_B_dn = nn.Parameter(torch.zeros(E, H, r, device=dev, dtype=dt))
        for a in (self.lora_A_gu, self.lora_A_dn):
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        # B stays zero -> initial delta is 0 -> model == base at start.

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]

            gu = F.linear(current_state, self.gate_up_proj[e])
            gu = gu + self.scaling * F.linear(
                F.linear(self.lora_dropout(current_state), self.lora_A_gu[e]),
                self.lora_B_gu[e],
            )
            gate, up = gu.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            dn = F.linear(current_hidden_states, self.down_proj[e])
            dn = dn + self.scaling * F.linear(
                F.linear(self.lora_dropout(current_hidden_states), self.lora_A_dn[e]),
                self.lora_B_dn[e],
            )
            dn = dn * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, dn.to(final_hidden_states.dtype))

        return final_hidden_states


def inject_expert_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    """Replace every ``OlmoeSparseMoeBlock.experts`` with a ``LoRAExperts`` wrapper.

    Returns the number of MoE blocks modified. Idempotent guard: a block whose
    experts are already a ``LoRAExperts`` is skipped.
    """
    count = 0
    for module in model.modules():
        if module.__class__.__name__ == "OlmoeSparseMoeBlock":
            if isinstance(module.experts, LoRAExperts):
                continue
            module.experts = LoRAExperts(module.experts, r=r, alpha=alpha, dropout=dropout)
            count += 1
    return count


def checkpoint_has_expert_lora(saved_keys) -> bool:
    """True if a saved trainable-params state dict contains expert-LoRA tensors."""
    return any(".experts.lora_" in str(k) for k in saved_keys)
