"""LoRA adapter injection for Xiaomi-Robotics-1's VLM language_model layers.

Target module names (self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj) were
verified against the real loaded model in Task 2 -- see NOTES.md. If those
differ from what's used here, this file's TARGET_ATTN_MODULES /
TARGET_MLP_MODULES lists are the single place to update.
"""
import math

import torch
import torch.nn as nn

TARGET_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TARGET_MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    """Wraps a frozen base nn.Linear with a trainable low-rank update.

    B is zero-initialized so LoRALinear(x) == base_linear(x) exactly at
    construction time -- fine-tuning starts from parity with the frozen
    base model, not a perturbed version of it.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        dtype = base_linear.weight.dtype
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def _replace_module(parent: nn.Module, attr_name: str, rank: int, alpha: float):
    base = getattr(parent, attr_name)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {attr_name}, got {type(base).__name__}")
    setattr(parent, attr_name, LoRALinear(base, rank=rank, alpha=alpha))


def inject_lora_into_vlm_layers(vlm_or_layers_container, rank: int = 4, alpha: float = 8.0):
    """Replaces target attention/MLP Linear layers in every decoder layer
    with LoRALinear wrappers, in place. `vlm_or_layers_container` must have
    a `.layers` attribute that is iterable of objects with `.self_attn` and
    `.mlp` submodules (matches both the DummyVLM test double and the real
    Qwen3-VL language_model structure verified in Task 2)."""
    for layer in vlm_or_layers_container.layers:
        for attn_name in TARGET_ATTN_MODULES:
            _replace_module(layer.self_attn, attn_name, rank, alpha)
        for mlp_name in TARGET_MLP_MODULES:
            _replace_module(layer.mlp, mlp_name, rank, alpha)
