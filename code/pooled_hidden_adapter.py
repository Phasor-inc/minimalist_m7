"""Bridges the VLM's real hidden size (verified against the loaded model in
Task 2 -- 2560 for Qwen3-VL-4B-Instruct -- passed in as a constructor
argument, never hardcoded here) to M7's memory core's 512-dim input space,
and defines exactly what "pooled VLM hidden states" means: mean-pooling
over the STATE token positions (the same input_ids range XR1.py's own
state_projector_choice consumes, i.e. where `state_embeds` gets attached in
their forward() -- see the real `self.state_projector_choice(state.flatten(0, 1))`
call in vendor/Xiaomi-Robotics-1/xr1/mibot/models/VLA/XR1.py). Mean-pooling
over state-token positions was chosen (over e.g. last-token pooling)
because the state tokens are the fixed-size, always-present anchor in every
batch, unlike the variable-length action-choice/score tokens which only
exist during their auxiliary choice-prediction training path.
"""
import torch
import torch.nn as nn


class PooledHiddenAdapter(nn.Module):
    def __init__(self, vlm_hidden_size: int, target_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(vlm_hidden_size, target_dim)

    def forward(self, pooled_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled_hidden_states)


def pool_state_token_hidden_states(hidden_states: torch.Tensor, state_token_mask: torch.Tensor) -> torch.Tensor:
    """hidden_states: [seq_len, hidden_size] (single sample) or
    [batch, seq_len, hidden_size]. state_token_mask: bool, same leading
    seq_len dim. Returns the mean over masked positions."""
    if not state_token_mask.any():
        raise ValueError("state_token_mask has no True positions to pool over")
    if hidden_states.dim() == 2:
        return hidden_states[state_token_mask].mean(dim=0)
    return torch.stack([
        hidden_states[i][state_token_mask[i]].mean(dim=0)
        for i in range(hidden_states.shape[0])
    ])
