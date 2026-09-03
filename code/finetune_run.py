"""Single, non-sequential LoRA fine-tuning run on top of frozen
Xiaomi-Robotics-1-RoboCasa365, with an optional M7 consolidation
regularization term. Supports two modes for the ablation required by the
design spec: LoRA-only (use_m7_consolidation=False) and LoRA+M7
(use_m7_consolidation=True) -- same script, one flag, so both runs are
guaranteed to share every other hyperparameter.
"""
import torch
import torch.nn.functional as F


def compute_total_loss(
    xiaomi_loss,
    pooled_hidden,
    adapter,
    memory_core,
    use_m7_consolidation: bool,
    consolidation_weight: float = 0.1,
):
    """xiaomi_loss: their real, unchanged 4-term loss (loss_mse + loss_freq
    + loss_choice + loss_score, computed exactly as in their forward()).
    pooled_hidden: [B, vlm_hidden_size] from pool_state_token_hidden_states,
    only used when use_m7_consolidation=True.
    Returns (total_loss, breakdown_dict) -- breakdown always has 'xiaomi',
    only has 'consolidation' when the M7 term is active, so callers can log
    both runs' breakdowns with the same code path."""
    breakdown = {"xiaomi": xiaomi_loss.detach().item()}
    if not use_m7_consolidation:
        return xiaomi_loss, breakdown

    # Capture the consolidation target BEFORE calling memory_core(...),
    # since forward() mutates the EMA consolidation_bias buffer using THIS
    # batch's own output as a side effect. Reading the bias AFTER the call
    # would make the regularization target trivially self-referential --
    # 100% self-leakage on the very first call (bias starts at zero, gets
    # set directly to this batch's mean, then immediately used as "the
    # target" for this same batch), and a 1% same-batch leak on every
    # subsequent call at ema_decay=0.99. Capturing prior_bias first
    # preserves the intended semantics: "regularize toward what prior
    # batches looked like," not "regularize toward itself."
    prior_bias = memory_core.consolidation_bias().detach().clone()

    projected = adapter(pooled_hidden)
    h_prev = torch.zeros_like(projected)
    combined, _ = memory_core(projected, h_prev)  # mutates the EMA bias as a side effect; regularization uses prior_bias captured above, not this call's own output
    target = prior_bias.unsqueeze(0).expand_as(combined)
    consolidation_loss = F.mse_loss(combined, target)

    total = xiaomi_loss + consolidation_weight * consolidation_loss
    breakdown["consolidation"] = consolidation_loss.detach().item()
    return total, breakdown
