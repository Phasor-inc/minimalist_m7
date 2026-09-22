# Why `m7_memory_core.py` isn't in this repo

This submission's eval code and deployed checkpoint do **not** need this file — the
memory core never runs at inference time (see `merge_lora_for_eval.py`'s own
docstring: "M7's extra adapter/memory_core state is never part of the served
inference graph"). The deployed, merged checkpoint is an ordinary frozen backbone
+ merged LoRA deltas, architecturally identical to a plain LoRA fine-tune.

Where it *was* used: as a training-time auxiliary loss (`--use-m7-consolidation`
in `finetune_run.py`) during the 220-step fine-tune, shaping what the LoRA deltas
converged to. It is Phasor's own architecture, shared across several of our other
projects, and we're keeping the literal implementation proprietary rather than
publishing it here.

In place of the source, here is the exact mechanism, precise enough to reproduce:

## `ModernHopfieldLayer`

Scaled dot-product attention over a learnable memory bank, with a LayerNorm
residual:

```
x_norm   = normalize(x, p=2, dim=1)
mem_norm = normalize(memory, p=2, dim=1)          # memory: [memory_size, hidden]
sim      = (x_norm @ mem_norm.T) / sqrt(hidden)
attn     = softmax(sim / temperature, dim=1)      # temperature = 0.1
retrieved = attn @ mem_norm
hop_out  = LayerNorm(retrieved + x)
```

## `ImprovedHGRNGate`

A full GRU-style gate (reset/update/candidate, three weight matrices — not the
simplified single-forget-gate formula sometimes seen in the literature):

```
combined       = concat(x, h_prev)
r              = sigmoid(LayerNorm(W_r(combined)))
z              = sigmoid(LayerNorm(W_z(combined)))
combined_reset = concat(x, r * h_prev)
h_tilde        = tanh(LayerNorm(W_h(combined_reset)))
hgrn_out       = (1 - z) * h_prev + z * h_tilde   # h_prev = zeros for this use
```

## Adaptive gate + combination

```
gate       = softmax(Linear(128, 2)(Dropout(0.1)(ReLU(LayerNorm(Linear(hidden, 128)(x))))))
w_h, w_g   = gate[:, 0:1], gate[:, 1:2]
combined_t = w_h * hop_out + w_g * hgrn_out
```

## Consolidation loss (what actually touched training)

```
consolidation_loss = MSE(combined_t, pooled_vlm_hidden_state.detach())
total_loss = task_loss + consolidation_weight * consolidation_loss   # weight = 0.1
```

Gradients from `total_loss` flow into the LoRA adapters and the memory core's own
parameters (Hopfield memory bank, HGRN weight matrices, adaptive gate) — never into
the frozen backbone or the DiT action head.

Happy to discuss further or answer specific reproducibility questions directly.
