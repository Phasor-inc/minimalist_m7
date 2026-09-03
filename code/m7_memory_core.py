"""M7's memory core (HGRN gate + Hopfield associative recall + adaptive
gate), ported from /workspace/ODI26/odi/m7_odi_full_run.py's
ImprovedHGRNGate and ModernHopfieldLayer classes (real code read directly
this session), plus a consolidation_bias mechanism in the same structural
pattern as ODI2026's IndexBindingStore.consolidation_bias(). This is the
regularizer applied to pooled VLM hidden states (see
code/pooled_hidden_adapter.py and code/finetune_run.py) -- it does NOT
touch the DiT action head at all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldLayer(nn.Module):
    """Continuous modern Hopfield associative memory: scaled dot-product
    attention + temperature over a bank of learnable memory patterns,
    residual + LayerNorm. Same architecture as M7's own class."""

    def __init__(self, input_size=512, memory_size=256, temperature=0.1):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.temperature = temperature
        self.memory = nn.Parameter(torch.randn(memory_size, input_size))
        nn.init.xavier_uniform_(self.memory)
        self.ln = nn.LayerNorm(input_size)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        similarity = torch.matmul(x_norm, mem_norm.T) / (self.input_size ** 0.5)
        attention_weights = F.softmax(similarity / self.temperature, dim=1)
        retrieved = torch.matmul(attention_weights, mem_norm)
        return self.ln(retrieved + x)


class ImprovedHGRNGate(nn.Module):
    """GRU-style hierarchical temporal gate with LayerNorm-stabilized reset/
    update/candidate paths. Same architecture as M7's own class."""

    def __init__(self, input_size=512, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_z.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        r = torch.sigmoid(self.ln_r(self.W_r(combined)))
        z = torch.sigmoid(self.ln_z(self.W_z(combined)))
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.ln_h(self.W_h(combined_reset)))
        return (1 - z) * h_prev + z * h_tilde


class M7MemoryCore(nn.Module):
    """Combines HGRN gating and Hopfield recall via a 2-way learned softmax
    (adaptive_gate, same structure as M7_ContinualAdaptiveModel's own
    self.adaptive_gate), and tracks a running consolidation_bias -- the
    exponential-moving-average of the combined output, used as a soft
    "what this representation should look like" target that the fine-tuning
    loss regularizes toward (code/finetune_run.py adds
    F.mse_loss(pooled_hidden, core.consolidation_bias().detach()) to the
    training loss). Starts at zero so the very first forward pass adds no
    regularization pressure -- the bias only reflects prior activity."""

    def __init__(self, input_size=512, hidden_size=512, memory_size=256, ema_decay=0.99):
        super().__init__()
        self.hopfield = ModernHopfieldLayer(input_size, memory_size)
        self.hgrn = ImprovedHGRNGate(input_size, hidden_size)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        self.ema_decay = ema_decay
        self.register_buffer("_consolidation_bias", torch.zeros(input_size))
        self.register_buffer("_has_seen_data", torch.tensor(False))

    def forward(self, x, h_prev):
        hopfield_out = self.hopfield(x)
        hgrn_out = self.hgrn(x, h_prev)
        gate = self.adaptive_gate(x)  # [B, 2]
        combined = gate[:, 0:1] * hopfield_out + gate[:, 1:2] * hgrn_out
        if self.training:
            with torch.no_grad():
                batch_mean = combined.mean(dim=0)
                if self._has_seen_data:
                    self._consolidation_bias.mul_(self.ema_decay).add_(
                        batch_mean, alpha=1 - self.ema_decay
                    )
                else:
                    self._consolidation_bias.copy_(batch_mean)
                    self._has_seen_data.fill_(True)
        return combined, hgrn_out

    def consolidation_bias(self):
        return self._consolidation_bias
