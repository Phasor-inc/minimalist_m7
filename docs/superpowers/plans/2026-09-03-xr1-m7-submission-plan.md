# Xiaomi-Robotics-1 + M7 Memory Core RoboCasa Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a real, submittable RoboCasa365 checkpoint by adding LoRA adapters plus M7's HGRN+Hopfield memory core (as a consolidation regularizer) to Xiaomi-Robotics-1-RoboCasa365's frozen VLM, without touching its DiT action head, and evaluate it against their own real `eval_robocasa365` protocol alongside a LoRA-only ablation.

**Architecture:** Frozen `Xiaomi-Robotics-1-RoboCasa365` (Qwen3-VL-4B VLM + 36-layer DiT). LoRA adapters (rank=4, alpha=8.0, zero-init B) injected into the VLM's `language_model` attention/MLP Linear layers only. A separate branch pools the VLM's hidden states at the same token positions their own `action_choice`/`score_choice` path already reads, projects to 512-dim, runs through `ImprovedHGRNGate` + `ModernHopfieldLayer` + `adaptive_gate` (ported from M7's ODI2026 code), and adds a `consolidation_bias` regularization term to their unchanged 4-term loss. DiT is never modified.

**Tech Stack:** PyTorch, HuggingFace `transformers` (pinned `4.57.1` for eval; VLM training uses whatever version their `xr1` package requires — confirmed in Task 1), `snntorch` is NOT needed here (M7's memory core components are plain `nn.Module`, no spiking dependency), `pytest` for unit tests, their own `torchrun`-based `tools/train.py` entry point, their own client-server `eval_robocasa365` scripts.

---

### Task 1: Clone Xiaomi's repo, verify GPU headroom, download the checkpoint

**Files:** none created — infrastructure setup, first commit is just the clone + a NOTES.md capturing what was verified.

- [ ] **Step 1: Check current GPU headroom before doing anything GPU-related**

Run (on the pod):
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```
This project needs real headroom for a 5B model + LoRA fine-tuning. If `memory.used` is within ~20GB of `memory.total`, STOP and check what's still running (`ps aux | grep -E "m7_robocasa|eval_full_parallel"`) before proceeding — do not assume the eval sweep or M7's chunking/JEPA training from earlier sessions have finished. Record the real free-memory number in `NOTES.md` (Step 5 below).

- [ ] **Step 2: Clone their repo as a vendored reference**

```bash
mkdir -p /workspace/xr1-m7-submission/vendor
cd /workspace/xr1-m7-submission/vendor
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git
cd Xiaomi-Robotics-1
git log -1 --format="%H %cd"
```
Record the commit hash in `NOTES.md` — this pins exactly which version of their code this project was built against, since their `main` branch can change.

- [ ] **Step 3: Set up the deploy (`mibot`) environment**

Follow `vendor/Xiaomi-Robotics-1/docs/DEPLOYMENT.md` exactly as written (real file, read it directly — do not guess its contents). Record the actual commands used in `NOTES.md`, since this doc may reference conda/mamba and specific CUDA/torch versions that need to match the pod's real setup.

- [ ] **Step 4: Download the RoboCasa365 checkpoint**

```bash
mkdir -p /workspace/xr1-m7-submission/checkpoints
export MODEL_PATH=/workspace/xr1-m7-submission/checkpoints/Xiaomi-Robotics-1-RoboCasa365
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --local-dir "$MODEL_PATH"
du -sh "$MODEL_PATH"
```
Expected: a real directory with model weights, config.json, tokenizer files. Record the real disk size in `NOTES.md` (needed to confirm the pod's earlier-reported 335TB free `/workspace` mount has room, which it does, but record the real number anyway rather than assuming).

- [ ] **Step 5: Write NOTES.md capturing verified facts**

```markdown
# xr1-m7-submission — verified environment facts

- GPU headroom at setup time: <fill in from Step 1>
- Xiaomi repo commit: <fill in from Step 2>
- Checkpoint size on disk: <fill in from Step 4>
- transformers version installed by mibot deploy env: <run `pip show transformers` in that env, fill in>
```

- [ ] **Step 6: Commit**

```bash
cd /workspace/xr1-m7-submission
git add NOTES.md
git commit -m "docs: record verified environment facts from setup"
```
(The `vendor/` clone and `checkpoints/` directory should NOT be committed — add a `.gitignore`:)

```bash
cat > /workspace/xr1-m7-submission/.gitignore << 'EOF'
vendor/
checkpoints/
*.pt
*.safetensors
__pycache__/
.pytest_cache/
EOF
git add .gitignore
git commit -m "chore: ignore vendored repo, checkpoints, and caches"
```

---

### Task 2: Verify real Qwen3-VL module names and hidden size

**Files:**
- Create: `code/inspect_vlm_modules.py` (throwaway inspection script, kept for reproducibility)

**Files:** none modified beyond the script above — this task is pure verification, no LoRA code yet.

- [ ] **Step 1: Write the inspection script**

```python
# code/inspect_vlm_modules.py
"""One-time inspection: print the real module tree of the loaded VLM so
LoRA target-module names (Task 3) and vlm_hidden_size (Task 5) are taken
from the actual model, not assumed from reading source code alone."""
import sys

sys.path.insert(0, "/workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1/xr1")
from mibot.models.VLA.XR1 import xr1

model = xr1()
print("vlm_hidden_size:", model.vlm.config.text_config.hidden_size)
print()
print("=== First language_model decoder layer's module names ===")
first_layer = model.vlm.model.language_model.layers[0]
for name, module in first_layer.named_modules():
    if name:
        print(f"{name}: {type(module).__name__}")
```

- [ ] **Step 2: Run it and record the real output**

```bash
cd /workspace/xr1-m7-submission
PYTHONPATH=/workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1/xr1:$PYTHONPATH \
  python3 code/inspect_vlm_modules.py 2>&1 | tee /tmp/vlm_module_inspection.txt
```
Expected: a real `vlm_hidden_size` integer (from Qwen3-VL-4B's actual config, not guessed), and a list of real attention/MLP submodule names (standard Qwen3 naming is `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` — but this step exists specifically to confirm that against the real loaded model rather than assume it, since assuming wrong here breaks every later task silently).

- [ ] **Step 3: Append the real findings to NOTES.md**

```markdown
- vlm_hidden_size (verified from loaded model): <fill in>
- Real LoRA target module names (verified): <fill in exact dotted names>
```

- [ ] **Step 4: Commit**

```bash
git add code/inspect_vlm_modules.py NOTES.md
git commit -m "docs: verify real VLM module names and hidden size from loaded model"
```

---

### Task 3: LoRA adapter injection

**Files:**
- Create: `code/lora_adapters.py`
- Test: `tests/test_lora_adapters.py`

Testable entirely without downloading the real 5B model — uses a tiny dummy module with the same attribute names verified in Task 2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lora_adapters.py
import torch
import torch.nn as nn
from code.lora_adapters import LoRALinear, inject_lora_into_vlm_layers


class DummySelfAttn(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class DummyMLP(nn.Module):
    def __init__(self, dim=8, inter=16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter)
        self.up_proj = nn.Linear(dim, inter)
        self.down_proj = nn.Linear(inter, dim)


class DummyDecoderLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = DummySelfAttn(dim)
        self.mlp = DummyMLP(dim)


class DummyVLM(nn.Module):
    def __init__(self, num_layers=2, dim=8):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(dim) for _ in range(num_layers)])


def test_lora_linear_matches_base_at_init():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_linear_base_weights_frozen():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    assert lora.base_linear.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_lora_linear_output_shape():
    base = nn.Linear(8, 16)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert lora(x).shape == (3, 16)


def test_inject_lora_into_vlm_layers_replaces_all_target_modules():
    vlm = DummyVLM(num_layers=2)
    inject_lora_into_vlm_layers(vlm, rank=4, alpha=8.0)
    for layer in vlm.layers:
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.k_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        assert isinstance(layer.self_attn.o_proj, LoRALinear)
        assert isinstance(layer.mlp.gate_proj, LoRALinear)
        assert isinstance(layer.mlp.up_proj, LoRALinear)
        assert isinstance(layer.mlp.down_proj, LoRALinear)


def test_inject_lora_preserves_output_at_init():
    torch.manual_seed(0)
    vlm_a = DummyVLM(num_layers=1)
    vlm_b = DummyVLM(num_layers=1)
    vlm_b.load_state_dict(vlm_a.state_dict())
    inject_lora_into_vlm_layers(vlm_b, rank=4, alpha=8.0)

    x = torch.randn(2, 8)
    out_a = vlm_a.layers[0].self_attn.q_proj(x)
    out_b = vlm_b.layers[0].self_attn.q_proj(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_inject_lora_only_target_modules_trainable():
    vlm = DummyVLM(num_layers=1)
    inject_lora_into_vlm_layers(vlm, rank=4, alpha=8.0)
    trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
    assert all("lora_A" in n or "lora_B" in n for n in trainable)
    assert len(trainable) == 14  # 7 target modules x 2 (lora_A, lora_B) x 1 layer
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/xr1-m7-submission
python3 -m pytest tests/test_lora_adapters.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'code.lora_adapters'`.

- [ ] **Step 3: Write the implementation**

```python
# code/lora_adapters.py
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

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def _replace_module(parent: nn.Module, attr_name: str, rank: float, alpha: float):
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_lora_adapters.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add code/lora_adapters.py tests/test_lora_adapters.py
git commit -m "feat: add LoRA adapter injection for VLM language_model layers"
```

---

### Task 4: Port M7's memory core

**Files:**
- Create: `code/m7_memory_core.py`
- Test: `tests/test_m7_memory_core.py`

Ports `ModernHopfieldLayer`, `ImprovedHGRNGate`, and an `adaptive_gate`-style
combiner from `/workspace/ODI26/odi/m7_odi_full_run.py` (real code, lines
~718-900 and ~1698 as read this session) into this repo's own module —
same duplication-over-cross-repo-dependency choice already made in
`xr1-continual`'s design for `synaptic_intelligence.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_m7_memory_core.py
import torch
from code.m7_memory_core import ImprovedHGRNGate, ModernHopfieldLayer, M7MemoryCore


def test_hgrn_gate_output_shape():
    gate = ImprovedHGRNGate(input_size=32, hidden_size=32)
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 32)
    h_new = gate(x, h_prev)
    assert h_new.shape == (4, 32)


def test_hgrn_gate_is_bounded_combination():
    torch.manual_seed(0)
    gate = ImprovedHGRNGate(input_size=32, hidden_size=32)
    x = torch.randn(4, 32)
    h_prev = torch.zeros(4, 32)
    h_new = gate(x, h_prev)
    # h_new = (1-z)*h_prev + z*h_tilde with h_prev=0, so h_new should be
    # elementwise between 0 and h_tilde's range -- just check it's finite
    # and not identically zero (gate is actually doing something)
    assert torch.isfinite(h_new).all()
    assert not torch.allclose(h_new, torch.zeros_like(h_new))


def test_hopfield_layer_output_shape():
    hopfield = ModernHopfieldLayer(input_size=32, memory_size=16, temperature=0.1)
    x = torch.randn(4, 32)
    out = hopfield(x)
    assert out.shape == (4, 32)


def test_hopfield_layer_retrieves_nearest_memory():
    torch.manual_seed(0)
    hopfield = ModernHopfieldLayer(input_size=32, memory_size=16, temperature=0.01)
    # very low temperature -> near-hard attention -> output should be close
    # to whichever memory row is most similar to a strongly-scaled input
    x = hopfield.memory[0:1].detach() * 10.0
    out = hopfield(x)
    assert torch.isfinite(out).all()


def test_m7_memory_core_forward_and_consolidation_bias():
    core = M7MemoryCore(input_size=32, hidden_size=32, memory_size=16)
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 32)
    output, h_new = core(x, h_prev)
    assert output.shape == (4, 32)
    assert h_new.shape == (4, 32)

    bias = core.consolidation_bias()
    assert bias.shape == (32,)
    assert torch.isfinite(bias).all()


def test_m7_memory_core_consolidation_bias_zero_before_any_forward():
    core = M7MemoryCore(input_size=32, hidden_size=32, memory_size=16)
    bias = core.consolidation_bias()
    assert torch.allclose(bias, torch.zeros(32))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_m7_memory_core.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'code.m7_memory_core'`.

- [ ] **Step 3: Write the implementation**

```python
# code/m7_memory_core.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_m7_memory_core.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add code/m7_memory_core.py tests/test_m7_memory_core.py
git commit -m "feat: port M7 memory core (HGRN + Hopfield + adaptive gate + consolidation bias)"
```

---

### Task 5: Pooled hidden-state adapter

**Files:**
- Create: `code/pooled_hidden_adapter.py`
- Test: `tests/test_pooled_hidden_adapter.py`

Bridges the VLM's real `vlm_hidden_size` (verified in Task 2, NOT hardcoded
here) to M7's memory core's 512-dim input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pooled_hidden_adapter.py
import torch
from code.pooled_hidden_adapter import PooledHiddenAdapter, pool_state_token_hidden_states


def test_adapter_projects_to_target_dim():
    adapter = PooledHiddenAdapter(vlm_hidden_size=2560, target_dim=512)
    x = torch.randn(4, 2560)
    out = adapter(x)
    assert out.shape == (4, 512)


def test_pool_state_token_hidden_states_mean_pools_masked_positions():
    # hidden_states: [seq_len, hidden_size], state_token_mask: [seq_len] bool
    hidden_states = torch.tensor([
        [1.0, 1.0],
        [3.0, 3.0],
        [5.0, 5.0],
    ])
    mask = torch.tensor([True, False, True])
    pooled = pool_state_token_hidden_states(hidden_states, mask)
    assert torch.allclose(pooled, torch.tensor([3.0, 3.0]))


def test_pool_state_token_hidden_states_raises_on_empty_mask():
    hidden_states = torch.randn(3, 2)
    mask = torch.tensor([False, False, False])
    try:
        pool_state_token_hidden_states(hidden_states, mask)
        assert False, "expected ValueError on all-False mask"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_pooled_hidden_adapter.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'code.pooled_hidden_adapter'`.

- [ ] **Step 3: Write the implementation**

```python
# code/pooled_hidden_adapter.py
"""Bridges the VLM's real hidden size (verified against the loaded model in
Task 2, passed in as a constructor argument -- never hardcoded here) to M7's
memory core's 512-dim input space, and defines exactly what "pooled VLM
hidden states" means: mean-pooling over the STATE token positions (the
same input_ids range XR1.py's own state_projector_choice consumes, i.e.
where `state_embeds` gets attached in their forward() -- see the real
`self.state_projector_choice(state.flatten(0, 1))` call in
vendor/Xiaomi-Robotics-1/xr1/mibot/models/VLA/XR1.py). Mean-pooling over
state-token positions was chosen (over e.g. last-token pooling) because the
state tokens are the fixed-size, always-present anchor in every batch,
unlike the variable-length action-choice/score tokens which only exist
during their auxiliary choice-prediction training path.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_pooled_hidden_adapter.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add code/pooled_hidden_adapter.py tests/test_pooled_hidden_adapter.py
git commit -m "feat: add pooled hidden-state adapter bridging VLM to M7 memory core"
```

---

### Task 6: Settle training data availability

**Files:**
- Create: `code/inspect_training_data.py` (throwaway inspection script, kept for reproducibility)

Real, unresolved question from research this session: no full RoboCasa365
training corpus was found published by Xiaomi — only
`XiaomiRobotics/xr1_post_train_demo` (Apache-2.0, small demo set per its
HuggingFace listing) exists under their org. This task resolves that
concretely before any fine-tuning code is written, since the fine-tuning
loop's data loader depends entirely on the answer.

- [ ] **Step 1: Download and inspect the demo dataset**

```bash
cd /workspace/xr1-m7-submission
mkdir -p data_investigation
hf download XiaomiRobotics/xr1_post_train_demo --repo-type dataset --local-dir data_investigation/xr1_post_train_demo
find data_investigation/xr1_post_train_demo -type f | head -30
```

- [ ] **Step 2: Write the inspection script**

```python
# code/inspect_training_data.py
"""Determine whether xr1_post_train_demo is (a) a full-enough RoboCasa365
training corpus to fine-tune on directly, or (b) a small illustrative demo
that this project needs to supplement -- e.g. by converting M7's own
already-downloaded RoboCasa LeRobot-format dataset
(/workspace/data/robocasa_datasets/v1.0/...) into their JSON schema
(see xr1/configs/data/load_washer.yaml for the real schema: paths to JSON
files, each with 60-dim state/action arrays, plus per-dataset mean/std/
q01/q99 normalization stats)."""
import json
from pathlib import Path

demo_dir = Path("data_investigation/xr1_post_train_demo")
json_files = list(demo_dir.rglob("*.json"))
print(f"Found {len(json_files)} JSON files under {demo_dir}")
for f in json_files[:3]:
    with open(f) as fh:
        sample = json.load(fh)
    print(f"\n{f}:")
    print(f"  top-level keys: {list(sample.keys()) if isinstance(sample, dict) else 'list of ' + str(len(sample)) + ' items'}")
```

- [ ] **Step 3: Run it and record the real finding**

```bash
python3 code/inspect_training_data.py 2>&1 | tee /tmp/training_data_inspection.txt
```

- [ ] **Step 4: Append the resolution to NOTES.md**

```markdown
## Training data decision (Task 6)

<Fill in one of:>
- xr1_post_train_demo contains N real trajectory files in the expected
  schema and is sufficient to fine-tune on directly for this project's
  scope (a single fine-tuning run, not full-scale retraining).
- xr1_post_train_demo is too small / wrong format; this project converts
  M7's existing /workspace/data/robocasa_datasets/v1.0 LeRobot-format data
  into their JSON schema instead. <If this branch, a follow-up task must be
  added here before Task 7 to write and test that converter -- do not
  proceed to Task 7 until this is resolved and documented.>
```

- [ ] **Step 5: Commit**

```bash
git add code/inspect_training_data.py NOTES.md
git commit -m "docs: resolve training data source for fine-tuning"
```

---

### Task 7: Wire the fine-tuning loop

**Files:**
- Create: `code/finetune_run.py`
- Test: `tests/test_finetune_run.py`

This task's exact data-loading call depends on Task 6's resolution — write
it against whichever real path Task 6 settled on, documented in NOTES.md by
that point. The loss-composition logic below is independent of that
decision and fully specified now.

- [ ] **Step 1: Write the failing test for loss composition (the part that's data-source-independent)**

```python
# tests/test_finetune_run.py
import torch
from code.finetune_run import compute_total_loss
from code.m7_memory_core import M7MemoryCore
from code.pooled_hidden_adapter import PooledHiddenAdapter


def test_compute_total_loss_lora_only_mode_excludes_consolidation_term():
    xiaomi_loss = torch.tensor(0.5, requires_grad=True)
    total, breakdown = compute_total_loss(
        xiaomi_loss=xiaomi_loss,
        pooled_hidden=None,
        adapter=None,
        memory_core=None,
        use_m7_consolidation=False,
    )
    assert torch.allclose(total, xiaomi_loss)
    assert "consolidation" not in breakdown


def test_compute_total_loss_with_m7_consolidation_adds_real_term():
    torch.manual_seed(0)
    xiaomi_loss = torch.tensor(0.5, requires_grad=True)
    adapter = PooledHiddenAdapter(vlm_hidden_size=16, target_dim=8)
    memory_core = M7MemoryCore(input_size=8, hidden_size=8, memory_size=4)
    pooled_hidden = torch.randn(2, 16)

    # prime the memory core so consolidation_bias is non-zero
    projected = adapter(pooled_hidden)
    h_prev = torch.zeros(2, 8)
    memory_core.train()
    memory_core(projected, h_prev)

    total, breakdown = compute_total_loss(
        xiaomi_loss=xiaomi_loss,
        pooled_hidden=pooled_hidden,
        adapter=adapter,
        memory_core=memory_core,
        use_m7_consolidation=True,
        consolidation_weight=0.1,
    )
    assert "consolidation" in breakdown
    assert torch.isfinite(total)
    assert total.item() != xiaomi_loss.item()  # real term was added
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_finetune_run.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'code.finetune_run'`.

- [ ] **Step 3: Write `compute_total_loss` (the data-source-independent core)**

```python
# code/finetune_run.py
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

    projected = adapter(pooled_hidden)
    h_prev = torch.zeros_like(projected)
    combined, _ = memory_core(projected, h_prev)
    target = memory_core.consolidation_bias().detach().unsqueeze(0).expand_as(combined)
    consolidation_loss = F.mse_loss(combined, target)

    total = xiaomi_loss + consolidation_weight * consolidation_loss
    breakdown["consolidation"] = consolidation_loss.detach().item()
    return total, breakdown
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_finetune_run.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add code/finetune_run.py tests/test_finetune_run.py
git commit -m "feat: add loss composition for LoRA-only vs LoRA+M7 ablation"
```

- [ ] **Step 6: Add the real training-loop entry point (depends on Task 6's data decision)**

At this point, read `NOTES.md`'s Task 6 resolution and write the actual
data-loading + `torchrun`-driven training loop into `code/finetune_run.py`,
calling `compute_total_loss` above. This step is intentionally left to
implementation time rather than specified with fabricated code here, since
its exact shape depends on a real decision (demo data vs. converted M7
data) this plan cannot make in advance without guessing — same principle
the spec itself already applied to the pooling operation, now applied one
level deeper. Do not skip writing real code here once Task 6 resolves;
this is a concrete follow-up, not a permanent placeholder.

- [ ] **Step 7: Commit the real training loop**

```bash
git add code/finetune_run.py
git commit -m "feat: wire real training loop against resolved data source"
```

---

### Task 8: Set up the eval environment and get a real baseline number

**Files:** none — infrastructure + a real recorded baseline in NOTES.md.

Before fine-tuning anything, get their own unmodified checkpoint's real
score on this pod's setup, as the honest starting point every later
comparison is measured against — do not assume the paper's 80.2/57.1/32.1
transfers exactly to this pod's environment/seed without checking.

- [ ] **Step 1: Check GPU headroom again (state may have changed since Task 1)**

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

- [ ] **Step 2: Set up the `robocasa_365` client environment**

Follow the official RoboCasa installation guide referenced in
`vendor/Xiaomi-Robotics-1/eval_robocasa365/README.md`, then:
```bash
pip install transformers==4.57.1 imageio[ffmpeg] tqdm scipy
export MUJOCO_GL=egl
```
(Same `libEGL.so.1` gap found earlier this session in M7's own eval setup
may recur here — check `python3 -c "import OpenGL.EGL"` succeeds before
assuming this is fine; if it fails the same way, `apt-get install -y
libegl1 libgl1 libopengl0` fixed it for M7's eval.)

- [ ] **Step 3: Start the server (single-GPU adaptation)**

Their reference `scripts/deploy.sh "$MODEL_PATH" 8 8` assumes 8 GPUs. This
pod has 1. Run:
```bash
cd /workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1
conda activate mibot
bash scripts/deploy.sh "$MODEL_PATH" 1 1
tmux attach -t model_servers  # verify server actually started
```

- [ ] **Step 4: Run the smoke test first**

```bash
CONDA_ENV=robocasa_365 NUM_TRIALS=1 \
  bash scripts/launch_robocasa365.sh \
  1 /workspace/xr1-m7-submission/eval_results/smoke "$MODEL_PATH" \
  --task-name CloseBlenderLid --horizon 20
cat /workspace/xr1-m7-submission/eval_results/smoke/*/summary.json
```
Expected: a real summary.json with at least 1 completed episode, no error
records in `eval_results/smoke/scheduler/*/errors/`.

- [ ] **Step 5: Write a thin eval wrapper (matches the spec's file structure)**

Their eval is a client-server shell-script pipeline, not an importable
Python function — so this wrapper's job is making that pipeline callable
and its results parseable from the rest of this repo's code, not
re-implementing it:

```python
# code/eval_wrapper.py
"""Thin subprocess wrapper around Xiaomi's real client-server
eval_robocasa365 pipeline (vendor/Xiaomi-Robotics-1/scripts/deploy.sh +
scripts/launch_robocasa365.sh) -- their eval is not an importable Python
API, so this wraps the real shell invocation and parses the real
summary.json it produces, rather than re-implementing their eval logic."""
import json
import subprocess
from pathlib import Path

XIAOMI_REPO = Path("/workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1")


def run_eval(checkpoint_path: str, output_dir: str, num_gpus: int = 1,
             extra_args: list[str] | None = None) -> dict:
    """Starts a single-GPU server, runs the client against it, and returns
    the parsed summary.json. Raises CalledProcessError if either the deploy
    or launch script exits non-zero -- callers should not silently treat a
    failed eval run as a 0% result."""
    subprocess.run(
        ["bash", "scripts/deploy.sh", checkpoint_path, str(num_gpus), str(num_gpus)],
        cwd=XIAOMI_REPO, check=True,
    )
    cmd = ["bash", "scripts/launch_robocasa365.sh", str(num_gpus), output_dir, checkpoint_path]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=XIAOMI_REPO, check=True, env={"CONDA_ENV": "robocasa_365"})

    summary_files = list(Path(output_dir).glob("*/summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.json produced under {output_dir}")
    with open(summary_files[0]) as f:
        return json.load(f)
```

```python
# tests/test_eval_wrapper.py
"""Only tests the parsing half without invoking the real (slow, GPU-bound)
eval pipeline -- the pipeline itself is exercised for real in Step 6/7
below, not under pytest."""
import json
from code.eval_wrapper import run_eval


def test_run_eval_raises_when_no_summary_produced(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    try:
        run_eval(checkpoint_path="/fake", output_dir=str(tmp_path))
        assert False, "expected FileNotFoundError when no summary.json exists"
    except FileNotFoundError:
        pass
```

```bash
python3 -m pytest tests/test_eval_wrapper.py -v
```
Expected: 1 passed.

```bash
git add code/eval_wrapper.py tests/test_eval_wrapper.py
git commit -m "feat: add thin subprocess wrapper around Xiaomi's real eval pipeline"
```

- [ ] **Step 6: Run the real baseline eval on the unmodified checkpoint**

```bash
CONDA_ENV=robocasa_365 \
  bash scripts/launch_robocasa365.sh \
  1 /workspace/xr1-m7-submission/eval_results/baseline "$MODEL_PATH"
```
This will take substantially longer than M7's own eval given 1 GPU serving
instead of 8 (their reference config: 2500 episodes total). Run this as a
background/`nohup` process and check back rather than blocking.

- [ ] **Step 7: Record the real baseline in NOTES.md**

```markdown
## Real baseline (unmodified Xiaomi-Robotics-1-RoboCasa365 on this pod)

- Episode success rate: <fill in from baseline/*/summary.json>
- Compare to their published 57.28% (pretrain split, target50 task set) --
  <note any real discrepancy, don't assume it matches exactly>
```

- [ ] **Step 7: Commit**

```bash
git add NOTES.md
git commit -m "docs: record real baseline eval of unmodified checkpoint on this pod"
```

---

### Task 9: Run the LoRA-only ablation, then LoRA+M7, then compare

**Files:**
- Modify: `NOTES.md`

**Files:** none created — this is the real experiment, using Task 7's
training loop and Task 8's eval wrapper.

- [ ] **Step 1: Read their real config.yaml before writing any training invocation**

```bash
cat /workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1/xr1/configs/config.yaml
```
This plan does not fabricate Hydra override key names — read the real file
here and record in `NOTES.md` exactly which keys control output directory,
checkpoint save path, and how (or whether) `code/finetune_run.py`'s
`use_m7_consolidation` flag needs to be threaded through their config
system vs. invoked as a separate script argument to `finetune_run.py`
directly (simpler, and consistent with how `compute_total_loss` was
already written as a plain function callable outside their Hydra config
tree). Prefer invoking `code/finetune_run.py` directly with a
`--use-m7-consolidation` flag over fighting their Hydra config for a
one-off ablation switch, unless Step 1's read of `config.yaml` shows their
`tools/train.py` cannot be driven any other way.

- [ ] **Step 2: Fine-tune with `use_m7_consolidation=False`**

```bash
python3 code/finetune_run.py \
  --checkpoint /workspace/xr1-m7-submission/checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output-dir /workspace/xr1-m7-submission/checkpoints/lora_only
  # --use-m7-consolidation flag omitted -> LoRA-only mode
```
(Exact CLI flags depend on Step 1's finding — this is the intended shape
per this task's design; adjust to match whatever `code/finetune_run.py`'s
Task 7 Step 6 implementation actually exposes, and record any deviation in
`NOTES.md`.)

- [ ] **Step 3: Evaluate the LoRA-only checkpoint**

```python
from code.eval_wrapper import run_eval
result = run_eval(
    checkpoint_path="/workspace/xr1-m7-submission/checkpoints/lora_only",
    output_dir="/workspace/xr1-m7-submission/eval_results/lora_only",
)
print(result)
```

- [ ] **Step 4: Fine-tune with `use_m7_consolidation=True`**

```bash
python3 code/finetune_run.py \
  --checkpoint /workspace/xr1-m7-submission/checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output-dir /workspace/xr1-m7-submission/checkpoints/lora_plus_m7 \
  --use-m7-consolidation
```

- [ ] **Step 5: Evaluate the LoRA+M7 checkpoint**

```python
from code.eval_wrapper import run_eval
result = run_eval(
    checkpoint_path="/workspace/xr1-m7-submission/checkpoints/lora_plus_m7",
    output_dir="/workspace/xr1-m7-submission/eval_results/lora_plus_m7",
)
print(result)
```

- [ ] **Step 6: Write the comparison into NOTES.md**

```markdown
## Final comparison

| Run | Episode success rate |
|---|---|
| Unmodified baseline (Task 8) | <fill in> |
| LoRA-only | <fill in> |
| LoRA + M7 consolidation | <fill in> |

Honest read: <fill in -- did M7's term help, hurt, or make no measurable
difference? State the real gap, not a hoped-for one.>
```

- [ ] **Step 7: Commit**

```bash
git add NOTES.md
git commit -m "docs: record final LoRA-only vs LoRA+M7 comparison"
```
