"""Single, non-sequential LoRA fine-tuning run on top of frozen
Xiaomi-Robotics-1-RoboCasa365, with an optional M7 consolidation
regularization term. Supports two modes for the ablation required by the
design spec: LoRA-only (use_m7_consolidation=False) and LoRA+M7
(use_m7_consolidation=True) -- same script, one flag, so both runs are
guaranteed to share every other hyperparameter, PROVIDED both are invoked
with the same --seed (see main()'s seeding comments and NOTES.md "Task 7"
for why this matters and what it does/doesn't cover).

Real training-loop wiring (below compute_total_loss) loads the real frozen
xr1() model (mibot.models.VLA.XR1), the real downloaded RoboCasa365
checkpoint weights, injects LoRA into the VLM's language_model layers only
(the DiT action head is never touched), and trains against the real
xr1_post_train_demo data via their own JsonDataset/CustomCollate classes --
see NOTES.md "Task 7" for real smoke-test evidence.
"""
import argparse
import glob
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XR1_SRC = os.path.join(REPO_ROOT, "vendor", "Xiaomi-Robotics-1", "xr1")
if XR1_SRC not in sys.path:
    sys.path.insert(0, XR1_SRC)

from code.lora_adapters import inject_lora_into_vlm_layers  # noqa: E402
from code.m7_memory_core import M7MemoryCore  # noqa: E402
from code.pooled_hidden_adapter import PooledHiddenAdapter, pool_state_token_hidden_states  # noqa: E402


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


# ---------------------------------------------------------------------------
# Real training-loop wiring (Step 6 of Task 7 -- see plan for full context).
# Imports of the real vendored mibot classes are deferred to inside
# functions below is NOT done here; they're imported at module scope above
# so a plain `import code.finetune_run` (e.g. from the test file) fails
# loudly if the vendored repo / mibot deps aren't importable, rather than
# silently succeeding and only failing at call time.
# ---------------------------------------------------------------------------
from mibot.models.VLA.XR1 import xr1  # noqa: E402
from mibot.models.VLM.qwen3vl import STATE_ID  # noqa: E402
from mibot.data.datasets.json_dataset import JsonDataset  # noqa: E402
from mibot.data.collate.custom_collate import CustomCollate  # noqa: E402


def load_xr1_with_checkpoint(checkpoint_dir: str):
    """Builds the real xr1() model (random-init base architecture, same
    construction path as Task 2's inspect_vlm_modules.py) then loads the
    real downloaded RoboCasa365 checkpoint weights into it.

    The checkpoint is an HF-style deploy checkpoint (MiBoTForActionGeneration,
    defined in the checkpoint's own modeling_mibot.py) -- a structurally
    identical but separate class from mibot.models.VLA.XR1's xr1, built for
    HF `from_pretrained`/deployment rather than their Hydra+Lightning
    training entrypoint. Its safetensors state-dict keys (vlm.*, dit.*,
    state_projector.*, action_projector.*, action_output_layer.*,
    t_embedder.*, t_projector.*, sink.*) match xr1()'s own module attribute
    names 1:1 -- verified directly against the real
    model.safetensors.index.json's weight_map before writing this loader
    (1120 keys, top-level prefixes exactly
    {action_output_layer, action_projector, dit, sink, state_projector,
    t_embedder, t_projector, vlm}), not assumed.

    The checkpoint deliberately omits state_projector_choice /
    action_projector_choice / score_projector_choice (xr1's auxiliary
    choice-loss heads, used only during their own async training recipe,
    not needed for inference) -- those stay randomly initialized from
    xr1()'s own construction. load_state_dict is therefore called with
    strict=False, and the missing keys are asserted to be exactly that
    expected set below (not silently ignored)."""
    model = xr1()
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"no .safetensors shards found in {checkpoint_dir}")

    state_dict = {}
    for shard_path in shard_paths:
        state_dict.update(load_file(shard_path, device="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict  # free the ~9.5GB CPU-side copy before moving model to GPU

    if unexpected:
        raise RuntimeError(f"checkpoint has keys xr1() does not define: {unexpected}")

    # Expected-missing prefixes: xr1()'s auxiliary choice-loss heads
    # (state_projector_choice / action_projector_choice / score_projector_choice)
    # are not part of the deployed RoboCasa365 checkpoint at all.
    expected_missing_prefixes = (
        "state_projector_choice.",
        "action_projector_choice.",
        "score_projector_choice.",
    )
    # Expected-missing exact keys, discovered by actually running this loader
    # and reading the real mismatch, not assumed up front:
    #  - vlm.lm_head.weight: config.json has tie_word_embeddings=true, so HF's
    #    _from_config()/post_init() ties lm_head.weight to
    #    vlm.model.language_model.embed_tokens.weight (same underlying storage).
    #    The checkpoint's safetensors legitimately never saves it separately
    #    (standard HF tied-weights behavior) -- loading embed_tokens.weight
    #    already updates what lm_head.weight reads.
    #  - vlm.model.action_embed.weight / vlm.model.score_embed.weight: real,
    #    confirmed absence -- grepped directly against
    #    model.safetensors.index.json's full key list (1120 keys) and found
    #    zero matches for "action_embed"/"score_embed" anywhere in the
    #    released checkpoint. These embed the <a_i>/<score> input tokens used
    #    only by xr1's auxiliary choice-loss training path (loss_l1/loss_score);
    #    the deployed/released checkpoint apparently never shipped trained
    #    weights for them. They stay at xr1()'s own random initialization and
    #    stay frozen along with the rest of the base model in LoRA-only mode
    #    -- consistent with this task's "only LoRA (+adapter/memory_core)
    #    trainable" design, but worth flagging: loss_l1/loss_score will be
    #    computed against untrained embeddings for this checkpoint. Recorded
    #    in NOTES.md.
    expected_missing_exact = {
        "vlm.lm_head.weight",
        "vlm.model.action_embed.weight",
        "vlm.model.score_embed.weight",
    }
    unexplained_missing = [
        key
        for key in missing
        if not key.startswith(expected_missing_prefixes) and key not in expected_missing_exact
    ]
    if unexplained_missing:
        raise RuntimeError(
            "checkpoint is missing keys xr1() defines that are NOT the expected "
            f"set: {unexplained_missing}"
        )

    print(
        f"Loaded RoboCasa365 checkpoint from {checkpoint_dir}: "
        f"{len(shard_paths)} shard(s), {len(missing)} missing keys "
        "(choice-head params + tied lm_head + untrained action/score embed -- see load_xr1_with_checkpoint docstring)."
    )
    return model


def freeze_and_inject_lora(model, rank: int, alpha: float):
    """Freezes every parameter in the loaded model (base VLM + DiT), then
    injects LoRA into the VLM's language_model decoder layers only.
    model.dit is never passed to inject_lora_into_vlm_layers -- the DiT
    action head is structurally untouched by this call.

    Caller must seed torch's RNG before calling this (see main()) --
    LoRALinear's lora_A uses kaiming_uniform_ init, so the two ablation
    runs (LoRA-only vs LoRA+M7) only start from the same LoRA weights if
    the global RNG state entering this call is identical."""
    for param in model.parameters():
        param.requires_grad_(False)
    inject_lora_into_vlm_layers(model.vlm.model.language_model, rank=rank, alpha=alpha)
    return model


def verify_trainable_params(model, adapter=None, memory_core=None):
    """Real verification (not assumption) that only LoRA params (+
    adapter/memory_core params when M7 consolidation is active) are
    trainable -- the frozen base VLM and the entire DiT must contribute
    zero trainable parameters to the optimizer. Raises if that's violated."""
    dit_trainable = [name for name, param in model.dit.named_parameters() if param.requires_grad]
    if dit_trainable:
        raise RuntimeError(f"DiT has trainable parameters (must be frozen): {dit_trainable}")

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    non_lora_trainable = [name for name, _ in trainable if "lora_A" not in name and "lora_B" not in name]
    if non_lora_trainable:
        raise RuntimeError(f"non-LoRA base-model params are trainable: {non_lora_trainable}")

    lora_param_count = sum(param.numel() for _, param in trainable)
    total_param_count = sum(param.numel() for param in model.parameters())
    print(f"Trainable (LoRA) params in base model: {lora_param_count:,} / {total_param_count:,} total")

    extra_params = []
    if adapter is not None:
        extra_params += list(adapter.parameters())
    if memory_core is not None:
        extra_params += list(memory_core.parameters())
    if extra_params:
        extra_count = sum(param.numel() for param in extra_params)
        print(f"Trainable adapter+memory_core params: {extra_count:,}")

    return [param for _, param in trainable] + extra_params


def _ensure_data_symlink(data_dir: str):
    """The xr1_post_train_demo JSON files' video references are relative
    (e.g. "data/videos/json1_ego.mp4"), matching load_washer.yaml's own
    convention -- see NOTES.md Task 6. Ensures a `data` symlink exists in
    the current working directory pointing at data_dir, so decord's
    relative VideoReader(path) calls resolve regardless of where this
    script is invoked from."""
    link_path = os.path.join(os.getcwd(), "data")
    target = os.path.abspath(data_dir)
    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(target):
            return
        raise RuntimeError(
            f"'{link_path}' already exists as a symlink to a different target "
            f"({os.readlink(link_path)} != {target}); refusing to overwrite"
        )
    if os.path.exists(link_path):
        raise RuntimeError(f"'{link_path}' already exists and is not the expected symlink")
    os.symlink(target, link_path)


def build_dataloader(data_dir: str, batch_size: int, action_length: int, total_steps: int, num_workers: int, seed: int):
    """Builds the real JsonDataset + CustomCollate DataLoader against the
    real xr1_post_train_demo data, using load_washer.yaml's own real
    mean/std/q01/q99 normalization stats (loaded directly from the YAML,
    not re-typed by hand) -- only the `paths` list is overridden, to point
    at the actual downloaded json file locations rather than
    load_washer.yaml's own CWD-relative convention (see NOTES.md).

    Shuffling uses a dedicated torch.Generator seeded with `seed`, kept
    separate from the global torch RNG -- so the batch order a run sees is
    reproducible given the same seed regardless of how much global RNG
    state model/adapter construction consumed beforehand (see main())."""
    washer_yaml_path = os.path.join(XR1_SRC, "configs", "data", "load_washer.yaml")
    with open(washer_yaml_path) as yaml_file:
        washer_cfg = yaml.safe_load(yaml_file)
    train_cfg = washer_cfg["data"]["params"]["train_datasets"]

    json_dir = os.path.join(data_dir, "json")
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no json files found under {json_dir}")

    _ensure_data_symlink(data_dir)

    params = {
        "max_steps": total_steps,
        "train_datasets": {
            "batch_size": batch_size,
            "action_length": action_length,
            "paths": paths,
            "mean": train_cfg["mean"],
            "std": train_cfg["std"],
            "q01": train_cfg["q01"],
            "q99": train_cfg["q99"],
        },
    }
    dataset = JsonDataset(params)
    collate_fn = CustomCollate()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=generator,
    )


def move_batch_to_device(batch, device):
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


class _VlmOutputCapture:
    """Captures the real Qwen3VLForConditionalGeneration output object
    (with its real .hidden_states) via a forward hook on model.vlm, so the
    real last-hidden-state tensor xr1.forward() already computes internally
    (for its own action/score choice heads) can also be read here for M7
    pooling -- without re-implementing xr1.forward() or calling the VLM a
    second time."""

    def __init__(self, vlm_module):
        self.output = None
        self._handle = vlm_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output

    def remove(self):
        self._handle.remove()


def run_training_step(model, batch, adapter, memory_core, use_m7_consolidation, consolidation_weight, capture):
    capture.output = None
    xiaomi_out = model(batch, return_loss=True)
    xiaomi_loss = xiaomi_out["loss"]

    pooled_hidden = None
    if use_m7_consolidation:
        if capture.output is None or capture.output.hidden_states is None:
            raise RuntimeError("VLM forward hook did not capture hidden_states for this step")
        hidden_states = capture.output.hidden_states
        state_mask = batch["input_ids"] == STATE_ID
        pooled_hidden = pool_state_token_hidden_states(hidden_states, state_mask).float()

    total_loss, breakdown = compute_total_loss(
        xiaomi_loss=xiaomi_loss,
        pooled_hidden=pooled_hidden,
        adapter=adapter,
        memory_core=memory_core,
        use_m7_consolidation=use_m7_consolidation,
        consolidation_weight=consolidation_weight,
    )
    breakdown.update(
        {
            "loss_mse": xiaomi_out["loss_mse"].detach().item(),
            "loss_freq": xiaomi_out["loss_freq"].detach().item(),
            "loss_l1": xiaomi_out["loss_l1"].detach().item(),
            "loss_score": xiaomi_out["loss_score"].detach().item(),
        }
    )
    return total_loss, breakdown


def save_trainable_state(output_dir, model, adapter, memory_core, use_m7_consolidation, args):
    """Saves only the trainable deltas (LoRA weights, plus adapter/
    memory_core weights when M7 consolidation was active) -- not the full
    frozen 5B base checkpoint, which would be wasteful to duplicate."""
    os.makedirs(output_dir, exist_ok=True)
    lora_state_dict = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora_A" in name or "lora_B" in name)
    }
    payload = {
        "lora_state_dict": lora_state_dict,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "use_m7_consolidation": use_m7_consolidation,
        "base_checkpoint": os.path.abspath(args.checkpoint),
    }
    if use_m7_consolidation:
        payload["adapter_state_dict"] = adapter.state_dict()
        payload["memory_core_state_dict"] = memory_core.state_dict()
        payload["adapter_vlm_hidden_size"] = args.vlm_hidden_size
        payload["adapter_target_dim"] = args.adapter_target_dim
        payload["memory_core_memory_size"] = args.memory_size

    out_path = os.path.join(output_dir, "finetuned_delta.pt")
    torch.save(payload, out_path)
    print(f"Saved trainable deltas to {out_path}")
    return out_path


def _dit_weight_sum(model):
    """Real, exact float sum over every DiT parameter, used to verify the
    DiT was genuinely never touched during training (not just assumed
    frozen because it received no optimizer -- a stray in-place op
    elsewhere could still mutate it silently). Accumulates a running
    per-parameter sum rather than torch.cat-ing every DiT parameter into
    one transient multi-GB fp32 tensor -- avoids that memory spike on an
    already GPU-memory-contended pod (see NOTES.md GPU-contention note)."""
    total = 0.0
    for param in model.dit.parameters():
        total += param.detach().float().sum().item()
    return total


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the downloaded RoboCasa365 checkpoint dir.")
    parser.add_argument("--output-dir", required=True, help="Where to save the trainable-delta checkpoint.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(REPO_ROOT, "data_investigation", "xr1_post_train_demo"),
        help="Dir containing json/ and videos/ subdirs (xr1_post_train_demo layout).",
    )
    parser.add_argument("--use-m7-consolidation", action="store_true", help="Enable the LoRA+M7 ablation mode.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seeds torch/random RNGs (LoRA init, DataLoader shuffle order, xr1.forward()'s internal "
        "stochasticity). Task 9's LoRA-only vs LoRA+M7 ablation is only valid if BOTH runs use the same "
        "--seed -- see NOTES.md.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=20,
        help="Total optimizer steps for this run. NOTE: 20 is a smoke-test-scale placeholder -- Task 9 must "
        "override this with a real training budget (see NOTES.md).",
    )
    parser.add_argument("--batch-size", type=int, default=48, help="Matches load_washer.yaml's real batch_size.")
    parser.add_argument("--action-length", type=int, default=30, help="Matches load_washer.yaml's real action_length.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--consolidation-weight", type=float, default=0.1)
    parser.add_argument("--vlm-hidden-size", type=int, default=2560, help="Verified in Task 2 -- see NOTES.md.")
    parser.add_argument("--adapter-target-dim", type=int, default=512)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = build_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed up front so LoRA's kaiming_uniform_ init (drawn during
    # freeze_and_inject_lora below) is identical between a LoRA-only and a
    # LoRA+M7 run given the same --seed -- both runs call xr1()/checkpoint
    # loading/freeze_and_inject_lora identically before any mode branching,
    # so seeding here is enough to guarantee LoRA starts from the same
    # weights in both modes.
    _seed_everything(args.seed)

    print(f"Loading xr1() and RoboCasa365 checkpoint from {args.checkpoint} ...")
    model = load_xr1_with_checkpoint(args.checkpoint)
    freeze_and_inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)
    model.train()

    dit_weight_sum_before = _dit_weight_sum(model)

    adapter = memory_core = None
    if args.use_m7_consolidation:
        adapter = PooledHiddenAdapter(args.vlm_hidden_size, args.adapter_target_dim).to(device)
        memory_core = M7MemoryCore(
            input_size=args.adapter_target_dim,
            hidden_size=args.adapter_target_dim,
            memory_size=args.memory_size,
        ).to(device)
        memory_core.train()

    trainable_params = verify_trainable_params(model, adapter, memory_core)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    loader = build_dataloader(
        args.data_dir, args.batch_size, args.action_length, args.total_steps, args.num_workers, args.seed
    )

    # Re-seed immediately before the training loop. LoRA+M7 mode's
    # adapter/memory_core construction above draws extra RNG samples (their
    # own nn.Linear/xavier_uniform_ init) that LoRA-only mode never draws --
    # left alone, that would desynchronize the global RNG state the two
    # modes enter the loop with, which would in turn desynchronize
    # xr1.forward()'s own internal stochasticity (prefix_length via
    # random.randint/random.random, the Beta-sampled flow-matching timestep,
    # torch.randn_like noise) between the two ablation runs -- differences
    # attributable to RNG bookkeeping, not to use_m7_consolidation. The
    # DataLoader's batch order is unaffected either way (build_dataloader
    # uses its own dedicated seeded torch.Generator, not the global RNG),
    # so this re-seed only re-synchronizes xr1.forward()'s per-step draws.
    _seed_everything(args.seed)

    capture = _VlmOutputCapture(model.vlm)
    step = 0
    start_time = time.time()
    try:
        for batch in loader:
            if step >= args.total_steps:
                break
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            total_loss, breakdown = run_training_step(
                model, batch, adapter, memory_core, args.use_m7_consolidation, args.consolidation_weight, capture
            )
            total_loss.backward()
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(f"step {step}/{args.total_steps} total_loss={total_loss.item():.6f} breakdown={breakdown} elapsed={elapsed:.1f}s")
    finally:
        capture.remove()

    dit_weight_sum_after = _dit_weight_sum(model)
    if dit_weight_sum_after != dit_weight_sum_before:
        raise RuntimeError(
            f"DiT weights changed during training (before={dit_weight_sum_before}, "
            f"after={dit_weight_sum_after}) -- the DiT must remain frozen/untouched."
        )
    print(f"Verified DiT untouched: weight-sum before={dit_weight_sum_before} after={dit_weight_sum_after}")

    save_trainable_state(args.output_dir, model, adapter, memory_core, args.use_m7_consolidation, args)


if __name__ == "__main__":
    main()
