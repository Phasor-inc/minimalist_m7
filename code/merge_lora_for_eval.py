"""Merges a code/finetune_run.py-produced finetuned_delta.pt (LoRA-only or
LoRA+M7 -- both save the same lora_state_dict shape, M7's extra
adapter/memory_core state is never part of the served inference graph, see
NOTES.md Task 7) into the frozen base RoboCasa365 checkpoint's own Linear
weights, and writes out a new, fully self-contained HF-style checkpoint
directory that Xiaomi's deploy.sh/AutoModel.from_pretrained(trust_remote_code)
path can load directly -- exactly like the unmodified baseline checkpoint,
just with the target attn/mlp Linear weights numerically updated in place.

Why this exists: run_eval()/deploy.sh load a full HF-style checkpoint dir
(config.json + safetensors + modeling_mibot.py etc.) via AutoModel, which
constructs a plain MiBoTForActionGeneration with ordinary nn.Linear layers --
it has no notion of LoRALinear wrappers. finetune_run.py's own
save_trainable_state() only ever saves the small LoRA delta (by design, to
avoid duplicating the ~9.5GB frozen base checkpoint on every ablation leg),
so evaluating either fine-tuned leg against the official pipeline requires
materializing one real merged checkpoint first. All CPU-only -- no GPU
needed for this step, safe to run at any time regardless of GPU contention
from a concurrent training/eval job.
"""
import argparse
import os
import shutil
import json

import torch
from safetensors.torch import save_file

from code.finetune_run import load_xr1_with_checkpoint, freeze_and_inject_lora
from code.lora_adapters import LoRALinear, TARGET_ATTN_MODULES, TARGET_MLP_MODULES


def merge_and_unwrap(language_model):
    """In place: for every LoRALinear this run injected, folds
    scaling * (lora_B @ lora_A) into base_linear.weight, then puts the plain
    base_linear back in the parent module -- restoring the exact original
    architecture (ordinary nn.Linear everywhere), just with updated weights."""
    merged_count = 0
    for layer in language_model.layers:
        for attn_name in TARGET_ATTN_MODULES:
            mod = getattr(layer.self_attn, attn_name)
            if isinstance(mod, LoRALinear):
                delta = mod.scaling * (mod.lora_B.data @ mod.lora_A.data)
                mod.base_linear.weight.data = mod.base_linear.weight.data + delta
                setattr(layer.self_attn, attn_name, mod.base_linear)
                merged_count += 1
        for mlp_name in TARGET_MLP_MODULES:
            mod = getattr(layer.mlp, mlp_name)
            if isinstance(mod, LoRALinear):
                delta = mod.scaling * (mod.lora_B.data @ mod.lora_A.data)
                mod.base_linear.weight.data = mod.base_linear.weight.data + delta
                setattr(layer.mlp, mlp_name, mod.base_linear)
                merged_count += 1
    return merged_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, help="Path to finetuned_delta.pt")
    parser.add_argument("--output-dir", required=True, help="Where to write the merged, deployable checkpoint")
    args = parser.parse_args()

    payload = torch.load(args.delta, map_location="cpu", weights_only=True)
    base_checkpoint = payload["base_checkpoint"]
    lora_rank = payload["lora_rank"]
    lora_alpha = payload["lora_alpha"]
    lora_state_dict = payload["lora_state_dict"]
    print(f"Loaded delta: base_checkpoint={base_checkpoint}, lora_rank={lora_rank}, "
          f"lora_alpha={lora_alpha}, use_m7_consolidation={payload['use_m7_consolidation']}, "
          f"{len(lora_state_dict)} lora tensors")

    model = load_xr1_with_checkpoint(base_checkpoint)
    freeze_and_inject_lora(model, rank=lora_rank, alpha=lora_alpha)

    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    assert not unexpected, f"unexpected keys loading lora_state_dict: {unexpected}"
    missing_lora = [k for k in missing if "lora_A" in k or "lora_B" in k]
    assert not missing_lora, f"LoRA keys present in model but not in saved delta: {missing_lora}"
    print(f"Loaded {len(lora_state_dict)} real fine-tuned LoRA tensors into the freshly-injected LoRA structure.")

    merged_count = merge_and_unwrap(model.vlm.model.language_model)
    print(f"Merged and unwrapped {merged_count} LoRALinear modules back to plain nn.Linear.")

    # Sanity: no LoRALinear should remain anywhere in the model.
    remaining = [name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    assert not remaining, f"LoRALinear modules remained unmerged: {remaining}"

    # Restrict the exported state dict to exactly the real baseline checkpoint's
    # own key set (its model.safetensors.index.json weight_map) -- this
    # guarantees (a) no tied-weight (lm_head.weight) safetensors aliasing
    # crash, since the original checkpoint never saved that key either, and
    # (b) an apples-to-apples deploy-time "missing keys" story identical to
    # the baseline eval (Task 8) for every non-LoRA-touched key -- the only
    # real difference in what gets deployed is the merged attn/mlp weights.
    with open(os.path.join(base_checkpoint, "model.safetensors.index.json")) as f:
        index = json.load(f)
    original_keys = set(index["weight_map"].keys())

    full_state_dict = model.state_dict()
    export_state_dict = {k: v.detach().cpu().contiguous() for k, v in full_state_dict.items() if k in original_keys}
    dropped = set(full_state_dict.keys()) - original_keys
    print(f"Exporting {len(export_state_dict)}/{len(full_state_dict)} tensors "
          f"(dropped {len(dropped)} keys not present in baseline checkpoint's own weight_map, "
          "e.g. tied lm_head / untrained action-score-embed / choice heads -- matches baseline exactly).")
    missing_from_export = original_keys - set(export_state_dict.keys())
    assert not missing_from_export, f"real baseline keys missing from merged export: {missing_from_export}"

    os.makedirs(args.output_dir, exist_ok=True)
    skip_names = set(index["weight_map"].values()) | {"model.safetensors.index.json", ".cache"}
    for name in os.listdir(base_checkpoint):
        if name in skip_names:
            continue
        src = os.path.join(base_checkpoint, name)
        if os.path.isdir(src):
            continue
        shutil.copy2(src, os.path.join(args.output_dir, name))

    save_file(export_state_dict, os.path.join(args.output_dir, "model.safetensors"))
    print(f"Wrote merged, deployable checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
