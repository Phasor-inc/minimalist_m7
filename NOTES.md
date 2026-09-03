# xr1-m7-submission — verified environment facts

- **GPU headroom at setup time (Step 1, before any GPU-related work):** RTX 4090, driver 610.43.02. `nvidia-smi` reported `utilization.gpu=2%, memory.used=37789 MiB, memory.total=49140 MiB` — only ~11.1 GB free. This is within the ~20GB threshold that requires a stop-and-check per the task instructions. Checked `ps aux | grep -E "m7_robocasa|eval_full_parallel"` and confirmed the M7 eval sweep (7 parallel `m7_robocasa_eval.py` rollout jobs), a flowmatching/chunking training run (`m7_robocasa_flowmatching_run.py`, running since Sep 2), and a JEPA training run (`m7_robocasa_jepa_run.py`, also since Sep 2) were all still actively running under `/workspace/M7` — none had finished. None of these processes were touched, killed, or interacted with. By the time env setup finished, GPU usage had dropped to `memory.used=19204 MiB / 49140 MiB` (~30GB free) as some eval-sweep jobs completed on their own — but the pod's GPU is shared and headroom should be re-checked immediately before any GPU-touching step in later tasks (e.g. LoRA fine-tuning), not assumed from this reading.

- **Xiaomi repo commit:** cloned `https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git` to `vendor/Xiaomi-Robotics-1` (not committed — see .gitignore). Pinned commit: `cfcab04e662514644d62a4f3cfcce97ce83b90b2`, dated Wed Aug 26 12:15:46 2026 +0800.

- **Checkpoint size on disk:** `hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --local-dir checkpoints/Xiaomi-Robotics-1-RoboCasa365` completed successfully. Real size on disk: **9.5G** (`du -sh`). Contents verified as a real checkpoint: 3 safetensors shards (model-00001/2/3-of-00003.safetensors, ~10GB total incl. index), config.json, tokenizer.json/vocab.json/merges.txt, and custom `modeling_mibot.py`/`configuration_mibot.py`/`processing_mibot.py` (matches the `trust_remote_code=True` requirement noted in their DEPLOYMENT.md). `/workspace` mount has 336T free (`df -h`), confirming ample room.

- **transformers version installed by mibot deploy env:** **4.57.1** (`pip show transformers` inside the `mibot` conda env) — the exact version DEPLOYMENT.md requires ("must be exactly 4.57.1 — other versions are unverified").

## Deploy (`mibot`) environment setup — deviations from DEPLOYMENT.md as written

DEPLOYMENT.md assumes conda/mamba is already present. This pod had **no conda, no mamba, and no python3.12** (only python3.11 and python3.10 via apt). Real commands used, in order:

1. Installed Miniconda project-locally (not touching any other project or system Python):
   ```
   curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash /tmp/miniconda.sh -b -p /workspace/xr1-m7-submission/miniconda3
   ```
2. `conda create -n mibot python=3.12 -y` initially failed twice:
   - First: `NoChannelsConfiguredError` — no channels configured out of the box. Added conda-forge.
   - Second: `CondaToSNonInteractiveError` — the installer's bundled `defaults` channel (anaconda.com pkgs/main, pkgs/r) requires interactive Terms-of-Service acceptance, which isn't appropriate to auto-accept non-interactively. Fixed by overwriting `miniconda3/.condarc` to use **conda-forge only** (removed `defaults`) rather than accepting ToS on the user's behalf.
   - Then `conda create -n mibot python=3.12 -y` succeeded: Python 3.12.14.
3. `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` — as written, succeeded (took several minutes for the large cu128 wheels + bundled NVIDIA CUDA 12.8 libs). Verified: `torch 2.8.0+cu128`, `torch.cuda.is_available() == True`.
4. `pip uninstall -y ninja && pip install ninja` then `pip install <flash-attn 2.8.3 prebuilt wheel URL from DEPLOYMENT.md>` — as written, succeeded without needing a source build. Verified: `flash_attn 2.8.3`.
5. `apt-get install -y libegl1 libgl1 libgles2` — DEPLOYMENT.md prefixes this with `sudo`, but `sudo` is not installed on this pod (already running as root). Ran without `sudo`. `libegl1` and `libgl1` were already present system-wide; `libgles2` was newly installed.

Final verified versions in the `mibot` env: torch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128, transformers 4.57.1, flash_attn 2.8.3.

Step 2 of DEPLOYMENT.md (launching the inference server via `scripts/deploy.sh` across multiple GPU ports) was **not** exercised in this task — Task 1 only calls for the deploy environment to be set up, not the server to be launched. That is left for whichever later task actually needs a running inference server.

## Task 2 — real VLM module names and hidden size (verified from loaded model)

- **vlm_hidden_size (verified from loaded model):** `2560` (`model.vlm.config.text_config.hidden_size`, from the real `Qwen/Qwen3-VL-4B-Instruct` config fetched by `xr1()`'s `_build_model()` — the `xr1` constructor builds the VLM from `Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")` with random init weights, i.e. base architecture, not the downloaded RoboCasa365 checkpoint weights — that's what the plan's script does and is expected).

- **Real LoRA target module names (verified, exact dotted names from `model.vlm.model.language_model.layers[0]`):**
  ```
  self_attn.q_proj: Linear
  self_attn.k_proj: Linear
  self_attn.v_proj: Linear
  self_attn.o_proj: Linear
  mlp.gate_proj: Linear
  mlp.up_proj: Linear
  mlp.down_proj: Linear
  ```
  Confirmed identical to the standard Qwen3 naming assumed in the plan — nothing unexpected on the attention/MLP projection names.

  Additional real submodules present in the layer that are **not** candidate LoRA targets but are worth recording since they were part of the real inspection: `self_attn.q_norm` and `self_attn.k_norm` (`LigerRMSNorm`), `mlp.act_fn` (`SiLUActivation`), `input_layernorm` and `post_attention_layernorm` (both `LigerRMSNorm`). Note the norm layers are `LigerRMSNorm` (from `liger-kernel`), not the vanilla HF `Qwen3RMSNorm` — a byproduct of this checkpoint's code using Liger kernel fused ops. Not relevant to LoRA target selection but potentially relevant if any later task tries to pattern-match on norm layer class names.

- **Real dependency gap found and fixed:** `mibot.models.VLA.XR1.xr1` (the vendored source-tree import path required by this task's script) needs the *full* vendored-repo training deps in `vendor/Xiaomi-Robotics-1/xr1/assets/requirements.txt`, not just the lightweight deploy-server deps DEPLOYMENT.md has you install (which Task 1 correctly followed as written). Importing `mibot.models.VLA.XR1` transitively imports `mibot.models.runner.base_runner` (needs `lightning`) and `mibot.models` `__init__.py` (needs `mmengine.Registry`), among others. Installed the remaining pinned packages from `xr1/assets/requirements.txt` into the `mibot` conda env (skipping `torch==2.8.0`/`torchvision==0.23.0`/`transformers==4.57.1`, which the installed `2.8.0+cu128`/`0.23.0+cu128`/`4.57.1` already satisfy): `mmengine==0.10.7`, `lightning==2.5.3`, `deepspeed==0.18.9` (built from source via pip, succeeded), `accelerate==1.11.0`, `safetensors==0.6.2`, `liger-kernel==0.6.5`, `numpy==2.1.3`, `Pillow==11.3.0`, `decord==0.6.0`, `omegaconf==2.3.0`, `hydra-core==1.3.2`, `wandb==0.23.1`, `tensorboard==2.20.0`. This downgraded `safetensors` 0.8.0→0.6.2, `Pillow` 12.3.0→11.3.0, `numpy` 2.5.2→2.1.3 to match the vendored repo's pins (torch/torchvision/transformers were untouched). The `mibot` env now matches the vendored repo's full `requirements.txt`, superset of what DEPLOYMENT.md's deploy-only instructions cover.

- **GPU headroom immediately before running the script (fresh check, not reused from Task 1):** `utilization.gpu=24%, memory.used=26086 MiB, memory.total=49140 MiB` — ~22.4GB free, sufficient for the ~5B-parameter bf16 model build (`xr1()` builds the full model including the 36-layer DiT, as expected per the task). After the script exited, GPU memory returned to baseline (`memory.free=24998 MiB`) — confirmed no leaked GPU memory from this task.

## Task 5 — pooled hidden-state adapter: NaN-on-empty-row bug found and fixed, packing note for Task 7

- **Bug found by review, fixed before Task 7 wires this in:** `pool_state_token_hidden_states`'s 3D (batched) branch originally only validated `state_token_mask.any()` across the *whole* tensor, not per-row. An all-False mask on a single row in an otherwise-valid batch (e.g. shape `[2,3,2]`, mask `[[True,False,True],[False,False,False]]`) silently produced `[[2.,2.],[nan,nan]]` instead of raising — `.mean(dim=0)` over an empty selection is a silent NaN, not an exception. This is now fixed with a per-row check (`state_token_mask[i].any()`) that raises `ValueError(f"state_token_mask row {i} has no True positions to pool over")` identifying the offending row. Verified fixed: reproducing the exact `[2,3,2]` scenario above now raises instead of returning NaN. Added 3D test coverage (`tests/test_pooled_hidden_adapter.py`): a batched happy-path test and a per-row-empty-mask test — the 3D branch had zero test coverage before this fix (all original 3 tests only exercised the 2D single-sample path).

- **Why this matters for Task 7:** `compute_total_loss` (not yet built) has no runtime `isfinite` guard per current design. A silent NaN from this adapter would flow uncaught into the loss and permanently poison Adam's optimizer state (NaN moment estimates don't self-heal) — the kind of bug that surfaces hours into a training run with no clear proximate cause. Task 7's implementer should either keep relying on this ValueError as the guard (i.e. never suppress/catch it upstream without deliberately deciding what an empty-mask row *means* for that batch), or add an explicit `isfinite` check in `compute_total_loss` as defense in depth.

- **Real batch shape at Task 7's call site is probably NOT a genuine multi-row batch.** `vendor/Xiaomi-Robotics-1/xr1/mibot/data/collate/custom_collate.py` **packs** samples together via `torch.cat(dim=1)` (concatenating along the sequence dimension) rather than padding samples into independent rows of a batch, and it drops any sample that would overflow the packed length whole (not truncating it). This means the real `hidden_states`/`state_token_mask` tensors reaching this adapter from the primary JSON-dataset path (`xr1/mibot/data/datasets/json_dataset.py`) are likely shaped closer to `[1, total_packed_length, hidden]` — a single packed row — than to `[batch_size>1, seq_len, hidden]` with independent per-sample rows. So the 3D per-row branch (and its per-row-empty-mask guard) may not actually be exercised by the primary JSON-dataset data path at all.
  - It **would** be exercised by Task 6's fallback LeRobot-conversion path, *if* that path produces genuine multi-row batches — but that path won't automatically inherit the primary path's implicit guarantee that every sample has a `<state>` token, unless whoever writes that converter deliberately preserves/enforces it. Worth double-checking when Task 6 is implemented, and worth a deliberate look at Task 7 Step 6 (real training-loop wiring) to confirm which shape actually shows up at this call site before assuming the per-row branch is dead code or live code.
