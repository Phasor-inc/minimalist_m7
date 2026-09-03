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
