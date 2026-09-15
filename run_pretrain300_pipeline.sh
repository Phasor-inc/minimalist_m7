#!/usr/bin/env bash
set -uo pipefail
cd /workspace/xr1-m7-submission

LOGFILE=/workspace/xr1-m7-submission/pretrain300_pipeline.log
exec >>"$LOGFILE" 2>&1

echo "=== run_pretrain300_pipeline.sh started at $(date -u +%FT%TZ) ==="

source miniconda3/etc/profile.d/conda.sh
conda activate mibot

fatal() {
    echo "FATAL: $1"
    tmux kill-session -t model_servers 2>/dev/null || true
    exit 1
}

# ---------- Leg 1: LoRA-only on pretrain300 ----------
echo "=== Training lora_only_pretrain300 ==="
python3 -u -m code.finetune_run \
  --checkpoint checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output-dir checkpoints/lora_only_pretrain300 \
  --seed 42 --total-steps 220 --batch-size 48 --num-workers 0 \
  --data-source robocasa --robocasa-task-set pretrain300
[ $? -eq 0 ] || fatal "lora_only_pretrain300 training failed"
echo "Training complete: lora_only_pretrain300"

echo "=== Merging lora_only_pretrain300 ==="
python3 -u -m code.merge_lora_for_eval \
  --delta checkpoints/lora_only_pretrain300/finetuned_delta.pt \
  --output-dir checkpoints/lora_only_pretrain300_merged
[ $? -eq 0 ] || fatal "lora_only_pretrain300 merge failed"
echo "Merge complete: lora_only_pretrain300"

# ---------- Leg 2: LoRA+M7 on pretrain300 ----------
echo "=== Training lora_plus_m7_pretrain300 ==="
python3 -u -m code.finetune_run \
  --checkpoint checkpoints/Xiaomi-Robotics-1-RoboCasa365 \
  --output-dir checkpoints/lora_plus_m7_pretrain300 \
  --seed 42 --total-steps 220 --batch-size 48 --num-workers 0 \
  --data-source robocasa --robocasa-task-set pretrain300 --use-m7-consolidation
[ $? -eq 0 ] || fatal "lora_plus_m7_pretrain300 training failed"
echo "Training complete: lora_plus_m7_pretrain300"

echo "=== Merging lora_plus_m7_pretrain300 ==="
python3 -u -m code.merge_lora_for_eval \
  --delta checkpoints/lora_plus_m7_pretrain300/finetuned_delta.pt \
  --output-dir checkpoints/lora_plus_m7_pretrain300_merged
[ $? -eq 0 ] || fatal "lora_plus_m7_pretrain300 merge failed"
echo "Merge complete: lora_plus_m7_pretrain300"

# ---------- Smoke test both before committing to full sweeps ----------
smoke_test() {
    local checkpoint_dir=$1
    local label=$2
    echo "=== Smoke test: $label ==="
    cd vendor/Xiaomi-Robotics-1
    bash scripts/deploy.sh "/workspace/xr1-m7-submission/checkpoints/${checkpoint_dir}" 1 1
    LOADED=0
    for i in $(seq 1 60); do
        if tmux capture-pane -p -t model_servers:server-00 2>/dev/null | grep -q "Server running"; then
            LOADED=1
            break
        fi
        sleep 15
    done
    if [ "$LOADED" -ne 1 ]; then
        cd /workspace/xr1-m7-submission
        fatal "$label smoke-test server did not report ready within 15 minutes"
    fi
    RUN_ID="smoke_${label}_$(date +%Y%m%d-%H%M%S)"
    CONDA_ENV=robocasa_365 NUM_TRIALS=5 RUN_ID="$RUN_ID" bash scripts/launch_robocasa365.sh 1 \
      "/workspace/xr1-m7-submission/eval_results/smoke_${label}" \
      "/workspace/xr1-m7-submission/checkpoints/${checkpoint_dir}" \
      --task-name CloseFridge --task-name OpenCabinet --task-name NavigateKitchen
    cd /workspace/xr1-m7-submission
    tmux kill-session -t model_servers 2>/dev/null || true
    sleep 10
    SUMMARY="eval_results/smoke_${label}/${RUN_ID}/summary.json"
    if [ ! -f "$SUMMARY" ]; then
        fatal "$label smoke summary.json not found at $SUMMARY"
    fi
    RATE=$(python3 -c "import json; print(json.load(open('$SUMMARY')).get('episode_success_rate', -1))")
    echo "$label smoke episode_success_rate: $RATE"
    GATE_OK=$(python3 -c "print(1 if float('$RATE') >= 0.3 else 0)")
    if [ "$GATE_OK" != "1" ]; then
        fatal "$label smoke success rate $RATE is below the 0.3 gate -- NOT launching full eval"
    fi
    echo "$label smoke gate passed ($RATE >= 0.3)."
}

smoke_test "lora_only_pretrain300_merged" "lora_only_pretrain300"
smoke_test "lora_plus_m7_pretrain300_merged" "lora_plus_m7_pretrain300"

# ---------- Full eval sweeps, sequential (GPU can't fit two 3-server sweeps) ----------
full_eval() {
    local checkpoint_dir=$1
    local label=$2
    echo "=== Full eval sweep: $label (2500 episodes, 50 tasks) ==="
    cd vendor/Xiaomi-Robotics-1
    bash scripts/deploy.sh "/workspace/xr1-m7-submission/checkpoints/${checkpoint_dir}" 3 1
    for w in server-00 server-01 server-02; do
        LOADED=0
        for i in $(seq 1 60); do
            if tmux capture-pane -p -t model_servers:$w 2>/dev/null | grep -q "Server running"; then
                LOADED=1
                break
            fi
            sleep 15
        done
        if [ "$LOADED" -ne 1 ]; then
            cd /workspace/xr1-m7-submission
            fatal "$label: $w did not report ready within 15 minutes"
        fi
    done
    echo "$label: all 3 full-eval servers ready."
    CONDA_ENV=robocasa_365 bash scripts/launch_robocasa365.sh 3 \
      "/workspace/xr1-m7-submission/eval_results/${label}" \
      "/workspace/xr1-m7-submission/checkpoints/${checkpoint_dir}"
    cd /workspace/xr1-m7-submission
    tmux kill-session -t model_servers 2>/dev/null || true
    sleep 10
}

full_eval "lora_only_pretrain300_merged" "lora_only_pretrain300"
full_eval "lora_plus_m7_pretrain300_merged" "lora_plus_m7_pretrain300"

echo "=== run_pretrain300_pipeline.sh finished at $(date -u +%FT%TZ) ==="
