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

# --- Extended inspection (real evidence for the sufficiency decision) ---
print("\n" + "=" * 70)
print("EXTENDED INSPECTION (all files, structure/dims/task-diversity check)")
print("=" * 70)

video_files = sorted((demo_dir / "videos").glob("*.mp4")) if (demo_dir / "videos").exists() else []
print(f"\nVideo files found: {len(video_files)}")
for v in video_files:
    print(f"  {v.name}  ({v.stat().st_size / 1e6:.1f} MB)")

for f in sorted(json_files):
    with open(f) as fh:
        data = json.load(fh)
    print(f"\n--- {f.name} ---")
    if isinstance(data, list):
        print(f"  type: list, length: {len(data)}")
        item = data[0] if data else None
    elif isinstance(data, dict):
        print(f"  type: dict, top-level keys: {list(data.keys())}")
        item = data
    else:
        print(f"  type: {type(data)}")
        item = None

    if isinstance(item, dict):
        print(f"  sample entry keys: {list(item.keys())}")
        # Look for a state/action-bearing conversation structure
        conv = item.get("conversations") or item.get("conversation")
        if conv:
            print(f"  conversation turns: {len(conv)}")
            for turn in conv[:4]:
                role = turn.get("from", turn.get("role", "?"))
                val = turn.get("value", turn.get("content", ""))
                val_preview = (str(val)[:150] + "...") if len(str(val)) > 150 else str(val)
                print(f"    [{role}] {val_preview}")
        # Look directly for state/action arrays and report dims
        for key in ("state", "action", "states", "actions"):
            if key in item:
                v = item[key]
                if isinstance(v, list):
                    inner_len = len(v[0]) if v and isinstance(v[0], list) else None
                    print(f"  '{key}': outer len={len(v)}, inner dim={inner_len}")
