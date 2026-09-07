# Copyright (C) 2026 -- this project (xr1-m7-submission), Task 9 pivot.
"""Real lerobot-format RoboCasa365 training data loader.

Bridges the real, official-format RoboCasa lerobot data already on this pod
(/workspace/data/robocasa_datasets/v1.0/pretrain/{atomic,composite}/<Task>/
<date>/lerobot/ -- LeRobot v2.1: one parquet file per episode plus 3
per-episode camera mp4s) into the exact per-sample dict schema
mibot.data.collate.custom_collate.CustomCollate and
mibot.models.VLA.XR1.xr1's own training forward() already consume (the same
schema mibot.data.datasets.json_dataset.JsonDataset produces for the
xr1_post_train_demo data path -- see code/finetune_run.py and NOTES.md
"Task 7"/"Task 9 pivot").

Why this is NOT a thin reuse of JsonDataset's own compose_action/
compose_state/build_action_mask/normalize_action_parts/normalize_quantile
(mibot/utils/io.py) -- read directly, not assumed:
  - Those functions hardcode ACTION_PARTS, a dual-arm-humanoid layout
    (left_ee_pos/aa/gripper, right_ee_pos/aa/gripper, waist, base -- dims
    0:20 of the model's 60-dim action head) matching xr1_post_train_demo's
    "Load washer" data. RoboCasa's PandaOmron embodiment is a completely
    different action space and would be silently wrong if forced through
    that layout.
  - The released Xiaomi-Robotics-1-RoboCasa365 checkpoint's own
    preprocessor_config.json action_config["robocasa365"] (read directly
    off the downloaded checkpoint on this pod) confirms the REAL convention
    that checkpoint's own action head was actually calibrated against:
    mean=0, std=1 (raw, unnormalized) on exactly the first 12 of 60 action
    dims, std=0 (unused/masked) on the remaining 48. This loader reproduces
    that exact real convention so LoRA fine-tuning stays consistent with
    what the frozen base model already expects, rather than inventing a new
    one.
  - eval_robocasa365/entry.py (the checkpoint's own real, working eval
    client, vendor/Xiaomi-Robotics-1/eval_robocasa365/entry.py) confirms the
    real state convention: a 14-dim EE-first state
    (end_effector_position_relative(3), end_effector_rotation_relative as
    axis-angle(3), gripper_qpos(2), base_position(3), base_rotation as
    axis-angle(3)) built via observation_to_state()/quat_xyzw_to_axis_angle()
    zero-padded into the model's 60-dim state head. quat_xyzw_to_axis_angle
    is reproduced byte-for-byte below (copied, not imported -- entry.py
    lives under eval_robocasa365/, which pulls in deploy/client.py, a live
    inference-server networking dependency finetune_run.py has no business
    importing).
  - The real RoboCasa lerobot data's own meta/modality.json (read directly
    off .../CloseFridge/20250819/lerobot/meta/modality.json, confirmed
    identical layout on other sampled tasks) gives the real raw-field slice
    layout matching this exactly: observation.state[7:10]=
    end_effector_position_relative, [10:14]=end_effector_rotation_relative
    (quat xyzw), [14:16]=gripper_qpos, [0:3]=base_position, [3:7]=
    base_rotation (quat xyzw); action[5:8]=end_effector_position,
    [8:11]=end_effector_rotation, [11:12]=gripper_close, [0:4]=base_motion,
    [4:5]=control_mode (12-dim total) -- i.e. this dataset's raw "action"
    column IS already the real robocasa365 12-dim action-head target, no
    transform needed beyond zero-padding into the 60-dim head.
  - xr1()'s own forward() (mibot/models/VLA/XR1.py, compute_flow_loss) was
    read directly and confirmed to apply action_mask as a genuine
    elementwise (timestep AND action-dim) boolean mask:
    `(F.mse_loss(pred, target, reduction="none") * weight)[action_mask].mean()`
    -- so zero-masking the unused 48/60 action dims here means they
    contribute exactly zero loss, not a near-zero pull toward an arbitrary
    padding value.

What IS reused as-is, confirmed embodiment-agnostic by reading it directly:
  - mibot.data.datasets.json_dataset.JsonDataset._messages (a @staticmethod,
    builds Qwen chat-format messages from a conversations list + image pool,
    no washer-specific logic).
  - mibot.utils.io.resize_image (plain PIL resize, no washer-specific
    logic).

Memory note (real constraint on this pod, not theoretical -- see NOTES.md):
this loader is deliberately built to avoid ever materializing the full
~2.16M-frame / 3,963-episode real dataset in memory. __init__ only indexes
episodes (one small dict per episode, ~3,963 total across the 34 real
task roots -- negligible), and builds a fixed-length `samples` list of
(episode_index, frame_index) tuples sized to the real training budget
(`total_steps * batch_size`, e.g. ~10k for a 220-step/batch-48 run), not to
the dataset size. Per-episode state/action arrays are read lazily via a
bounded LRU cache (maxsize=64, ~a few hundred KB per episode at most).
Video frames are decoded one at a time via a fresh decord.VideoReader per
read (never a whole video held in memory) -- the same pattern
JsonDataset._images already uses for the xr1_post_train_demo path.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation

# Self-contained sys.path setup (matches code/finetune_run.py's own
# top-of-module pattern exactly) -- don't rely on finetune_run.py having
# already been imported first to put the vendored mibot source tree on
# sys.path, since this module is also imported directly by its own test
# file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XR1_SRC = os.path.join(_REPO_ROOT, "vendor", "Xiaomi-Robotics-1", "xr1")
if _XR1_SRC not in sys.path:
    sys.path.insert(0, _XR1_SRC)

from mibot.data.datasets.json_dataset import JsonDataset  # noqa: E402
from mibot.utils.io import resize_image  # noqa: E402

# Real, verified facts about the model's fixed-width action/state heads and
# the robocasa365 embodiment's real sub-slice within them (see module
# docstring for the checkpoint/entry.py evidence).
ROBOCASA_ACTION_DIM = 60
ROBOCASA_STATE_DIM = 60
ROBOCASA_ACTIVE_ACTION_DIMS = 12
ROBOCASA_ACTIVE_STATE_DIMS = 14

# Real camera video keys, matching this dataset's own
# meta/info.json "features" keys AND eval_robocasa365/entry.py's
# CAMERA_KEYS (modulo entry.py's "video." client-side prefix, which is a
# deploy/client.py wire-protocol detail, not part of the on-disk lerobot
# schema).
CAMERA_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)

# Copied directly from /workspace/M7/code/m7_robocasa_eval.py's
# OFFICIAL_ATOMIC_SEEN_TASKS / OFFICIAL_COMPOSITE_SEEN_TASKS (read from that
# real source on this pod, not guessed -- see NOTES.md). Deliberately NOT
# imported cross-project from M7/code, which is a separate, unrelated
# project directory outside this repo.
OFFICIAL_ATOMIC_SEEN_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
OFFICIAL_COMPOSITE_SEEN_TASKS = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Byte-for-byte reproduction of eval_robocasa365/entry.py's real,
    working function of the same name (copied rather than imported -- see
    module docstring for why)."""
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_state_to_14d(state16: np.ndarray) -> np.ndarray:
    """Real modality.json slice layout (verified directly against
    .../CloseFridge/20250819/lerobot/meta/modality.json): base_position
    [0:3], base_rotation quat-xyzw [3:7], end_effector_position_relative
    [7:10], end_effector_rotation_relative quat-xyzw [10:14], gripper_qpos
    [14:16]. Reassembled into entry.py's real 14-dim EE-first order:
    ee_pos_rel(3), ee_rot_rel axis-angle(3), gripper_qpos(2),
    base_position(3), base_rotation axis-angle(3)."""
    state16 = np.asarray(state16, dtype=np.float64).reshape(-1)
    if state16.shape != (16,):
        raise ValueError(f"expected a 16-dim observation.state, got {state16.shape}")
    base_position = state16[0:3]
    base_rotation = state16[3:7]
    ee_pos_rel = state16[7:10]
    ee_rot_rel = state16[10:14]
    gripper_qpos = state16[14:16]
    return np.concatenate(
        [
            ee_pos_rel.astype(np.float32),
            quat_xyzw_to_axis_angle(ee_rot_rel),
            gripper_qpos.astype(np.float32),
            base_position.astype(np.float32),
            quat_xyzw_to_axis_angle(base_rotation),
        ]
    ).astype(np.float32)


def _augment(images):
    """Functional copy of JsonDataset._augment's real jitter recipe (that
    method never references `self`, so this is a plain copy rather than an
    awkward unbound-method call) -- kept identical so RoboCasa training
    images get the same real augmentation distribution xr1_post_train_demo's
    images already went through."""
    ops = (
        (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
        (adjust_contrast, random.uniform(0.5, 1.5)),
        (adjust_saturation, random.uniform(0.5, 1.5)),
        (adjust_hue, random.uniform(0.0, 0.0)),
    )
    flags = [random.randint(0, 1) == 0 for _ in ops]
    out = []
    for image in images:
        image = resize_image(image, factor=32, max_pixels=160000)
        for use, (op, value) in zip(flags, ops):
            if use:
                image = op(image, value)
        out.append(image)
    return out


def discover_task_roots(data_root, atomic_tasks, composite_tasks):
    """Real glob against the verified directory layout
    (<data_root>/{atomic,composite}/<TaskName>/<date>/lerobot). Raises if
    any requested task has zero matches -- silently training on a subset of
    the intended task set would be a real, hard-to-notice correctness bug,
    not a graceful degradation."""
    roots = []
    missing = []
    for kind, names in (("atomic", atomic_tasks), ("composite", composite_tasks)):
        for name in names:
            matches = sorted(glob.glob(os.path.join(data_root, kind, name, "*", "lerobot")))
            if not matches:
                missing.append(f"{kind}/{name}")
                continue
            roots.append(matches[0])
    if missing:
        raise FileNotFoundError(f"no lerobot data found under {data_root!r} for tasks: {missing}")
    return roots


class RoboCasaLerobotDataset(Dataset):
    """Flat sampler over real RoboCasa365 lerobot-format training data (see
    module docstring for the full schema-compatibility rationale). Indexes
    episodes (not individual frames -- see the memory note in the module
    docstring), then builds a fixed-length, seed-deterministic list of
    (episode, frame) samples sized to the real training budget."""

    def __init__(self, task_roots, action_length, max_samples, seed):
        self.action_length = int(action_length)
        self.episodes = self._index_episodes(task_roots)
        if not self.episodes:
            raise ValueError(f"no episodes found under task_roots: {task_roots}")

        rng = random.Random(seed)
        order = list(range(len(self.episodes)))
        rng.shuffle(order)
        max_samples = max(int(max_samples), 1)
        quotient, remainder = divmod(max_samples, len(order))
        tiled = order * quotient + order[:remainder]

        # Built once, here, not per __getitem__ call -- deterministic given
        # `seed`, and bounded to max_samples regardless of how many real
        # episodes/frames exist (see memory note in module docstring).
        self.samples = [
            (episode_index, rng.randint(0, max(0, self.episodes[episode_index]["length"] - 1)))
            for episode_index in tiled
        ]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _index_episodes(task_roots):
        episodes = []
        for root in task_roots:
            episodes_path = os.path.join(root, "meta", "episodes.jsonl")
            with open(episodes_path) as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    episodes.append(
                        {
                            "root": root,
                            "episode_index": record["episode_index"],
                            "length": record["length"],
                            "task": record["tasks"][0],
                        }
                    )
        return episodes

    @staticmethod
    @lru_cache(maxsize=64)
    def _read_episode_arrays(root, episode_index):
        path = os.path.join(root, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        action = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        return state, action

    @staticmethod
    def _video_path(root, episode_index, camera_key):
        return os.path.join(root, "videos", "chunk-000", camera_key, f"episode_{episode_index:06d}.mp4")

    def _read_frame(self, root, episode_index, camera_key, frame_index):
        path = self._video_path(root, episode_index, camera_key)
        video = VideoReader(path, num_threads=2)
        return video.get_batch([frame_index]).asnumpy()[0]

    def _pad(self, value, steps):
        value = np.asarray(value)
        if steps == self.action_length:
            return value
        return np.concatenate([value, np.repeat(value[-1:], self.action_length - steps, axis=0)], axis=0)

    def __getitem__(self, index):
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        root, real_episode_index, length, task = (
            episode["root"],
            episode["episode_index"],
            episode["length"],
            episode["task"],
        )
        steps = min(self.action_length, length - frame_index)
        if steps <= 0:
            raise IndexError(f"invalid sample: frame_index={frame_index} episode_length={length}")

        state_arr, action_arr = self._read_episode_arrays(root, real_episode_index)

        state60 = np.zeros((1, ROBOCASA_STATE_DIM), dtype=np.float32)
        state60[0, :ROBOCASA_ACTIVE_STATE_DIMS] = observation_state_to_14d(state_arr[frame_index])

        action_window = self._pad(action_arr[frame_index : frame_index + steps], steps)
        action60 = np.zeros((self.action_length, ROBOCASA_ACTION_DIM), dtype=np.float32)
        action60[:, :ROBOCASA_ACTIVE_ACTION_DIMS] = action_window

        action_mask60 = np.zeros((self.action_length, ROBOCASA_ACTION_DIM), dtype=np.int32)
        action_mask60[:steps, :ROBOCASA_ACTIVE_ACTION_DIMS] = 1

        images = [
            Image.fromarray(self._read_frame(root, real_episode_index, camera_key, frame_index))
            for camera_key in CAMERA_KEYS
        ]
        images = _augment(images)

        conversations = [
            {
                "from": "human",
                "value": (
                    "Left camera: <image>\nRight camera: <image>\nWrist camera: <image>\n\n"
                    f"Generate robot actions for the task:\n{task} /no_cot"
                ),
            },
            {"from": "gpt", "value": "<cot></cot>"},
            {"from": "human", "value": "Robot state: <state>"},
            {"from": "gpt", "value": "".join(f"<a_{i}>" for i in range(steps)) + "<score>"},
        ]

        return {
            "messages": JsonDataset._messages(conversations, images),
            "action": torch.from_numpy(action60),
            "action_mask": torch.from_numpy(action_mask60),
            "state": torch.from_numpy(state60),
            "vlm_action_target": torch.from_numpy(action60[:steps]),
            "vlm_action_mask": torch.from_numpy(action_mask60[:steps]),
            "vlm_action_actual_length": steps,
        }
