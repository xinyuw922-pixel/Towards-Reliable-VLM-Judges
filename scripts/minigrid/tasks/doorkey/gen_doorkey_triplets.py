#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoorKey Triplet Generator for GridWM-Judge.

Generates Full/NoCue/CF triplets for DoorKey task with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python gen_doorkey_triplets.py --out-dir data_doorkey_vFinal --num 100 --env-id MiniGrid-DoorKey-8x8-v0 --resume
"""

import argparse
import copy
import json
import shutil
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import gymnasium as gym

# MiniGrid imports
from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.core.world_object import Goal
try:
    from scripts.minigrid.tasks.nocue_common import build_nocue_meta, compute_nocue_metrics, select_nocue_indices
except ModuleNotFoundError:
    import sys

    _tasks_root = Path(__file__).resolve().parents[1]
    if str(_tasks_root) not in sys.path:
        sys.path.append(str(_tasks_root))
    from nocue_common import build_nocue_meta, compute_nocue_metrics, select_nocue_indices


def _repo_root() -> Path:
    # gen_doorkey_triplets.py is at scripts/minigrid/tasks/doorkey/
    # repo root is 4 levels up
    return Path(__file__).resolve().parents[4]


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (_repo_root() / p).resolve()

# -------------------------
# Constants & Helpers
# -------------------------

ACTION_ID2NAME = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

def append_jsonl_batch(path: Path, objs: List[Dict]) -> None:
    """Atomic-ish append with fsync to survive hard crashes"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def enforce_triplet_hard_gates(records: List[Dict]) -> None:
    by_variant = {str(r.get("variant", "")): r for r in records}
    required = ("full", "nocue", "cf")
    if set(by_variant.keys()) != set(required):
        raise ValueError(f"triplet_variants_invalid:{sorted(by_variant.keys())}")

    full = by_variant["full"]
    nocue = by_variant["nocue"]
    cf = by_variant["cf"]

    for name in required:
        rec = by_variant[name]
        actions = list(rec.get("actions_id", []))
        frames = list(rec.get("frames", []))
        state_seq = list(rec.get("state_seq", []))
        if len(frames) != len(actions) + 1:
            raise ValueError(f"{name}_frame_action_misaligned:{len(frames)}!={len(actions)}+1")
        if len(state_seq) != len(frames):
            raise ValueError(f"{name}_state_frame_misaligned:{len(state_seq)}!={len(frames)}")

    if list(nocue.get("actions_id", [])) != list(full.get("actions_id", [])):
        raise ValueError("nocue_actions_mismatch")
    if list(cf.get("actions_id", [])) != list(full.get("actions_id", [])):
        raise ValueError("cf_actions_mismatch")

    for key in ("success", "terminated", "truncated"):
        if nocue.get(key) != full.get(key):
            raise ValueError(f"nocue_{key}_mismatch")
    if float(nocue.get("reward", 0.0)) != float(full.get("reward", 0.0)):
        raise ValueError("nocue_reward_mismatch")

def render_agent_pov(env, tile_size: int, grid_h: int, grid_w: int) -> Optional[np.ndarray]:
    try:
        frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)
        # Dynamic shape check
        if isinstance(frame, np.ndarray) and frame.shape[0] == grid_h * tile_size and frame.shape[1] == grid_w * tile_size:
            return frame
    except Exception:
        pass
    return None

def _door_state(door_obj) -> int:
    if getattr(door_obj, "is_locked", False): return 2
    if getattr(door_obj, "is_open", False): return 0
    return 1

def extract_state(env_unwrapped, obs: Dict) -> Dict:
    img_enc = obs.get("image", None)
    visible_types_gt: Set[str] = set()
    if img_enc is not None:
        H, W = img_enc.shape[0], img_enc.shape[1]
        for i in range(H):
            for j in range(W):
                obj_idx = int(img_enc[i, j, 0])
                name = IDX_TO_OBJECT.get(obj_idx, None)
                if name and name not in ("unseen", "empty", "wall", "floor"):
                    visible_types_gt.add(name)

    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None: continue
            t = getattr(cell, "type", "unknown")
            if t in ("wall", "floor"): continue
            o = {"type": t, "pos": [int(x), int(y)]}
            if hasattr(cell, "color"): o["color"] = str(getattr(cell, "color"))
            if t == "door": o["state"] = _door_state(cell)
            objects.append(o)

    fx, fy = env_unwrapped.front_pos
    f_cell = env_unwrapped.grid.get(fx, fy)
    f_type = getattr(f_cell, "type", "empty") if f_cell else "empty"
    if f_type == "floor":
        f_type = "empty"
    front_info = {"pos": [int(fx), int(fy)], "type": str(f_type)}
    if f_cell and f_type == "door":
        front_info["state"] = _door_state(f_cell)

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": {"type": str(env_unwrapped.carrying.type), "color": str(env_unwrapped.carrying.color)} if env_unwrapped.carrying else None
        },
        "front_cell": front_info,
        "visible_types_gt": sorted(list(visible_types_gt)),
        "visible_types": sorted(list(visible_types_gt)),
        "state_encoding": img_enc.astype(int).tolist() if img_enc is not None else None,
        "objects": objects,
        "mask_applied": False,
        "mask_type": None
    }

# -------------------------
# Planning & Rollout
# -------------------------

@dataclass(frozen=True)
class DKState:
    ax: int; ay: int; ad: int; has_key: bool; door_state: int; key_present: bool

def _find_key_door_goal(env_unwrapped):
    key_pos = door_pos = goal_pos = None
    key_obj = door_obj = None
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            c = env_unwrapped.grid.get(x, y)
            if c:
                if c.type == "key": key_pos = (x, y); key_obj = c
                elif c.type == "door": door_pos = (x, y); door_obj = c
                elif c.type == "goal": goal_pos = (x, y)
    return key_pos, key_obj, door_pos, door_obj, goal_pos

def plan_doorkey_bfs(env_unwrapped, max_nodes=200000) -> Optional[List[int]]:
    key_pos, _, door_pos, door_obj, goal_pos = _find_key_door_goal(env_unwrapped)
    if not (key_pos and door_pos and goal_pos and door_obj): return None

    start_node = DKState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir), False, _door_state(door_obj), True)
    from collections import deque
    q = deque([start_node]); visited = {start_node}; parent = {start_node: (None, None)}

    steps = 0
    while q:
        curr = q.popleft()
        steps += 1
        if steps > max_nodes: return None
        if (curr.ax, curr.ay) == goal_pos:
            path = []
            while parent[curr][0]:
                prev, act = parent[curr]; path.append(act); curr = prev
            return list(reversed(path))

        for act in (0, 1):
            nd = (curr.ad - 1) % 4 if act == 0 else (curr.ad + 1) % 4
            nxt = DKState(curr.ax, curr.ay, nd, curr.has_key, curr.door_state, curr.key_present)
            if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, act); q.append(nxt)
        dx, dy = DIR2VEC[curr.ad]; nx, ny = curr.ax + dx, curr.ay + dy
        if 0 <= nx < env_unwrapped.width and 0 <= ny < env_unwrapped.height:
            passable = True
            if (nx, ny) == door_pos and curr.door_state != 0: passable = False
            if (nx, ny) == key_pos and curr.key_present: passable = False
            cell = env_unwrapped.grid.get(nx, ny);
            if cell and cell.type == "wall": passable = False
            if passable:
                nxt = DKState(nx, ny, curr.ad, curr.has_key, curr.door_state, curr.key_present)
                if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 2); q.append(nxt)
        if (nx, ny) == key_pos and curr.key_present and not curr.has_key:
            nxt = DKState(curr.ax, curr.ay, curr.ad, True, curr.door_state, False)
            if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 3); q.append(nxt)
        if (nx, ny) == door_pos:
            if (curr.door_state == 2 and curr.has_key) or curr.door_state == 1:
                nxt = DKState(curr.ax, curr.ay, curr.ad, curr.has_key, 0, curr.key_present)
                if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 5); q.append(nxt)
    return None

def rollout_from_current(env, obs0: Dict, actions: List[int], tile_size: int, out_dir: Path, rel_dir: str, grid_h: int, grid_w: int) -> Optional[Dict]:
    frames, states = [], []
    reward_sum = 0.0
    terminated = truncated = False

    frame0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    if frame0 is None: return None
    save_png(frame0, out_dir / rel_dir / "step_000.png")
    frames.append(str(Path(rel_dir) / "step_000.png"))
    states.append(extract_state(env.unwrapped, obs0))

    obs = obs0
    for t, a in enumerate(actions):
        obs, r, term, trunc, _ = env.step(a)
        reward_sum += float(r)
        terminated |= bool(term); truncated |= bool(trunc)
        frame = render_agent_pov(env, tile_size, grid_h, grid_w)
        if frame is None: return None
        save_png(frame, out_dir / rel_dir / f"step_{t+1:03d}.png")
        frames.append(str(Path(rel_dir) / f"step_{t+1:03d}.png"))
        states.append(extract_state(env.unwrapped, obs))
        if term or trunc: break

    return {
        "frames": frames, "state_seq": states, "reward": reward_sum,
        "success": bool(terminated and reward_sum > 0),
        "terminated": terminated, "truncated": truncated
    }

def mask_key_tiles_inplace(pov_img: np.ndarray, state_encoding: List, threshold_obj_name="key") -> Tuple[int, int]:
    if state_encoding is None: return 0, 0
    enc = np.array(state_encoding, dtype=int)
    Ht, Wt = enc.shape[0], enc.shape[1]
    H, W = pov_img.shape[0], pov_img.shape[1]
    tile_h, tile_w = H // Ht, W // Wt
    masked_count, target_count = 0, 0
    for i in range(Ht):
        for j in range(Wt):
            if IDX_TO_OBJECT.get(int(enc[i, j, 0])) == threshold_obj_name:
                target_count += 1
                # state_encoding uses (vx, vy); image tiles use (row=y, col=x)
                y1, y2 = j * tile_h, (j + 1) * tile_h
                x1, x2 = i * tile_w, (i + 1) * tile_w
                pov_img[y1:y2, x1:x2] = 0
                masked_count += 1
    return masked_count, target_count

def relocate_goal_cf(env_unwrapped, forbidden: Set[Tuple[int, int]], rng: np.random.Generator) -> Optional[Dict]:
    _, _, _, _, goal_pos = _find_key_door_goal(env_unwrapped)
    if not goal_pos: return None
    cands = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            if (x,y) != goal_pos and (x,y) not in forbidden and env_unwrapped.grid.get(x, y) is None:
                cands.append((x,y))
    if not cands: return None
    new_pos = cands[rng.integers(0, len(cands))]
    env_unwrapped.grid.set(goal_pos[0], goal_pos[1], None)
    env_unwrapped.grid.set(new_pos[0], new_pos[1], Goal())
    return {"cf_mode": "move_goal", "goal_from": list(goal_pos), "goal_to": list(new_pos), "intervention_step": 0}

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=str(_repo_root() / "datasets" / "raw_data" / "doorkey"),
    )
    ap.add_argument("--num", "--n", dest="num", type=int, default=50)
    ap.add_argument("--env-id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--nocue-max-visible", type=int, default=5)
    ap.add_argument("--alignment-min", type=float, default=0.7)
    ap.add_argument("--mask-ratio-max", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true", help="Resume from existing directory (skip existing seeds)")
    args = ap.parse_args()

    out_dir = _resolve_path(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triplets.jsonl"

    prefix = "doorkey_s"

    # Resume Logic: Only delete jsonl if NOT resuming
    if not args.resume:
        if jsonl_path.exists(): jsonl_path.unlink()
        for child in out_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                shutil.rmtree(child, ignore_errors=True)
        tmp_root = out_dir / "_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

    existing_group_dirs = sorted(
        p.name for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    existing_seeds = []
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("variant") == "full":
                existing_seeds.append(int(rec["seed"]))

    made_count = len(existing_group_dirs) if args.resume else 0
    current_seed = (max(existing_seeds) + 1) if existing_seeds else args.seed_start

    while made_count < args.num:
        gid = f"{prefix}{made_count:06d}"
        final_root = out_dir / gid

        # Resume Check
        if final_root.exists():
            if args.resume:
                print(f"[RESUME] Skipping existing {gid}")
                made_count += 1
                continue
            else:
                shutil.rmtree(final_root)

        work_dir = out_dir / "_tmp" / f"{gid}"
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        env = None
        try:
            # 1. FULL
            env = gym.make(args.env_id, render_mode="rgb_array")
            obs0, _ = env.reset(seed=current_seed)
            gh, gw = obs0["image"].shape[0], obs0["image"].shape[1]

            actions = plan_doorkey_bfs(env.unwrapped)
            if not actions or len(actions) > args.max_steps: raise ValueError("Plan failed")

            full_traj = rollout_from_current(env, obs0, actions, args.tile_size, work_dir, "full", gh, gw)
            mission = env.unwrapped.mission
            env.close(); env = None # Release Resource

            if not full_traj or not full_traj["success"]: raise ValueError("Full rollout failed")

            pickup_step = next((t for t, a in enumerate(actions) if a == 3 and full_traj["state_seq"][t]["front_cell"]["type"] == "key"), None)
            toggle_step = next((t for t, a in enumerate(actions) if a == 5 and full_traj["state_seq"][t]["front_cell"]["type"] == "door"), None)
            if pickup_step is None or toggle_step is None: raise ValueError("Semantic events missing")

            # 2. CF
            forbidden = {tuple(st["agent"]["pos"]) for st in full_traj["state_seq"]} | \
                        {tuple(st["front_cell"]["pos"]) for st in full_traj["state_seq"]}

            cf_traj, cf_meta = None, None
            for k in range(20):
                # Clean CF dir before each attempt
                cf_path = work_dir / "cf"
                if cf_path.exists(): shutil.rmtree(cf_path)

                env_cf = gym.make(args.env_id, render_mode="rgb_array")
                try:
                    env_cf.reset(seed=current_seed)
                    rng = np.random.default_rng(current_seed + 1000 + k)
                    meta = relocate_goal_cf(env_cf.unwrapped, forbidden, rng)
                    if meta:
                        obs_cf = env_cf.unwrapped.gen_obs()
                        traj = rollout_from_current(env_cf, obs_cf, actions, args.tile_size, work_dir, "cf", gh, gw)

                        if traj is None: continue

                        # CF Gate: Reward 0, No Truncation, No Goal Visibility
                        goal_visible = any("goal" in st["visible_types_gt"] for st in traj["state_seq"])
                        if (not traj["success"] and
                            traj["reward"] == 0 and
                            not traj["truncated"] and
                            not goal_visible):
                            cf_traj, cf_meta = traj, meta
                            break
                finally:
                    env_cf.close() # Always close CF env

            if not cf_traj: raise ValueError("CF generation failed")

            # 3. NOCUE
            first_interaction_step = int(pickup_step)
            masked_indices, leak_indices, d5_excluded_indices = select_nocue_indices(
                state_seq=full_traj["state_seq"],
                first_interaction_step=first_interaction_step,
                max_visible=args.nocue_max_visible,
                is_target_visible=lambda st: ("key" in st.get("visible_types_gt", [])),
                is_d5_excluded=lambda st: st["front_cell"]["type"] == "key",
            )

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
            nocue_states = copy.deepcopy(full_traj["state_seq"])
            per_frame_mask_ratios, alignment_ratios, masked_tiles_total = [], [], 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                # FIX: Context Manager for FD Safety
                with Image.open(work_dir / fp_rel) as im:
                    img = np.array(im.convert("RGB"))

                st = nocue_states[i]
                if i in masked_indices:
                    cnt, tgt = mask_key_tiles_inplace(img, st["state_encoding"], "key")
                    masked_tiles_total += cnt
                    per_frame_mask_ratios.append(cnt / (gh*gw))
                    alignment_ratios.append(cnt/tgt if tgt > 0 else 0.0)
                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    st["visible_types"] = [v for v in st.get("visible_types_gt", []) if v != "key"]
                else:
                    st["mask_applied"] = False; st["mask_type"] = None; st["visible_types"] = st["visible_types_gt"]
                save_png(img, work_dir / "nocue" / f"step_{i:03d}.png")

            mask_strength_target = 1.0 / float(gh * gw)
            metrics = compute_nocue_metrics(
                per_frame_mask_ratios=per_frame_mask_ratios,
                alignment_ratios=alignment_ratios,
                masked_tiles_total=int(masked_tiles_total),
                gh=int(gh),
                gw=int(gw),
                masked_count=len(masked_indices),
                mask_ratio_max=float(args.mask_ratio_max),
                alignment_min=float(args.alignment_min),
            )
            nocue_meta = build_nocue_meta(
                targets=["key"],
                first_interaction_step=int(first_interaction_step),
                masked_indices=masked_indices,
                leak_indices=leak_indices,
                d5_excluded_indices=d5_excluded_indices,
                mask_strength_target=float(mask_strength_target),
                mask_strength_actual=float(np.mean(per_frame_mask_ratios)) if per_frame_mask_ratios else 0.0,
                mask_strength_threshold=float(args.mask_ratio_max),
                alignment_threshold=float(args.alignment_min),
                metrics=metrics,
                extra={"max_visible_frames": int(args.nocue_max_visible)},
            )

            # Finalize
            if final_root.exists(): shutil.rmtree(final_root)
            shutil.move(str(work_dir), str(final_root))

            # Record
            base_rec = {
                "group_id": gid, "task": "doorkey", "env_id": args.env_id, "seed": int(current_seed),
                "actions_id": actions, "actions_text": [ACTION_ID2NAME[a] for a in actions],
                "mission": mission, "model_input_fields": ["frames", "mission"]
            }

            nocue_traj_struct = {
                "frames": full_traj["frames"], "state_seq": nocue_states,
                "success": True, "reward": full_traj["reward"],
                "terminated": full_traj["terminated"], "truncated": full_traj["truncated"]
            }

            def mk_rec(var, traj, meta_c=None, meta_n=None):
                return {**base_rec, "variant": var,
                        "frames": [f"{gid}/{var}/{Path(p).name}" for p in traj["frames"]],
                        "state_seq": traj["state_seq"], "success": traj["success"], "reward": traj["reward"],
                        "terminated": traj["terminated"], "truncated": traj["truncated"], "cf_meta": meta_c, "nocue_meta": meta_n}

            records = [
                mk_rec("full", full_traj),
                mk_rec("nocue", nocue_traj_struct, None, nocue_meta),
                mk_rec("cf", cf_traj, cf_meta, None)
            ]
            enforce_triplet_hard_gates(records)
            append_jsonl_batch(jsonl_path, records)

            print(f"[{made_count+1}/{args.num}] Generated {gid} (seed={current_seed})")
            made_count += 1

        except ValueError:
            if work_dir.exists(): shutil.rmtree(work_dir)
            if env: env.close()
        except Exception as e:
            if work_dir.exists(): shutil.rmtree(work_dir)
            if env: env.close()
            print(f"Error seed {current_seed}: {e}")

        current_seed += 1

if __name__ == "__main__":
    main()
