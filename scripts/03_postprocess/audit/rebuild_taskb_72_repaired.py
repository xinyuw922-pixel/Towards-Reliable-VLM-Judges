#!/usr/bin/env python3
"""
Rebuild 24 blocked lavagap + memory exam entries with corrected steps.

Strategy B: lavagap (12 UIDs) + memory (12 UIDs) replacement.
Writes repaired exam to datasets/exams_taskb_v9_72_repaired_candidate/.
"""

import json, shutil
from pathlib import Path
from PIL import Image
import numpy as np

RAW_ROOT = Path("datasets/raw_data_taskb72")
EXAM_IN = Path("datasets/exams_taskb_v9_72_candidate/task_b_exam.jsonl")
EXAM_IN_DIR = EXAM_IN.parent
EXAM_OUT_DIR = Path("datasets/exams_taskb_v9_72_repaired_candidate")
OUT_JSONL = EXAM_OUT_DIR / "task_b_exam.jsonl"
OUT_IMG_DIR = EXAM_OUT_DIR / "images" / "taskB"

# ============================================================
# TRANSFORM FUNCTIONS (copied from generate_task_b_v9_72.py)
# ============================================================

def world_to_image(world_x, world_y, agent_x, agent_y, agent_dir):
    dx = world_x - agent_x
    dy = world_y - agent_y
    if agent_dir == 0:
        x_local =  dy;  y_local =  dx
    elif agent_dir == 1:
        x_local = -dx;  y_local =  dy
    elif agent_dir == 2:
        x_local = -dy;  y_local = -dx
    elif agent_dir == 3:
        x_local =  dx;  y_local = -dy
    else:
        raise ValueError(f"Invalid dir: {agent_dir}")
    return x_local + 4, y_local + 1

def agent_to_image(agent_x, agent_y, agent_dir):
    return world_to_image(agent_x, agent_y, agent_x, agent_y, agent_dir)

def compute_agent_dir_from_image(agent_pos_img, front_cell_pos_img):
    dx = front_cell_pos_img[0] - agent_pos_img[0]
    dy = front_cell_pos_img[1] - agent_pos_img[1]
    if   (dx, dy) == (1, 0): return 0
    elif (dx, dy) == (0,-1): return 1
    elif (dx, dy) == (-1,0): return 2
    elif (dx, dy) == (0, 1): return 3
    else:
        raise ValueError(f"agent.dir derivation failed: agent={agent_pos_img}, front={front_cell_pos_img}")

def canonical_state_for_type(obj_type, state):
    if obj_type == "door":
        return 0 if state is None else state
    return None

def image_to_observe(x_img, y_img):
    x_local = x_img - 4
    y_local = y_img - 1
    return x_local + 3, 6 - y_local

def is_observe_cell_visible(obs_x, obs_y, state_encoding):
    if not (0 <= obs_x < 7 and 0 <= obs_y < 7):
        return False
    try:
        cell = state_encoding[obs_x][obs_y]
    except (IndexError, TypeError):
        return False
    return isinstance(cell, (list, tuple)) and not (cell[0] == 0 and cell[1] == 0)

def is_visible_object(x_img, y_img, state_encoding):
    if not (1 <= x_img <= 7 and 1 <= y_img <= 7):
        return False
    obs_x, obs_y = image_to_observe(x_img, y_img)
    return is_observe_cell_visible(obs_x, obs_y, state_encoding)

def _source_frame_abs(env, triplet, t, raw_root):
    frames = triplet.get("frames", [])
    frame_rel = frames[t] if t < len(frames) else None
    if not frame_rel:
        frame_rel = f"{env}_{triplet['group_id']}/full/step_{t:03d}.png"
    return raw_root / frame_rel

# ============================================================
# REPLACEMENT MAPS
# ============================================================
LAVAGAP_REPLACEMENTS = {
    'lavagap_s000000': 9,
    'lavagap_s000003': 0,
    'lavagap_s000004': 1,
    'lavagap_s000007': 2,
    'lavagap_s000008': 3,
    'lavagap_s000010': 5,
    'lavagap_s000001': 9,
    'lavagap_s000005': 0,
    'lavagap_s000009': 1,
    'lavagap_s000002': 9,
    'lavagap_s000006': 0,
    'lavagap_s000011': 1,
}

MEMORY_REPLACEMENTS = {
    'B.memory.memory_s000000.t2': 4,
    'B.memory.memory_s000001.t6': 0,
    'B.memory.memory_s000002.t2': 4,
    'B.memory.memory_s000003.t2': 0,
    'B.memory.memory_s000004.t3': 0,
    'B.memory.memory_s000005.t4': 0,
    'B.memory.memory_s000006.t6': 1,
    'B.memory.memory_s000007.t1': 0,
    'B.memory.memory_s000008.t3': 0,
    'B.memory.memory_s000009.t6': 2,
    'B.memory.memory_s000010.t3': 2,
    'B.memory.memory_s000011.t9': 7,
}

def load_triplets(env):
    with open(RAW_ROOT / env / "triplets.jsonl") as f:
        return {t["group_id"]: t for t in [json.loads(l) for l in f if l.strip()]}

# ============================================================
# REBUILD A SINGLE ENTRY
# ============================================================
def rebuild_entry(env, group_id, step, source_exam_rec, triplets_cache):
    triplet = triplets_cache.get(group_id)
    if not triplet:
        raise RuntimeError(f"Triplet not found: {env}/{group_id}")
    state_seq = triplet.get("state_seq", [])
    if step >= len(state_seq):
        raise RuntimeError(f"Step {step} out of range for {env}/{group_id} (max={len(state_seq)-1})")

    world_state = state_seq[step]
    state_encoding = world_state.get("state_encoding", [])
    triplet_agent = world_state.get("agent", {})
    world_agent_pos = triplet_agent.get("pos")
    world_dir = triplet_agent.get("dir")
    world_carrying = triplet_agent.get("carrying")

    triplet_front = world_state.get("front_cell", {})
    world_front_type = triplet_front.get("type", "empty")
    if world_front_type == "floor":
        world_front_type = "empty"
    front_cell_state = canonical_state_for_type(world_front_type, triplet_front.get("state", 0))
    front_cell_world_pos = triplet_front.get("pos")

    world_objects_raw = world_state.get("objects", [])
    _source_frame_abs(env, triplet, step, RAW_ROOT)  # side-effect: validates path

    agent_x, agent_y = world_agent_pos
    agent_pos_img = agent_to_image(agent_x, agent_y, world_dir)
    fc_x, fc_y = front_cell_world_pos
    front_cell_pos_img = world_to_image(fc_x, fc_y, agent_x, agent_y, world_dir)

    objects_img = []
    for obj in world_objects_raw:
        wx, wy = obj["pos"]
        x_img, y_img = world_to_image(wx, wy, agent_x, agent_y, world_dir)
        if is_visible_object(x_img, y_img, state_encoding):
            if [x_img, y_img] != list(front_cell_pos_img):
                objects_img.append({
                    "type": obj.get("type", "unknown"),
                    "pos": [x_img, y_img],
                    "color": obj.get("color"),
                    "state": canonical_state_for_type(obj.get("type", ""), obj.get("state", 0)),
                })

    agent_dir_img = compute_agent_dir_from_image(agent_pos_img, front_cell_pos_img)

    answer = {
        "agent": {
            "pos": list(agent_pos_img),
            "dir": agent_dir_img,
            "carrying": world_carrying,
        },
        "front_cell": {
            "pos": list(front_cell_pos_img),
            "type": world_front_type,
            "state": front_cell_state,
        },
        "objects": objects_img,
    }

    new_uid = f"B.{env}.{group_id}.t{step}"
    new_image_basename = f"{group_id}_t{step}.png"

    new_rec = dict(source_exam_rec)
    new_rec["uid"] = new_uid
    new_rec["exam_id"] = new_uid
    new_rec["image"] = f"images/taskB/{new_image_basename}"
    new_rec["answer"] = json.dumps(answer, separators=(', ', ': '))
    new_rec["answer_json"] = answer
    new_rec["_replaced_from"] = source_exam_rec["uid"]
    new_rec["_old_step"] = int(source_exam_rec["uid"].split(".")[-1].replace("t", ""))
    new_rec["_new_step"] = step
    new_rec["_replacement_reason"] = "image_hash_duplicate_repair"
    return new_rec

# ============================================================
# MAIN
# ============================================================
print("Loading existing exam...")
with open(EXAM_IN) as f:
    existing = {json.loads(l)["uid"]: json.loads(l) for l in f if l.strip()}

EXAM_OUT_DIR.mkdir(parents=True, exist_ok=True)
(EXAM_OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Build replacement index
all_replacements = {}
for uid, step in MEMORY_REPLACEMENTS.items():
    gid = uid.split(".")[2]
    all_replacements[uid] = {"env": "memory", "gid": gid, "new_step": step}
for gid, step in LAVAGAP_REPLACEMENTS.items():
    uid = f"B.lavagap.{gid}.t8"
    all_replacements[uid] = {"env": "lavagap", "gid": gid, "new_step": step}

# Cache triplets
lavagap_triplets = load_triplets("lavagap")
memory_triplets = load_triplets("memory")

output_rows = []
replaced = []
kept = []

for uid, rec in existing.items():
    if uid in all_replacements:
        rep = all_replacements[uid]
        triplets = lavagap_triplets if rep["env"] == "lavagap" else memory_triplets
        triplets_cache = triplets
        new_rec = rebuild_entry(rep["env"], rep["gid"], rep["new_step"], rec, triplets_cache)
        output_rows.append(new_rec)
        replaced.append((uid, new_rec["uid"]))
    else:
        output_rows.append(rec)
        kept.append(uid)

print(f"\nTotal: {len(kept)} kept + {len(replaced)} replaced = {len(output_rows)} rows")
for old_uid, new_uid in replaced:
    print(f"  {old_uid}  ->  {new_uid}")

# Write repaired JSONL
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for row in output_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"\nWrote: {OUT_JSONL}")

# Copy images and verify
print("\nCopying images to output directory...")
missing = []
for row in output_rows:
    env = row["uid"].split(".")[1]
    if env in ("lavagap", "memory"):
        gid = row["uid"].split(".")[2]
        step = row["_new_step"]
        img_src = RAW_ROOT / env / gid / "full" / f"step_{step:03d}.png"
        img_dst = OUT_IMG_DIR / f"{gid}_t{step}.png"
    else:
        img_src = EXAM_IN_DIR / row["image"]
        img_dst = OUT_IMG_DIR / Path(row["image"]).name

    if not img_src.exists():
        missing.append(str(img_src))
        print(f"  MISSING: {img_src}")
    else:
        shutil.copy2(img_src, img_dst)

if not missing:
    print(f"All {len(output_rows)} image copies OK.")
else:
    print(f"WARNING: {len(missing)} missing images!")

# Final uniqueness audit
print("\n=== FINAL IMAGE UNIQUENESS AUDIT ===")
from collections import defaultdict, Counter
all_hashes = defaultdict(dict)
for row in output_rows:
    uid = row["uid"]
    env = uid.split(".")[1]
    if env in ("lavagap", "memory"):
        gid = uid.split(".")[2]
        step = row["_new_step"]
        img_path = RAW_ROOT / env / gid / "full" / f"step_{step:03d}.png"
    else:
        img_path = EXAM_IN_DIR / row["image"]
    all_hashes[env][uid] = hash(np.array(Image.open(img_path)).tobytes())

for env in sorted(all_hashes.keys()):
    hashes = list(all_hashes[env].values())
    unique = len(set(hashes))
    total = len(hashes)
    status = "OK" if unique == total else f"COLLISION ({unique}/{total})"
    print(f"  {env}: {unique}/{total} unique — {status}")
    if unique < total:
        dup = Counter(hashes)
        for h, cnt in dup.items():
            if cnt > 1:
                dups = [u for u, hh in all_hashes[env].items() if hh == h]
                print(f"    hash={h}: {dups}")

print("\n=== DONE ===")
