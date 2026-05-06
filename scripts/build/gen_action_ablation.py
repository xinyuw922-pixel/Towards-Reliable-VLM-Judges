#!/usr/bin/env python3
"""
Action Ablation Generator

Reference: Stats Spec §3.2
Generate action ablation versions of Task A exam
- shuffle: Randomly replace action
- mask: Replace action with [UNKNOWN]
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass


ACTION_NAMES = {
    0: "left",
    1: "right", 
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done"
}


@dataclass
class AblationConfig:
    seed: int = 42
    mask_token: str = "[UNKNOWN]"


def load_exam(exam_path: str) -> List[Dict]:
    """Load Task A exam"""
    items = []
    with open(exam_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def get_action_pool(exam_items: List[Dict], current_action: str) -> List[str]:
    """Get other actions from the same environment as candidate pool"""
    pool = []
    for item in exam_items:
        if item.get('action') and item['action'] != current_action:
            pool.append(item['action'])
    return list(set(pool)) if pool else []


def generate_shuffle_version(exam_items: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Generate shuffle version.
    Randomly replace action with another action from the same environment.
    """
    random.seed(seed)
    
    # Group by task
    by_task = defaultdict(list)
    for item in exam_items:
        if item.get('task') == 'A':
            task_name = item.get('uid', '').split('.')[1] if '.' in item.get('uid', '') else 'unknown'
            by_task[task_name].append(item)
    
    result = []
    
    for task_name, task_items in by_task.items():
        # Build action pool
        all_actions = set()
        for item in task_items:
            if 'action' in item and item['action']:
                all_actions.add(item['action'])
        
        action_list = list(all_actions)
        
        for item in task_items:
            new_item = item.copy()
            
            # Randomly select a different action
            original_action = item.get('action', '')
            if original_action and action_list:
                candidates = [a for a in action_list if a != original_action]
                if candidates:
                    new_action = random.choice(candidates)
                    # Replace action in prompt
                    prompt = new_item.get('prompt', '')
                    new_prompt = prompt.replace(original_action, new_action)
                    new_item['prompt'] = new_prompt
                    new_item['action'] = new_action
                    new_item['ablation_type'] = 'shuffle'
                    new_item['original_action'] = original_action
            
            result.append(new_item)
    
    return result


def generate_mask_version(exam_items: List[Dict], mask_token: str = "[UNKNOWN]") -> List[Dict]:
    """
    Generate mask version.
    Replace action with [UNKNOWN].
    """
    result = []
    
    for item in exam_items:
        if item.get('task') == 'A':
            new_item = item.copy()
            
            # Replace action with mask token
            original_action = item.get('action', '')
            if original_action:
                prompt = new_item.get('prompt', '')
                new_prompt = prompt.replace(original_action, mask_token)
                new_item['prompt'] = new_prompt
                new_item['action'] = mask_token
                new_item['ablation_type'] = 'mask'
                new_item['original_action'] = original_action
            
            result.append(new_item)
    
    return result


def save_exam(items: List[Dict], output_path: str):
    """Save exam to JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Action Ablation Generator")
    parser.add_argument("--exam", "-e", required=True, help="Input Task A exam JSONL")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--mask-token", default="[UNKNOWN]", help="Mask token")
    args = parser.parse_args()
    
    # Load original exam
    exam_items = load_exam(args.exam)
    print(f"Loaded {len(exam_items)} Task A items")
    
    # Generate shuffle version
    shuffle_items = generate_shuffle_version(exam_items, args.seed)
    shuffle_path = Path(args.output_dir) / "task_a_exam_shuffle.jsonl"
    save_exam(shuffle_items, str(shuffle_path))
    print(f"Generated {len(shuffle_items)} shuffle items: {shuffle_path}")
    
    # Generate mask version
    mask_items = generate_mask_version(exam_items, args.mask_token)
    mask_path = Path(args.output_dir) / "task_a_exam_mask.jsonl"
    save_exam(mask_items, str(mask_path))
    print(f"Generated {len(mask_items)} mask items: {mask_path}")


if __name__ == "__main__":
    main()
