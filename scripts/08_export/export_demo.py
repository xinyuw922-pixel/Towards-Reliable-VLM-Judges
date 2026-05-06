#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    from common import (  # type: ignore
        CANDIDATE_ENVS,
        action_name,
        concat_vertical_rgb,
        dump_json,
        ensure_output_dir,
        export_rgb,
        make_env,
        maybe_render_top_view,
        runtime_context,
        try_import_stack,
    )
else:
    from .common import (
        CANDIDATE_ENVS,
        action_name,
        concat_vertical_rgb,
        dump_json,
        ensure_output_dir,
        export_rgb,
        make_env,
        maybe_render_top_view,
        runtime_context,
        try_import_stack,
    )


DEFAULT_ACTIONS = [1, 2, 2, 0, 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="tmp_miniworld/demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--topview_width", type=int, default=320)
    ap.add_argument("--topview_height", type=int, default=240)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    try:
        gym, _ = try_import_stack()
    except RuntimeError as exc:
        print(f"[MiniWorld demo] {exc}", file=sys.stderr)
        raise SystemExit(1)

    manifest = {
        "runtime": runtime_context(),
        "seed": args.seed,
        "default_actions": DEFAULT_ACTIONS,
        "render_contract": {
            "obs_width": args.obs_width,
            "obs_height": args.obs_height,
            "topview_width": args.topview_width,
            "topview_height": args.topview_height,
            "concat_kind": "ego_topview_concat",
            "concat_layout": "vertical_top_bottom",
        },
        "envs": [],
    }

    for env_id in CANDIDATE_ENVS:
        env_slug = env_id.replace("MiniWorld-", "").replace("-v0", "").lower()
        env_dir = ensure_output_dir(out_dir / env_slug)
        item = {"env_id": env_id, "output_dir": str(env_dir)}
        try:
            env, obs, info = make_env(
                gym,
                env_id,
                seed=args.seed,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                window_width=args.topview_width,
                window_height=args.topview_height,
            )
            export_rgb(env_dir / "reset_ego.png", obs)
            top = maybe_render_top_view(env)
            if top is not None:
                export_rgb(env_dir / "reset_topview.png", top)
                concat = concat_vertical_rgb(obs, top)
                export_rgb(env_dir / "reset_ego_topview_concat.png", concat)

            item["frames"] = [
                {"kind": "reset_ego", "path": str(env_dir / "reset_ego.png")},
            ]
            if top is not None:
                item["frames"].append({"kind": "reset_topview", "path": str(env_dir / "reset_topview.png")})
                item["frames"].append(
                    {
                        "kind": "reset_ego_topview_concat",
                        "path": str(env_dir / "reset_ego_topview_concat.png"),
                    }
                )

            for idx, action in enumerate(DEFAULT_ACTIONS, start=1):
                obs, reward, terminated, truncated, step_info = env.step(action)
                ego_path = env_dir / f"step_{idx:02d}_ego.png"
                export_rgb(ego_path, obs)
                item["frames"].append(
                    {
                        "kind": "ego",
                        "step": idx,
                        "action": action,
                        "action_name": action_name(env, action),
                        "reward": float(reward),
                        "path": str(ego_path),
                    }
                )

                top = maybe_render_top_view(env)
                if top is not None:
                    top_path = env_dir / f"step_{idx:02d}_topview.png"
                    export_rgb(top_path, top)
                    item["frames"].append(
                        {
                            "kind": "topview",
                            "step": idx,
                            "path": str(top_path),
                        }
                    )
                if terminated or truncated:
                    item["terminated_at"] = idx
                    break

            env.close()
            item["ok"] = True
            item["reset_info_keys"] = sorted(info.keys())
        except Exception as exc:
            item["ok"] = False
            item["error"] = repr(exc)
        manifest["envs"].append(item)

    dump_json(out_dir / "demo_manifest.json", manifest)
    print(f"Saved demo manifest to {out_dir / 'demo_manifest.json'}")


if __name__ == "__main__":
    main()
