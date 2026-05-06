#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_inference.py - GridWM-Judge Universal VLM Inference Gateway

This script provides a unified, capability-preserving interface for running inference
on GridWM-Judge exam questions across multiple VLM backends. Designed with scientific
standards to prevent engineering artifacts from distorting capability assessments.

Key Features:
- Backend Agnostic: OpenAI-compatible (SiliconFlow/OpenAI/Zhizengzeng), Gemini, Local models
- Capability Preservation: Task-aware token budgets, PNG fidelity, robust error handling
- Real-time Progress: Live progress updates with ETA and error tracking
- Experiment Isolation: Automatic directory structure for clean result management
- Zero-Trust Security: Advanced secret masking and API parameter validation
- Scientific Standards: Sentinel errors, metadata preservation, reproducible execution
- Provider Compatibility: Unified API key resolution (ZZZ_API_KEY + legacy support)

Supported Backends:
  - openai_compatible: OpenAI, SiliconFlow, Zhizengzeng, custom OpenAI-compatible APIs
      * Task-aware token budgets (A:512, B:2048, C:256)
      * PNG-first image encoding for fidelity preservation
      * CamelCase token key normalization (maxOutputTokens -> max_output_tokens)
      * Dedicated __GW_ERR__: sentinel to avoid model output collision
      * Zhizengzeng unified endpoint (all models via api.zhizengzeng.com/v1)
      * API key compatibility: ZZZ_API_KEY (official) + ZHIZENGZENG_API_KEY (legacy)
  - gemini: Google Gemini API + Zhizengzeng Gemini
      * Secure parameter merging and validation
  - local: Local model inference (Qwen2.5-VL, InternVL2.5, LLaVA-NeXT-Video)
      * Device mapping and dtype configuration
      * Torch-based inference with proper memory management

Progress Reporting:
- Real-time updates every N items and T seconds
- Includes processing rate, ETA, error count, and current UID
- Immediate stdout flushing (no buffering issues)

Experiment Management:
- Automatic experiment ID generation: {backend}_{provider}_{model}_{protocol}_shard{shard_id}
- Isolated directory structure prevents result conflicts
- Resume capability with processed UID tracking

Supported Providers (OpenAI-compatible):
- openai: Official OpenAI API
- siliconflow: SiliconFlow API (CN optimized)
- zhizengzeng*: Zhizengzeng unified platform (ChatGPT, Claude, Grok, etc.)
- zhizengzeng_qwen: Zhizengzeng Qwen VL models (qwen-vl-plus, qwen-vl-max, etc.)
- closeai: CloseAI proxy
- custom: Custom base URL

Author: GridWM-Judge Team
Version: 7.6.12 (Zhizengzeng-Qwen-VL-Compatible Edition)
Date: 2025-01-14
"""

# ── Setup path BEFORE other imports ──────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_scripts_dir = _Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent  # repo root is 2 levels up from scripts/<module>/
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

# ── Load .env BEFORE any other imports that may use environment variables ────────
from scripts.schema.env_bootstrap import load_project_dotenv as _load_project_dotenv
_load_project_dotenv(_Path(__file__).resolve().parents[1] / ".env", override=False)

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image
try:
    import torch  # Required for local model inference (Qwen2.5-VL, etc.)
    _torch_available = True
except ImportError:
    _torch_available = False
    torch = None

from scripts.schema.exam_schema import (
    EXAM_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    build_exam_meta,
    parse_exam_id as parse_exam_id_ssot,
    resolve_uid,
)

# D7 payload scan integration
D7_SCAN_ENABLED = True
try:
    from d7_payload_scan import scan_payload as d7_scan_payload
    D7_SCAN_AVAILABLE = True
except ImportError:
    D7_SCAN_AVAILABLE = False
    def d7_scan_payload(prompt):
        class DummyResult:
            passed = True
            violations = []
        return DummyResult()


# -------------------------
# Constants
# -------------------------

# Dedicated sentinel for gateway errors to avoid collision with model outputs
API_ERROR_PREFIX = "__GW_ERR__:"

_RE_TASK_FROM_UID = re.compile(r"^(?P<task>[ABC])\.")


# -------------------------
# Utilities
# -------------------------

def stable_hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _extract_request_options(q: Dict[str, Any]) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    content_order = q.get("api_content_order")
    image_detail = q.get("api_image_detail")
    if content_order:
        opts["content_order"] = str(content_order)
    if image_detail:
        opts["image_detail"] = str(image_detail)
    return opts

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def parse_dtype(s: str):
    if s in (None, "", "auto"): return "auto"
    if not _torch_available:
        raise ImportError("torch is not available; local model inference requires torch. For API-based inference, use --backend openai_compatible or --backend gemini.")
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def safe_load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()

def encode_image_base64(path: str, max_side: int, quality: int, prefer_png: bool = True) -> Tuple[str, str]:
    """
    Returns (mime_type, b64).
    - prefer_png=True: try PNG to avoid JPEG artifacts (esp. storyboard / grid images)
    - max_side<=0: no resize
    """
    p = Path(path)
    with Image.open(p) as im0:
        im = im0.convert("RGB")
        if max_side and max_side > 0:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        # Keep PNG if asked; otherwise JPEG for smaller payload.
        if prefer_png:
            im.save(buf, format="PNG", optimize=True)
            return ("image/png", base64.b64encode(buf.getvalue()).decode("utf-8"))
        else:
            im.save(buf, format="JPEG", quality=int(quality), optimize=True)
            return ("image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8"))

def _infer_task(uid: str) -> Optional[str]:
    if not uid:
        return None
    m = _RE_TASK_FROM_UID.match(uid)
    return m.group("task") if m else None

def _task_max_tokens(uid: str, default_max: int, task_overrides: Dict[str, int]) -> int:
    """
    Task-aware token budget to avoid truncation causing fake failures.
    - A: 512 tokens (atomic transitions)
    - B: 2048 tokens (JSON structures)
    - C: 256 tokens (success/failure judgment)
    """
    t = _infer_task(uid)
    if not t:
        return int(default_max)
    if t in task_overrides:
        return int(task_overrides[t])
    return int(default_max)

def fuse_images(paths: List[str], out_path: Path, mode: str, quality: int) -> str:
    imgs = [safe_load_rgb(p) for p in paths]
    ws, hs = zip(*[im.size for im in imgs])
    if mode == "h":
        W, H = sum(ws), max(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        x = 0
        for im in imgs: canvas.paste(im, (x, 0)); x += im.size[0]
    else:
        W, H = max(ws), sum(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for im in imgs: canvas.paste(im, (0, y)); y += im.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=quality)
    return str(out_path)

def _need(name: str, v: Optional[str]) -> str:
    if not v:
        raise ValueError(f"Missing {name}")
    return v

def _pick_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None

def _mask_any(x: Any) -> Any:
    """
    Recursively mask sensitive data in dicts/lists.
    - Hard-mask high-risk blobs: 'messages', 'contents'
    - Mask secrets with precise rules (handles snake_case, kebab-case, camelCase)
    """
    SECRET_EXACT = {
        "api_key", "apikey", "access_token", "refresh_token", "id_token",
        "authorization", "auth", "secret", "password", "x-api-key", "x-goog-api-key"
    }
    # Allow these for observability (case-insensitive checking below)
    ALLOW_OBSERVABILITY = {
        "max_tokens", "max_completion_tokens", "max_output_tokens",
        "maxtokens", "maxcompletiontokens", "maxoutputtokens"
    }

    def is_secret_key(key_str: str) -> bool:
        lk = key_str.lower()
        clean_lk = lk.replace("_", "").replace("-", "")

        if clean_lk in ALLOW_OBSERVABILITY:
            return False

        if lk in ("messages", "contents"):
            return True
        if lk in SECRET_EXACT:
            return True

        # Heuristics for variations: *_token, *_key, *-token, *-key
        if lk.endswith("token") or lk.endswith("key"):
            # Check if it looks like a known safe token param
            if clean_lk in ALLOW_OBSERVABILITY:
                return False
            return True # Default to secret if it ends in token/key and not whitelisted

        if "authorization" in lk:
            return True
        return False

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if is_secret_key(str(k)):
                # Distinguish content vs secret masking
                if str(k).lower() in ("messages", "contents"):
                    out[k] = "***MASKED_CONTENT***"
                else:
                    out[k] = "***MASKED_SECRET***"
            else:
                out[k] = _mask_any(v)
        return out
    if isinstance(x, list):
        return [_mask_any(v) for v in x]
    return x

def is_o_series_model(model: str) -> bool:
    """
    Robust heuristic for OpenAI o-series and models requiring max_completion_tokens.
    Handles namespaced model ids like 'openai/o1-mini', 'zzz/o3-mini', 'models/o1-preview'.
    Also identifies gpt-5.4 which requires max_completion_tokens (not max_tokens).
    """
    m = (model or "").strip().lower()
    m = m.split("/")[-1]
    m = m.split(":", 1)[0]
    if m.startswith(("o1", "o3", "o4", "o5")) or (len(m) >= 2 and m[0] == "o" and m[1].isdigit()):
        return True
    # gpt-5.4 also requires max_completion_tokens (not max_tokens) via zhizengzeng
    if m.startswith("gpt-5") or m.startswith("chatgpt-5"):
        return True
    return False


def prefers_responses_api(model: str) -> bool:
    """
    Heuristic: GPT-5-class models are more stable on the Responses API.
    Keep this as a recommendation rather than a hard switch so existing
    experiment configs remain reproducible.
    """
    m = (model or "").strip().lower()
    m = m.split("/")[-1]
    m = m.split(":", 1)[0]
    return m.startswith("gpt-5") or m.startswith("chatgpt-5")

def sanitize_openai_extra_body(model: str, extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fail-safe: prevent extra_body from sabotaging evaluation or breaking models.
    1. Canonicalization: camelCase tokens -> snake_case
    2. Integrity: forbid overriding core fields (linear scan)
    3. O-series / Standard logic: remap & clean
    """
    eb = dict(extra_body or {})
    modified_logs = []

    # --- 1. Canonicalization (Camel -> Snake) ---
    # Map common variations to standard snake_case keys
    # This catches maxOutputTokens, MaxTokens, etc.
    TOKEN_MAP = {
        "maxoutputtokens": "max_output_tokens",
        "maxcompletiontokens": "max_completion_tokens",
        "maxtokens": "max_tokens"
    }

    # Snapshot keys to allow modification
    original_keys = list(eb.keys())
    for k in original_keys:
        lk = k.lower().replace("_", "").replace("-", "")
        if lk in TOKEN_MAP:
            target = TOKEN_MAP[lk]
            if target != k:
                # Move value to standard key
                eb[target] = eb.pop(k)
                modified_logs.append(f"normalize: {k} -> {target}")

    # --- 2. Linear Scan Integrity Check ---
    # Scan ALL keys to catch 'streamOptions', 'Stream', etc.
    forbid = {"messages", "model", "stream", "stream_options", "streamoptions"}
    for k in eb.keys():
        if str(k).lower() in forbid:
             raise ValueError(f"extra_body must NOT contain '{k}' (would override/break constructed request).")

    # --- 3. Model-Specific Logic ---
    if is_o_series_model(model):
        # Remap aliases -> max_completion_tokens
        if "max_completion_tokens" not in eb:
            if "max_tokens" in eb:
                eb["max_completion_tokens"] = eb["max_tokens"]
                modified_logs.append("remap: max_tokens -> max_completion_tokens")
            elif "max_output_tokens" in eb:
                eb["max_completion_tokens"] = eb["max_output_tokens"]
                modified_logs.append("remap: max_output_tokens -> max_completion_tokens")

        # Drop incompatible aliases
        for k in ("max_tokens", "max_output_tokens"):
            if k in eb:
                eb.pop(k, None)
                modified_logs.append(f"drop: {k} (incompatible with o-series)")

        if "temperature" in eb:
            eb.pop("temperature", None)
            modified_logs.append("drop: temperature (fail-safe for o-series)")

    else:
        # Standard models
        # 1. Remap max_output_tokens -> max_tokens
        if "max_output_tokens" in eb and "max_tokens" not in eb:
            eb["max_tokens"] = eb["max_output_tokens"]
            modified_logs.append("remap: max_output_tokens -> max_tokens")

        # 2. Remap max_completion_tokens -> max_tokens (Prevent o-series config fail)
        if "max_completion_tokens" in eb and "max_tokens" not in eb:
            eb["max_tokens"] = eb["max_completion_tokens"]
            modified_logs.append("remap: max_completion_tokens -> max_tokens")

        # 3. Drop aliases unconditionally
        for k in ("max_output_tokens", "max_completion_tokens"):
            if k in eb:
                eb.pop(k, None)
                modified_logs.append(f"drop: {k} (standard model cleanup)")

    if modified_logs:
        print(f"🛡️ [Auto-Sanitize] Model='{model}': {', '.join(modified_logs)}", flush=True)

    return eb

def validate_backend_provider(backend: str, provider: str):
    allowed = {
        "openai_compatible": {
            "custom", "openai", "siliconflow", "closeai",
            # Unified Zhizengzeng providers (all use OpenAI-compatible interface)
            "zhizengzeng", "zhizengzeng_xai", "zhizengzeng_claude", "zhizengzeng_grok", "zhizengzeng_qwen"
        },
        "gemini": {"custom", "gemini", "zhizengzeng_gemini"},
        "qwen2.5-vl": None,
        "internvl2.5": None,
        "llava-next-video": None,
    "llava": None,
    "llava-next": None,
    }

    valid_set = allowed.get(backend)
    if valid_set is None:
        return

    if provider not in valid_set:
        raise ValueError(
            f"❌ Invalid provider '{provider}' for backend '{backend}'.\n"
            f"   Allowed providers: {sorted(list(valid_set))}"
        )


# --- Runners ---
class Runner:
    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

class OpenAICompatibleRunner(Runner):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_side: int,
        quality: int,
        max_retries: int,
        backoff_base: float,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        # Some OpenAI-compatible providers are sensitive to trailing slashes.
        self.client = OpenAI(api_key=api_key, base_url=(base_url or "").rstrip("/"), timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_side = max_side
        self.quality = quality
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.seed = seed
        self.extra_body = sanitize_openai_extra_body(model, extra_body)

    def _is_retryable(self, e: Exception) -> bool:
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg or "timeout" in msg:
            return True
        if "502" in msg or "503" in msg or "504" in msg:
            return True
        return False

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # SiliconFlow's multimodal examples follow OpenAI's content schema with image_url parts.
        # Default to images-first to preserve current behavior, but allow small request-level
        # probes for provider compatibility diagnostics.
        request_options = dict(request_options or {})
        content_order = str(request_options.get("content_order") or "images_first").strip().lower()
        image_detail = request_options.get("image_detail")

        image_parts = []
        for p in images:
            mime, b64 = encode_image_base64(
                p,
                max_side=self.max_side,
                quality=self.quality,
                prefer_png=getattr(self, "prefer_png", True),
            )
            image_url: Dict[str, Any] = {"url": f"data:{mime};base64,{b64}"}
            if image_detail:
                image_url["detail"] = image_detail
            image_parts.append({"type": "image_url", "image_url": image_url})

        text_part = {"type": "text", "text": prompt}
        if content_order in {"text_first", "text-first", "textfirst"}:
            content = [text_part] + image_parts
        else:
            content = image_parts + [text_part]

        last_err = None
        for i in range(self.max_retries + 1):
            try:
                req_kwargs = {}
                if self.seed is not None:
                    req_kwargs["seed"] = self.seed

                # Token limit logic (Sanitized)
                token_keys = ("max_completion_tokens", "max_tokens")
                if not any(k in self.extra_body for k in token_keys):
                    budget = int(max_tokens_override) if max_tokens_override is not None else int(self.max_tokens)
                    if is_o_series_model(self.model):
                        req_kwargs["max_completion_tokens"] = budget
                    else:
                        req_kwargs["max_tokens"] = budget

                # Temperature logic (Sanitized)
                if "temperature" not in self.extra_body and (not is_o_series_model(self.model)):
                    req_kwargs["temperature"] = float(self.temperature)

                if self.extra_body:
                    req_kwargs["extra_body"] = self.extra_body

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    **req_kwargs,
                )

                # Null-Safety
                if not getattr(resp, "choices", None):
                    return {"text": f"{API_ERROR_PREFIX} empty choices (gateway error)", "finish_reason": "error", "usage": None}

                ch0 = resp.choices[0]
                msg = getattr(ch0, "message", None)
                text = getattr(msg, "content", None) if msg else None
                finish_reason = getattr(ch0, "finish_reason", "unknown")
                usage_raw = getattr(resp, "usage", None)
                # Convert CompletionUsage to dict for JSON serialization
                usage = None
                if usage_raw:
                    usage = {
                        "prompt_tokens": getattr(usage_raw, "prompt_tokens", None),
                        "completion_tokens": getattr(usage_raw, "completion_tokens", None),
                        "total_tokens": getattr(usage_raw, "total_tokens", None),
                    }

                if not text:
                    return {"text": f"{API_ERROR_PREFIX} empty message.content (finish_reason={finish_reason})", "finish_reason": finish_reason, "usage": usage}
                return {"text": text, "finish_reason": finish_reason, "usage": usage}

            except Exception as e:
                last_err = e
                if i < self.max_retries and self._is_retryable(e):
                    time.sleep(self.backoff_base * (2**i))
                    continue
                return {"text": f"{API_ERROR_PREFIX} {str(e)}", "finish_reason": "error", "usage": None}
        return {"text": f"{API_ERROR_PREFIX} {str(last_err)}", "finish_reason": "error", "usage": None}


class OpenAIResponsesRunner(Runner):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_side: int,
        quality: int,
        max_retries: int,
        backoff_base: float,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=(base_url or "").rstrip("/"), timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_side = max_side
        self.quality = quality
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.seed = seed
        self.extra_body = dict(extra_body or {})

    def _is_retryable(self, e: Exception) -> bool:
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg or "timeout" in msg:
            return True
        if "502" in msg or "503" in msg or "504" in msg:
            return True
        return False

    def _extract_output_text(self, resp: Any) -> Optional[str]:
        text = getattr(resp, "output_text", None)
        if text:
            return text

        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for content_item in getattr(item, "content", []) or []:
                if getattr(content_item, "type", None) == "output_text":
                    chunk = getattr(content_item, "text", None)
                    if chunk:
                        parts.append(chunk)
        merged = "".join(parts).strip()
        return merged or None

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request_options = dict(request_options or {})
        content_order = str(request_options.get("content_order") or "images_first").strip().lower()
        image_detail = request_options.get("image_detail")

        image_items = []
        for p in images:
            mime, b64 = encode_image_base64(
                p,
                max_side=self.max_side,
                quality=self.quality,
                prefer_png=getattr(self, "prefer_png", True),
            )
            image_item = {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
            if image_detail:
                image_item["detail"] = image_detail
            image_items.append(image_item)

        text_item = {"type": "input_text", "text": prompt}
        if content_order in {"text_first", "text-first", "textfirst"}:
            content = [text_item] + image_items
        else:
            content = image_items + [text_item]

        last_err = None
        for i in range(self.max_retries + 1):
            try:
                req_kwargs = {
                    "model": self.model,
                    "input": [{"role": "user", "content": content}],
                }
                budget = int(max_tokens_override) if max_tokens_override is not None else int(self.max_tokens)
                req_kwargs["max_output_tokens"] = budget

                if self.seed is not None:
                    req_kwargs["seed"] = self.seed

                if "temperature" not in self.extra_body and (not is_o_series_model(self.model)):
                    req_kwargs["temperature"] = float(self.temperature)

                if self.extra_body:
                    req_kwargs["extra_body"] = self.extra_body

                resp = self.client.responses.create(**req_kwargs)

                usage_raw = getattr(resp, "usage", None)
                usage = None
                if usage_raw:
                    usage = {
                        "prompt_tokens": getattr(usage_raw, "input_tokens", None) or getattr(usage_raw, "prompt_tokens", None),
                        "completion_tokens": getattr(usage_raw, "output_tokens", None) or getattr(usage_raw, "completion_tokens", None),
                        "total_tokens": getattr(usage_raw, "total_tokens", None),
                    }

                text = self._extract_output_text(resp)
                status = getattr(resp, "status", None) or "unknown"
                incomplete = getattr(resp, "incomplete_details", None)
                reason = getattr(incomplete, "reason", None) if incomplete else None
                finish_reason = reason or status

                if not text:
                    return {
                        "text": f"{API_ERROR_PREFIX} empty response.output_text (status={status}, reason={reason})",
                        "finish_reason": finish_reason,
                        "usage": usage,
                    }
                return {"text": text, "finish_reason": finish_reason, "usage": usage}

            except Exception as e:
                last_err = e
                if i < self.max_retries and self._is_retryable(e):
                    time.sleep(self.backoff_base * (2**i))
                    continue
                return {"text": f"{API_ERROR_PREFIX} {str(e)}", "finish_reason": "error", "usage": None}
        return {"text": f"{API_ERROR_PREFIX} {str(last_err)}", "finish_reason": "error", "usage": None}


class GeminiRunner(Runner):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_side: int,
        quality: int,
        max_retries: int,
        backoff_base: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.model = self._canon_model(model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_side = max_side
        self.quality = quality
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.extra_body = extra_body or {}

    @staticmethod
    def _canon_model(m: str) -> str:
        m = (m or "").strip()
        if m.startswith("models/"):
            m = m[len("models/"):]
        if ":generatecontent" in m.lower():
            m = m.split(":", 1)[0]
        return m

    def _is_retryable(self, status: Optional[int], err_text: str) -> bool:
        t = (err_text or "").lower()
        if status in (429, 500, 502, 503, 504):
            return True
        if "rate" in t and "limit" in t:
            return True
        if "timeout" in t:
            return True
        if "connection" in t or "reset" in t:
            return True
        return False

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        import requests

        url = f"{self.base_url}v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }

        request_options = dict(request_options or {})
        content_order = str(request_options.get("content_order") or "images_first").strip().lower()

        image_parts = []
        for p in images:
            mime, b64 = encode_image_base64(
                p,
                max_side=self.max_side,
                quality=self.quality,
                prefer_png=getattr(self, "prefer_png", True),
            )
            image_parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        text_part = {"text": prompt}
        if content_order in {"text_first", "text-first", "textfirst"}:
            parts = [text_part] + image_parts
        else:
            parts = image_parts + [text_part]

        # Use max_tokens_override if provided, otherwise use default max_tokens
        max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }

        if self.extra_body:
            eb = dict(self.extra_body)
            # Integrity Check
            if "contents" in eb:
                raise ValueError("extra_body must NOT contain 'contents' (would override constructed prompt/images).")

            gc = eb.pop("generationConfig", None)
            if isinstance(gc, dict):
                payload["generationConfig"].update(gc)

            payload.update(eb)

        last_err = None
        for i in range(self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    last_err = f"HTTP {r.status_code}: {r.text}"
                    if i < self.max_retries and self._is_retryable(r.status_code, r.text):
                        time.sleep(self.backoff_base * (2**i))
                        continue
                    return f"{API_ERROR_PREFIX} {last_err}"

                data = r.json()
                cands = data.get("candidates") or []
                if not cands:
                    return f"{API_ERROR_PREFIX} empty candidates"

                content = (cands[0].get("content") or {})
                out_parts = content.get("parts") or []
                texts = [p["text"] for p in out_parts if isinstance(p, dict) and "text" in p]
                return "".join(texts).strip() if texts else f"{API_ERROR_PREFIX} empty text"
            except Exception as e:
                last_err = str(e)
                if i < self.max_retries and self._is_retryable(None, last_err):
                    time.sleep(self.backoff_base * (2**i))
                    continue
                return f"{API_ERROR_PREFIX} {last_err}"
        return f"{API_ERROR_PREFIX} {str(last_err)}"


class Qwen25VLRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)
        self.process_vision_info = process_vision_info
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
        else: self.cfg.update({"do_sample": False})

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        content = [{"type": "image", "image": p} for p in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_cfg)
        return self.processor.batch_decode([out[len(inp):] for inp, out in zip(inputs["input_ids"], ids)], skip_special_tokens=True)[0]

class InternVLRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Handle multi-image scenarios for GridWM-Judge tasks
        if len(images) == 1:
            # Single image (Task B)
            img = safe_load_rgb(images[0])
        else:
            # Multi-image fusion (Task A: 5 images, Task C: storyboard)
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Smart fusion strategy based on image count
                if len(images) <= 3:
                    mode = "h"  # Horizontal fusion for few images (Task A)
                else:
                    mode = "v"  # Vertical fusion for many images (Task C)

                # Use high quality for fusion to preserve details
                fused_path = fuse_images(images, Path(tmp_path), mode, quality=95)
                img = safe_load_rgb(fused_path)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Resize image to square for InternVL compatibility (448x448 as per model config)
        img = img.resize((448, 448), Image.Resampling.LANCZOS)

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        # For InternVL, use direct generation instead of chat method
        try:
            import numpy as np
            import torch
            from transformers import AutoTokenizer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)

            # Prepare image
            img_array = np.array(img)
            pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            pixel_values = pixel_values.to(dtype=self.model.dtype, device=self.model.device)

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Add pixel values to inputs
            inputs['pixel_values'] = pixel_values
            inputs['image_flags'] = torch.tensor([[1]], dtype=torch.long, device=self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_cfg)

            # Decode response
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"InternVL generation failed: {str(e)}") from e

class LlavaRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        import numpy as np

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.np = np
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            # For single image, use standard LLaVA format
            if len(images) == 1:
                image = safe_load_rgb(images[0])
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt")
            else:
                # For multiple images, concatenate them horizontally
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=95)
                    image = safe_load_rgb(fused_path)

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(text=text, images=image, return_tensors="pt")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Remove image_sizes if present (compatibility issue)
            if 'image_sizes' in inputs:
                del inputs['image_sizes']

            # Create generation config with potential token override
            gen_cfg = dict(self.cfg)
            if max_tokens_override is not None:
                gen_cfg["max_new_tokens"] = max_tokens_override

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_cfg)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0]
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"LLaVA generation failed: {str(e)}") from e

class LlavaNextRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        import numpy as np

        # Set up attention implementation for maximum speed
        attn_config = {}
        if use_flash_attention:
            try:
                # Check if flash_attn is available
                import flash_attn
                attn_config["attn_implementation"] = "flash_attention_2"
                print("🎯 Using Flash Attention 2 for accelerated inference", flush=True)
            except ImportError:
                print("⚠️ Flash Attention 2 not available, falling back to eager attention", flush=True)
                attn_config["attn_implementation"] = "eager"

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.np = np

        # Optimized generation config
        self.cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        # Remove None values
        self.cfg = {k: v for k, v in self.cfg.items() if v is not None}

        # Set batch processing capability
        self.max_batch_size = 4  # Conservative batch size for stability

    def _prepare_batch_inputs(self, batch_requests: List[Tuple[List[str], str, Optional[int]]]) -> Tuple[Dict, List]:
        """
        Prepare batched inputs for multiple requests.
        Returns: (batched_inputs, original_indices)
        """
        batch_images = []
        batch_texts = []
        batch_indices = []

        for idx, (images, prompt, _) in enumerate(batch_requests):
            # High-quality image processing
            if len(images) == 1:
                image = safe_load_rgb(images[0])
            else:
                # Lossless PNG fusion for multiple images
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=100)
                    image = safe_load_rgb(fused_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            batch_images.append(image)
            batch_indices.append(idx)

            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            batch_texts.append(text)

        # Batch process
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs, batch_indices

    def generate_batch(self, batch_requests: List[Tuple[List[str], str, Optional[int]]]) -> List[str]:
        """
        Generate responses for a batch of requests.
        Each request is (images, prompt, max_tokens_override)
        """
        if len(batch_requests) == 0:
            return []

        try:
            # Prepare batched inputs
            inputs, batch_indices = self._prepare_batch_inputs(batch_requests)

            # Get max tokens for the batch
            max_tokens = max((req[2] if req[2] is not None else self.cfg["max_new_tokens"])
                           for req in batch_requests)

            # Create generation config
            gen_cfg = dict(self.cfg)
            gen_cfg["max_new_tokens"] = max_tokens

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_cfg)

            # Decode each response
            responses = []
            for i in range(len(batch_requests)):
                response = self.processor.decode(outputs[i], skip_special_tokens=True)
                responses.append(response.strip())

            return responses

        except Exception as e:
            # Fallback to individual processing if batch fails
            print(f"⚠️ Batch processing failed ({e}), falling back to individual processing", flush=True)
            return [self.generate(images, prompt, max_tokens) for images, prompt, max_tokens in batch_requests]

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            # High-quality image processing (no lossy compression)
            if len(images) == 1:
                # Load image with original quality and size
                image = safe_load_rgb(images[0])
                # Keep original size for maximum quality (LLaVA-NeXT handles resizing internally)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt")
            else:
                # For multiple images, use PNG for lossless fusion
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    # Use PNG quality=100 for lossless fusion
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=100)
                    image = safe_load_rgb(fused_path)

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(text=text, images=image, return_tensors="pt")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Create generation config with potential token override
            gen_cfg = dict(self.cfg)
            if max_tokens_override is not None:
                gen_cfg["max_new_tokens"] = max_tokens_override

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_cfg)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"LLaVA-NeXT generation failed: {str(e)}") from e

class LlavaNextVideoRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        import numpy as np

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        self.np = np
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(
        self,
        images: List[str],
        prompt: str,
        max_tokens_override: Optional[int] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        frames = [self.np.array(safe_load_rgb(p)) for p in images]
        video = self.np.stack(frames, axis=0)
        conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, videos=video, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_cfg)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

def resolve_openai_provider(provider, base_url, api_key):
    if provider == "openai":
        key = api_key or _pick_env("OPENAI_API_KEY")
        return (base_url or "https://api.openai.com/v1"), _need("OPENAI_API_KEY", key)
    if provider == "siliconflow":
        key = api_key or _pick_env("SILICONFLOW_API_KEY")
        # SiliconFlow provides an OpenAI-compatible Chat Completions endpoint.
        # Default CN endpoint: https://api.siliconflow.cn/v1
        # Allow overrides via --base_url or env SILICONFLOW_BASE_URL (useful for non-CN endpoints).
        sf_base = base_url or _pick_env("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        return sf_base.rstrip("/"), _need("SILICONFLOW_API_KEY", key)
    if provider == "closeai":
        key = api_key or _pick_env("CLOSEAI_API_KEY")
        return "https://api.openai-proxy.org/v1", _need("CLOSEAI_API_KEY", key)
    if provider == "zhizengzeng_qwen":
        # Zhizengzeng Qwen VL models - uses Alibaba base URL
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_API_KEY")
        return "https://api.zhizengzeng.com/alibaba", _need("ZZZ_API_KEY or ZHIZENGZENG_API_KEY", key)
    if provider.startswith("zhizengzeng"):
        # Unified Zhizengzeng endpoint - all models via OpenAI-compatible interface
        # Support both official ZZZ_API_KEY and legacy ZHIZENGZENG_API_KEY for compatibility
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_API_KEY")
        return "https://api.zhizengzeng.com/v1", _need("ZZZ_API_KEY or ZHIZENGZENG_API_KEY", key)
    if provider == "custom":
        if not base_url: raise ValueError("provider=custom requires --base_url")
        key = api_key or _pick_env("OPENAI_API_KEY")
        return base_url, _need("custom api_key (or OPENAI_API_KEY)", key)
    return base_url, api_key

def resolve_gemini_provider(provider, base_url, api_key):
    if provider == "gemini":
        key = api_key or _pick_env("GEMINI_API_KEY")
        return (base_url or "https://generativelanguage.googleapis.com/"), _need("GEMINI_API_KEY", key)
    if provider == "zhizengzeng_gemini":
        # Zhizengzeng Gemini API uses Google-compatible format but different base_url
        # Official Google: https://generativelanguage.googleapis.com/
        # Zhizengzeng: https://api.zhizengzeng.com/google/
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_GEMINI_API_KEY")
        return "https://api.zhizengzeng.com/google/", _need("ZZZ_API_KEY or ZHIZENGZENG_GEMINI_API_KEY", key)
    if provider == "custom":
        if not base_url: raise ValueError("provider=custom requires --base_url")
        key = api_key or _pick_env("GEMINI_API_KEY")
        return base_url, _need("custom api_key (or GEMINI_API_KEY)", key)
    return base_url, api_key

def run(args):
    validate_backend_provider(args.backend, args.provider)

    # Resolve requests path (support exam mode)
    if hasattr(args, 'exam_dir') and args.exam_dir:
        gen_dir = Path(args.responses_dir) / "_requests_from_exam"
        out_files = generate_requests_from_exam(
            Path(args.exam_dir),
            gen_dir,
            getattr(args, 'exam_task', 'all'),
            task_c_exam_path=getattr(args, 'task_c_exam', None),
            task_d_exam_path=getattr(args, 'task_d_exam', None),
            task_e_exam_path=getattr(args, 'task_e_exam', None),
            taskb_prompt_variant=getattr(args, 'taskb_prompt_variant', 'v7'),
        )
        exam_task = getattr(args, 'exam_task', 'all')
        if exam_task == "all":
            combined = gen_dir / "requests_exam_all.jsonl"
            with combined.open("w") as fout:
                for f in out_files: shutil.copyfileobj(f.open("r"), fout)
            req_path = combined
        else: req_path = out_files[0]
        if getattr(args, 'build_exam_requests_only', False): return
    else:
        req_path = Path(args.requests)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    extra_body = {}
    if args.extra_body:
        with open(args.extra_body, "r", encoding="utf-8") as f:
            extra_body = json.load(f)
        # Deep secure masking
        print(f"🔧 Loaded extra_body (Masked): {_mask_any(extra_body)}", flush=True)

    protocol_tag = f"fused_{args.force_fuse}" if args.force_fuse else "native"
    model_tag = args.model_tag or args.model.replace("/", "_").replace(":", "_")[:50]  # Limit length
    backend_tag = args.backend.replace("_", "").replace("-", "")
    provider_tag = args.provider.replace("_", "").replace("-", "")

    # Create experiment identifier based on key parameters
    request_api_tag = ""
    if args.backend == "openai_compatible":
        request_api_tag = f"_{args.openai_request_api}"

    experiment_id = f"{backend_tag}_{provider_tag}_{model_tag}{request_api_tag}"
    if args.num_shards > 1:
        experiment_id += f"_shard{args.shard_id}"
    if args.force_fuse:
        experiment_id += f"_{protocol_tag}"

    resp_dir = Path(args.responses_dir) / experiment_id
    resp_dir.mkdir(parents=True, exist_ok=True)
    resp_path = resp_dir / f"{req_path.stem}.jsonl"

    dtype = parse_dtype(args.dtype)

    if args.backend == "openai_compatible":
        url, key = resolve_openai_provider(args.provider, args.base_url, args.api_key)
        if args.openai_request_api == "chat" and prefers_responses_api(args.model):
            print(
                f"⚠️  Model '{args.model}' may be more stable via --openai_request_api responses.",
                flush=True,
            )
        runner_cls = OpenAIResponsesRunner if args.openai_request_api == "responses" else OpenAICompatibleRunner
        runner = runner_cls(
            model=args.model,
            api_key=key,
            base_url=url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            max_side=args.api_image_max_side,
            quality=args.api_jpeg_quality,
            max_retries=args.api_max_retries,
            backoff_base=args.api_backoff_base,
            seed=args.seed,
            extra_body=extra_body,
        )
        # Set PNG preference for image fidelity
        if hasattr(runner, "__dict__"):
            runner.prefer_png = bool(getattr(args, "api_prefer_png", True))

    elif args.backend == "gemini":
        url, key = resolve_gemini_provider(args.provider, args.base_url, args.api_key)
        runner = GeminiRunner(
            model=args.model,
            api_key=key,
            base_url=url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            max_side=args.api_image_max_side,
            quality=args.api_jpeg_quality,
            max_retries=args.api_max_retries,
            backoff_base=args.api_backoff_base,
            extra_body=extra_body,
        )
        if hasattr(runner, "__dict__"):
            runner.prefer_png = bool(getattr(args, "api_prefer_png", True))

    elif args.backend == "qwen2.5-vl":
        runner = Qwen25VLRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "internvl2.5":
        runner = InternVLRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava-next-video":
        runner = LlavaNextVideoRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava":
        runner = LlavaRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava-next":
        runner = LlavaNextRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------------
    # Progress configuration
    # -------------------------
    task_overrides = {"A": args.max_tokens_A, "B": args.max_tokens_B, "C": args.max_tokens_C}
    t_start = time.time()
    n_err = 0

    processed = set()
    if args.resume and resp_path.exists():
        for r in load_jsonl(resp_path):
            done_uid = resolve_uid(r)
            if done_uid:
                processed.add(done_uid)
        print(f"🔄 Resuming {resp_path}: {len(processed)} done.", flush=True)

    # Optional total counting for better ETA (fast enough for ~1-10k lines)
    args._total_requests = 0
    if args.progress_total:
        try:
            with req_path.open("r", encoding="utf-8") as fcnt:
                for _ in fcnt:
                    args._total_requests += 1
        except Exception:
            args._total_requests = 0

    n_do = 0
    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time
    processed_count = len(processed)
    total_processed = processed_count
    last_progress_count = processed_count  # Track items since last progress report

    print(f"🔄 Starting inference on shard {args.shard_id} (resume: {processed_count} already done)", flush=True)

    # Check if runner supports batch processing
    use_batch = hasattr(runner, 'generate_batch') and hasattr(runner, 'max_batch_size')

    def _resolve_response_exam_meta(q: Dict[str, Any], uid: str) -> Tuple[str, str, Dict[str, Any]]:
        exam_id = str(q.get("exam_id") or uid)
        schema_version = str(q.get("schema_version") or EXAM_SCHEMA_VERSION)
        if isinstance(q.get("exam"), dict):
            exam_meta = dict(q["exam"])
        else:
            exam_meta = build_exam_meta(exam_id, row=q, schema_version=schema_version)
        exam_meta.setdefault("exam_id", exam_id)
        exam_meta.setdefault("uid", exam_id)
        if "schema_version" not in exam_meta:
            exam_meta["schema_version"] = schema_version
        if "task" not in exam_meta:
            exam_meta.update(parse_exam_id(exam_id))
        return exam_id, schema_version, exam_meta

    def _write_response_row(
        fout_handle: Any,
        q: Dict[str, Any],
        uid: str,
        pred: str,
        raw: str,
        ok: bool,
        err: Optional[str],
        latency_ms: float,
        fused: bool,
        finish_reason: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> None:
        if isinstance(raw, str) and raw.startswith(API_ERROR_PREFIX):
            ok = False
            if not err:
                err = raw[len(API_ERROR_PREFIX):].strip() or raw
            pred = ""
            if not finish_reason:
                finish_reason = "error"
        exam_id, schema_version, exam_meta = _resolve_response_exam_meta(q, uid)
        payload = {
            "uid": uid,
            "exam_id": exam_id,
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "exam_schema_version": schema_version,
            "pred": pred,
            "raw": raw,
            "meta": {
                "ok": ok,
                "error": err,
                "shard": args.shard_id,
                "latency_ms": latency_ms,
                "fused": fused,
                "protocol": protocol_tag,
                "backend": args.backend,
                "provider": args.provider,
                "model": args.model,
                "finish_reason": finish_reason,
                "usage": usage,
                "exam": exam_meta,
            },
        }
        fout_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fout_handle.flush()

    with req_path.open("r", encoding="utf-8") as fin, resp_path.open("a", encoding="utf-8") as fout:
        if use_batch:
            # Batch processing mode
            batch_requests = []
            batch_metadata = []

            for line in fin:
                if not line.strip(): continue
                q = json.loads(line)
                uid = q.get("uid") or q.get("exam_id")
                if not uid:
                    continue

                if args.num_shards > 1:
                    if (stable_hash64(uid) % args.num_shards) != args.shard_id: continue
                if uid in processed: continue

                images = q["images"]
                prompt = q["prompt"]
                fused = False

                # P0-2: Hard Protocol Enforcement & Prompt Injection
                if args.force_fuse:
                    if len(images) <= 1:
                        raise ValueError(f"Protocol Violation: --force_fuse set but found single image for {uid}. Task C Storyboard should run WITHOUT force_fuse.")

                    fused = True
                    fuse_out = tmp_dir / f"{uid.replace(':','_').replace('/','_')}.jpg"
                    images = [fuse_images(images, fuse_out, args.force_fuse, args.api_jpeg_quality)]

                    # Dynamic Prompt Replacement
                    if args.force_fuse == "h" and q.get("prompt_fused_h"):
                        prompt = q["prompt_fused_h"]
                    elif args.force_fuse == "v" and q.get("prompt_fused_v"):
                        prompt = q["prompt_fused_v"]
                    else:
                        layout = "horizontal" if args.force_fuse == "h" else "vertical"
                        prompt = f"NOTE: The provided image is a {layout} concatenation.\n\n{prompt}"

                budget = _task_max_tokens(uid, args.max_new_tokens, task_overrides)
                batch_requests.append((images, prompt, budget))
                batch_metadata.append((q, fused))

                # Process batch when full or at end
                if len(batch_requests) >= runner.max_batch_size:
                    t0 = time.time()
                    try:
                        results = runner.generate_batch(batch_requests)
                        for i, result in enumerate(results):
                            q, fused = batch_metadata[i]
                            uid = q.get("uid") or q.get("exam_id")
                            if not uid:
                                continue

                            if isinstance(result, dict):
                                pred = result["text"]
                                finish_reason = result.get("finish_reason")
                                usage = result.get("usage")
                            else:
                                pred = result
                                finish_reason = None
                                usage = None

                            err = None
                            ok = True

                            _write_response_row(
                                fout_handle=fout,
                                q=q,
                                uid=uid,
                                pred=pred,
                                raw=pred,
                                ok=ok,
                                err=err,
                                latency_ms=(time.time() - t0) * 1000.0 / len(batch_requests),
                                fused=fused,
                                finish_reason=finish_reason,
                                usage=usage,
                            )
                            n_do += 1
                            total_processed += 1

                    except Exception as e:
                        # Fallback to individual processing for failed batch
                        pass

                    # Progress reporting for batch processing
                    current_time = time.time()
                    items_since_last_report = total_processed - last_progress_count
                    time_since_last_report = current_time - last_progress_time

                    should_report = False
                    if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                        should_report = True
                    if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                        should_report = True

                    if should_report:
                        elapsed = current_time - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        print(f"📊 Shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                        last_progress_time = current_time
                        last_progress_count = total_processed
                        print(f"⚠️ Batch processing failed, falling back to individual processing", flush=True)
                        for i, (images, prompt, budget) in enumerate(batch_requests):
                            q, fused = batch_metadata[i]
                            uid = q.get("uid") or q.get("exam_id")
                            if not uid:
                                continue
                            t0 = time.time()
                            err = None
                            try:
                                request_options = _extract_request_options(q)
                                result = runner.generate(
                                    images,
                                    prompt,
                                    max_tokens_override=budget,
                                    request_options=request_options if request_options else None,
                                )
                                if isinstance(result, dict):
                                    pred = result["text"]
                                    finish_reason = result.get("finish_reason")
                                    usage = result.get("usage")
                                else:
                                    pred = result
                                    finish_reason = None
                                    usage = None
                            except Exception as e:
                                pred = f"{API_ERROR_PREFIX} {str(e)}"
                                err = str(e)
                                finish_reason = "error"
                                usage = None

                            ok = (err is None)
                            raw = pred
                            if not ok:
                                pred = ""

                            _write_response_row(
                                fout_handle=fout,
                                q=q,
                                uid=uid,
                                pred=pred,
                                raw=raw,
                                ok=ok,
                                err=err,
                                latency_ms=(time.time() - t0) * 1000.0,
                                fused=fused,
                                finish_reason=finish_reason,
                                usage=usage,
                            )
                            n_do += 1
                            total_processed += 1

                    batch_requests = []
                    batch_metadata = []

            # Process remaining batch
            if batch_requests:
                t0 = time.time()
                try:
                    results = runner.generate_batch(batch_requests)
                    for i, result in enumerate(results):
                        q, fused = batch_metadata[i]
                        uid = q.get("uid") or q.get("exam_id")
                        if not uid:
                            continue

                        if isinstance(result, dict):
                            pred = result["text"]
                            finish_reason = result.get("finish_reason")
                            usage = result.get("usage")
                        else:
                            pred = result
                            finish_reason = None
                            usage = None

                        err = None
                        ok = True

                        _write_response_row(
                            fout_handle=fout,
                            q=q,
                            uid=uid,
                            pred=pred,
                            raw=pred,
                            ok=ok,
                            err=err,
                            latency_ms=(time.time() - t0) * 1000.0 / len(batch_requests),
                            fused=fused,
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                        n_do += 1
                        total_processed += 1

                except Exception as e:
                    # Fallback to individual processing
                    print(f"⚠️ Final batch processing failed, falling back to individual processing", flush=True)

                # Progress reporting for batch mode
                current_time = time.time()
                items_since_last_report = total_processed - last_progress_count
                time_since_last_report = current_time - last_progress_time

                should_report = False
                if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                    should_report = True
                if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                    should_report = True

                if should_report:
                    elapsed = current_time - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"📊 Shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                    last_progress_time = current_time
                    last_progress_count = total_processed
                    for i, (images, prompt, budget) in enumerate(batch_requests):
                        q, fused = batch_metadata[i]
                        uid = q.get("uid") or q.get("exam_id")
                        if not uid:
                            continue
                        t0 = time.time()
                        err = None
                        try:
                            request_options = _extract_request_options(q)
                            result = runner.generate(
                                images,
                                prompt,
                                max_tokens_override=budget,
                                request_options=request_options if request_options else None,
                            )
                            if isinstance(result, dict):
                                pred = result["text"]
                                finish_reason = result.get("finish_reason")
                                usage = result.get("usage")
                            else:
                                pred = result
                                finish_reason = None
                                usage = None
                        except Exception as e:
                            pred = f"{API_ERROR_PREFIX} {str(e)}"
                            err = str(e)
                            finish_reason = "error"
                            usage = None

                        ok = (err is None)
                        raw = pred
                        if not ok:
                            pred = ""

                        _write_response_row(
                            fout_handle=fout,
                            q=q,
                            uid=uid,
                            pred=pred,
                            raw=raw,
                            ok=ok,
                            err=err,
                            latency_ms=(time.time() - t0) * 1000.0,
                            fused=fused,
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                        n_do += 1
                        total_processed += 1
        else:
            # Original single-item processing mode
            for line in fin:
                if not line.strip(): continue
                q = json.loads(line)
                uid = q.get("uid") or q.get("exam_id")
                if not uid:
                    continue

                if args.num_shards > 1:
                    if (stable_hash64(uid) % args.num_shards) != args.shard_id: continue
                if uid in processed: continue

                images = q["images"]
                prompt = q["prompt"]
                fused = False

                # P0-2: Hard Protocol Enforcement & Prompt Injection
                if args.force_fuse:
                    if len(images) <= 1:
                        # Single image: skip Fuse, raise error to prevent contamination
                        raise ValueError(f"Protocol Violation: --force_fuse set but found single image for {uid}. Task C Storyboard should run WITHOUT force_fuse.")

                    fused = True
                    fuse_out = tmp_dir / f"{uid.replace(':','_').replace('/','_')}.jpg"
                    images = [fuse_images(images, fuse_out, args.force_fuse, args.api_jpeg_quality)]

                    # Dynamic Prompt Replacement
                    if args.force_fuse == "h" and q.get("prompt_fused_h"):
                        prompt = q["prompt_fused_h"]
                    elif args.force_fuse == "v" and q.get("prompt_fused_v"):
                        prompt = q["prompt_fused_v"]
                    else:
                        layout = "horizontal" if args.force_fuse == "h" else "vertical"
                        prompt = f"NOTE: The provided image is a {layout} concatenation.\n\n{prompt}"

                # P0-1: D7 scan BEFORE sending to model (SSOT §4.5 + Stats Spec §1)
                # Previously d7_result was used but the scan call was missing — prompts with
                # forbidden tokens leaked to models, invalidating all SCI metrics
                if D7_SCAN_ENABLED and D7_SCAN_AVAILABLE:
                    d7_result = d7_scan_payload(prompt)
                    if not d7_result.passed:
                        err = f"D7_VIOLATION:{','.join(d7_result.violations)}"
                        print(f"[D7] {uid}: BLOCKED - Forbidden tokens: {d7_result.violations}")
                        _write_response_row(
                            fout_handle=fout,
                            q=q,
                            uid=uid,
                            pred="",
                            raw="",
                            ok=False,
                            err=err,
                            latency_ms=0,
                            fused=fused,
                            finish_reason="d7_rejected",
                            usage=None,
                        )
                        n_do += 1
                        total_processed += 1
                        continue

                if args.min_interval_ms > 0 and n_do > 0:
                    time.sleep(args.min_interval_ms / 1000.0)

                t0 = time.time()
                err = None
                try:
                    # Task-aware budget to avoid truncation -> fake JSON failures
                    budget = _task_max_tokens(uid, args.max_new_tokens, task_overrides)
                    request_options = _extract_request_options(q)
                    result = runner.generate(
                        images,
                        prompt,
                        max_tokens_override=budget,
                        request_options=request_options if request_options else None,
                    )
                    if isinstance(result, dict):
                        pred = result["text"]
                        finish_reason = result.get("finish_reason")
                        usage = result.get("usage")
                    else:
                        # Fallback for backward compatibility
                        pred = result
                        finish_reason = None
                        usage = None
                except Exception as e:
                    pred = f"{API_ERROR_PREFIX} {str(e)}"
                    err = str(e)
                    finish_reason = "error"
                    usage = None

                ok = (err is None)
                raw = pred
                if not ok:
                    pred = ""

                _write_response_row(
                    fout_handle=fout,
                    q=q,
                    uid=uid,
                    pred=pred,
                    raw=raw,
                    ok=ok,
                    err=err,
                    latency_ms=(time.time() - t0) * 1000.0,
                    fused=fused,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                n_do += 1
                total_processed += 1

                # Progress reporting - check both time interval and item count
                current_time = time.time()
                items_since_last_report = total_processed - last_progress_count
                time_since_last_report = current_time - last_progress_time

                should_report = False
                if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                    should_report = True
                if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                    should_report = True

                if should_report:
                    elapsed = current_time - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"📊 Shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                    last_progress_time = current_time
                    last_progress_count = total_processed

    # Final progress report
    final_time = time.time()
    total_elapsed = final_time - start_time
    final_rate = total_processed / total_elapsed if total_elapsed > 0 else 0
    print(f"✅ Shard {args.shard_id} completed: {total_processed} total, {final_rate:.2f}/sec, total time: {total_elapsed:.1f}s", flush=True)


# -------------------------
# Utilities
# -------------------------

def stable_hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def parse_dtype(s: str):
    if s in (None, "", "auto"): return "auto"
    if not _torch_available:
        raise ImportError("torch is not available; local model inference requires torch. For API-based inference, use --backend openai_compatible or --backend gemini.")
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def safe_load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()

def encode_image_base64_jpeg(path: str, max_side: int, quality: int) -> str:
    img = safe_load_rgb(path)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def fuse_images(paths: List[str], out_path: Path, mode: str, quality: int) -> str:
    imgs = [safe_load_rgb(p) for p in paths]
    ws, hs = zip(*[im.size for im in imgs])
    if mode == "h":
        W, H = sum(ws), max(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        x = 0
        for im in imgs:
            canvas.paste(im, (x, 0))
            x += im.size[0]
    else:
        W, H = max(ws), sum(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for im in imgs:
            canvas.paste(im, (0, y))
            y += im.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=quality)
    return str(out_path)

def _need(name: str, v: Optional[str]) -> str:
    if not v: raise ValueError(f"Missing {name}")
    return v

def _pick_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v: return v
    return None

def _mask_any(x: Any) -> Any:
    SECRET_EXACT = {"api_key", "apikey", "access_token", "authorization", "auth", "secret", "password"}
    def is_secret_key(key_str: str) -> bool:
        lk = key_str.lower().replace("_", "").replace("-", "")
        return lk in SECRET_EXACT or lk.endswith("token") or lk.endswith("key")

    if isinstance(x, dict):
        return {k: ("***MASKED***" if is_secret_key(str(k)) else _mask_any(v)) for k, v in x.items()}
    if isinstance(x, list): return [_mask_any(v) for v in x]
    return x

# -------------------------
# Exam helpers (Metadata Resurrection)
# -------------------------

def parse_exam_id(uid: str) -> Dict[str, Any]:
    return parse_exam_id_ssot(uid)

# -------------------------
# Exam -> Requests Logic
# -------------------------

# =============================================================================
# Task B Prompt Variants (for paired smoke comparison)
# =============================================================================

_TASK_B_PROMPT_V7 = """Return ONLY a JSON object with exactly these keys: agent, front_cell, objects.

COORDINATE SYSTEM:
- Use 1-based image coordinates.
- The origin is the BOTTOM-LEFT corner of the 7x7 image.
- x increases to the RIGHT.
- y increases UPWARD.
- Report every position as [x=col, y=row].
- Valid range: x in [1,7], y in [1,7].

COLOR VOCABULARY:
- Only use: red, green, blue, yellow, purple, grey.
- Use canonical colors only.
- Lava must be red.
- Goal must be green.
- Do not invent color names.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use this exact encoding:
  - 0 = east / right
  - 1 = south / down
  - 2 = west / left
  - 3 = north / up
- dir must be one of: 0, 1, 2, 3.
- Do not omit dir.
- Do not output null for dir.

STATE FIELD:
- You MUST output a numeric `state` for front_cell and for every object.
- For doors only, use:
  - 0 = open
  - 1 = closed
  - 2 = locked
- For all non-door types, use `state = null`.
- Do not output 0 for non-door types unless the tile/object is a door.
- Do not output strings such as "open" or "closed".

RULES:
- Judge only from the image.
- Do not assume any fixed agent position.
- front_cell must be reported as a separate field.
- front_cell.type is the OCCUPANCY type of the tile directly in front of the agent.
- Allowed front_cell.type values are: empty, wall, door, goal, ball, key, lava, box.
- Do NOT use floor, grey, color words, or free-form descriptions as front_cell.type.
- Do NOT include the front_cell tile in objects.
- Only report NON-BACKGROUND objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.
- Do not invent objects, colors, or states.

OUTPUT:
- Return raw JSON only.
- No markdown.
- No code blocks.
- No explanation.

Required schema:
{
  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},
  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},
  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]
}"""

_TASK_B_PROMPT_V7_LEGEND = """Return ONLY a JSON object with exactly these keys: agent, front_cell, objects.

COORDINATE SYSTEM:
- Use 1-based image coordinates.
- The origin is the BOTTOM-LEFT corner of the 7x7 image.
- x increases to the RIGHT.
- y increases UPWARD.
- Report every position as [x=col, y=row].
- Valid range: x in [1,7], y in [1,7].

COLOR VOCABULARY:
- Only use: red, green, blue, yellow, purple, grey.
- Use canonical colors only.
- Lava must be red.
- Goal must be green.
- Do not invent color names.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use this exact encoding:
  - 0 = east / right
  - 1 = south / down
  - 2 = west / left
  - 3 = north / up
- dir must be one of: 0, 1, 2, 3.
- Do not omit dir.
- Do not output null for dir.

STATE FIELD — DOOR VISUAL LEGEND (CRITICAL):
- You MUST output a numeric `state` for front_cell and for every object.
- For doors only, infer `state` from the visible door sprite itself:
  - state = 0: the door is open; the door panel does not block the whole doorway, and the doorway looks passable
  - state = 1: the handle marker looks like a small circle
  - state = 2: the handle marker looks like a short horizontal bar ("-")
- Judge door state from the visible door sprite itself, not from task context or guessing.
- For all non-door types, use `state = null`.
- Do not output 0 for non-door types unless the tile/object is a door.
- Do not output strings such as "open" or "closed".

RULES:
- Judge only from the image.
- Do not assume any fixed agent position.
- front_cell must be reported as a separate field.
- front_cell.type is the OCCUPANCY type of the tile directly in front of the agent.
- Allowed front_cell.type values are: empty, wall, door, goal, ball, key, lava, box.
- Do NOT use floor, grey, color words, or free-form descriptions as front_cell.type.
- Do NOT include the front_cell tile in objects.
- Only report NON-BACKGROUND objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.
- Do not invent objects, colors, or states.

OUTPUT:
- Return raw JSON only.
- No markdown.
- No code blocks.
- No explanation.

Required schema:
{
  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},
  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},
  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]
}

PROMPT VERSION: Task B prompt v7.1 (door-state visual legend added)
"""

_TASK_B_PROMPT_V7_1C = """Return ONLY a single-line JSON object with exactly these keys: agent, front_cell, objects.

Do NOT use markdown fences. Do NOT pretty-print. Do NOT add extra keys. In agent.carrying, use only {type, color} or null — never add state there.

COORDINATES: 1-based on 7x7 view. Origin=bottom-left. x,y in [1,7]. Report [x,y] as [col,row].

COLORS: red, green, blue, yellow, purple, grey only. Lava=red, Goal=green.

AGENT DIRECTION: You MUST output agent.dir as an integer. 0=east, 1=south, 2=west, 3=north. Never omit or set to null.

STATE FIELD — DOOR VISUAL LEGEND (CRITICAL): Output numeric state for front_cell and every object.
  For doors: infer state from the visible sprite.
    - state=0: door is open; door panel does NOT block doorway; passage looks passable
    - state=1: handle marker looks like a small circle
    - state=2: handle marker looks like a short horizontal bar ("-")
  For all non-door types: state=null. Never output 0 for a non-door. Never output "open"/"closed" as a string.
  Judge state from the visible sprite, not from mission text, environment name, or task context.

RULES: Judge only from the image. Do not infer from mission text, environment name, or task context.
front_cell is the occupancy type directly in front of the agent.
Allowed front_cell.type: empty, wall, door, goal, ball, key, lava, box. Never include the front_cell tile in objects.
objects: visible non-background objects only — keys, balls, doors, goals, lava, boxes. Never report wall, floor, or empty as objects. If not clearly visible, do not report.

Required schema (single-line):
{"agent":{"pos":[x,y],"dir":d,"carrying":null_or_object},"front_cell":{"pos":[x,y],"type":"TYPE","state":null_or_door_state},"objects":[{"type":"TYPE","pos":[x,y],"color":"COLOR","state":null_or_door_state}]}

PROMPT VERSION: Task B prompt v7.1c (door visual legend + compact output format)
"""

_TASK_B_PROMPT_V7_2 = """Return ONLY a JSON object with exactly these keys: agent, front_cell, objects.

COORDINATE SYSTEM:
- Use 1-based image coordinates.
- The origin is the BOTTOM-LEFT corner of the 7x7 image.
- x increases to the RIGHT.
- y increases UPWARD.
- Report every position as [x=col, y=row].
- Valid range: x in [1,7], y in [1,7].

COLOR VOCABULARY:
- Only use: red, green, blue, yellow, purple, grey.
- Use canonical colors only.
- Lava must be red.
- Goal must be green.
- Do not invent color names.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use this exact encoding:
  - 0 = east / right
  - 1 = south / down
  - 2 = west / left
  - 3 = north / up
- dir must be one of: 0, 1, 2, 3.
- Do not omit dir.
- Do not output null for dir.

STATE FIELD - DOOR STATE HIERARCHY (CRITICAL):
- You MUST output a numeric `state` for front_cell and for every object.
- For doors only, use:
  - 0 = open
  - 1 = closed
  - 2 = locked
- For all non-door types, use `state = null`.
- Do not output 0 for non-door types unless the tile/object is a door.
- Do not output strings such as "open" or "closed".

DOOR DECISION RULES (IMPORTANT):
- First decide whether a visible tile/object is a DOOR.
- Only after deciding it is a door, classify its state.
- Do NOT require an exact circle/bar template in order to recognize a door.
- A tile can clearly be a door even if the small state marker is hard to read.

DOOR STATE HIERARCHY:
- state = 0 (open): the doorway looks passable; the door panel does not block the opening.
- state = 2 (locked): the tile is clearly a door, it is NOT open, and there is a CLEAR short horizontal-bar / lock-like marker.
- state = 1 (closed): the tile is clearly a door, it is NOT open, and the lock cue is absent, weak, or ambiguous.

ANTI-OVERCORRECTION RULE:
- If a tile is clearly a visible door but the exact closed-vs-locked cue is hard to read, STILL report it as a door.
- Do NOT drop a visible door from `objects`.
- Do NOT change a visible door into `empty` just because the state cue is ambiguous.
- Ambiguous non-open door state should default to `state = 1` rather than deleting the door.

EVIDENCE RULES:
- Judge from the visible image only.
- Use doorway passability as the strongest cue for `open`.
- Use small marker details as secondary cues for distinguishing `closed` vs `locked`.
- Do NOT infer door state from the mission, environment name, or task context.

RULES:
- Judge only from the image.
- Do not assume any fixed agent position.
- front_cell must be reported as a separate field.
- front_cell.type is the OCCUPANCY type of the tile directly in front of the agent.
- Allowed front_cell.type values are: empty, wall, door, goal, ball, key, lava, box.
- Do NOT use floor, grey, color words, or free-form descriptions as front_cell.type.
- Do NOT include the front_cell tile in objects.
- Only report NON-BACKGROUND objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.
- Do not invent objects, colors, or states.

OUTPUT:
- Return raw JSON only.
- No markdown.
- No code blocks.
- No explanation.

Required schema:
{
  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},
  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},
  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]
}

PROMPT VERSION: Task B prompt v7.2 (door-state hierarchy + no-drop rule)
"""

_TASK_B_PROMPT_V7_2B = """Return ONLY a single-line JSON object with exactly these keys: agent, front_cell, objects.

Do NOT use markdown fences.
Do NOT pretty-print.
Do NOT add extra keys anywhere.
In agent.carrying, use only {type, color} or null. Do NOT add state under carrying.

COORDINATES:
- Use 1-based image coordinates on the 7x7 view.
- Origin = bottom-left.
- x increases right, y increases up.
- Report every position as [x, y] with x,y in [1,7].

COLORS:
- Only use: red, green, blue, yellow, purple, grey.
- Lava must be red. Goal must be green.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use: 0=east, 1=south, 2=west, 3=north.

STATE FIELD:
- You MUST output numeric state for front_cell and every object.
- For doors only: 0=open, 1=closed, 2=locked.
- For all non-door types: state=null.

DOOR RULES:
- First decide whether a visible tile/object is a door.
- Do NOT require an exact marker template in order to recognize a door.
- A tile can clearly be a door even if the small state marker is hard to read.
- Use doorway passability as the strongest cue for state=0 (open).
- Use state=2 (locked) only when the tile is clearly a door, not open, and there is a clear short horizontal-bar / lock-like cue.
- If the tile is clearly a door, not open, and the closed-vs-locked cue is weak or ambiguous, use state=1 (closed).
- If a visible door is present, do NOT drop it just because the state cue is ambiguous.

RULES:
- Judge only from the image.
- Do not infer from mission text, environment name, or task context.
- front_cell is the occupancy type directly in front of the agent.
- Allowed front_cell.type values: empty, wall, door, goal, ball, key, lava, box.
- Do NOT include the front_cell tile in objects.
- objects should include only non-background visible objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.

Required schema:
{"agent":{"pos":[x,y],"dir":d,"carrying":null_or_object},"front_cell":{"pos":[x,y],"type":"TYPE","state":null_or_door_state},"objects":[{"type":"TYPE","pos":[x,y],"color":"COLOR","state":null_or_door_state}]}

PROMPT VERSION: Task B prompt v7.2b (compact door-state hierarchy + no-drop rule)
"""

_TASK_B_PROMPT_V7_2C = """Return ONLY a single-line JSON object with exactly these keys: agent, front_cell, objects.

Do NOT use markdown fences.
Do NOT pretty-print.
Do NOT add extra keys anywhere.
In agent.carrying, use only {type, color} or null. Do NOT add state under carrying.

COORDINATES:
- Use 1-based image coordinates on the 7x7 view.
- Origin = bottom-left.
- x increases right, y increases up.
- Report every position as [x, y] with x,y in [1,7].

COLORS:
- Only use: red, green, blue, yellow, purple, grey.
- Lava must be red. Goal must be green.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use: 0=east, 1=south, 2=west, 3=north.

STATE FIELD:
- You MUST output numeric state for front_cell and every object.
- For doors only: 0=open, 1=closed, 2=locked.
- For all non-door types: state=null.

DOOR RULES:
- First decide whether a visible tile/object is a door.
- Do NOT require an exact marker template in order to recognize a door.
- If a visible door is present, do NOT drop it just because the state cue is ambiguous.
- OPEN FIRST: if the doorway/opening looks visually passable or unobstructed, use state=0 even if the small marker is hard to read.
- Do NOT label a door as closed just because a marker is unclear.
- Use state=1 (closed) only when the tile is clearly a door and the opening looks visibly blocked by a closed door panel.
- Use state=2 (locked) only when the tile is clearly a door, not open, and there is a clear short horizontal-bar / lock-like cue.
- If the door is clearly non-open but the closed-vs-locked cue is ambiguous, use state=1.

RULES:
- Judge only from the image.
- Do not infer from mission text, environment name, or task context.
- front_cell is the occupancy type directly in front of the agent.
- Allowed front_cell.type values: empty, wall, door, goal, ball, key, lava, box.
- Do NOT include the front_cell tile in objects.
- objects should include only non-background visible objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.

Required schema:
{"agent":{"pos":[x,y],"dir":d,"carrying":null_or_object},"front_cell":{"pos":[x,y],"type":"TYPE","state":null_or_door_state},"objects":[{"type":"TYPE","pos":[x,y],"color":"COLOR","state":null_or_door_state}]}

PROMPT VERSION: Task B prompt v7.2c (compact open-first door hierarchy + no-drop rule)
"""

_TASK_B_PROMPT_V7_LAVA_GUARD = """Return ONLY a JSON object with exactly these keys: agent, front_cell, objects.

COORDINATE SYSTEM:
- Use 1-based image coordinates.
- The origin is the BOTTOM-LEFT corner of the 7x7 image.
- x increases to the RIGHT.
- y increases UPWARD.
- Report every position as [x=col, y=row].
- Valid range: x in [1,7], y in [1,7].

COLOR VOCABULARY:
- Only use: red, green, blue, yellow, purple, grey.
- Use canonical colors only.
- Lava must be red.
- Goal must be green.
- Do not invent color names.

AGENT DIRECTION:
- You MUST output agent.dir.
- Use this exact encoding:
  - 0 = east / right
  - 1 = south / down
  - 2 = west / left
  - 3 = north / up
- dir must be one of: 0, 1, 2, 3.
- Do not omit dir.
- Do not output null for dir.

STATE FIELD:
- You MUST output a numeric `state` for front_cell and for every object.
- For doors only, use:
  - 0 = open
  - 1 = closed
  - 2 = locked
- For all non-door types, use `state = null`.
- Do not output 0 for non-door types unless the tile/object is a door.
- Do not output strings such as "open" or "closed".

RULES:
- Judge only from the image.
- Do not assume any fixed agent position.
- front_cell must be reported as a separate field.
- front_cell.type is the OCCUPANCY type of the tile directly in front of the agent.
- Allowed front_cell.type values are: empty, wall, door, goal, ball, key, lava, box.
- Do NOT use floor, grey, color words, or free-form descriptions as front_cell.type.
- Do NOT include the front_cell tile in objects.
- Only report NON-BACKGROUND objects: keys, balls, doors, goals, lava, boxes.
- Do NOT report wall, floor, or empty as objects.
- If an object is not clearly visible, do not report it.
- Do not invent objects, colors, or states.
- Do not report lava unless clearly visible as contiguous red hazard tiles.
- Do not infer object classes from task priors; report only clearly visible objects.

OUTPUT:
- Return raw JSON only.
- No markdown.
- No code blocks.
- No explanation.

Required schema:
{
  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},
  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},
  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]
}

PROMPT VERSION: Task B prompt v7_lava_guard (anti-hallucination ablation variant)
"""
def generate_requests_from_exam(exam_dir: Path, out_requests_dir: Path, exam_task: str = "all",
                                image_path_mode: str = "absolute", task_c_exam_path: str = None,
                                task_d_exam_path: str = None,
                                task_e_exam_path: str = None,
                                taskb_prompt_variant: str = "v7") -> List[Path]:
    """
    Generate inference requests from exam directory.

    Args:
        exam_dir: Base exam directory
        out_requests_dir: Output directory for requests
        exam_task: Which tasks to process ("all", "A", "B", "C", "D", "E")
        image_path_mode: "absolute" or "relative"
        task_c_exam_path: Explicit path to legacy Task C exam JSONL
        task_d_exam_path: Explicit path to Task D exam JSONL (preferred over task_c_exam_path for Task D)
        taskb_prompt_variant: "v7" = original prompt; "v7_legend" = adds door-state visual legend;
            "v7_1c" = v7_legend core + compact single-line output; "v7_2" = door-state hierarchy + no-drop rule;
            "v7_2b" = compact door-state hierarchy + no-drop rule
    """
    # Select Task B prompt variant
    _VALID_TASKB_VARIANTS = {"v7", "v7_legend", "v7_1c", "v7_2", "v7_2b", "v7_2c", "v7_lava_guard"}
    if taskb_prompt_variant not in _VALID_TASKB_VARIANTS:
        raise ValueError(f"taskb_prompt_variant must be one of {_VALID_TASKB_VARIANTS}")
    _TASK_B_PROMPT_MAP = {
        "v7": _TASK_B_PROMPT_V7,
        "v7_legend": _TASK_B_PROMPT_V7_LEGEND,
        "v7_1c": _TASK_B_PROMPT_V7_1C,
        "v7_2": _TASK_B_PROMPT_V7_2,
        "v7_2b": _TASK_B_PROMPT_V7_2B,
        "v7_2c": _TASK_B_PROMPT_V7_2C,
        "v7_lava_guard": _TASK_B_PROMPT_V7_LAVA_GUARD,
    }
    _TASK_B_PROMPT = _TASK_B_PROMPT_MAP[taskb_prompt_variant]
    exam_dir = exam_dir.resolve()
    out_requests_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"A", "B", "C", "D", "E"} if exam_task == "all" else {exam_task}
    telemetry = {
        "exam_dir": str(exam_dir),
        "tasks": sorted(list(wanted)),
        "counts": {},
        "skips": defaultdict(int),
        "request_schema_version": REQUEST_SCHEMA_VERSION,
    }

    manifest_schema = EXAM_SCHEMA_VERSION
    manifest_path = exam_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_schema = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("schema_version", EXAM_SCHEMA_VERSION))
        except Exception:
            manifest_schema = EXAM_SCHEMA_VERSION

    def img_full_path(rel: str) -> str:
        # Do NOT use .resolve() — keep symlinks intact so request paths
        # reference the intended exam version (v7 → v4 via symlink is OK
        # but the request record should say "v7", not resolved "v4")
        p = exam_dir / rel  # preserved symlink
        return str(p) if image_path_mode == "absolute" else str(p.relative_to(exam_dir))

    generated_files = []

    def build_request_item(exam_row: Dict[str, Any], prompt_text: str) -> Optional[Dict[str, Any]]:
        uid = resolve_uid(exam_row)
        rel = exam_row.get("image")
        if not uid or not rel:
            return None
        schema_version = exam_row.get("schema_version") or manifest_schema
        exam = build_exam_meta(uid, row=exam_row, schema_version=schema_version)
        return {
            "uid": uid,
            "exam_id": exam.get("exam_id", uid),
            "schema_version": exam.get("schema_version", EXAM_SCHEMA_VERSION),
            "images": [img_full_path(rel)],
            "prompt": prompt_text,
            "exam": exam,
        }

    # Task A
    if "A" in wanted:
        reqs = []
        for it in load_jsonl(exam_dir / "task_a_exam.jsonl") if (exam_dir / "task_a_exam.jsonl").exists() else []:
            prompt = (
                it.get("prompt", "")
                + "\nAnswer with ONLY the correct letter (A/B/C/D). "
                + "Return exactly one uppercase letter and no other text."
            ).strip()
            req = build_request_item(it, prompt)
            if req:
                reqs.append(req)

        if reqs:
            out = out_requests_dir / "requests_A.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["A"] = len(reqs)

    # Task B (Structure JSON)
    if "B" in wanted:
        reqs = []
        # Use the selected prompt variant (set at function entry based on taskb_prompt_variant)
        b_prompt = _TASK_B_PROMPT
        for it in load_jsonl(exam_dir / "task_b_exam.jsonl") if (exam_dir / "task_b_exam.jsonl").exists() else []:
            req = build_request_item(it, b_prompt)
            if req:
                reqs.append(req)

        if reqs:
            out = out_requests_dir / "requests_B.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["B"] = len(reqs)

    # Task C (legacy storyboard Success/Fail) — only if exam_dir has task_c_exam.jsonl
    # NOTE: prefer task_d_exam.jsonl (Task D) over task_c_exam.jsonl (legacy C) when both exist.
    #       If neither explicit path nor exam_dir default exists, skip.
    if "C" in wanted:
        reqs = []
        if task_c_exam_path:
            task_c_path = Path(task_c_exam_path)
        else:
            task_c_path = exam_dir / "task_c_exam.jsonl"
        for it in load_jsonl(task_c_path) if task_c_path.exists() else []:
            prompt = it.get("prompt", "Determine whether the agent accomplished its goal. Answer with ONLY: Yes or No.").strip()
            req = build_request_item(it, prompt)
            if req:
                reqs.append(req)
        if reqs:
            out = out_requests_dir / "requests_C.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["C"] = len(reqs)

    # Task D (full-trajectory Success/Fail montage) — preferred Task C/D variant
    if "D" in wanted:
        reqs = []
        if task_d_exam_path:
            task_d_path = Path(task_d_exam_path)
        else:
            # Priority: task_d_exam.jsonl → task_c_exam.jsonl (legacy C, same semantics)
            d_path = exam_dir / "task_d_exam.jsonl"
            c_path = exam_dir / "task_c_exam.jsonl"
            task_d_path = d_path if d_path.exists() else c_path if c_path.exists() else None
        for it in load_jsonl(task_d_path) if task_d_path and task_d_path.exists() else []:
            prompt = it.get("prompt", "Determine whether the agent accomplished its goal. Answer with ONLY: Yes or No.").strip()
            req = build_request_item(it, prompt)
            if req:
                reqs.append(req)
        if reqs:
            out = out_requests_dir / "requests_D.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["D"] = len(reqs)

    # Task E (Long-Range Discrete State Tracking) — structured JSON answer
    if "E" in wanted:
        reqs = []
        if task_e_exam_path:
            task_e_path = Path(task_e_exam_path)
        else:
            task_e_path = None
            for fname in ("task_e_exam_v13.jsonl", "task_e_exam_v12.jsonl", "task_e_exam_v11.jsonl", "task_e_exam_v1.jsonl"):
                candidate = exam_dir / fname
                if candidate.exists():
                    task_e_path = candidate
                    break
        for it in load_jsonl(task_e_path) if task_e_path and task_e_path.exists() else []:
            # Task E exam rows have a pre-built "prompt" field
            prompt = it.get("prompt", "").strip()
            if not prompt:
                continue
            req = build_request_item(it, prompt)
            if req:
                reqs.append(req)
        if reqs:
            out = out_requests_dir / "requests_E.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["E"] = len(reqs)

    write_json(out_requests_dir / "requests_gen_manifest.json", telemetry)
    return generated_files

# -------------------------
# Provider Resolvers (CRITICAL FIX)
# -------------------------

# -------------------------
# Runners (OpenAI/Gemini/Local)
# -------------------------

# -------------------------
# Main Run Loop
# -------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Universal VLM Inference Gateway")

    # Core paths
    p.add_argument("--exam_dir", help="Path to exam directory (enables exam mode)")
    p.add_argument("--exam_task", default="all", choices=["all", "A", "B", "C", "D", "E"], help="Which exam tasks to process")
    p.add_argument("--task_c_exam", help="Explicit path to legacy Task C exam JSONL (overrides --exam_dir)")
    p.add_argument("--task_d_exam", help="Explicit path to Task D exam JSONL (overrides --exam_dir/task_d_exam.jsonl)")
    p.add_argument("--task_e_exam", help="Explicit path to Task E exam JSONL (overrides auto-discovery: v13 -> v12 -> v11 -> v1)")
    p.add_argument("--taskb_prompt_variant", default="v7", choices=["v7", "v7_legend", "v7_1c", "v7_2", "v7_2b", "v7_2c", "v7_lava_guard"],
                   help="Task B prompt variant: v7=original, v7_legend=adds door-state visual legend, v7_1c=v7_legend+compact output, v7_2=door-state hierarchy + no-drop rule, v7_2b=compact door-state hierarchy + no-drop rule, v7_2c=compact open-first door hierarchy + no-drop rule, v7_lava_guard=anti-hallucination ablation")
    p.add_argument("--requests", help="Path to requests JSONL file (alternative to exam_dir)")
    p.add_argument("--responses_dir", required=True, help="Directory to save responses")

    # Backend configuration
    p.add_argument("--backend", default="openai_compatible", choices=["openai_compatible", "gemini", "qwen2.5-vl", "internvl2.5", "llava-next-video", "llava", "llava-next"])
    p.add_argument("--provider", default="siliconflow", help="Provider for the backend (openai, siliconflow, zhizengzeng*, custom, etc.)")
    p.add_argument("--model", required=True, help="Model identifier")
    p.add_argument("--api_key", help="API key (can also use environment variables)")
    p.add_argument("--openai_request_api", default="chat", choices=["chat", "responses"],
                   help="OpenAI-compatible request API: chat=completions, responses=/v1/responses (recommended for GPT-5-class models)")

    # Generation parameters
    p.add_argument("--max_new_tokens", type=int, default=512, help="Default max tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")

    # Task-aware token budgets (capability preservation)
    p.add_argument("--max_tokens_A", type=int, default=512, help="Max tokens for Task A")
    p.add_argument("--max_tokens_B", type=int, default=2048, help="Max tokens for Task B (JSON)")
    p.add_argument("--max_tokens_C", type=int, default=256, help="Max tokens for Task C")

    # Image fidelity
    p.add_argument("--api_prefer_png", action="store_true", help="Prefer PNG for images (recommended)")
    p.add_argument("--api_image_max_side", type=int, default=2048, help="Max image side (<=0 disables resize)")
    p.add_argument("--api_jpeg_quality", type=int, default=85, help="JPEG quality when using JPEG")

    # Progress and logging
    p.add_argument("--progress_every", type=int, default=50, help="Progress update every N items")
    p.add_argument("--progress_interval_s", type=float, default=10.0, help="Heartbeat interval seconds")
    p.add_argument("--progress_show_uid", action="store_true", help="Show current UID in progress")
    p.add_argument("--progress_total", action="store_true", help="Count total items for ETA")

    # Exam mode specific
    p.add_argument("--build_exam_requests_only", action="store_true", help="Only generate requests, don't run inference")

    # Other options
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--shard_id", type=int, default=0, help="Current shard ID")
    p.add_argument("--resume", action="store_true", help="Resume from existing responses")
    p.add_argument("--tmp_dir", default="./tmp", help="Temporary directory")
    p.add_argument("--dtype", default="auto", help="Data type for local models")
    p.add_argument("--device_map", default="auto", help="Device mapping for local models")
    p.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for acceleration (requires flash-attn)")
    p.add_argument("--seed", type=int, help="Random seed")

    # API settings
    p.add_argument("--api_timeout", type=float, default=60.0, help="API timeout seconds")
    p.add_argument("--api_max_retries", type=int, default=3, help="Max API retries")
    p.add_argument("--api_backoff_base", type=float, default=2.0, help="Retry backoff base")
    p.add_argument("--min_interval_ms", type=int, default=0, help="Min interval between requests")

    # Advanced options
    p.add_argument("--force_fuse", choices=["h", "v"], help="Force image fusion mode")
    p.add_argument("--base_url", help="Custom base URL")
    p.add_argument("--model_tag", help="Custom model tag for response directory")
    p.add_argument("--extra_body", help="Path to JSON file with extra request body")

    args = p.parse_args()
    run(args)
