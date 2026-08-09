"""Resident scorer for the auto approve/deny models.

Speaks newline-delimited JSON on stdin/stdout so the pi extension pays the model load
cost once per session instead of once per tool call.

Protocol
    -> {"id": 1, "text": "### PROPOSED TOOL CALL\n..."}
    <- {"id": 1, "p_deny": 0.0123, "ms": 9.4}
    -> {"id": 2, "op": "info"}
    <- {"id": 2, "device": "cuda", "dtype": "bfloat16", "model": "...", "attn": "flash_attention_2"}

Every response carries the request id back. Errors come back as {"id": N, "error": "..."}
rather than killing the process -- one bad request must not take the gate down mid-session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg):
    # stderr is the extension's log channel; stdout is reserved for protocol frames.
    sys.stderr.write(f"[auto-pi] {msg}\n")
    sys.stderr.flush()


def pick_device(requested):
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    import torch

    # bf16 is verified bit-identical to fp32 on this model's benchmark, and carries no
    # overflow risk. MPS gets fp16 because bf16 support there is still uneven.
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def pick_attn(device):
    """flash-attn is required to reach the full 64k window; sdpa is the safe fallback.

    The non-flash path materialises a dense (B,1,L,L) sliding-window mask, so very long
    inputs get expensive. The extension caps input length accordingly when this returns sdpa.
    """
    if device != "cuda":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=65536)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # dependencies missing -> tell the extension, do not crash silently
        emit({"id": 0, "ready": False, "error": f"missing python deps: {exc}"})
        return 1

    device = pick_device(args.device)
    dtype = pick_dtype(device)
    attn = pick_attn(device)

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, dtype=dtype, attn_implementation=attn
        )
        model = model.to(device).eval()
    except Exception as exc:
        emit({"id": 0, "ready": False, "error": f"failed to load {args.model}: {exc}"})
        return 1

    # sdpa cannot afford the full window; clamp so a huge transcript degrades to a
    # truncated judgement instead of an out-of-memory kill.
    max_len = args.max_len if attn == "flash_attention_2" else min(args.max_len, 8192)

    # id2label may be absent or reordered on a fine-tune; resolve "deny" by name, and only
    # fall back to index 1 when the label map is unusable.
    deny_index = 1
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).strip().lower() == "deny":
            deny_index = int(idx)
            break

    emit({
        "id": 0,
        "ready": True,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "attn": attn,
        "max_len": max_len,
        "model": args.model,
        "deny_index": deny_index,
    })
    log(f"ready on {device} ({str(dtype).replace('torch.', '')}, {attn}, max_len={max_len})")

    # Warm the kernels so the first real tool call is not the one that pays for autotuning.
    try:
        with torch.no_grad():
            warm = tok("### PROPOSED TOOL CALL\ntool: bash\nargs: ls", return_tensors="pt",
                       truncation=True, max_length=max_len)
            model(**{k: v.to(device) for k, v in warm.items()})
    except Exception as exc:
        log(f"warmup skipped: {exc}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id", 0)

        if req.get("op") == "shutdown":
            emit({"id": rid, "ok": True})
            return 0
        if req.get("op") == "info":
            emit({"id": rid, "device": device, "dtype": str(dtype).replace("torch.", ""),
                  "attn": attn, "max_len": max_len, "model": args.model})
            continue

        text = req.get("text")
        if not isinstance(text, str) or not text:
            emit({"id": rid, "error": "missing text"})
            continue

        try:
            t0 = time.time()
            with torch.no_grad():
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.float()
                probs = logits.softmax(-1)[0]
                p_deny = float(probs[deny_index].item())
            emit({
                "id": rid,
                "p_deny": p_deny,
                "ms": round((time.time() - t0) * 1000, 2),
                "tokens": int(enc["input_ids"].shape[-1]),
            })
        except Exception as exc:
            emit({"id": rid, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
