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


def release_memory(device):
    """Return the caching allocator's block back to the OS between scorings."""
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


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

    The non-flash path materialises dense (B,1,L,L) attention, so memory grows with the
    SQUARE of input length. See safe_window() for why that matters a great deal.
    """
    if device != "cuda":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


# Why the non-flash path needs a window cap at all.
#
# With flash-attention the full 64k window is fine -- that is how this model was benchmarked
# (97.02% on 16k-64k items). flash-attn is CUDA-only, so on Apple/CPU transformers falls back
# to sdpa, which materialises a dense (B, heads, L, L) score matrix. Memory then grows with
# the SQUARE of input length. Measured on an M4 Max with auto-1b (fp16, sdpa), peak allocator
# memory for ONE forward pass:
#
#     1024 tok ->  2.4 GB      3072 tok ->  7.0 GB
#     2048 tok ->  4.5 GB      4096 tok -> 12.2 GB      8192 tok -> ~40 GB (extrapolated)
#
# Fitting those points: overhead ~2.0 GB for weights, plus ~6e-7 GB per token squared. An
# earlier release hardcoded 8192 here and exhausted a 64 GB machine badly enough to crash it.
QUADRATIC_GB_PER_TOKEN_SQ = 6.0e-7
MODEL_OVERHEAD_GB = 2.2

# Never exceed this share of total system RAM for one forward pass, so the gate cannot
# starve the editor, the browser, or the agent it is supposed to be protecting.
RAM_FRACTION = 0.125

NONFLASH_FLOOR = 1024
NONFLASH_CEILING = 8192


def total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 8.0


def safe_window(ram_gb):
    """Largest sdpa window whose measured peak stays inside the RAM budget."""
    budget = max(0.5, ram_gb * RAM_FRACTION - MODEL_OVERHEAD_GB)
    tokens = int((budget / QUADRATIC_GB_PER_TOKEN_SQ) ** 0.5)
    tokens = (tokens // 512) * 512  # round down to a tidy multiple
    return max(NONFLASH_FLOOR, min(NONFLASH_CEILING, tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--allow-large-window", action="store_true",
                    help=f"permit up to {NONFLASH_CEILING} tokens without flash-attn "
                         f"(quadratic memory — can exhaust RAM)")
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

    # With flash-attn the full window is affordable. Without it, size the window to what this
    # machine can actually hold rather than a hardcoded guess.
    ram = total_ram_gb()
    if attn == "flash_attention_2":
        max_len = args.max_len
    else:
        auto = safe_window(ram)
        max_len = min(args.max_len, auto)
        if args.allow_large_window:
            max_len = min(args.max_len, NONFLASH_CEILING)
            if max_len > auto:
                est = MODEL_OVERHEAD_GB + QUADRATIC_GB_PER_TOKEN_SQ * max_len ** 2
                log(f"WARNING: window {max_len} without flash-attn on {ram:.0f}GB RAM. "
                    f"Estimated peak ~{est:.0f}GB per call. Safe value here is {auto}.")
        log(f"sdpa window {max_len} tokens (~{MODEL_OVERHEAD_GB + QUADRATIC_GB_PER_TOKEN_SQ * max_len ** 2:.1f}GB peak) "
            f"on {ram:.0f}GB RAM")

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
    release_memory(device)

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
                n_tok = int(enc["input_ids"].shape[-1])
            emit({
                "id": rid,
                "p_deny": p_deny,
                "ms": round((time.time() - t0) * 1000, 2),
                "tokens": n_tok,
            })
        except Exception as exc:
            emit({"id": rid, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            # Attention buffers for a single pass are large (GBs at multi-k tokens). Hand them
            # back rather than letting the caching allocator sit on them between tool calls.
            # Rebind rather than `del` — the names may be unbound if tokenising itself threw.
            enc = logits = probs = None
            release_memory(device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
