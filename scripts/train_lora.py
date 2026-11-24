"""LoRA fine-tuning for the selective-grounding conditions (A, B, A', A'').

Plain transformers + PEFT training loop, so every step is visible and the budget is explicit.

- Loss is computed only on assistant tokens (user and system tokens are masked with -100).
- The budget is reported in training tokens and optimiser updates, and written to the run's
  manifest.json together with data hashes, so conditions can be checked for matched budgets.
- Device: CUDA if available, else Intel XPU, else CPU.

Usage:
  python scripts/train_lora.py --model Qwen/Qwen3.5-0.8B --data a.jsonl [b.jsonl ...] \
      --out adapters/smoke --epochs 1 --max-steps 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_examples(paths: list[str]) -> tuple[list[dict], dict]:
    examples, hashes = [], {}
    for p in paths:
        raw = Path(p).read_bytes()
        hashes[p] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            if line.strip():
                examples.append(json.loads(line))
    return examples, hashes


def tokenize_example(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Build input_ids and labels with loss on every assistant reply (content + end-of-turn token).

    The full conversation is rendered once with the chat template. Each assistant reply is then
    located by character offsets. Rendering turn by turn is NOT safe: some templates (e.g. Qwen3.5)
    render earlier assistant turns differently from the last one, which silently drops targets.
    Returns None if the example is too long or a reply cannot be located exactly.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    cursor = 0
    for m in messages:
        content = m["content"].strip()
        pos = text.find(content, cursor)
        if pos < 0:
            return None
        cursor = pos + len(content)
        if m["role"] != "assistant":
            continue
        span_end = cursor
        hit_last = None
        for k, (a, b) in enumerate(offsets):
            if b > pos and a < span_end:
                labels[k] = ids[k]
                hit_last = k
        if hit_last is None:
            return None
        if hit_last + 1 < len(ids):  # end-of-turn token, so the model learns to stop
            labels[hit_last + 1] = ids[hit_last + 1]
    return {"input_ids": ids, "labels": labels}


def check_targets(tokenizer, encoded: list[dict], examples: list[dict]) -> int:
    """Count examples whose trained tokens do not contain every assistant reply (should be 0)."""
    bad = 0
    for enc, ex in zip(encoded, examples):
        trained = tokenizer.decode([i for i, l in zip(enc["input_ids"], enc["labels"]) if l != -100])
        replies = [m["content"].strip() for m in ex["messages"] if m["role"] == "assistant"]
        if not all(r in trained for r in replies):
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", nargs="+", required=True, help="JSONL files with a 'messages' list per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser updates (0 = no cap)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="base-weight dtype; use float16 on GPUs without bf16 (e.g. Colab T4)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N examples (smoke tests)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(torch)
    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples, data_hashes = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]
    encoded, kept, skipped = [], [], 0
    for ex in examples:
        enc = tokenize_example(tokenizer, ex["messages"], args.max_len)
        if enc is None:
            skipped += 1
        else:
            encoded.append(enc)
            kept.append(ex)
    if not encoded:
        raise SystemExit("no usable examples")
    bad_targets = check_targets(tokenizer, encoded, kept)
    if bad_targets:
        raise SystemExit(f"{bad_targets} examples do not train on every assistant reply; refusing to train")
    target_tokens = sum(sum(1 for l in e["labels"] if l != -100) for e in encoded)
    total_tokens = sum(len(e["input_ids"]) for e in encoded)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        task_type="CAUSAL_LM",
    )
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, lora)
    # Keep the trainable LoRA weights in float32 even when the frozen base is fp16/bf16: this is
    # numerically stable without a loss scaler.
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    steps_per_epoch = math.ceil(len(encoded) / (args.batch_size * args.grad_accum))
    planned_updates = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        planned_updates = min(planned_updates, args.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, planned_updates)))

    log: list[dict] = []
    updates = micro = seen_target_tokens = 0
    started = time.time()
    model.train()
    order: list[int] = []
    while updates < planned_updates:
        if not order:
            order = list(range(len(encoded)))
            random.shuffle(order)
        idx = order.pop()
        e = encoded[idx]
        input_ids = torch.tensor([e["input_ids"]], device=device)
        labels = torch.tensor([e["labels"]], device=device)
        loss = model(input_ids=input_ids, labels=labels).loss / args.grad_accum
        loss.backward()
        seen_target_tokens += sum(1 for l in e["labels"] if l != -100)
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            entry = {"update": updates, "loss": round(loss.item() * args.grad_accum, 4), "elapsed_s": round(time.time() - started, 1)}
            log.append(entry)
            print(json.dumps(entry), flush=True)

    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    peak_mem = None
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 2**30
    elif device == "xpu":
        peak_mem = torch.xpu.max_memory_allocated() / 2**30
    manifest = {
        "model": args.model,
        "data": data_hashes,
        "n_examples": len(encoded),
        "n_skipped": skipped,
        "target_check_failures": bad_targets,
        "dataset_total_tokens": total_tokens,
        "dataset_target_tokens": target_tokens,
        "optimizer_updates": updates,
        "micro_steps": micro,
        "target_tokens_seen": seen_target_tokens,
        "epochs_equivalent": round(micro / len(encoded), 3),
        "trainable_params": trainable,
        "total_params": total_params,
        "device": device,
        "dtype": args.dtype,
        "peak_memory_gib": round(peak_mem, 2) if peak_mem is not None else None,
        "wall_time_s": round(time.time() - started, 1),
        "args": vars(args),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "loss_log": log,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "loss_log"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
