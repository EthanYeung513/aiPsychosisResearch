"""Minimal OpenAI-compatible chat server: a Hugging Face model, optionally with a LoRA adapter.

All conditions in a comparison must be served the same way (same engine, precision and decoding).
Base and adapter runs therefore both go through this server, with or without --adapter.

  python scripts/serve_hf.py --model Qwen/Qwen3.5-0.8B [--adapter adapters/smoke/adapter] --port 8091

Implements POST /v1/chat/completions (messages, max_tokens, temperature, seed) and GET /health.
finish_reason is "length" when max_tokens is reached, otherwise "stop". Single-threaded on purpose.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--name", default=None, help="model name reported in responses")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()
    name = args.name or (f"{args.model}+{args.adapter}" if args.adapter else args.model)

    def generate(body: dict) -> dict:
        messages = body["messages"]
        max_new = int(body.get("max_tokens") or 512)
        temperature = float(body.get("temperature", 0.7))
        if body.get("seed") is not None:
            torch.manual_seed(int(body["seed"]))
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new = out[0, inputs["input_ids"].shape[1]:]
        finish = "length" if len(new) >= max_new and new[-1].item() != tok.eos_token_id else "stop"
        content = tok.decode(new, skip_special_tokens=True).strip()
        return {
            "object": "chat.completion",
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(len(new))},
        }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": name, "device": device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, generate(body))
            except Exception as e:  # report, never crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *a):
            pass

    print(f"serving {name} on http://127.0.0.1:{args.port} ({device})", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
