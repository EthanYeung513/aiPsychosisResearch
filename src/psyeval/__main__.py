"""CLI: python -m psyeval run ... | python -m psyeval summarize RUN_DIR

Example (local OpenAI-compatible server, e.g. llama.cpp `llama-server` or Ollama, judge disabled):
  python -m psyeval run --target-base-url http://localhost:8080/v1 --target-model qwen \
      --no-judge --cases conduit_explicit --out runs/smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

from .cases import CASES_SHA256, PSYCHOSIS, SUITES, UPSTREAM_COMMIT, load_suite
from .cases import REPO_ROOT
from .chat import OpenAICompatibleClient
from .rubrics import JUDGE_STYLES, metrics_for, start_turn_for
from .runner import run_cases
from .summary import compare_runs, summarize_run


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_suite(args.suite, args.split)
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c.id for c in cases}
        if unknown:
            sys.exit(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]
    target = OpenAICompatibleClient(
        base_url=args.target_base_url,
        model=args.target_model,
        api_key_env=args.target_api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        timeout_s=args.timeout,
        extra_body=json.loads(args.target_extra_body) if args.target_extra_body else {},
    )
    judge = None
    if not args.no_judge:
        if not (args.judge_base_url and args.judge_model):
            sys.exit("--judge-base-url and --judge-model are required unless --no-judge")
        judge = OpenAICompatibleClient(
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key_env=args.judge_api_key_env,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            timeout_s=args.judge_timeout,
        )
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8") if args.system_prompt_file else None
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "condition_label": args.label,
        "target": target.describe(),
        "judge": judge.describe() if judge else None,
        "system_prompt": system_prompt,
        "suite": args.suite,
        "cases": [c.id for c in cases],
        "case_turns": {c.id: len(c.prompts) for c in cases},
        "case_metric_start": {
            c.id: {n: start_turn_for(m, c.help_from_turn) for n, m in metrics_for(c.suite).items()} for c in cases
        },
        "split": args.split if args.suite != PSYCHOSIS else None,
        "upstream_commit": UPSTREAM_COMMIT,
        "cases_sha256": CASES_SHA256,
        "repo_commit": _git_head(),
        "python": platform.python_version(),
        "extra": json.loads(args.manifest_extra) if args.manifest_extra else {},
    }
    manifest["workers"] = args.workers
    manifest["judge_style"] = args.judge_style
    path = run_cases(cases, target, judge, Path(args.out), manifest, system_prompt, workers=args.workers, judge_style=args.judge_style)
    print(f"wrote {path}")
    if judge:
        print(json.dumps(summarize_run(Path(args.out)), indent=2))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .runner import rejudge_record
    from .summary import load_records

    src = Path(args.run_dir)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    cases = {c.id: c for c in load_suite(manifest["suite"], manifest.get("split") or "dev")}
    judge = OpenAICompatibleClient(
        base_url=args.judge_base_url, model=args.judge_model, api_key_env=args.judge_api_key_env,
        temperature=args.judge_temperature, max_tokens=args.judge_max_tokens, timeout_s=args.judge_timeout,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    new_manifest = {
        **manifest,
        "judge_style": args.judge_style,
        "judge_context_turns": args.judge_context_turns,
        "judge_context_mode": args.judge_context_mode,
        "judge": judge.describe(),
        "rejudged_from": str(src),
        "rejudged_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": _git_head(),
    }
    (out / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    records = load_records(src / "conversations.jsonl")
    from concurrent.futures import ThreadPoolExecutor

    with (out / "conversations.jsonl").open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(rejudge_record, rec, cases[rec["case_id"]], judge, args.judge_style,
                        args.judge_context_turns, args.judge_context_mode)
            for rec in records
        ]
        for n, fut in enumerate(futures, 1):  # in source order
            new = fut.result()
            print(f"[{n}/{len(records)}] {new['case_id']}")
            f.write(json.dumps(new, ensure_ascii=False) + chr(10))
            f.flush()
    print(json.dumps(summarize_run(out), indent=2))
    return 0


def _excl(args: argparse.Namespace) -> set[str]:
    return {f for f in (args.exclude_families or "").split(",") if f}


def cmd_summarize(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_run(Path(args.run_dir), _excl(args)), indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(json.dumps(compare_runs(Path(args.run_a), Path(args.run_b), _excl(args)), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="psyeval")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run cases against a target model")
    r.add_argument("--suite", choices=SUITES, default=PSYCHOSIS)
    r.add_argument("--split", choices=["dev", "test"], default="dev", help="for grounded suites")
    r.add_argument("--target-base-url", required=True)
    r.add_argument("--target-model", required=True)
    r.add_argument("--target-api-key-env", help="env var holding the target API key, if any")
    r.add_argument("--target-extra-body", help='JSON merged into each target request, e.g. '
                   '\'{"chat_template_kwargs": {"enable_thinking": false}}\' for llama.cpp or vLLM (recorded in the manifest)')
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--max-tokens", type=int, default=768)
    r.add_argument("--seed", type=int)
    r.add_argument("--timeout", type=float, default=120.0, help="seconds per target request (raise for slow local hardware)")
    r.add_argument("--judge-timeout", type=float, default=120.0)
    r.add_argument("--system-prompt-file")
    r.add_argument("--judge-base-url")
    r.add_argument("--judge-model")
    r.add_argument("--judge-api-key-env")
    r.add_argument("--judge-temperature", type=float, default=0.0)
    r.add_argument("--judge-max-tokens", type=int, default=16)
    r.add_argument("--no-judge", action="store_true", help="collect transcripts only")
    r.add_argument("--judge-style", choices=JUDGE_STYLES, default="bare", help="project judges only; 'reasoned' = revision R1")
    r.add_argument("--cases", help="comma-separated case ids (default: every case in the suite)")
    r.add_argument("--label", default="", help="condition label, e.g. baseline / prompt / generic / targeted")
    r.add_argument("--out", required=True, help="new run directory (must not exist)")
    r.add_argument("--workers", type=int, default=1, help="conversations to run concurrently (for batching servers)")
    r.add_argument("--manifest-extra", help="JSON object of provenance to record (model file hashes, server build, ...)")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="summarize an existing run directory")
    s.add_argument("run_dir")
    s.add_argument("--exclude-families", help="comma-separated, e.g. conduit (dev-exposed) for the primary result")
    s.set_defaults(func=cmd_summarize)

    j = sub.add_parser("judge", help="score an existing run's transcripts with a judge (no generation)")
    j.add_argument("run_dir")
    j.add_argument("--judge-base-url", required=True)
    j.add_argument("--judge-model", required=True)
    j.add_argument("--judge-api-key-env")
    j.add_argument("--judge-temperature", type=float, default=0.0)
    j.add_argument("--judge-max-tokens", type=int, default=16)
    j.add_argument("--judge-timeout", type=float, default=120.0)
    j.add_argument("--out", required=True, help="new run directory (must not exist)")
    j.add_argument("--workers", type=int, default=1, help="transcripts to judge concurrently")
    j.add_argument("--judge-style", choices=JUDGE_STYLES, default="bare", help="project judges only; 'reasoned' = revision R1")
    j.add_argument("--judge-context-turns", type=int, help="judge sees only the last N turns (deviation; tests H-len)")
    j.add_argument("--judge-context-mode", choices=["full", "turn-aware"], default="full",
                   help="grounded suites: 'turn-aware' = case facts as established by each turn (candidate R3)")
    j.set_defaults(func=cmd_judge)

    c = sub.add_parser("compare", help="paired per-family difference, RUN_B minus RUN_A")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--exclude-families", help="comma-separated, e.g. conduit")
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
