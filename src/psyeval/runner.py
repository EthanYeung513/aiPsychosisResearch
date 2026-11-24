"""Run Psychosis-bench conversations against a target model and score each turn with a judge.

Validity rules (deliberately stricter than upstream):
- A target-model failure aborts that conversation. The error text is never scored or fed back as a reply.
- A judge failure, truncated judge output or unparseable judge output is an invalid score (score=None),
  stored with the raw text and reason.
- A target reply cut off at the token limit is kept (it is what a user would see) but flagged truncated.
- Every user message, reply, judge prompt version and raw judge output is kept for auditing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .cases import Case
from .chat import ChatClient, ChatError, Message
from .rubrics import build_prompt, metrics_for, parse_score, start_turn_for


def judge_turn(
    judge: ChatClient,
    history: list[tuple[str, str]],
    case: Case,
    turn_number: int,
    style: str = "bare",
    context_turns: int | None = None,
    context_mode: str = "full",
) -> dict:
    """context_turns: if set, the judge sees only the last N turns. This deviates from the upstream
    'review the entire conversation' instruction and is recorded in the manifest (hypothesis H-len).
    context_mode: 'full' (default) or 'turn-aware' (candidate R3: case facts as established by this
    turn; this affects only the grounded suites, whose rubrics use {context})."""
    if context_turns:
        history = history[-context_turns:]
    case_context = case.context_at(turn_number, context_mode)
    scores: dict[str, dict] = {}
    for name, metric in metrics_for(case.suite).items():
        if turn_number < start_turn_for(metric, case.help_from_turn):
            continue
        mstyle = "bare" if metric.upstream else style
        prompt = build_prompt(metric, history, case.harm_type, case_context, mstyle)
        try:
            reply = judge.chat([{"role": "user", "content": prompt}])
        except ChatError as e:
            scores[name] = {"score": None, "raw": None, "error": str(e)}
            continue
        if reply.truncated:
            scores[name] = {"score": None, "raw": reply.content, "error": "judge output truncated"}
            continue
        score = parse_score(reply.content, metric.max_score, mstyle)
        entry: dict = {"score": score, "raw": reply.content}
        if score is None:
            entry["error"] = "unparseable judge output"
        scores[name] = entry
    return scores


def run_conversation(
    case: Case,
    target: ChatClient,
    judge: ChatClient | None,
    system_prompt: str | None = None,
    judge_style: str = "bare",
) -> dict:
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    history: list[tuple[str, str]] = []
    turns: list[dict] = []
    record: dict = {
        "case_id": case.id,
        "suite": case.suite,
        "n_prompts": len(case.prompts),
        "metric_start": {n: start_turn_for(m, case.help_from_turn) for n, m in metrics_for(case.suite).items()},
        "family": case.family,
        "condition": case.condition,
        "theme": case.theme,
        "harm_type": case.harm_type,
        "status": "ok",
        "error": None,
        "turns": turns,
    }
    for i, user_message in enumerate(case.prompts):
        turn_number = i + 1
        messages.append({"role": "user", "content": user_message})
        started = time.perf_counter()
        try:
            reply = target.chat(list(messages))
        except ChatError as e:
            record["status"] = "target_error"
            record["error"] = f"turn {turn_number}: {e}"
            break
        latency = time.perf_counter() - started
        messages.append({"role": "assistant", "content": reply.content})
        history.append((user_message, reply.content))
        turn: dict = {
            "turn": turn_number,
            "user": user_message,
            "assistant": reply.content,
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "latency_s": round(latency, 3),
        }
        if judge is not None:
            turn["scores"] = judge_turn(judge, history, case, turn_number, judge_style)
        turns.append(turn)
    return record


def rejudge_record(
    record: dict,
    case: Case,
    judge: ChatClient,
    judge_style: str = "bare",
    context_turns: int | None = None,
    context_mode: str = "full",
) -> dict:
    """Score an existing transcript with a (possibly different) judge. Generation is untouched."""
    if record["case_id"] != case.id:
        raise ValueError(f"record {record['case_id']} does not match case {case.id}")
    for i, t in enumerate(case.prompts[: len(record["turns"])]):
        if record["turns"][i]["user"] != t:
            raise ValueError(f"{case.id} turn {i + 1}: transcript does not match the case script")
    new = {k: v for k, v in record.items() if k != "turns"}
    new["metric_start"] = {n: start_turn_for(m, case.help_from_turn) for n, m in metrics_for(case.suite).items()}
    history: list[tuple[str, str]] = []
    turns = []
    for t in record["turns"]:
        history.append((t["user"], t["assistant"]))
        nt = {k: v for k, v in t.items() if k != "scores"}
        nt["scores"] = judge_turn(judge, history, case, t["turn"], judge_style, context_turns, context_mode)
        turns.append(nt)
    new["turns"] = turns
    return new


def run_cases(
    cases: list[Case],
    target: ChatClient,
    judge: ChatClient | None,
    out_dir: Path,
    manifest: dict,
    system_prompt: str | None = None,
    log=print,
    workers: int = 1,
    judge_style: str = "bare",
) -> Path:
    """Run every case and write records in case order.

    With workers > 1, conversations run concurrently, so a batching server (vLLM, llama.cpp with
    -np N) can serve several at once. Turns within a conversation stay sequential. Records are written
    as soon as every earlier case has finished, so a crash loses only unfinished cases, and those
    show up as `missing` in the summary.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path = out_dir / "conversations.jsonl"

    def one(case: Case) -> dict:
        return run_conversation(case, target, judge, system_prompt, judge_style)

    with results_path.open("w", encoding="utf-8") as f:
        if workers <= 1:
            results = (one(c) for c in cases)
            for n, (case, record) in enumerate(zip(cases, results), 1):
                _write(f, record, n, len(cases), log)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one, c) for c in cases]
                for n, fut in enumerate(futures, 1):  # in case order
                    _write(f, fut.result(), n, len(cases), log)
    return results_path


def _write(f, record: dict, n: int, total: int, log) -> None:
    log(f"[{n}/{total}] {record['case_id']} {record['status']}")
    f.write(json.dumps(record, ensure_ascii=False) + chr(10))
    f.flush()
    if record["status"] != "ok":
        log(f"  ABORTED: {record['error']}")
