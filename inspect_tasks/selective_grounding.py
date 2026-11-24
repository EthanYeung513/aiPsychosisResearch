"""Inspect (UK AISI) port of the psyeval evaluation.

It uses the same cases, judge rubric prompts and strict score parsing as `src/psyeval`, so the two
implementations score identically. Inspect supplies the run infrastructure: logging, the log viewer,
retries, concurrency and model providers.

  inspect eval inspect_tasks/selective_grounding.py -T suite=psychosis-bench \
      --model openai-api/local/qwen3.5-4b --model-role grader=openai-api/judge/gemma-3-12b-it
  (with LOCAL_BASE_URL=http://127.0.0.1:8081/v1, JUDGE_BASE_URL=http://127.0.0.1:8082/v1, and dummy *_API_KEY)

Behaviour aligned with psyeval:
- A target-model error makes the sample error out. Inspect reports it and does not score it.
- Replies cut off at max_tokens are counted in the score metadata (`truncated_turns`).
- A judge failure or unparseable judge output is invalid. A metric with no valid turn scores is NaN,
  never 0, and every score's metadata includes valid/expected counts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model  # noqa: E402
from inspect_ai.scorer import Score, Target, mean, scorer  # noqa: E402
from inspect_ai.solver import Generate, TaskState, solver  # noqa: E402

from psyeval.cases import load_suite  # noqa: E402
from psyeval.rubrics import build_prompt, metrics_for, parse_score, start_turn_for  # noqa: E402


def _dataset(suite: str, split: str, cases: str | None) -> MemoryDataset:
    wanted = set(cases.split(",")) if cases else None
    samples = []
    for c in load_suite(suite, split):
        if wanted and c.id not in wanted:
            continue
        samples.append(
            Sample(
                input=c.prompts[0],
                id=c.id,
                metadata={
                    "suite": suite,
                    "family": c.family,
                    "prompts": list(c.prompts),
                    "harm_type": c.harm_type,
                    "context": c.context,
                    "help_from_turn": c.help_from_turn,
                },
            )
        )
    return MemoryDataset(samples)


@solver
def scripted_conversation(system_prompt: str | None = None):
    """Play the fixed user turns, generating one assistant reply after each."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = [ChatMessageSystem(content=system_prompt)] if system_prompt else []
        truncated = 0
        for user_turn in state.metadata["prompts"]:
            state.messages.append(ChatMessageUser(content=user_turn))
            state = await generate(state)
            if state.output.stop_reason == "max_tokens":
                truncated += 1
        state.metadata["truncated_turns"] = truncated
        return state

    return solve


def _turns(state: TaskState) -> list[tuple[str, str]]:
    msgs = [m for m in state.messages if m.role in ("user", "assistant")]
    return [(msgs[i].text, msgs[i + 1].text) for i in range(0, len(msgs) - 1, 2) if msgs[i].role == "user"]


def judge_scorer(suite: str):
    metric_names = list(metrics_for(suite))

    @scorer(metrics={name: [mean()] for name in metric_names})
    def turn_judges():
        async def score(state: TaskState, target: Target) -> Score:
            grader = get_model(role="grader", required=True)
            meta = state.metadata
            turns = _turns(state)
            values, counts, raw = {}, {}, {}
            for name, metric in metrics_for(suite).items():
                start = start_turn_for(metric, meta.get("help_from_turn"))
                expected = max(0, len(meta["prompts"]) - start + 1)
                got = []
                for t in range(start, len(turns) + 1):
                    prompt = build_prompt(metric, turns[:t], meta["harm_type"], meta["context"])
                    out = await grader.generate(prompt)
                    s = None if out.stop_reason == "max_tokens" else parse_score(out.completion, metric.max_score)
                    raw.setdefault(name, []).append(out.completion[:40])
                    if s is not None:
                        got.append(s)
                values[name] = sum(got) / len(got) if got else math.nan
                counts[name] = {"valid": len(got), "expected": expected}
            return Score(
                value=values,
                explanation="; ".join(f"{k}: {v['valid']}/{v['expected']} valid" for k, v in counts.items()),
                metadata={"counts": counts, "truncated_turns": meta.get("truncated_turns"), "raw_judge": raw},
            )

        return score

    return turn_judges()


@task
def selective_grounding(
    suite: str = "psychosis-bench",
    split: str = "dev",
    cases: str | None = None,
    system_prompt_file: str | None = None,
) -> Task:
    system_prompt = Path(system_prompt_file).read_text(encoding="utf-8") if system_prompt_file else None
    return Task(
        dataset=_dataset(suite, split, cases),
        solver=scripted_conversation(system_prompt),
        scorer=judge_scorer(suite),
    )
