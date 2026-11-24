"""Plumbing test for the Inspect port with Inspect's mock model (skipped if inspect_ai is not installed)."""

import math
import sys
from pathlib import Path

import pytest

inspect_ai = pytest.importorskip("inspect_ai")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_inspect_task_runs_and_marks_unparseable_judge_as_nan(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from inspect_tasks.selective_grounding import selective_grounding

    target = get_model("mockllm/model")  # default output: unparseable as a score
    grader = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "1")] * 200)
    logs = inspect_eval(
        selective_grounding(suite="grounded", cases="heat_pump_hum"),
        model=target,
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    log = logs[0]
    assert log.status == "success"
    sample = log.samples[0]
    assert len([m for m in sample.messages if m.role == "assistant"]) == 6
    score = next(iter(sample.scores.values()))
    assert score.value["overreact"] == 1 and score.metadata["counts"]["overreact"] == {"valid": 6, "expected": 6}


def test_unparseable_grader_output_is_nan_not_zero(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from inspect_tasks.selective_grounding import selective_grounding

    logs = inspect_eval(
        selective_grounding(suite="grounded", cases="heat_pump_hum"),
        model=get_model("mockllm/model"),
        model_roles={"grader": get_model("mockllm/model")},  # "Default output..." cannot be parsed
        log_dir=str(tmp_path),
        display="none",
    )
    score = next(iter(logs[0].samples[0].scores.values()))
    assert math.isnan(score.value["overreact"])
    assert score.metadata["counts"]["overreact"]["valid"] == 0
