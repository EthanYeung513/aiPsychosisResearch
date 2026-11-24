from pathlib import Path

import pytest

from psyeval.cases import GROUNDED, PSYCHOSIS, UPSTREAM_DIR, load_cases, load_grounded_cases
from psyeval.chat import ChatError, Reply, parse_completion
from psyeval.rubrics import METRICS, build_prompt, metrics_for, parse_score
from psyeval.runner import rejudge_record, run_cases, run_conversation
from psyeval.summary import compare, summarize


class ScriptedClient:
    """Returns canned replies; raises ChatError on the listed (1-based) call numbers."""

    def __init__(self, reply="ok", fail_on=(), finish_reason="stop"):
        self.reply = reply
        self.fail_on = set(fail_on)
        self.finish_reason = finish_reason
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) in self.fail_on:
            raise ChatError("boom")
        text = self.reply(messages) if callable(self.reply) else self.reply
        return Reply(text, self.finish_reason)

    def describe(self):
        return {}


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_cases_load_and_pair_into_families(cases):
    assert len(cases) == 16
    assert all(len(c.prompts) == 12 for c in cases)
    families = {c.family for c in cases}
    assert len(families) == 8
    assert {c.condition for c in cases} == {"Explicit", "Implicit"}


def test_rubric_templates_match_upstream_verbatim():
    upstream = (UPSTREAM_DIR / "psy_bench" / "core" / "scoring.py").read_text(encoding="utf-8")
    upstream_metrics = [m for m in METRICS.values() if m.upstream]
    assert {m.name for m in upstream_metrics} == {"dcs", "hes", "sis"}
    for metric in upstream_metrics:
        assert metric.template in upstream, f"{metric.name} template drifted from upstream"


@pytest.mark.parametrize(
    "text,max_score,expected",
    [
        ("2", 2, 2),
        (" 0\n", 2, 0),
        ("Score: 1", 2, 1),
        ("**2**", 2, 2),
        ("1.", 1, 1),
        ("3", 2, None),
        ("2", 1, None),
        ("0, 1 or 2? I'd say 2", 2, None),  # upstream would read this as 0
        ("Score: 4/5", 2, None),
        ("", 2, None),
    ],
)
def test_parse_score_is_strict(text, max_score, expected):
    assert parse_score(text, max_score) == expected


def test_judge_windows_match_upstream(cases):
    target = ScriptedClient("reply")
    judge = ScriptedClient("1")
    record = run_conversation(cases[0], target, judge)
    assert record["status"] == "ok"
    scored = {name: [t["turn"] for t in record["turns"] if name in t["scores"]] for name in METRICS}
    from4, from7 = list(range(4, 13)), list(range(7, 13))
    assert scored == {
        "dcs": from4, "hes": from7, "sis": from7,
        "endorse": from4, "confront": from4, "warmth": from4,
        "overreact": [], "helpful": [], "help": [],
    }


def test_target_error_aborts_and_is_never_scored(cases):
    target = ScriptedClient("reply", fail_on={5})
    judge = ScriptedClient("2")
    record = run_conversation(cases[0], target, judge)
    assert record["status"] == "target_error"
    assert len(record["turns"]) == 4
    assert all("ERROR" not in t["assistant"] for t in record["turns"])
    assert len(target.calls) == 5  # stopped, did not continue without an assistant turn


def test_history_alternates_and_includes_system_prompt(cases):
    target = ScriptedClient("reply")
    run_conversation(cases[0], target, None, system_prompt="be kind")
    last = target.calls[-1]
    assert last[0] == {"role": "system", "content": "be kind"}
    roles = [m["role"] for m in last[1:]]
    assert roles == ["user", "assistant"] * 11 + ["user"]


def test_invalid_judge_output_is_missing_not_zero(cases):
    target = ScriptedClient("reply")
    judge = ScriptedClient("I cannot score this")
    record = run_conversation(cases[0], target, judge)
    summary = summarize([record], {cases[0].id: 12}, PSYCHOSIS, iters=10)
    dcs = summary["metrics"]["dcs"]["complete_case"]
    assert dcs["mean_of_conversation_means"] is None
    assert dcs["valid_turn_scores"] == 0
    assert dcs["expected_turn_scores"] == 9
    assert summary["all_scores_valid"] is False


def test_summary_denominators_with_aborted_conversation(cases):
    judge = ScriptedClient("2")
    ok = run_conversation(cases[0], ScriptedClient("r"), judge)
    aborted = run_conversation(cases[1], ScriptedClient("r", fail_on={1}), judge)
    summary = summarize([ok, aborted], {cases[0].id: 12, cases[1].id: 12}, PSYCHOSIS, iters=10)
    assert [p["case_id"] for p in summary["partial"]] == [cases[1].id]
    dcs = summary["metrics"]["dcs"]["complete_case"]
    assert dcs["mean_of_conversation_means"] == 2
    assert dcs["valid_turn_scores"] == 9
    assert dcs["expected_turn_scores_all_cases"] == 18
    assert summary["all_cases_complete"] is False


def test_prompt_with_braces_does_not_break(cases):
    target = ScriptedClient("use {curly} braces {conversation}")
    judge = ScriptedClient(lambda m: "0")
    record = run_conversation(cases[0], target, judge)
    assert record["status"] == "ok"


def test_cases_lost_before_writing_count_as_missing(cases):
    ok = run_conversation(cases[0], ScriptedClient("r"), ScriptedClient("1"))
    summary = summarize([ok], {c.id: 12 for c in cases[:4]}, PSYCHOSIS, iters=10)
    assert summary["missing"] == [c.id for c in cases[1:4]]
    assert summary["metrics"]["dcs"]["complete_case"]["expected_turn_scores_all_cases"] == 36
    assert summary["all_cases_complete"] is False


def test_partial_conversation_excluded_from_primary_but_in_sensitivity(cases):
    judge = ScriptedClient("2")
    partial = run_conversation(cases[0], ScriptedClient("r", fail_on={10}), judge)
    summary = summarize([partial], {cases[0].id: 12}, PSYCHOSIS, iters=10)
    dcs = summary["metrics"]["dcs"]
    assert dcs["complete_case"]["mean_of_conversation_means"] is None
    assert dcs["available_case_sensitivity"]["mean_of_conversation_means"] == 2


def test_truncated_target_flagged_and_truncated_judge_invalid(cases):
    record = run_conversation(cases[0], ScriptedClient("r", finish_reason="length"), ScriptedClient("1", finish_reason="length"))
    assert all(t["truncated"] for t in record["turns"])
    assert all(s["score"] is None for t in record["turns"] for s in t["scores"].values())
    summary = summarize([record], {cases[0].id: 12}, PSYCHOSIS, iters=10)
    assert summary["truncated_target_turns"] == 12


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        {"choices": None},
        {"choices": []},
        {"choices": ["text"]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {"content": "hi"}, "finish_reason": 3}]},
    ],
)
def test_malformed_bodies_raise_chat_error(body):
    with pytest.raises(ChatError):
        parse_completion(body)


def test_parse_completion_reads_finish_reason():
    reply = parse_completion({"choices": [{"message": {"content": " hi "}, "finish_reason": "length"}]})
    assert reply == Reply("hi", "length") and reply.truncated


def test_compare_pairs_by_family_and_excludes_incomplete(cases):
    fam_cases = [c for c in cases if c.family in {"conduit", "digital_romance", "turing_test"}]
    def run(score, fail_ids=()):
        return [
            run_conversation(c, ScriptedClient("r", fail_on={3} if c.id in fail_ids else ()), ScriptedClient(score))
            for c in fam_cases
        ]
    a = run("2")
    b = run("1", fail_ids={"turing_test_implicit"})
    result = compare(a, b, "dcs", iters=10)
    assert result["n_families"] == 2
    assert result["excluded_families"] == ["turing_test"]
    assert result["mean_paired_diff_b_minus_a"] == -1


def test_grounded_cases_load_and_are_disjoint_from_benchmark(cases):
    grounded = load_grounded_cases()
    assert len(grounded) == 8 and all(g.suite == GROUNDED and g.context for g in grounded)
    assert len({g.family for g in grounded}) == 8
    assert not {g.family for g in grounded} & {c.family for c in cases}
    bench_text = {p for c in cases for p in c.prompts}
    assert not any(p in bench_text for g in grounded for p in g.prompts)


def test_grounded_suite_uses_grounded_metrics_with_context():
    case = load_grounded_cases()[0]
    judge = ScriptedClient("0")
    record = run_conversation(case, ScriptedClient("reply"), judge)
    assert record["suite"] == GROUNDED and record["n_prompts"] == 6
    assert set(record["turns"][0]["scores"]) == {"overreact", "helpful"}
    assert set(record["turns"][3]["scores"]) == {"overreact", "helpful", "confront", "warmth"}
    overreact_prompts = [m[0]["content"] for m in judge.calls if "OVER-REACT" in m[0]["content"]]
    assert overreact_prompts and all(case.context in p for p in overreact_prompts)
    summary = summarize([record], {case.id: 6}, GROUNDED, iters=10)
    assert set(summary["metrics"]) == set(metrics_for(GROUNDED))
    assert summary["metrics"]["overreact"]["complete_case"]["expected_turn_scores_all_cases"] == 6
    assert summary["metrics"]["warmth"]["complete_case"]["expected_turn_scores_all_cases"] == 3
    assert summary["all_cases_complete"] and summary["all_scores_valid"]


def test_build_prompt_never_substitutes_inside_conversation():
    metric = METRICS["overreact"]
    prompt = build_prompt(metric, [("my {context} and {harm_type}", "ok")], "None", "CTX")
    assert "my {context} and {harm_type}" in prompt
    assert prompt.count("CTX") == 1


def test_new_rubrics_end_with_single_integer_instruction():
    for m in METRICS.values():
        if not m.upstream:
            assert "{conversation}" in m.template
            assert m.template.rstrip().endswith(("0 or 1.", "0, 1 or 2."))


def test_help_metric_only_for_grounded_risk_suite():
    assert set(metrics_for("grounded-risk")) == {"overreact", "helpful", "confront", "warmth", "help"}
    assert "help" not in metrics_for("grounded") and "help" not in metrics_for(PSYCHOSIS)


def test_help_judge_waits_for_case_specific_risk_turn():
    from psyeval.cases import load_suite
    case = next(c for c in load_suite("grounded-risk", "dev") if c.id == "grief_passive_si")
    assert case.help_from_turn == 5
    record = run_conversation(case, ScriptedClient("reply"), ScriptedClient("1"))
    help_turns = [t["turn"] for t in record["turns"] if "help" in t["scores"]]
    assert help_turns == [5, 6]
    starts = {case.id: record["metric_start"]}
    summary = summarize([record], {case.id: 6}, "grounded-risk", iters=10, starts=starts)
    h = summary["metrics"]["help"]["complete_case"]
    assert h["valid_turn_scores"] == h["expected_turn_scores_all_cases"] == 2
    assert summary["all_scores_valid"]


def test_rejudge_scores_existing_transcript_without_regenerating(cases):
    target = ScriptedClient("reply")
    unjudged = run_conversation(cases[0], target, None)
    assert all("scores" not in t for t in unjudged["turns"])
    judge = ScriptedClient("1")
    judged = rejudge_record(unjudged, cases[0], judge)
    assert len(target.calls) == 12  # no new generation
    assert [t["assistant"] for t in judged["turns"]] == [t["assistant"] for t in unjudged["turns"]]
    summary = summarize([judged], {cases[0].id: 12}, PSYCHOSIS, iters=10)
    assert summary["all_scores_valid"] and summary["metrics"]["endorse"]["complete_case"]["mean_of_conversation_means"] == 1


def test_rejudge_refuses_mismatched_script(cases):
    rec = run_conversation(cases[0], ScriptedClient("r"), None)
    rec["turns"][0]["user"] = "tampered"
    with pytest.raises(ValueError):
        rejudge_record(rec, cases[0], ScriptedClient("1"))


def test_cli_module_imports_and_parses():
    import psyeval.__main__ as cli
    with pytest.raises(SystemExit) as e:
        cli.main(["judge", "--help"])
    assert e.value.code == 0


def test_cli_passes_target_extra_body_to_the_target_client(monkeypatch, tmp_path):
    import psyeval.__main__ as cli
    seen = []

    class Stop(Exception):
        pass

    def fake_client(**kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(cli, "OpenAICompatibleClient", fake_client)
    with pytest.raises(Stop):
        cli.main(["run", "--target-base-url", "http://x/v1", "--target-model", "m", "--no-judge",
                  "--target-extra-body", '{"chat_template_kwargs": {"enable_thinking": false}}',
                  "--out", str(tmp_path / "r")])
    assert seen[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_concurrent_run_preserves_case_order_and_matches_sequential(cases, tmp_path):
    import json as _json
    import threading
    import time as _time

    class SlowEcho:
        """Thread-safe target: reply depends only on the conversation, with jittered latency."""
        lock = threading.Lock()

        def chat(self, messages):
            _time.sleep(0.001 * (len(messages) % 3))
            return Reply("echo:" + messages[-1]["content"][:20], "stop")

        def describe(self):
            return {}

    subset = cases[:5]
    seq = run_cases(subset, SlowEcho(), None, tmp_path / "seq", {}, log=lambda *_: None, workers=1)
    par = run_cases(subset, SlowEcho(), None, tmp_path / "par", {}, log=lambda *_: None, workers=4)
    a = [_json.loads(l) for l in seq.read_text(encoding="utf-8").splitlines()]
    b = [_json.loads(l) for l in par.read_text(encoding="utf-8").splitlines()]
    assert [r["case_id"] for r in b] == [c.id for c in subset]
    strip = lambda rs: [[(t["user"], t["assistant"]) for t in r["turns"]] for r in rs]
    assert strip(a) == strip(b)


def test_summarize_run_can_exclude_dev_exposed_family(cases, tmp_path):
    from psyeval.summary import summarize_run
    subset = [c for c in cases if c.family in {"conduit", "turing_test"}]
    manifest = {"suite": PSYCHOSIS, "case_turns": {c.id: 12 for c in subset}}
    run_cases(subset, ScriptedClient("r"), ScriptedClient("2"), tmp_path / "run", manifest, log=lambda *_: None)
    full = summarize_run(tmp_path / "run")
    primary = summarize_run(tmp_path / "run", {"conduit"})
    assert full["n_expected_conversations"] == 4 and primary["n_expected_conversations"] == 2
    assert primary["excluded_families"] == ["conduit"] and primary["all_cases_complete"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("It elaborates the decoding as real.\nSCORE: 1", 1),
        ("Reasoning.\n**SCORE: 0**", 0),
        ("SCORE: 1\nActually I changed my mind.", None),  # score must be on the final line
        ("The answer is 1", None),
        ("Reasoning.\nSCORE: 3", None),
    ],
)
def test_reasoned_parse_uses_final_line_only(text, expected):
    assert parse_score(text, 1, "reasoned") == expected


def test_reasoned_style_never_changes_upstream_rubrics(cases):
    from psyeval.rubrics import template_for
    for m in METRICS.values():
        if m.upstream:
            assert template_for(m, "reasoned") == m.template
        else:
            assert "SCORE: <n>" in template_for(m, "reasoned")


def test_reasoned_judge_end_to_end(cases):
    judge = ScriptedClient(lambda msgs: "SCORE: 1" if "SCORE: <n>" in msgs[0]["content"] else "2")
    record = run_conversation(cases[0], ScriptedClient("r"), judge, judge_style="reasoned")
    t = record["turns"][-1]["scores"]
    assert t["endorse"]["score"] == 1 and t["dcs"]["score"] == 2  # upstream stays bare


def test_judge_context_turns_limits_what_the_judge_sees(cases):
    rec = run_conversation(cases[0], ScriptedClient(lambda m: f"reply-{len(m)}"), None)
    judge = ScriptedClient("1")
    rejudge_record(rec, cases[0], judge, context_turns=2)
    last_prompt = judge.calls[-1][0]["content"]
    assert last_prompt.count("Human: ") == 2
    assert rec["turns"][-1]["assistant"] in last_prompt and rec["turns"][0]["assistant"] not in last_prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning here. SCORE: 2", 2),
        ("Reasoning." + chr(10) + "SCORE: 1", 1),
        ("Reasoning. **SCORE: 0**", 0),
        ("SCORE: 1 but on reflection SCORE: 0", None),  # two scores: ambiguous
        ("SCORE: 1" + chr(10) + "more text after", None),  # must end with the score
        ("Reasoning. SCORE: 3", None),
        ("no score at all", None),
    ],
)
def test_parse_score_end(text, expected):
    from psyeval.rubrics import parse_score_end
    assert parse_score_end(text, 2) == expected


def test_turn_aware_context_changes_grounded_prompts_only_when_enabled():
    case = next(c for c in load_grounded_cases() if c.id == "heat_pump_hum")
    rec = run_conversation(case, ScriptedClient("reply"), None)

    def overreact_prompts(mode):
        judge = ScriptedClient("0")
        rejudge_record(rec, case, judge, context_mode=mode)
        return [m[0]["content"] for m in judge.calls if "OVER-REACT" in m[0]["content"]]

    full, aware = overreact_prompts("full"), overreact_prompts("turn-aware")
    assert all(case.context in p for p in full)  # default unchanged
    assert case.context not in aware[1] and case.context_at(2, "turn-aware") in aware[1]  # turn 2
    assert case.context in aware[-1]  # from turn 4 the full facts apply


def test_turn_aware_mode_is_a_no_op_for_cases_without_per_turn_contexts(cases):
    rec = run_conversation(cases[0], ScriptedClient("r"), None)
    a, b = ScriptedClient("1"), ScriptedClient("1")
    rejudge_record(rec, cases[0], a, context_mode="full")
    rejudge_record(rec, cases[0], b, context_mode="turn-aware")
    assert [m[0]["content"] for m in a.calls] == [m[0]["content"] for m in b.calls]


def test_r5_helpful_v2_is_recorded_but_not_judged():
    from psyeval.rubrics import HELPFUL_V2_TEMPLATE
    assert "helpful_v2" not in METRICS and "materially wrong fact" in HELPFUL_V2_TEMPLATE
