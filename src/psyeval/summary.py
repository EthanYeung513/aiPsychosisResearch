"""Summaries that keep denominators visible.

- Expected cases come from the run manifest, not from what reached the output file, so a crash that loses
  later cases shows up as `missing` instead of shrinking the denominator.
- Unit of analysis is the conversation (turns within one conversation are dependent).
- Primary values use complete conversations only (all 12 target turns). Partial conversations are
  excluded because late, harder turns are the ones most likely to be lost. An available-case value
  including them is reported as a sensitivity check.
- Uncertainty: cluster bootstrap over scenario families (explicit + implicit variants resampled together).
- Condition comparisons use paired per-family differences on the same families (`compare`).
- Missing data is reported, never turned into 0.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

from .rubrics import METRICS, metrics_for


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_complete(record: dict) -> bool:
    return record["status"] == "ok" and len(record["turns"]) == record["n_prompts"]


def conversation_metric(record: dict, name: str) -> dict:
    metric = METRICS[name]
    values = [
        t["scores"][name]["score"]
        for t in record["turns"]
        if name in t.get("scores", {}) and t["scores"][name]["score"] is not None
    ]
    return {
        "n_expected": max(0, record["n_prompts"] - record.get("metric_start", {}).get(name, metric.start_turn) + 1),
        "n_valid": len(values),
        "mean": mean(values) if values else None,
    }


def _bootstrap(per_family: dict[str, list[float]], iters: int, seed: int) -> list[float] | None:
    families = sorted(per_family)
    if len(families) < 2:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        sample = [v for fam in rng.choices(families, k=len(families)) for v in per_family[fam]]
        stats.append(mean(sample))
    stats.sort()
    return [stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1]]


def _metric_block(records: list[dict], name: str, iters: int, seed: int) -> dict:
    per_family: dict[str, list[float]] = {}
    n_valid = n_expected = 0
    for r in records:
        m = conversation_metric(r, name)
        n_valid += m["n_valid"]
        n_expected += m["n_expected"]
        if m["mean"] is not None:
            per_family.setdefault(r["family"], []).append(m["mean"])
    values = [v for vs in per_family.values() for v in vs]
    return {
        "mean_of_conversation_means": mean(values) if values else None,
        "n_conversations_with_scores": len(values),
        "valid_turn_scores": n_valid,
        "expected_turn_scores": n_expected,
        "ci95_cluster_bootstrap": _bootstrap(per_family, iters, seed),
    }


def summarize(
    records: list[dict],
    expected: dict[str, int],
    suite: str,
    iters: int = 5000,
    seed: int = 0,
    starts: dict[str, dict[str, int]] | None = None,
) -> dict:
    """`expected` maps every case id the run was meant to cover to its number of user turns.
    `starts` optionally maps case id -> metric -> first judged turn (per-case overrides)."""
    starts = starts or {}
    expected_case_ids = list(expected)
    by_id = {r["case_id"]: r for r in records}
    unexpected = sorted(set(by_id) - set(expected_case_ids))
    if unexpected:
        raise ValueError(f"records for cases not in manifest: {unexpected}")
    missing = [c for c in expected_case_ids if c not in by_id]
    complete = [by_id[c] for c in expected_case_ids if c in by_id and is_complete(by_id[c])]
    partial = [by_id[c] for c in expected_case_ids if c in by_id and not is_complete(by_id[c])]
    truncated = sum(t.get("truncated", False) for r in records for t in r["turns"])
    summary: dict = {
        "suite": suite,
        "n_expected_conversations": len(expected_case_ids),
        "n_complete": len(complete),
        "missing": missing,
        "partial": [{"case_id": r["case_id"], "turns": len(r["turns"]), "error": r["error"]} for r in partial],
        "truncated_target_turns": truncated,
        "metrics": {},
    }
    for name, metric in metrics_for(suite).items():
        primary = _metric_block(complete, name, iters, seed)
        primary["expected_turn_scores_all_cases"] = sum(
            max(0, n - starts.get(cid, {}).get(name, metric.start_turn) + 1) for cid, n in expected.items()
        )
        summary["metrics"][name] = {
            "complete_case": primary,
            "available_case_sensitivity": _metric_block(complete + partial, name, iters, seed),
        }
    summary["all_cases_complete"] = not missing and not partial
    summary["all_scores_valid"] = all(
        m["complete_case"]["valid_turn_scores"] == m["complete_case"]["expected_turn_scores_all_cases"]
        for m in summary["metrics"].values()
    )
    return summary


def compare(
    records_a: list[dict], records_b: list[dict], name: str, iters: int = 5000, seed: int = 0
) -> dict:
    """Paired difference B - A per scenario family, using only families complete in both runs."""

    def family_means(records: list[dict]) -> tuple[dict[str, float], set[str]]:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r["family"], []).append(r)
        means, unusable = {}, set()
        for fam, rs in groups.items():
            vals = [conversation_metric(r, name)["mean"] for r in rs if is_complete(r)]
            if len(vals) != len(rs) or any(v is None for v in vals):
                unusable.add(fam)
            else:
                means[fam] = mean(vals)
        return means, unusable

    a, bad_a = family_means(records_a)
    b, bad_b = family_means(records_b)
    shared = sorted(set(a) & set(b))
    diffs = {fam: b[fam] - a[fam] for fam in shared}
    excluded = sorted((set(a) | set(b) | bad_a | bad_b) - set(shared))
    return {
        "metric": name,
        "n_families": len(shared),
        "excluded_families": excluded,
        "mean_paired_diff_b_minus_a": mean(diffs.values()) if diffs else None,
        "ci95_family_bootstrap": _bootstrap({f: [d] for f, d in diffs.items()}, iters, seed),
        "per_family_diff": diffs,
    }


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _family_of(case_id: str, suite: str) -> str:
    # Psychosis-bench pairs explicit/implicit variants; grounded case ids are their own family.
    return case_id.rsplit("_", 1)[0] if suite == "psychosis-bench" else case_id


def _filtered(run_dir: Path, exclude: set[str]) -> tuple[dict, list[dict], dict]:
    m = _manifest(run_dir)
    expected = {c: n for c, n in m["case_turns"].items() if _family_of(c, m["suite"]) not in exclude}
    records = [r for r in load_records(run_dir / "conversations.jsonl") if r["family"] not in exclude]
    return m, records, expected


def summarize_run(run_dir: Path, exclude_families: set[str] | None = None) -> dict:
    exclude = exclude_families or set()
    m, records, expected = _filtered(run_dir, exclude)
    out = summarize(records, expected, m["suite"], starts=m.get("case_metric_start"))
    out["excluded_families"] = sorted(exclude)
    return out


def compare_runs(run_a: Path, run_b: Path, exclude_families: set[str] | None = None) -> dict:
    exclude = exclude_families or set()
    ma, ra, ea = _filtered(run_a, exclude)
    mb, rb, eb = _filtered(run_b, exclude)
    if ea != eb or ma["suite"] != mb["suite"]:
        raise ValueError("runs cover different suites or case sets")
    out = {name: compare(ra, rb, name) for name in metrics_for(ma["suite"])}
    out["excluded_families"] = sorted(exclude)
    return out
