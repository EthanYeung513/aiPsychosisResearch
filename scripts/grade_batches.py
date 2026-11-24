"""Blind grading of help and helpfulness by a stronger grader (revision R7, docs/rubric-revisions.md).

The local judge failed calibration on these two metrics, so they are graded by a stronger model
following prompts/grader_help_helpfulness.txt, in blinded batches.

  make : collect every scored turn of the grounded suites from complete conversations in the given
         runs, give each a random id, shuffle across runs (so every batch mixes conditions), add a
         fraction of duplicates under new ids (grader-consistency check), and write
         <out>/batch_NN/items.json plus <out>/key.json (do not show key.json to the grader).
  apply: read <out>/batch_NN/labels.claude-ai.json, report duplicate consistency and coverage, and
         write a copy of each run to <run>-graded/ with the 'helpful' and 'help' scores replaced by
         the grader's. Missing grades stay None (never 0). Other metrics are unchanged.

  python scripts/grade_batches.py make RUN... --out grading/dev
  python scripts/grade_batches.py apply grading/dev
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psyeval.cases import load_all_grounded_contexts  # noqa: E402
from psyeval.summary import is_complete, load_records  # noqa: E402

GRADED = {"helpful": "helpfulness", "help": "help"}  # run metric -> grader label name
LABELS_FILE = "labels.claude-ai.json"


def make(runs: list[str], out: Path, batch_size: int, dup_frac: float, seed: int) -> int:
    contexts = load_all_grounded_contexts()
    items, key = [], {}
    for run in map(Path, runs):
        for rec in load_records(run / "conversations.jsonl"):
            if rec["suite"] == "psychosis-bench" or not is_complete(rec):
                continue
            for t in rec["turns"]:
                questions = [GRADED[m] for m in GRADED if m in t.get("scores", {})]
                if not questions:
                    continue
                convo = []
                for u in rec["turns"][: t["turn"]]:
                    convo += [{"role": "user", "content": u["user"]}, {"role": "assistant", "content": u["assistant"]}]
                item_id = secrets.token_hex(4)
                items.append({"id": item_id, "context": contexts.get(rec["case_id"], ""), "conversation": convo,
                              "questions": questions})
                key[item_id] = {"run": str(run), "case_id": rec["case_id"], "turn": t["turn"], "dup_of": None}
    rng = random.Random(seed)
    for original in rng.sample(items, round(len(items) * dup_frac)):
        dup_id = secrets.token_hex(4)
        items.append({**original, "id": dup_id})
        key[dup_id] = {**key[original["id"]], "dup_of": original["id"]}
    rng.shuffle(items)
    out.mkdir(parents=True, exist_ok=False)
    for b in range(0, len(items), batch_size):
        d = out / f"batch_{b // batch_size:02d}"
        d.mkdir()
        (d / "items.json").write_text(json.dumps(items[b:b + batch_size], indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} items ({sum(v['dup_of'] is not None for v in key.values())} duplicates) "
          f"in {(len(items) + batch_size - 1) // batch_size} batches -> {out}")
    return 0


def apply(out: Path) -> int:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    labels: dict[str, dict] = {}
    for f in sorted(out.glob(f"batch_*/{LABELS_FILE}")):
        labels.update(json.loads(f.read_text(encoding="utf-8")))
    unknown = set(labels) - set(key)
    if unknown:
        raise ValueError(f"Unknown item ids in grader labels: {sorted(unknown)[:5]}")
    pairs = {m: [] for m in GRADED.values()}
    for item, k in key.items():
        if k["dup_of"] and item in labels and k["dup_of"] in labels:
            for m in pairs:
                a, b = labels[item].get(m), labels[k["dup_of"]].get(m)
                if a is not None and b is not None:
                    pairs[m].append(a == b)
    report = {
        "items": len(key), "graded": len(labels),
        "duplicate_consistency": {m: (sum(v) / len(v) if v else None, len(v)) for m, v in pairs.items()},
    }
    grades: dict[tuple[str, str, int], dict] = {}
    for item, k in key.items():
        if k["dup_of"] is None:
            grades[(k["run"], k["case_id"], k["turn"])] = labels.get(item, {})
    for run in sorted({k["run"] for k in key.values()}):
        src, dst = Path(run), Path(run + "-graded")
        dst.mkdir(exist_ok=False)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        manifest["graded_metrics"] = {m: {"grader": "blind stronger-model grader (Claude)",
                                          "prompt": "prompts/grader_help_helpfulness.txt", "dir": str(out)}
                                      for m in GRADED}
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        lines = []
        for rec in load_records(src / "conversations.jsonl"):
            for t in rec["turns"]:
                g = grades.get((run, rec["case_id"], t["turn"]))
                for metric, label in GRADED.items():
                    if metric in t.get("scores", {}):
                        score = (g or {}).get(label)
                        t["scores"][metric] = {"score": score, "raw": "grader" if score is not None else "grader: missing"}
            lines.append(json.dumps(rec, ensure_ascii=False))
        (dst / "conversations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("runs", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--batch-size", type=int, default=40)
    m.add_argument("--dup-frac", type=float, default=0.1)
    m.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("apply")
    a.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "make":
        return make(args.runs, Path(args.out), args.batch_size, args.dup_frac, args.seed)
    return apply(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
