"""Build the training set for each adapter condition from the drafted or reviewed data.

  A   : every concerning example
  B   : per concerning family, one early, one middle and one late example, plus their matched grounded twins
  A'  : B's concerning examples only (same exposure to them as in B)

Only examples with review in {accepted, edited} are used by default. --allow-ai-reviewed explicitly
admits AI-reviewed data for exploratory training; --include-pending is for pipeline testing only.
Split exclusions apply in every mode. Incomplete paired strata fail. Selection is deterministic (--seed). Writes
<out>/{A,B,A_prime}.jsonl and <out>/manifest.json with counts, ids and, with --tokenizer, target-token
budgets computed with the same masking as scripts/train_lora.py.

  python scripts/build_conditions.py --out data/conditions [--include-pending] [--tokenizer Qwen/Qwen3.5-0.8B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONCERNING = ROOT / "data" / "drafts" / "train_concerning.jsonl"
GROUNDED = ROOT / "data" / "drafts" / "train_grounded.jsonl"
OK_REVIEW = {"accepted", "edited"}
AI_REVIEW = {"ai_accepted", "ai_edited"}
SPLITS = ROOT / "data" / "splits.json"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_reviewed(records, allowed, excluded):
    return [r for r in records if r["review"] in allowed
            and r["family"] not in excluded and r.get("split", "train") == "train"
            and not r.get("review_metadata", {}).get("split_review_required", False)]


def validate_pool(concerning, grounded, families):
    if not concerning:
        raise ValueError("No eligible data. AI-reviewed drafts require explicit --allow-ai-reviewed.")
    all_rows = concerning + list(grounded.values())
    if len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate example IDs")
    if {r['family'] for r in concerning} != set(families):
        raise ValueError("Training family coverage differs from splits.json")
    for family in families:
        for stage in ('early', 'middle', 'late'):
            rs = [r for r in concerning if r['family'] == family and r['stage'] == stage]
            if len(rs) != 2:
                raise ValueError(f"Expected two concerning examples in {family}/{stage}")
    for r in concerning:
        twin = grounded.get(r['pair_id'])
        if not twin or twin['pair_id'] != r['id'] or twin['stage'] != r['stage']:
            raise ValueError(f"Missing or inconsistent reviewed counterpart: {r['id']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--include-pending", action="store_true")
    ap.add_argument("--allow-ai-reviewed", action="store_true", help="Explicitly use AI review for exploratory training; not human validation")
    ap.add_argument("--tokenizer", help="HF tokenizer id/path, to report target-token budgets")
    args = ap.parse_args()

    allowed = OK_REVIEW | ({"pending"} if args.include_pending else set()) | (AI_REVIEW if args.allow_ai_reviewed else set())
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    excluded = set(splits['training'].get('excluded_families', []))
    concerning = select_reviewed(read(CONCERNING), allowed, excluded)
    grounded = {e['id']: e for e in select_reviewed(read(GROUNDED), allowed, excluded)}
    families = set(splits['training']['concerning']['families']) - excluded
    validate_pool(concerning, grounded, families)

    rng = random.Random(args.seed)
    by_family: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for e in concerning:
        by_family[e["family"]][e["stage"]].append(e)
    b_concerning, missing = [], []
    for fam in sorted(by_family):
        for stage in ("early", "middle", "late"):
            options = [e for e in by_family[fam][stage] if e["pair_id"] in grounded]
            if not options:
                missing.append(f"{fam}/{stage}")
                continue
            b_concerning.append(rng.choice(options))
    b_grounded = [grounded[e["pair_id"]] for e in b_concerning]
    if missing:
        raise ValueError(f"Incomplete strata: {missing}")

    conditions = {
        "A": sorted(concerning, key=lambda e: e["id"]),
        "B": sorted(b_concerning + b_grounded, key=lambda e: e["id"]),
        "A_prime": sorted(b_concerning, key=lambda e: e["id"]),
    }

    budgets = {}
    if args.tokenizer:
        sys.path.insert(0, str(ROOT / "scripts"))
        from transformers import AutoTokenizer

        import train_lora

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for name, exs in conditions.items():
            encs = [train_lora.tokenize_example(tok, e["messages"], 4096) for e in exs]
            if any(x is None for x in encs):
                raise SystemExit(f"{name}: some examples could not be encoded")
            budgets[name] = {
                "target_tokens": sum(sum(l != -100 for l in x["labels"]) for x in encs),
                "total_tokens": sum(len(x["input_ids"]) for x in encs),
                "target_check_failures": train_lora.check_targets(tok, encs, exs),
            }
            if budgets[name]['target_check_failures']:
                raise ValueError(f"{name}: assistant-target verification failed")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "include_pending": args.include_pending,
        "allow_ai_reviewed": args.allow_ai_reviewed,
        "review_basis": sorted({e['review'] for e in concerning + b_grounded}),
        "excluded_families": sorted(excluded),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONCERNING, GROUNDED, SPLITS)},
        "missing_strata_for_B": missing,
        "tokenizer": args.tokenizer,
        "target_token_relative_difference_A_B": (
            abs(budgets['A']['target_tokens'] - budgets['B']['target_tokens']) / budgets['A']['target_tokens']
            if budgets else None
        ),
        "conditions": {},
    }
    for name, exs in conditions.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exs), encoding="utf-8")
        manifest["conditions"][name] = {
            "n": len(exs),
            "n_concerning": sum(e["arm"] == "concerning" for e in exs),
            "n_grounded": sum(e["arm"] == "grounded" for e in exs),
            "ids": [e["id"] for e in exs],
            **budgets.get(name, {}),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {k: {kk: v for kk, v in c.items() if kk != "ids"} for k, c in manifest["conditions"].items()}
    print(json.dumps({"include_pending": args.include_pending, "missing_strata_for_B": missing, "conditions": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
