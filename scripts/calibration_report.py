"""Compare judges with explicitly identified human or AI reference labels.

python scripts/calibration_report.py labels/round1
python scripts/calibration_report.py labels/round1 --reference-kind ai --labels-file labels.codex-ai.json

Agreement is descriptive, not evidence that a reference rater is correct. Setups
with different coverage are not ranked against each other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_agreement import kappa  # noqa: E402

VARIANTS = {
    "original": None,
    "ctx2": "runs/rejudge-pilot-m2/ctx2/{run}",
    "r1": "runs/rejudge-pilot-m2/r1/{run}",
    "r4": "runs/rejudge-pilot-m2/r4/{run}-reparsed",
    "r4r3": "runs/rejudge-pilot-m2/r4r3/{run}-reparsed",
}
METRICS = ["endorse", "dcs", "hes", "sis", "overreact", "confront", "warmth", "helpful", "help"]


def load_variant(pattern: str, run_name: str) -> dict:
    path = ROOT / pattern.format(run=run_name) / "conversations.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for t in rec["turns"]:
            out[(rec["case_id"], t["turn"])] = {k: v["score"] for k, v in t.get("scores", {}).items()}
    return out


def main(label_dir: str, labels_file: str = "labels.json", reference_kind: str = "human") -> int:
    d = Path(label_dir)
    label_path = d / labels_file
    # AI export filenames must never silently become a human reference.
    if "ai" in label_path.stem.lower() and reference_kind != "ai":
        raise ValueError("AI label files require --reference-kind ai")
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    unknown = set(labels) - set(key)
    if unknown:
        raise ValueError(f"Unknown label IDs: {sorted(unknown)}")
    cache, pairs = {}, {}
    abstentions = {}
    for item, reference in labels.items():
        k = key[item]
        run_name = Path(k["run"]).name
        for metric, value in reference.items():
            if value is None:
                abstentions[metric] = abstentions.get(metric, 0) + 1
        for variant, pattern in VARIANTS.items():
            if pattern is None:
                scores = k["judge"]
            else:
                if (variant, run_name) not in cache:
                    cache[(variant, run_name)] = load_variant(pattern, run_name)
                scores = cache[(variant, run_name)].get((k["case_id"], k["turn"]), {})
            for metric, h in reference.items():
                j = scores.get(metric)
                if h is not None and j is not None:
                    pairs.setdefault((variant, metric), []).append((int(j), int(h), item))

    print(f"Reference: {reference_kind.upper()}; labelled items: {len(labels)}\n")
    if reference_kind == "ai":
        print("AI-to-AI agreement only. No independent human calibration or clinical validation.\n")
    print("| Metric | " + " | ".join(f"{v} agree / kappa (n)" for v in VARIANTS) + " |")
    print("| --- " * (len(VARIANTS) + 1) + "|")
    for m in METRICS:
        cells = []
        for v in VARIANTS:
            p = pairs.get((v, m), [])
            if not p:
                cells.append("-")
                continue
            agree = sum(a == b for a, b, _ in p) / len(p)
            kap = kappa([(a, b) for a, b, _ in p])
            cells.append(f"{agree:.0%} / {'-' if kap is None else f'{kap:.2f}'} ({len(p)})")
        if any(c != "-" for c in cells):
            print(f"| {m} | " + " | ".join(cells) + " |")
    print("\nNo automatic winner: coverage differs, n is small, and selection on these labels would require fresh validation.")
    print("\nReference abstentions (excluded, never scored as zero):", json.dumps(abstentions, sort_keys=True))
    print("\nDisagreements (judge, reference):")
    for (v, m), p in sorted(pairs.items()):
        bad = [(i, a, b) for a, b, i in p if a != b]
        if bad:
            print(f"- {m} [{v}]: " + ", ".join(f"{i} ({a}, {b})" for i, a, b in bad))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label_dir", nargs="?", default="labels/round1")
    ap.add_argument("--labels-file", default="labels.json", help="Filename relative to label_dir (or absolute path)")
    ap.add_argument("--reference-kind", choices=("human", "ai"), default="human")
    args = ap.parse_args()
    raise SystemExit(main(args.label_dir, args.labels_file, args.reference_kind))
