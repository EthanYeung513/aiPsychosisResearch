"""Judge-human agreement per metric (M3).

  python scripts/judge_agreement.py labels/round1/key.json labels/round1/labels.json
      [--rescored runs/A=runs/A_reasoned ...]

--rescored maps an original run directory to a re-judged copy (made with `psyeval judge`). Judge
scores are then read from the copy for the same case and turn, so two judge setups can be compared
against the SAME human labels.

Reports, per metric: n, exact agreement, Cohen's kappa (unweighted), and the confusion counts. It also
lists every disagreement (item id, case, turn, judge vs human) for inspection. With small n these are
rough indicators, not proof that the judge is valid.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _rescored_lookup(mapping: list[str]) -> dict:
    table: dict[tuple, dict] = {}
    for m in mapping:
        old, new = m.split("=", 1)
        for line in (Path(new) / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            for t in rec["turns"]:
                table[(str(Path(old)), rec["case_id"], t["turn"])] = {k: v["score"] for k, v in t.get("scores", {}).items()}
    return table


def main(key_path: str, labels_path: str, rescored: list[str]) -> int:
    key = json.load(open(key_path, encoding="utf-8"))
    labels = json.load(open(labels_path, encoding="utf-8"))  # {item_id: {metric: value}}
    alt = _rescored_lookup(rescored) if rescored else None
    by_metric: dict[str, list[tuple[int, int]]] = {}
    disagreements = []
    for item_id, human in labels.items():
        k = key.get(item_id)
        if not k:
            continue
        judge_scores = k["judge"]
        if alt is not None:
            judge_scores = alt.get((str(Path(k["run"])), k["case_id"], k["turn"]))
            if judge_scores is None:
                continue
        for metric, h in human.items():
            j = judge_scores.get(metric)
            if j is None or h is None:
                continue
            by_metric.setdefault(metric, []).append((int(j), int(h)))
            if int(j) != int(h):
                disagreements.append({"item": item_id, "case": k["case_id"], "turn": k["turn"], "metric": metric, "judge": j, "human": h})
    report = {
        m: {
            "n": len(p),
            "agreement": round(sum(a == b for a, b in p) / len(p), 3),
            "cohens_kappa": None if kappa(p) is None else round(kappa(p), 3),
            "confusion_judge_human": {f"{a}->{b}": c for (a, b), c in sorted(Counter(p).items())},
        }
        for m, p in sorted(by_metric.items())
    }
    print(json.dumps({"judge_source": "rescored" if alt is not None else "original", "per_metric": report, "disagreements": disagreements}, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("labels")
    ap.add_argument("--rescored", nargs="*", default=[], help="ORIGINAL_RUN=RESCORED_RUN pairs")
    a = ap.parse_args()
    raise SystemExit(main(a.key, a.labels, a.rescored))
