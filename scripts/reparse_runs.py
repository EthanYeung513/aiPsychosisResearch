"""Re-parse stored raw judge outputs with parser revision P-parse (no judge calls).

Only reasoned-style scores of PROJECT metrics are re-parsed. Upstream metrics and bare-style runs
are copied unchanged. Writes <run>-reparsed/ next to each run and prints invalid counts before and
after.

  python scripts/reparse_runs.py runs/rejudge-pilot-m2/r4/* runs/rejudge-pilot-m2/r1/*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psyeval.rubrics import METRICS, parse_score_end  # noqa: E402


def main(paths: list[str]) -> int:
    for p in map(Path, paths):
        if p.name.endswith("-reparsed") or not (p / "conversations.jsonl").exists():
            continue
        manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        reasoned = manifest.get("judge_style") == "reasoned"
        before = after = changed = 0
        out_lines = []
        for line in (p / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            for t in rec["turns"]:
                for name, s in t.get("scores", {}).items():
                    before += s["score"] is None
                    if reasoned and not METRICS[name].upstream and s.get("raw") is not None:
                        new = parse_score_end(s["raw"], METRICS[name].max_score)
                        if new != s["score"]:
                            changed += 1
                        s["score"] = new
                        if new is None:
                            s["error"] = s.get("error") or "unparseable judge output (P-parse)"
                        else:
                            s.pop("error", None)
                    after += s["score"] is None
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        dest = p.with_name(p.name + "-reparsed")
        dest.mkdir(exist_ok=False)
        manifest["parser_revision"] = "P-parse (output must end with 'SCORE: n'; reasoned project metrics only)"
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (dest / "conversations.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"{p}: invalid {before} -> {after}, scores changed {changed}, reasoned={reasoned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
