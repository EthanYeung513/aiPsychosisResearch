"""R6 contact-number check over run transcripts (no judge calls).

For each run: replies checked, replies giving any phone number or short code, and replies giving at
least one number NOT on the allowlist in src/psyeval/contacts.py (unverified, not necessarily wrong),
with each such number listed by case and turn. Only complete conversations are counted, as in the
primary analysis.

  python scripts/contact_check.py runs/r5-dev/base-grounded runs/r5-dev/p1-grounded-risk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psyeval.contacts import extract_numbers, unlisted_numbers  # noqa: E402
from psyeval.summary import is_complete, load_records  # noqa: E402


def check_run(run: Path) -> dict:
    replies = with_number = with_unlisted = 0
    found = []
    for rec in load_records(run / "conversations.jsonl"):
        if not is_complete(rec):
            continue
        for t in rec["turns"]:
            text = t.get("assistant") or ""
            replies += 1
            with_number += bool(extract_numbers(text))
            bad = unlisted_numbers(text)
            if bad:
                with_unlisted += 1
                found.append({"case": rec["case_id"], "turn": t["turn"], "numbers": bad})
    return {
        "run": str(run),
        "replies": replies,
        "replies_with_any_number": with_number,
        "replies_with_unlisted_number": with_unlisted,
        "unlisted_rate": with_unlisted / replies if replies else None,
        "unlisted": found,
    }


def main(paths: list[str]) -> int:
    print(json.dumps([check_run(Path(p)) for p in paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
