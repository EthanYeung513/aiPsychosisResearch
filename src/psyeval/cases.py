"""Evaluation cases.

- Psychosis-bench: read from the pinned upstream checkout, with the file hash verified on every load.
- Grounded controls: our original conversations in data/grounded_controls.json. Here the user's claims
  are true or ordinary, and a per-case `context` note tells the judge so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "psychosis-bench"
UPSTREAM_COMMIT = "73966f95be2e40f1ceee76dfe08eebe939ad6c21"
CASES_PATH = UPSTREAM_DIR / "data" / "test_cases.json"
# sha256 of upstream's file with LF line endings (Windows checkouts may convert to CRLF; the hash
# normalises CRLF to LF so it matches on every platform). Same content as the earlier CRLF pin 69598ca5…99b4.
CASES_SHA256 = "d9b7820c0bebb6ec845e5825378535e8e35b79b0244a72be64ec5e49d8da439f"
GROUNDED_PATH = REPO_ROOT / "data" / "grounded_controls.json"
GROUNDED_TEST_PATH = REPO_ROOT / "data" / "grounded_test.json"
GROUNDED_RISK_PATH = REPO_ROOT / "data" / "grounded_at_risk.json"

PSYCHOSIS = "psychosis-bench"
GROUNDED = "grounded"  # low-risk grounded controls (dev or test file)
GROUNDED_RISK = "grounded-risk"  # true claims plus real risk: help IS appropriate
SUITES = (PSYCHOSIS, GROUNDED, GROUNDED_RISK)


@dataclass(frozen=True)
class Case:
    id: str
    name: str
    theme: str
    condition: str  # Psychosis-bench: "Explicit"/"Implicit"; grounded: "Grounded"
    harm_type: str
    prompts: tuple[str, ...]
    suite: str = PSYCHOSIS
    family_id: str | None = None
    context: str = ""  # judge-visible ground truth for grounded controls
    help_from_turn: int | None = None  # grounded-risk: first turn at which concrete help is appropriate
    # Candidate revision R3: judge-visible facts as established BY a given turn. A tuple of
    # (from_turn, text), sorted. Used only when the judge runs with context_mode="turn-aware".
    context_by_turn: tuple[tuple[int, str], ...] = ()

    @property
    def family(self) -> str:
        """Scenario family; for Psychosis-bench the explicit/implicit pair, e.g. 'conduit'."""
        return self.family_id or self.id.rsplit("_", 1)[0]

    def context_at(self, turn: int, mode: str = "full") -> str:
        """Judge context for a turn. 'full' is the whole-case context (the default). 'turn-aware'
        (R3) uses the latest context_by_turn entry at or before this turn, and falls back to the
        full context when the case defines no per-turn contexts."""
        if mode == "full" or not self.context_by_turn:
            return self.context
        if mode != "turn-aware":
            raise ValueError(f"unknown context mode {mode!r}")
        current = self.context
        for from_turn, text in self.context_by_turn:
            if from_turn <= turn:
                current = text
        return current


def load_cases(path: Path = CASES_PATH, expected_sha256: str | None = CASES_SHA256) -> list[Case]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path} sha256 {digest} != pinned {expected_sha256}; re-pin deliberately if upstream changed")
    data = json.loads(raw.decode("utf-8"))
    return [
        Case(
            id=c["id"],
            name=c["name"],
            theme=c["theme"],
            condition=c["condition"],
            harm_type=c["harm_type"],
            prompts=tuple(c["prompts"]),
        )
        for c in data["cases"]
    ]


def load_grounded_cases(path: Path = GROUNDED_PATH, suite: str = GROUNDED, split: str | None = None) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if split is None or c.get("split", data.get("split")) == split]
    return [
        Case(
            id=c["id"],
            name=c["id"],
            theme=c.get("mirrors", ""),
            condition="Grounded",
            harm_type="None",
            prompts=tuple(c["prompts"]),
            suite=suite,
            family_id=c["family"],
            context=c["context"],
            help_from_turn=c.get("help_from_turn"),
            context_by_turn=tuple(sorted((int(k), v) for k, v in c.get("context_by_turn", {}).items())),
        )
        for c in cases
    ]


def load_suite(suite: str, split: str = "dev") -> list[Case]:
    """split applies to the grounded suites: 'dev' or 'test'."""
    if suite == PSYCHOSIS:
        return load_cases()
    if suite == GROUNDED:
        return load_grounded_cases(GROUNDED_PATH if split == "dev" else GROUNDED_TEST_PATH)
    if suite == GROUNDED_RISK:
        return load_grounded_cases(GROUNDED_RISK_PATH, suite=GROUNDED_RISK, split=split)
    raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")


def load_all_grounded_contexts() -> dict[str, str]:
    """case id -> judge context, across every grounded file that exists."""
    out: dict[str, str] = {}
    for path, suite in ((GROUNDED_PATH, GROUNDED), (GROUNDED_TEST_PATH, GROUNDED), (GROUNDED_RISK_PATH, GROUNDED_RISK)):
        if path.exists():
            out.update({c.id: c.context for c in load_grounded_cases(path, suite=suite)})
    return out
