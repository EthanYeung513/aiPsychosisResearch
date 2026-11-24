import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "calibration_report", Path(__file__).resolve().parents[1] / "scripts/calibration_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_files(tmp_path, filename="labels.codex-ai.json"):
    (tmp_path / "key.json").write_text(json.dumps({
        "one": {"run": "test", "case_id": "c", "turn": 1, "judge": {"hes": 0, "helpful": 2}},
        "two": {"run": "test", "case_id": "c", "turn": 2, "judge": {"hes": 0, "helpful": 1}},
    }))
    (tmp_path / filename).write_text(json.dumps({
        "one": {"hes": None, "helpful": 0}, "two": {"hes": 0, "helpful": 1},
    }))


def test_ai_report_preserves_identity_abstentions_and_coverage(tmp_path, monkeypatch, capsys):
    fixture_files(tmp_path)
    monkeypatch.setattr(report, "VARIANTS", {"original": None})
    report.main(str(tmp_path), "labels.codex-ai.json", "ai")
    text = capsys.readouterr().out
    assert "AI-to-AI agreement only" in text
    assert '| helpful | 50%' in text
    assert '| hes | 100% / - (1)' in text
    assert '"hes": 1' in text
    assert "No automatic winner" in text
    assert "Human-labelled" not in text


def test_ai_filename_cannot_default_to_human(tmp_path):
    fixture_files(tmp_path)
    with pytest.raises(ValueError, match="reference-kind ai"):
        report.main(str(tmp_path), "labels.codex-ai.json")


def test_unknown_items_fail_instead_of_silently_disappearing(tmp_path):
    fixture_files(tmp_path, "labels.json")
    (tmp_path / "labels.json").write_text('{"unmapped": {"hes": 0}}')
    with pytest.raises(ValueError, match="Unknown label IDs"):
        report.main(str(tmp_path))
