import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grade_batches  # noqa: E402


def _run(tmp_path, name, suite="grounded-risk"):
    run = tmp_path / name
    run.mkdir()
    turns = []
    for n in (1, 2, 3):
        scores = {"overreact": {"score": 0, "raw": "0"}, "helpful": {"score": 2, "raw": "2"}}
        if n >= 2:
            scores["help"] = {"score": 1, "raw": "1"}
        turns.append({"turn": n, "user": f"u{n}", "assistant": f"a{n}", "scores": scores})
    rec = {"case_id": "c1", "suite": suite, "n_prompts": 3, "status": "ok", "turns": turns}
    (run / "conversations.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"suite": suite}), encoding="utf-8")
    return run


def test_make_and_apply_round_trip(tmp_path):
    runs = [_run(tmp_path, "base"), _run(tmp_path, "p1")]
    out = tmp_path / "grading"
    grade_batches.make([str(r) for r in runs], out, batch_size=4, dup_frac=0.5, seed=1)
    key = json.loads((out / "key.json").read_text())
    batches = sorted(out.glob("batch_*"))
    assert len(key) == 9 and len(batches) == 3  # 6 originals + 3 duplicates
    items = [i for b in batches for i in json.loads((b / "items.json").read_text())]
    assert all("run" not in i and "condition" not in i for i in items)  # blinded
    assert {tuple(i["questions"]) for i in items} <= {("helpfulness",), ("helpfulness", "help")}

    # Grade everything as helpfulness 1 / help 0, except leave one original ungraded.
    originals = [k for k, v in key.items() if v["dup_of"] is None]
    skipped = next(k for k in originals if k not in {v["dup_of"] for v in key.values()})
    for b in batches:
        labels = {i["id"]: {"helpfulness": 1, **({"help": 0} if "help" in i["questions"] else {})}
                  for i in json.loads((b / "items.json").read_text()) if i["id"] != skipped}
        (b / grade_batches.LABELS_FILE).write_text(json.dumps(labels))
    grade_batches.apply(out)

    for r in runs:
        graded = tmp_path / (r.name + "-graded")
        rec = json.loads((graded / "conversations.jsonl").read_text().splitlines()[0])
        assert json.loads((graded / "manifest.json").read_text())["graded_metrics"]["help"]["grader"]
        for t in rec["turns"]:
            assert t["scores"]["overreact"]["score"] == 0  # untouched
            assert t["scores"]["helpful"]["score"] in (1, None)  # replaced; missing stays None, never 0
            if "help" in t["scores"]:
                assert t["scores"]["help"]["score"] in (0, None)
    missing = [t for r in runs for t in json.loads((tmp_path / (r.name + "-graded") / "conversations.jsonl")
               .read_text().splitlines()[0])["turns"] if t["scores"]["helpful"]["score"] is None]
    assert len(missing) == 1
