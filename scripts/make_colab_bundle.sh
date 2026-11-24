#!/usr/bin/env bash
# Package code + data (no private notes, no third_party, no runs) for upload to Colab.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/colab_bundle.zip
python -I - <<'PY'
import zipfile, pathlib
root = pathlib.Path(".")
include = ["pyproject.toml", "src", "scripts", "data", "prompts", "tests", "inspect_tasks"]
with zipfile.ZipFile("dist/colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in include:
        p = root / item
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
        for f in files:
            z.write(f, f.as_posix())
print("wrote dist/colab_bundle.zip with", len(zipfile.ZipFile("dist/colab_bundle.zip").namelist()), "files")
PY
