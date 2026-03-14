#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo ".venv not found. Create it first." >&2
  exit 1
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install py2app

rm -rf build dist

PYPROJECT_PATH="$ROOT_DIR/pyproject.toml"
PYPROJECT_BAK_PATH="$ROOT_DIR/pyproject.toml.py2app.bak"

restore_pyproject() {
  if [[ -f "$PYPROJECT_BAK_PATH" ]]; then
    mv "$PYPROJECT_BAK_PATH" "$PYPROJECT_PATH"
  fi
}

trap restore_pyproject EXIT

if [[ -f "$PYPROJECT_PATH" ]]; then
  mv "$PYPROJECT_PATH" "$PYPROJECT_BAK_PATH"
fi

python setup_py2app.py py2app -A

echo ""
echo "Built app bundle:"
echo "  $ROOT_DIR/dist/JalLijiye.app"
echo "Note: this is an alias app bundle and depends on this repo and .venv staying in place."
