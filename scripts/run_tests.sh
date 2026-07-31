#!/usr/bin/env bash
set -eu

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" -m unittest discover -s tests -v
