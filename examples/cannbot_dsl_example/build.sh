#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

rm -rf build dist cann_bench.egg-info

python setup.py bdist_wheel

test -n "$(find dist -maxdepth 1 -type f -name 'cann_bench*.whl' -print -quit)"

echo "=== Build complete ==="
ls -la dist/
