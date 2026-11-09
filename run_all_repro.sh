#!/usr/bin/env bash
# One-click reproduction into results2/ (Qwen 4B/9B strict + MiniCPM).
# See reproduce_all.py --help for options.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}
exec "${PY}" reproduce_all.py "$@"
