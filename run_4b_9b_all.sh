#!/usr/bin/env bash
# Run all 4B/9B experiments with the paper protocol and verify the results.
# See reproduce_4b_9b.py --help for options.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/home/suzeqin/miniconda3/envs/sp/bin/python}
exec "${PY}" reproduce_4b_9b.py "$@"
