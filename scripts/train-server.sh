#!/usr/bin/env bash
set -euo pipefail
cd /workspace/A2A_pushT
exec .venv/bin/python -u -m pusht_a2a.train "$@"
