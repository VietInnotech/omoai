#!/usr/bin/env bash
set -euo pipefail

# Print the fully resolved OMOAI configuration (YAML after env merges)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE%/scripts}"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
from src.omoai.config.schemas import get_config

cfg = get_config()
print(cfg.model_dump_yaml(sort_keys=False))
PY

