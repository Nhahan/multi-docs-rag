#!/usr/bin/env bash
set -euo pipefail

missing=0

if ! command -v uv >/dev/null 2>&1; then
  echo "[MISSING] uv (required)"
  missing=1
else
  echo "[OK] uv"
fi

if ! command -v uvx >/dev/null 2>&1; then
  echo "[MISSING] uvx (required for direct execution path)"
  missing=1
else
  echo "[OK] uvx"
fi

if command -v git >/dev/null 2>&1; then
  echo "[OK] git"
else
  echo "[MISSING] git (required for optional local source checkout)"
  missing=1
fi

if [ $missing -ne 0 ]; then
  echo
  echo "Some prerequisites are missing. Install first, then run: npm run ouroboros:install"
  exit 1
fi

echo
echo "Ouroboros install check completed."
