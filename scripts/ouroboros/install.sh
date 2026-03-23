#!/usr/bin/env bash
set -euo pipefail

echo "Checking ouroboros-ai install prerequisites..."

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Installing Ouroboros CLI via uvx..."
uv tool install ouroboros-ai

echo "Installed. Verify with: uvx --from ouroboros-ai ouroboros --help"
