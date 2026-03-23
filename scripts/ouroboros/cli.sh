#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
  echo "Usage: npm run ouroboros -- <ooo-command>"
  echo "Examples:"
  echo "  npm run ouroboros -- setup"
  echo "  npm run ouroboros -- init \"Build multi-doc RAG with local LLMs\""
  echo "  npm run ouroboros -- tui monitor"
  echo "  npm run ouroboros -- --help"
  exit 0
fi

if [ "$1" = "setup" ]; then
  echo "Ouroboros setup: initializing configuration for this environment."
  uvx --from ouroboros-ai ouroboros config init
  echo "Setup done. Next: run interview/init as needed."
  exit 0
fi

uvx --from ouroboros-ai ouroboros "$@"
