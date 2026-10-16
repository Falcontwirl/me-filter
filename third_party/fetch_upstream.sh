#!/bin/bash
# Fetch the authors' code (github.com/arnab-api/filter) at the pinned commit.
# It is not redistributed in this repo (upstream has no license), so each checkout runs this once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/filter"
COMMIT=d4a3010b391eda4b60f824e37c63335bb8940b7f
if [ ! -d "$DEST/.git" ]; then
  rm -rf "$DEST"
  git clone -q https://github.com/arnab-api/filter.git "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"
cp "$HERE/env.yml" "$DEST/env.yml"
echo "upstream ready at $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
