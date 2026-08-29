#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Resolve data/ symlinks into real files (Render's git clone
# may not preserve symlinks depending on the OS/config).
echo "Resolving data/ symlinks..."
for link in data/*.csv; do
  if [ -L "$link" ]; then
    target=$(readlink "$link")
    cp --remove-destination "$target" "$link" 2>/dev/null \
      || (rm "$link" && cp "$(dirname "$link")/$target" "$link")
    echo "  resolved $link -> real file"
  fi
done

# Ensure audit directory exists and is writable
mkdir -p audit
echo "Build complete ✓"
