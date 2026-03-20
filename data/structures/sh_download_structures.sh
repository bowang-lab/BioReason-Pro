#!/bin/bash

# Usage: bash sh_download_structures.sh --cache-dir /path/to/cache [options]
# All arguments are forwarded to download_structures.py
# --cache-dir is required

python3 download_structures.py "$@"
