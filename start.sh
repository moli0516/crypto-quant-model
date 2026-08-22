#!/usr/bin/env bash
# 本地快捷啟動腳本

set -e
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python cli.py "$@"
