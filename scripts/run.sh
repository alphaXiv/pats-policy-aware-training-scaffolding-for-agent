#!/usr/bin/env bash
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential libgl1
python -m pip install --disable-pip-version-check -r requirements.txt
if [ ! -f "${ALFWORLD_DATA}/logic/alfred.pddl" ]; then
  alfworld-download
fi
python run_reproduction.py
