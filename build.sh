#!/usr/bin/env bash
# Exit on error
set -o errexit

# Ensure uv is available
pip install --upgrade uv

# Install lightweight CPU-only PyTorch first using uv (keeps memory usage well under 512MB)
uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of your app requirements using uv
uv pip install --system --no-cache-dir -r requirements.txt