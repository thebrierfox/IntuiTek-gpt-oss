#!/bin/bash
# bootstrap.sh - High-level script to set up GPT-OSS environment and run a test inference.
# This script:
# 1. Checks that Python 3.10+ is installed.
# 2. Installs required Python dependencies.
# 3. Downloads GPT-OSS models using setup_gpt_oss.py.
# 4. Runs a sample inference using example_inference.py.

set -e

# 1. Check Python version
PYTHON_CMD=${PYTHON_CMD:-python3}
PY_VER=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED="3.10"
if [ "$(printf '%s\n' "$REQUIRED" "$PY_VER" | sort -V | head -n1)" != "$REQUIRED" ]; then
  echo "Python $REQUIRED or newer required. Current: $PY_VER"
  exit 1
fi

# 2. Install dependencies
echo "Installing Python dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  $PYTHON_CMD -m pip install -r requirements.txt
fi
$PYTHON_CMD -m pip install huggingface_hub==0.20.0

# 3. Download models
echo "Running setup script to download models..."
export GPT_OSS_DOWNLOAD_ALL=${GPT_OSS_DOWNLOAD_ALL:-0}
MODEL_DIR="models"
$PYTHON_CMD setup_gpt_oss.py --target_dir "$MODEL_DIR"

# 4. Run sample inference
echo "Running sample inference..."
PROMPT=${PROMPT:-"Explain the significance of the open-source GPT-OSS models in accessible terms."}
REASONING_LEVEL=${REASONING_LEVEL:-medium}
# Determine first model directory if available
if [ -d "$MODEL_DIR" ]; then
  MODEL_NAME=$(ls -1 "$MODEL_DIR" | head -n1)
  MODEL_ID="$MODEL_DIR/$MODEL_NAME"
else
  MODEL_ID="gpt-oss-20b" # fallback to default model id
fi
$PYTHON_CMD example_inference.py --model_id "$MODEL_ID" --prompt "$PROMPT" --reasoning_level "$REASONING_LEVEL" --max_tokens 512
