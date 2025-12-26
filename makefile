# --- Configuration ---
# Your Python interpreter (uv handles the venv automatically)
PYTHON_CMD = uv run python

# File paths
INFERENCE_SCRIPT = src/inference.py
TRAIN_SCRIPT = src/train.py
CHECKPOINT_DIR = checkpoints

# --- Targets ---

# Default target (what runs if you just type 'make')
.PHONY: all
all: setup inference

# 1. Setup: Installs dependencies using uv
.PHONY: setup
setup:
	@echo ">>> Installing dependencies with uv..."
	uv sync

# 2. Inference: Runs your LLM
.PHONY: inference
inference:
	@echo ">>> Running inference..."
	$(PYTHON_CMD) $(INFERENCE_SCRIPT)

.PHONY: train
train:
	@echo ">> Starting training..."
	$(PYTHON_CMD) $(TRAIN_SCRIPT) 
