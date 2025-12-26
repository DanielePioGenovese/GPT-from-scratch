# --- Configuration ---
# Your Python interpreter (uv handles the venv automatically)
PYTHON_CMD = uv run python

# File paths
INFERENCE_SCRIPT = src/inference.py
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

# 2. Download Weights: (Optional helper to fetch from HuggingFace)
# You can use the huggingface-cli or a python script to do this.
.PHONY: download-model
download-model:
	@echo ">>> Downloading model from Hugging Face..."
	# Example command:
	# uv run huggingface-cli download <your-username>/<your-repo> --local-dir $(CHECKPOINT_DIR)

# 3. Inference: Runs your LLM
.PHONY: inference
inference:
	@echo ">>> Running inference..."
	$(PYTHON_CMD) $(INFERENCE_SCRIPT)

# 4. Export: Creates a requirements.txt for non-uv users (Compatibility)
.PHONY: export
export:
	@echo ">>> Generating requirements.txt for legacy pip users..."
	uv export --format requirements-txt > requirements.txt

# 5. Clean: Removes the environment and cache
.PHONY: clean
clean:
	@echo ">>> Cleaning project..."
	rm -rf .venv
	rm -rf __pycache__