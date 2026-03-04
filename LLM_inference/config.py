"""
Configuration for LLM Inference on Mac Studio

This file contains all configurable parameters for running LLM inference.
Modify these settings to experiment with different models and parameters.
"""

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Choose backend: "ollama" or "mlx"
BACKEND = "ollama"

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

# Ollama API endpoint (default local server)
OLLAMA_BASE_URL = "http://localhost:11434"

# Model to use with Ollama
# Popular options for 90GB Mac Studio:
#   - "llama3.3:70b"      # Latest Llama, best quality (~40GB)
#   - "llama3.1:70b"      # Stable Llama 70B (~40GB)
#   - "qwen2.5:72b"       # Strong multilingual (~40GB)
#   - "deepseek-r1:70b"   # Reasoning focused (~40GB)
#   - "mixtral:8x7b"      # Fast MoE model (~26GB)
#   - "llama3.2:3b"       # Fast for testing (~2GB)
#   - "llama3.2:1b"       # Very fast for testing (~1GB)
OLLAMA_MODEL = "llama3.2:3b"

# =============================================================================
# MLX-LM CONFIGURATION
# =============================================================================

# MLX-LM server endpoint
MLX_BASE_URL = "http://localhost:8080"

# Model to use with MLX-LM (HuggingFace model ID or local path)
# Popular options:
#   - "mlx-community/Llama-3.3-70B-Instruct-4bit"
#   - "mlx-community/Llama-3.1-70B-Instruct-4bit"
#   - "mlx-community/Qwen2.5-72B-Instruct-4bit"
#   - "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
#   - "mlx-community/Llama-3.2-3B-Instruct-4bit"
MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

# Maximum tokens to generate
MAX_TOKENS = 512

# Temperature (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.7

# Top-p (nucleus sampling)
TOP_P = 0.9

# Seed for reproducibility (None for random)
SEED = None

# =============================================================================
# TEST PROMPTS
# =============================================================================

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about artificial intelligence.",
]

# Benchmark prompt for consistent timing measurements
BENCHMARK_PROMPT = """You are a helpful medical assistant. Please provide a detailed explanation of the following medical concept:

What is the difference between Type 1 and Type 2 diabetes? Include information about:
1. Causes
2. Symptoms
3. Treatment options
4. Long-term management

Please be thorough but concise."""
