#!/bin/bash
# Setup script for LLM inference on Mac Studio
# This script installs and configures Ollama for local LLM inference

set -e

echo "=============================================="
echo "  LLM Inference Setup for Mac Studio"
echo "=============================================="
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is designed for macOS (Mac Studio)"
    exit 1
fi

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (M1/M2/M3)"
    echo "   Performance may vary on Intel Macs"
fi

echo "📦 Step 1: Installing Ollama..."
echo "-----------------------------------------"

if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
    ollama --version
else
    if command -v brew &> /dev/null; then
        echo "Installing Ollama via Homebrew..."
        brew install ollama
    else
        echo "Homebrew not found. Installing Ollama via curl..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

echo ""
echo "📦 Step 2: Starting Ollama server..."
echo "-----------------------------------------"

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server is already running"
else
    echo "Starting Ollama server in background..."
    # Start Ollama as a background service
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server started successfully"
    else
        echo "⚠️  Ollama server may not have started. Check /tmp/ollama.log"
        echo "   You can also run 'ollama serve' manually in another terminal"
    fi
fi

echo ""
echo "📦 Step 3: Pulling recommended models..."
echo "-----------------------------------------"

# Small model for testing
echo "Pulling llama3.2:3b (small model for testing)..."
ollama pull llama3.2:3b

echo ""
echo "=============================================="
echo "  Setup Complete! ✅"
echo "=============================================="
echo ""
echo "Available models:"
ollama list
echo ""
echo "Quick start:"
echo "  1. Test the setup:  python test_inference.py"
echo "  2. Run benchmark:   python test_inference.py --benchmark"
echo "  3. Use custom prompt: python test_inference.py --prompt 'Your question'"
echo ""
echo "To pull larger models for 90GB Mac Studio:"
echo "  ollama pull llama3.3:70b      # Best quality, ~40GB"
echo "  ollama pull qwen2.5:72b       # Strong multilingual, ~40GB"
echo "  ollama pull deepseek-r1:70b   # Reasoning focused, ~40GB"
echo "  ollama pull mixtral:8x7b      # Fast MoE model, ~26GB"
echo ""
echo "Then update config.py to use the new model."
