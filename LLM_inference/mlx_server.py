#!/usr/bin/env python3
"""
MLX-LM Server - OpenAI-compatible API server for MLX models

This script starts a local server that provides an OpenAI-compatible API
for MLX models. This is an alternative to Ollama for running LLMs on Mac.

Usage:
    python mlx_server.py                                    # Use default model
    python mlx_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python mlx_server.py --port 8080
"""

import argparse
import sys


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import mlx
    except ImportError:
        missing.append("mlx")

    try:
        import mlx_lm
    except ImportError:
        missing.append("mlx-lm")

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install with:")
        print("   pip install mlx mlx-lm")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Start MLX-LM OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python mlx_server.py
    python mlx_server.py --model mlx-community/Llama-3.3-70B-Instruct-4bit
    python mlx_server.py --port 8080 --host 0.0.0.0
        """,
    )

    # Import config for defaults
    try:
        from config import MLX_MODEL

        default_model = MLX_MODEL
    except ImportError:
        default_model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model to serve (default: {default_model})",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return 1

    print(f"🚀 Starting MLX-LM server...")
    print(f"📦 Model: {args.model}")
    print(f"🌐 Host: {args.host}:{args.port}")
    print(f"🔗 API URL: http://{args.host}:{args.port}/v1")
    print()

    # Try to use mlx_lm.server if available
    try:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")

        print(f"💡 Running: {' '.join(cmd)}")
        print("-" * 60)
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Alternative: Use Ollama instead:")
        print("   brew install ollama")
        print("   ollama serve")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
