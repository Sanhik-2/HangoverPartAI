"""
main.py — CLI REPL entry point for the State-Driven Developer Agent.

Provides an interactive loop that accepts user prompts and delegates
to the 3-phase cognitive loop, pretty-printing state transitions
and Cognee query results.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment before any Cognee imports to prevent default settings override
load_dotenv()

from agent_loop import run_cognitive_loop
from cognee_memory import initialize_memory, reset_memory

# ─── Logging Setup ────────────────────────────────────────────────────


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for the agent."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-15s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down external library warnings unless verbose debugging is enabled
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("cognee").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# ─── Banner ───────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗████████╗ █████╗ ████████╗███████╗                ║
║   ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝                ║
║   ███████╗   ██║   ███████║   ██║   █████╗                  ║
║   ╚════██║   ██║   ██╔══██║   ██║   ██╔══╝                  ║
║   ███████║   ██║   ██║  ██║   ██║   ███████╗                ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝                ║
║                                                              ║
║   State-Driven Developer Agent                               ║
║   Cognee Knowledge Graph + Ollama LLM                        ║
║                                                              ║
║   Commands:                                                  ║
║     /reset   — Wipe all memory and start fresh               ║
║     /status  — Show current Cognee status                    ║
║     /verbose — Toggle verbose logging                        ║
║     /quit    — Exit the agent                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─── REPL Loop ────────────────────────────────────────────────────────


async def repl() -> None:
    """Interactive REPL loop for the agent."""
    verbose = False
    setup_logging(verbose)

    print(BANNER)

    # Check Ollama connectivity
    print("  Checking Ollama connectivity...")
    try:
        import httpx

        # We target the root Ollama port dynamically derived from environmental endpoint variable
        ollama_base = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1").replace(
            "/v1", ""
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{ollama_base}/api/tags",
                timeout=5.0,
            )
            models = resp.json().get("models", [])
            if models:
                print(f"  ✓ Ollama connected — {len(models)} model(s) available:")
                for m in models:
                    size_gb = m.get("size", 0) / (1024**3)
                    print(f"    • {m['name']} ({size_gb:.2f} GB)")
            else:
                print("  ⚠ Ollama connected but no model signatures found!")
                print("    Recommended for your system profile (8GB RAM):")
                print("    Run: ollama pull qwen2.5:3b && ollama pull nomic-embed-text")
    except Exception as e:
        print(f"  ❌ Cannot reach Ollama execution context: {e}")
        print("    Please ensure Ollama is active. Run: ollama serve")
        return

    # Initialize Cognee
    print("\n  Initializing Cognee memory layer...")
    await initialize_memory()
    print("  ✓ Memory layer ready.\n")

    print("  Enter your prompts below. The agent will process each through")
    print("  the optimized, state-driven 3-phase cognitive loop.\n")

    # Main Command / Input REPL Loop
    while True:
        try:
            user_input = input("  agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting safely. Goodbye.\n")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/quit", "/exit"):
                print("\n  Exiting safely. Goodbye.\n")
                break

            elif cmd == "/reset":
                confirm = input(
                    "  ⚠ This will delete ALL database and graph index partitions. Confirm? (yes/no): "
                )
                if confirm.strip().lower() == "yes":
                    await reset_memory()
                    print("  ✓ Memory wiped successfully.\n")
                else:
                    print("  Cancelled.\n")

            elif cmd == "/status":
                print("\n  ── Cognee Status ──")
                print(
                    f"  LLM Provider:   {os.getenv('LLM_PROVIDER', 'ollama (local)')}"
                )
                print(
                    f"  LLM Model:      {os.getenv('LLM_MODEL', 'qwen2.5:3b (optimized)')}"
                )
                print(
                    f"  LLM Endpoint:   {os.getenv('LLM_ENDPOINT', 'http://localhost:11434/v1')}"
                )
                print(
                    f"  Embed Model:    {os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')}"
                )
                print(
                    f"  Embed Endpoint: {os.getenv('EMBEDDING_ENDPOINT', 'http://localhost:11434/v1')}"
                )
                print()

            elif cmd == "/verbose":
                verbose = not verbose
                setup_logging(verbose)
                print(f"  Verbose logging: {'ON' if verbose else 'OFF'}\n")

            else:
                print(f"  Unknown command: {user_input}")
                print("  Available options: /reset, /status, /verbose, /quit\n")

            continue

        # ── Execute cognitive loop ──
        try:
            state = await run_cognitive_loop(user_input)
        except Exception as e:
            logging.getLogger("agent_loop").error(f"Loop failed: {e}", exc_info=True)
            print(f"\n  ❌ Agent loop collapsed: {e}")
            print(
                "  The diagnostic error traces have been logged. Try mutating your prompt parameters.\n"
            )


# ─── Entry Point ──────────────────────────────────────────────────────


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        print("\n\n  Exiting safely. Goodbye.\n")


if __name__ == "__main__":
    main()
