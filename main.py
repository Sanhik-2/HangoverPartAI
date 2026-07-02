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
║   Cognee Knowledge Graph + Groq Cloud LLM                    ║
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

    import httpx

    # ── Check Cloud LLM connectivity ──
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_endpoint = os.getenv("LLM_ENDPOINT", "https://api.groq.com/openai/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    # Detect Groq from endpoint URL (provider is labeled 'openai' for Cognee compatibility)
    is_groq = "groq.com" in llm_endpoint
    provider_display = "GROQ" if is_groq else llm_provider.upper()

    print(f"  Checking {provider_display} LLM connectivity...")
    if is_groq:
        if not llm_api_key or llm_api_key == "YOUR_GROQ_API_KEY_HERE":
            print("  ❌ Groq API key not configured!")
            print("    1. Sign up at https://console.groq.com (free, no credit card)")
            print("    2. Generate an API key")
            print("    3. Paste it into .env as LLM_API_KEY=gsk_...")
            return
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{llm_endpoint.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {llm_api_key}"},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    model_ids = [m.get("id", "") for m in models]
                    if llm_model in model_ids:
                        print(f"  ✓ {provider_display} connected — model '{llm_model}' available")
                    else:
                        print(f"  ✓ {provider_display} connected — {len(models)} model(s) available")
                        print(f"    ⚠ Configured model '{llm_model}' not found in list")
                else:
                    print(f"  ❌ {provider_display} API returned status {resp.status_code}")
                    print(f"    Check your API key in .env")
                    return
        except Exception as e:
            print(f"  ❌ Cannot reach {provider_display} API: {e}")
            return
    else:
        print(f"  ℹ Using provider '{llm_provider}' — skipping Groq-specific check")

    # ── Check Ollama connectivity (needed for embeddings) ──
    print("  Checking Ollama connectivity (embeddings)...")
    try:
        # Use EMBEDDING_ENDPOINT for Ollama check — it always points to local Ollama.
        # Strip any /api/* path to get the base Ollama URL for the health check.
        embedding_ep = os.getenv("EMBEDDING_ENDPOINT", "http://localhost:11434")
        # Extract base URL: http://localhost:11434/api/embed -> http://localhost:11434
        if "/api/" in embedding_ep:
            ollama_base = embedding_ep[:embedding_ep.index("/api/")]
        else:
            ollama_base = embedding_ep.rstrip("/")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{ollama_base}/api/tags",
                timeout=5.0,
            )
            models = resp.json().get("models", [])
            embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            model_names = [m.get("name", "") for m in models]
            if any(embed_model in name for name in model_names):
                print(f"  ✓ Ollama connected — embedding model '{embed_model}' available")
            elif models:
                print(f"  ✓ Ollama connected — {len(models)} model(s) available")
                print(f"    ⚠ Embedding model '{embed_model}' not found. Run: ollama pull {embed_model}")
            else:
                print("  ⚠ Ollama connected but no models found.")
                print(f"    Run: ollama pull {embed_model}")
    except Exception as e:
        print(f"  ❌ Cannot reach Ollama (needed for embeddings): {e}")
        print("    Please ensure Ollama is running: ollama serve")
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
                print("\n  ── Agent Status ──")
                _provider = os.getenv('LLM_PROVIDER', 'openai')
                _endpoint = os.getenv('LLM_ENDPOINT', '')
                _display = "GROQ" if "groq.com" in _endpoint else _provider.upper()
                print(f"  LLM Provider:   {_display}")
                print(
                    f"  LLM Model:      {os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')}"
                )
                print(
                    f"  LLM Endpoint:   {os.getenv('LLM_ENDPOINT', 'https://api.groq.com/openai/v1')}"
                )
                api_key = os.getenv('LLM_API_KEY', '')
                key_display = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else '(not set)'
                print(f"  API Key:        {key_display}")
                print(
                    f"  Embed Model:    {os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')} (local Ollama)"
                )
                print(
                    f"  Embed Endpoint: {os.getenv('EMBEDDING_ENDPOINT', 'http://localhost:11434/api/embed')}"
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
