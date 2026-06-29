"""
agent_loop.py — Three-phase cognitive loop for the State-Driven Developer Agent.

Implements the deterministic state machine that structures learning,
breakthroughs, and failures into the Cognee knowledge graph.

Optimized to enforce absolute string token density for local CPU runtimes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import textwrap
from typing import Any, Optional

from state_schemas import (
    ExecutionHistory,
    ExecutionStatus,
    FailureEdge,
    GoalState,
    KnowledgePoint,
    SolutionObjectState,
)

import cognee_memory as memory

logger = logging.getLogger("agent_loop")


# ═══════════════════════════════════════════════════════════════════════
# LLM Interface — Uses Ollama via litellm (Cognee's internal dep)
# ═══════════════════════════════════════════════════════════════════════

async def llm_generate(prompt: str, system_prompt: str = "") -> str:
    """
    Generate text from the local Ollama LLM.
    Tuned with tight token constraints for low-overhead CPU parsing.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    model = os.getenv("LLM_MODEL", "qwen2.5:3b")
    endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")

    try:
        import litellm
        response = await litellm.acompletion(
            model=f"ollama/{model}",
            messages=[
                {"role": "system", "content": system_prompt or "You are a precise technical assistant."},
                {"role": "user", "content": prompt},
            ],
            api_base=endpoint.rstrip("/v1"),
            timeout=90, # Prevent indefinite thread locking on slow CPU cycles
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"[LLM ERROR: {e}]"


async def llm_generate_json(prompt: str, system_prompt: str = "") -> dict[str, Any]:
    """Generate structured JSON from the LLM, with strict cleaning fallbacks."""
    raw = await llm_generate(prompt, system_prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Clean markdown formatting blocks if the model ignores the systemic directive
    import re
    json_match = re.search(r"
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1
http://googleusercontent.com/immersive_entry_chip/2
