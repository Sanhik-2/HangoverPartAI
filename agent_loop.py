"""
agent_loop.py — Three-phase cognitive loop for the State-Driven Developer Agent.

Implements the deterministic state machine that structures learning,
breakthroughs, and failures into the Cognee knowledge graph.

Phase 1: Intent Deconstruction & Goal Serialization
Phase 2: Hierarchical Object Mapping
Phase 3: Runtime Execution with FAILED_BY edge recording

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
# LLM Interface — Uses litellm for provider-agnostic model routing
# ═══════════════════════════════════════════════════════════════════════

async def llm_generate(prompt: str, system_prompt: str = "") -> str:
    """
    Generate text from the configured LLM provider.
    Supports both local Ollama and cloud providers via LLM_PROVIDER env var.
    Tuned with tight token constraints for low-overhead CPU parsing.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b")
    endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "")

    try:
        import litellm

        # Build kwargs based on provider
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt or "You are a precise technical assistant."},
                {"role": "user", "content": prompt},
            ],
            "timeout": 90,
        }

        if provider == "ollama":
            # Ollama routing: prefix model and set api_base
            kwargs["model"] = f"ollama/{model}" if not model.startswith("ollama/") else model
            kwargs["api_base"] = endpoint.rstrip("/v1").rstrip("/")
        else:
            # Cloud provider (openai, gemini via OpenAI gateway, etc.)
            kwargs["model"] = model
            kwargs["api_base"] = endpoint.rstrip("/")
            if api_key:
                kwargs["api_key"] = api_key

        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return f"[LLM ERROR: {e}]"


async def llm_generate_json(prompt: str, system_prompt: str = "") -> dict[str, Any]:
    """Generate structured JSON from the LLM, with strict cleaning fallbacks."""
    raw = await llm_generate(prompt, system_prompt)

    # Attempt direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Clean markdown formatting blocks if the model ignores the systemic directive
    import re
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find any JSON-like block bounded by { }
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM output: {raw[:200]}")
    return {"error": "Failed to parse LLM response", "raw": raw[:500]}


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Intent Deconstruction & Goal Serialization
# ═══════════════════════════════════════════════════════════════════════

async def phase_1_intent_deconstruction(user_prompt: str) -> GoalState:
    """
    Deconstruct the user's prompt into a structured GoalState.
    Query Cognee for similar historical goals to prevent redundant exploration.
    """
    logger.info("═══ Phase 1: Intent Deconstruction ═══")

    system_prompt = textwrap.dedent("""\
        You are a goal decomposition engine. Given a user prompt, extract:
        1. A precise technical goal name (short title)
        2. A list of preparation steps (ordered execution strategies)
        3. Understanding notes (architectural constraints and background)

        Respond ONLY with valid JSON in this exact format:
        {
            "goal_name": "...",
            "preparation_steps": ["step1", "step2"],
            "understanding_notes": ["note1", "note2"]
        }
    """)

    result = await llm_generate_json(
        f"Decompose this prompt into a structured goal:\n\n{user_prompt}",
        system_prompt,
    )

    goal = GoalState(
        goal_name=result.get("goal_name", user_prompt[:80]),
        preparation_steps=result.get("preparation_steps", []),
        understanding_notes=result.get("understanding_notes", []),
    )

    logger.info(f"  Goal created: {goal.goal_name} ({goal.goal_id})")

    # Query Cognee for similar historical goals
    similar = await memory.memory.query_similar_goals(goal)
    if similar:
        logger.info(f"  Found {len(similar)} similar historical goal(s):")
        for s in similar[:3]:
            logger.info(f"    • {s}")
    else:
        logger.info("  No similar historical goals found — new exploration path.")

    # Persist goal to graph
    await memory.memory.store_goal(goal)

    return goal


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Hierarchical Object Mapping
# ═══════════════════════════════════════════════════════════════════════

async def phase_2_object_mapping(goal: GoalState) -> SolutionObjectState:
    """
    Map the goal into a SolutionObjectState with knowledge points.
    Check ancestor constraints to prevent conflicting solution branches.
    """
    logger.info("═══ Phase 2: Hierarchical Object Mapping ═══")

    system_prompt = textwrap.dedent("""\
        You are a solution architect. Given a goal, create a solution plan with:
        1. A descriptive state name
        2. Knowledge points — facts, invariants, or formulas needed
        3. An execution plan (code or command to run)

        Respond ONLY with valid JSON:
        {
            "state_name": "...",
            "knowledge_points": [
                {"topic": "...", "description": "...", "is_verified": false}
            ],
            "execution_plan": "..."
        }
    """)

    goal_text = goal.to_cognee_text()
    result = await llm_generate_json(
        f"Create a solution plan for this goal:\n\n{goal_text}",
        system_prompt,
    )

    # Build knowledge points from LLM output
    kp_data = result.get("knowledge_points", [])
    knowledge_points = []
    for kp in kp_data:
        if isinstance(kp, dict):
            knowledge_points.append(KnowledgePoint(
                topic=kp.get("topic", "unknown"),
                description=kp.get("description", ""),
                is_verified=kp.get("is_verified", False),
            ))

    state = SolutionObjectState(
        state_name=result.get("state_name", f"Solution for {goal.goal_name}"),
        goal_id=goal.goal_id,
        knowledge_points=knowledge_points,
    )

    logger.info(f"  Solution state: {state.state_name} ({state.state_id})")
    logger.info(f"  Knowledge points: {len(state.knowledge_points)}")

    # Check ancestor constraints if this node has a parent
    if state.parent_state_id:
        ancestors = await memory.memory.query_ancestors(state.parent_state_id)
        if ancestors:
            logger.info(f"  Checking {len(ancestors)} ancestor constraint(s)...")

    # Check for prior failures on similar paths
    failure_history = await memory.memory.get_failure_history(state.state_id)
    if failure_history:
        logger.warning(
            f"  ⚠ Found {len(failure_history)} prior failure(s) on similar paths. "
            "Adjusting solution trajectory."
        )

    # Store the execution plan for Phase 3
    state._execution_plan = result.get("execution_plan", "")

    # Persist solution state to graph
    await memory.memory.store_state(state)

    return state


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Runtime Execution & Failure Recording
# ═══════════════════════════════════════════════════════════════════════

async def phase_3_runtime_execution(state: SolutionObjectState) -> SolutionObjectState:
    """
    Execute the solution plan. Record failures as FAILED_BY edges
    to prevent re-traversal of dead-end paths.
    """
    logger.info("═══ Phase 3: Runtime Execution ═══")

    execution_plan = getattr(state, "_execution_plan", "")

    if not execution_plan:
        logger.info("  No executable plan — generating analysis response.")
        state.execution_history = ExecutionHistory(
            last_status=ExecutionStatus.SUCCESS,
        )
        # Mark knowledge points as verified for analysis-only tasks
        for kp in state.knowledge_points:
            kp.is_verified = True
        await memory.memory.store_state(state)
        return state

    # Attempt execution
    logger.info(f"  Executing plan: {execution_plan[:100]}...")

    try:
        # Run the execution plan as a subprocess with timeout
        proc = await asyncio.create_subprocess_shell(
            execution_plan,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=60,
        )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            # ── SUCCESS ──
            logger.info("  ✓ Execution succeeded.")
            if stdout_text:
                logger.info(f"  Output: {stdout_text[:200]}")

            state.execution_history = ExecutionHistory(
                last_status=ExecutionStatus.SUCCESS,
            )
            # Mark knowledge points as verified on successful execution
            for kp in state.knowledge_points:
                kp.is_verified = True
        else:
            # ── FAILED ──
            error_msg = stderr_text or f"Exit code {proc.returncode}"
            logger.error(f"  ✗ Execution failed: {error_msg[:200]}")

            state.execution_history = ExecutionHistory(
                last_status=ExecutionStatus.FAILED,
                error_log=error_msg,
            )

            # Record failure edge to prevent dead-end re-traversal
            await memory.memory.record_failure(
                source_state_id=state.state_id,
                error_message=error_msg,
                failed_code=execution_plan,
                root_cause=await _analyze_failure(error_msg, execution_plan),
            )

    except asyncio.TimeoutError:
        error_msg = "Execution timed out after 60 seconds"
        logger.error(f"  ✗ {error_msg}")

        state.execution_history = ExecutionHistory(
            last_status=ExecutionStatus.FAILED,
            error_log=error_msg,
        )
        await memory.memory.record_failure(
            source_state_id=state.state_id,
            error_message=error_msg,
            failed_code=execution_plan,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"  ✗ Execution error: {error_msg}")

        state.execution_history = ExecutionHistory(
            last_status=ExecutionStatus.FAILED,
            error_log=error_msg,
        )
        await memory.memory.record_failure(
            source_state_id=state.state_id,
            error_message=error_msg,
            failed_code=execution_plan,
        )

    # Persist final state to graph
    await memory.memory.store_state(state)
    return state


async def _analyze_failure(error_msg: str, code: str) -> Optional[str]:
    """Use the LLM to perform root cause analysis on a failure."""
    try:
        result = await llm_generate(
            f"Analyze this execution failure and provide a one-sentence root cause:\n\n"
            f"Error: {error_msg[:300]}\n"
            f"Code: {code[:300]}",
            "You are a debugging expert. Respond with a single concise sentence.",
        )
        return result if not result.startswith("[LLM ERROR") else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main Cognitive Loop — Orchestrates all 3 phases
# ═══════════════════════════════════════════════════════════════════════

async def run_cognitive_loop(user_prompt: str) -> SolutionObjectState:
    """
    Execute the full 3-phase cognitive loop for a given user prompt.

    Phase 1: Intent Deconstruction → GoalState
    Phase 2: Hierarchical Object Mapping → SolutionObjectState
    Phase 3: Runtime Execution → Final SolutionObjectState

    Returns the final SolutionObjectState with execution results.
    """
    logger.info(f"\n{'━' * 60}")
    logger.info(f"Processing: {user_prompt[:80]}...")
    logger.info(f"{'━' * 60}")

    # Phase 1: Deconstruct intent into a GoalState
    goal = await phase_1_intent_deconstruction(user_prompt)

    # Phase 2: Map goal into a SolutionObjectState
    state = await phase_2_object_mapping(goal)

    # Phase 3: Execute and record results
    final_state = await phase_3_runtime_execution(state)

    # ── Pretty-print final state ──
    status = final_state.execution_history.last_status.value
    status_icon = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "◌"

    print(f"\n  {'─' * 50}")
    print(f"  {status_icon} Status: {status}")
    print(f"  Goal:   {goal.goal_name}")
    print(f"  State:  {final_state.state_name}")
    print(f"  KPs:    {len(final_state.knowledge_points)} knowledge point(s)")
    if final_state.execution_history.error_log:
        print(f"  Error:  {final_state.execution_history.error_log[:150]}")
    print(f"  {'─' * 50}\n")

    return final_state
