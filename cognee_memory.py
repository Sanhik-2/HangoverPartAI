"""
cognee_memory.py — Persistent memory layer wrapping the Cognee API.

Optimized version designed to fit within tight local hardware constraints
by compressing token structures and switching to fast semantic vectors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import cognee
from cognee.api.v1.search import SearchType

from state_schemas import (
    ExecutionHistory,
    ExecutionStatus,
    FailureEdge,
    GoalState,
    KnowledgePoint,
    SolutionObjectState,
)

logger = logging.getLogger("cognee_memory")


# ─── Dataset Namespaces ───────────────────────────────────────────────

GOALS_DATASET = "agent_goals"
STATES_DATASET = "agent_states"
FAILURES_DATASET = "agent_failures"


# ─── Initialization ───────────────────────────────────────────────────


async def initialize_memory() -> None:
    """Bootstrap the Cognee memory system."""
    logger.info("Initializing Cognee memory layer...")
    logger.info("Cognee memory layer initialized.")


async def reset_memory() -> None:
    """Nuclear option: wipe all Cognee data and start fresh."""
    logger.warning("Resetting ALL Cognee memory — this is destructive!")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    logger.info("Memory reset complete.")


# ─── Goal State Operations ────────────────────────────────────────────


async def store_goal(goal: GoalState) -> str:
    """
    Persist a GoalState to the knowledge graph.

    Optimized: Minimizes token footprint by compressing structural spaces
    and eliminating indented redundant JSON payloads.
    """
    text = goal.to_cognee_text()
    json_compact = goal.model_dump_json()  # Strips formatting whitespaces completely

    combined = f"{text}\nDATA:{json_compact}"

    logger.info(f"Storing goal: {goal.goal_name} ({goal.goal_id})")
    await cognee.add(combined, dataset_name=GOALS_DATASET)
    await cognee.cognify(datasets=[GOALS_DATASET])
    logger.info(f"Goal {goal.goal_id} committed to graph.")

    return goal.goal_id


async def query_similar_goals(goal: GoalState, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Phase 1 graph query: search for overlapping historical goals.

    Optimized: Uses SearchType.SEARCH for rapid localized vector matching
    instead of triggering global graph induction logic.
    """
    query_text = f"Goal target: {goal.goal_name}. Components: {' '.join(goal.understanding_notes)}"
    logger.info(f"Searching for similar goals via semantic vector paths...")

    try:
        results = await cognee.search(
            query_text=query_text,
            query_type=SearchType.SEARCH,  # Direct vector semantic indexing
        )
        logger.info(f"Found {len(results)} similar goal matches.")
        return results if results else []
    except Exception as e:
        logger.warning(f"Goal search failed (may be empty graph): {e}")
        return []


# ─── Solution State Operations ────────────────────────────────────────


async def store_state(state: SolutionObjectState) -> str:
    """
    Persist a SolutionObjectState to the knowledge graph.
    Creates or updates a node in the minified object tree.
    """
    text = state.to_cognee_text()
    json_compact = state.model_dump_json()
    combined = f"{text}\nDATA:{json_compact}"

    logger.info(f"Storing state: {state.state_name} ({state.state_id})")
    await cognee.add(combined, dataset_name=STATES_DATASET)
    await cognee.cognify(datasets=[STATES_DATASET])
    logger.info(f"State {state.state_id} committed to graph.")

    return state.state_id


async def query_ancestors(state_id: str) -> list[dict[str, Any]]:
    """
    Phase 2 inheritance check: traverse the parent state vectors.

    Optimized: Shifts to semantic trace to prevent multi-hop query lockouts on local CPU.
    """
    query_text = f"Ancestry chain and inherited constraints for state_id: {state_id}"
    logger.info(f"Querying ancestors via fast semantic lookup for: {state_id}")

    try:
        results = await cognee.search(
            query_text=query_text,
            query_type=SearchType.SEARCH,  # Direct path retrieval
        )
        return results if results else []
    except Exception as e:
        logger.warning(f"Ancestor query failed: {e}")
        return []


async def query_states_by_goal(goal_id: str) -> list[dict[str, Any]]:
    """Retrieve all solution states linked to a specific goal."""
    query_text = f"Solution states mapping directly to goal identifier: {goal_id}"

    try:
        results = await cognee.search(
            query_text=query_text,
            query_type=SearchType.SEARCH,
        )
        return results if results else []
    except Exception as e:
        logger.warning(f"Goal-state query failed: {e}")
        return []


# ─── Failure Operations ───────────────────────────────────────────────


async def record_failure(
    source_state_id: str,
    error_message: str,
    failed_code: Optional[str] = None,
    root_cause: Optional[str] = None,
) -> FailureEdge:
    """
    Phase 3 failure recording: create a FAILED_BY tracking point.

    Acts as a logical boundary ("scar") that prevents execution tracking
    from re-traversing identical dead-ends.
    """
    failure = FailureEdge(
        source_state_id=source_state_id,
        error_message=error_message,
        failed_code_snippet=failed_code,
        root_cause_analysis=root_cause,
    )

    text = failure.to_cognee_text()
    json_compact = failure.model_dump_json()
    combined = f"{text}\nDATA:{json_compact}"

    logger.info(f"Recording execution dead-end boundary for state {source_state_id}")
    await cognee.add(combined, dataset_name=FAILURES_DATASET)
    await cognee.cognify(datasets=[FAILURES_DATASET])
    logger.info(f"Failure record {failure.failure_id} committed to graph substrate.")

    return failure
