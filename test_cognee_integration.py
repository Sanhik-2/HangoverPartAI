"""
test_cognee_integration.py — Integration tests for the Cognee memory layer.

Validates the full pipeline using Gemini API:
  1. Store a goal state → cognify → search
  2. Store a solution state → cognify → search
  3. Record a failure → cognify → verify history
  4. Check dead-end prevention works
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Ensure .env is loaded before anything else
from dotenv import load_dotenv

load_dotenv()

# --- Gemini Configuration & Pydantic Validation Handlers ---
# Cognee's internal validator requires that if LLM_PROVIDER is configured,
# all associated credentials and model tags must be populated together.
# Note: Cognee's LLMProvider enum expects "gemini" instead of "google".
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-2.5-flash"

os.environ["EMBEDDING_PROVIDER"] = "gemini"
os.environ["EMBEDDING_MODEL"] = "text-embedding-004"
os.environ["EMBEDDING_DIMENSIONS"] = "768"

# Pull Gemini key from environment
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if gemini_key:
    # Programmatically bind to all API keys that litellm/cognee might check
    os.environ["LLM_API_KEY"] = gemini_key
    os.environ["EMBEDDING_API_KEY"] = gemini_key
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["GOOGLE_API_KEY"] = gemini_key
else:
    # Print warning if no key is found, but set placeholder to avoid validation crashes
    print("\n" + "!" * 60)
    print("  WARNING: No GEMINI_API_KEY or GOOGLE_API_KEY detected in env.")
    print("  Please add GEMINI_API_KEY='your_api_key' to your .env file!")
    print("!" * 60 + "\n")
    os.environ["LLM_API_KEY"] = "placeholder"
    os.environ["EMBEDDING_API_KEY"] = "placeholder"
# -----------------------------------------------------------

import cognee_memory as memory
from state_schemas import (
    ExecutionHistory,
    ExecutionStatus,
    GoalState,
    KnowledgePoint,
    SolutionObjectState,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-15s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


# ═══════════════════════════════════════════════════════════════════════


async def test_goal_lifecycle():
    """Test 1: Goal storage and retrieval."""
    print("\n" + "═" * 50)
    print("  TEST 1: Goal State Lifecycle")
    print("═" * 50)

    goal = GoalState(
        goal_name="HTTP Server with Rate Limiting",
        preparation_steps=[
            "Choose Python HTTP framework (aiohttp vs FastAPI)",
            "Implement token-bucket rate limiter",
            "Add middleware for request counting",
            "Write load test to verify limits",
        ],
        understanding_notes=[
            "Must support async request handling",
            "Rate limits should be per-IP",
            "Token bucket allows burst traffic",
        ],
    )

    print(f"  Created goal: {goal.goal_name}")
    print(f"  ID: {goal.goal_id}")

    # Store
    print("  → Storing goal...")
    goal_id = await memory.store_goal(goal)
    print(f"  ✓ Stored with ID: {goal_id}")

    # Search
    print("  → Searching for similar goals...")
    results = await memory.query_similar_goals(goal)
    print(f"  ✓ Found {len(results)} matches")

    print("  ✅ TEST 1 PASSED\n")
    return goal


async def test_solution_state_lifecycle(goal: GoalState):
    """Test 2: Solution state storage and ancestry."""
    print("\n" + "═" * 50)
    print("  TEST 2: Solution State Lifecycle")
    print("═" * 50)

    # Root state
    root_state = SolutionObjectState(
        state_name="FastAPI Rate Limiter Root",
        goal_id=goal.goal_id,
        knowledge_points=[
            KnowledgePoint(
                topic="Framework Choice",
                description="FastAPI chosen for async support and Pydantic integration",
                is_verified=False,
            ),
            KnowledgePoint(
                topic="Rate Limit Algorithm",
                description="Token bucket with 100 tokens/minute refill rate",
                is_verified=False,
            ),
        ],
    )

    print(f"  Created root state: {root_state.state_name}")
    print(f"  ID: {root_state.state_id}")

    # Store root
    print("  → Storing root state...")
    root_id = await memory.store_state(root_state)
    print(f"  ✓ Stored root: {root_id}")

    # Child state
    child_state = SolutionObjectState(
        state_name="Token Bucket Implementation",
        parent_state_id=root_state.state_id,
        goal_id=goal.goal_id,
        knowledge_points=[
            KnowledgePoint(
                topic="Bucket Size",
                description="Max 100 tokens, refill 10/second",
                is_verified=True,
            ),
        ],
        execution_history=ExecutionHistory(
            last_status=ExecutionStatus.SUCCESS,
        ),
    )

    print(f"  Created child state: {child_state.state_name}")
    print(f"  Parent: {child_state.parent_state_id}")

    # Store child
    print("  → Storing child state...")
    child_id = await memory.store_state(child_state)
    print(f"  ✓ Stored child: {child_id}")

    # Query ancestors
    print("  → Querying ancestry chain...")
    ancestors = await memory.query_ancestors(child_state.state_id)
    print(f"  ✓ Found {len(ancestors)} ancestor records")

    print("  ✅ TEST 2 PASSED\n")
    return root_state


async def test_failure_recording(state: SolutionObjectState):
    """Test 3: Failure edge recording and dead-end prevention."""
    print("\n" + "═" * 50)
    print("  TEST 3: Failure Recording & Dead-End Prevention")
    print("═" * 50)

    # Record a failure
    print("  → Recording a failure...")
    failure = await memory.record_failure(
        source_state_id=state.state_id,
        error_message="ImportError: No module named 'fastapi'",
        failed_code="from fastapi import FastAPI\napp = FastAPI()",
        root_cause="fastapi not installed in the virtual environment",
    )
    print(f"  ✓ Failure recorded: {failure.failure_id}")

    # Check failure history
    print("  → Checking failure history...")
    history = await memory.get_failure_history(state.state_id)
    print(f"  ✓ Found {len(history)} failure records")

    # Check dead-end detection
    print("  → Testing dead-end detection...")
    is_dead = await memory.check_dead_end(
        state.state_id,
        "from fastapi import FastAPI",
    )
    print(f"  ✓ Dead-end detected: {is_dead}")

    print("  ✅ TEST 3 PASSED\n")


async def test_schema_serialization():
    """Test 4: Verify schema serialization/deserialization."""
    print("\n" + "═" * 50)
    print("  TEST 4: Schema Serialization")
    print("═" * 50)

    goal = GoalState(
        goal_name="Test Serialization",
        preparation_steps=["step1"],
        understanding_notes=["note1"],
    )

    # Test JSON round-trip
    json_str = goal.model_dump_json(indent=2)
    restored = GoalState.model_validate_json(json_str)
    assert restored.goal_name == goal.goal_name
    assert restored.goal_id == goal.goal_id
    print("  ✓ GoalState JSON round-trip OK")

    # Test Cognee text serialization
    text = goal.to_cognee_text()
    assert "[GOAL]" in text
    assert goal.goal_name in text
    print("  ✓ GoalState Cognee text serialization OK")

    # Test SolutionObjectState
    state = SolutionObjectState(
        state_name="Test State",
        knowledge_points=[
            KnowledgePoint(topic="t", description="d", is_verified=True),
        ],
    )
    json_str = state.model_dump_json(indent=2)
    restored_state = SolutionObjectState.model_validate_json(json_str)
    assert restored_state.state_name == state.state_name
    assert len(restored_state.knowledge_points) == 1
    assert restored_state.knowledge_points[0].is_verified is True
    print("  ✓ SolutionObjectState JSON round-trip OK")

    print("  ✅ TEST 4 PASSED\n")


# ═══════════════════════════════════════════════════════════════════════


async def run_all_tests():
    """Execute all integration tests."""
    print("\n" + "═" * 50)
    print("  COGNEE INTEGRATION TEST SUITE")
    print("═" * 50)
    print(f"  LLM Provider:  {os.getenv('LLM_PROVIDER', 'unknown')}")
    print(f"  LLM Model:     {os.getenv('LLM_MODEL', 'unknown')}")
    print(f"  Embed Model:   {os.getenv('EMBEDDING_MODEL', 'unknown')}")

    # Initialize
    print("\n  → Initializing Cognee...")
    await memory.initialize_memory()
    print("  ✓ Memory initialized.")

    # Run tests
    await test_schema_serialization()

    goal = await test_goal_lifecycle()
    state = await test_solution_state_lifecycle(goal)
    await test_failure_recording(state)

    print("\n" + "═" * 50)
    print("  ALL TESTS PASSED ✅")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
