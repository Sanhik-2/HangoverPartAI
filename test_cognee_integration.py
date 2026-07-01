"""
test_cognee_integration.py — Integration tests for the Cognee memory layer.

Validates the full pipeline:
  1. Schema serialization round-trips (to_cognee_text)
  2. Store a goal state → cognify → search
  3. Store a solution state → cognify → search
  4. Record a failure → cognify → search
  5. Agent loop import and module integrity checks

Can run standalone: python test_cognee_integration.py
"""

import asyncio
import json
import logging
import os
import sys

# Force Cognee to disable multi-tenant authentication and access control.
# This must be set programmatically before any cognee imports.
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

from dotenv import load_dotenv
load_dotenv()

from state_schemas import (
    ExecutionHistory,
    ExecutionStatus,
    FailureEdge,
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
# Unit Tests — Schema Serialization (no Cognee required)
# ═══════════════════════════════════════════════════════════════════════

def test_goal_state_serialization():
    """Test GoalState creation and text serialization."""
    goal = GoalState(
        goal_name="Implement Cognee integration",
        preparation_steps=["Install cognee", "Configure .env", "Write schemas"],
        understanding_notes=["Cognee uses async API", "Requires Ollama for embeddings"],
    )

    assert goal.goal_name == "Implement Cognee integration"
    assert goal.goal_id.startswith("goal_")
    assert len(goal.preparation_steps) == 3
    assert goal.state_type.value == "Goal"

    text = goal.to_cognee_text()
    assert "[GOAL]:" in text
    assert "Implement Cognee integration" in text
    assert "Install cognee" in text

    # JSON round-trip
    json_str = goal.model_dump_json()
    restored = GoalState.model_validate_json(json_str)
    assert restored.goal_name == goal.goal_name
    assert restored.goal_id == goal.goal_id

    logger.info("  ✓ GoalState serialization test passed")


def test_solution_state_serialization():
    """Test SolutionObjectState creation and text serialization."""
    state = SolutionObjectState(
        state_name="Cognee memory wrapper",
        goal_id="goal_test123",
        knowledge_points=[
            KnowledgePoint(
                topic="Cognee API",
                description="Uses add/cognify/search pipeline",
                is_verified=True,
            ),
            KnowledgePoint(
                topic="SearchType",
                description="CHUNKS type for basic text retrieval",
                is_verified=False,
            ),
        ],
        execution_history=ExecutionHistory(
            last_status=ExecutionStatus.SUCCESS,
        ),
    )

    assert state.state_name == "Cognee memory wrapper"
    assert state.state_id.startswith("state_")
    assert len(state.knowledge_points) == 2
    assert state.execution_history.last_status == ExecutionStatus.SUCCESS

    text = state.to_cognee_text()
    assert "[STATE]:" in text
    assert "Cognee memory wrapper" in text
    assert "Cognee API:True" in text
    assert "Status: SUCCESS" in text

    # JSON round-trip
    json_str = state.model_dump_json()
    restored = SolutionObjectState.model_validate_json(json_str)
    assert restored.state_name == state.state_name
    assert len(restored.knowledge_points) == 2

    logger.info("  ✓ SolutionObjectState serialization test passed")


def test_failure_edge_serialization():
    """Test FailureEdge creation and text serialization."""
    failure = FailureEdge(
        source_state_id="state_test456",
        error_message="ModuleNotFoundError: No module named 'nonexistent'",
        failed_code_snippet="import nonexistent\nnonexistent.run()",
        root_cause_analysis="Module not installed in the virtual environment",
    )

    assert failure.failure_id.startswith("fail_")
    assert failure.source_state_id == "state_test456"
    assert failure.state_type.value == "Failure"

    text = failure.to_cognee_text()
    assert "[FAILURE]:" in text
    assert "state_test456" in text
    assert "ModuleNotFoundError" in text

    # JSON round-trip
    json_str = failure.model_dump_json()
    restored = FailureEdge.model_validate_json(json_str)
    assert restored.source_state_id == failure.source_state_id
    assert restored.error_message == failure.error_message

    logger.info("  ✓ FailureEdge serialization test passed")


def test_execution_history_defaults():
    """Test ExecutionHistory default values."""
    history = ExecutionHistory()
    assert history.last_status == ExecutionStatus.PENDING
    assert history.error_log is None
    assert history.timestamp is not None

    logger.info("  ✓ ExecutionHistory defaults test passed")


# ═══════════════════════════════════════════════════════════════════════
# Module Import Tests
# ═══════════════════════════════════════════════════════════════════════

def test_module_imports():
    """Verify all core modules import without error."""
    from cognee_memory import CogneeMemory, memory, initialize_memory, reset_memory
    assert isinstance(memory, CogneeMemory)

    from agent_loop import run_cognitive_loop, llm_generate, llm_generate_json
    assert callable(run_cognitive_loop)
    assert callable(llm_generate)
    assert callable(llm_generate_json)

    logger.info("  ✓ Module import test passed")


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — Cognee Pipeline (requires running Ollama)
# ═══════════════════════════════════════════════════════════════════════

async def test_cognee_goal_lifecycle():
    """Integration test: Store a goal → cognify → search."""
    from cognee_memory import memory

    goal = GoalState(
        goal_name="Test goal lifecycle integration",
        preparation_steps=["Step A", "Step B"],
        understanding_notes=["Note 1"],
    )

    logger.info("  Storing goal...")
    goal_id = await memory.store_goal(goal)
    assert goal_id == goal.goal_id
    logger.info(f"  Goal stored: {goal_id}")

    logger.info("  Querying similar goals...")
    results = await memory.query_similar_goals(goal)
    logger.info(f"  Found {len(results)} similar goal(s)")

    logger.info("  ✓ Goal lifecycle integration test passed")


async def test_cognee_state_lifecycle():
    """Integration test: Store a solution state → cognify → query ancestors."""
    from cognee_memory import memory

    state = SolutionObjectState(
        state_name="Test state lifecycle",
        goal_id="goal_lifecycle_test",
        knowledge_points=[
            KnowledgePoint(
                topic="Testing",
                description="Validates the Cognee pipeline",
                is_verified=False,
            ),
        ],
    )

    logger.info("  Storing solution state...")
    state_id = await memory.store_state(state)
    assert state_id == state.state_id
    logger.info(f"  State stored: {state_id}")

    logger.info("  Querying ancestors...")
    ancestors = await memory.query_ancestors(state.state_id)
    logger.info(f"  Found {len(ancestors)} ancestor(s)")

    logger.info("  ✓ State lifecycle integration test passed")


async def test_cognee_failure_recording():
    """Integration test: Record a failure → cognify → query history."""
    from cognee_memory import memory

    logger.info("  Recording failure edge...")
    failure = await memory.record_failure(
        source_state_id="state_failure_test",
        error_message="TestError: Intentional failure for integration testing",
        failed_code="raise TestError('integration test')",
        root_cause="Intentional test failure",
    )
    assert failure.failure_id.startswith("fail_")
    logger.info(f"  Failure recorded: {failure.failure_id}")

    logger.info("  Querying failure history...")
    history = await memory.get_failure_history("state_failure_test")
    logger.info(f"  Found {len(history)} failure record(s)")

    logger.info("  ✓ Failure recording integration test passed")


# ═══════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════

def run_unit_tests():
    """Run all unit tests (no external dependencies needed)."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Unit Tests — Schema Serialization              ║")
    print("╚══════════════════════════════════════════════════╝\n")

    tests = [
        test_goal_state_serialization,
        test_solution_state_serialization,
        test_failure_edge_serialization,
        test_execution_history_defaults,
        test_module_imports,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


async def run_integration_tests():
    """Run all integration tests (requires Cognee + Ollama)."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Integration Tests — Cognee Pipeline            ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Initialize memory layer
    from cognee_memory import initialize_memory
    await initialize_memory()

    tests = [
        test_cognee_goal_lifecycle,
        test_cognee_state_lifecycle,
        test_cognee_failure_recording,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"  ✗ {test.__name__} FAILED: {e}", exc_info=True)
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "═" * 54)
    print("  CogneeAIProject — Test Suite")
    print("═" * 54)

    # Always run unit tests
    unit_ok = run_unit_tests()

    # Run integration tests if --integration flag is passed
    if "--integration" in sys.argv:
        integration_ok = asyncio.run(run_integration_tests())
    else:
        print("  ℹ Skipping integration tests (pass --integration to run them)")
        integration_ok = True

    print("\n" + "═" * 54)
    if unit_ok and integration_ok:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("═" * 54 + "\n")

    sys.exit(0 if (unit_ok and integration_ok) else 1)


if __name__ == "__main__":
    main()
