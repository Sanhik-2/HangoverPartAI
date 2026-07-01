import os

from dotenv import load_dotenv

# ─── CRITICAL: BOOTSTRAP ENVIRONMENT BEFORE IMPORTING COGNEE ─────────
# We load .env and set path variables statically here to prevent Cognee
# from initializing with default site-packages SQLite locks.
load_dotenv()

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

if not os.environ.get("COGNEE_SYSTEM_PATH"):
    os.environ["COGNEE_SYSTEM_PATH"] = "/home/Sanhik/CogneeAIProject/.cognee_system"

import json
import logging
from typing import List, Optional

import cognee
from cognee.infrastructure.llm.config import LLMConfig

from state_schemas import FailureEdge, GoalState, SolutionObjectState

logger = logging.getLogger("cognee_memory")


def _get_ollama_llm_config() -> LLMConfig:
    """Build an Ollama-based LLMConfig for Cognee's internal graph extraction.

    Cognee's cognify() pipeline uses `instructor` for structured output,
    which requires full OpenAI function-calling support. Gemini's OpenAI
    compatibility gateway doesn't support this, causing MALFORMED_FUNCTION_CALL
    errors. We route Cognee's internal LLM calls through local Ollama instead.
    """
    return LLMConfig(
        llm_provider="ollama",
        llm_model="ollama/llama3.1:8b",
        llm_endpoint="http://localhost:11434/v1",
        llm_api_key="ollama",
    )

# ─── Robust Version-Agnostic SearchType Import Cascade ────────────────
try:
    # Primary import path for Cognee v1.2.x
    from cognee.modules.search.types.SearchType import SearchType
except ImportError:
    try:
        # Fallback for cognee.modules.search.types (non-submodule)
        from cognee.modules.search.types import SearchType
    except ImportError:
        try:
            # Fallback for root export mapping
            from cognee import SearchType
        except ImportError:
            # Absolute fail-safe: Define string-compatible matching mock
            from enum import Enum

            class SearchType(str, Enum):
                CHUNKS = "CHUNKS"
                GRAPH_COMPLETION = "GRAPH_COMPLETION"
                SUMMARIES = "SUMMARIES"


GOALS_DATASET = "goals_dataset"
STATES_DATASET = "states_dataset"
FAILURES_DATASET = "failures_dataset"


class CogneeMemory:
    async def store_goal(self, goal: GoalState) -> str:
        """Serialize the goal and ingest it into the Cognee graph database."""
        text_payload = goal.to_cognee_text()
        # Compact JSON representation for data payload preservation
        json_compact = goal.model_dump_json()
        combined = f"{text_payload}\nDATA:{json_compact}"

        logger.info(f"Storing goal: {goal.goal_name} ({goal.goal_id})")
        await cognee.add(combined, dataset_name=GOALS_DATASET)
        await cognee.cognify(datasets=[GOALS_DATASET], llm_config=_get_ollama_llm_config())
        logger.info(f"Goal {goal.goal_id} committed to graph.")
        return goal.goal_id

    async def query_similar_goals(self, goal: GoalState) -> List[dict]:
        """Perform vector similarity search on ingested goal configurations."""
        try:
            results = await cognee.search(
                query_text=goal.goal_name, query_type=SearchType.CHUNKS, datasets=[GOALS_DATASET]
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Error querying similar goals: {e}")
            return []

    async def store_state(self, state: SolutionObjectState) -> str:
        """Serialize the solution state and commit it to the Cognee memory graph."""
        text_payload = state.to_cognee_text()
        json_compact = state.model_dump_json()
        combined = f"{text_payload}\nDATA:{json_compact}"

        logger.info(f"Storing solution state: {state.state_name} ({state.state_id})")
        await cognee.add(combined, dataset_name=STATES_DATASET)
        await cognee.cognify(datasets=[STATES_DATASET], llm_config=_get_ollama_llm_config())
        logger.info(f"Solution state {state.state_id} committed.")
        return state.state_id

    async def query_ancestors(self, parent_state_id: str) -> List[dict]:
        """Retrieve ancestors of a given state node to check behavioral constraints."""
        try:
            results = await cognee.search(
                query_text=parent_state_id, query_type=SearchType.CHUNKS, datasets=[STATES_DATASET]
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Error querying ancestors: {e}")
            return []

    async def get_failure_history(self, state_id: str) -> List[dict]:
        """Query vector indexes for recorded FAILED_BY relationships intersecting this node path."""
        try:
            results = await cognee.search(
                query_text=state_id, query_type=SearchType.CHUNKS, datasets=[FAILURES_DATASET]
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Error retrieving failure history: {e}")
            return []

    async def record_failure(
        self,
        source_state_id: str,
        error_message: str,
        failed_code: str,
        root_cause: Optional[str] = None,
    ) -> FailureEdge:
        """Instantiate and record a FailureEdge to prevent tracing redundant dead-ends."""
        failure = FailureEdge(
            source_state_id=source_state_id,
            error_message=error_message,
            failed_code_snippet=failed_code,
            root_cause_analysis=root_cause,
        )
        text_payload = failure.to_cognee_text()
        json_compact = failure.model_dump_json()
        combined = f"{text_payload}\nDATA:{json_compact}"

        logger.info(f"Recording failure edge for state: {source_state_id}")
        await cognee.add(combined, dataset_name=FAILURES_DATASET)
        await cognee.cognify(datasets=[FAILURES_DATASET], llm_config=_get_ollama_llm_config())
        return failure


# Export a global memory instance to satisfy `from cognee_memory import memory`
memory = CogneeMemory()


async def initialize_memory():
    """Bootstraps directories and system files for Cognee."""
    system_path = os.getenv("COGNEE_SYSTEM_PATH", os.path.expanduser("~/.cognee"))
    db_path = os.path.join(system_path, "databases")
    os.makedirs(db_path, exist_ok=True)
    logger.info("Cognee memory layer initialized.")


async def reset_memory():
    """Wipes all local graph database records and cached vector indices."""
    import shutil

    system_path = os.getenv("COGNEE_SYSTEM_PATH", os.path.expanduser("~/.cognee"))
    if os.path.exists(system_path):
        shutil.rmtree(system_path)
        os.makedirs(system_path, exist_ok=True)
        logger.info("Local database cache partitions cleared.")
