"""
state_schemas.py — Pydantic models for the State-Driven Developer Agent.

Enforces structured JSON schemas for Goal States, Solution Objects,
Knowledge Points, and Failure Edges that flow through the Cognee graph.

Optimized to output highly compressed, token-dense string signatures.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ─── Enums ─────────────────────────────────────────────────────────────


class ExecutionStatus(str, Enum):
    """Possible outcomes of a runtime execution step."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


class StateType(str, Enum):
    """Discriminator for top-level state payloads."""

    GOAL = "Goal"
    SOLUTION = "Solution"
    FAILURE = "Failure"


# ─── Knowledge Point ──────────────────────────────────────────────────


class KnowledgePoint(BaseModel):
    """
    An individual unit of verified or unverified knowledge.

    These accumulate inside SolutionObjectState nodes and represent
    truths, formulas, or system invariants discovered during execution.
    """

    topic: str = Field(..., description="Short identifier for the knowledge domain")
    description: str = Field(..., description="Detailed explanation of the knowledge")
    is_verified: bool = Field(
        default=False,
        description="True if this fact was confirmed by successful execution",
    )


# ─── Execution History ────────────────────────────────────────────────


class ExecutionHistory(BaseModel):
    """Tracks the last execution outcome for a solution state."""

    last_status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    error_log: Optional[str] = Field(
        default=None, description="stderr or exception traceback if FAILED"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─── Goal State ───────────────────────────────────────────────────────


class GoalState(BaseModel):
    """
    Phase 1 output: Intent Deconstruction & Goal Serialization.

    Created from every user prompt before any code generation begins.
    Serialized and queried against Cognee to find overlapping historical
    attempts.
    """

    state_type: StateType = Field(default=StateType.GOAL, frozen=True)
    goal_id: str = Field(
        default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this goal",
    )
    goal_name: str = Field(
        ..., description="Precise technical title for the breakthrough target"
    )
    preparation_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of deterministic execution strategies",
    )
    understanding_notes: list[str] = Field(
        default_factory=list,
        description="Deep architectural background and constraints",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_cognee_text(self) -> str:
        """Serialize to a hyper-dense text block for low-overhead Cognee ingestion."""
        steps_flat = ",".join(self.preparation_steps)
        notes_flat = ",".join(self.understanding_notes)
        return f"[GOAL]: {self.goal_name}\nID: {self.goal_id}\nSteps: {steps_flat}\nNotes: {notes_flat}"


# ─── Solution Object State ────────────────────────────────────────────


class SolutionObjectState(BaseModel):
    """
    Phase 2 output: Hierarchical Object Mapping.

    Each logical component, state, or isolated solution is a discrete
    node in the object root tree graph. Child nodes inherit constraints
    from parent nodes via parent_state_id.
    """

    state_type: StateType = Field(default=StateType.SOLUTION, frozen=True)
    state_id: str = Field(
        default_factory=lambda: f"state_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this solution state",
    )
    state_name: str = Field(
        ..., description="Human-readable name for this solution branch"
    )
    parent_state_id: Optional[str] = Field(
        default=None,
        description="Links to parent node in the inheritance tree (null = root)",
    )
    goal_id: Optional[str] = Field(
        default=None, description="Links back to the originating GoalState"
    )
    knowledge_points: list[KnowledgePoint] = Field(
        default_factory=list, description="Accumulated facts and invariants"
    )
    execution_history: ExecutionHistory = Field(
        default_factory=ExecutionHistory, description="Last runtime execution outcome"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_cognee_text(self) -> str:
        """Serialize to text with zero structural padding to conserve model context."""
        kp_summary = ",".join(
            [f"{kp.topic}:{kp.is_verified}" for kp in self.knowledge_points]
        )
        err_msg = (
            f" | Error: {self.execution_history.error_log[:100]}"
            if self.execution_history.error_log
            else ""
        )
        return (
            f"[STATE]: {self.state_name}\n"
            f"ID: {self.state_id}\n"
            f"Parent: {self.parent_state_id or 'ROOT'}\n"
            f"Goal: {self.goal_id or 'N/A'}\n"
            f"Status: {self.execution_history.last_status.value}{err_msg}\n"
            f"KPs: {kp_summary}"
        )


# ─── Failure Edge ─────────────────────────────────────────────────────


class FailureEdge(BaseModel):
    """
    Phase 3 output: Failure recording.

    Represents the relationship:
      (CurrentStateObject)-[:FAILED_BY {stderr: "..."}]->(InvalidCodePayload)

    Stored permanently to prevent re-traversal of dead-end paths.
    """

    state_type: StateType = Field(default=StateType.FAILURE, frozen=True)
    failure_id: str = Field(
        default_factory=lambda: f"fail_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this failure record",
    )
    source_state_id: str = Field(
        ..., description="The state that attempted execution and failed"
    )
    error_message: str = Field(..., description="The stderr or exception message")
    failed_code_snippet: Optional[str] = Field(
        default=None, description="The code or command that caused the failure"
    )
    root_cause_analysis: Optional[str] = Field(
        default=None, description="Agent's analysis of why the failure occurred"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_cognee_text(self) -> str:
        """Serialize failure signature tightly to safeguard context frames."""
        code_flat = (
            self.failed_code_snippet.replace("\n", " ")
            if self.failed_code_snippet
            else "None"
        )
        return (
            f"[FAILURE]: State {self.source_state_id} crashed.\n"
            f"ID: {self.failure_id}\n"
            f"Error: {self.error_message[:150]}\n"
            f"Code: {code_flat[:150]}"
        )
