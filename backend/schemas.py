from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    blocker = "blocker"
    sub_issue = "sub_issue"
    curiosity_branch = "curiosity_branch"
    related_concept = "related_concept"


class IssueStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    parked = "parked"
    suggested = "suggested"


class GoalStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    parked = "parked"


class AnalyzeRequest(BaseModel):
    conversation_text: str = Field(min_length=1)


class ConversationTurn(BaseModel):
    turn_id: str = Field(
        pattern=r"^T[1-9]\d*$",
        description="Stable sequential identifier within the conversation, such as T1",
    )
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class GoalThreadCandidate(BaseModel):
    goal_id: str = Field(
        pattern=r"^G[1-9]\d*$",
        description="Stable sequential identifier within the analysis, such as G1",
    )
    main_goal: str = Field(min_length=1)
    summary: str = Field(
        min_length=1,
        description="Concise description of the goal thread and its scope",
    )
    turn_ids: list[str] = Field(
        min_length=1,
        description="Conversation turns that provide evidence for this goal",
    )


class GoalDetectionResult(BaseModel):
    goal_threads: list[GoalThreadCandidate] = Field(min_length=1)


class Issue(BaseModel):
    title: str = Field(description="A concise name for the learning issue")
    type: IssueType
    status: IssueStatus
    trigger: str = Field(description="Why this issue arose in the conversation")
    summary: str
    final_understanding: str = Field(
        description="The learner's demonstrated understanding; empty when unavailable"
    )
    review_note: str = Field(description="A concise note for later review")


class GoalThread(BaseModel):
    goal_id: str = Field(description="Stable identifier within this response, such as G1")
    main_goal: str
    status: GoalStatus
    summary: str
    issues: list[Issue]


class AnalysisResult(BaseModel):
    goal_threads: list[GoalThread] = Field(min_length=1)
