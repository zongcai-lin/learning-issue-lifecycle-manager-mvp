import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.schemas import (
    AnalysisResult,
    ConversationTurn,
    GoalDetectionResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(DOTENV_PATH)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_goal_threads(turns: list[ConversationTurn]) -> GoalDetectionResult:
    """Detect goal-level threads without extracting their learning issues."""

    if not turns:
        raise ValueError("At least one conversation turn is required")

    serialized_turns = json.dumps(
        [turn.model_dump() for turn in turns],
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""
Identify the distinct Goal Threads in the ordered learning conversation below.

A Goal Thread is a concrete learner-facing outcome with independent value. A step
being locally testable is not enough to make it a separate Goal when it is only a
diagnostic, prerequisite, or necessary concept within one delivery workflow.
This is Stage 1 only: detect goals and their source turns. Do not extract Issues,
classify Issue types, judge Issue lifecycle status, or produce final understanding.

Rules:
- Return candidates with exactly: goal_id, main_goal, summary, and turn_ids.
- Use sequential goal IDs G1, G2, G3, and so on.
- Use only turn_ids present in the input.
- A turn may support more than one Goal Thread.
- Goal membership does not need to form a continuous range.
- A pure navigation turn such as "continue", "what next", or "接下来做什么"
  never creates a Goal Thread by itself and may remain unassigned. Create a new
  Goal only after the conversation actually begins work on a new concrete outcome.
- Separate goals when their concrete outcomes or completion criteria are independent.
- Infer the outcome from the whole conversation arc, including later success
  confirmation. Do not freeze an early diagnostic request as a separate Goal if
  later turns show that it served one broader outcome.
- Keep diagnostic steps under the outcome they help accomplish. For example,
  testing an API directly and fixing CORS are one Goal when both are steps toward
  making the frontend request succeed.
- Keep syntax and conceptual questions under the implementation Goal that caused
  them. A question about const, fetch, JSON, or a result div while building a
  rendered page must not become an independent Goal unless the learner explicitly
  begins a separate course or deliverable about that concept.
- Assign all substantive turns that materially support a Goal, including its
  diagnostic, implementation, and necessary conceptual turns.
- Base every candidate on conversation evidence and do not invent goals.

Required boundary examples:
- Directly testing /analyze, diagnosing CORS, applying middleware, and confirming
  that the frontend request succeeds are one Goal: establish frontend/backend
  connectivity. The API test is a diagnostic step, not its own Goal.
- Rendering JSON as issue cards, asking how the result div, async/await, fetch,
  const, and JSON conversion work, and confirming the page works are one Goal:
  render a readable Issue List. The JavaScript questions are necessary concepts,
  not a separate Goal.

Ordered conversation turns:
{serialized_turns}
"""

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You perform Stage 1 goal-thread detection for a learning "
                    "conversation and return only the requested structured result."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GoalDetectionResult,
        temperature=0,
        store=False,
    )

    message = completion.choices[0].message
    result = message.parsed
    if result is None:
        if message.refusal:
            raise ValueError(f"Goal detection was refused: {message.refusal}")
        raise ValueError("Goal detection returned no parsed result")

    known_turn_ids = {turn.turn_id for turn in turns}
    expected_goal_ids = [f"G{index}" for index in range(1, len(result.goal_threads) + 1)]
    actual_goal_ids = [goal.goal_id for goal in result.goal_threads]
    if actual_goal_ids != expected_goal_ids:
        raise ValueError(
            "Goal IDs must be sequential from G1; "
            f"received {', '.join(actual_goal_ids)}"
        )

    for goal in result.goal_threads:
        if len(goal.turn_ids) != len(set(goal.turn_ids)):
            raise ValueError(f"{goal.goal_id} contains duplicate turn references")

        unknown_turn_ids = sorted(set(goal.turn_ids) - known_turn_ids)
        if unknown_turn_ids:
            raise ValueError(
                f"{goal.goal_id} references unknown turns: "
                f"{', '.join(unknown_turn_ids)}"
            )

    return result


def analyze_learning_conversation(conversation_text: str) -> AnalysisResult:
    # The Pydantic model is the source of truth for the final output contract.
    response_schema = json.dumps(
        AnalysisResult.model_json_schema(), ensure_ascii=False, indent=2
    )

    prompt = f"""
You are analyzing an AI-assisted learning conversation.

Your task is to reconstruct the learner's evolving learning structure as one or
more Goal Threads. Each Goal Thread represents a distinct goal pursued in the
conversation and contains the issues that arose while pursuing that goal.

Return ONLY valid JSON that conforms to this JSON Schema:
{response_schema}

Issue type definitions:
- blocker: an issue that must be addressed before its Goal Thread can continue
- sub_issue: a prerequisite, diagnostic step, or smaller question within a Goal Thread
- curiosity_branch: a side question the learner actually explored but that does not block the Goal Thread
- related_concept: a potentially useful concept mentioned in the conversation that the learner did not explore

Issue status definitions:
- open: the issue remains active or unresolved
- resolved: the conversation contains evidence that the learner understood or completed it
- parked: the learner touched the issue but deferred further work
- suggested: the system identified a related concept that the learner has not chosen to explore

Goal status definitions:
- open: the goal remains active or incomplete
- resolved: the conversation contains evidence that the goal was completed
- parked: the learner explicitly deferred the goal

Analysis rules:
- A conversation can contain one or multiple Goal Threads.
- Use task-or-milestone-level granularity, not project-level granularity.
- A Goal Thread represents a concrete outcome with its own independent completion criterion.
- Separate independently completable goals even when they are closely related or belong to the same project.
- Merge activities only when they share the same concrete outcome and completion criterion.
- Start a new Goal Thread when the previous outcome has been completed or confirmed and the learner begins pursuing a new deliverable or outcome.
- Also start a new Goal Thread when the new outcome can be completed and tested independently, or when the success criterion changes substantially.
- Do not start a new Goal Thread merely because the learner asks a diagnostic, conceptual, or implementation question needed to complete the current goal. Record it as an issue under the current Goal Thread.
- Navigation language such as "continue" or "what next" does not create a Goal Thread by itself. If it is followed by work on a new concrete outcome, create the new Goal Thread when that work begins.
- Before generating the JSON, identify every point where the learner moves from one independently testable outcome to another.
- Use sequential goal IDs: G1, G2, G3, and so on.
- Assign every issue to the Goal Thread that explains why it arose.
- Do not create a main_issue. The Goal Thread's main_goal already represents the main line.
- Do not use parking_lot as an issue type. Represent deferred work with status parked.
- Use suggested primarily with related_concept. If the learner actually explored a concept, classify it as blocker, sub_issue, or curiosity_branch instead.
- An assistant answer alone does not prove resolved status. Look for learner confirmation, successful observable progress, or clear evidence in the conversation.
- Do not create issues from navigation messages such as "continue" or "what next" unless they include a substantive question.
- Keep distinct user questions separate when they require different explanations.
- Preserve returns to an earlier goal by assigning later related issues back to that Goal Thread.
- Prefer concise, evidence-based summaries. Do not invent learner understanding that is not present.

Goal boundary example:

Conversation progression:
1. The learner tests POST /analyze in FastAPI /docs.
2. The learner fixes CORS and confirms that the frontend receives backend data.
3. The learner then starts converting the returned raw JSON into readable issue cards.
4. The learner confirms that the issue-list page works.

Correct structure:
- G1: Make the frontend Analyze button successfully request POST /analyze.
  Completion criterion: the frontend successfully receives the backend response.
  Issues: testing /analyze, understanding CORS, and fixing the cross-origin request.
- G2: Render the returned JSON as a readable Issue List page.
  Completion criterion: the page displays the goal and issue fields as readable cards.
  Issues: understanding response.json(), DOM rendering, and relevant JavaScript concepts.

These are separate Goal Threads even though they belong to the same MVP project.

Conversation:
{conversation_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict learning issue lifecycle analyzer. "
                    "Return only JSON matching the supplied schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The analyzer returned an empty response")

    return AnalysisResult.model_validate_json(content)
