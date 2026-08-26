import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.conversation import parse_conversation_turns
from backend.schemas import (
    AnalysisResult,
    ConversationTurn,
    GoalDetectionResult,
    GoalIssueExtraction,
    GoalThread,
    GoalThreadCandidate,
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


def extract_issues_for_goal(
    turns: list[ConversationTurn],
    goal: GoalThreadCandidate,
) -> GoalThread:
    """Extract Stage 2 Issues from only the turns assigned to one goal."""

    turns_by_id = {turn.turn_id: turn for turn in turns}
    unknown_turn_ids = [
        turn_id for turn_id in goal.turn_ids if turn_id not in turns_by_id
    ]
    if unknown_turn_ids:
        raise ValueError(
            f"{goal.goal_id} references unknown turns: {', '.join(unknown_turn_ids)}"
        )

    goal_turn_ids = set(goal.turn_ids)
    goal_turns = [turn for turn in turns if turn.turn_id in goal_turn_ids]
    if not goal_turns:
        raise ValueError(f"{goal.goal_id} has no conversation turns")

    serialized_goal = json.dumps(goal.model_dump(), ensure_ascii=False, indent=2)
    serialized_turns = json.dumps(
        [turn.model_dump() for turn in goal_turns],
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""
Extract the learning Issues for exactly one already-detected Goal Thread.
This is Stage 2. Do not create, merge, rename, or re-scope Goal Threads. Determine
the Goal Thread status and extract its Issues only from the supplied goal turns.

Issue type definitions:
- blocker: an issue that must be addressed before its Goal Thread can continue
- sub_issue: a prerequisite, diagnostic step, validation step, or smaller question within a Goal Thread
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
- Return status and at least one Issue. Every Issue must contain exactly:
  title, type, status, trigger, summary, final_understanding, and review_note.
- Use only the supplied turns. They are the complete evidence boundary for this
  Goal Thread; do not import Issues or evidence from another Goal Thread.
- Do not create a main_issue. The Goal Thread's main_goal already represents the main line.
- Do not use parking_lot as an issue type. Represent deferred work with status parked.
- If the user explicitly asked about or explored a concept such as const, fetch,
  JSON, CORS, or async/await, it is not an unchosen system recommendation. Never
  classify such an Issue as related_concept with suggested status; use blocker,
  sub_issue, or curiosity_branch according to its role.
- Use related_concept with suggested only when the system introduced a potentially
  useful concept that the learner did not ask about or explore.
- resolved requires positive evidence: explicit user confirmation, observed
  successful execution, a passing result, or another observable completion signal.
  An assistant explanation, proposed fix, or instruction alone is not evidence of
  resolution. When the evidence is ambiguous, keep the Issue or Goal open.
- Do not create an Issue from pure navigation such as "continue", "what next",
  "下一步是什么", or "继续". A navigation turn with a separate substantive
  question may contribute only that substantive question.
- Taxonomy precedence is mandatory: every diagnostic or validation activity is a
  sub_issue, even when it temporarily blocks progress. Testing an endpoint,
  running code to check whether it works, reproducing an error, inspecting output,
  and verifying a fix are diagnostic/validation activities. Never label one of
  these activities blocker. Reserve blocker for the underlying problem that stops
  the Goal, such as the CORS failure itself.
- Keep distinct substantive user questions separate when they require different explanations.
- Prefer concise, evidence-based summaries. Do not invent learner understanding that is not present.
- If no final understanding is demonstrated, return an empty string for final_understanding.

Goal Thread candidate:
{serialized_goal}

Turns assigned to this Goal Thread:
{serialized_turns}
"""

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You perform Stage 2 issue extraction for one fixed Goal "
                    "Thread and return only the requested structured result."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GoalIssueExtraction,
        temperature=0,
        store=False,
    )

    message = completion.choices[0].message
    extraction = message.parsed
    if extraction is None:
        if message.refusal:
            raise ValueError(f"Issue extraction was refused: {message.refusal}")
        raise ValueError(
            f"Issue extraction returned no parsed result for {goal.goal_id}"
        )

    return GoalThread(
        goal_id=goal.goal_id,
        main_goal=goal.main_goal,
        status=extraction.status,
        summary=goal.summary,
        issues=extraction.issues,
    )


def analyze_learning_conversation(conversation_text: str) -> AnalysisResult:
    """Run the parser, Stage 1 detection, and per-goal Stage 2 extraction."""

    turns = parse_conversation_turns(conversation_text)
    detected_goals = detect_goal_threads(turns)
    goal_threads = [
        extract_issues_for_goal(turns, goal) for goal in detected_goals.goal_threads
    ]

    return AnalysisResult(goal_threads=goal_threads)
