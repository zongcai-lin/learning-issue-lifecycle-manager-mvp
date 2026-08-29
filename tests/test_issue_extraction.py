import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.analyzer import analyze_learning_conversation, extract_issues_for_goal
from backend.main import app
from backend.schemas import (
    AnalysisResult,
    ConversationTurn,
    GoalDetectionResult,
    GoalIssueExtraction,
    GoalThreadCandidate,
    Issue,
)


def completion_with(result, refusal: str | None = None):
    message = SimpleNamespace(parsed=result, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def issue(title: str, status: str = "open") -> Issue:
    return Issue(
        title=title,
        type="sub_issue",
        status=status,
        trigger=f"The learner raised {title}.",
        summary=f"The learner worked on {title}.",
        final_understanding="" if status == "open" else f"Understood {title}.",
        review_note=f"Review {title}.",
    )


class ExtractIssuesForGoalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = [
            ConversationTurn(
                turn_id="T1", role="user", content="Why is CORS blocking fetch?"
            ),
            ConversationTurn(turn_id="T2", role="assistant", content="Add CORS middleware."),
            ConversationTurn(turn_id="T3", role="user", content="The request succeeds now."),
            ConversationTurn(turn_id="T4", role="user", content="What does const mean?"),
            ConversationTurn(turn_id="T5", role="assistant", content="const creates a binding."),
        ]

    @patch("backend.analyzer.get_openai_client")
    def test_uses_only_turns_assigned_to_the_goal(self, get_client) -> None:
        parse = get_client.return_value.chat.completions.parse
        goal = GoalThreadCandidate(
            goal_id="G2",
            main_goal="Understand the JavaScript rendering code",
            summary="The learner asks about const while rendering results.",
            turn_ids=["T4", "T5"],
        )
        parse.return_value = completion_with(
            GoalIssueExtraction(status="open", issues=[issue("Understand const")])
        )

        result = extract_issues_for_goal(self.turns, goal)

        self.assertEqual(result.goal_id, "G2")
        self.assertEqual(result.main_goal, goal.main_goal)
        self.assertEqual(result.summary, goal.summary)
        prompt = parse.call_args.kwargs["messages"][-1]["content"]
        supplied_turns = json.loads(prompt.split("Turns assigned to this Goal Thread:\n", 1)[1])
        self.assertEqual([turn["turn_id"] for turn in supplied_turns], ["T4", "T5"])
        self.assertNotIn("CORS", json.dumps(supplied_turns))

    @patch("backend.analyzer.get_openai_client")
    def test_uses_structured_output_and_encodes_product_rules(self, get_client) -> None:
        parse = get_client.return_value.chat.completions.parse
        goal = GoalThreadCandidate(
            goal_id="G1",
            main_goal="Connect the frontend to the API",
            summary="The learner resolves a browser request failure.",
            turn_ids=["T1", "T2", "T3"],
        )
        parse.return_value = completion_with(
            GoalIssueExtraction(status="resolved", issues=[issue("Fix CORS", "resolved")])
        )

        extract_issues_for_goal(self.turns, goal)

        kwargs = parse.call_args.kwargs
        self.assertIs(kwargs["response_format"], GoalIssueExtraction)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertFalse(kwargs["store"])
        prompt = kwargs["messages"][-1]["content"]
        self.assertIn("Never\n  classify such an Issue as related_concept with suggested", prompt)
        self.assertIn("resolved requires positive evidence", prompt)
        self.assertIn("Do not create an Issue from pure navigation", prompt)
        self.assertIn("Taxonomy precedence is mandatory", prompt)
        self.assertIn("Never label one of\n  these activities blocker", prompt)

    def test_rejects_unknown_goal_turn_before_calling_openai(self) -> None:
        goal = GoalThreadCandidate(
            goal_id="G1",
            main_goal="Connect the frontend to the API",
            summary="The learner resolves a browser request failure.",
            turn_ids=["T1", "T99"],
        )

        with self.assertRaisesRegex(ValueError, "G1 references unknown turns: T99"):
            extract_issues_for_goal(self.turns, goal)


class StagedAnalysisTests(unittest.TestCase):
    @patch("backend.analyzer.get_openai_client")
    def test_combines_independent_extractions_into_analysis_result(self, get_client) -> None:
        parse = get_client.return_value.chat.completions.parse
        goals = GoalDetectionResult(
            goal_threads=[
                GoalThreadCandidate(
                    goal_id="G1",
                    main_goal="Connect the frontend to the API",
                    summary="Fix the browser request.",
                    turn_ids=["T1", "T2", "T3"],
                ),
                GoalThreadCandidate(
                    goal_id="G2",
                    main_goal="Understand the JavaScript rendering code",
                    summary="Explain const in the result renderer.",
                    turn_ids=["T4", "T5"],
                ),
            ]
        )
        parse.side_effect = [
            completion_with(goals),
            completion_with(
                GoalIssueExtraction(status="resolved", issues=[issue("Fix CORS", "resolved")])
            ),
            completion_with(
                GoalIssueExtraction(status="open", issues=[issue("Understand const")])
            ),
        ]
        conversation = """User: Why is CORS blocking fetch?
Assistant: Add CORS middleware.
User: The request succeeds now.
User: What does const mean?
Assistant: const creates a binding.
"""

        result = analyze_learning_conversation(conversation)

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual([goal.goal_id for goal in result.goal_threads], ["G1", "G2"])
        self.assertEqual(result.goal_threads[0].issues[0].title, "Fix CORS")
        self.assertEqual(result.goal_threads[1].issues[0].title, "Understand const")
        self.assertEqual(parse.call_count, 3)
        response_formats = [
            invocation.kwargs["response_format"] for invocation in parse.call_args_list
        ]
        self.assertEqual(
            response_formats,
            [GoalDetectionResult, GoalIssueExtraction, GoalIssueExtraction],
        )
        get_client.assert_called_once_with()


class PublicAnalyzeContractTests(unittest.TestCase):
    @patch("backend.main.analyze_learning_conversation")
    def test_analyze_returns_goal_threads_consumable_by_current_frontend(
        self, analyze
    ) -> None:
        analyze.return_value = AnalysisResult.model_validate(
            {
                "goal_threads": [
                    {
                        "goal_id": "G1",
                        "main_goal": "Connect the frontend to the API",
                        "status": "resolved",
                        "summary": "The browser request succeeds.",
                        "issues": [issue("Fix CORS", "resolved").model_dump()],
                    },
                    {
                        "goal_id": "G2",
                        "main_goal": "Render readable issue cards",
                        "status": "open",
                        "summary": "The learner is understanding the renderer.",
                        "issues": [issue("Understand const").model_dump()],
                    },
                ]
            }
        )

        response = TestClient(app).post(
            "/analyze",
            json={"conversation_text": "User: Please analyze this conversation."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [goal["goal_id"] for goal in payload["goal_threads"]], ["G1", "G2"]
        )
        self.assertEqual(
            payload["goal_threads"][0]["issues"][0]["title"], "Fix CORS"
        )
        self.assertEqual(
            set(payload["goal_threads"][1]["issues"][0]),
            {
                "title",
                "type",
                "status",
                "trigger",
                "summary",
                "final_understanding",
                "review_note",
            },
        )

    def test_final_contract_requires_at_least_one_issue_per_goal(self) -> None:
        with self.assertRaises(ValidationError):
            AnalysisResult.model_validate(
                {
                    "goal_threads": [
                        {
                            "goal_id": "G1",
                            "main_goal": "Connect the frontend to the API",
                            "status": "open",
                            "summary": "The browser request is being diagnosed.",
                            "issues": [],
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
