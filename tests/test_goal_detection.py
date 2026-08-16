import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.analyzer import DOTENV_PATH, detect_goal_threads
from backend.schemas import ConversationTurn, GoalDetectionResult, GoalThreadCandidate


def completion_with(result: GoalDetectionResult | None, refusal: str | None = None):
    message = SimpleNamespace(parsed=result, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DetectGoalThreadsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = [
            ConversationTurn(turn_id="T1", role="user", content="Why is fetch blocked?"),
            ConversationTurn(turn_id="T2", role="assistant", content="Check CORS."),
            ConversationTurn(turn_id="T3", role="user", content="The request works now."),
        ]

    def test_dotenv_path_is_fixed_to_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]

        self.assertEqual(DOTENV_PATH, repository_root / ".env")

    @patch("backend.analyzer.client.chat.completions.parse")
    def test_uses_pydantic_structured_output_and_preserves_turn_data(self, parse) -> None:
        expected = GoalDetectionResult(
            goal_threads=[
                GoalThreadCandidate(
                    goal_id="G1",
                    main_goal="Connect the frontend to the API",
                    summary="The learner resolves the blocked browser request.",
                    turn_ids=["T1", "T2", "T3"],
                )
            ]
        )
        parse.return_value = completion_with(expected)

        result = detect_goal_threads(self.turns)

        self.assertEqual(result, expected)
        kwargs = parse.call_args.kwargs
        self.assertIs(kwargs["response_format"], GoalDetectionResult)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertFalse(kwargs["store"])

        prompt = kwargs["messages"][-1]["content"]
        serialized = prompt[prompt.index("[") :]
        supplied_turns = json.loads(serialized)
        self.assertEqual(supplied_turns[0]["turn_id"], "T1")
        self.assertEqual(supplied_turns[0]["role"], "user")
        self.assertEqual(supplied_turns[2]["content"], "The request works now.")

    @patch("backend.analyzer.client.chat.completions.parse")
    def test_allows_a_turn_to_support_multiple_goals(self, parse) -> None:
        expected = GoalDetectionResult(
            goal_threads=[
                GoalThreadCandidate(
                    goal_id="G1",
                    main_goal="Connect the frontend to the API",
                    summary="Resolve the browser request failure.",
                    turn_ids=["T1", "T2"],
                ),
                GoalThreadCandidate(
                    goal_id="G2",
                    main_goal="Confirm the integration works",
                    summary="Use the successful request as integration evidence.",
                    turn_ids=["T2", "T3"],
                ),
            ]
        )
        parse.return_value = completion_with(expected)

        result = detect_goal_threads(self.turns)

        self.assertIn("T2", result.goal_threads[0].turn_ids)
        self.assertIn("T2", result.goal_threads[1].turn_ids)

    @patch("backend.analyzer.client.chat.completions.parse")
    def test_rejects_unknown_turn_references(self, parse) -> None:
        parse.return_value = completion_with(
            GoalDetectionResult(
                goal_threads=[
                    GoalThreadCandidate(
                        goal_id="G1",
                        main_goal="Connect the frontend to the API",
                        summary="Resolve the browser request failure.",
                        turn_ids=["T1", "T99"],
                    )
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "G1 references unknown turns: T99"):
            detect_goal_threads(self.turns)

    @patch("backend.analyzer.client.chat.completions.parse")
    def test_rejects_non_sequential_goal_ids(self, parse) -> None:
        parse.return_value = completion_with(
            GoalDetectionResult(
                goal_threads=[
                    GoalThreadCandidate(
                        goal_id="G2",
                        main_goal="Connect the frontend to the API",
                        summary="Resolve the browser request failure.",
                        turn_ids=["T1", "T2"],
                    )
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "sequential from G1"):
            detect_goal_threads(self.turns)

    @patch("backend.analyzer.client.chat.completions.parse")
    def test_rejects_duplicate_turn_references_within_one_goal(self, parse) -> None:
        parse.return_value = completion_with(
            GoalDetectionResult(
                goal_threads=[
                    GoalThreadCandidate(
                        goal_id="G1",
                        main_goal="Connect the frontend to the API",
                        summary="Resolve the browser request failure.",
                        turn_ids=["T1", "T1"],
                    )
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "duplicate turn references"):
            detect_goal_threads(self.turns)

    def test_rejects_empty_turn_list_before_calling_openai(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one conversation turn"):
            detect_goal_threads([])

    def test_stage_one_schema_contains_no_issue_or_lifecycle_fields(self) -> None:
        candidate_fields = set(GoalThreadCandidate.model_fields)
        self.assertEqual(
            candidate_fields,
            {"goal_id", "main_goal", "summary", "turn_ids"},
        )
        self.assertNotIn("issues", GoalDetectionResult.model_json_schema())


if __name__ == "__main__":
    unittest.main()
