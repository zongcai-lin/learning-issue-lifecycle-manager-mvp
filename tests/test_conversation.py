import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.conversation import parse_conversation_turns
from backend.schemas import (
    AnalysisResult,
    ConversationTurn,
    GoalDetectionResult,
    GoalThreadCandidate,
    IssueStatus,
    IssueType,
)


class ParseConversationTurnsTests(unittest.TestCase):
    def test_preserves_role_order_and_assigns_stable_turn_ids(self) -> None:
        conversation = """Context:
The learner is debugging CORS.

User:
What is CORS?

Assistant:
CORS controls which browser origins may call the API.

User:
I added the middleware and the request now succeeds.
"""

        first_parse = parse_conversation_turns(conversation)
        second_parse = parse_conversation_turns(conversation)

        self.assertEqual([turn.turn_id for turn in first_parse], ["T1", "T2", "T3"])
        self.assertEqual(
            [turn.role for turn in first_parse],
            ["user", "assistant", "user"],
        )
        self.assertEqual(first_parse, second_parse)
        self.assertIn("now succeeds", first_parse[2].content)

    def test_supports_inline_markers_and_ignores_markers_in_code_fences(self) -> None:
        conversation = """User: Show me the expected format.
Assistant:
Use this example:
```text
User:
This line is example data, not a new turn.
```
User: Understood.
"""

        turns = parse_conversation_turns(conversation)

        self.assertEqual(len(turns), 3)
        self.assertIn("This line is example data", turns[1].content)
        self.assertEqual(turns[2].content, "Understood.")

    def test_rejects_empty_turn_content(self) -> None:
        with self.assertRaisesRegex(ValueError, r"T1 \(user\) has no content"):
            parse_conversation_turns("User:\nAssistant:\nA response")

        with self.assertRaisesRegex(ValueError, "Conversation text cannot be empty"):
            parse_conversation_turns("   \n")

    def test_rejects_text_without_role_markers(self) -> None:
        with self.assertRaisesRegex(ValueError, "No User: or Assistant:"):
            parse_conversation_turns("A conversation without explicit role markers")


class SchemaTests(unittest.TestCase):
    def test_conversation_turn_rejects_unstable_id(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationTurn(turn_id="turn-1", role="user", content="Question")

    def test_goal_detection_result_contains_candidates(self) -> None:
        result = GoalDetectionResult(
            goal_threads=[
                GoalThreadCandidate(
                    goal_id="G1",
                    main_goal="Connect the frontend and backend",
                    summary="The learner is making the browser request reach the API.",
                    turn_ids=["T1", "T2", "T3"],
                )
            ]
        )

        self.assertEqual(result.goal_threads[0].turn_ids, ["T1", "T2", "T3"])

    def test_analysis_result_is_the_final_response_contract(self) -> None:
        result = AnalysisResult.model_validate(
            {
                "goal_threads": [
                    {
                        "goal_id": "G1",
                        "main_goal": "Understand CORS",
                        "status": "resolved",
                        "summary": "The learner fixed the browser request.",
                        "issues": [
                            {
                                "title": "Understand CORS",
                                "type": "blocker",
                                "status": "resolved",
                                "trigger": "The browser blocked the request.",
                                "summary": "The learner added restricted CORS middleware.",
                                "final_understanding": "Different ports are different origins.",
                                "review_note": "Test the API and browser path separately.",
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(result.goal_threads[0].goal_id, "G1")
        self.assertEqual(result.goal_threads[0].status.value, "resolved")

    def test_issue_taxonomy_matches_the_product_decision(self) -> None:
        self.assertEqual(
            {issue_type.value for issue_type in IssueType},
            {"blocker", "sub_issue", "curiosity_branch", "related_concept"},
        )
        self.assertEqual(
            {issue_status.value for issue_status in IssueStatus},
            {"open", "resolved", "parked", "suggested"},
        )

        schema_text = str(AnalysisResult.model_json_schema())
        self.assertNotIn("main_issue", schema_text)
        self.assertNotIn("parking_lot", schema_text)

    def test_fastapi_route_uses_analysis_result(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from backend.main import app

        analyze_route = next(route for route in app.routes if route.path == "/analyze")
        self.assertIs(analyze_route.response_model, AnalysisResult)


if __name__ == "__main__":
    unittest.main()
