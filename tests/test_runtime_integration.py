import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from openai import OpenAIError

from backend.analyzer import (
    AnalyzerConfigurationError,
    AnalyzerResponseError,
    NoGoalThreadsError,
    detect_goal_threads,
    get_openai_client,
)
from backend.main import app
from backend.schemas import ConversationTurn, GoalDetectionResult


class LazyClientConfigurationTests(unittest.TestCase):
    def test_missing_api_key_fails_when_client_is_requested(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                AnalyzerConfigurationError, "OPENAI_API_KEY is not configured"
            ):
                get_openai_client()

    @patch("backend.analyzer.OpenAI")
    def test_client_receives_configured_key_without_logging_it(self, openai) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}, clear=True):
            client = get_openai_client()

        self.assertIs(client, openai.return_value)
        openai.assert_called_once_with(api_key="test-secret")


class AnalyzerBoundaryTests(unittest.TestCase):
    def test_no_goal_threads_has_a_distinct_domain_error(self) -> None:
        openai_client = MagicMock()
        openai_client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=GoalDetectionResult(goal_threads=[]),
                        refusal=None,
                    )
                )
            ]
        )

        with self.assertRaisesRegex(NoGoalThreadsError, "No substantive Goal Thread"):
            detect_goal_threads(
                [ConversationTurn(turn_id="T1", role="user", content="Continue")],
                openai_client=openai_client,
            )


class AnalyzeRouteErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_empty_string_is_rejected_by_request_schema(self) -> None:
        response = self.client.post("/analyze", json={"conversation_text": ""})

        self.assertEqual(response.status_code, 422)

    def test_whitespace_only_input_returns_clear_client_error(self) -> None:
        response = self.client.post("/analyze", json={"conversation_text": "  \n"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Conversation text cannot be empty")

    @patch("backend.main.analyze_learning_conversation")
    def test_missing_configuration_returns_service_unavailable(self, analyze) -> None:
        analyze.side_effect = AnalyzerConfigurationError("OPENAI_API_KEY is missing")

        response = self.client.post(
            "/analyze", json={"conversation_text": "User: Explain CORS"}
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    @patch("backend.main.analyze_learning_conversation")
    def test_invalid_structured_output_returns_bad_gateway(self, analyze) -> None:
        analyze.side_effect = AnalyzerResponseError("invalid model payload")

        response = self.client.post(
            "/analyze", json={"conversation_text": "User: Explain CORS"}
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "The analyzer returned an invalid structured response",
        )

    @patch("backend.main.analyze_learning_conversation")
    def test_openai_api_error_returns_bad_gateway_without_internal_details(
        self, analyze
    ) -> None:
        analyze.side_effect = OpenAIError("secret upstream detail")

        response = self.client.post(
            "/analyze", json={"conversation_text": "User: Explain CORS"}
        )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)
        self.assertEqual(
            response.json()["detail"],
            "The OpenAI API request failed. Please try again later.",
        )

    @patch("backend.main.analyze_learning_conversation")
    def test_no_goal_threads_returns_unprocessable_content(self, analyze) -> None:
        analyze.side_effect = NoGoalThreadsError("No substantive Goal Thread was found")

        response = self.client.post(
            "/analyze", json={"conversation_text": "User: Continue"}
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("No substantive Goal Thread", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
