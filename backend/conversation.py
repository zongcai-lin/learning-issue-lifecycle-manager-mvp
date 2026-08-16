import re

from backend.schemas import ConversationTurn


ROLE_MARKER = re.compile(
    r"^\s*(?:\*\*)?(User|Assistant)\s*:(?:\*\*)?\s*(.*)$",
    re.IGNORECASE,
)


def parse_conversation_turns(conversation_text: str) -> list[ConversationTurn]:
    """Parse exported User/Assistant sections into ordered, stable turns.

    Text before the first role marker is treated as context rather than a turn.
    Role-like lines inside fenced code blocks remain part of the current turn.
    """

    if not conversation_text.strip():
        raise ValueError("Conversation text cannot be empty")

    turns: list[ConversationTurn] = []
    current_role: str | None = None
    current_lines: list[str] = []
    in_code_fence = False

    def append_current_turn() -> None:
        if current_role is None:
            return

        content = "\n".join(current_lines).strip()
        turn_id = f"T{len(turns) + 1}"
        if not content:
            raise ValueError(f"{turn_id} ({current_role}) has no content")

        turns.append(
            ConversationTurn(
                turn_id=turn_id,
                role=current_role,
                content=content,
            )
        )

    for line in conversation_text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence

        marker = None if in_code_fence else ROLE_MARKER.match(line)
        if marker:
            append_current_turn()
            current_role = marker.group(1).lower()
            inline_content = marker.group(2)
            current_lines = [inline_content] if inline_content else []
            continue

        if current_role is not None:
            current_lines.append(line)

    append_current_turn()

    if not turns:
        raise ValueError("No User: or Assistant: conversation turns were found")

    return turns
