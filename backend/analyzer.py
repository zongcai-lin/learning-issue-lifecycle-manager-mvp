import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_learning_conversation(conversation_text: str) -> dict:
    prompt = f"""
You are analyzing an AI-assisted learning conversation.

Your task:
Extract the user's learning structure from the conversation.

Return ONLY valid JSON with this exact structure:
{{
  "main_goal": "string",
  "issues": [
    {{
      "title": "string",
      "type": "main_issue | blocker | sub_issue | curiosity_branch | related_concept",
      "status": "open | resolved | parked | suggested",
      "trigger": "string",
      "summary": "string",
      "final_understanding": "string",
      "review_note": "string"
    }}
  ]
}}

Definitions:
- main_issue: the main problem the user is trying to solve
- blocker: a problem that must be solved before the main goal can continue
- sub_issue: a smaller question, diagnostic step, or follow-up under another issue
- curiosity_branch: a related side question that the user has already asked or explored, but it is not mandatory to continue the main goal
- related_concept: a concept that may be useful for later understanding, but the user has not necessarily asked about it directly

Status definitions:
- open: the issue is still active or not fully resolved
- resolved: the issue has been answered, fixed, or confirmed by the user
- parked: the user has touched this issue, but it should not be expanded now
- suggested: the system thinks this concept may be useful later, but the user has not confirmed interest

Input format note:
The conversation may contain multiple turns between User and Assistant.
Use the User messages to identify the user's actual confusion, goals, follow-up questions, and unresolved issues.
Use the Assistant messages to infer whether an issue was explained, resolved, or still needs review.

Granularity rules:
- Treat each distinct User question, confusion, debugging obstacle, or follow-up as a candidate issue.
- Do not merge different user confusions only because they belong to the same broad topic.
- If one User message contains multiple separate questions, split them into multiple issues when they require different explanations.
- Use main_issue only when an issue directly represents the central problem the user is trying to solve. Do not label diagnostic checks or setup steps as main_issue; use sub_issue instead.
- The top-level main_goal already represents the user's central goal. Do not force an issue to be main_issue. Use main_issue only when the user explicitly discusses a central unresolved problem that is separate from the top-level main_goal.
- Do not create issues from workflow/navigation messages such as "what should I do next", "where should I continue", "this step is done", or "continue", unless the user also asks a specific technical, conceptual, or product-design question.
- A diagnostic step can be a sub_issue if it helps the user solve the main goal.
- A blocker is something that prevents the user from continuing the main goal unless it is understood or fixed.
- Use Assistant messages to decide whether an issue is resolved, still open, or parked.
- For a realistic 5-12 turn learning/debugging conversation, prefer 3-8 issues if there are enough distinct user questions.
- Do not create related_concept unless the conversation explicitly mentions a useful concept that the user has not directly explored.

- If the User explicitly asks about a concept, term, syntax, function, error, or code line, do not classify it as related_concept. Use sub_issue, blocker, or curiosity_branch depending on its role.
- When the User asks "what is X", "why use X", "what does this line do", or "what is the difference between X and Y", treat that as a separate candidate issue unless it is truly trivial or already covered by another issue.

Conversation:
{conversation_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a strict learning issue lifecycle analyzer. Return only JSON."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)