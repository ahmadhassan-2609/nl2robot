import anthropic
import json
import re

from planner.prompts import SYSTEM_PROMPT

client = anthropic.Anthropic()


def plan_task(command: str, scene_state: dict) -> dict:
    """
    Sends the command + scene state to Claude and returns a parsed action plan.
    Raises ValueError if the LLM response cannot be parsed as JSON.
    """
    prompt = SYSTEM_PROMPT.format(scene_state=json.dumps(scene_state, indent=2))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=prompt,
        messages=[{"role": "user", "content": command}],
    )

    raw = response.content[0].text

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract a JSON block from surrounding text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"LLM returned invalid JSON:\n{raw}")
