"""
Agent prompting users for additional details when the query is ambiguous.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ClarificationAgent:
    """
    Generates clarifying follow-up prompts.
    """

    def run(self, user_input: str) -> Dict[str, str]:
        return {
            "message": (
                "I need a bit more information to help you. "
                "Please specify the property addresses or metrics you are interested in."
            ),
            "original_query": user_input,
        }


__all__ = ["ClarificationAgent"]

