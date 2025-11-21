"""
Agent prompting users for additional details when the query is ambiguous.
"""

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class ClarificationAgent:
    """
    Generates clarifying follow-up prompts.
    """

    def run(self, user_input: str) -> Dict[str, str]:
        logger.info("ClarificationAgent invoked for query '%s'", user_input)
        return {
            "message": (
                "I need a bit more information to help you. "
                "Please specify the property addresses or metrics you are interested in."
            ),
            "original_query": user_input,
        }


__all__ = ["ClarificationAgent"]

