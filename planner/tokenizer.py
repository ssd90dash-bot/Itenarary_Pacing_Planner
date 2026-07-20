"""Local token counting with `tiktoken`, and comparison against billed counts.

Two independent ways of counting exist here, and they deliberately disagree:

  * **`tiktoken`** encodes text locally with the model's BPE vocabulary. Free,
    instant, works before a call is made — so it is what you use to *predict*
    cost or to guard against oversized prompts.
  * **The API's `usage` block** is what you are actually billed for.

The local count comes in *below* the billed count, because a structured-output
request has the JSON schema and message scaffolding added server-side. That gap
is a real, measurable overhead rather than an error in either number, and
`compare()` exists to quantify it.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

# Current OpenAI models encode with o200k_base; older GPT-3.5/4 used cl100k_base.
FALLBACK_ENCODING = "o200k_base"

# Chat messages are not just their text: each carries role/delimiter tokens, and
# the reply is primed with a few more. These are the documented constants for
# the chat format.
TOKENS_PER_MESSAGE = 3
TOKENS_PER_REPLY_PRIMING = 3


def encoding_for(model: str):
    """Get the model's encoding, falling back for unknown or future models."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def count_text(text: str, model: str = "gpt-4o-mini") -> int:
    return len(encoding_for(model).encode(text))


def count_messages(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Count a chat payload including per-message and reply-priming overhead."""
    encoding = encoding_for(model)
    total = TOKENS_PER_REPLY_PRIMING
    for message in messages:
        total += TOKENS_PER_MESSAGE
        for key, value in message.items():
            if isinstance(value, str):
                total += len(encoding.encode(value))
                if key == "name":
                    total += 1
    return total


@dataclass
class TokenComparison:
    """Local estimate vs what the API actually billed."""

    local_estimate: int
    api_reported: int

    @property
    def delta(self) -> int:
        """Positive means the API billed more than we counted locally."""
        return self.api_reported - self.local_estimate

    @property
    def delta_pct(self) -> float | None:
        if not self.local_estimate:
            return None
        return 100.0 * self.delta / self.local_estimate

    @property
    def explanation(self) -> str:
        if self.delta > 0:
            return (
                f"The API billed {self.delta:,} more input tokens than a local "
                "count of the prompt text. That gap is the structured-output "
                "JSON schema and message scaffolding, which are added server-side "
                "and never appear in the strings we encode locally."
            )
        if self.delta < 0:
            return (
                f"The local count is {abs(self.delta):,} tokens higher than billed "
                "— usually prompt caching or a server-side optimisation."
            )
        return "Local and billed counts agree exactly."


def compare(local_estimate: int, api_reported: int) -> TokenComparison:
    return TokenComparison(local_estimate=local_estimate, api_reported=api_reported)
