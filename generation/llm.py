"""
llm.py — Anthropic Claude API wrapper.

Stateless: each call is independent.
Supports streaming for the UI, non-streaming for the eval pipeline.
"""
import os
from typing import Iterator, Optional


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> dict:
        """Non-streaming generation. Returns full response + usage stats."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": self.model,
        }

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Streaming generation. Yields text chunks as they arrive."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text
