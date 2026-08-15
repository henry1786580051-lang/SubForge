"""MiniMax Anthropic-compatible client with explicit prompt caching."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx

PROMPT_CACHE_LOCAL_TTL = 240.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192


@dataclass
class _PromptCacheGate:
    condition: threading.Condition = field(default_factory=threading.Condition)
    creating: bool = False
    ready_at: float = 0.0


class _CompletionsAdapter:
    def __init__(self, owner: "MiniMaxAnthropicClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner.create_message(**kwargs)


class _ChatAdapter:
    def __init__(self, owner: "MiniMaxAnthropicClient") -> None:
        self.completions = _CompletionsAdapter(owner)


class MiniMaxAnthropicClient:
    """Expose Anthropic Messages through the app's existing chat interface."""

    is_minimax_anthropic = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        http_client: httpx.Client,
    ) -> None:
        self._client = anthropic.Anthropic(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
        self.chat = _ChatAdapter(self)
        self.models = self._client.models
        self._gates: dict[str, _PromptCacheGate] = {}
        self._gates_lock = threading.Lock()

    def close(self) -> None:
        """Release the underlying Anthropic/httpx connection pool."""
        self._client.close()

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        system_parts: list[str] = []
        conversation: list[dict] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            conversation.append({"role": role, "content": content})

        system: list[dict] = []
        if system_parts:
            system.append(
                {
                    "type": "text",
                    "text": "\n\n".join(system_parts),
                    "cache_control": {"type": "ephemeral"},
                }
            )
        return system, conversation

    @staticmethod
    def _cache_key(model: str, system: list[dict]) -> str:
        text = "\n".join(str(block.get("text") or "") for block in system)
        return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _supports_explicit_prompt_cache(model: str) -> bool:
        """Match only the MiniMax models currently listed for active caching."""
        normalized = re.sub(r"[^a-z0-9]+", "", str(model or "").lower())
        return bool(re.fullmatch(r"minimaxm2(?:1|5|7)?(?:highspeed)?", normalized))

    def _gate_for(self, key: str) -> _PromptCacheGate:
        with self._gates_lock:
            return self._gates.setdefault(key, _PromptCacheGate())

    @staticmethod
    def _claim_cache_creation(gate: _PromptCacheGate) -> bool:
        with gate.condition:
            while gate.creating:
                gate.condition.wait()
            if gate.ready_at and time.monotonic() - gate.ready_at < PROMPT_CACHE_LOCAL_TTL:
                return False
            gate.creating = True
            return True

    @staticmethod
    def _finish_cache_creation(gate: _PromptCacheGate, succeeded: bool) -> None:
        with gate.condition:
            gate.creating = False
            if succeeded:
                gate.ready_at = time.monotonic()
            gate.condition.notify_all()

    def create_message(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1,
        **kwargs: Any,
    ) -> Any:
        system, conversation = self._convert_messages(messages)
        explicit_cache = bool(system) and self._supports_explicit_prompt_cache(model)
        if system and not explicit_cache:
            system[0].pop("cache_control", None)
        request = {
            "model": model,
            "messages": conversation,
            "system": system,
            "temperature": temperature,
            "max_tokens": int(
                kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
            ),
            **kwargs,
        }
        if not explicit_cache:
            return self._client.messages.create(**request)

        gate = self._gate_for(self._cache_key(model, system))
        creator = self._claim_cache_creation(gate)
        if not creator:
            return self._client.messages.create(**request)

        succeeded = False
        try:
            response = self._client.messages.create(**request)
            succeeded = True
            return response
        finally:
            self._finish_cache_creation(gate, succeeded)
