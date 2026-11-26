"""LLM clients.

OllamaClient  talks to an Ollama server (on your PC, Colab or Kaggle).
              mode="chat"       -> instruct models, /api/chat with a system prompt
              mode="completion" -> base models, /api/generate with raw=True (no chat template)
MockClient    deterministic fake answers for tests and dry runs; `optimism` shifts
              its answers so the analysis pipeline can be checked end to end.

All calls are logged by the caller; clients only return text.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class LLMClient(Protocol):
    name: str
    mode: str

    def generate(self, prompt: str, system: str | None = None, *, seed: int = 0,
                 temperature: float | None = None, max_tokens: int | None = None,
                 schema: dict | None = None) -> str: ...


@dataclass
class OllamaClient:
    name: str
    tag: str
    mode: str = "chat"
    url: str = "http://localhost:11434"
    timeout_s: float = 120
    temperature: float = 0.7
    max_tokens: int = 160
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.Client(base_url=self.url, timeout=self.timeout_s)

    def generate(self, prompt: str, system: str | None = None, *, seed: int = 0,
                 temperature: float | None = None, max_tokens: int | None = None,
                 schema: dict | None = None) -> str:
        options = {"seed": seed, "temperature": self.temperature if temperature is None else temperature,
                   "num_predict": max_tokens or self.max_tokens}
        if self.mode == "chat":
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            body: dict[str, Any] = {"model": self.tag, "messages": messages, "stream": False, "options": options}
            if schema:
                body["format"] = schema  # Ollama structured outputs: constrains the JSON shape
            r = self._http.post("/api/chat", json=body)
            r.raise_for_status()
            return r.json()["message"]["content"]
        # base model: plain text continuation, no chat template
        body = {"model": self.tag, "prompt": prompt, "raw": True, "stream": False, "options": options}
        r = self._http.post("/api/generate", json=body)
        r.raise_for_status()
        return r.json()["response"]

    def close(self) -> None:
        self._http.close()


@dataclass
class MockClient:
    """Fake LLM. Picks options from the prompt so every code path can be tested without a GPU."""
    name: str = "mock"
    mode: str = "chat"
    optimism: float = 0.0       # >0 favours hopeful / adapt answers, <0 the opposite
    latency_s: float = 0.0

    def generate(self, prompt: str, system: str | None = None, *, seed: int = 0,
                 temperature: float | None = None, max_tokens: int | None = None,
                 schema: dict | None = None) -> str:
        if self.latency_s:
            time.sleep(self.latency_s)
        h = int(hashlib.sha256(f"{self.name}|{seed}|{system}|{prompt}".encode()).hexdigest()[:12], 16)
        rng = random.Random(h)
        options = re.findall(r"^\s*([A-H])\)\s", prompt, flags=re.M)
        actions = re.findall(r'"([a-z_]+)"\s*:\s*\d', prompt)  # e.g. "hire": 0 in a JSON template
        choice_list = re.search(r"choose one of:\s*([a-z_|, ]+)", prompt)
        if self.mode == "completion" and options:
            return f"{self._pick(options, rng)}) and that is my answer."
        if options:
            return json.dumps({"answer": self._pick(options, rng), "reason": "Mock answer."})
        if choice_list:
            acts = [a.strip() for a in re.split(r"[|,]", choice_list.group(1)) if a.strip()]
            return json.dumps({"action": self._pick(acts, rng), "reason": "Mock decision."})
        if actions:
            # layoffs are rare, other counts small; enough to exercise every code path
            counts = {a: (int(rng.random() < 0.05) if "layoff" in a else rng.randint(0, 3)) for a in actions}
            return json.dumps(counts | {"reason": "Mock counts."})
        return json.dumps({"reason": "Mock."})

    def _pick(self, items: list[str], rng: random.Random) -> str:
        # optimism tilts the choice toward the first items (by convention the more hopeful / adaptive ones)
        weights = [1.0 + self.optimism * (len(items) - i - 1) / max(1, len(items) - 1) for i in range(len(items))]
        weights = [max(w, 0.05) for w in weights]
        return rng.choices(items, weights=weights)[0]


# ----------------------------------------------------------------------------- parsing

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a model answer (handles ```json fences and extra prose)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(t[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_letter(text: str, valid: list[str]) -> str | None:
    """First option letter in a base-model continuation, e.g. 'B) Somewhat worried' -> 'B'."""
    m = re.match(r"\s*\(?([A-H])\)?", text or "")
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"\b([A-H])\)", text or "")
    return m.group(1) if m and m.group(1) in valid else None


def make_client(model_key: str, cfg: dict, **mock_kwargs: Any) -> LLMClient:
    llm = cfg["llm"]
    spec = llm["models"].get(model_key, {"tag": model_key, "mode": "chat"})
    if llm["backend"] == "mock":
        return MockClient(name=model_key, mode=spec["mode"], **mock_kwargs)
    return OllamaClient(name=model_key, tag=spec["tag"], mode=spec["mode"], url=llm["ollama_url"],
                        timeout_s=llm["timeout_s"], temperature=llm["temperature"], max_tokens=llm["max_tokens"])
