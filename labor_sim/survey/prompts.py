"""Prompt formats for Experiment 1.

instruct: chat format with a fixed system prompt, JSON answer.
base:     survey-completion transcript (Argyle et al. 2023); the model continues
          the respondent's line and we read the option letter it writes.
"""
from __future__ import annotations

import string

from labor_sim.survey.personas import Persona

SYSTEM_PROMPT = (
    "You are taking part in a survey of working adults in the United States. "
    "Answer as the person described, based on their situation, not as an AI assistant. "
    "There are no right or wrong answers."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["answer", "reason"],
}


def letters(n: int) -> list[str]:
    return list(string.ascii_uppercase[:n])


def format_options(options: list[dict]) -> str:
    return "\n".join(f"{l}) {o['label']}" for l, o in zip(letters(len(options)), options))


def instruct_prompt(persona: Persona, question: dict) -> tuple[str, str]:
    user = (f"About you: {persona.backstory()}\n\n"
            f"Question: {question['text']}\n{format_options(question['options'])}\n\n"
            'Reply only with JSON: {"answer": "<option letter>", "reason": "<one sentence>"}')
    return SYSTEM_PROMPT, user


def base_prompt(persona: Persona, question: dict) -> str:
    return ("The following is a transcript of an interview from a 2024 survey of working adults "
            "in the United States.\n\n"
            "Interviewer: Please tell me a little about yourself.\n"
            f"Respondent: {persona.backstory()}\n\n"
            f"Interviewer: {question['text']}\n{format_options(question['options'])}\n"
            "Respondent: I would choose option ")
