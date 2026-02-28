"""
nsys_tui.agent — The nsys-ai agent: a CUDA ML systems performance expert.

This package provides:
    persona.py  — Agent identity, system prompt, knowledge layers
    loop.py     — Core analysis loop: profile → skill selection → execution → report
"""
from .persona import SYSTEM_PROMPT, AGENT_IDENTITY
from .loop import Agent

__all__ = ["SYSTEM_PROMPT", "AGENT_IDENTITY", "Agent"]
