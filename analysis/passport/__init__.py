"""Deterministic HTML rendering for patient state descriptions (describe, abstain, or refuse modes)."""
from .render import render_passport, build_passport, PassportState

__all__ = ["render_passport", "build_passport", "PassportState"]
