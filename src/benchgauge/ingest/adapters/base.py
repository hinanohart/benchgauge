"""Adapter contract: sniff a path, load it into an EvalLog."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from benchgauge.model import EvalLog


@runtime_checkable
class Adapter(Protocol):
    name: str

    def sniff(self, path: Path) -> bool:
        """Cheap structural check: does this path look like our format?"""
        ...

    def load(self, path: Path) -> EvalLog:
        """Parse the path into an EvalLog (or raise AbstainError/InputError)."""
        ...
