"""Adapter registry + sniff dispatch."""

from __future__ import annotations

from benchgauge.ingest.adapters.base import Adapter
from benchgauge.ingest.adapters.lm_eval import LmEvalAdapter
from benchgauge.ingest.adapters.open_llm_leaderboard import OpenLLMLeaderboardAdapter

# Order matters only for display; sniff() must be mutually exclusive enough that
# at most one matches a given path (ambiguity -> abstain in ingest.load_evallog).
# The Adapter annotation makes the structural contract (name/sniff/load) checked.
ADAPTERS: list[Adapter] = [LmEvalAdapter(), OpenLLMLeaderboardAdapter()]
ADAPTERS_BY_NAME: dict[str, Adapter] = {a.name: a for a in ADAPTERS}


def get_adapter(name: str) -> Adapter:
    if name not in ADAPTERS_BY_NAME:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS_BY_NAME)}")
    return ADAPTERS_BY_NAME[name]
