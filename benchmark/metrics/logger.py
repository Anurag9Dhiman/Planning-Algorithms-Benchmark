"""Writes episode results to JSONL and aggregated summaries to JSON."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

from benchmark.metrics.evaluator import AggregateResult, EpisodeResult


class ResultLogger:
    def __init__(self, run_dir: str):
        self._dir = Path(run_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ep_file = self._dir / "episodes.jsonl"
        self._summary_file = self._dir / "summary.json"
        self._summaries: List[dict] = []

    def log_episode(self, result: EpisodeResult) -> None:
        try:
            with open(self._ep_file, "a") as f:
                f.write(json.dumps(_serialise(asdict(result))) + "\n")
        except OSError as e:
            print(f"[logger] WARNING: could not write episode ({e})")

    def log_aggregate(self, agg: AggregateResult) -> None:
        self._summaries.append(_serialise(asdict(agg)))
        try:
            with open(self._summary_file, "w") as f:
                json.dump(self._summaries, f, indent=2)
        except OSError as e:
            print(f"[logger] WARNING: could not write summary ({e})")

    @property
    def run_dir(self) -> Path:
        return self._dir


def _serialise(obj):
    """Recursively convert numpy / dataclass types to JSON-serialisable forms."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalar (int64, float32, …)
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):  # nested dataclass
        return _serialise(dataclasses.asdict(obj))
    return obj
