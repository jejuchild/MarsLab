"""Backward-compatible re-export of AnalysisModule from shared package."""
from analysis.shared.base import AnalysisModule  # noqa: F401

__all__ = ["AnalysisModule"]
