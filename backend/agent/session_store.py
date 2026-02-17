"""
Session persistence for workflow sessions.

In-memory cache with JSON file persistence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from agent.schemas import WorkflowSession

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 30


class SessionStore:
    """Persist workflow sessions to disk with in-memory cache."""

    def __init__(self, data_dir: str = "backend/data/workflow_sessions") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, WorkflowSession] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing sessions from disk into cache."""
        for path in sorted(self.data_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                session = WorkflowSession.model_validate_json(path.read_text())
                self._cache[session.session_id] = session
            except Exception as e:
                logger.warning(f"Failed to load session {path.name}: {e}")

    def save(self, session: WorkflowSession) -> None:
        """Save session to cache and disk."""
        session.updated_at = datetime.now(timezone.utc)
        self._cache[session.session_id] = session

        path = self.data_dir / f"{session.session_id}.json"
        try:
            path.write_text(session.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

        # LRU eviction
        if len(self._cache) > _MAX_SESSIONS:
            oldest_id = min(
                self._cache,
                key=lambda sid: self._cache[sid].updated_at,
            )
            self.delete(oldest_id)

    def load(self, session_id: str) -> Optional[WorkflowSession]:
        """Load session from cache or disk."""
        if session_id in self._cache:
            return self._cache[session_id]

        path = self.data_dir / f"{session_id}.json"
        if path.exists():
            try:
                session = WorkflowSession.model_validate_json(path.read_text())
                self._cache[session_id] = session
                return session
            except Exception as e:
                logger.warning(f"Failed to load session {session_id}: {e}")
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """List sessions sorted by most recent first."""
        sessions = sorted(
            self._cache.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )[:limit]

        return [
            {
                "session_id": s.session_id,
                "user_goal": s.user_goal,
                "state": s.state,
                "agent_type": s.agent_type,
                "created_at": s.created_at.isoformat(),
                "step_count": len(s.steps),
                "completed_steps": s.completed_step_count,
            }
            for s in sessions
        ]

    def delete(self, session_id: str) -> None:
        """Remove a session from cache and disk."""
        self._cache.pop(session_id, None)
        path = self.data_dir / f"{session_id}.json"
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


# Singleton
session_store = SessionStore()
