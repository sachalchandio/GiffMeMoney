"""In-memory session store for the Real-Time mode (paper only).

Sessions are ephemeral, per-browser-session simulation state — there is nothing
to persist (no real money, no account-critical data), so a process-local,
thread-safe dict is exactly right. A small cap evicts the oldest sessions so a
long-lived server can't grow unbounded.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from app.livesim.engine import LiveSimSession, create_session, tick

__all__ = ["LiveSimStore", "get_store"]

#: Most concurrent sessions kept in memory before the oldest is evicted.
_MAX_SESSIONS: int = 256


class LiveSimStore:
    """A thread-safe, capped store of live-sim sessions."""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, LiveSimSession]" = OrderedDict()
        self._max = max(8, int(max_sessions))

    def start(self, **config) -> LiveSimSession:
        """Create a new session with a fresh id and store it."""
        session_id = uuid.uuid4().hex[:16]
        session = create_session(session_id, **config)
        with self._lock:
            self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        return session

    def get(self, session_id: str) -> LiveSimSession | None:
        """Return a session by id (and mark it most-recently-used), or ``None``."""
        with self._lock:
            session = self._sessions.get(str(session_id))
            if session is not None:
                self._sessions.move_to_end(str(session_id))
            return session

    def tick(self, session_id: str, steps: int | None = None) -> LiveSimSession | None:
        """Advance a session by ``steps``; returns the session or ``None`` if missing."""
        session = self.get(session_id)
        if session is None:
            return None
        return tick(session, steps)

    def stop(self, session_id: str) -> bool:
        """Remove a session. Returns ``True`` if one was removed."""
        with self._lock:
            return self._sessions.pop(str(session_id), None) is not None

    def count(self) -> int:
        """Number of live sessions held."""
        with self._lock:
            return len(self._sessions)


_STORE: LiveSimStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> LiveSimStore:
    """Return the process-wide :class:`LiveSimStore` singleton."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = LiveSimStore()
    return _STORE
