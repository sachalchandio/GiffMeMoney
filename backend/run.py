"""Development entry point for the GiffMeMoney backend.

Run ``python run.py`` from the ``backend/`` directory to start a hot-reloading
uvicorn server on ``http://0.0.0.0:8000`` (interactive docs at ``/docs``,
WebSocket feed at ``/ws``).

This is a thin convenience wrapper; in production you would invoke uvicorn (or
gunicorn with uvicorn workers) directly, e.g. ``uvicorn app.main:app``.
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the uvicorn development server with auto-reload enabled.

    Binds to all interfaces on port 8000 and serves the ASGI app exported as
    ``app.main:app``. ``reload=True`` watches the source tree so edits take
    effect without a manual restart.
    """
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
