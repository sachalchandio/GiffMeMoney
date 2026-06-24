"""FastAPI application factory and ASGI entry point for GiffMeMoney.

This module wires the whole backend together:

* creates the :class:`fastapi.FastAPI` app (title ``GiffMeMoney``);
* configures CORS from :data:`app.config.settings.cors_origins` (plus wildcard
  methods/headers, per section 5 of the contract);
* mounts the five REST routers under the ``/api`` prefix
  (``market``, ``assets``, ``recommendations``, ``strategies``, ``portfolio``);
* exposes the ``/ws`` WebSocket endpoint implementing the section-6 protocol
  (snapshot on connect, subscribe/unsubscribe actions, periodic ticks and
  heartbeats) via :class:`app.market.feed.ConnectionManager`;
* runs :func:`app.market.feed.price_tick_loop` as a background task for the
  lifetime of the process using the lifespan context manager.

The exported ASGI app is ``app`` (referenced by ``uvicorn app.main:app`` and by
``run.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

from app.api import (
    assets as assets_api,
    auth as auth_api,
    bot as bot_api,
    broker as broker_api,
    invest as invest_api,
    market as market_api,
    portfolio as portfolio_api,
    recommendations as recommendations_api,
    strategies as strategies_api,
)
from app.config import settings
from app.market.feed import ConnectionManager, LivePriceBook, price_tick_loop
from app.market.provider import get_provider

__all__ = ["app", "create_app", "manager"]


# A single process-wide connection manager shared by the ``/ws`` endpoint and
# the background tick loop.
manager = ConnectionManager()


# ---------------------------------------------------------------------------
# OpenAPI / docs metadata
# ---------------------------------------------------------------------------

#: Rich, multi-paragraph API description rendered at the top of the docs page.
API_DESCRIPTION = """\
**GiffMeMoney** is an intelligent investment-advisory API that runs 18+ quant
models over a deterministic market simulator. It powers asset analysis, ranked
recommendations, a strategy library with backtesting, Markowitz portfolio
optimization, and a fully simulated paper-trading brokerage (wallet, positions,
P&L history and an allocation advisor).

Every JSON payload is **camelCase** on the wire (request bodies accept both
camelCase and snake_case). Concrete request/response examples are attached to
each schema below.

**Authentication.** Sign up or log in under `auth` to receive a signed JWT, then
send it as `Authorization: Bearer <token>`. On the `invest`/`wallet`/`advisor`
routes the token selects the caller's own isolated account; anonymous callers
may instead pass an `X-Account-Id` header (or fall back to the shared `demo`
account).

**Real-time feed.** A WebSocket lives at `/ws` (not shown below): it pushes a
full price snapshot on connect, then `subscribe`/`unsubscribe`/`set`/`ping`
control messages drive periodic price ticks and heartbeats.

> _Educational simulation on synthetic market data — not financial advice._
"""

#: Tag-group metadata: ordering + per-section descriptions for the docs sidebar.
OPENAPI_TAGS: List[Dict[str, str]] = [
    {"name": "market", "description": "Live prices, candles and the market summary dashboard."},
    {"name": "assets", "description": "The tradable universe and per-asset composite analysis."},
    {"name": "recommendations", "description": "Ranked investment ideas across the universe."},
    {"name": "strategies", "description": "The strategy catalog, cross-asset rankings and backtests."},
    {"name": "portfolio", "description": "Markowitz mean-variance optimization and the simulated portfolio view."},
    {"name": "invest", "description": "The simulated brokerage: wallet, positions, P&L history and advisor."},
    {"name": "wallet", "description": "Cash, cards and the account ledger (simulated funding)."},
    {"name": "advisor", "description": "Risk-profiled allocation advice for a dollar amount."},
    {"name": "bot", "description": "Simulated paper-trading auto-trader: preset modes, backtests and side-by-side comparison (synthetic data, no real funds)."},
    {"name": "broker", "description": "Pluggable broker execution layer (go-live, opt-in). Ships in simulated paper mode with live trading hard-gated OFF; every payload carries a paper flag + disclaimer."},
    {"name": "auth", "description": "Email/password signup, login and the current-user probe."},
]


def _scalar_html(openapi_url: str, title: str) -> str:
    """Render the Scalar API-reference page for a given OpenAPI document.

    Scalar is a modern, themeable replacement for Swagger UI. It is loaded from
    the public CDN at view time (no extra Python dependency); the page simply
    points it at our generated ``/openapi.json``.

    Args:
        openapi_url: URL of the OpenAPI document to render (e.g. ``/openapi.json``).
        title: Page ``<title>``.

    Returns:
        A complete standalone HTML document as a string.
    """
    return f"""<!doctype html>
<html>
  <head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {{ margin: 0; }}
    </style>
  </head>
  <body>
    <script id="api-reference" data-url="{openapi_url}"></script>
    <script>
      var configuration = {{ theme: "purple", layout: "modern", darkMode: true }};
      var el = document.getElementById("api-reference");
      el.dataset.configuration = JSON.stringify(configuration);
    </script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""


def _normalize_symbols(raw: Any) -> Optional[List[str]]:
    """Coerce a client-supplied ``symbols`` payload into a clean symbol list.

    Args:
        raw: The ``symbols`` value from an inbound WebSocket message. Expected to
            be a list of strings, but defensively handles a bare string or
            missing/None value.

    Returns:
        A list of upper-cased, non-empty symbols, or ``None`` if nothing usable
        was provided (which the caller treats as "no change / all symbols").
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            sym = item.strip().upper()
            if sym:
                out.append(sym)
    return out or None


def _mount(application: FastAPI, api_router: APIRouter) -> None:
    """Include a router so its routes resolve under the ``/api`` prefix exactly once.

    The contract (section 5) puts every REST route under ``/api``. Some sibling
    router modules bake ``/api`` into their own ``APIRouter(prefix=...)`` and some
    don't. To produce the correct final paths in both cases, this helper adds the
    ``/api`` prefix only when the router's existing prefix doesn't already start
    with ``/api`` — avoiding a doubled ``/api/api/...`` path.

    Args:
        application: The FastAPI app to mount onto.
        api_router: The router to include.
    """
    existing_prefix = getattr(api_router, "prefix", "") or ""
    if existing_prefix == "/api" or existing_prefix.startswith("/api/"):
        application.include_router(api_router)
    else:
        application.include_router(api_router, prefix="/api")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown: launch and cleanly stop the price-tick loop.

    On startup a :class:`asyncio.Event` and a background task running
    :func:`app.market.feed.price_tick_loop` are created and stored on the app
    state. On shutdown the event is set and the task is awaited (cancelled if it
    does not exit promptly) so no work leaks past the process lifetime.

    Args:
        application: The FastAPI application being started.

    Yields:
        Control back to the server while the app serves requests.
    """
    provider = get_provider()
    stop_event = asyncio.Event()
    application.state.stop_event = stop_event
    application.state.manager = manager
    application.state.provider = provider
    # A live price book exposed for the /ws snapshot on connect (kept in sync by
    # the tick loop which maintains its own book; this one is only read for the
    # initial snapshot so a fresh connection sees current prices immediately).
    application.state.tick_task = asyncio.create_task(
        price_tick_loop(manager, provider, stop_event)
    )

    try:
        yield
    finally:
        stop_event.set()
        task: Optional[asyncio.Task] = getattr(application.state, "tick_task", None)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Wires CORS, mounts the five ``/api`` routers, and registers the ``/ws``
    WebSocket endpoint. The background tick loop is started by the ``lifespan``
    handler attached here.

    Returns:
        The fully-configured :class:`fastapi.FastAPI` instance.
    """
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary=(
            "Quant investment-advisory API: analysis, recommendations, "
            "strategies, portfolio optimization and a simulated brokerage."
        ),
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        contact={"name": "GiffMeMoney", "url": "https://github.com/"},
        license_info={"name": "Educational / sandbox use"},
        # The default Swagger UI is replaced by a modern Scalar page served at
        # ``/docs`` below; ReDoc and the raw ``/openapi.json`` stay on defaults.
        docs_url=None,
        lifespan=lifespan,
    )

    # CORS: explicit Vite dev origins from settings, wildcard methods/headers.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the five REST routers so every final path is exactly the contract
    # path under ``/api`` (section 5). Sibling router modules are inconsistent
    # about whether they bake ``/api`` into their own ``APIRouter(prefix=...)``:
    # some do (e.g. ``prefix="/api"``), some don't. ``_mount`` adds the ``/api``
    # prefix only when the router doesn't already carry it, so we never end up
    # with a doubled ``/api/api/...`` path regardless of which convention a
    # given module chose.
    for api_router in (
        market_api.router,
        assets_api.router,
        recommendations_api.router,
        strategies_api.router,
        portfolio_api.router,
        invest_api.router,
        bot_api.router,
        broker_api.router,
        auth_api.router,
    ):
        _mount(application, api_router)

    @application.get("/docs", include_in_schema=False)
    async def scalar_docs() -> HTMLResponse:
        """Serve the modern Scalar API-reference UI (replacing Swagger UI).

        Renders the Scalar standalone reference against this app's generated
        ``/openapi.json``. Excluded from the schema itself so it does not appear
        as an API operation.

        Returns:
            The Scalar HTML page.
        """
        return HTMLResponse(
            _scalar_html(
                openapi_url=application.openapi_url or "/openapi.json",
                title=f"{application.title} — API Reference",
            )
        )

    @application.get("/swagger", include_in_schema=False)
    async def swagger_docs() -> HTMLResponse:
        """Serve the classic Swagger UI against this app's ``/openapi.json``.

        The default ``/docs`` route is replaced by the modern Scalar page, so
        this route preserves the familiar Swagger UI for callers that still
        prefer it. Excluded from the schema itself so it does not appear as an
        API operation.

        Returns:
            The Swagger UI HTML page.
        """
        return get_swagger_ui_html(
            openapi_url=application.openapi_url or "/openapi.json",
            title=f"{application.title} — Swagger UI",
        )

    @application.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        """Serve the live market feed (section-6 protocol).

        On connect the full-universe price snapshot is sent. The client may then
        send ``{"action":"subscribe"|"unsubscribe","symbols":[...]}`` messages to
        narrow the stream; ticks and heartbeats are pushed by the background
        :func:`price_tick_loop`. The default subscription is the whole universe.

        Args:
            websocket: The inbound Starlette/FastAPI WebSocket connection.
        """
        await manager.connect(websocket)
        try:
            # Initial full-universe snapshot built from current live prices.
            provider = get_provider()
            book = LivePriceBook(provider)
            snapshot = {
                "type": "snapshot",
                "data": [p.model_dump(by_alias=True) for p in book.points()],
            }
            await manager.send_json(websocket, snapshot)

            # Listen for subscribe/unsubscribe control messages until the client
            # disconnects. Tick/heartbeat pushes are handled by the loop task.
            while True:
                message = await websocket.receive_json()
                if not isinstance(message, dict):
                    continue
                action = str(message.get("action", "")).strip().lower()
                symbols = _normalize_symbols(message.get("symbols"))

                if action == "subscribe":
                    if symbols is None:
                        # Subscribe to everything.
                        await manager.set_subscription(websocket, None)
                    else:
                        await manager.subscribe(websocket, symbols)
                elif action == "unsubscribe":
                    if symbols is not None:
                        await manager.unsubscribe(websocket, symbols)
                elif action == "set":
                    # Convenience: replace the subscription set wholesale.
                    await manager.set_subscription(websocket, symbols)
                elif action == "ping":
                    await manager.send_json(
                        websocket, {"type": "pong", "t": int(time.time() * 1000)}
                    )
                # Unknown actions are ignored (forward-compatible).
        except WebSocketDisconnect:
            pass
        except Exception:
            # Any unexpected error: close gracefully without crashing the server.
            pass
        finally:
            await manager.disconnect(websocket)
            if websocket.client_state != WebSocketState.DISCONNECTED:
                with contextlib.suppress(Exception):
                    await websocket.close()

    return application


# The ASGI application instance imported by uvicorn / run.py.
app = create_app()
