"""Application configuration for the GiffMeMoney backend.

Settings are environment-driven via ``pydantic-settings``. Every field has a
sane default so the application boots with **no environment variables and no
third-party API keys** — it runs entirely on the built-in market simulator.

Environment variables are matched case-insensitively to the field names
(e.g. ``RISK_FREE_RATE=0.05``, ``PROVIDER=simulated``). Optional provider API
keys are read from the environment when present so a real ``MarketDataProvider``
adapter can be dropped in later without code changes.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource


class _CsvEnvSource(EnvSettingsSource):
    """Env source that accepts comma-separated strings for ``list`` fields.

    Pydantic-settings JSON-decodes "complex" fields (lists, dicts) directly
    from the environment. This subclass first lets a bare comma-separated
    value (e.g. ``a.com,b.com``) be parsed into a list, while still allowing a
    proper JSON array. Only ``cors_origins`` is treated specially.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        """Split comma-separated CORS origins before JSON decoding.

        Args:
            field_name: Name of the settings field.
            field: Field metadata.
            value: Raw environment value.
            value_is_complex: Whether pydantic considers the field complex.

        Returns:
            A parsed list for ``cors_origins`` given a plain CSV string,
            otherwise the default pydantic-settings behavior.
        """
        if (
            field_name == "cors_origins"
            and isinstance(value, str)
            and not value.strip().startswith("[")
        ):
            return [item.strip() for item in value.split(",") if item.strip()]
        return super().prepare_field_value(
            field_name, field, value, value_is_complex
        )


class Settings(BaseSettings):
    """Runtime configuration for the API.

    Attributes:
        app_name: Human-readable application name.
        cors_origins: Allowed CORS origins for the Vite dev server.
        risk_free_rate: Annual risk-free rate (decimal, e.g. ``0.04`` = 4%).
        tick_interval_ms: Interval between live WebSocket price ticks, in ms.
        provider: Market-data provider key used by ``get_provider()``.
        history_days: Number of daily closes the simulator generates per asset.
        jwt_secret: HMAC signing secret for auth JWTs. Defaults to a **dev**
            secret — this is a sandbox/demo app, so override it via the
            ``JWT_SECRET`` environment variable in any non-toy deployment.
        jwt_expire_days: Lifetime of an issued auth token, in days.
        finnhub_api_key: Optional API key for a future Finnhub adapter.
        polygon_api_key: Optional API key for a future Polygon adapter.
        coingecko_api_key: Optional API key for a future CoinGecko adapter.
        binance_api_key: Optional API key for a future Binance adapter.
        broker: Broker execution backend key (``'simulated'`` default,
            ``'alpaca'`` opt-in). Selected by ``get_broker()``.
        alpaca_api_key: Optional Alpaca API key id (None on the simulator).
        alpaca_secret_key: Optional Alpaca API secret (None on the simulator).
        alpaca_base_url: Alpaca REST base URL. **Defaults to the PAPER
            (sandbox) endpoint** — no real money — so the broker is safe even
            when keys are present. Only a deliberate override to the live host
            plus the live gates below ever places real orders.
        alpaca_live: Master live-trading flag. ``False`` by default; live is
            additionally gated on real keys and an explicit ``broker_ack``.
        broker_ack: Free-text live-trading acknowledgement. Live orders are only
            permitted when this exactly equals
            ``"I understand this places real orders"``.
        persist: Persistence backend key (``'memory'`` default, ``'sqlite'``
            opt-in). SQLite initializes its own new database file only.
        db_url: SQLAlchemy database URL used when ``persist == 'sqlite'``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "GiffMeMoney"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    risk_free_rate: float = 0.04
    tick_interval_ms: int = 1000
    provider: str = "simulated"
    history_days: int = 1300

    # Authentication (sandbox/demo): real hashed passwords + signed JWTs, but a
    # dev secret by default. Override JWT_SECRET in any non-toy deployment.
    jwt_secret: str = "dev-secret-change-me"
    jwt_expire_days: int = 7

    # Optional provider credentials (None when running on the simulator).
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    coingecko_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None

    # --- Broker execution (go-live, OPT-IN; ships OFF) ---
    # Default is the simulated paper broker. The real Alpaca adapter defaults to
    # Alpaca's PAPER (sandbox) endpoint, so even with keys set no real money
    # moves. LIVE trading is hard-gated and OFF by default: it requires
    # ``broker == 'alpaca'`` AND ``alpaca_live`` AND real keys AND
    # ``broker_ack == 'I understand this places real orders'``.
    broker: str = "simulated"
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_live: bool = False
    broker_ack: Optional[str] = None

    # --- Persistence (go-live, OPT-IN; ships in-memory) ---
    # Default keeps the in-memory stores so tests and the running app are
    # unaffected. When ``sqlite`` the app initializes its OWN new database file
    # (never an existing one) with no auto schema-sync beyond first create.
    persist: str = "memory"
    db_url: str = "sqlite:///giffmemoney.db"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use the CSV-aware env source so ``CORS_ORIGINS`` accepts ``a,b,c``.

        Returns:
            The ordered tuple of settings sources, with the default process
            environment source swapped for :class:`_CsvEnvSource`.
        """
        csv_env = _CsvEnvSource(settings_cls)
        return (
            init_settings,
            csv_env,
            dotenv_settings,
            file_secret_settings,
        )


settings = Settings()
