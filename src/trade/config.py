"""Runtime configuration for the trading platform.

Values come from environment variables (prefixed ``TRADE_``) or a local ``.env``
file. Nothing is hardcoded; secrets live outside the repo. See ``.env.example``
for the starter template.

The single ``get_settings()`` entry point is cached; tests that mutate the
environment must call ``get_settings.cache_clear()`` before re-reading.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    PAPER = "paper"
    LIVE = "live"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRADE_",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    bybit_base_url: str = "https://api.bybit.com"
    bybit_testnet_url: str = "https://api-testnet.bybit.com"
    use_bybit_testnet: bool = True

    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None

    default_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    default_category: str = "linear"

    daily_drawdown_pct: float = Field(default=0.035, gt=0.0, lt=0.5)
    weekly_drawdown_pct: float = Field(default=0.08, gt=0.0, lt=0.5)
    monthly_drawdown_pct: float = Field(default=0.12, gt=0.0, lt=0.5)

    prometheus_port: int = Field(default=9464, gt=0, lt=65536)

    @model_validator(mode="after")
    def _drawdowns_are_layered(self) -> Settings:
        # The whole point of layered drawdowns is that each longer window is
        # looser than the shorter one; misordering them defeats the safeguard.
        if not self.daily_drawdown_pct <= self.weekly_drawdown_pct <= self.monthly_drawdown_pct:
            raise ValueError(
                "Drawdown limits must satisfy daily <= weekly <= monthly "
                f"(got {self.daily_drawdown_pct=}, {self.weekly_drawdown_pct=}, "
                f"{self.monthly_drawdown_pct=})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
