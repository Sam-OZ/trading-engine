"""Infrastructure configuration: where things live and which keys to use.

Business values (risk limits, budgets, toggles) are NOT here; they are admin
settings stored in the database (see settings_store.py).

Safety: TRADING_MODE accepts only "demo", and the broker modules contain only
demo trading endpoints. Pointing this engine at a real-money account needs a
code change, not a config change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfraConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    trading_mode: Literal["demo"] = "demo"
    database_url: str = "sqlite+aiosqlite:///./data/engine.db"
    data_dir: Path = Path("./data")

    capital_api_key: SecretStr | None = None
    capital_identifier: SecretStr | None = None
    capital_api_password: SecretStr | None = None

    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None

    anthropic_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None

    @property
    def kill_switch_file(self) -> Path:
        return self.data_dir / "KILL"

    def has_capital(self) -> bool:
        return all(
            s is not None and s.get_secret_value()
            for s in (self.capital_api_key, self.capital_identifier, self.capital_api_password)
        )

    def has_binance(self) -> bool:
        return all(
            s is not None and s.get_secret_value()
            for s in (self.binance_api_key, self.binance_api_secret)
        )


def load_config() -> InfraConfig:
    cfg = InfraConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
