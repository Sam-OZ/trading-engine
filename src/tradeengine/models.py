"""Domain types shared by every module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class Venue(StrEnum):
    CAPITAL = "capital"
    BINANCE = "binance"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class AssetKind(StrEnum):
    FX = "fx"  # Forex CFD (Capital.com)
    CRYPTO_CFD = "crypto_cfd"  # crypto CFD (Capital.com): can short, no coins held
    CRYPTO_SPOT = "crypto_spot"  # real coins (Binance): long-only


class Source(StrEnum):
    STRATEGY = "strategy"  # deterministic rule-based strategy
    CLAUDE = "claude"  # Claude trading its own budget


class OrderStatus(StrEnum):
    PENDING = "pending"  # written to journal, not yet sent
    SUBMITTED = "submitted"  # sent, outcome not yet known
    FILLED = "filled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"  # send failed ambiguously; reconciler resolves


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class Instrument:
    """One tradable market, with the canonical symbol used across the engine."""

    symbol: str  # canonical, e.g. "EUR/USD", "BTC/USDT"
    venue: Venue
    broker_symbol: str  # Capital.com epic ("EURUSD") or Binance symbol ("BTCUSDT")
    kind: AssetKind
    base: str
    quote: str

    @property
    def currencies(self) -> tuple[str, str]:
        return (self.base, self.quote)


# Phase 1 starting instruments. AUD/USD, USD/CAD and SOL/USDT are added after Phase 3.
DEFAULT_INSTRUMENTS: dict[str, Instrument] = {
    i.symbol: i
    for i in (
        Instrument("EUR/USD", Venue.CAPITAL, "EURUSD", AssetKind.FX, "EUR", "USD"),
        Instrument("GBP/USD", Venue.CAPITAL, "GBPUSD", AssetKind.FX, "GBP", "USD"),
        Instrument("USD/JPY", Venue.CAPITAL, "USDJPY", AssetKind.FX, "USD", "JPY"),
        Instrument("BTC/USDT", Venue.BINANCE, "BTCUSDT", AssetKind.CRYPTO_SPOT, "BTC", "USDT"),
        Instrument("ETH/USDT", Venue.BINANCE, "ETHUSDT", AssetKind.CRYPTO_SPOT, "ETH", "USDT"),
    )
}


@dataclass(frozen=True)
class MarketSpec:
    """Broker dealing rules for one instrument, read from the broker at start-up."""

    min_size: Decimal
    size_step: Decimal
    price_tick: Decimal
    min_notional: Decimal = Decimal("0")
    contract_multiplier: Decimal = Decimal("1")  # P&L per unit size per 1.0 price move, in quote ccy
    tradeable: bool = True


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    ts: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Bar:
    symbol: str
    timeframe: str
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")


@dataclass(frozen=True)
class TradeIntent:
    """What a strategy or Claude wants to do, before risk sizing."""

    symbol: str
    side: Side
    entry: Decimal  # reference entry price used for sizing
    stop: Decimal  # mandatory protective stop
    target: Decimal | None
    source: Source
    strategy: str
    signal_time: datetime
    reason: str = ""
    size_factor: Decimal = Decimal("1")  # Claude review may reduce (0 < f <= 1)


@dataclass(frozen=True)
class OrderPlan:
    """A sized, risk-approved order ready for execution."""

    client_order_id: str
    instrument: Instrument
    intent: TradeIntent
    size: Decimal
    risk_amount: Decimal  # account-currency loss if the stop fills at its level


@dataclass
class AccountSnapshot:
    venue: Venue
    currency: str
    balance: Decimal
    equity: Decimal
    available: Decimal


@dataclass
class BrokerPosition:
    """A position as the broker reports it."""

    venue: Venue
    symbol: str
    side: Side
    size: Decimal
    entry_price: Decimal
    stop: Decimal | None
    target: Decimal | None
    broker_position_id: str
    protection_id: str | None = None  # Binance OCO list id
    client_order_id: str | None = None
    opened_at: datetime | None = None
    unrealised_pnl: Decimal | None = None


@dataclass
class ExitInfo:
    """What happened to a journal position the broker no longer shows."""

    status: str  # "closed" | "unprotected" | "unknown"
    exit_price: Decimal | None = None
    reason: str = ""


@dataclass
class OpenResult:
    ok: bool
    broker_position_id: str | None = None
    fill_price: Decimal | None = None
    filled_size: Decimal | None = None
    protection_id: str | None = None
    protected: bool = False
    error: str | None = None
    raw: dict = field(default_factory=dict)
