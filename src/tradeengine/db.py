"""Trade journal schema (SQLAlchemy 2.0, async). SQLite locally, Postgres in Docker/cloud."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PRICE = Numeric(28, 10)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Unique key derived from the signal: a retried or duplicated submit cannot create a second row.
    client_order_id: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    venue: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(4))
    size: Mapped[Decimal] = mapped_column(PRICE)
    source: Mapped[str] = mapped_column(String(16))
    strategy: Mapped[str] = mapped_column(String(64))
    entry_ref: Mapped[Decimal] = mapped_column(PRICE)
    stop: Mapped[Decimal] = mapped_column(PRICE)
    target: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    risk_amount: Mapped[Decimal] = mapped_column(PRICE)
    status: Mapped[str] = mapped_column(String(16))
    broker_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    broker_position_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fill_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class PositionRow(Base):
    __tablename__ = "positions"
    __table_args__ = (Index("ix_positions_status", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("orders.client_order_id"), unique=True, nullable=True
    )
    venue: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(4))
    size: Mapped[Decimal] = mapped_column(PRICE)
    entry_price: Mapped[Decimal] = mapped_column(PRICE)
    stop: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    target: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    broker_position_id: Mapped[str] = mapped_column(String(64))
    protection_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(16))
    strategy: Mapped[str] = mapped_column(String(64))
    risk_amount: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))
    status: Mapped[str] = mapped_column(String(8))
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    realised_pnl: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)


class PositionMetaRow(Base):
    """How a strategy position is managed after entry (trail, breakeven, time exit)."""

    __tablename__ = "position_meta"

    client_order_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(64))
    initial_risk: Mapped[Decimal] = mapped_column(PRICE)  # price distance entry -> first stop
    trail_atr: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    breakeven_at_r: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    max_bars: Mapped[int] = mapped_column(Integer)
    signal_bar: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EventRow(Base):
    """Audit log: every decision, rejection, reconcile action and alert."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String(8))
    component: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class SettingRow(Base):
    """Admin-editable runtime settings (risk limits, budgets, toggles)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    updated_by: Mapped[str] = mapped_column(String(64), default="system")


class StateRow(Base):
    """Engine state that must survive restarts: halt flag, start-of-day and peak equity."""

    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class EquityRow(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    venue: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(8))
    balance: Mapped[Decimal] = mapped_column(PRICE)
    equity: Mapped[Decimal] = mapped_column(PRICE)
    available: Mapped[Decimal] = mapped_column(PRICE)


class CandleRow(Base):
    __tablename__ = "candles"
    __table_args__ = (UniqueConstraint("venue", "symbol", "timeframe", "open_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[Decimal] = mapped_column(PRICE)
    high: Mapped[Decimal] = mapped_column(PRICE)
    low: Mapped[Decimal] = mapped_column(PRICE)
    close: Mapped[Decimal] = mapped_column(PRICE)
    volume: Mapped[Decimal] = mapped_column(PRICE, default=Decimal("0"))


class FollowSnapshotRow(Base):
    """Round 4 Track A: one followed trader's open positions, recorded once an hour (read-only, public data)."""

    __tablename__ = "follow_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    address: Mapped[str] = mapped_column(String(64), index=True)
    account_value: Mapped[Decimal] = mapped_column(PRICE)
    positions: Mapped[dict] = mapped_column(JSON)  # coin -> {"szi": signed size, "value": USD notional}


class ShadowTradeRow(Base):
    """Round 4 Track A: a copy trade that was only recorded, never sent. Scored at the end of the record."""

    __tablename__ = "shadow_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coin: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))  # Binance USDⓈ-M futures symbol
    side: Mapped[str] = mapped_column(String(4))
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    entry: Mapped[Decimal] = mapped_column(PRICE)
    stop: Mapped[Decimal] = mapped_column(PRICE)
    size: Mapped[Decimal] = mapped_column(PRICE)
    risk_usd: Mapped[Decimal] = mapped_column(PRICE)
    reason: Mapped[str] = mapped_column(String(128))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fees: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    funding: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)
    pnl: Mapped[Decimal | None] = mapped_column(PRICE, nullable=True)  # after fees and funding
    r: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)


class ShadowEquityRow(Base):
    __tablename__ = "shadow_equity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    equity: Mapped[Decimal] = mapped_column(PRICE)


class Database:
    def __init__(self, url: str):
        sqlite = url.startswith("sqlite")
        # SQLite on the server is shared by the running engine and short CLI commands (`te status`):
        # WAL lets readers and the writer work side by side, and a 30 s busy timeout rides out the rest.
        # Backups use SQLite's .backup API, which is safe with WAL (never file-copy a live WAL database).
        self.engine: AsyncEngine = create_async_engine(url, future=True, connect_args={"timeout": 30} if sqlite else {})
        if sqlite:
            from sqlalchemy import event

            @event.listens_for(self.engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()
