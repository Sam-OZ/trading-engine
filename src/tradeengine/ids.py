"""Deterministic client order IDs.

The same signal always produces the same ID, so a retry, a restart mid-submit
or a duplicated signal maps onto one journal row and at most one broker order.
Format fits Binance's newClientOrderId rule: ^[.A-Z:/a-z0-9_-]{1,36}$.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from .models import Side, Source


def client_order_id(source: Source, strategy: str, symbol: str, side: Side, signal_time: datetime) -> str:
    key = f"{source}|{strategy}|{symbol}|{side}|{signal_time.isoformat()}"
    return "te-" + hashlib.sha256(key.encode()).hexdigest()[:20]


def leg_id(cid: str, leg: str) -> str:
    """Child IDs for protective orders: te-<20 hex>-tp / -sl / -oco (max 27 chars)."""
    if leg not in {"tp", "sl", "oco", "x"}:
        raise ValueError(leg)
    return f"{cid}-{leg}"
