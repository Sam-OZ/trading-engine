"""Position sizing: size so that a stop-out loses (at most) the risk amount."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from .models import MarketSpec

# Stablecoins treated as their fiat peg for currency conversion only.
_PEGS = {"USDT": "USD", "USDC": "USD", "FDUSD": "USD"}


def _norm(ccy: str) -> str:
    return _PEGS.get(ccy.upper(), ccy.upper())


def conversion_rate(from_ccy: str, to_ccy: str, rates: Mapping[str, Decimal]) -> Decimal | None:
    """Units of `to_ccy` per 1 unit of `from_ccy`, from mid prices keyed like "EUR/USD".

    Returns None when no route exists; callers must then refuse the trade rather
    than guess, because a wrong rate means a wrong size.
    """
    f, t = _norm(from_ccy), _norm(to_ccy)
    if f == t:
        return Decimal(1)

    def direct(a: str, b: str) -> Decimal | None:
        if f"{a}/{b}" in rates and rates[f"{a}/{b}"] > 0:
            return rates[f"{a}/{b}"]
        if f"{b}/{a}" in rates and rates[f"{b}/{a}"] > 0:
            return Decimal(1) / rates[f"{b}/{a}"]
        return None

    d = direct(f, t)
    if d is not None:
        return d
    via_usd_a, via_usd_b = direct(f, "USD"), direct("USD", t)
    if via_usd_a is not None and via_usd_b is not None:
        return via_usd_a * via_usd_b
    return None


def round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    if tick <= 0:
        return price
    return (price / tick).to_integral_value() * tick


@dataclass(frozen=True)
class SizeResult:
    size: Decimal | None
    risk_at_stop: Decimal  # account currency, using the rounded size
    notional: Decimal  # account currency
    reason: str = ""


def loss_per_unit(entry: Decimal, stop: Decimal, spec: MarketSpec, quote_to_account: Decimal) -> Decimal:
    return abs(entry - stop) * spec.contract_multiplier * quote_to_account


def position_size(
    entry: Decimal,
    stop: Decimal,
    risk_amount: Decimal,
    spec: MarketSpec,
    quote_to_account: Decimal,
) -> SizeResult:
    per_unit = loss_per_unit(entry, stop, spec, quote_to_account)
    if per_unit <= 0:
        return SizeResult(None, Decimal(0), Decimal(0), "stop distance is zero")
    size = round_down(risk_amount / per_unit, spec.size_step)
    notional = size * entry * spec.contract_multiplier * quote_to_account
    if size < spec.min_size:
        min_risk = spec.min_size * per_unit
        return SizeResult(
            None,
            Decimal(0),
            Decimal(0),
            f"broker minimum size {spec.min_size} would risk {min_risk:.2f}, above the {risk_amount:.2f} allowed",
        )
    if spec.min_notional > 0 and size * entry * spec.contract_multiplier < spec.min_notional:
        return SizeResult(
            None,
            Decimal(0),
            Decimal(0),
            f"order value below the exchange minimum of {spec.min_notional}",
        )
    return SizeResult(size, size * per_unit, notional)
