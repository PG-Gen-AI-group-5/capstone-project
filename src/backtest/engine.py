"""Trade simulation and portfolio (tasks 17.1, 17.3-17.5) on raw, as-traded prices.

Each trading day, in order (`Book.session`):
1. At the open, for open positions: an `exit` request (Exit Monitor or manual) sells at the open; a tightened
   stop applies from today. A gap through the stop sells at the open (`gap_stop`); a gap above the target
   sells at the open.
2. New signals dated today buy at the open, best confidence first, while slots and cash allow. Size: risk
   `risk_per_trade` of equity between the open and the stop, capped at `max_position_pct` of equity.
   No fill when the stock is locked at its band (open = high = low = close) or opens outside stop..target.
3. During the day: stop and target both inside the range -> the stop counts (task 17.1); else whichever was hit.
4. At the close: positions that reached `max_hold_days` (the entry day is day 1) sell at the close.
5. Equity = cash + positions at the close.

`simulate` runs a whole backtest through one `Book`; the paper-trading desk (src/desk) keeps a `Book` between
days and calls `session` once per trading day, so both apply exactly the same rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import order_cost
from src.contracts import Signal, TradeRecord

PORTFOLIO_DEFAULTS = {"capital": 1_000_000.0, "risk_per_trade": 0.01, "max_positions": 5, "max_position_pct": 0.2}


@dataclass
class Position:
    signal: Signal
    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    stop: float
    entry_cost: float
    days: int = 0

    def to_dict(self) -> dict:
        return {"signal": self.signal.model_dump(mode="json"), "entry_date": str(self.entry_date.date()),
                "entry_price": self.entry_price, "quantity": self.quantity, "stop": self.stop,
                "entry_cost": self.entry_cost, "days": self.days}

    @classmethod
    def from_dict(cls, d: dict) -> Position:
        return cls(Signal(**d["signal"]), pd.Timestamp(d["entry_date"]), float(d["entry_price"]), int(d["quantity"]),
                   float(d["stop"]), float(d["entry_cost"]), int(d.get("days", 0)))


@dataclass
class Order:
    """One fill (or refused entry) in a session. `at`: open, intraday or close. Price and costs are exact, so the
    orders add up to the cash."""
    symbol: str
    side: str                                 # buy | sell
    kind: str                                 # entry | no_fill | exit | gap_stop | target | stop | time
    at: str
    quantity: int
    price: float | None
    costs: float
    source: str
    ref: str | None = None


@dataclass
class SessionResult:
    trades: list[TradeRecord] = field(default_factory=list)      # closed round trips and refused entries
    orders: list[Order] = field(default_factory=list)
    equity: dict = field(default_factory=dict)


class Book:
    """Cash and open positions, advanced one trading day at a time."""

    def __init__(self, cash: float, portfolio: dict | None = None, cost_rates: dict | None = None,
                 positions: dict[str, Position] | None = None):
        self.p = {**PORTFOLIO_DEFAULTS, **(portfolio or {})}
        self.cash = float(cash)
        self.cost_rates = cost_rates
        self.positions: dict[str, Position] = dict(positions or {})

    # ---- state for the desk's database
    def state(self) -> dict:
        return {"cash": self.cash, "positions": [p.to_dict() for p in self.positions.values()]}

    @classmethod
    def from_state(cls, state: dict, portfolio: dict | None = None, cost_rates: dict | None = None) -> Book:
        positions = {d["signal"]["symbol"]: Position.from_dict(d) for d in state.get("positions", [])}
        return cls(state["cash"], portfolio, cost_rates, positions)

    # ---- one trading day
    def session(self, d, today: pd.DataFrame, signals=(), exits: dict | None = None) -> SessionResult:
        """today: that day's raw bars indexed by symbol (open, high, low, close). exits: {symbol: decision} with
        `action` (exit | tighten_stop | hold) and `new_stop`."""
        d = pd.Timestamp(d)
        exits = exits or {}
        out = SessionResult()
        bar = lambda sym, k: float(today.at[sym, k])

        # 1. at the open
        for sym in list(self.positions):
            pos = self.positions[sym]
            if sym not in today.index:
                continue
            o = bar(sym, "open")
            dec = exits.get(sym)
            action = getattr(dec, "action", None)
            if action == "exit":
                self._close(pos, d, o, "exit", "open", out)
                continue
            new_stop = getattr(dec, "new_stop", None)
            if action == "tighten_stop" and new_stop and new_stop > pos.stop:
                pos.stop = float(new_stop)
            if o <= pos.stop:
                self._close(pos, d, o, "gap_stop", "open", out)
            elif o >= pos.signal.target:
                self._close(pos, d, o, "target", "open", out)

        # 2. new entries
        marked = self.cash + sum(q.quantity * bar(s, "open") for s, q in self.positions.items() if s in today.index)
        for s in sorted(signals, key=lambda x: -(x.confidence or 0)):
            if s.symbol in self.positions or len(self.positions) >= self.p["max_positions"] or s.symbol not in today.index:
                continue
            o, h, lo, c = (bar(s.symbol, k) for k in ("open", "high", "low", "close"))
            locked = o == h == lo == c
            if locked or not (s.stop < o < s.target):
                out.trades.append(TradeRecord(symbol=s.symbol, source=s.source, signal_date=s.signal_date, entry_date=None,
                                              entry_price=None, exit_date=None, exit_price=None, exit_reason="no_fill",
                                              quantity=0, gross_pnl=0.0, costs=0.0, net_pnl=0.0, return_pct=0.0))
                out.orders.append(Order(s.symbol, "buy", "no_fill", "open", 0, o, 0.0, s.source, s.ref))
                continue
            qty = math.floor(min(marked * self.p["risk_per_trade"] / (o - s.stop), marked * self.p["max_position_pct"] / o,
                                 self.cash / (o * 1.002)))
            if qty <= 0:
                continue
            cost = order_cost(qty * o, "buy", self.cost_rates)
            self.cash -= qty * o + cost
            self.positions[s.symbol] = Position(s, d, o, qty, s.stop, cost)
            out.orders.append(Order(s.symbol, "buy", "entry", "open", qty, o, cost, s.source, s.ref))

        # 3. during the day, 4. at the close
        for sym in list(self.positions):
            if sym not in today.index:
                continue
            pos = self.positions[sym]
            h, lo, c = bar(sym, "high"), bar(sym, "low"), bar(sym, "close")
            pos.days += 1
            if lo <= pos.stop:
                self._close(pos, d, pos.stop, "stop", "intraday", out)
            elif h >= pos.signal.target:
                self._close(pos, d, pos.signal.target, "target", "intraday", out)
            elif pos.days >= pos.signal.max_hold_days:
                self._close(pos, d, c, "time", "close", out)

        # 5. equity at the close
        held = sum(q.quantity * bar(s, "close") for s, q in self.positions.items() if s in today.index)
        out.equity = {"trade_date": d, "cash": round(self.cash, 2), "positions": round(held, 2),
                      "equity": round(self.cash + held, 2), "open_positions": len(self.positions)}
        return out

    def _close(self, pos: Position, d: pd.Timestamp, price: float, reason: str, at: str, out: SessionResult) -> None:
        value = pos.quantity * price
        cost = order_cost(value, "sell", self.cost_rates)
        self.cash += value - cost
        gross = (price - pos.entry_price) * pos.quantity
        net = gross - pos.entry_cost - cost
        out.trades.append(TradeRecord(symbol=pos.signal.symbol, source=pos.signal.source, signal_date=pos.signal.signal_date,
                                      entry_date=pos.entry_date.date(), entry_price=pos.entry_price, exit_date=d.date(),
                                      exit_price=round(price, 2), exit_reason=reason, quantity=pos.quantity,
                                      gross_pnl=round(gross, 2), costs=round(pos.entry_cost + cost, 2),
                                      net_pnl=round(net, 2), return_pct=round(net / (pos.entry_price * pos.quantity), 5)))
        out.orders.append(Order(pos.signal.symbol, "sell", reason, at, pos.quantity, price, cost, pos.signal.source,
                                pos.signal.ref))
        del self.positions[pos.signal.symbol]

    def mark_open(self, last_bars: pd.DataFrame | None) -> list[TradeRecord]:
        """Positions still open, marked at the last close (exit_reason `open`)."""
        out = []
        for sym, pos in self.positions.items():
            price = float(last_bars.at[sym, "close"]) if last_bars is not None and sym in last_bars.index else pos.entry_price
            gross = (price - pos.entry_price) * pos.quantity
            out.append(TradeRecord(symbol=sym, source=pos.signal.source, signal_date=pos.signal.signal_date,
                                   entry_date=pos.entry_date.date(), entry_price=pos.entry_price, exit_date=None,
                                   exit_price=round(price, 2), exit_reason="open", quantity=pos.quantity,
                                   gross_pnl=round(gross, 2), costs=round(pos.entry_cost, 2),
                                   net_pnl=round(gross - pos.entry_cost, 2),
                                   return_pct=round((gross - pos.entry_cost) / (pos.entry_price * pos.quantity), 5)))
        return out


def simulate(signals: list[Signal], bars: pd.DataFrame, start=None, end=None, portfolio: dict | None = None,
             cost_rates: dict | None = None, exits: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (journal, equity). bars: symbol, trade_date, open, high, low, close (raw prices).
    exits: Exit Monitor decisions (trade_date, symbol, action, new_stop)."""
    p = {**PORTFOLIO_DEFAULTS, **(portfolio or {})}
    bars = bars.assign(trade_date=pd.to_datetime(bars["trade_date"]))
    days = pd.DatetimeIndex(sorted(bars["trade_date"].unique()))
    first = pd.Timestamp(start) if start is not None else (min(s.signal_date for s in signals) if signals else days[0])
    days = days[days >= pd.Timestamp(first)]
    if end is not None:
        last_hold = max((s.max_hold_days for s in signals), default=0)
        cutoff = days.searchsorted(pd.Timestamp(end), side="right") + last_hold
        days = days[:cutoff]
    by_day = {d: g.set_index("symbol") for d, g in bars[bars["trade_date"].isin(days)].groupby("trade_date")}
    todays: dict[pd.Timestamp, list[Signal]] = {}
    for s in signals:
        todays.setdefault(pd.Timestamp(s.signal_date), []).append(s)
    day_exits: dict[pd.Timestamp, dict] = {}
    if exits is not None and not exits.empty:
        for r in exits.itertuples():
            day_exits.setdefault(pd.Timestamp(r.trade_date), {})[r.symbol] = r

    book = Book(p["capital"], p, cost_rates)
    journal, equity = [], []
    for d in days:
        today = by_day.get(d)
        if today is None:
            continue
        res = book.session(d, today, todays.get(d, []), day_exits.get(d))
        journal += res.trades
        equity.append(res.equity)
    last = days[-1] if len(days) else None
    journal += book.mark_open(by_day.get(last) if last is not None else None)
    j = pd.DataFrame([t.model_dump() for t in journal])
    return j, pd.DataFrame(equity)


def buy_and_hold(bars: pd.DataFrame, symbols: list[str], start, end, capital: float = 1_000_000.0,
                 cost_rates: dict | None = None, source: str = "buy_and_hold") -> pd.DataFrame:
    """Equal-weight buy at the first open, hold to the end (task 16.3). Returns the equity curve."""
    b = bars.assign(trade_date=pd.to_datetime(bars["trade_date"]))
    b = b[(b["trade_date"] >= pd.Timestamp(start)) & (b["trade_date"] <= pd.Timestamp(end)) & b["symbol"].isin(symbols)]
    first = b["trade_date"].min()
    opens = b[b["trade_date"] == first].set_index("symbol")["open"]
    per = capital / max(len(opens), 1)
    qty = {s: math.floor(per / (o * 1.002)) for s, o in opens.items()}
    spent = sum(qty[s] * opens[s] + order_cost(qty[s] * opens[s], "buy", cost_rates) for s in qty)
    cash = capital - spent
    wide = b.pivot_table(index="trade_date", columns="symbol", values="close").ffill()
    value = sum(wide[s] * q for s, q in qty.items() if s in wide)
    return pd.DataFrame({"trade_date": wide.index, "cash": cash, "positions": value.round(2).to_numpy(),
                         "equity": (cash + value).round(2).to_numpy(), "open_positions": len(qty), "source": source})


def summarise(journal: pd.DataFrame, equity: pd.DataFrame, capital: float) -> dict:
    """Trade and portfolio metrics: trades, win rate, average return, total return, CAGR, max drawdown, Sharpe."""
    filled = journal[journal["exit_reason"] != "no_fill"] if len(journal) else journal
    out = {"trades": len(filled), "no_fills": int((journal["exit_reason"] == "no_fill").sum()) if len(journal) else 0}
    if len(filled):
        out.update(win_rate=float((filled["net_pnl"] > 0).mean()), avg_return=float(filled["return_pct"].mean()),
                   costs=float(filled["costs"].sum()),
                   exits={k: int(v) for k, v in filled["exit_reason"].value_counts().items()})
    if len(equity):
        e = equity.set_index("trade_date")["equity"]
        r = e.pct_change().dropna()
        years = max((e.index[-1] - e.index[0]).days / 365.25, 1 / 365.25)
        out.update(total_return=float(e.iloc[-1] / capital - 1), cagr=float((e.iloc[-1] / capital) ** (1 / years) - 1),
                   max_drawdown=float((e / e.cummax() - 1).min()),
                   sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
                   exposure=float((equity["open_positions"] > 0).mean()))
    return out
