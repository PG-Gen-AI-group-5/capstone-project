"""Indian equity delivery costs per order (task 17.2). Rates come from `costs` in config.yaml; check them
against a broker's calculator when they change.

buy:  brokerage + STT + exchange charges + SEBI fee + stamp duty + GST on (brokerage + exchange + SEBI)
sell: brokerage + STT + exchange charges + SEBI fee + DP charge + GST on (brokerage + exchange + SEBI)
"""
from __future__ import annotations

DEFAULTS = {
    "brokerage_pct": 0.0,            # discount brokers charge nothing for delivery
    "brokerage_max": 20.0,           # rupees per order, when brokerage_pct > 0
    "stt_pct": 0.001,                # 0.1% on buy and sell value (delivery)
    "exchange_pct": 0.0000297,       # NSE transaction charge, 0.00297%
    "sebi_per_crore": 10.0,          # Rs 10 per crore of turnover
    "stamp_buy_pct": 0.00015,        # 0.015% on the buy side
    "gst_pct": 0.18,                 # on brokerage + exchange + SEBI charges
    "dp_per_sell": 15.93,            # depository charge per stock sold in a day, incl. GST
}


def order_cost(value: float, side: str, rates: dict | None = None) -> float:
    r = {**DEFAULTS, **(rates or {})}
    brokerage = min(value * r["brokerage_pct"], r["brokerage_max"]) if r["brokerage_pct"] else 0.0
    exchange = value * r["exchange_pct"]
    sebi = value * r["sebi_per_crore"] / 1e7
    gst = (brokerage + exchange + sebi) * r["gst_pct"]
    cost = brokerage + value * r["stt_pct"] + exchange + sebi + gst
    cost += value * r["stamp_buy_pct"] if side == "buy" else r["dp_per_sell"]
    return round(cost, 2)
