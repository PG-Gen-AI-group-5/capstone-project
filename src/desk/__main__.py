"""Run the paper-trading desk from the command line (the same steps the app's jobs run).

    python -m src.desk list
    python -m src.desk create --name "2023 rules" --start 2023-01-02 [--provider offline] [--capital 1000000]
    python -m src.desk advance --desk 1 --days 20
"""
from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.desk import service
from src.desk.data import DeskData, desk_settings
from src.desk.store import Store


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m src.desk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--end")
    c.add_argument("--provider", default="offline", choices=service.PROVIDERS)
    c.add_argument("--capital", type=float)
    a = sub.add_parser("advance")
    a.add_argument("--desk", type=int, required=True)
    a.add_argument("--days", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = Config.load()
    store = Store(cfg.data_dir / desk_settings(cfg)["db"])
    if args.cmd == "list":
        for d in store.rows("SELECT id, name, provider, current_date, phase FROM desks ORDER BY id"):
            print(f"{d['id']:>3}  {d['name']:<28} {d['provider']:<18} {d['current_date']}  {d['phase']}")
        return
    data = DeskData.load(cfg)
    say = lambda msg, *_: print(f"  {msg}", flush=True)
    try:
        if args.cmd == "create":
            desk_id = service.create_desk(store, data, cfg, args.name, args.start, args.capital, args.provider, args.end)
            print(f"Desk {desk_id} created; making the first morning's plans")
            out = service.run_morning(store, data, cfg, desk_id, say)
            print(f"{out['trade_date']}: {out['analysed']} analysed, {out['accepted']} accepted")
        else:
            n = service.advance(store, data, cfg, args.desk, args.days, say)
            desk = service.get_desk(store, args.desk)
            print(f"Advanced {n} day(s); desk {args.desk} is at {desk['current_date']} ({desk['phase']}), "
                  f"cash {desk['book']['cash']:,.0f}, {len(desk['book']['positions'])} open position(s)")
    except service.DeskError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
