"""The live suggester: each trading day, trade suggestions for the next session and advice on your holdings.

    python -m src.live desk --name "Live" [--provider openai_compatible] [--capital 1000000]   # make a live desk
    python -m src.live daily                 # download, rebuild, forecast, suggest (what the 08:00 task runs)
    python -m src.live daily --no-update     # re-plan with the data already downloaded
    python -m src.live update                # download and rebuild only
    python -m src.live refit                 # refit the live TFT and LSTM (what the weekly task runs, ~1 hour)
    python -m src.live status
    python -m src.live schedule install | remove | show

Open the app (python -m src.api) to see the suggestions, mark what you bought, and get the exit advice.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import Config


def say(msg: str) -> None:
    print(f"{datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m src.live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("desk", help="create a live desk")
    d.add_argument("--name", required=True)
    d.add_argument("--provider", default=None, help="offline | openai_compatible | anthropic (default: llm.provider)")
    d.add_argument("--capital", type=float)
    dl = sub.add_parser("daily", help="download, rebuild, forecast and suggest")
    dl.add_argument("--no-update", action="store_true", help="skip the download and rebuild")
    dl.add_argument("--refit", choices=["missing", "due", "always"], default="missing")
    sub.add_parser("update", help="download and rebuild the live tables")
    r = sub.add_parser("refit", help="refit the live models")
    r.add_argument("--models", help="comma-separated (default: live.models)")
    sub.add_parser("status")
    s = sub.add_parser("schedule")
    s.add_argument("action", choices=["install", "remove", "show"])
    args = ap.parse_args(argv)

    from src.desk.data import desk_settings
    from src.desk.service import DeskError
    from src.desk.store import Store
    from src.live import schedule
    from src.live import update as live_update

    cfg = Config.load()
    store = Store(cfg.data_dir / desk_settings(cfg)["db"])
    try:
        if args.cmd == "desk":
            from src.desk import live
            provider = args.provider or (cfg.get("llm") or {}).get("provider", "offline")
            desk_id = live.create_live_desk(store, cfg, args.name, args.capital, provider)
            say(f"Live desk {desk_id} ({args.name}, {provider}) created. Run: python -m src.live daily")
        elif args.cmd == "daily":
            from src.desk import live
            say("Live daily run")
            out = live.daily(cfg, store, update_data=not args.no_update, refit_if=args.refit, progress=say)
            for desk_id, res in out.get("desks", {}).items():
                say(f"desk {desk_id}: suggestions for {res['trade_date']}: {res['analysed']} analysed, "
                    f"{res['accepted']} accepted; {res['holdings']} holding(s) checked")
            if not out.get("desks"):
                say("No live desks yet: python -m src.live desk --name Live")
        elif args.cmd == "update":
            info = live_update.ingest(cfg, log=say)
            say(f"prices to {info['last_price_day']} (were to {info['last_price_day_before']})")
            b = live_update.build(cfg, log=say)
            say(f"live tables built for {b['day']} in {b['sec']}s")
        elif args.cmd == "refit":
            from datetime import date

            from src.live import models as live_models
            from src.live.env import live_settings
            info = live_update.built(cfg) or live_update.build(cfg, log=say)
            day = date.fromisoformat(str(info["day"]))
            names = args.models.split(",") if args.models else live_settings(cfg)["models"]
            from src.live.env import live_config
            from src.models.data import load_panel
            panel = load_panel(live_config(cfg))
            for name in names:
                live_models.refit(cfg, name.strip(), day, panel, log=say)
        elif args.cmd == "status":
            from src.live import models as live_models
            from src.live.env import live_config, live_settings
            lcfg = live_config(cfg)
            print("lake:   ", live_update.lake_status(cfg))
            print("built:  ", live_update.built(cfg) or "not yet")
            for m in live_settings(cfg)["models"]:
                info = live_models.fit_info(lcfg, m)
                print(f"{m:7} ", f"fitted for {info['fitted_for']}, labels to {info['train_end']}" if info else "not fitted")
            for row in store.rows("SELECT id, name, provider, current_date, phase FROM desks WHERE mode = 'live'"):
                print("desk:   ", row)
            print("tasks:  ", schedule.installed() or "none")
        else:
            if args.action == "install":
                for line in schedule.install(cfg):
                    say(f"scheduled {line}")
            elif args.action == "remove":
                for line in schedule.remove():
                    say(line)
            else:
                print(schedule.installed() or "No scheduled tasks.")
    except DeskError as exc:
        say(f"Stopped: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
