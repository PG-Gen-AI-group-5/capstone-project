"""Serve the desk app: python -m src.api [--host 127.0.0.1] [--port 8000]"""
from __future__ import annotations

import argparse

import uvicorn

from src.api.app import create_app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.api", description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    print(f"Trading with AI: http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
