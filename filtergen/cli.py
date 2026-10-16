"""Shared argparse/seed/logging setup for experiments/*.py."""

import argparse
import logging

from filtergen import upstream  # noqa: F401
from src.utils import experiment_utils, logging_utils


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=upstream.DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    logging_utils.add_logging_args(p)
    return p


def setup(args) -> None:
    logging_utils.configure(args)
    # upstream logs every sampled prompt at INFO; keep our logs readable
    for name in ("src.selection.data", "src.selection.optimization"):
        logging.getLogger(name).setLevel(max(args.log_level, logging.WARNING))
    experiment_utils.set_seed(args.seed)
