"""Reusable persistence, configuration, logging, and reproducibility tools."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pandas as pd
import yaml

P = ParamSpec("P")
R = TypeVar("R")


def setup_logger(
    name: str = "redrob_ai_ranker",
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create an idempotent application logger with optional file output."""
    logger = logging.getLogger(name)
    resolved_level = (
        logging.getLevelName(level.upper()) if isinstance(level, str) else level
    )
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unknown logging level: {level}")

    logger.setLevel(resolved_level)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handlers = [
        handler
        for handler in logger.handlers
        if type(handler) is logging.StreamHandler
    ]
    if not stream_handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_files = {
            Path(handler.baseFilename).resolve()
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if log_path not in existing_files:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(resolved_level)
    return logger


def timer(
    function: Callable[P, R] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Log wall-clock execution time for a function.

    Supports both ``@timer`` and ``@timer(logger=my_logger)`` usage.
    """

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        active_logger = logger or logging.getLogger(target.__module__)

        @wraps(target)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            started_at = time.perf_counter()
            try:
                return target(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started_at
                active_logger.info(
                    "%s completed in %.3f seconds", target.__name__, elapsed
                )

        return wrapped

    if function is not None:
        return decorate(function)
    return decorate


def _temporary_output_path(destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".tmp{destination.suffix}",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    compression: str = "zstd",
) -> Path:
    """Atomically save a DataFrame based on its filename extension."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_output_path(destination)

    try:
        suffix = destination.suffix.casefold()
        if suffix == ".parquet":
            dataframe.to_parquet(
                temporary_path,
                engine="pyarrow",
                compression=compression,
                index=index,
            )
        elif suffix == ".csv":
            dataframe.to_csv(temporary_path, index=index, encoding="utf-8")
        elif suffix in {".pkl", ".pickle"}:
            dataframe.to_pickle(temporary_path)
        elif suffix == ".feather":
            if index:
                raise ValueError("Feather output does not support index=True")
            dataframe.to_feather(temporary_path, compression=compression)
        else:
            raise ValueError(
                f"Unsupported DataFrame output format: {destination.suffix}"
            )
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logging.getLogger(__name__).info(
        "Saved %d rows to %s", len(dataframe), destination
    )
    return destination


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame saved by :func:`save_dataframe`."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DataFrame file not found: {source}")

    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        return pd.read_parquet(source, engine="pyarrow", dtype_backend="pyarrow")
    if suffix == ".csv":
        return pd.read_csv(source, encoding="utf-8")
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    if suffix == ".feather":
        return pd.read_feather(source)
    raise ValueError(f"Unsupported DataFrame input format: {source.suffix}")


def load_configuration(path: str | Path) -> dict[str, Any]:
    """Load a mapping from YAML, JSON, or TOML configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.casefold()
    try:
        if suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            value = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            with config_path.open("rb") as handle:
                value = tomllib.load(handle)
        else:
            raise ValueError(
                f"Unsupported configuration format: {config_path.suffix}"
            )
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Unable to load configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {config_path}: {exc}") from exc

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return dict(value)


def seed_everything(seed: int = 42) -> None:
    """Seed Python and NumPy and propagate the seed to child processes."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
