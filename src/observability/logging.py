"""Structured logging + timing. Swap the handler for OpenTelemetry/LangSmith in production."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"level": record.levelname, "event": record.getMessage()}
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS})
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(log: logging.Logger, event: str, **fields: Any) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(event, extra={**fields, "latency_ms": ms})
