import logging
import sys
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter as _JsonFormatter

from app.config import settings

conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="-")


class CorrelationFilter(logging.Filter):
    """Injects conversation_id from contextvars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.conversation_id = conversation_id_var.get("-")
        return True


def setup_logging() -> None:
    """Configure structured JSON logging with correlation IDs."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_FORMAT == "json":
        formatter = _JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(conversation_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(conversation_id)s] %(name)s: %(message)s"
        )

    handler.setFormatter(formatter)
    handler.addFilter(CorrelationFilter())
    root.addHandler(handler)
