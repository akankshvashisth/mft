"""Logging interface of the Nautilus-free services.

Shells inject PortsLog (Nautilus logger); without an injected log a service uses its stdlib module logger,
which logging.Logger satisfies structurally.
"""
from __future__ import annotations

from typing import Protocol


class LogLike(Protocol):
    def info(self, msg: str) -> None: ...

    def warning(self, msg: str) -> None: ...

    def error(self, msg: str) -> None: ...
