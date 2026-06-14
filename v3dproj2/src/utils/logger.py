"""
Structured logging module for the V3D Drone Control System.
Provides file and console logging with configurable levels and formatting.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.config import get_config


class AppLogger:
    """
    Centralized logger with console and file output.
    Uses rotating file handler to prevent unbounded log growth.
    """

    _instance: Optional[AppLogger] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls) -> AppLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._setup()

    def _setup(self) -> None:
        self._logger = logging.getLogger("V3DDroneControl")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        try:
            cfg = get_config()
            log_level_str = cfg.get("logging.level", "INFO")
            log_format = cfg.get(
                "logging.format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            log_file = cfg.get("logging.file", "logs/app.log")
            console_enabled = cfg.get("logging.console", True)
        except Exception:
            log_level_str = "INFO"
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            log_file = "logs/app.log"
            console_enabled = True

        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        self._logger.setLevel(log_level)

        formatter = logging.Formatter(log_format)

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        if self._logger is None:
            self._setup()
        if self._logger is None:
            self._logger = logging.getLogger("V3DDroneControl_Fallback")
        return self._logger

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        self.get_logger().exception(msg, *args, **kwargs)


def get_logger() -> AppLogger:
    return AppLogger()
