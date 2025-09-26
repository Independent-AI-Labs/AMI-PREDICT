"""
Logging system for CryptoBot Pro
"""

import json
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class Logger:
    """Centralized logging system"""

    def __init__(self, config: Optional[dict] = None):
        """Initialize logger

        Args:
            config: Logging configuration
        """
        self.config = config or {}
        self._setup_logger()

    def _setup_logger(self):
        """Configure logger"""
        # Remove default handler
        logger.remove()

        # Console handler with color
        level = self.config.get("level", "INFO")
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            colorize=True,
        )

        # File handler with JSON format
        log_file = self.config.get("file", "logs/cryptobot.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.get("format") == "json":
            logger.add(
                log_file,
                level=level,
                format=self._json_format,
                rotation=self.config.get("max_size", "100 MB"),
                retention=f"{self.config.get('max_age', 30)} days",
                compression="zip" if self.config.get("compress", True) else None,
                serialize=True,
            )
        else:
            logger.add(
                log_file,
                level=level,
                rotation=self.config.get("max_size", "100 MB"),
                retention=f"{self.config.get('max_age', 30)} days",
                compression="zip" if self.config.get("compress", True) else None,
            )

    def _json_format(self, record):
        """Format log record as JSON"""
        return json.dumps(
            {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "extra": record.get("extra", {}),
            }
        )

    def get_logger(self, name: Optional[str] = None):
        """Get logger instance

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name:
            return logger.bind(name=name)
        return logger

    @staticmethod
    def debug(message: str, **kwargs):
        """Log debug message"""
        logger.debug(message, **kwargs)

    @staticmethod
    def info(message: str, **kwargs):
        """Log info message"""
        logger.info(message, **kwargs)

    @staticmethod
    def warning(message: str, **kwargs):
        """Log warning message"""
        logger.warning(message, **kwargs)

    @staticmethod
    def error(message: str, **kwargs):
        """Log error message"""
        logger.error(message, **kwargs)

    @staticmethod
    def critical(message: str, **kwargs):
        """Log critical message"""
        logger.critical(message, **kwargs)

    @staticmethod
    def exception(message: str, **kwargs):
        """Log exception with traceback"""
        logger.exception(message, **kwargs)
