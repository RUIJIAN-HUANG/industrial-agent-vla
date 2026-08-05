"""Standalone YOLO perception service."""

from .config import load_config
from .routes import YoloService

__version__ = "0.1.0"

__all__ = ["YoloService", "__version__", "load_config"]
