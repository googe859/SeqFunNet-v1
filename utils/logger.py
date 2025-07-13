"""
Logging utilities for MixerEncoding.
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file (optional)
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class Logger:
    """
    Custom logger that writes to both console and file.
    """
    
    def __init__(self, filename: str):
        """
        Initialize logger.
        
        Args:
            filename: Log file path
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message: str):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        """Flush both terminal and file."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """Close the log file."""
        self.log.close() 