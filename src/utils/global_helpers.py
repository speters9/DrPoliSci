import functools
import sys
import logging
import yaml

from pyprojroot.here import here

from src.utils.config_types import *

def graceful_exit(func):
    """Gracefully exit a longrunning script."""
    @functools.wraps(func)
    def wrapper_graceful_exit(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("Keyboard Interrupt detected. Exiting gracefully...")
            sys.exit(0)
    return wrapper_graceful_exit

def logger_setup(logger_name="query_logger", log_level=logging.INFO):
    """
    Set up and return a logger with the specified name and level.
    Avoids affecting the root logger by setting propagate to False.

    Args:
        logger_name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Retrieve or create a logger
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if already set up
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)  # Match handler level to logger level

        # Set the format for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - raised_by: %(name)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    # Set the logger level explicitly and prevent it from propagating to the root
    logger.setLevel(log_level)
    logger.propagate = True

    return logger


def load_config(path="config.yaml") -> FullConfig:
    with open(here(path), "r") as f:
        raw = yaml.safe_load(f)

    return FullConfig(
        app=AppConfig(**raw["app"]),
        model=ModelConfig(**raw["model"]),
        paths=PathsConfig(**{k: here(v) for k, v in raw["paths"].items()}),
        timeouts=TimeoutsConfig(**raw["timeouts"]),
        summary=SummaryConfig(**raw["summary"]),
        graph=GraphConfig(**raw["graph"]),
    )
