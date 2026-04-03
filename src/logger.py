# Import the logging module to create application logs.
import logging

# Import the os module for directory creation.
import os

# Import log directory and file path settings from the configuration file.
from src.config import LOGS_DIR, LOG_FILE_PATH

# Create the logs directory if it does not already exist.
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure the logging system for the project.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Define a helper function that returns a logger object for a given module name.
def get_logger(name: str):
    # Return a logger instance using the provided name.
    return logging.getLogger(name)