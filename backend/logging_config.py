# logging_config.py
import logging

def setup_logging():
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set logging level
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a file handler and set its level and file name
    file_handler = logging.FileHandler('loggingdata.txt')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Initialize the logger
logger = setup_logging()
