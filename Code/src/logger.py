
import logging
import time
import sys
from contextlib import contextmanager
import datetime

def setup_logger(name="ProjectLogger", log_file="../Logs/execution.log"):
    """
    Sets up a logger that writes to both console and a log file.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs
    if logger.hasHandlers():
        return logger

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

@contextmanager
def log_execution_time(logger, task_name="Task"):
    """
    Context manager to log the execution time of a block of code.
    
    Args:
        logger (logging.Logger): Logger instance to use.
        task_name (str): Name of the task being measured.
    """
    start_time = time.time()
    logger.info(f"START: {task_name}")
    try:
        yield
    except Exception as e:
        logger.error(f"FAILED: {task_name} - Error: {e}")
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"END: {task_name} - Duration: {duration:.4f} seconds")
