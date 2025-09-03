import logging
import sys
import os
from multiprocessing import Queue, current_process
from logging.handlers import QueueHandler, QueueListener
import threading

# Global variables for multiprocessing logging
log_queue = None
queue_listener = None
_lock = threading.Lock()


def setup_logger(name=None, level=None):
    """
    Get a logger instance configured for both local and production environments.
    Works correctly with multiprocessing.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Logging level (defaults to INFO, or DEBUG if DEBUG env var is set)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = current_process().name
    
    if level is None:
        level = logging.DEBUG if os.getenv('DEBUG') else logging.INFO
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # If we're in a worker process, use QueueHandler
        if log_queue is not None and current_process().name != 'MainProcess':
            handler = QueueHandler(log_queue)
            logger.addHandler(handler)
        else:
            # Main process or single-process mode: use StreamHandler
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(processName)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    return logger


def setup_multiprocess_logging(level=None):
    """
    Initialize multiprocessing-aware logging.
    Call this once in the main process before spawning workers.
    
    Args:
        level: Logging level for all loggers
    """
    global log_queue, queue_listener
    
    with _lock:
        if log_queue is not None:
            return  # Already initialized
        
        if level is None:
            level = logging.DEBUG if os.getenv('DEBUG') else logging.INFO
        
        # Create queue for multiprocessing
        log_queue = Queue()
        
        # Set up the main handler that will write to stdout
        main_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(processName)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(formatter)
        
        # Create and start the listener
        queue_listener = QueueListener(log_queue, main_handler, respect_handler_level=True)
        queue_listener.start()
        
        # Force unbuffered output for Docker/Railway
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)


def worker_init(queue):
    """
    Initialize logging in a worker process.
    Pass this to ProcessPoolExecutor's initializer parameter.
    
    Args:
        queue: The multiprocessing Queue for log messages
    """
    global log_queue
    log_queue = queue
    
    # Reconfigure stdout/stderr for unbuffered output in worker
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Set up the logger for this worker - this will now use the queue
    setup_logger()


def get_multiprocess_queue():
    """
    Get the multiprocessing log queue.
    Used for passing to worker processes.
    
    Returns:
        The log queue if multiprocess logging is set up, None otherwise
    """
    return log_queue


def shutdown_logging():
    """
    Clean shutdown of the logging system.
    Call this when the application exits.
    """
    global queue_listener
    
    if queue_listener:
        queue_listener.stop()
        queue_listener = None