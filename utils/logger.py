import os
import logging
from datetime import datetime

def get_logger(config, main_rank=True):
    """Create a logger for logging training progress.
    
    Args:
        config: Configuration object containing logging settings
        main_rank: Whether this is the main process (for DDP training)
    
    Returns:
        logger: Configured logger instance
    """
    if not main_rank:
        return None
        
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(config.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('semantic_segmentation')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 