import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import functools

class StructuredLogger:
    """Structured logging with correlation IDs and context management."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log with structured context."""
        correlation_id = kwargs.pop('correlation_id', str(uuid.uuid4())[:8])
        extra_data = {
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Add extra data to logger's extra
        extra = {'correlation_id': correlation_id}
        getattr(self.logger, level.lower())(
            f"{message} | Context: {json.dumps(extra_data)}", 
            extra=extra
        )
    
    def info(self, message: str, **kwargs):
        self._log_with_context("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context("DEBUG", message, **kwargs)

def log_execution_time(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        logger = StructuredLogger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Function executed successfully",
                function=func.__name__,
                execution_time=execution_time,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Function execution failed",
                function=func.__name__,
                execution_time=execution_time,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    return wrapper