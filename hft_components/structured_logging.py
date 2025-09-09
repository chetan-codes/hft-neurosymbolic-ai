#!/usr/bin/env python3
"""
Structured Logging with Correlation IDs for HFT Neurosymbolic AI System
"""

import structlog
import logging
import uuid
import time
import json
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime
import asyncio

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class CorrelationIDProcessor:
    """Processor to add correlation ID to log records"""
    
    def __call__(self, logger, method_name, event_dict):
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        return event_dict

class TimingProcessor:
    """Processor to add timing information"""
    
    def __call__(self, logger, method_name, event_dict):
        if 'start_time' in event_dict:
            duration = time.time() - event_dict['start_time']
            event_dict['duration_ms'] = round(duration * 1000, 2)
            del event_dict['start_time']
        return event_dict

class JSONFormatter:
    """Custom JSON formatter for structured logs"""
    
    def __call__(self, logger, method_name, event_dict):
        # Add timestamp
        event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service name
        event_dict['service'] = 'hft_neurosymbolic'
        
        # Add log level
        event_dict['level'] = method_name.upper()
        
        return event_dict

def setup_structured_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Setup structured logging with correlation IDs
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (json, console)
        log_file: Optional log file path
    
    Returns:
        Configured structlog logger
    """
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        CorrelationIDProcessor(),
        TimingProcessor(),
        JSONFormatter(),
    ]
    
    if log_format == "console":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()

class CorrelationContext:
    """Context manager for correlation IDs"""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.token = None
    
    def __enter__(self):
        self.token = correlation_id_var.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            correlation_id_var.reset(self.token)

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()

def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context"""
    correlation_id_var.set(correlation_id)

class LoggedFunction:
    """Decorator for logging function calls with correlation IDs"""
    
    def __init__(self, logger: structlog.BoundLogger, log_args: bool = True, log_result: bool = False):
        self.logger = logger
        self.log_args = log_args
        self.log_result = log_result
    
    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                correlation_id = get_correlation_id()
                start_time = time.time()
                
                log_data = {
                    'function': func.__name__,
                    'start_time': start_time,
                    'correlation_id': correlation_id
                }
                
                if self.log_args:
                    log_data['args'] = str(args)[:200]  # Truncate long args
                    log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
                
                self.logger.info("Function call started", **log_data)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if self.log_result:
                        log_data['result'] = str(result)[:200]
                    
                    self.logger.info("Function call completed", **log_data)
                    return result
                    
                except Exception as e:
                    log_data['error'] = str(e)
                    log_data['error_type'] = type(e).__name__
                    self.logger.error("Function call failed", **log_data)
                    raise
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                correlation_id = get_correlation_id()
                start_time = time.time()
                
                log_data = {
                    'function': func.__name__,
                    'start_time': start_time,
                    'correlation_id': correlation_id
                }
                
                if self.log_args:
                    log_data['args'] = str(args)[:200]
                    log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
                
                self.logger.info("Function call started", **log_data)
                
                try:
                    result = func(*args, **kwargs)
                    
                    if self.log_result:
                        log_data['result'] = str(result)[:200]
                    
                    self.logger.info("Function call completed", **log_data)
                    return result
                    
                except Exception as e:
                    log_data['error'] = str(e)
                    log_data['error_type'] = type(e).__name__
                    self.logger.error("Function call failed", **log_data)
                    raise
            
            return sync_wrapper

class TradingSignalLogger:
    """Specialized logger for trading signals"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_signal_generation(self, symbol: str, strategy: str, **kwargs):
        """Log trading signal generation"""
        self.logger.info(
            "Trading signal generation started",
            symbol=symbol,
            strategy=strategy,
            **kwargs
        )
    
    def log_signal_result(self, symbol: str, action: str, confidence: float, **kwargs):
        """Log trading signal result"""
        self.logger.info(
            "Trading signal generated",
            symbol=symbol,
            action=action,
            confidence=confidence,
            **kwargs
        )
    
    def log_ai_prediction(self, symbol: str, model_type: str, confidence: float, **kwargs):
        """Log AI prediction"""
        self.logger.info(
            "AI prediction completed",
            symbol=symbol,
            model_type=model_type,
            confidence=confidence,
            **kwargs
        )
    
    def log_symbolic_analysis(self, symbol: str, regime: str, signal: str, **kwargs):
        """Log symbolic analysis"""
        self.logger.info(
            "Symbolic analysis completed",
            symbol=symbol,
            market_regime=regime,
            technical_signal=signal,
            **kwargs
        )

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_database_query(self, database: str, query_type: str, duration_ms: float, **kwargs):
        """Log database query performance"""
        self.logger.info(
            "Database query",
            database=database,
            query_type=query_type,
            duration_ms=duration_ms,
            **kwargs
        )

# Global logger instance
logger = setup_structured_logging()

# Specialized loggers
trading_logger = TradingSignalLogger(logger)
performance_logger = PerformanceLogger(logger)

# Convenience functions
def log_function(log_args: bool = True, log_result: bool = False):
    """Decorator for logging function calls"""
    return LoggedFunction(logger, log_args, log_result)

def with_correlation_id(correlation_id: Optional[str] = None):
    """Context manager for correlation IDs"""
    return CorrelationContext(correlation_id)
