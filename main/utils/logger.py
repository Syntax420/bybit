import logging
import os
import sys
import time
import pytz
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Any
from utils.data_storage import save_trade_to_csv
import json
from datetime import datetime
import yaml

# Global logger instances
system_logger = None
error_logger = None
critical_logger = None
strategy_logger = None
api_logger = None

def _serialize_json_safe(obj: Any) -> Any:
    """
    Convert an object to be JSON serializable
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, bool):
        return obj  # Return booleans as is - JSON can handle them directly
    elif isinstance(obj, dict):
        return {k: _serialize_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_json_safe(item) for item in obj]
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO format string
    else:
        return str(obj)  # Convert any other types to strings

def get_current_time_with_timezone():
    """
    Get current time with timezone information
    
    Returns:
        String with formatted datetime and timezone
    """
    # Use UTC by default
    utc_now = datetime.now(pytz.UTC)
    
    # Format with timezone info
    return utc_now.strftime("%Y-%m-%d %H:%M:%S %Z")

def setup_logger(config: Dict = None) -> logging.Logger:
    """
    Set up and configure the logger
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    global system_logger, error_logger, critical_logger, strategy_logger, api_logger
    
    # Get log level from config
    if config and 'logging' in config:
        log_level_str = config.get('logging', {}).get('level', 'INFO')
    else:
        log_level_str = 'INFO'
        
    # Convert string to logging level
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create system logger with rotation (1MB max size)
    system_logger = logging.getLogger()
    system_log_handler = RotatingFileHandler(
        os.path.join(log_dir, 'system.log'),
        maxBytes=1024*1024,  # 1MB
        backupCount=5  # Keep 5 backup files
    )
    system_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    system_logger.addHandler(system_log_handler)
    
    # Create error logger (separate file for errors)
    error_logger = logging.getLogger('error')
    error_log_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=1024*1024,  # 1MB
        backupCount=5  # Keep 5 backup files
    )
    error_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    error_log_handler.setLevel(logging.ERROR)  # Only log ERROR level and above
    error_logger.addHandler(error_log_handler)
    
    # Create critical logger (separate file for critical errors)
    critical_logger = logging.getLogger('critical')
    critical_log_handler = RotatingFileHandler(
        os.path.join(log_dir, 'critical.log'),
        maxBytes=1024*1024,  # 1MB
        backupCount=5  # Keep 5 backup files
    )
    critical_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    critical_log_handler.setLevel(logging.CRITICAL)  # Only log CRITICAL level
    critical_logger.addHandler(critical_log_handler)
    
    # Create specific loggers for API calls and strategy decisions
    api_logger = logging.getLogger('api')
    api_log_handler = RotatingFileHandler(
        os.path.join(log_dir, 'api.log'),
        maxBytes=1024*1024,  # 1MB
        backupCount=3  # Keep 3 backup files
    )
    api_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    api_logger.addHandler(api_log_handler)
    
    strategy_logger = logging.getLogger('strategy')
    strategy_log_handler = RotatingFileHandler(
        os.path.join(log_dir, 'strategy.log'),
        maxBytes=1024*1024,  # 1MB
        backupCount=3  # Keep 3 backup files
    )
    strategy_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    strategy_logger.addHandler(strategy_log_handler)
    
    # Log startup information
    system_logger.info(f"Logger initialized with level: {log_level_str}")
    system_logger.info(f"System logs will be written to: {os.path.join(log_dir, 'system.log')}")
    system_logger.info(f"Error logs will be written to: {os.path.join(log_dir, 'error.log')}")
    system_logger.info(f"Critical logs will be written to: {os.path.join(log_dir, 'critical.log')}")
    system_logger.info(f"Current time with timezone: {get_current_time_with_timezone()}")
    
    return system_logger

def log_api_call(logger: logging.Logger, endpoint: str, method: str, params: Dict, response: Optional[Dict] = None, error: Optional[str] = None):
    """
    Log API calls with details about request and response
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        params: Request parameters
        response: API response (if available)
        error: Error message (if any)
    """
    # Create API call log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "timezone": datetime.now(pytz.UTC).tzname(),
        "endpoint": endpoint,
        "method": method,
        "params": _serialize_json_safe(params)  # Use serialization helper
    }
    
    # Add response or error
    if error:
        log_entry["status"] = "error"
        log_entry["error"] = error
        logger.error(f"API {method} to {endpoint} failed: {error}")
        # Also log to error log
        if error_logger:
            error_logger.error(f"API {method} to {endpoint} failed: {error}")
    else:
        log_entry["status"] = "success"
        if response:
            # Don't log the full response to avoid huge log files
            if isinstance(response, dict):
                status = response.get("retCode", "N/A")
                message = response.get("retMsg", "N/A")
                log_entry["response_status"] = status
                log_entry["response_message"] = message
                logger.info(f"API {method} to {endpoint} completed: status={status}, message={message}")
            else:
                logger.info(f"API {method} to {endpoint} completed with non-dict response")
    
    # Log to API-specific log
    if api_logger:
        api_logger.debug(json.dumps(_serialize_json_safe(log_entry)))
    
    return log_entry

def log_strategy_decision(logger: logging.Logger, symbol: str, timeframe: str, decision: str, 
                         signals: Dict, indicators: Dict = None, reason: str = None,
                         strategy_name: str = None, leverage: float = None):
    """
    Log strategy decisions with relevant indicators and signals
    
    Args:
        logger: Logger instance
        symbol: Trading symbol
        timeframe: Analysis timeframe
        decision: Decision taken (buy, sell, neutral)
        signals: Signal values
        indicators: Indicator values
        reason: Reason for the decision
        strategy_name: Name of the strategy used
        leverage: Leverage used for trading
    """
    # Create strategy log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "timezone": datetime.now(pytz.UTC).tzname(),
        "symbol": symbol,
        "timeframe": timeframe,
        "decision": decision
    }
    
    # Add strategy name if provided
    if strategy_name:
        log_entry["strategy_name"] = strategy_name
    
    # Add leverage if provided
    if leverage:
        log_entry["leverage"] = leverage
    
    # Ensure signals and indicators are JSON serializable
    log_entry["signals"] = _serialize_json_safe(signals)
    
    if indicators:
        log_entry["indicators"] = _serialize_json_safe(indicators)
        
    if reason:
        log_entry["reason"] = reason
    
    # Log to strategy-specific log
    if strategy_logger:
        strategy_logger.info(json.dumps(log_entry))
    
    # Log a summary to the main system log
    strategy_info = f"[{strategy_name}] " if strategy_name else ""
    leverage_info = f" (leverage: {leverage}x)" if leverage else ""
    
    logger.info(f"Strategy {strategy_info}{decision.upper()} signal for {symbol} ({timeframe}){leverage_info}: {reason if reason else 'Based on technical analysis'}")
    
    return log_entry

def log_data_load(logger: logging.Logger, source: str, symbol: str = None, 
                 timeframe: str = None, rows: int = None, success: bool = True, error: str = None):
    """
    Log data loading operations (CSV, API, etc.)
    
    Args:
        logger: Logger instance
        source: Data source (CSV, API, etc.)
        symbol: Trading symbol
        timeframe: Data timeframe
        rows: Number of data rows loaded
        success: Whether the load was successful
        error: Error message if not successful
    """
    if success:
        if symbol and timeframe:
            logger.info(f"Loaded {rows} rows of {symbol} {timeframe} data from {source}")
        else:
            logger.info(f"Loaded data from {source}")
    else:
        error_msg = f"Failed to load data from {source}: {error}"
        logger.error(error_msg)
        # Also log to error log
        if error_logger:
            error_logger.error(error_msg)

def log_trade(logger: logging.Logger, trade_data: Dict):
    """
    Log executed trades
    
    Args:
        logger: Logger instance
        trade_data: Trading data
    """
    # Extract key information
    symbol = trade_data.get('symbol', 'unknown')
    side = trade_data.get('side', 'unknown')
    price = trade_data.get('price', 0)
    qty = trade_data.get('quantity', 0)
    order_type = trade_data.get('order_type', 'unknown')
    strategy = trade_data.get('strategy', 'unknown')
    leverage = trade_data.get('leverage', 1)
    
    # Log trade execution with additional info
    logger.info(f"Trade executed: {side} {qty} {symbol} at {price} ({order_type}) - Strategy: {strategy}, Leverage: {leverage}x - Time: {get_current_time_with_timezone()}")
    
    # Add timezone info to trade data
    trade_data['timezone'] = datetime.now(pytz.UTC).tzname()
    
    # Write detailed trade log to trades.log
    trade_log_dir = 'logs'
    if not os.path.exists(trade_log_dir):
        os.makedirs(trade_log_dir)
        
    trade_log_path = os.path.join(trade_log_dir, 'trades.log')
    
    # Use rotating file handler for trades
    with open(trade_log_path, 'a') as f:
        # Use safe serialization for the trade data
        f.write(json.dumps(_serialize_json_safe(trade_data)) + '\n')
    
    # Save trade to CSV for historical analysis
    try:
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = time.time()
        save_result = save_trade_to_csv(trade_data)
        if not save_result:
            logger.warning(f"Failed to save trade data to CSV for {symbol}")
    except Exception as e:
        logger.error(f"Error saving trade to CSV: {str(e)}")

def log_performance(logger: logging.Logger, performance_data: Dict):
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        performance_data: Performance metrics
    """
    # Extract key metrics
    balance = performance_data.get('balance', 0)
    pnl = performance_data.get('daily_pnl', 0)
    win_rate = performance_data.get('win_rate', 0)
    
    # Add timezone info
    performance_data['timezone'] = datetime.now(pytz.UTC).tzname()
    performance_data['timestamp_iso'] = datetime.now().isoformat()
    
    # Log performance summary
    logger.info(f"Performance update: Balance=${balance:.2f}, PnL=${pnl:.2f}, Win Rate={win_rate:.1f}% - Time: {get_current_time_with_timezone()}")
    
    # Write detailed performance log to performance.log
    perf_log_dir = 'logs'
    if not os.path.exists(perf_log_dir):
        os.makedirs(perf_log_dir)
        
    perf_log_path = os.path.join(perf_log_dir, 'performance.log')
    
    with open(perf_log_path, 'a') as f:
        # Use safe serialization for the performance data
        f.write(json.dumps(_serialize_json_safe(performance_data)) + '\n')

def log_error(logger: logging.Logger, error_type: str, error_msg: str, context: Dict = None):
    """
    Log detailed error information
    
    Args:
        logger: Logger instance
        error_type: Type of error
        error_msg: Error message
        context: Additional context information
    """
    # Add timezone info
    tz_info = f" - Time: {get_current_time_with_timezone()}"
    
    # Log to main logger
    logger.error(f"{error_type}: {error_msg}{tz_info}")
    
    # Create detailed error entry
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "timezone": datetime.now(pytz.UTC).tzname(),
        "error_type": error_type,
        "error_msg": error_msg
    }
    
    if context:
        error_entry["context"] = _serialize_json_safe(context)  # Use serialization helper
    
    # Also log to error log
    if error_logger:
        # Use safe serialization for contexts that might contain non-serializable objects
        safe_context = json.dumps(_serialize_json_safe(context)) if context else 'None'
        error_logger.error(f"{error_type}: {error_msg}{tz_info} | Context: {safe_context}")
    
    # Write to error log file
    error_log_dir = 'logs'
    if not os.path.exists(error_log_dir):
        os.makedirs(error_log_dir)
        
    error_log_path = os.path.join(error_log_dir, 'error.log')
    
    with open(error_log_path, 'a') as f:
        # Use safe serialization for the whole error entry
        f.write(json.dumps(_serialize_json_safe(error_entry)) + '\n')

def log_critical_error(logger: logging.Logger, error_type: str, error_msg: str, context: Dict = None):
    """
    Log critical error information (separate from normal errors)
    
    Args:
        logger: Logger instance
        error_type: Type of error
        error_msg: Error message
        context: Additional context information
    """
    # Add timezone info
    tz_info = f" - Time: {get_current_time_with_timezone()}"
    
    # Log to main logger with CRITICAL level
    logger.critical(f"CRITICAL {error_type}: {error_msg}{tz_info}")
    
    # Create detailed error entry
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "timezone": datetime.now(pytz.UTC).tzname(),
        "error_type": error_type,
        "error_severity": "CRITICAL",
        "error_msg": error_msg
    }
    
    if context:
        error_entry["context"] = context
    
    # Also log to critical log
    if critical_logger:
        critical_logger.critical(f"{error_type}: {error_msg}{tz_info} | Context: {json.dumps(context) if context else 'None'}")
    
    # Write to critical log file
    critical_log_dir = 'logs'
    if not os.path.exists(critical_log_dir):
        os.makedirs(critical_log_dir)
        
    critical_log_path = os.path.join(critical_log_dir, 'critical.log')
    
    with open(critical_log_path, 'a') as f:
        f.write(json.dumps(error_entry) + '\n')

def log_exception(logger: logging.Logger, exc: Exception, context: str = None, stack_trace: str = None, is_critical: bool = False):
    """
    Log exception information
    
    Args:
        logger: Logger instance
        exc: Exception object
        context: Context where the exception occurred
        stack_trace: Stack trace of the exception
        is_critical: Whether this exception is critical
    """
    error_msg = f"Exception: {str(exc)}"
    if context:
        error_msg += f" in {context}"
    
    # Add timezone info
    tz_info = f" - Time: {get_current_time_with_timezone()}"
    
    # Log to main logger with appropriate level
    if is_critical:
        logger.critical(f"{error_msg}{tz_info}")
    else:
        logger.error(f"{error_msg}{tz_info}")
    
    # Create detailed exception entry
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "timezone": datetime.now(pytz.UTC).tzname(),
        "exception_type": exc.__class__.__name__,
        "exception_msg": str(exc),
        "severity": "CRITICAL" if is_critical else "ERROR",
        "context": context
    }
    
    if stack_trace:
        error_entry["stack_trace"] = stack_trace
    
    # Also log to appropriate log
    if is_critical and critical_logger:
        critical_logger.critical(f"{error_msg}{tz_info}\nStack trace: {stack_trace if stack_trace else 'Not provided'}")
    elif error_logger:
        error_logger.error(f"{error_msg}{tz_info}\nStack trace: {stack_trace if stack_trace else 'Not provided'}")
    
    # Write to appropriate log file
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, 'critical.log' if is_critical else 'error.log')
    
    with open(log_path, 'a') as f:
        # Use safe serialization for the whole error entry
        f.write(json.dumps(_serialize_json_safe(error_entry)) + '\n')