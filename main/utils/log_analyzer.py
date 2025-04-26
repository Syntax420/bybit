import os
import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger("log_analyzer")

class LogAnalyzer:
    """
    Analyzes log files for patterns, errors, and statistics
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the log analyzer
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = log_dir
        self.error_types = [
            "ImportError", 
            "ValueError", 
            "KeyError", 
            "TypeError", 
            "IndexError", 
            "AttributeError", 
            "JSONDecodeError", 
            "APIError", 
            "NetworkError", 
            "TimeoutError",
            "NameError",
            "SyntaxError"
        ]
        self.error_patterns = {
            "API Error": r"API .* failed",
            "Connection Error": r"Connection (?:error|timeout|refused)",
            "Data Error": r"No data|Missing data|Empty response|Invalid data",
            "Authentication Error": r"Authentication (?:failed|error|invalid)",
            "Rate Limit": r"Rate limit|Too many requests",
            "Permission Error": r"Permission denied|Not authorized|Unauthorized"
        }
    
    def analyze_error_log(self, days_back: int = 1) -> Dict:
        """
        Analyze the error log file and generate statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Dictionary with error statistics
        """
        error_log_path = os.path.join(self.log_dir, "error.log")
        critical_log_path = os.path.join(self.log_dir, "critical.log")
        
        if not os.path.exists(error_log_path):
            logger.warning(f"Error log file not found: {error_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "total_errors": 0,
            "total_critical": 0,
            "error_types": defaultdict(int),
            "error_patterns": defaultdict(int),
            "most_frequent_errors": [],
            "recent_errors": [],
            "errors_by_strategy": defaultdict(int),
            "errors_by_symbol": defaultdict(int),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Process error log
        self._process_log_file(error_log_path, stats, cutoff_date, is_critical=False)
        
        # Process critical log if it exists
        if os.path.exists(critical_log_path):
            self._process_log_file(critical_log_path, stats, cutoff_date, is_critical=True)
        
        # Sort the most frequent errors
        stats["most_frequent_errors"] = sorted(
            [{"error_type": k, "count": v} for k, v in stats["error_types"].items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]  # Top 10 errors
        
        return stats
    
    def _process_log_file(self, log_path: str, stats: Dict, cutoff_date: datetime, is_critical: bool = False):
        """
        Process a single log file and update the statistics
        
        Args:
            log_path: Path to the log file
            stats: Statistics dictionary to update
            cutoff_date: Cutoff date for filtering old entries
            is_critical: Whether the log contains critical errors
        """
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON
                        entry = json.loads(line.strip())
                        
                        # Skip if older than cutoff date
                        if "timestamp" in entry:
                            try:
                                entry_date = datetime.fromisoformat(entry["timestamp"])
                                if entry_date < cutoff_date:
                                    continue
                            except (ValueError, TypeError):
                                # If we can't parse the date, include it anyway
                                pass
                        
                        # Update the total count
                        if is_critical:
                            stats["total_critical"] += 1
                        else:
                            stats["total_errors"] += 1
                        
                        # Extract error type
                        error_type = entry.get("exception_type") or entry.get("error_type") or "Unknown"
                        stats["error_types"][error_type] += 1
                        
                        # Extract error message for pattern matching
                        error_msg = entry.get("exception_msg") or entry.get("error_msg") or ""
                        
                        # Match against known patterns
                        for pattern_name, pattern in self.error_patterns.items():
                            if re.search(pattern, error_msg, re.IGNORECASE):
                                stats["error_patterns"][pattern_name] += 1
                        
                        # Extract strategy and symbol information if available
                        context = entry.get("context", {})
                        if isinstance(context, dict):
                            if "strategy" in context:
                                stats["errors_by_strategy"][context["strategy"]] += 1
                            if "symbol" in context:
                                stats["errors_by_symbol"][context["symbol"]] += 1
                        elif isinstance(context, str):
                            # Try to extract symbol from context string
                            symbol_match = re.search(r"for ([A-Z0-9]+)", context)
                            if symbol_match:
                                stats["errors_by_symbol"][symbol_match.group(1)] += 1
                        
                        # Add to recent errors list (limit to 20)
                        if len(stats["recent_errors"]) < 20:
                            stats["recent_errors"].append({
                                "type": error_type,
                                "message": error_msg[:100] + ("..." if len(error_msg) > 100 else ""),
                                "timestamp": entry.get("timestamp", "unknown")
                            })
                    
                    except json.JSONDecodeError:
                        # Not a JSON line, try to extract info using regex
                        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)
                        timestamp = timestamp_match.group(0) if timestamp_match else "unknown"
                        
                        # Look for known error types
                        for error_type in self.error_types:
                            if error_type in line:
                                stats["error_types"][error_type] += 1
                                if is_critical:
                                    stats["total_critical"] += 1
                                else:
                                    stats["total_errors"] += 1
                                break
                        
                        # Check for patterns
                        for pattern_name, pattern in self.error_patterns.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                stats["error_patterns"][pattern_name] += 1
                                break
        
        except Exception as e:
            logger.error(f"Error processing log file {log_path}: {str(e)}")
    
    def generate_summary(self, days_back: int = 1) -> str:
        """
        Generate a text summary of error statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text summary of error statistics
        """
        stats = self.analyze_error_log(days_back)
        
        if not stats:
            return "No error logs found for analysis."
        
        # Format the timestamp
        timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M Uhr")
        
        summary = [
            f"Error Summary (Stand: {timestamp}, last {days_back} day{'s' if days_back > 1 else ''}):",
            f"- Total Errors: {stats['total_errors']}",
            f"- Critical Errors: {stats['total_critical']}",
            "\nError Types:",
        ]
        
        # Add error types
        for error in sorted(stats["error_types"].items(), key=lambda x: x[1], reverse=True):
            summary.append(f"- {error[1]}x {error[0]}")
        
        # Add error patterns if any
        if stats["error_patterns"]:
            summary.append("\nError Patterns:")
            for pattern, count in sorted(stats["error_patterns"].items(), key=lambda x: x[1], reverse=True):
                summary.append(f"- {count}x {pattern}")
        
        # Add strategy errors if any
        if stats["errors_by_strategy"]:
            summary.append("\nErrors by Strategy:")
            for strategy, count in sorted(stats["errors_by_strategy"].items(), key=lambda x: x[1], reverse=True):
                summary.append(f"- {count}x {strategy}")
        
        # Add symbol errors if any
        if stats["errors_by_symbol"]:
            summary.append("\nMost Affected Symbols:")
            for symbol, count in sorted(stats["errors_by_symbol"].items(), key=lambda x: x[1], reverse=True)[:5]:
                summary.append(f"- {count}x {symbol}")
        
        # Add recent errors
        if stats["recent_errors"]:
            summary.append("\nMost Recent Errors:")
            for i, error in enumerate(stats["recent_errors"][:5], 1):
                summary.append(f"{i}. [{error['timestamp']}] {error['type']}: {error['message']}")
        
        return "\n".join(summary)
    
    def print_summary(self, days_back: int = 1):
        """
        Print the error summary to the console
        
        Args:
            days_back: Number of days to look back in logs
        """
        summary = self.generate_summary(days_back)
        print(summary)
        return summary
    
    def log_summary(self, days_back: int = 1):
        """
        Log the error summary to the system log
        
        Args:
            days_back: Number of days to look back in logs
        """
        summary = self.generate_summary(days_back)
        system_logger = logging.getLogger()
        system_logger.info("LOG ANALYSIS SUMMARY\n" + summary)
        return summary
    
    def run_scheduled_analysis(self, interval_hours: int = 6):
        """
        Schedule regular log analysis
        
        Args:
            interval_hours: How often to run the analysis (in hours)
        """
        import schedule
        import time
        
        # Run immediately
        self.log_summary()
        
        # Schedule future runs
        schedule.every(interval_hours).hours.do(self.log_summary)
        
        logger.info(f"Scheduled log analysis to run every {interval_hours} hours")

    def analyze_strategy_log(self, days_back: int = 1, strategy_filter: Optional[str] = None) -> Dict:
        """
        Analyze the strategy log file and generate statistics
        
        Args:
            days_back: Number of days to look back in logs
            strategy_filter: Only analyze logs for this strategy (if provided)
            
        Returns:
            Dictionary with strategy statistics
        """
        strategy_log_path = os.path.join(self.log_dir, "strategy.log")
        
        if not os.path.exists(strategy_log_path):
            logger.warning(f"Strategy log file not found: {strategy_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "neutral_signals": 0,
            "signals_by_strategy": defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0, "total": 0}),
            "signals_by_symbol": defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0, "total": 0}),
            "signals_by_timeframe": defaultdict(lambda: {"buy": 0, "sell": 0, "neutral": 0, "total": 0}),
            "signals_timeline": [],
            "most_active_symbols": [],
            "most_active_strategies": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            with open(strategy_log_path, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON
                        entry = json.loads(line.strip())
                        
                        # Skip if older than cutoff date
                        if "timestamp" in entry:
                            try:
                                entry_date = datetime.fromisoformat(entry["timestamp"])
                                if entry_date < cutoff_date:
                                    continue
                            except (ValueError, TypeError):
                                # If we can't parse the date, include it anyway
                                pass
                        
                        # Skip if not matching strategy filter
                        if strategy_filter and entry.get("strategy_name") != strategy_filter:
                            continue
                        
                        # Extract basic info
                        symbol = entry.get("symbol", "unknown")
                        timeframe = entry.get("timeframe", "unknown")
                        decision = entry.get("decision", "neutral").lower()
                        strategy_name = entry.get("strategy_name", "unknown")
                        
                        # Update total counts
                        stats["total_signals"] += 1
                        if decision == "buy":
                            stats["buy_signals"] += 1
                        elif decision == "sell":
                            stats["sell_signals"] += 1
                        else:
                            stats["neutral_signals"] += 1
                        
                        # Update by strategy
                        stats["signals_by_strategy"][strategy_name]["total"] += 1
                        stats["signals_by_strategy"][strategy_name][decision] += 1
                        
                        # Update by symbol
                        stats["signals_by_symbol"][symbol]["total"] += 1
                        stats["signals_by_symbol"][symbol][decision] += 1
                        
                        # Update by timeframe
                        stats["signals_by_timeframe"][timeframe]["total"] += 1
                        stats["signals_by_timeframe"][timeframe][decision] += 1
                        
                        # Add to timeline
                        stats["signals_timeline"].append({
                            "timestamp": entry.get("timestamp", "unknown"),
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "decision": decision,
                            "strategy": strategy_name,
                            "reason": entry.get("reason", "")
                        })
                    
                    except json.JSONDecodeError:
                        # Not a JSON line, try to extract info using regex
                        continue
        
        except Exception as e:
            logger.error(f"Error processing strategy log file: {str(e)}")
        
        # Calculate most active symbols and strategies
        stats["most_active_symbols"] = sorted(
            [{"symbol": k, "count": v["total"]} for k, v in stats["signals_by_symbol"].items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]  # Top 10 symbols
        
        stats["most_active_strategies"] = sorted(
            [{"strategy": k, "count": v["total"]} for k, v in stats["signals_by_strategy"].items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]  # Top 10 strategies
        
        return stats
    
    def generate_strategy_summary(self, days_back: int = 1, strategy_filter: Optional[str] = None) -> str:
        """
        Generate a text summary of strategy statistics
        
        Args:
            days_back: Number of days to look back in logs
            strategy_filter: Only analyze logs for this strategy (if provided)
            
        Returns:
            Text summary of strategy statistics
        """
        stats = self.analyze_strategy_log(days_back, strategy_filter)
        
        if not stats or stats["total_signals"] == 0:
            return "No strategy logs found for analysis."
        
        # Format the timestamp
        timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M Uhr")
        
        strategy_text = f" ({strategy_filter})" if strategy_filter else ""
        
        summary = [
            f"Strategy Analysis{strategy_text} (Stand: {timestamp}, last {days_back} day{'s' if days_back > 1 else ''}):",
            f"- Total Signals: {stats['total_signals']}",
            f"- Buy Signals: {stats['buy_signals']} ({stats['buy_signals']/stats['total_signals']*100:.1f}%)",
            f"- Sell Signals: {stats['sell_signals']} ({stats['sell_signals']/stats['total_signals']*100:.1f}%)",
            f"- Neutral Signals: {stats['neutral_signals']} ({stats['neutral_signals']/stats['total_signals']*100:.1f}%)",
            "\nMost Active Symbols:",
        ]
        
        # Add most active symbols
        for i, symbol_data in enumerate(stats["most_active_symbols"][:5], 1):
            symbol = symbol_data["symbol"]
            count = symbol_data["count"]
            buy = stats["signals_by_symbol"][symbol]["buy"]
            sell = stats["signals_by_symbol"][symbol]["sell"]
            summary.append(f"{i}. {symbol}: {count} signals (Buy: {buy}, Sell: {sell})")
        
        # Add most active strategies
        if len(stats["most_active_strategies"]) > 1:  # Only if we have multiple strategies
            summary.append("\nMost Active Strategies:")
            for i, strategy_data in enumerate(stats["most_active_strategies"][:5], 1):
                strategy = strategy_data["strategy"]
                count = strategy_data["count"]
                buy = stats["signals_by_strategy"][strategy]["buy"]
                sell = stats["signals_by_strategy"][strategy]["sell"]
                summary.append(f"{i}. {strategy}: {count} signals (Buy: {buy}, Sell: {sell})")
        
        # Add signals by timeframe
        summary.append("\nSignals by Timeframe:")
        for timeframe, data in sorted(stats["signals_by_timeframe"].items(), 
                                     key=lambda x: x[1]["total"], reverse=True):
            summary.append(f"- {timeframe}: {data['total']} signals (Buy: {data['buy']}, Sell: {data['sell']})")
        
        return "\n".join(summary)

    def analyze_api_log(self, days_back: int = 1) -> Dict:
        """
        Analyze the API log file and generate statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Dictionary with API call statistics
        """
        api_log_path = os.path.join(self.log_dir, "api.log")
        
        if not os.path.exists(api_log_path):
            logger.warning(f"API log file not found: {api_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "calls_by_endpoint": defaultdict(lambda: {"success": 0, "error": 0, "total": 0}),
            "calls_by_method": defaultdict(lambda: {"success": 0, "error": 0, "total": 0}),
            "error_types": defaultdict(int),
            "recent_errors": [],
            "hourly_distribution": defaultdict(int),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            with open(api_log_path, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON
                        entry = json.loads(line.strip())
                        
                        # Skip if older than cutoff date
                        if "timestamp" in entry:
                            try:
                                entry_date = datetime.fromisoformat(entry["timestamp"])
                                if entry_date < cutoff_date:
                                    continue
                                
                                # Extract hour for hourly distribution
                                hour = entry_date.hour
                                stats["hourly_distribution"][hour] += 1
                                
                            except (ValueError, TypeError):
                                # If we can't parse the date, include it anyway
                                pass
                        
                        # Extract basic info
                        endpoint = entry.get("endpoint", "unknown")
                        method = entry.get("method", "unknown")
                        status = entry.get("status", "unknown")
                        
                        # Update total counts
                        stats["total_calls"] += 1
                        if status == "success":
                            stats["success_calls"] += 1
                        elif status == "error":
                            stats["error_calls"] += 1
                            error_msg = entry.get("error", "Unknown error")
                            stats["error_types"][error_msg] += 1
                            
                            # Add to recent errors list (limit to 20)
                            if len(stats["recent_errors"]) < 20:
                                stats["recent_errors"].append({
                                    "timestamp": entry.get("timestamp", "unknown"),
                                    "endpoint": endpoint,
                                    "method": method,
                                    "error": error_msg[:100] + ("..." if len(error_msg) > 100 else "")
                                })
                        
                        # Update by endpoint
                        stats["calls_by_endpoint"][endpoint]["total"] += 1
                        if status == "success":
                            stats["calls_by_endpoint"][endpoint]["success"] += 1
                        elif status == "error":
                            stats["calls_by_endpoint"][endpoint]["error"] += 1
                        
                        # Update by method
                        stats["calls_by_method"][method]["total"] += 1
                        if status == "success":
                            stats["calls_by_method"][method]["success"] += 1
                        elif status == "error":
                            stats["calls_by_method"][method]["error"] += 1
                    
                    except json.JSONDecodeError:
                        # Not a JSON line, try to extract info using regex
                        continue
        
        except Exception as e:
            logger.error(f"Error processing API log file: {str(e)}")
        
        return stats
    
    def generate_api_summary(self, days_back: int = 1) -> str:
        """
        Generate a text summary of API call statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text summary of API call statistics
        """
        stats = self.analyze_api_log(days_back)
        
        if not stats or stats["total_calls"] == 0:
            return "No API logs found for analysis."
        
        # Format the timestamp
        timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M Uhr")
        
        summary = [
            f"API Call Analysis (Stand: {timestamp}, last {days_back} day{'s' if days_back > 1 else ''}):",
            f"- Total API Calls: {stats['total_calls']}",
            f"- Successful Calls: {stats['success_calls']} ({stats['success_calls']/stats['total_calls']*100:.1f}%)",
            f"- Failed Calls: {stats['error_calls']} ({stats['error_calls']/stats['total_calls']*100:.1f}%)",
            "\nMost Used Endpoints:",
        ]
        
        # Add most used endpoints
        for i, (endpoint, data) in enumerate(sorted(stats["calls_by_endpoint"].items(), 
                                                  key=lambda x: x[1]["total"], reverse=True)[:5], 1):
            success_rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            summary.append(f"{i}. {endpoint}: {data['total']} calls (Success Rate: {success_rate:.1f}%)")
        
        # Add methods
        summary.append("\nAPI Methods:")
        for method, data in sorted(stats["calls_by_method"].items(), 
                                 key=lambda x: x[1]["total"], reverse=True):
            success_rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            summary.append(f"- {method}: {data['total']} calls (Success Rate: {success_rate:.1f}%)")
        
        # Add common errors if any
        if stats["error_types"]:
            summary.append("\nMost Common API Errors:")
            for i, (error, count) in enumerate(sorted(stats["error_types"].items(), 
                                                   key=lambda x: x[1], reverse=True)[:5], 1):
                summary.append(f"{i}. {error[:100]}: {count} occurrences")
        
        # Add recent errors
        if stats["recent_errors"]:
            summary.append("\nMost Recent API Errors:")
            for i, error in enumerate(stats["recent_errors"][:5], 1):
                summary.append(f"{i}. {error['method']} {error['endpoint']}: {error['error']}")
        
        return "\n".join(summary)

    def analyze_trade_log(self, days_back: int = 1) -> Dict:
        """
        Analyze the trades log file and generate statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Dictionary with trade statistics
        """
        trade_log_path = os.path.join(self.log_dir, "trades.log")
        
        if not os.path.exists(trade_log_path):
            logger.warning(f"Trade log file not found: {trade_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "volume_total": 0.0,
            "trades_by_symbol": defaultdict(lambda: {"buy": 0, "sell": 0, "total": 0, "volume": 0.0}),
            "trades_by_strategy": defaultdict(lambda: {"buy": 0, "sell": 0, "total": 0, "volume": 0.0}),
            "trades_by_day": defaultdict(int),
            "daily_volumes": defaultdict(float),
            "recent_trades": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            with open(trade_log_path, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON
                        trade = json.loads(line.strip())
                        
                        # Skip if older than cutoff date
                        if "timestamp" in trade:
                            try:
                                trade_date = datetime.fromisoformat(trade["timestamp"])
                                if trade_date < cutoff_date:
                                    continue
                                
                                # Format date for daily stats
                                date_str = trade_date.strftime("%Y-%m-%d")
                                stats["trades_by_day"][date_str] += 1
                                
                            except (ValueError, TypeError):
                                # If we can't parse the date, include it anyway
                                pass
                        
                        # Extract basic info
                        symbol = trade.get("symbol", "unknown")
                        side = trade.get("side", "unknown").lower()
                        quantity = float(trade.get("quantity", 0))
                        price = float(trade.get("price", 0))
                        strategy = trade.get("strategy", "unknown")
                        
                        # Calculate trade value/volume
                        trade_value = quantity * price
                        
                        # Update total counts
                        stats["total_trades"] += 1
                        stats["volume_total"] += trade_value
                        
                        if side == "buy":
                            stats["buy_trades"] += 1
                        elif side == "sell":
                            stats["sell_trades"] += 1
                        
                        # Update daily volumes
                        if "timestamp" in trade:
                            try:
                                trade_date = datetime.fromisoformat(trade["timestamp"])
                                date_str = trade_date.strftime("%Y-%m-%d")
                                stats["daily_volumes"][date_str] += trade_value
                            except (ValueError, TypeError):
                                pass
                        
                        # Update by symbol
                        stats["trades_by_symbol"][symbol]["total"] += 1
                        stats["trades_by_symbol"][symbol]["volume"] += trade_value
                        if side == "buy":
                            stats["trades_by_symbol"][symbol]["buy"] += 1
                        elif side == "sell":
                            stats["trades_by_symbol"][symbol]["sell"] += 1
                        
                        # Update by strategy
                        stats["trades_by_strategy"][strategy]["total"] += 1
                        stats["trades_by_strategy"][strategy]["volume"] += trade_value
                        if side == "buy":
                            stats["trades_by_strategy"][strategy]["buy"] += 1
                        elif side == "sell":
                            stats["trades_by_strategy"][strategy]["sell"] += 1
                        
                        # Add to recent trades list (limit to 20)
                        if len(stats["recent_trades"]) < 20:
                            stats["recent_trades"].append({
                                "timestamp": trade.get("timestamp", "unknown"),
                                "symbol": symbol,
                                "side": side,
                                "quantity": quantity,
                                "price": price,
                                "value": trade_value,
                                "strategy": strategy
                            })
                    
                    except json.JSONDecodeError:
                        # Not a JSON line, skip
                        continue
        
        except Exception as e:
            logger.error(f"Error processing trade log file: {str(e)}")
        
        return stats
    
    def generate_trade_summary(self, days_back: int = 1) -> str:
        """
        Generate a text summary of trade statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text summary of trade statistics
        """
        stats = self.analyze_trade_log(days_back)
        
        if not stats or stats["total_trades"] == 0:
            return "No trade logs found for analysis."
        
        # Format the timestamp
        timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M Uhr")
        
        summary = [
            f"Trade Analysis (Stand: {timestamp}, last {days_back} day{'s' if days_back > 1 else ''}):",
            f"- Total Trades: {stats['total_trades']}",
            f"- Buy Trades: {stats['buy_trades']} ({stats['buy_trades']/stats['total_trades']*100:.1f}%)",
            f"- Sell Trades: {stats['sell_trades']} ({stats['sell_trades']/stats['total_trades']*100:.1f}%)",
            f"- Total Volume: ${stats['volume_total']:.2f}",
            "\nMost Active Symbols:",
        ]
        
        # Add most active symbols by trade count
        for i, (symbol, data) in enumerate(sorted(stats["trades_by_symbol"].items(), 
                                                key=lambda x: x[1]["total"], reverse=True)[:5], 1):
            summary.append(f"{i}. {symbol}: {data['total']} trades (Volume: ${data['volume']:.2f})")
        
        # Add most active strategies
        if len(stats["trades_by_strategy"]) > 1:  # Only if we have multiple strategies
            summary.append("\nMost Active Strategies:")
            for i, (strategy, data) in enumerate(sorted(stats["trades_by_strategy"].items(), 
                                                    key=lambda x: x[1]["total"], reverse=True)[:5], 1):
                summary.append(f"{i}. {strategy}: {data['total']} trades (Volume: ${data['volume']:.2f})")
        
        # Add daily trade summary
        if stats["trades_by_day"]:
            summary.append("\nDaily Trade Count:")
            for date, count in sorted(stats["trades_by_day"].items(), key=lambda x: x[0], reverse=True):
                volume = stats["daily_volumes"].get(date, 0)
                summary.append(f"- {date}: {count} trades (Volume: ${volume:.2f})")
        
        # Add recent trades
        if stats["recent_trades"]:
            summary.append("\nMost Recent Trades:")
            for i, trade in enumerate(sorted(stats["recent_trades"], 
                                          key=lambda x: x.get("timestamp", ""), reverse=True)[:5], 1):
                summary.append(f"{i}. {trade['symbol']} {trade['side'].upper()} {trade['quantity']} @ ${trade['price']:.6f} (${trade['value']:.2f})")
        
        return "\n".join(summary)

    def analyze_performance_log(self, days_back: int = 30) -> Dict:
        """
        Analyze the performance log file and generate statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Dictionary with performance statistics
        """
        performance_log_path = os.path.join(self.log_dir, "performance.log")
        
        if not os.path.exists(performance_log_path):
            logger.warning(f"Performance log file not found: {performance_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "current_balance": None,
            "starting_balance": None,
            "daily_pnl": {},
            "cumulative_pnl": {},
            "win_rate": 0.0,
            "performance_by_symbol": defaultdict(lambda: {"trades": 0, "pnl": 0.0, "win_count": 0, "loss_count": 0}),
            "performance_by_strategy": defaultdict(lambda: {"trades": 0, "pnl": 0.0, "win_count": 0, "loss_count": 0}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Store data for timeline
        timestamps = []
        balances = []
        
        try:
            with open(performance_log_path, 'r') as f:
                for line in f:
                    try:
                        # Try to parse as JSON
                        entry = json.loads(line.strip())
                        
                        # Skip if older than cutoff date
                        if "timestamp_iso" in entry:
                            try:
                                entry_date = datetime.fromisoformat(entry["timestamp_iso"])
                                if entry_date < cutoff_date:
                                    # If this is the earliest entry we've seen, save as starting balance
                                    if stats["starting_balance"] is None:
                                        stats["starting_balance"] = entry.get("balance", 0)
                                    continue
                                
                                # Format date for daily stats
                                date_str = entry_date.strftime("%Y-%m-%d")
                                
                                # Save timestamp and balance for timeline
                                timestamps.append(entry_date)
                                balances.append(entry.get("balance", 0))
                                
                            except (ValueError, TypeError):
                                # If we can't parse the date, include it anyway
                                pass
                        
                        # Extract basic info
                        balance = entry.get("balance", 0)
                        daily_pnl = entry.get("daily_pnl", 0)
                        win_rate = entry.get("win_rate", 0)
                        symbol_performance = entry.get("symbol_performance", {})
                        strategy_performance = entry.get("strategy_performance", {})
                        
                        # Update current balance
                        stats["current_balance"] = balance
                        
                        # If we don't have a starting balance yet, set it
                        if stats["starting_balance"] is None:
                            stats["starting_balance"] = balance
                        
                        # Update daily PnL if we have date info
                        if "timestamp_iso" in entry:
                            try:
                                entry_date = datetime.fromisoformat(entry["timestamp_iso"])
                                date_str = entry_date.strftime("%Y-%m-%d")
                                stats["daily_pnl"][date_str] = daily_pnl
                            except (ValueError, TypeError):
                                pass
                        
                        # Update win rate
                        stats["win_rate"] = win_rate
                        
                        # Update symbol performance
                        for symbol, perf in symbol_performance.items():
                            stats["performance_by_symbol"][symbol]["trades"] += perf.get("trades", 0)
                            stats["performance_by_symbol"][symbol]["pnl"] += perf.get("pnl", 0)
                            stats["performance_by_symbol"][symbol]["win_count"] += perf.get("win_count", 0)
                            stats["performance_by_symbol"][symbol]["loss_count"] += perf.get("loss_count", 0)
                        
                        # Update strategy performance
                        for strategy, perf in strategy_performance.items():
                            stats["performance_by_strategy"][strategy]["trades"] += perf.get("trades", 0)
                            stats["performance_by_strategy"][strategy]["pnl"] += perf.get("pnl", 0)
                            stats["performance_by_strategy"][strategy]["win_count"] += perf.get("win_count", 0)
                            stats["performance_by_strategy"][strategy]["loss_count"] += perf.get("loss_count", 0)
                    
                    except json.JSONDecodeError:
                        # Not a JSON line, skip
                        continue
        
        except Exception as e:
            logger.error(f"Error processing performance log file: {str(e)}")
        
        # Calculate cumulative PnL
        cumulative = 0
        for date in sorted(stats["daily_pnl"].keys()):
            cumulative += stats["daily_pnl"][date]
            stats["cumulative_pnl"][date] = cumulative
        
        # Create timeline data if we have timestamps
        if timestamps and balances:
            # Convert to pandas Series for easier manipulation
            timeline_df = pd.DataFrame({
                'timestamp': timestamps,
                'balance': balances
            })
            
            # Sort by timestamp
            timeline_df = timeline_df.sort_values('timestamp')
            
            # Create dictionary format for easier JSON serialization
            stats['balance_timeline'] = {
                'timestamps': [ts.isoformat() for ts in timeline_df['timestamp']],
                'balances': timeline_df['balance'].tolist()
            }
        
        return stats
    
    def generate_performance_summary(self, days_back: int = 30) -> str:
        """
        Generate a text summary of performance statistics
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text summary of performance statistics
        """
        stats = self.analyze_performance_log(days_back)
        
        if not stats or stats["current_balance"] is None:
            return "No performance logs found for analysis."
        
        # Format the timestamp
        timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M Uhr")
        
        # Calculate overall profit/loss
        starting_balance = stats["starting_balance"] or 0
        current_balance = stats["current_balance"] or 0
        pnl_amount = current_balance - starting_balance
        pnl_percent = (pnl_amount / starting_balance * 100) if starting_balance > 0 else 0
        
        summary = [
            f"Performance Analysis (Stand: {timestamp}, last {days_back} day{'s' if days_back > 1 else ''}):",
            f"- Current Balance: ${current_balance:.2f}",
            f"- Starting Balance: ${starting_balance:.2f}",
            f"- Overall P&L: ${pnl_amount:.2f} ({pnl_percent:+.2f}%)",
            f"- Overall Win Rate: {stats['win_rate']:.1f}%",
        ]
        
        # Add best performing symbols
        if stats["performance_by_symbol"]:
            summary.append("\nBest Performing Symbols:")
            top_symbols = sorted(stats["performance_by_symbol"].items(), key=lambda x: x[1]["pnl"], reverse=True)[:5]
            for i, (symbol, perf) in enumerate(top_symbols, 1):
                win_rate = perf["win_count"] / (perf["win_count"] + perf["loss_count"]) * 100 if (perf["win_count"] + perf["loss_count"]) > 0 else 0
                summary.append(f"{i}. {symbol}: ${perf['pnl']:.2f} ({perf['trades']} trades, {win_rate:.1f}% win rate)")
        
        # Add worst performing symbols
        if stats["performance_by_symbol"]:
            summary.append("\nWorst Performing Symbols:")
            bottom_symbols = sorted(stats["performance_by_symbol"].items(), key=lambda x: x[1]["pnl"])[:5]
            for i, (symbol, perf) in enumerate(bottom_symbols, 1):
                win_rate = perf["win_count"] / (perf["win_count"] + perf["loss_count"]) * 100 if (perf["win_count"] + perf["loss_count"]) > 0 else 0
                summary.append(f"{i}. {symbol}: ${perf['pnl']:.2f} ({perf['trades']} trades, {win_rate:.1f}% win rate)")
        
        # Add strategy performance
        if len(stats["performance_by_strategy"]) > 1:  # Only if we have multiple strategies
            summary.append("\nStrategy Performance:")
            for strategy, perf in sorted(stats["performance_by_strategy"].items(), key=lambda x: x[1]["pnl"], reverse=True):
                win_rate = perf["win_count"] / (perf["win_count"] + perf["loss_count"]) * 100 if (perf["win_count"] + perf["loss_count"]) > 0 else 0
                summary.append(f"- {strategy}: ${perf['pnl']:.2f} ({perf['trades']} trades, {win_rate:.1f}% win rate)")
        
        # Add daily PnL for recent days
        if stats["daily_pnl"]:
            summary.append("\nRecent Daily P&L:")
            recent_days = sorted(stats["daily_pnl"].items(), key=lambda x: x[0], reverse=True)[:7]  # Last 7 days
            for date, pnl in recent_days:
                summary.append(f"- {date}: ${pnl:+.2f}")
        
        return "\n".join(summary)
    
    def generate_comprehensive_report(self, days_back: int = 7) -> str:
        """
        Generate a comprehensive report of all log types
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text report with information from all log types
        """
        try:
            # Get the various reports
            error_stats = self.analyze_error_log(days_back)
            strategy_stats = self.analyze_strategy_log(days_back)
            api_stats = self.analyze_api_log(days_back)
            trade_stats = self.analyze_trade_log(days_back)
            performance_stats = self.analyze_performance_log(days_back)
            candle_stats = self.analyze_candle_warnings(days_back)
            
            # Format the timestamp
            timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M")
            
            # Create the report sections
            sections = [
                f"=== COMPREHENSIVE LOG ANALYSIS REPORT ===",
                f"Generated: {timestamp}",
                f"Analysis Period: Last {days_back} days",
                "\n"
            ]
            
            # Add error summary
            sections.append("=== ERROR SUMMARY ===")
            if error_stats:
                sections.append(f"Total Errors: {error_stats['total_errors']}")
                sections.append(f"Critical Errors: {error_stats['total_critical']}")
                
                if error_stats["error_types"]:
                    sections.append("\nTop Error Types:")
                    for error_type, count in sorted(error_stats["error_types"].items(), key=lambda x: x[1], reverse=True)[:5]:
                        sections.append(f"  {error_type}: {count}")
            else:
                sections.append("No error logs found for analysis.")
                
            # Add API summary
            sections.append("\n=== API SUMMARY ===")
            if api_stats:
                total_calls = api_stats.get('total_calls', 0)
                failed_calls = api_stats.get('failed_calls', api_stats.get('error_calls', 0))
                failure_rate = 0 if total_calls == 0 else (failed_calls / total_calls * 100)
                
                sections.append(f"Total API Calls: {total_calls}")
                sections.append(f"Failed API Calls: {failed_calls} ({failure_rate:.1f}%)")
                
                if 'endpoints' in api_stats:
                    sections.append("\nTop API Endpoints:")
                    for endpoint, count in sorted(api_stats["endpoints"].items(), key=lambda x: x[1], reverse=True)[:5]:
                        sections.append(f"  {endpoint}: {count}")
                elif 'error_types' in api_stats:
                    sections.append("\nTop API Errors:")
                    for error, count in sorted(api_stats["error_types"].items(), key=lambda x: x[1], reverse=True)[:5]:
                        sections.append(f"  {error}: {count}")
            else:
                sections.append("No API logs found for analysis.")
                
            # Add strategy summary
            sections.append("\n=== STRATEGY SUMMARY ===")
            if strategy_stats:
                sections.append(f"Total Strategy Signals: {strategy_stats.get('total_signals', 0)}")
                
                if 'strategy_types' in strategy_stats:
                    sections.append("\nStrategy Signal Distribution:")
                    for strategy, count in sorted(strategy_stats["strategy_types"].items(), key=lambda x: x[1], reverse=True):
                        sections.append(f"  {strategy}: {count}")
                elif 'signals_by_strategy' in strategy_stats:
                    sections.append("\nStrategy Signal Distribution:")
                    for strategy, data in sorted(strategy_stats["signals_by_strategy"].items(), 
                                               key=lambda x: (x[1].get('total', 0) if isinstance(x[1], dict) else x[1]), 
                                               reverse=True):
                        count = data.get('total', 0) if isinstance(data, dict) else data
                        sections.append(f"  {strategy}: {count} signals")
                        
                if 'signal_distribution' in strategy_stats:
                    sections.append("\nSignal Distribution:")
                    for signal, count in sorted(strategy_stats["signal_distribution"].items(), key=lambda x: x[1], reverse=True):
                        sections.append(f"  {signal}: {count}")
                elif 'signals_by_symbol' in strategy_stats:
                    sections.append("\nSignals by Symbol:")
                    for symbol, data in sorted(strategy_stats["signals_by_symbol"].items(), 
                                            key=lambda x: (x[1].get('total', 0) if isinstance(x[1], dict) else x[1]), 
                                            reverse=True)[:5]:
                        count = data.get('total', 0) if isinstance(data, dict) else data
                        sections.append(f"  {symbol}: {count} signals")
            else:
                sections.append("No strategy logs found for analysis.")
                
            # Add trade summary
            sections.append("\n=== TRADE SUMMARY ===")
            if trade_stats:
                total_trades = trade_stats.get('total_trades', 0)
                winning_trades = trade_stats.get('winning_trades', trade_stats.get('buy_trades', 0))
                win_rate = 0 if total_trades == 0 else (winning_trades / total_trades * 100)
                total_pnl = trade_stats.get('total_pnl', trade_stats.get('volume_total', 0))
                
                sections.append(f"Total Trades: {total_trades}")
                sections.append(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
                sections.append(f"Total P&L: {total_pnl:.2f}")
                
                if 'trades_by_symbol' in trade_stats:
                    sections.append("\nTop Traded Symbols:")
                    symbol_data = []
                    for symbol, data in trade_stats["trades_by_symbol"].items():
                        if isinstance(data, dict) and 'total' in data:
                            symbol_data.append((symbol, data['total']))
                        else:
                            symbol_data.append((symbol, data))
                    
                    for symbol, count in sorted(symbol_data, key=lambda x: x[1], reverse=True)[:5]:
                        sections.append(f"  {symbol}: {count}")
            else:
                sections.append("No trade logs found for analysis.")
                
            # Add performance summary
            sections.append("\n=== PERFORMANCE SUMMARY ===")
            if performance_stats:
                initial_balance = performance_stats.get('initial_balance', performance_stats.get('starting_balance', 0))
                current_balance = performance_stats.get('current_balance', 0)
                
                # Calculate total return based on available data
                total_return = 0
                if 'total_return' in performance_stats:
                    total_return = performance_stats['total_return']
                elif 'cumulative_pnl' in performance_stats and performance_stats['cumulative_pnl']:
                    # Get the most recent cumulative PNL if available
                    try:
                        if performance_stats['cumulative_pnl'] and isinstance(performance_stats['cumulative_pnl'], dict) and len(performance_stats['cumulative_pnl']) > 0:
                            total_return = performance_stats['cumulative_pnl'].get(
                                max(performance_stats['cumulative_pnl'].keys()), 0
                            )
                    except (ValueError, TypeError, AttributeError, KeyError) as e:
                        # If there's an error, use 0 and log the issue
                        logger.debug(f"Could not calculate total return from cumulative_pnl: {str(e)}")
                        total_return = 0
                
                # Calculate return percentage
                total_return_percent = 0
                if initial_balance > 0:
                    total_return_percent = (total_return / initial_balance) * 100
                
                max_drawdown = performance_stats.get('max_drawdown', 0)
                max_drawdown_percent = performance_stats.get('max_drawdown_percent', 0)
                
                sections.append(f"Initial Balance: {initial_balance:.2f}")
                sections.append(f"Current Balance: {current_balance:.2f}")
                sections.append(f"Total Return: {total_return:.2f} ({total_return_percent:.1f}%)")
                sections.append(f"Max Drawdown: {max_drawdown:.2f} ({max_drawdown_percent:.1f}%)")
            else:
                sections.append("No performance logs found for analysis.")
                
            # Add candle warnings summary
            sections.append("\n=== CANDLE DATA WARNINGS ===")
            if candle_stats and candle_stats.get("total_warnings", 0) > 0:
                sections.append(f"Total Warnings: {candle_stats['total_warnings']}")
                
                # Top symbols with issues
                if candle_stats["warnings_by_symbol"]:
                    sections.append("\nTop Symbols with Missing Data:")
                    for symbol, count in sorted(candle_stats["warnings_by_symbol"].items(), key=lambda x: x[1], reverse=True)[:5]:
                        sections.append(f"  {symbol}: {count}")
                
                # Top timeframes with issues
                if candle_stats["warnings_by_timeframe"]:
                    sections.append("\nTimeframes with Missing Data:")
                    for timeframe, count in sorted(candle_stats["warnings_by_timeframe"].items(), key=lambda x: x[1], reverse=True)[:3]:
                        sections.append(f"  {timeframe}: {count}")
                
                # Add recommendations
                if candle_stats.get("recommendations"):
                    sections.append("\nRecommendations:")
                    for rec in candle_stats["recommendations"][:3]:  # Top 3 recommendations
                        sections.append(f"  • {rec}")
            else:
                sections.append("No candle data warnings found for analysis.")
            
            return "\n".join(sections)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def generate_performance_chart(self, days_back: int = 30) -> Optional[str]:
        """
        Generate a base64-encoded chart image of performance data
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Base64-encoded PNG image or None if no data
        """
        try:
            # Get performance data
            stats = self.analyze_performance_log(days_back)
            
            # Check if we have enough valid data to create a chart
            if not stats:
                logger.warning("No performance data available for charting")
                return None
                
            if 'balance_timeline' not in stats:
                logger.warning("No balance timeline data available for charting")
                return None
                
            balance_timeline = stats.get('balance_timeline', {})
            if not balance_timeline or not isinstance(balance_timeline, dict):
                logger.warning("Invalid balance timeline data format")
                return None
                
            timestamps = balance_timeline.get('timestamps', [])
            balances = balance_timeline.get('balances', [])
            
            if not timestamps or not balances or len(timestamps) != len(balances) or len(timestamps) < 2:
                logger.warning(f"Insufficient data points for charting: {len(timestamps)} timestamps, {len(balances)} balances")
                return None
            
            # Convert timestamps to datetime objects
            try:
                datetime_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting timestamps to datetime: {str(e)}")
                return None
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(datetime_timestamps, balances, 'b-', linewidth=2)
            
            # Add titles and labels
            plt.title(f'Account Balance (Last {days_back} Days)')
            plt.xlabel('Date')
            plt.ylabel('Balance ($)')
            plt.grid(True)
            
            # Format dates
            plt.gcf().autofmt_xdate()
            
            # Save to bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Convert to base64
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return None
        
    def save_comprehensive_report(self, days_back: int = 7, include_charts: bool = True) -> str:
        """
        Generate and save a comprehensive report to a file
        
        Args:
            days_back: Number of days to look back in logs
            include_charts: Whether to include charts in the report
            
        Returns:
            Path to the saved report
        """
        # Generate the report
        report_text = self.generate_comprehensive_report(days_back)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "reports"
        
        # Ensure directory exists
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # Save HTML report if charts are included
        if include_charts:
            report_path = os.path.join(report_dir, f"trading_report_{timestamp}.html")
            
            # Get performance chart
            performance_chart = self.generate_performance_chart(days_back)
            
            # Basic HTML template
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f5f5f5; padding: 10px; white-space: pre-wrap; }}
        .chart {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <h1>Trading Bot Analytics Report</h1>
    <h3>Generated on: {datetime.now().strftime("%d.%m.%Y - %H:%M")}</h3>
    
    <h2>Performance Overview</h2>
    """
            
            # Add performance chart if available
            if performance_chart:
                html_content += f"""
    <div class="chart">
        <img src="data:image/png;base64,{performance_chart}" alt="Performance Chart">
    </div>
    """
            
            # Add report text
            html_content += f"""
    <h2>Detailed Analysis</h2>
    <pre>{report_text}</pre>
</body>
</html>
            """
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write(html_content)
        else:
            # Save text report
            report_path = os.path.join(report_dir, f"trading_report_{timestamp}.txt")
            with open(report_path, 'w') as f:
                f.write(report_text)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        return report_path

    def analyze_candle_warnings(self, days_back: int = 1) -> Dict:
        """
        Analyze the system log file for warnings about missing candle data
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Dictionary with statistics on missing candle data warnings
        """
        system_log_path = os.path.join(self.log_dir, "system.log")
        
        if not os.path.exists(system_log_path):
            logger.warning(f"System log file not found: {system_log_path}")
            return {}
            
        # Collect statistics
        stats = {
            "total_warnings": 0,
            "warnings_by_symbol": defaultdict(int),
            "warnings_by_timeframe": defaultdict(int),
            "symbols_with_persistent_issues": [],
            "recent_warnings": [],
            "warnings_by_hour": defaultdict(int),  # Track warnings by hour to identify time patterns
            "timeframe_symbol_pairs": defaultdict(list),  # Track specific symbol+timeframe combinations
            "recommendations": [],  # Will store recommendations based on analysis
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Missing candle warning pattern (both English and German)
        warning_patterns = [
            r"No candle data for ([A-Z0-9]+) \((\w+)\)",  # English pattern
            r"Keine Kerzendaten für ([A-Z0-9]+) \((\w+)\)",  # German pattern with ü
            r"Keine Kerzendaten fur ([A-Z0-9]+) \((\w+)\)",  # German pattern without ü
            r"Keine Kerzendaten f.r ([A-Z0-9]+) \((\w+)\)"   # German pattern with any character for ü
        ]
        
        try:
            with open(system_log_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    # Check if this is a warning line
                    if "WARNING" not in line:
                        continue
                        
                    # Try to extract timestamp
                    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    if timestamp_match:
                        try:
                            log_date = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                            if log_date < cutoff_date:
                                continue
                            # Record hour for time pattern analysis
                            stats["warnings_by_hour"][log_date.hour] += 1
                        except ValueError:
                            # If we can't parse the date, include it anyway
                            pass
                    
                    # Check for missing candle warnings
                    for pattern in warning_patterns:
                        match = re.search(pattern, line)
                        if match:
                            symbol = match.group(1)
                            timeframe = match.group(2)
                            
                            stats["total_warnings"] += 1
                            stats["warnings_by_symbol"][symbol] += 1
                            stats["warnings_by_timeframe"][timeframe] += 1
                            
                            # Track specific symbol+timeframe combinations
                            pair_key = f"{symbol}_{timeframe}"
                            if timestamp_match:
                                time_str = timestamp_match.group(1)
                                if len(stats["timeframe_symbol_pairs"][pair_key]) < 10:  # Limit to 10 timestamps per pair
                                    stats["timeframe_symbol_pairs"][pair_key].append(time_str)
                            
                            # Add to recent warnings list (limit to 20)
                            if len(stats["recent_warnings"]) < 20:
                                warning_text = f"No candle data for {symbol} ({timeframe})"
                                timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                                stats["recent_warnings"].append({
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "message": warning_text,
                                    "timestamp": timestamp
                                })
                            break
            
            # Identify symbols with persistent issues (high warning count)
            threshold = 5  # Consider symbols with 5+ warnings as having persistent issues
            stats["symbols_with_persistent_issues"] = [
                {"symbol": symbol, "count": count}
                for symbol, count in stats["warnings_by_symbol"].items()
                if count >= threshold
            ]
            
            # Sort by count
            stats["symbols_with_persistent_issues"].sort(key=lambda x: x["count"], reverse=True)
            
            # Generate recommendations based on analysis
            self._generate_candle_recommendations(stats)
            
        except Exception as e:
            logger.error(f"Error analyzing candle warnings: {str(e)}")
            
        return stats
    
    def _generate_candle_recommendations(self, stats: Dict):
        """
        Generate recommendations based on candle data warning analysis
        
        Args:
            stats: Dictionary with warning statistics
        """
        # Check for time patterns (e.g., warnings concentrated during certain hours)
        peak_hours = sorted(stats["warnings_by_hour"].items(), key=lambda x: x[1], reverse=True)[:3]
        if peak_hours and peak_hours[0][1] > 5:  # If there's a clear peak with more than 5 warnings
            peak_hour = peak_hours[0][0]
            count = peak_hours[0][1]
            stats["recommendations"].append(
                f"Consider adjusting data fetching schedule: {count} warnings occurred during hour {peak_hour}."
            )
        
        # Check for problematic timeframes
        problem_timeframes = sorted(stats["warnings_by_timeframe"].items(), key=lambda x: x[1], reverse=True)[:2]
        if problem_timeframes and problem_timeframes[0][1] > 10:
            timeframe = problem_timeframes[0][0]
            count = problem_timeframes[0][1]
            stats["recommendations"].append(
                f"Consider alternative data sources for {timeframe} timeframe which had {count} warnings."
            )
        
        # Check for problematic symbols
        if stats["symbols_with_persistent_issues"]:
            top_symbols = stats["symbols_with_persistent_issues"][:3]
            symbol_list = ", ".join([f"{item['symbol']} ({item['count']})" for item in top_symbols])
            stats["recommendations"].append(
                f"Consider excluding or using alternative sources for these symbols: {symbol_list}"
            )
        
        # Check for specific symbol+timeframe combinations that occur frequently
        problematic_pairs = sorted([(k, len(v)) for k, v in stats["timeframe_symbol_pairs"].items()], 
                                 key=lambda x: x[1], reverse=True)[:3]
        if problematic_pairs and problematic_pairs[0][1] >= 3:
            pair_info = []
            for pair, count in problematic_pairs:
                if count >= 3:  # Only include pairs with at least 3 occurrences
                    symbol, timeframe = pair.split('_')
                    pair_info.append(f"{symbol} ({timeframe})")
            
            if pair_info:
                stats["recommendations"].append(
                    f"Most problematic symbol-timeframe combinations: {', '.join(pair_info)}"
                )
        
        # General recommendations if there are many warnings
        if stats["total_warnings"] > 20:
            stats["recommendations"].append(
                "Consider implementing a more robust data caching strategy to handle missing candle data."
            )
            
        if not stats["recommendations"]:
            stats["recommendations"].append(
                "No specific recommendations - current missing candle data warnings are within normal range."
            )
        
    def generate_candle_warnings_summary(self, days_back: int = 1) -> str:
        """
        Generate a text summary of missing candle data warnings
        
        Args:
            days_back: Number of days to look back in logs
            
        Returns:
            Text summary of candle data warnings
        """
        stats = self.analyze_candle_warnings(days_back)
        
        if not stats or stats.get("total_warnings", 0) == 0:
            return "No candle data warnings found in the specified time period."
            
        summary = [
            f"=== Missing Candle Data Analysis (Last {days_back} Days) ===",
            f"Total Warnings: {stats['total_warnings']}",
            "\nTop Symbols with Missing Data:",
        ]
        
        # Add top symbols with missing data
        for item in sorted(stats["warnings_by_symbol"].items(), key=lambda x: x[1], reverse=True)[:10]:
            summary.append(f"  {item[0]}: {item[1]} warnings")
            
        summary.append("\nWarnings by Timeframe:")
        for timeframe, count in sorted(stats["warnings_by_timeframe"].items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {timeframe}: {count} warnings")
        
        # Add time pattern information if available
        if stats["warnings_by_hour"]:
            summary.append("\nWarnings by Hour of Day:")
            peak_hours = sorted(stats["warnings_by_hour"].items(), key=lambda x: x[1], reverse=True)[:5]
            for hour, count in peak_hours:
                summary.append(f"  Hour {hour}: {count} warnings")
            
        if stats["symbols_with_persistent_issues"]:
            summary.append("\nSymbols with Persistent Issues:")
            for item in stats["symbols_with_persistent_issues"]:
                summary.append(f"  {item['symbol']}: {item['count']} warnings")
                
        if stats["recent_warnings"]:
            summary.append("\nRecent Warnings:")
            for warning in stats["recent_warnings"][:5]:  # Show only the 5 most recent
                summary.append(f"  [{warning['timestamp']}] {warning['message']}")
        
        # Add recommendations
        if stats.get("recommendations"):
            summary.append("\nRecommendations:")
            for rec in stats["recommendations"]:
                summary.append(f"  • {rec}")
                
        summary.append(f"\nAnalysis completed at: {stats['timestamp']}")
        
        return "\n".join(summary)

    def export_candle_warnings_to_csv(self, days_back: int = 7, output_file: str = None) -> str:
        """
        Export missing candle data warnings to a CSV file for further analysis
        
        Args:
            days_back: Number of days to look back in logs
            output_file: Path to output file (default: 'candle_warnings_{date}.csv')
            
        Returns:
            Path to the saved CSV file
        """
        stats = self.analyze_candle_warnings(days_back)
        
        if not stats or stats.get("total_warnings", 0) == 0:
            logger.warning("No candle data warnings found to export")
            return ""
            
        # Create output file path if not provided
        if not output_file:
            date_str = datetime.now().strftime("%Y%m%d")
            output_file = f"candle_warnings_{date_str}.csv"
            
        # Create DataFrame with warning data
        warnings_data = []
        
        # Add symbol warnings
        for symbol, count in stats["warnings_by_symbol"].items():
            warnings_data.append({
                "type": "symbol",
                "name": symbol,
                "warnings": count,
                "details": ""
            })
            
        # Add timeframe warnings
        for timeframe, count in stats["warnings_by_timeframe"].items():
            warnings_data.append({
                "type": "timeframe",
                "name": timeframe,
                "warnings": count,
                "details": ""
            })
        
        # Add hour analysis
        for hour, count in stats["warnings_by_hour"].items():
            warnings_data.append({
                "type": "hour",
                "name": str(hour),
                "warnings": count,
                "details": ""
            })
            
        # Add symbol+timeframe pairs
        for pair, timestamps in stats["timeframe_symbol_pairs"].items():
            symbol, timeframe = pair.split("_")
            warnings_data.append({
                "type": "pair",
                "name": pair,
                "warnings": len(timestamps),
                "details": ", ".join(timestamps[:5])  # Include first 5 timestamps
            })
            
        # Add recent warnings
        for warning in stats["recent_warnings"]:
            warnings_data.append({
                "type": "recent",
                "name": f"{warning['symbol']}_{warning['timeframe']}",
                "warnings": 1,
                "details": warning['timestamp']
            })
            
        # Create DataFrame and save to CSV
        try:
            df = pd.DataFrame(warnings_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Exported {len(warnings_data)} candle warning records to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting candle warnings to CSV: {str(e)}")
            return ""


# Update the analyze_logs function to support different log types
def analyze_logs(log_type: str = "error", days_back: int = 1, print_to_console: bool = True):
    """
    Quick function to analyze logs without creating an analyzer instance
    
    Args:
        log_type: Type of log to analyze (error, strategy, api, trade, performance, candle, all)
        days_back: Number of days to look back in logs
        print_to_console: Whether to print the summary to console
    
    Returns:
        Summary text
    """
    analyzer = LogAnalyzer()
    
    if log_type == "error":
        if print_to_console:
            return analyzer.print_summary(days_back)
        else:
            return analyzer.log_summary(days_back)
    elif log_type == "strategy":
        summary = analyzer.generate_strategy_summary(days_back)
    elif log_type == "api":
        summary = analyzer.generate_api_summary(days_back)
    elif log_type == "trade":
        summary = analyzer.generate_trade_summary(days_back)
    elif log_type == "performance":
        summary = analyzer.generate_performance_summary(days_back)
    elif log_type == "candle":
        summary = analyzer.generate_candle_warnings_summary(days_back)
    elif log_type == "all":
        summary = analyzer.generate_comprehensive_report(days_back)
    else:
        return f"Invalid log type: {log_type}. Must be one of: error, strategy, api, trade, performance, candle, all"
    
    if print_to_console:
        print(summary)
    else:
        system_logger = logging.getLogger()
        system_logger.info(f"LOG ANALYSIS SUMMARY ({log_type.upper()})\n" + summary)
    
    return summary


if __name__ == "__main__":
    # When run directly, analyze all logs for the last day and print to console
    analyze_logs(log_type="all", days_back=1, print_to_console=True) 