import os
import sys
import json
import time
import logging
import traceback
import importlib
import pandas as pd
import numpy as np
import threading

import schedule
import argparse
import random
import signal
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

# Import custom modules
from api.bybit_api import BybitAPI
from strategy.strategy_rsi_macd import RSIMACDStrategy
from strategy.donchian_channel import DonchianChannelStrategy
from risk.risk_manager import RiskManager
from symbols.market_fetcher import MarketFetcher
from utils.logger import setup_logger, log_trade, log_performance, log_error, log_exception, log_critical_error, log_data_load, log_strategy_decision, log_api_call
from utils.data_storage import save_candles_to_csv, load_candles_from_csv
from utils.log_analyzer import LogAnalyzer, analyze_logs

# Globale Konstanten für Gebühren
MAKER_FEE_RATE = -0.01 / 100  # -0.01% (Rabatt)
TAKER_FEE_RATE = 0.06 / 100   # 0.06%

class BybitTradingBot:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the trading bot with configuration"""
        # Load configuration
        self.config_path = config_path
        self.config = self.load_config(config_path)
        
        # Set up logger
        self.logger = setup_logger(self.config)
        
        # Set log levels for specific modules to ensure detailed logging
        logging.getLogger('api.bybit_api').setLevel(logging.DEBUG)
        logging.getLogger('strategy.strategy_rsi_macd').setLevel(logging.DEBUG)
        logging.getLogger('strategy.donchian_channel').setLevel(logging.DEBUG)
        logging.getLogger('risk.risk_manager').setLevel(logging.DEBUG)
        logging.getLogger('symbols.market_fetcher').setLevel(logging.DEBUG)
        logging.getLogger('utils.data_storage').setLevel(logging.DEBUG)
        
        # Get trading mode from config
        self.paper_trading = self.config.get("general", {}).get("paper_trading", True)
        
        # Initialize API connection
        testnet = self.config.get("api", {}).get("testnet", False)
        api_key = self.config.get("api", {}).get("api_key", "")
        api_secret = self.config.get("api", {}).get("api_secret", "")
        
        # For paper trading, we can use testnet or real network data
        # but we won't execute real trades
        if self.paper_trading:
            # Force testnet to True for paper trading for safety
            testnet = True
            self.logger.info("Paper trading enabled - using testnet for market data")
        
        self.api = BybitAPI(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # Initialize modules
        self.market_fetcher = MarketFetcher(self.api, self.config)
        
        # Initialize strategy based on config
        self.strategy = self._load_strategy()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.api, self.config)
        
        # Initialize log analyzer
        self.log_analyzer = LogAnalyzer()
        
        # Trading parameters
        self.trading_interval = self.config.get("trading", {}).get("trading_interval", "15m")
        self.default_leverage = self.config.get("trading", {}).get("default_leverage", 3)
        self.max_positions = self.config.get("trading", {}).get("max_positions", 5)
        self.take_profit_percent = self.config.get("trading", {}).get("take_profit_percent", 3.0)
        self.stop_loss_percent = self.config.get("trading", {}).get("stop_loss_percent", 2.0)
        self.use_trailing_stop = self.config.get("trading", {}).get("use_trailing_stop", True)
        
        # State variables
        self.is_running = False
        self.active_symbols = []
        self.open_positions = {}
        
        # Time filtering
        self.time_filtering = self.config.get("trading", {}).get("time_filtering", False)
        self.trading_hours_start = self.config.get("trading", {}).get("trading_hours_start", 0)
        self.trading_hours_end = self.config.get("trading", {}).get("trading_hours_end", 23)
        
        # API Error Handling
        self.max_api_retries = self.config.get("api", {}).get("max_retries", 3)
        self.api_retry_delay = self.config.get("api", {}).get("retry_delay_seconds", 2)
        
        # Position tracking for trailing stops
        self.position_high_prices = {}
        self.position_low_prices = {}
        
        self.logger.info(f"Bot initialized with {self.strategy.__class__.__name__} strategy")
        self.logger.info(f"Paper trading mode: {'ON' if self.paper_trading else 'OFF'}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
            
    def start(self):
        """Start the trading bot with the configured strategy"""
        try:
            self.logger.info("Starting Bybit trading bot...")
            self.logger.info(f"Trading mode: {'Paper Trading' if self.paper_trading else 'Live Trading'}")
            
            # Log active trading symbols
            self.logger.info(f"Active trading symbols: {self.config.get('trading', {}).get('symbols', {})}")
            
            # Initialize API connection
            if not self.api:
                api_key = self.config.get("api", {}).get("api_key", "")
                api_secret = self.config.get("api", {}).get("api_secret", "")
                self.api = BybitAPI(api_key=api_key, api_secret=api_secret, testnet=self.config.get("api", {}).get("testnet", False))
            
            # Initialize API fully (including WebSocket connections)
            self.api.initialize()
            
            # Ensure time is properly synchronized with server
            # This is critical for proper API authentication
            server_time = self.api.get_server_time()
            if server_time:
                local_time = int(time.time() * 1000)
                time_diff = abs(server_time - local_time)
                self.logger.info(f"Time difference with server: {time_diff} ms")
                
                # Warn if time difference is significant
                if time_diff > 1000:  # More than 1 second
                    self.logger.warning(f"Time difference with server is significant ({time_diff} ms). "
                                      f"This can cause authentication issues. Consider using NTP for time synchronization.")
            
            # Clean up old cache files
            self.api.manage_cache(max_age_days=7, max_cache_size_mb=2000)
            
            # Initialize market data
            self.market_fetcher.initialize()
            
            # Explicitly update symbols before starting
            self.update_symbols()
            self.logger.info(f"Updated active symbols: {self.active_symbols}")
            
            # Check if we have symbols to trade
            if not self.active_symbols:
                self.logger.warning("No active symbols found for trading. Check your whitelist configuration.")
                
                # Try to use whitelist directly if available
                whitelist = self.config.get("trading", {}).get("symbols", {}).get("whitelist", [])
                if whitelist:
                    self.logger.info(f"Using whitelist symbols directly: {whitelist}")
                    self.active_symbols = whitelist
            
            # Setup scheduled tasks
            self.setup_schedules()
            
            # Start the main loop
            self.is_running = True
            self.analyze_and_trade()  # Run once immediately
            
            # Start background threads
            self._start_background_threads()
            
            # Wait for background threads to finish (if this is not a blocking call)
            self._wait_for_threads()
            
            self.logger.info("Bot operation completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {str(e)}")
            log_exception(self.logger, e, "Bot Startup", traceback.format_exc())
            self.logger.critical(f"Exception: {str(e)} in Bot Startup - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logging.getLogger('critical').critical(f"Exception: {str(e)} in Bot Startup - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\nStack trace: {traceback.format_exc()}")
            return False
            
    def create_initial_performance_log(self):
        """Create initial performance log entry to ensure file exists for analysis"""
        try:
            wallet_balance = 0.0
            
            # Get current balance from risk manager
            if self.paper_trading:
                wallet_balance = self.risk_manager.get_account_balance()
            else:
                # For live trading, get real balance from API
                balance_info = self.api.get_wallet_balance()
                if balance_info and 'list' in balance_info and balance_info['list']:
                    for coin in balance_info['list'][0].get('coin', []):
                        if coin.get('coin') == 'USDT':
                            wallet_balance = float(coin.get('equity', '0'))
                            break
            
            # Create simple initial performance record
            performance_data = {
                'balance': wallet_balance,
                'daily_pnl': 0.0,
                'win_rate': 0.0,
                'consecutive_losses': 0,
                'starting_balance': wallet_balance,
                'symbol_performance': {},
                'strategy_performance': {},
                'cumulative_pnl': {"2025-01-01": 0.0},  # Add at least one entry so we have a valid key
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'total_return': 0.0,
                'total_return_percent': 0.0,
                'initial_balance': wallet_balance
            }
            
            # Make sure we have a valid timeline
            performance_data['balance_timeline'] = {
                'timestamps': [datetime.now().isoformat(), (datetime.now() - timedelta(days=1)).isoformat()],
                'balances': [wallet_balance, wallet_balance]
            }
            
            # Log the performance data
            log_performance(self.logger, performance_data)
            self.logger.info(f"Created initial performance log with balance: {wallet_balance}")
            
        except Exception as e:
            self.logger.error(f"Error creating initial performance log: {e}")
            log_error(self.logger, "INITIAL_PERFORMANCE_LOG_ERROR", str(e))
            
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping Bybit trading bot...")
        self.is_running = False
        
        # Properly close all API connections
        try:
            if hasattr(self, 'api') and self.api is not None:
                self.logger.info("Closing API connections...")
                self.api.close()
                self.logger.info("API connections closed")
        except Exception as e:
            self.logger.error(f"Error closing API connections: {e}")
        
        # Log final state
        self.logger.info("Bot stopped successfully")
        return True

    def setup_schedules(self):
        """Setup scheduled tasks"""
        # Convert interval string to minutes for scheduling
        interval_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
        interval_minutes = interval_map.get(self.trading_interval, 15)
        
        # Schedule main analysis task
        schedule.every(interval_minutes).minutes.do(self.analyze_and_trade)
        
        # Update symbols list hourly
        schedule.every(1).hours.do(self.update_symbols)
        
        # Check open positions every minute
        schedule.every(1).minutes.do(self.check_open_positions)
        
        # Update performance metrics every 30 minutes
        schedule.every(30).minutes.do(self.update_performance)
        
        # Run log analysis every 4 hours
        schedule.every(4).hours.do(self.run_log_analysis)
        
        # Generate a comprehensive daily report at midnight
        schedule.every().day.at("00:01").do(self.generate_daily_report)
        
        self.logger.info(f"Scheduled tasks setup completed with {interval_minutes} minute trading interval")
        
    def run_log_analysis(self):
        """Run log analysis and report the results"""
        try:
            self.logger.info("Running scheduled log analysis...")
            
            try:
                # Run comprehensive analysis for the last day
                analysis_summary = self.log_analyzer.generate_comprehensive_report(days_back=1)
                self.logger.info("Log analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Error generating comprehensive report: {str(e)}")
                analysis_summary = f"Error generating report: {str(e)}"
            
            # Save report to file (HTML format with charts)
            if self.config.get("logging", {}).get("save_reports", True):
                try:
                    report_path = self.log_analyzer.save_comprehensive_report(
                        days_back=1, 
                        include_charts=True
                    )
                    self.logger.info(f"Analysis report saved to: {report_path}")
                except Exception as e:
                    self.logger.error(f"Error saving analysis report: {str(e)}")
            
            # Check for critical error patterns that might require intervention
            try:
                error_stats = self.log_analyzer.analyze_error_log(days_back=1)
                if error_stats.get("total_critical", 0) > 0:
                    self.logger.warning(f"ATTENTION: {error_stats.get('total_critical')} critical errors detected in the last 24 hours")
                    
                    # Notify administrators if critical errors are detected
                    if "critical_notification" in self.config.get("notifications", {}):
                        # This would typically call a notification service
                        self.logger.info("Critical error notification sent to administrators")
            except Exception as e:
                self.logger.error(f"Error analyzing error log: {str(e)}")
            
            # Analyze API performance to detect potential issues
            try:
                api_stats = self.log_analyzer.analyze_api_log(days_back=1)
                if api_stats.get("error_calls", 0) > 0:
                    error_rate = api_stats.get("error_calls", 0) / max(api_stats.get("total_calls", 1), 1) * 100
                    if error_rate > 10:  # Over 10% error rate
                        self.logger.warning(f"High API error rate detected: {error_rate:.1f}% ({api_stats.get('error_calls')} errors)")
            except Exception as e:
                self.logger.error(f"Error analyzing API log: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error running log analysis: {str(e)}")
            error_details = traceback.format_exc()
            log_exception(self.logger, e, context="Log Analysis", stack_trace=error_details)
            return False
        
    def update_symbols(self):
        """Update the list of tradable symbols"""
        try:
            self.logger.info("Updating tradable symbols...")
            
            # Get optimal trading symbols based on volume and activity
            self.active_symbols = self.market_fetcher.get_optimal_trading_symbols(
                max_symbols=self.max_positions
            )
            
            self.logger.info(f"Active symbols updated: {self.active_symbols}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating symbols: {e}")
            log_error(self.logger, "SYMBOL_UPDATE_ERROR", str(e))
            return False
    
    def check_margin_health(self) -> bool:
        """
        Check if account has enough margin/balance for healthy trading
        
        Returns:
            bool: True if margin is healthy, False otherwise
        """
        try:
            # Get current account balance
            if self.paper_trading:
                balance = self.risk_manager.get_account_balance()
            else:
                balance_info = self._api_call_with_retry(self.api.get_wallet_balance)
                balance = 0.0
                if balance_info and 'list' in balance_info and balance_info['list']:
                    for coin in balance_info['list'][0].get('coin', []):
                        if coin.get('coin') == 'USDT':
                            balance = float(coin.get('equity', '0'))
                            break
            
            # Get minimum required balance from config or use default
            min_balance = self.config.get("trading", {}).get("min_balance", 10.0)
            
            # Check if balance is above minimum
            if balance < min_balance:
                self.logger.warning(f"Margin health check failed: Balance {balance:.2f} USDT below minimum {min_balance:.2f} USDT")
                return False
            
            # Check if we have any active positions
            positions_data = None
            if not self.paper_trading:
                positions_data = self._api_call_with_retry(self.api.get_positions)
            
            # Check margin ratio if positions data is available
            if positions_data and hasattr(positions_data, 'get'):
                # In a real implementation, you would check margin ratios here
                # For now, we just use a simple balance check
                pass
            
            self.logger.debug(f"Margin health check passed: Balance {balance:.2f} USDT")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking margin health: {e}")
            log_error(self.logger, "MARGIN_HEALTH_ERROR", str(e))
            # On error, assume margin is healthy to avoid stopping trading
            # Alternatively, could return False to be more conservative
            return True
    
    def _is_trading_hour(self) -> bool:
        """Check if current time is within trading hours"""
        if not self.time_filtering:
            return True
            
        now = datetime.now()
        current_hour = now.hour
        
        return self.trading_hours_start <= current_hour < self.trading_hours_end
            
    def _api_call_with_retry(self, api_function, *args, **kwargs):
        """Execute API call with retry functionality"""
        # Check if we're in paper trading mode and handle wallet balance requests specially
        if self.paper_trading and api_function == self.api.get_wallet_balance:
            self.logger.info("Paper trading mode: Using mock wallet balance")
            mock_balance = self.risk_manager.get_account_balance()
            # Format the response to match the structure expected from the actual API
            return {
                'list': [
                    {
                        'coin': [
                            {
                                'coin': 'USDT',
                                'equity': str(mock_balance),
                                'walletBalance': str(mock_balance)
                            }
                        ]
                    }
                ]
            }
        
        # For non-paper trading or other API calls, use normal retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = api_function(*args, **kwargs)
                return response
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"API call attempt {attempt+1}/{max_retries} failed: {error_msg}")
                
                # If this is the last attempt, raise the exception
                if attempt == max_retries - 1:
                    self.logger.error(f"API call failed after {max_retries} attempts")
                    return None
                
                # Wait before retrying
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2

    def analyze_and_trade(self):
        """Main trading function that analyzes symbols and executes trades"""
        try:
            # Check if we're in trading hours
            if not self._is_trading_hour():
                self.logger.info("Outside trading hours, skipping trade cycle")
                return
                
            self.logger.info(f"Running analysis on {len(self.active_symbols)} symbols")
            
            # Convert interval to API format (e.g., "15m" -> "15")
            interval = self.trading_interval.replace("m", "").replace("h", "60")
            
            # Check if we can open new positions
            can_open, reason = self.risk_manager.can_open_new_position()
            
            if not can_open:
                self.logger.warning(f"Skipping trade cycle: {reason}")
                return
                
            # Check margin health
            margin_healthy = self.check_margin_health()
            if not margin_healthy:
                self.logger.warning("Margin health check failed, skipping new trades")
                return
                
            # Analyze each symbol
            for symbol in self.active_symbols:
                try:
                    # Skip if we already have a position for this symbol
                    if symbol in self.open_positions:
                        self.logger.debug(f"Skipping {symbol} - already have an open position")
                        continue
                        
                    # Get volatility metrics for adaptive position sizing
                    volatility_metrics = self.market_fetcher.calculate_volatility_metrics(symbol, interval)
                    
                    if not volatility_metrics:
                        self.logger.warning(f"Could not calculate volatility metrics for {symbol}, skipping")
                        continue
                    
                    # Skip extremely volatile assets for safety
                    if volatility_metrics.get("volatility_class") == "very_high":
                        self.logger.warning(f"Skipping {symbol} due to very high volatility")
                        continue
                    
                    # Multi-timeframe analysis for confirmation
                    self.logger.debug(f"Performing multi-timeframe analysis for {symbol}")
                    analysis_result = self.analyze_multiple_timeframes(symbol, interval)
                    
                    if "error" in analysis_result:
                        self.logger.warning(f"Error in analysis for {symbol}: {analysis_result['error']}")
                        continue
                        
                    signal = analysis_result.get("signal", "neutral")
                    data_source = analysis_result.get("data_source", "unknown")
                    
                    # Check market condition (range/breakout)
                    is_range, range_low, range_high, range_height = self.market_fetcher.detect_range(symbol, interval)
                    is_breakout, breakout_dir, breakout_strength, consolidation_range = self.market_fetcher.detect_breakout(symbol, interval)
                    
                    # Log analysis results
                    self.logger.info(f"Analysis for {symbol}: Signal={signal}, "
                                    f"RSI={analysis_result.get('rsi', 0):.2f}, "
                                    f"MACD Histogram={analysis_result.get('macd_histogram', 0):.8f}, "
                                    f"Market: {'Range' if is_range else 'Trending'}, "
                                    f"Volatility: {volatility_metrics.get('volatility_class', 'unknown')}, "
                                    f"Data Source: {data_source}")
                    
                    # Additional breakout logging
                    if is_breakout:
                        self.logger.info(f"Breakout detected for {symbol}: Direction={breakout_dir}, Strength={breakout_strength:.1f}, Range={consolidation_range:.2f}%")
                        
                    # Execute trade if valid signal is generated
                    # Adjust trading strategy based on market conditions
                    if signal in ["buy", "sell"] and can_open:
                        # For range markets, only take mean reversion trades
                        if is_range:
                            current_price = analysis_result.get("price", 0)
                            
                            # For ranges, only take buy signals near support or sell signals near resistance
                            if (signal == "buy" and current_price < range_low * 1.01) or \
                               (signal == "sell" and current_price > range_high * 0.99):
                                self.logger.info(f"Taking range trade for {symbol} - {signal} near range boundary")
                                self.execute_trade(symbol, signal, analysis_result)
                            else:
                                self.logger.info(f"Skipping {signal} signal for {symbol} - not at range boundary")
                                
                        # For breakouts, take trades in breakout direction
                        elif is_breakout and breakout_strength > 30:
                            # Only take trades aligned with breakout direction
                            if (breakout_dir == "up" and signal == "buy") or \
                               (breakout_dir == "down" and signal == "sell"):
                                self.logger.info(f"Taking breakout trade for {symbol} - {signal} with breakout")
                                self.execute_trade(symbol, signal, analysis_result)
                            else:
                                self.logger.info(f"Skipping {signal} signal for {symbol} - against breakout direction")
                                
                        # For normal trending markets
                        else:
                            self.logger.info(f"Taking trend trade for {symbol} - {signal} signal | Reason: {analysis_result.get('reason', 'N/A')}")
                            self.execute_trade(symbol, signal, analysis_result)
                    elif signal != "neutral":
                        self.logger.info(f"Signal {signal} for {symbol}, but can_open is {can_open}")
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    log_error(self.logger, "ANALYSIS_ERROR", str(e), {
                        "symbol": symbol,
                        "stacktrace": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in analyze_and_trade: {e}")
            log_error(self.logger, "TRADE_CYCLE_ERROR", str(e), {
                "stacktrace": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "active_symbols": self.active_symbols
            })
            
    def analyze_multiple_timeframes(self, symbol: str, interval: str) -> Dict:
        """
        Perform analysis across multiple timeframes for confirmation
        
        Args:
            symbol: Trading symbol
            interval: Base interval for analysis
            
        Returns:
            Combined analysis result
        """
        try:
            self.logger.debug(f"Starting multi-timeframe analysis for {symbol}")
            
            # Get timeframes to analyze
            # Use the base timeframe plus higher timeframes for trend confirmation
            timeframes = [interval]  # Base timeframe (e.g., "15")
            
            # Add higher timeframe
            higher_tf = None
            if int(interval) <= 15:
                higher_tf = "60"  # Add 1-hour timeframe
            elif int(interval) <= 60:
                higher_tf = "240"  # Add 4-hour timeframe
            else:
                higher_tf = "1440"  # Add daily timeframe
                
            timeframes.append(higher_tf)
            self.logger.debug(f"Analyzing {symbol} on timeframes: {', '.join(timeframes)}m")
                
            # Analyze each timeframe
            results = {}
            
            # First try the base timeframe
            base_result = self.strategy.analyze(symbol, interval=interval)
            
            if "error" in base_result:
                self.logger.warning(f"Error analyzing {symbol} on base timeframe ({interval}): {base_result['error']}")
                return {"signal": "neutral", "error": f"Base timeframe error: {base_result['error']}"}
                
            results[interval] = base_result
            
            # Try the higher timeframe
            higher_result = self.strategy.analyze(symbol, interval=higher_tf)
            
            if "error" in higher_result:
                self.logger.warning(f"Error analyzing {symbol} on higher timeframe ({higher_tf}): {higher_result['error']}")
                # Continue with just the base timeframe if higher timeframe fails
                self.logger.info(f"Proceeding with only base timeframe analysis for {symbol}")
                return base_result
                
            results[higher_tf] = higher_result
            
            # Start with the base timeframe result
            combined_result = base_result.copy()
            
            # Check if signals align
            base_signal = base_result.get("signal", "neutral")
            higher_tf_signal = higher_result.get("signal", "neutral")
            
            # Log analysis
            self.logger.info(f"Multi-timeframe analysis for {symbol}: " +
                           f"Base ({interval}m): {base_signal}, " +
                           f"Higher ({higher_tf}m): {higher_tf_signal}")
            
            # Add confidence level based on timeframe alignment
            confidence = 0.5  # Default medium confidence
            
            # If signals contradict, reduce confidence or neutralize
            if base_signal != "neutral" and higher_tf_signal != "neutral" and base_signal != higher_tf_signal:
                self.logger.info(f"Timeframe conflict for {symbol}: reducing confidence")
                confidence = 0.3  # Low confidence
                
                # If we want to be conservative, we could neutralize contradicting signals
                # combined_result["signal"] = "neutral"
            
            # If signals align, increase confidence
            elif base_signal == higher_tf_signal and base_signal != "neutral":
                self.logger.info(f"Timeframe alignment for {symbol}: increasing confidence")
                confidence = 1.0  # High confidence
                
            # Add higher timeframe data to result
            combined_result["higher_tf_signal"] = higher_tf_signal
            combined_result["confidence"] = confidence
            combined_result["data_sources"] = {
                "base_tf": base_result.get("data_source", "unknown"),
                "higher_tf": higher_result.get("data_source", "unknown")
            }
            
            # If higher timeframe has a reason field, include it
            if "reason" in higher_result:
                combined_result["higher_tf_reason"] = higher_result["reason"]
            
            return combined_result
            
        except Exception as e:
            error_msg = f"Error in multi-timeframe analysis for {symbol}: {e}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"Multi-timeframe analysis for {symbol}", traceback.format_exc())
            # Return neutral signal on error
            return {"signal": "neutral", "error": str(e)}
            
    def execute_trade(self, symbol: str, signal: str, analysis: Dict):
        """Execute a trade based on the analysis signal"""
        try:
            # Get strategy info
            strategy_name = self.strategy.__class__.__name__
            
            # Convert signal to Bybit API format
            side = "Buy" if signal == "buy" else "Sell"
            
            # Get current price
            price = analysis.get("price", 0.0)
            if price <= 0:
                price = self._api_call_with_retry(self.api.get_latest_price, symbol)
                
            if not price:
                self.logger.error(f"Could not get price for {symbol}")
                log_error(self.logger, "PRICE_ERROR", "Could not get latest price", {
                    "symbol": symbol,
                    "signal": signal,
                    "strategy": strategy_name
                })
                return False
                
            # Für Limit-Orders etwas bessere Preise anbieten, um sicherzustellen, dass sie gefüllt werden
            # Buy etwas unter dem Marktpreis, Sell etwas über dem Marktpreis
            limit_price_adjustment = 0.05 / 100  # 0.05% Anpassung
            if side == "Buy":
                # Für Käufe: Preis etwas unter Marktpreis (besserer Preis für Käufer)
                limit_price = price * (1 - limit_price_adjustment)
            else:
                # Für Verkäufe: Preis etwas über Marktpreis (besserer Preis für Verkäufer)
                limit_price = price * (1 + limit_price_adjustment)
                
            # Auf angemessene Dezimalstellen runden
            limit_price = round(limit_price, 6)
            
            # Calculate stop loss based on volatility (ATR)
            stop_loss = self.calculate_atr_stop_loss(symbol, side, price)
            
            # Calculate position size based on risk management
            position_size = self.calculate_position_size(symbol, price, stop_loss)
            
            # Check if position size is valid
            if position_size <= 0:
                self.logger.warning(f"Position size too small for {symbol}, skipping trade")
                return False
                
            # Calculate risk percentage for risk-reward ratio
            stop_distance = abs(price - stop_loss)
            risk_percent = (stop_distance / price) * 100
            
            # Use risk-reward ratio to calculate take profit
            reward_ratio = self.config.get("trading", {}).get("reward_ratio", 1.5)
            reward_distance = stop_distance * reward_ratio
            
            if side == "Buy":
                take_profit = price + reward_distance
            else:
                take_profit = price - reward_distance
                
            self.logger.info(f"Executing {side} order for {symbol} at {limit_price} (Limit), size: {position_size:.6f} - Strategy: {strategy_name}, Leverage: {self.default_leverage}x")
            self.logger.info(f"Risk: {risk_percent:.2f}%, R:R: 1:{reward_ratio}, SL: {stop_loss:.6f}, TP: {take_profit:.6f}")
            
            # Execute the order
            order_id = "unknown"
            fee_info = None
            
            if not self.paper_trading:
                # Set leverage first
                set_leverage_result = self._api_call_with_retry(
                    self.api.set_leverage, 
                    symbol=symbol, 
                    leverage=self.default_leverage
                )
                
                if not set_leverage_result:
                    self.logger.error(f"Failed to set leverage for {symbol}")
                    log_error(self.logger, "LEVERAGE_ERROR", "Failed to set leverage", {
                        "symbol": symbol,
                        "leverage": self.default_leverage,
                        "strategy": strategy_name
                    })
                    return False
            
                # Place the order as Limit PostOnly für Maker-Gebühren
                order_result = self._api_call_with_retry(
                    self.api.place_order,
                    symbol=symbol,
                    side=side,
                    qty=position_size,
                    price=limit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_type="Limit",
                    time_in_force="PostOnly"  # Garantiert Maker-Gebühren
                )
                
                if not order_result or not order_result.get("success"):
                    self.logger.error(f"Failed to place {side} Limit order for {symbol}: {order_result}")
                    log_error(self.logger, "ORDER_ERROR", "Failed to place order", {
                        "symbol": symbol,
                        "side": side,
                        "price": limit_price,
                        "size": position_size,
                        "strategy": strategy_name,
                        "response": str(order_result)
                    })
                    return False
                    
                self.logger.info(f"Limit order placed successfully for {symbol}: {order_result}")
                order_id = order_result.get("order_id", "unknown")
                
                # Get fee information from execution
                if order_id != "unknown":
                    # Wait a bit for order execution
                    time.sleep(2)
                    execution_info = self._api_call_with_retry(
                        self.api.get_order_history,
                        symbol=symbol,
                        order_id=order_id
                    )
                    if execution_info:
                        fee_amount = float(execution_info.get("cumExecFee", "0"))
                        fee_currency = execution_info.get("feeCurrency", "USDT")
                        fee_info = {
                            "fee": fee_amount,
                            "fee_currency": fee_currency
                        }
                        self.logger.info(f"Order fee for {symbol}: {fee_amount} {fee_currency}")
                    
                    # Überprüfen, ob die Order gefüllt wurde
                    order_status = execution_info.get("orderStatus", "")
                    if order_status != "Filled":
                        self.logger.warning(f"Order for {symbol} not yet filled. Status: {order_status}")
                        # Hier könnte man eine Logik zum Stornieren und neu Platzieren als Market-Order implementieren
            else:
                # Paper trading - simulate order
                self.logger.info(f"[PAPER] Simulated {side} Limit order for {symbol} at {limit_price}, size: {position_size} - Strategy: {strategy_name}")
                order_id = f"paper-{int(time.time())}"
                
                # Simulate trading fee - Maker-Gebühr für Limit-Orders
                fee_amount = price * position_size * MAKER_FEE_RATE  # Negative Gebühr (Rabatt)
                fee_info = {
                    "fee": fee_amount,
                    "fee_currency": "USDT"
                }
                self.logger.info(f"[PAPER] Simulated Maker fee for {symbol}: {fee_amount:.6f} USDT (rebate)")
                
            # Track the position
            position_info = {
                "symbol": symbol,
                "side": side,
                "entry_price": limit_price if not self.paper_trading else price,  # Bei Paper-Trading den echten Preis verwenden
                "quantity": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(),
                "order_id": order_id,
                "risk_percent": risk_percent,
                "reward_ratio": reward_ratio,
                "partial_tp_taken": False,
                "fee": fee_info,
                "order_type": "Limit",
                "strategy": strategy_name,
                "leverage": self.default_leverage
            }
            
            self.open_positions[symbol] = position_info
            
            # Reset tracking prices for trailing stop
            self.position_high_prices[symbol] = price
            self.position_low_prices[symbol] = price
            
            # Log the trade
            trade_data = {
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "price": limit_price if not self.paper_trading else price,
                "quantity": position_size,
                "order_type": "Limit",
                "status": "Executed",
                "leverage": self.default_leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trade_id": order_id,
                "risk_percent": risk_percent,
                "reward_ratio": reward_ratio,
                "fee": fee_info.get("fee", 0) if fee_info else 0,
                "fee_currency": fee_info.get("fee_currency", "USDT") if fee_info else "USDT",
                "strategy": strategy_name
            }
            
            log_trade(self.logger, trade_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            log_error(self.logger, "TRADE_EXECUTION_ERROR", str(e), {
                "symbol": symbol,
                "side": signal,
                "price": analysis.get("price", 0),
                "strategy": self.strategy.__class__.__name__,
                "leverage": self.default_leverage,
                "stacktrace": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    def take_partial_profit(self, symbol: str, position: Dict, current_price: float):
        """Take partial profit on a position"""
        try:
            # Get position details
            side = position["side"]
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            leverage = position.get("leverage", 1)
            
            # Calculate the size to close for partial profit
            partial_size = quantity * (self.risk_manager.partial_tp_size_1 / 100)
            remaining_size = quantity - partial_size
            
            # Determine the limit close price
            limit_price_adjustment = 0.05 / 100  # 0.05% adjustment for limit orders
            
            if side == "Buy":
                limit_close_price = current_price * (1 - limit_price_adjustment)
                close_side = "Sell"
            else:
                limit_close_price = current_price * (1 + limit_price_adjustment)
                close_side = "Buy"
                
            # Round the price appropriately
            limit_close_price = round(limit_close_price, 6)
            
            self.logger.info(f"Taking partial profit ({self.risk_manager.partial_tp_size_1}%) for {symbol}: {side} position, "
                           f"Entry: {entry_price}, Current: {current_price}, Limit Close: {limit_close_price}")
            
            close_result = None
            fee_amount = 0
            
            if not self.paper_trading:
                # Place limit order to take partial profit
                close_result = self._api_call_with_retry(
                    self.api.place_order,
                    symbol=symbol,
                    side=close_side,
                    qty=partial_size,
                    price=limit_close_price,
                    order_type="Limit",
                    time_in_force="PostOnly",
                    reduce_only=True
                )
                
                if not close_result or not close_result.get("success"):
                    self.logger.error(f"Failed to take partial profit with limit order for {symbol}: {close_result}")
                    return False
                
                # Estimate fee (will be updated later with actual fee)
                fee_rate = 0.00055  # 0.055% maker fee
                fee_amount = partial_size * current_price * fee_rate
                
                self.logger.info(f"Partial profit order placed for {symbol}: {close_result}")
            else:
                # Simulate paper trading execution
                self.logger.info(f"PAPER TRADING: Taking partial profit for {symbol} at {current_price}")
                fee_rate = 0.00055  # 0.055% maker fee
                fee_amount = partial_size * current_price * fee_rate
            
            # Calculate the partial profit
            if side == "Buy":
                pnl_percent = ((current_price - entry_price) / entry_price) * 100 * leverage
                pnl_amount = (partial_size * (current_price - entry_price))
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100 * leverage
                pnl_amount = (partial_size * (entry_price - current_price))
                
            # Nettogewinn nach Gebühren
            net_pnl = pnl_amount - fee_amount
                
            # Log the partial trade
            log_trade(self.logger, {
                "timestamp": datetime.now().timestamp(),
                "symbol": symbol,
                "side": close_side,
                "price": limit_close_price if not self.paper_trading else current_price,
                "quantity": partial_size,
                "order_type": "Limit",
                "status": "PAPER_PARTIAL" if self.paper_trading else "PARTIAL",
                "pnl": pnl_amount,
                "net_pnl": net_pnl,
                "pnl_percent": pnl_percent,
                "fee": fee_amount,
                "leverage": leverage
            })
            
            # Update the position with remaining size
            position["quantity"] = remaining_size
            position["partial_tp_taken"] = True
            
            self.logger.info(f"Partial profit taken for {symbol} ({self.risk_manager.partial_tp_size_1}%): {pnl_percent:.2f}%, Net PnL: {net_pnl:.6f} USDT after {fee_amount:.6f} USDT fees")
            return True
                
        except Exception as e:
            self.logger.error(f"Error taking partial profit for {symbol}: {e}")
            log_error(self.logger, "PARTIAL_TP_ERROR", str(e), {"symbol": symbol})
            return False
            
    def check_open_positions(self):
        """Check and update open positions, apply trailing stops and take profits"""
        try:
            if not self.open_positions:
                return
                
            self.logger.info(f"Checking {len(self.open_positions)} open positions")
            
            positions_to_remove = []
            
            for symbol, position in self.open_positions.items():
                try:
                    side = position["side"]
                    entry_price = position["entry_price"]
                    stop_loss = position["stop_loss"]
                    take_profit = position["take_profit"]
                    
                    # Get current price
                    price_response = self._api_call_with_retry(self.api.get_latest_price, symbol)
                    
                    # Extract the actual price value from the ApiResponse object
                    if hasattr(price_response, 'success') and hasattr(price_response, 'data'):
                        # This is an ApiResponse object, extract the data
                        if price_response.success:
                            current_price = float(price_response.data)
                        else:
                            self.logger.warning(f"Failed to get price for {symbol}: {price_response.error_message}")
                            continue
                    elif isinstance(price_response, (int, float)):
                        # Already a numeric value
                        current_price = float(price_response)
                    elif isinstance(price_response, dict) and 'result' in price_response:
                        # Direct API response format
                        current_price = float(price_response['result'])
                    else:
                        self.logger.warning(f"Unknown response format for {symbol} price: {type(price_response)}")
                        continue
                        
                    if not current_price or current_price <= 0:
                        self.logger.warning(f"Could not get valid current price for {symbol}, skipping position check")
                        continue
                    
                    # Calculate unrealized PnL
                    if side == "Buy":
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                        # Update highest price for trailing stop
                        if current_price > self.position_high_prices.get(symbol, 0):
                            self.position_high_prices[symbol] = current_price
                    else:  # "Sell"
                        pnl_percent = (entry_price - current_price) / entry_price * 100
                        # Update lowest price for trailing stop
                        if current_price < self.position_low_prices.get(symbol, float('inf')):
                            self.position_low_prices[symbol] = current_price
                    
                    # Log current position status
                    self.logger.info(f"Position {symbol} {side}: Entry={entry_price}, Current={current_price}, "
                                   f"PnL={pnl_percent:.2f}%, SL={stop_loss}, TP={take_profit}")
                    
                    # Check for stop loss hit
                    if (side == "Buy" and current_price <= stop_loss) or \
                       (side == "Sell" and current_price >= stop_loss):
                        self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                        self.close_position(symbol, "stop_loss", current_price, pnl_percent)
                        positions_to_remove.append(symbol)
                        continue
                    
                    # Check for take profit hit
                    if (side == "Buy" and current_price >= take_profit) or \
                       (side == "Sell" and current_price <= take_profit):
                        self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                        self.close_position(symbol, "take_profit", current_price, pnl_percent)
                        positions_to_remove.append(symbol)
                        continue
                    
                    # Apply trailing stop if enabled and in profit
                    if self.use_trailing_stop:
                        self.apply_trailing_stop(symbol, position, current_price, pnl_percent)
                    
                    # Check for partial take profit
                    if pnl_percent >= self.take_profit_percent * 0.6 and not position.get("partial_tp_taken", False):
                        self.take_partial_profit(symbol, position, current_price)
                        
                except Exception as e:
                    self.logger.error(f"Error checking position for {symbol}: {e}")
                    log_error(self.logger, "POSITION_CHECK_ERROR", str(e), {
                        "symbol": symbol,
                        "position": position,
                        "stacktrace": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Remove closed positions
            for symbol in positions_to_remove:
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error in check_open_positions: {e}")
            log_error(self.logger, "POSITION_CHECK_CYCLE_ERROR", str(e), {
                "stacktrace": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "open_positions_count": len(self.open_positions)
            })
            
    def close_position(self, symbol: str, reason: str, close_price: float, pnl_percent: float):
        """Close a position"""
        try:
            # Get the position data
            if symbol not in self.open_positions:
                self.logger.warning(f"Cannot close position for {symbol} - not found in open positions")
                return
                
            position = self.open_positions[symbol]
            side = position["side"]
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            entry_fee = position.get("fee", {}).get("fee", 0) if isinstance(position.get("fee"), dict) else 0
            
            # Ensure close_price is a float
            if hasattr(close_price, 'success') and hasattr(close_price, 'data'):
                # Handle ApiResponse object
                if close_price.success:
                    close_price = float(close_price.data)
                else:
                    self.logger.error(f"Invalid close price ApiResponse for {symbol}: {close_price.error_message}")
                    # Get a fresh price to use instead
                    price_response = self._api_call_with_retry(self.api.get_latest_price, symbol)
                    if hasattr(price_response, 'success') and price_response.success:
                        close_price = float(price_response.data)
                    else:
                        self.logger.error(f"Cannot get valid close price for {symbol}, using entry price")
                        close_price = entry_price
            
            # Double-check that close_price is now a float
            if not isinstance(close_price, (int, float)):
                self.logger.warning(f"Close price for {symbol} is still not a number: {type(close_price)}")
                close_price = float(entry_price)  # Fallback to entry price
            
            # Limit-Order-Preis berechnen für bessere Ausführung
            limit_price_adjustment = 0.05 / 100  # 0.05% Anpassung
            if side == "Buy":
                # Für Close von Buy: Verkaufspreis etwas unter Marktpreis, damit Order schneller ausgeführt wird
                limit_close_price = close_price * (1 - limit_price_adjustment)
                close_side = "Sell"
            else:
                # Für Close von Sell: Kaufpreis etwas über Marktpreis
                limit_close_price = close_price * (1 + limit_price_adjustment)
                close_side = "Buy"
            
            # Auf angemessene Dezimalstellen runden
            limit_close_price = round(limit_close_price, 6)
            
            self.logger.info(f"Closing {side} position for {symbol} - Reason: {reason}, "
                           f"Entry: {entry_price}, Exit: {limit_close_price} (Limit), PnL: {pnl_percent:.2f}%")
            
            # Close the position in the exchange
            exit_fee = 0
            fee_currency = "USDT"
            close_result = None
            
            if not self.paper_trading:
                # Place Limit order to close position
                close_result = self._api_call_with_retry(
                    self.api.place_order,
                    symbol=symbol,
                    side=close_side,
                    qty=quantity,
                    price=limit_close_price,
                    order_type="Limit",
                    time_in_force="PostOnly",  # Garantiert Maker-Gebühren
                    reduce_only=True  # Stellt sicher, dass die Order nur die Position reduziert
                )
                
                if not close_result or not close_result.get("success"):
                    self.logger.error(f"Failed to close position with Limit order for {symbol}: {close_result}")
                    self.logger.warning(f"Attempting to close position with Market order as fallback")
                    
                    # Fallback: Versuche mit Market-Order zu schließen
                    close_result = self._api_call_with_retry(
                        self.api.close_position,
                        symbol=symbol
                    )
                    
                    if not close_result or not close_result.get("success"):
                        self.logger.error(f"Failed to close position for {symbol}: {close_result}")
                        log_error(self.logger, "CLOSE_POSITION_ERROR", "Failed to close position", {
                            "symbol": symbol,
                            "side": side,
                            "reason": reason,
                            "response": str(close_result),
                            "timestamp": datetime.now().isoformat()
                        })
                        return
                    
                self.logger.info(f"Position close order placed successfully for {symbol}")
                
                # Get the exit fee if available
                if close_result and "order_id" in close_result:
                    exit_order_id = close_result["order_id"]
                    
                    # Warten, damit die Order Zeit hat, ausgeführt zu werden
                    time.sleep(3)
                    
                    exit_order_info = self._api_call_with_retry(
                        self.api.get_order_history,
                        symbol=symbol,
                        order_id=exit_order_id
                    )
                    
                    if exit_order_info:
                        exit_fee = float(exit_order_info.get("cumExecFee", "0"))
                        fee_currency = exit_order_info.get("feeCurrency", "USDT")
                        
                        # Überprüfen, ob die Order als Maker oder Taker ausgeführt wurde
                        is_maker = exit_order_info.get("isMaker", False)
                        fee_type = "Maker" if is_maker else "Taker"
                        
                        self.logger.info(f"Exit fee for {symbol}: {exit_fee} {fee_currency} ({fee_type} fee)")
                        
                        # Order-Status überprüfen
                        order_status = exit_order_info.get("orderStatus", "")
                        self.logger.info(f"Close order status for {symbol}: {order_status}")
            else:
                # Paper trading - simulate limit exit
                self.logger.info(f"[PAPER] Simulated Limit order close for {symbol} at {limit_close_price}")
                
                # Bei Paper-Trading immer Maker-Gebühren simulieren
                exit_fee = close_price * quantity * MAKER_FEE_RATE  # Negative Gebühr (Rabatt)
                self.logger.info(f"[PAPER] Simulated Maker exit fee for {symbol}: {exit_fee:.6f} USDT (rebate)")
            
            # Calculate final PnL for the position
            pnl = 0.0
            if side == "Buy":
                pnl = (close_price - entry_price) * quantity
            else:
                pnl = (entry_price - close_price) * quantity
                
            # Subtract entry and exit fees from PnL
            total_fees = entry_fee + exit_fee
            net_pnl = pnl - total_fees
            
            # Log the closed trade
            trade_data = {
                "timestamp": time.time(),
                "symbol": symbol,
                "side": "Close " + side,
                "price": limit_close_price if not self.paper_trading else close_price,
                "quantity": quantity,
                "order_type": "Limit",
                "status": "Executed",
                "pnl": pnl,
                "net_pnl": net_pnl,
                "pnl_percent": pnl_percent,
                "entry_price": entry_price,
                "exit_price": close_price,
                "reason": reason,
                "trade_id": position.get("order_id", ""),
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "total_fees": total_fees,
                "fee_currency": fee_currency
            }
            
            log_trade(self.logger, trade_data)
            
            # Update risk manager with trade result
            self.risk_manager.update_trade_result({
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "exit_price": close_price,
                "quantity": quantity,
                "pnl": pnl,
                "net_pnl": net_pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": reason,
                "timestamp": datetime.now().timestamp(),
                "fees": total_fees
            })
            
            # Remove from open positions
            self.open_positions.pop(symbol, None)
            self.position_high_prices.pop(symbol, None)
            self.position_low_prices.pop(symbol, None)
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            log_error(self.logger, "CLOSE_POSITION_ERROR", str(e), {
                "symbol": symbol,
                "reason": reason,
                "close_price": close_price,
                "pnl_percent": pnl_percent,
                "stacktrace": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            
    def update_performance(self):
        """Log performance metrics, retrieve wallet balance, and update performance metrics."""
        try:
            # Log performance statistics
            self.logger.info("Updating performance statistics...")
            
            # Get account balance
            if self.paper_trading:
                # For paper trading, use the simulated balance
                wallet_balance = self.risk_manager.get_account_balance()
            else:
                # For live trading, get real balance from API
                balance_info = self.api.get_wallet_balance()
                wallet_balance = 0.0
                if balance_info and 'list' in balance_info and balance_info['list']:
                    for coin in balance_info['list'][0].get('coin', []):
                        if coin.get('coin') == 'USDT':
                            wallet_balance = float(coin.get('equity', '0'))
                            # Update risk manager's account balance with real balance
                            self.risk_manager.set_account_balance(wallet_balance)
                            break
                
            # Update performance metrics
            self.performance_metrics = self.risk_manager.get_performance_metrics()
            
            # Log current wallet balance
            self.logger.info(f"Current wallet balance: {wallet_balance:.2f} USDT")
            
            # Log daily P&L
            daily_pnl = self.performance_metrics.get('daily_pnl', 0.0)
            self.logger.info(f"Daily P&L: {daily_pnl:.2f} USDT ({(daily_pnl/wallet_balance)*100 if wallet_balance else 0:.2f}%)")
            
            # Log win rate
            win_rate = self.performance_metrics.get('win_rate', 0.0)
            self.logger.info(f"Win rate: {win_rate*100:.2f}%")
            
            # Log consecutive losses
            consecutive_losses = self.performance_metrics.get('consecutive_losses', 0)
            if consecutive_losses > 0:
                self.logger.warning(f"Current consecutive losses: {consecutive_losses}")
            
            # Create a comprehensive performance record
            performance_data = {
                'balance': wallet_balance,
                'daily_pnl': daily_pnl,
                'win_rate': win_rate * 100 if isinstance(win_rate, (int, float)) else 0,
                'consecutive_losses': consecutive_losses,
                'starting_balance': self.risk_manager.starting_balance,
                'symbol_performance': {},
                'strategy_performance': {}
            }
            
            # Add symbol performance if available
            trade_history = self.risk_manager.trade_history
            if trade_history:
                # Group trades by symbol
                symbol_performance = {}
                for trade in trade_history:
                    symbol = trade.get('symbol', 'unknown')
                    if symbol not in symbol_performance:
                        symbol_performance[symbol] = {
                            'trades': 0,
                            'pnl': 0.0,
                            'win_count': 0,
                            'loss_count': 0
                        }
                    
                    symbol_performance[symbol]['trades'] += 1
                    profit = trade.get('profit', 0)
                    symbol_performance[symbol]['pnl'] += profit
                    
                    if profit > 0:
                        symbol_performance[symbol]['win_count'] += 1
                    elif profit < 0:
                        symbol_performance[symbol]['loss_count'] += 1
                
                performance_data['symbol_performance'] = symbol_performance
            
            # Add max drawdown information if available
            # This would come from a more detailed analytics system
            
            # Log the performance data using the logger utility
            log_performance(self.logger, performance_data)
            
            # Return updated balance
            return wallet_balance
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            log_error(self.logger, "PERFORMANCE_ERROR", str(e))
            return 0.0
            
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate optimal position size based on account risk percentage and stop loss.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in base currency
        """
        try:
            # Get current account balance
            if self.paper_trading:
                balance = self.risk_manager.get_account_balance()
            else:
                balance_info = self._api_call_with_retry(self.api.get_wallet_balance)
                balance = 0.0
                if balance_info and 'list' in balance_info and balance_info['list']:
                    for coin in balance_info['list'][0].get('coin', []):
                        if coin.get('coin') == 'USDT':
                            balance = float(coin.get('equity', '0'))
                            break
                
            # If balance is zero or invalid, use a fallback or default value
            if balance <= 0:
                self.logger.warning("Invalid account balance. Using fallback balance of 1000 USDT")
                balance = 1000.0
                
            # Get risk percentage from config
            risk_per_trade_percent = self.config.get("trading", {}).get("risk_per_trade_percent", 1.0)
            
            # Calculate risk amount in USDT
            risk_amount = balance * (risk_per_trade_percent / 100)
            
            # Calculate stop loss distance
            stop_distance = abs(entry_price - stop_loss_price)
            
            if stop_distance <= 0 or stop_distance > entry_price * 0.1:
                # If stop distance is invalid or too large, use a reasonable default
                self.logger.warning(f"Invalid stop distance for {symbol}, using default percentage")
                stop_percent = self.config.get("trading", {}).get("default_stop_percent", 2.0) / 100
                stop_distance = entry_price * stop_percent
                
            # Calculate position size based on risk amount and stop distance
            position_size = risk_amount / stop_distance
            
            # Apply leverage if using futures
            leverage = self.config.get("trading", {}).get("leverage", 1.0)
            if leverage > 1.0:
                position_size = position_size * leverage
                
            # Apply min/max position size constraints
            min_position = self.config.get("trading", {}).get("min_position_size", 0.001)
            max_position = self.config.get("trading", {}).get("max_position_size", balance * 0.25)
            
            position_size = max(min_position, min(position_size, max_position))
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size:.6f} " +
                           f"(Risk: {risk_per_trade_percent}%, Stop distance: {stop_distance:.2f})")
                           
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            log_error(self.logger, "POSITION_SIZE_ERROR", str(e))
            return 0.0

    def calculate_atr_stop_loss(self, symbol: str, side: str, entry_price: float, multiplier: float = 2.5) -> float:
        """
        Berechnet einen Stop-Loss basierend auf dem Average True Range (ATR) Indikator.
        
        Args:
            symbol: Trading-Paar-Symbol
            side: 'buy' oder 'sell'
            entry_price: Einstiegspreis
            multiplier: ATR-Multiplikator (Standardwert: 2.5)
        
        Returns:
            Stop-Loss-Preis
        """
        try:
            # Kerzendaten für 1-Stunden-Chart holen (stabilerer ATR)
            df = self._strategy.fetch_candles(symbol, interval="60", limit=20)
            
            if df.empty:
                self.logger.warning(f"Keine Daten für die ATR-Berechnung für {symbol}, verwende Standard-SL")
                return self._calculate_default_stop_loss(side, entry_price)
                
            # Sicherstellen, dass alle erforderlichen Spalten vorhanden sind
            required_columns = ['high', 'low', 'close']
            
            # Log all available columns for debugging
            self.logger.debug(f"Available columns for {symbol}: {df.columns.tolist()}")
            
            # First check for case-insensitive matches
            for req_col in required_columns:
                # Check if column exists case-insensitive
                matches = [col for col in df.columns if col.lower() == req_col]
                if matches and req_col not in df.columns:
                    # Use the first match and create a lowercase column
                    df[req_col] = df[matches[0]]
                    self.logger.debug(f"Renamed column {matches[0]} to {req_col}")
            
            # Check for numeric column indices (common in some API responses)
            numeric_cols = [col for col in df.columns if str(col).isdigit()]
            if numeric_cols and len(numeric_cols) >= 5:
                # Typical order: timestamp(0), open(1), high(2), low(3), close(4), volume(5)
                if 'high' not in df.columns and '2' in df.columns:
                    df['high'] = df['2'].astype(float)
                    self.logger.debug("Mapped column '2' to 'high'")
                if 'low' not in df.columns and '3' in df.columns:
                    df['low'] = df['3'].astype(float)
                    self.logger.debug("Mapped column '3' to 'low'")
                if 'close' not in df.columns and '4' in df.columns:
                    df['close'] = df['4'].astype(float)
                    self.logger.debug("Mapped column '4' to 'close'")
            
            # Check for similar column names
            similar_columns = {
                'high': ['highest', 'max_price', 'upper', 'h'],
                'low': ['lowest', 'min_price', 'lower', 'l'],
                'close': ['closed', 'last', 'closing_price', 'c']
            }
            
            for req_col, alternatives in similar_columns.items():
                if req_col not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            df[req_col] = df[alt].astype(float)
                            self.logger.debug(f"Using alternative column '{alt}' for '{req_col}'")
                            break
            
            # Final check if required columns are available
            column_check = all(col in df.columns for col in required_columns)
            if not column_check:
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns for ATR calculation: {missing}. Available: {df.columns.tolist()}")
                return self._calculate_default_stop_loss(side, entry_price)
            
            # Ensure columns are numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # ATR berechnen
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            # Letzten ATR-Wert erhalten
            if df['atr'].dropna().empty:
                self.logger.warning(f"Keine gültigen ATR-Werte für {symbol}, verwende Standard-SL")
                return self._calculate_default_stop_loss(side, entry_price)
                
            atr = df['atr'].dropna().iloc[-1]
            
            if pd.isna(atr) or atr == 0:
                self.logger.warning(f"Ungültiger ATR-Wert für {symbol}, verwende Standard-SL")
                return self._calculate_default_stop_loss(side, entry_price)
                
            # Stop-Loss berechnen
            if side == 'buy':
                stop_loss = entry_price - (atr * multiplier)
            else:
                stop_loss = entry_price + (atr * multiplier)
                
            # Größenbeschränkung
            max_distance = entry_price * 0.05  # Max. 5% vom Einstiegspreis entfernt
            
            if side == 'buy' and (entry_price - stop_loss) > max_distance:
                stop_loss = entry_price - max_distance
            elif side == 'sell' and (stop_loss - entry_price) > max_distance:
                stop_loss = entry_price + max_distance
                
            self.logger.info(f"ATR-basierter Stop-Loss für {symbol} ({side}): {stop_loss:.6f} (Einstieg: {entry_price:.6f}, ATR: {atr:.6f})")
            return stop_loss
        except Exception as e:
            self.logger.error(f"Fehler bei der ATR-Stop-Loss-Berechnung für {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._calculate_default_stop_loss(side, entry_price)
        
    def _calculate_default_stop_loss(self, side: str, entry_price: float) -> float:
        """
        Berechnet einen Standard-Stop-Loss basierend auf einem festen Prozentsatz.
        
        Args:
            side: 'buy' oder 'sell'
            entry_price: Einstiegspreis
        
        Returns:
            Stop-Loss-Preis
        """
        sl_percent = self.config.get("risk", {}).get("stop_loss_percent", 2.0) / 100
        
        if side == 'buy':
            stop_loss = entry_price * (1 - sl_percent)
        else:
            stop_loss = entry_price * (1 + sl_percent)
            
        self.logger.info(f"Standard-Stop-Loss verwendet: {stop_loss:.6f} (Einstieg: {entry_price:.6f}, {sl_percent*100:.1f}%)")
        return stop_loss

    def apply_trailing_stop(self, symbol: str, position: Dict, current_price: float, pnl_percent: float):
        """
        Apply dynamic trailing stop based on ATR and profit level
        
        Args:
            symbol: Trading pair symbol
            position: Position information dictionary
            current_price: Current market price
            pnl_percent: Current profit/loss percentage
        """
        try:
            side = position["side"]
            entry_price = position["entry_price"]
            current_stop = position["stop_loss"]
            
            # Ensure current_price is a float
            if hasattr(current_price, 'success') and hasattr(current_price, 'data'):
                # Handle ApiResponse object
                if current_price.success:
                    current_price = float(current_price.data)
                else:
                    self.logger.error(f"Invalid current price for trailing stop on {symbol}: {current_price.error_message}")
                    return
            elif not isinstance(current_price, (int, float)):
                self.logger.error(f"Current price for trailing stop is not a number: {type(current_price)}")
                return
            
            # Only trail once we have some profit (defined by config)
            min_profit_to_trail = self.config.get("trading", {}).get("min_profit_to_trail", 1.0)
            
            # Check if we have enough profit to start trailing
            if side == "Buy" and current_price <= entry_price * (1 + min_profit_to_trail/100):
                return
            elif side == "Sell" and current_price >= entry_price * (1 - min_profit_to_trail/100):
                return
                
            # Get ATR value for dynamic trailing
            interval = self.trading_interval.replace("m", "").replace("h", "60")
            api_response = self._api_call_with_retry(self.api.get_kline, symbol=symbol, interval=interval, category="linear", limit=30)
            
            # Process the ApiResponse object
            if not api_response or not hasattr(api_response, 'success') or not api_response.success:
                # Fallback to simple trailing
                self.logger.warning(f"Failed to get kline data for ATR calculation, using simple trailing for {symbol}")
                self.simple_trailing_stop(symbol, position, current_price)
                return
                
            # Extract the actual candle data from the response
            candles_data = api_response.data
            
            # For API v5 format, the candles might be nested in a 'list' key
            if isinstance(candles_data, dict) and 'list' in candles_data:
                candles = candles_data.get('list', [])
            else:
                candles = candles_data
            
            if not candles or len(candles) < 14:
                # Fallback to simple trailing
                self.logger.warning(f"Insufficient candle data for ATR calculation ({len(candles) if candles else 0}), using simple trailing for {symbol}")
                self.simple_trailing_stop(symbol, position, current_price)
                return
                
            # Calculate ATR
            df = pd.DataFrame(candles)
            
            # Log available columns for debugging
            self.logger.debug(f"Candle columns for trailing stop of {symbol}: {df.columns.tolist()}")
            
            # Required columns for ATR calculation
            required_columns = ['high', 'low', 'close']
            
            # First check for case-insensitive matches
            for req_col in required_columns:
                matches = [col for col in df.columns if col.lower() == req_col]
                if matches and req_col not in df.columns:
                    df[req_col] = df[matches[0]]
                    self.logger.debug(f"Renamed trailing stop column {matches[0]} to {req_col}")
            
            # Check for numeric column indices
            numeric_cols = [col for col in df.columns if str(col).isdigit()]
            if numeric_cols and len(numeric_cols) >= 5:
                # Typical order: timestamp(0), open(1), high(2), low(3), close(4), volume(5)
                if 'high' not in df.columns and '2' in df.columns:
                    df['high'] = df['2']
                if 'low' not in df.columns and '3' in df.columns:
                    df['low'] = df['3']
                if 'close' not in df.columns and '4' in df.columns:
                    df['close'] = df['4']
            
            # Check for similar column names
            similar_columns = {
                'high': ['highest', 'max_price', 'upper', 'h'],
                'low': ['lowest', 'min_price', 'lower', 'l'],
                'close': ['closed', 'last', 'closing_price', 'c']
            }
            
            for req_col, alternatives in similar_columns.items():
                if req_col not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            df[req_col] = df[alt]
                            self.logger.debug(f"Using alternative column '{alt}' for '{req_col}' in trailing stop")
                            break
            
            # Final check if required columns are available
            column_check = all(col in df.columns for col in required_columns)
            if not column_check:
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns for trailing ATR calculation: {missing}. Available: {df.columns.tolist()}")
                self.simple_trailing_stop(symbol, position, current_price)
                return
            
            # Ensure columns are numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate True Range
            df["prev_close"] = df["close"].shift(1)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["prev_close"])
            df["tr3"] = abs(df["low"] - df["prev_close"])
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate ATR (14-period)
            df["atr"] = df["tr"].rolling(14).mean()
            
            # Check if we have valid ATR values
            if df["atr"].dropna().empty:
                self.logger.warning(f"No valid ATR values for trailing stop of {symbol}, using simple trailing")
                self.simple_trailing_stop(symbol, position, current_price)
                return
            
            # Get the current ATR
            atr = df["atr"].dropna().iloc[-1]
            
            if pd.isna(atr) or atr == 0:
                self.logger.warning(f"Invalid ATR value for trailing stop of {symbol}, using simple trailing")
                self.simple_trailing_stop(symbol, position, current_price)
                return
            
            # Adjust ATR multiple based on profit level
            # Higher profit = tighter trailing stop
            multiplier = 3.0  # Default
            
            # Adjust based on profit level
            if pnl_percent > 10.0:
                multiplier = 1.5  # Tight trailing
            elif pnl_percent > 5.0:
                multiplier = 2.0  # Medium trailing
            
            # Calculate new stop loss based on ATR
            if side == "Buy":
                new_stop = current_price - (atr * multiplier)
                
                # Only move stop up
                if new_stop > current_stop:
                    position["stop_loss"] = new_stop
                    self.logger.info(f"Updated trailing stop for {symbol}: {current_stop:.6f} -> {new_stop:.6f} " + 
                                   f"(ATR: {atr:.6f}, Multiplier: {multiplier})")
            else:  # Sell position
                new_stop = current_price + (atr * multiplier)
                
                # Only move stop down
                if new_stop < current_stop:
                    position["stop_loss"] = new_stop
                    self.logger.info(f"Updated trailing stop for {symbol}: {current_stop:.6f} -> {new_stop:.6f} " + 
                                   f"(ATR: {atr:.6f}, Multiplier: {multiplier})")
                                   
            # Also update breakeven setting if profit is significant
            self.update_breakeven_stop(symbol, position, current_price, pnl_percent)
            
        except Exception as e:
            self.logger.error(f"Error applying trailing stop for {symbol}: {e}")
            log_error(self.logger, "TRAILING_STOP_ERROR", str(e), {
                "symbol": symbol,
                "position": position,
                "current_price": current_price,
                "stacktrace": traceback.format_exc()
            })
            # Fallback to simple trailing in case of error
            try:
                self.simple_trailing_stop(symbol, position, current_price)
            except Exception as inner_e:
                self.logger.error(f"Failed to apply simple trailing stop as fallback: {inner_e}")
    
    def simple_trailing_stop(self, symbol: str, position: Dict, current_price: float):
        """
        Apply simple percentage-based trailing stop
        
        Args:
            symbol: Trading pair symbol
            position: Position information dictionary
            current_price: Current market price
        """
        try:
            side = position["side"]
            entry_price = position["entry_price"]
            current_stop = position["stop_loss"]
            
            # Percentage for trailing (adapt to market volatility)
            trail_percent = self.config.get("trading", {}).get("trail_percent", 1.0) / 100
            
            # Calculate new stop
            if side == "Buy":
                new_stop = current_price * (1 - trail_percent)
                
                # Only move stop up
                if new_stop > current_stop:
                    position["stop_loss"] = new_stop
                    self.logger.info(f"Updated simple trailing stop for {symbol}: {current_stop:.6f} -> {new_stop:.6f}")
            else:  # Sell position
                new_stop = current_price * (1 + trail_percent)
                
                # Only move stop down
                if new_stop < current_stop:
                    position["stop_loss"] = new_stop
                    self.logger.info(f"Updated simple trailing stop for {symbol}: {current_stop:.6f} -> {new_stop:.6f}")
                    
        except Exception as e:
            self.logger.error(f"Error applying simple trailing stop for {symbol}: {e}")
            
    def update_breakeven_stop(self, symbol: str, position: Dict, current_price: float, pnl_percent: float):
        """
        Move stop loss to breakeven once sufficient profit is achieved
        
        Args:
            symbol: Trading pair symbol
            position: Position information dictionary
            current_price: Current market price
            pnl_percent: Current profit/loss percentage
        """
        try:
            # Only move to breakeven if not already past it
            if position.get("breakeven_set", False):
                return
                
            side = position["side"]
            entry_price = position["entry_price"]
            current_stop = position["stop_loss"]
            
            # Config: profit percentage needed to move to breakeven
            min_profit_for_breakeven = self.config.get("trading", {}).get("min_profit_for_breakeven", 3.0)
            
            # Add small buffer for fees
            buffer = 0.1  # 0.1% buffer
            
            # Check if we have enough profit
            if pnl_percent < min_profit_for_breakeven:
                return
                
            # Calculate breakeven stop (with small buffer)
            if side == "Buy":
                breakeven_stop = entry_price * (1 + buffer/100)
                
                # Only adjust if it would move stop up
                if breakeven_stop > current_stop:
                    position["stop_loss"] = breakeven_stop
                    position["breakeven_set"] = True
                    self.logger.info(f"Set breakeven stop for {symbol}: {current_stop:.6f} -> {breakeven_stop:.6f}")
            else:  # Sell position
                breakeven_stop = entry_price * (1 - buffer/100)
                
                # Only adjust if it would move stop down
                if breakeven_stop < current_stop:
                    position["stop_loss"] = breakeven_stop
                    position["breakeven_set"] = True
                    self.logger.info(f"Set breakeven stop for {symbol}: {current_stop:.6f} -> {breakeven_stop:.6f}")
                    
        except Exception as e:
            self.logger.error(f"Error updating breakeven stop for {symbol}: {e}")

    def _load_strategy(self) -> Any:
        """
        Load the trading strategy based on configuration
        
        Returns:
            Initialized strategy instance
        """
        strategy_name = self.config.get("strategy", {}).get("active", "rsi_macd").lower()
        self.logger.info(f"Loading strategy: {strategy_name}")
        
        try:
            # Strategy mapping
            strategy_map = {
                "rsi_macd": RSIMACDStrategy,
                "donchian_channel": DonchianChannelStrategy
            }
            
            # Get the strategy class
            if strategy_name in strategy_map:
                strategy_class = strategy_map[strategy_name]
                self.logger.info(f"Using {strategy_class.__name__} strategy")
                return strategy_class(self.api, self.config)
            else:
                # Try dynamic import
                try:
                    # Convert strategy_name to CamelCase for class name
                    class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
                    if not class_name.endswith('Strategy'):
                        class_name += 'Strategy'
                    
                    # Try to import from strategy module
                    module_name = f"strategy.{strategy_name}"
                    module = importlib.import_module(module_name)
                    strategy_class = getattr(module, class_name)
                    
                    self.logger.info(f"Dynamically loaded {class_name} strategy from {module_name}")
                    return strategy_class(self.api, self.config)
                except (ImportError, AttributeError) as e:
                    self.logger.error(f"Failed to dynamically load strategy {strategy_name}: {str(e)}")
                    self.logger.warning(f"Falling back to RSI-MACD strategy")
                    return RSIMACDStrategy(self.api, self.config)
        except Exception as e:
            self.logger.error(f"Error loading strategy: {str(e)}")
            log_exception(self.logger, e, "Strategy loading", is_critical=True)
            self.logger.warning("Falling back to RSI-MACD strategy")
            return RSIMACDStrategy(self.api, self.config)

    def generate_daily_report(self):
        """Generate a comprehensive daily report"""
        try:
            self.logger.info("Generating daily report...")
            
            # Generate a report for the last 7 days
            report_path = self.log_analyzer.save_comprehensive_report(
                days_back=7, 
                include_charts=True
            )
            
            self.logger.info(f"Daily report generated and saved to: {report_path}")
            
            # This would typically involve sending the report via email or other notification
            # if "daily_report_notification" in self.config.get("notifications", {}):
            #     send_report_notification(report_path)
            
            return True
        except Exception as e:
            self.logger.error(f"Error generating daily report: {str(e)}")
            error_details = traceback.format_exc()
            log_exception(self.logger, e, context="Daily Report Generation", stack_trace=error_details)
            return False

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received shutdown signal {signum}, stopping trading bot")
        try:
            self.logger.info("Initiating graceful shutdown sequence...")
            
            # Close any open positions if configured to do so
            if self.config.get("trading", {}).get("close_positions_on_shutdown", False) and not self.paper_trading:
                self.logger.info("Closing all open positions on shutdown...")
                for symbol in list(self.open_positions.keys()):
                    try:
                        position = self.open_positions[symbol]
                        self.logger.info(f"Closing position for {symbol} due to shutdown")
                        self.close_position(symbol, "shutdown", position["entry_price"], 0.0)
                    except Exception as e:
                        self.logger.error(f"Error closing position for {symbol} during shutdown: {e}")
            
            # Set running flag to false to stop threads
            self.is_running = False
            
            # Properly close API connections
            if hasattr(self, 'api') and self.api is not None:
                self.logger.info("Closing API connections...")
                self.api.close()
            
            self.logger.info("Graceful shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
        # Force exit if needed
        # import os
        # os._exit(0)  # Uncomment if needed for forced exit

    def _start_background_threads(self):
        """Initialize and start all background threads"""
        self.logger.info("Starting background threads...")
        
        # Create scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler_loop,
            name="SchedulerThread",
            daemon=True
        )
        
        # Start scheduler thread
        self.scheduler_thread.start()
        self.logger.info("Background threads started successfully")
    
    def _wait_for_threads(self):
        """Wait for all background threads to complete"""
        if hasattr(self, 'scheduler_thread') and self.scheduler_thread.is_alive():
            self.logger.info("Waiting for scheduler thread to complete...")
            self.scheduler_thread.join()
            self.logger.info("Scheduler thread completed")
    
    def _run_scheduler_loop(self):
        """Run the scheduler in a background thread"""
        self.logger.info("Scheduler thread started")
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in scheduler thread: {str(e)}")
            log_exception(self.logger, e, "Scheduler Thread", traceback.format_exc())
        finally:
            self.logger.info("Scheduler thread exiting")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bybit Trading Bot")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--paper", action="store_true", help="Enable paper trading mode (non-interactive)")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode (non-interactive)")
    parser.add_argument("--add-blacklist", type=str, help="Add symbol to blacklist")
    parser.add_argument("--remove-blacklist", type=str, help="Remove symbol from blacklist")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Handle blacklist operations if requested
        if args.add_blacklist or args.remove_blacklist:
            # Initialize the API and market fetcher with minimal configuration
            config = {"api": {"testnet": True}}
            if os.path.exists(args.config):
                with open(args.config, 'r') as f:
                    config = json.load(f)
            
            api_key = config.get("api", {}).get("api_key", "")
            api_secret = config.get("api", {}).get("api_secret", "")
            api = BybitAPI(api_key=api_key, api_secret=api_secret, testnet=config.get("api", {}).get("testnet", True))
            market_fetcher = MarketFetcher(api, config)
            
            if args.add_blacklist:
                symbol = args.add_blacklist.upper()
                if market_fetcher.add_to_blacklist(symbol):
                    print(f"Added {symbol} to blacklist")
                else:
                    print(f"Failed to add {symbol} to blacklist")
                sys.exit(0)
            
            if args.remove_blacklist:
                symbol = args.remove_blacklist.upper()
                if market_fetcher.remove_from_blacklist(symbol):
                    print(f"Removed {symbol} from blacklist")
                else:
                    print(f"Failed to remove {symbol} from blacklist")
                sys.exit(0)
        
        # Load configuration
        config_path = args.config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"Configuration file {config_path} not found!")
            sys.exit(1)
        
        # Check if trading mode is specified via command line
        trading_mode_specified = args.paper or args.live
        
        # If no trading mode specified via command line, prompt the user
        if not trading_mode_specified:
            while True:
                print("\n===== Bybit Trading Bot =====")
                print("1. Paper Trading Mode (no real trades)")
                print("2. Live Trading Mode (real trades with real money)")
                print("3. Exit")
                choice = input("\nSelect trading mode [1-3]: ").strip()
                
                if choice == "1":
                    print("\nStarting in PAPER TRADING mode - No real trades will be executed")
                    # Set paper_trading to True in config
                    config["general"]["paper_trading"] = True
                    break
                elif choice == "2":
                    print("\n⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY ⚠️")
                    confirmation = input("Type 'CONFIRM' to proceed with live trading: ").strip()
                    if confirmation == "CONFIRM":
                        print("\nStarting in LIVE TRADING mode - REAL trades will be executed")
                        # Set paper_trading to False in config
                        config["general"]["paper_trading"] = False
                        break
                    else:
                        print("Live trading not confirmed. Please try again.")
                elif choice == "3":
                    print("Exiting program...")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please try again.")
        else:
            # Use command line argument for trading mode
            config["general"]["paper_trading"] = bool(args.paper)
            
            # Double-check to make sure we don't accidentally start live mode
            if not config["general"]["paper_trading"]:
                print("\n⚠️ WARNING: Starting in LIVE TRADING mode with REAL MONEY ⚠️")
                print("Press Ctrl+C within 5 seconds to cancel...")
                try:
                    for i in range(5, 0, -1):
                        print(f"Starting live trading in {i} seconds...", end="\r")
                        time.sleep(1)
                    print("\nLIVE TRADING mode activated                     ")
                except KeyboardInterrupt:
                    print("\nLive trading cancelled. Exiting...")
                    sys.exit(0)
        
        # Additional API configuration based on trading mode
        if config["general"]["paper_trading"]:
            # For paper trading, use testnet for data but not for execution
            config["api"]["testnet"] = True
        
        # Initialize and start the bot
        bot = BybitTradingBot(config_path=args.config)
        
        # Override config with runtime changes
        bot.paper_trading = config["general"]["paper_trading"]
        
        # Start the bot
        bot.start()
        
        # Keep the main thread alive
        try:
            while bot.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down bot...")
            bot.stop()
            print("Bot stopped. Goodbye!")
            
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        logging.error(traceback.format_exc())