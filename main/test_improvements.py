"""
Test Script for Bybit Bot Improvements

This script tests the enhancements made to the Bybit Bot:
1. Enhanced logging with timezone information
2. Log analysis capabilities
3. Dynamic strategy loading
4. Compatibility of Donchian Channel strategy
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# Add the parent directory to sys.path if running from subdirectory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from utils.logger import setup_logger, log_error, log_critical_error, log_strategy_decision
from utils.log_analyzer import LogAnalyzer
from api.bybit_api import BybitAPI
from strategy.strategy_rsi_macd import RSIMACDStrategy
from strategy.donchian_channel import DonchianChannelStrategy

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('test_improvements')

def load_config():
    """Load configuration from file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return default config
        return {
            "api": {
                "testnet": True
            },
            "strategy": {
                "active": "donchian_channel",
                "parameters": {
                    "donchian_channel": {
                        "dc_period": 20,
                        "breakout_confirmation": 2,
                        "use_adx_filter": True
                    },
                    "rsi_macd": {
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30
                    }
                }
            },
            "general": {
                "paper_trading": True
            }
        }

def test_logger_improvements():
    """Test the improved logger functionality"""
    logger.info("Testing improved logger functionality...")
    
    # Set up enhanced logger
    config = load_config()
    system_logger = setup_logger(config)
    
    # Log various types of messages
    system_logger.info("Standard info message with timezone")
    log_error(system_logger, "TEST_ERROR", "Test error message", {"test_context": True})
    log_critical_error(system_logger, "TEST_CRITICAL", "Test critical error", {"critical_context": True})
    
    # Test strategy decision logging with new parameters
    log_strategy_decision(
        system_logger,
        symbol="BTCUSDT",
        timeframe="15m",
        decision="buy",
        signals={"breakout": True, "trend": "up"},
        indicators={"rsi": 32.5, "macd": 0.002},
        reason="Test strategy signal",
        strategy_name="DonchianChannelStrategy",
        leverage=3.0
    )
    
    logger.info("Logger tests completed")
    return True

def test_log_analyzer():
    """Test the log analyzer functionality"""
    logger.info("Testing log analyzer...")
    
    # Create analyzer instance
    analyzer = LogAnalyzer()
    
    # Generate a summary
    summary = analyzer.generate_summary(days_back=1)
    
    # Print the summary
    print("\nLog Analysis Summary:")
    print(summary)
    
    logger.info("Log analyzer test completed")
    return True

def test_strategy_loading():
    """Test dynamic strategy loading and switching"""
    logger.info("Testing strategy loading...")
    
    config = load_config()
    api = BybitAPI(config)
    
    # 1. Test RSI-MACD strategy
    config["strategy"]["active"] = "rsi_macd"
    rsi_strategy = RSIMACDStrategy(api, config)
    logger.info(f"Loaded RSI-MACD strategy: {rsi_strategy.__class__.__name__}")
    
    # 2. Test Donchian Channel strategy
    config["strategy"]["active"] = "donchian_channel"
    donchian_strategy = DonchianChannelStrategy(api, config)
    logger.info(f"Loaded Donchian Channel strategy: {donchian_strategy.__class__.__name__}")
    
    logger.info("Strategy loading tests completed")
    return True

def test_donchian_compatibility():
    """Test compatibility of Donchian Channel strategy with various coins"""
    logger.info("Testing Donchian Channel strategy compatibility...")
    
    config = load_config()
    api = BybitAPI(config)
    
    # Create strategy instance
    strategy = DonchianChannelStrategy(api, config)
    
    # List of test symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"]
    
    # Test strategy on each symbol
    results = {}
    for symbol in test_symbols:
        try:
            logger.info(f"Testing Donchian Channel strategy on {symbol}...")
            result = strategy.analyze(symbol, interval="15", limit=200)
            
            # Extract and display signal
            signal = result.get("signal", "unknown")
            logger.info(f"{symbol} signal: {signal}")
            
            results[symbol] = signal
        except Exception as e:
            logger.error(f"Error testing {symbol}: {e}")
            results[symbol] = f"error: {str(e)}"
    
    # Print summary of results
    print("\nDonchian Channel Strategy Test Results:")
    for symbol, signal in results.items():
        print(f"{symbol}: {signal}")
    
    logger.info("Donchian compatibility tests completed")
    return results

def main():
    """Main test function"""
    # Set up test path
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a few test log entries to ensure we have something to analyze
    with open('logs/error.log', 'a') as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "error_type": "TestError",
            "error_msg": "Test error message for analysis",
            "context": "Test context"
        }) + '\n')
    
    try:
        print("\n=== TESTING BYBIT BOT IMPROVEMENTS ===\n")
        
        # Test logger improvements
        print("\n--- Testing Logger Improvements ---")
        test_logger_improvements()
        
        # Test log analyzer
        print("\n--- Testing Log Analyzer ---")
        test_log_analyzer()
        
        # Test strategy loading
        print("\n--- Testing Strategy Loading ---")
        test_strategy_loading()
        
        # Test Donchian compatibility
        print("\n--- Testing Donchian Strategy Compatibility ---")
        test_donchian_compatibility()
        
        print("\n=== ALL TESTS COMPLETED ===\n")
        return True
    
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    main() 