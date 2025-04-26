import logging
import sys
import os
import json
from api.bybit_api import BybitAPI
from strategy.donchian_channel import DonchianChannelStrategy
from utils.logger import setup_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('test_donchian')

def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join('config', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return default config
        return {
            "api": {
                "bybit": {
                    "testnet": True
                }
            },
            "strategy": {
                "parameters": {
                    "donchian_channel": {
                        "dc_period": 20,
                        "breakout_confirmation": 2,
                        "trailing_exit": True,
                        "atr_multiplier": 2.0,
                        "use_adx_filter": True,
                        "adx_period": 14,
                        "adx_threshold": 25,
                        "use_middle_channel": True,
                        "exit_opposite_band": False
                    }
                }
            },
            "general": {
                "use_cache": True,
                "cache_dir": "cache"
            }
        }

def main():
    """Main function to test Donchian Channel strategy"""
    # Set up logger
    setup_logger()
    
    # Load config
    config = load_config()
    
    # Initialize API client
    api = BybitAPI(config)
    
    # Initialize strategy
    strategy = DonchianChannelStrategy(api, config)
    
    # Symbols to test
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Run analysis for each symbol
    for symbol in test_symbols:
        logger.info(f"Analyzing {symbol}...")
        try:
            result = strategy.analyze(symbol, interval="15", limit=200)
            
            # Print result summary
            signal = result.get("signal", "neutral")
            price = result.get("price", 0)
            params = result.get("params", {})
            
            logger.info(f"Signal for {symbol}: {signal.upper()}")
            
            if signal != "neutral":
                entry = params.get("entry_price", 0)
                stop = params.get("stop_loss", 0)
                target = params.get("take_profit", 0)
                reason = params.get("reason", "No reason provided")
                
                logger.info(f"Entry: {entry:.2f}, Stop: {stop:.2f}, Target: {target:.2f}")
                logger.info(f"Reason: {reason}")
                
                # Calculate risk-reward ratio
                if signal == "buy":
                    risk = entry - stop
                    reward = target - entry
                else:
                    risk = stop - entry
                    reward = entry - target
                    
                if risk > 0:
                    rr_ratio = reward / risk
                    logger.info(f"Risk-Reward Ratio: 1:{rr_ratio:.2f}")
                
            logger.info("-" * 50)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 