#!/usr/bin/env python3
"""
Script to test Bybit API connectivity with current credentials
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_test")

# Import our API module
try:
    from api.bybit_api import BybitAPI
except ImportError:
    logger.error("Failed to import BybitAPI. Make sure you're running from the project root directory.")
    sys.exit(1)

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        return None

def test_api_connection(api_key, api_secret, testnet=False):
    """Test API connection with provided credentials"""
    logger.info(f"Testing Bybit API connection (testnet: {testnet})")
    
    # Initialize API
    api = BybitAPI(api_key=api_key, api_secret=api_secret, testnet=testnet)
    
    # Test connectivity
    success = api.test_connectivity()
    
    if success:
        logger.info("✅ API connection test successful!")
        
        # Try to get some basic information
        try:
            # Get server time
            server_time = api.get_server_time()
            if server_time:
                local_time = int(time.time() * 1000)
                time_diff = abs(server_time - local_time)
                logger.info(f"Server time: {datetime.fromtimestamp(server_time/1000).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"Time difference with server: {time_diff} ms")
            
            # Try listing some instruments
            response = api._make_request("GET", "/v5/market/instruments-info", {"category": "linear", "limit": 5})
            
            if response and response.get("retCode") == 0:
                instruments = response.get("result", {}).get("list", [])
                logger.info(f"Successfully fetched {len(instruments)} instruments")
                
                # Print some information about first few instruments
                for i, instrument in enumerate(instruments[:3]):
                    symbol = instrument.get("symbol", "UNKNOWN")
                    status = instrument.get("status", "UNKNOWN")
                    logger.info(f"Symbol {i+1}: {symbol} (Status: {status})")
            
            # Test WebSocket connection if credentials provided
            if api_key and api_secret:
                logger.info("Testing WebSocket connection...")
                ws_success = api.connect_ws()
                if ws_success:
                    logger.info("✅ WebSocket connection successful")
                    logger.info("Listening for a few seconds to check WebSocket activity...")
                    time.sleep(5)  # Wait a few seconds to see if we get any WebSocket messages
                else:
                    logger.error("❌ WebSocket connection failed")
        except Exception as e:
            logger.error(f"Error during additional tests: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.error("❌ API connection test failed")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Bybit API connection')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--api-key', type=str, help='Bybit API key (overrides config)')
    parser.add_argument('--api-secret', type=str, help='Bybit API secret (overrides config)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet (overrides config)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load config if available
    config = load_config(args.config)
    
    # Get API credentials from args or config
    api_key = args.api_key or (config.get("api", {}).get("api_key") if config else None)
    api_secret = args.api_secret or (config.get("api", {}).get("api_secret") if config else None)
    testnet = args.testnet or (config.get("api", {}).get("testnet") if config else False)
    
    if not api_key or not api_secret:
        logger.error("API key and secret are required. Provide them in config.json or via command line arguments.")
        sys.exit(1)
    
    # Run the test
    test_api_connection(api_key, api_secret, testnet) 