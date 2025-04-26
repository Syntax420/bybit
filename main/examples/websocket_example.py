#!/usr/bin/env python
"""
Bybit API WebSocket Example
--------------------------
This example demonstrates how to use the WebSocket functionality
of the Bybit API to receive real-time market data and account updates.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API
from api.bybit_api import BybitAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("websocket_example")

# Custom callback functions
def orderbook_callback(data):
    """Callback for orderbook data"""
    symbol = "unknown"
    if "topic" in data:
        parts = data["topic"].split(".")
        if len(parts) >= 3:
            symbol = parts[2]
            
    logger.info(f"Received orderbook for {symbol}: {len(data.get('data', {}).get('a', []))} asks, {len(data.get('data', {}).get('b', []))} bids")
    
    # Access top of book (best bid and ask)
    if "data" in data:
        if "a" in data["data"] and len(data["data"]["a"]) > 0:
            best_ask = data["data"]["a"][0]
            logger.info(f"Best ask: {best_ask[0]} - {best_ask[1]}")
        
        if "b" in data["data"] and len(data["data"]["b"]) > 0:
            best_bid = data["data"]["b"][0]
            logger.info(f"Best bid: {best_bid[0]} - {best_bid[1]}")

def ticker_callback(data):
    """Callback for ticker data"""
    symbol = "unknown"
    if "topic" in data:
        parts = data["topic"].split(".")
        if len(parts) >= 2:
            symbol = parts[1]
            
    if "data" in data:
        ticker_data = data["data"]
        logger.info(f"Ticker {symbol}: Last price: {ticker_data.get('lastPrice', 'N/A')}, 24h Change: {ticker_data.get('price24hPcnt', 'N/A')}, Volume: {ticker_data.get('volume24h', 'N/A')}")

def kline_callback(data):
    """Callback for kline/candlestick data"""
    symbol = "unknown"
    interval = "unknown"
    if "topic" in data:
        parts = data["topic"].split(".")
        if len(parts) >= 3:
            interval = parts[1]
            symbol = parts[2]
            
    if "data" in data:
        candle = data["data"][0] if data["data"] else {}
        logger.info(f"Kline {symbol} {interval}: Open: {candle.get('open', 'N/A')}, High: {candle.get('high', 'N/A')}, Low: {candle.get('low', 'N/A')}, Close: {candle.get('close', 'N/A')}")

def trade_callback(data):
    """Callback for trade data"""
    symbol = "unknown"
    if "topic" in data:
        parts = data["topic"].split(".")
        if len(parts) >= 2:
            symbol = parts[1]
            
    if "data" in data:
        trades = data["data"]
        for trade in trades:
            direction = "BUY" if trade.get("S") == "Buy" else "SELL"
            logger.info(f"Trade {symbol}: {direction} {trade.get('v', 'N/A')} @ {trade.get('p', 'N/A')}")

def position_callback(data):
    """Callback for position updates"""
    if "data" in data:
        positions = data["data"]
        for position in positions:
            symbol = position.get("symbol", "unknown")
            size = position.get("size", "0")
            entry_price = position.get("entryPrice", "0")
            leverage = position.get("leverage", "0")
            unrealized_pnl = position.get("unrealisedPnl", "0")
            
            if float(size) != 0:
                logger.info(f"Position {symbol}: Size: {size}, Entry: {entry_price}, Leverage: {leverage}x, PnL: {unrealized_pnl}")

def order_callback(data):
    """Callback for order updates"""
    if "data" in data:
        orders = data["data"]
        for order in orders:
            symbol = order.get("symbol", "unknown")
            order_status = order.get("orderStatus", "unknown")
            side = order.get("side", "unknown")
            price = order.get("price", "0")
            qty = order.get("qty", "0")
            
            logger.info(f"Order {symbol}: {side} {qty} @ {price} - Status: {order_status}")

def wallet_callback(data):
    """Callback for wallet updates"""
    if "data" in data:
        wallet_data = data["data"]
        for wallet in wallet_data:
            for coin in wallet.get("coin", []):
                currency = coin.get("coin", "unknown")
                equity = coin.get("equity", "0")
                available = coin.get("available", "0")
                
                logger.info(f"Wallet {currency}: Equity: {equity}, Available: {available}")

def main():
    """Main function to run the example"""
    logger.info("Starting Bybit WebSocket example")
    
    # Initialize API client (use testnet by default)
    testnet = True  # Set to False for production
    api = BybitAPI(testnet=testnet)
    
    # Initialize API and connect to WebSocket
    if not api.initialize():
        logger.error("Failed to initialize API")
        return
    
    # Subscribe to public channels for BTCUSDT
    symbol = "BTCUSDT"
    
    # Subscribe to orderbook with custom depth
    api.subscribe_orderbook(
        symbol=symbol,
        depth=50,
        callback=orderbook_callback
    )
    
    # Subscribe to ticker
    api.subscribe_ticker(
        symbol=symbol,
        callback=ticker_callback
    )
    
    # Subscribe to kline/candlestick with 1-minute interval
    api.subscribe_kline(
        symbol=symbol,
        interval="1",
        callback=kline_callback
    )
    
    # Subscribe to trades
    api.subscribe_trade(
        symbol=symbol,
        callback=trade_callback
    )
    
    # Subscribe to private channels (if credentials are provided)
    if api.api_key and api.api_secret:
        # Subscribe to position updates
        api.subscribe_position(callback=position_callback)
        
        # Subscribe to order updates
        api.subscribe_order(callback=order_callback)
        
        # Subscribe to wallet updates
        api.subscribe_wallet(callback=wallet_callback)
    else:
        logger.warning("API credentials not found, private channels will not be subscribed")
    
    # Keep the program running to receive WebSocket updates
    try:
        logger.info("WebSocket example running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Closing WebSocket connections...")
        api.unsubscribe_all()
        logger.info("Example completed")

if __name__ == "__main__":
    main() 