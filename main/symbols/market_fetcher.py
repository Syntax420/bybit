import logging
import os
from typing import List, Dict, Set
from api.bybit_api import BybitAPI
import json
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_storage import normalize_candle_data
import time
import signal
import math

class MarketFetcher:
    # Cache für validierte Symbole, um wiederholte API-Aufrufe zu vermeiden
    _valid_symbols_cache = {}
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialize MarketFetcher
        
        Args:
            api: BybitAPI instance
            config: Configuration dictionary
        """
        self.api = api
        self.config = config
        self.logger = logging.getLogger("market_fetcher")
        self.category = config.get("market", {}).get("category", "linear")
        self.blacklist_file = config.get("market", {}).get("blacklist_file", "blacklist.txt")
        self.blacklist = self._load_blacklist()
        
        # Get whitelist from trading.symbols config
        self.whitelist = config.get("trading", {}).get("symbols", {}).get("whitelist", [])
        if self.whitelist:
            self.logger.info(f"Loaded whitelist with {len(self.whitelist)} symbols: {self.whitelist}")
        
        self.tradable_symbols = set()
        
        # Initialisiere den Cache für dieses Objekt
        self._valid_symbols_cache = {}
        
        # Load market parameters from config
        market_config = config.get("market_fetcher", {})
        self.min_volume = market_config.get("min_volume", 5000000)  # Minimum 24h volume in USDT
        self.min_volatility = market_config.get("min_volatility", 0.5)  # Minimum volatility in %
        self.max_volatility = market_config.get("max_volatility", 15.0)  # Maximum volatility in %
        
        # Initial update of tradable symbols
        self.update_tradable_symbols()
        
    def _load_blacklist(self) -> Set[str]:
        """Load blacklisted symbols from file"""
        blacklist = set()
        try:
            if os.path.exists(self.blacklist_file):
                with open(self.blacklist_file, 'r') as f:
                    for line in f:
                        symbol = line.strip()
                        if symbol and not symbol.startswith('#'):
                            blacklist.add(symbol)
                self.logger.info(f"Loaded {len(blacklist)} symbols to blacklist")
            else:
                self.logger.info(f"No blacklist file found at {self.blacklist_file}")
        except Exception as e:
            self.logger.error(f"Error loading blacklist: {e}")
        return blacklist
        
    def _save_blacklist(self) -> bool:
        """
        Speichert die Blacklist-Datei
        
        Returns:
            True wenn erfolgreich, False sonst
        """
        try:
            with open(self.blacklist_file, 'w') as f:
                f.write("# Diese Symbole haben Probleme mit der Kerzendaten-API verursacht\n")
                for symbol in sorted(self.blacklist):
                    f.write(f"{symbol}\n")
            self.logger.info(f"Blacklist mit {len(self.blacklist)} Symbolen gespeichert")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Blacklist: {str(e)}", exc_info=True)
            return False

    def update_tradable_symbols(self) -> List[str]:
        """
        Fetch and update the list of tradable symbols from Bybit
        
        Returns:
            List of symbols available for trading
        """
        try:
            # First try to get symbols directly from the API
            self.logger.info("Fetching available symbols from API...")
            all_symbols = self.api.get_all_symbols(category=self.category)
            
            if not all_symbols:
                self.logger.warning("API returned no symbols, attempting to load from cached data")
                # Try to load from cached data
                cache_file = os.path.join('data', 'symbols_cache.json')
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            all_symbols = cached_data.get('symbols', [])
                            cache_time = cached_data.get('timestamp', 0)
                            cache_age = (datetime.now().timestamp() - cache_time) / 3600  # hours
                            self.logger.info(f"Loaded {len(all_symbols)} symbols from cache (age: {cache_age:.1f} hours)")
                    except Exception as e:
                        self.logger.error(f"Failed to load symbols from cache: {e}")
                        all_symbols = []
            else:
                # Cache the successfully retrieved symbols
                try:
                    cache_file = os.path.join('data', 'symbols_cache.json')
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'symbols': all_symbols,
                            'timestamp': datetime.now().timestamp()
                        }, f)
                    self.logger.debug(f"Cached {len(all_symbols)} symbols to {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to cache symbols: {e}")
            
            # Apply whitelist if provided
            if self.whitelist:
                filtered_symbols = [s for s in all_symbols if s in self.whitelist]
                self.logger.info(f"Applied whitelist: {len(filtered_symbols)} symbols remaining")
            else:
                filtered_symbols = all_symbols
                
            # Apply blacklist
            filtered_symbols = [s for s in filtered_symbols if s not in self.blacklist]
            self.logger.info(f"Applied blacklist: {len(filtered_symbols)} symbols remaining")
            
            # Process symbols in smaller batches to prevent overwhelming the API
            # and provide better progress tracking
            batch_size = 10  # Process 10 symbols at a time
            max_symbols_to_validate = 50  # Only validate a limited number for performance
            
            # Trim to max symbols if needed
            symbols_to_validate = filtered_symbols[:max_symbols_to_validate]
            total_batches = (len(symbols_to_validate) + batch_size - 1) // batch_size
            
            self.logger.info(f"Validating {len(symbols_to_validate)} of {len(filtered_symbols)} symbols in {total_batches} batches")
            
            # Validate symbols to ensure they have proper candle data
            valid_symbols = []
            blacklist_updated = False
            
            # Configure delays between batches to respect rate limits
            batch_delay = 3.0  # seconds between batches
            
            for batch_num, i in enumerate(range(0, len(symbols_to_validate), batch_size)):
                batch = symbols_to_validate[i:i+batch_size]
                self.logger.info(f"Processing batch {batch_num+1}/{total_batches} ({len(batch)} symbols)")
                
                for symbol in batch:
                    # Try to validate the symbol with increasing timeouts
                    for timeout_attempt in range(3):  # Try with increasing timeouts
                        timeout = 10 * (timeout_attempt + 1)  # 10, 20, or 30 seconds
                        
                        try:
                            validation_result, validation_source = self.is_symbol_valid(symbol, timeout=timeout)
                            
                            if validation_result:
                                valid_symbols.append(symbol)
                                self.logger.debug(f"Symbol {symbol} validated successfully using {validation_source}")
                                break  # Break out of timeout retry loop
                            elif timeout_attempt == 2:  # Last attempt failed
                                self.logger.warning(f"Symbol {symbol} validation failed after {timeout_attempt+1} attempts, adding to blacklist")
                                # Add to runtime blacklist
                                self.blacklist.add(symbol)
                                blacklist_updated = True
                        except Exception as e:
                            self.logger.error(f"Error during validation of {symbol}: {str(e)}")
                            if timeout_attempt == 2:  # Last attempt
                                self.logger.warning(f"Symbol {symbol} validation failed with errors, adding to blacklist")
                                self.blacklist.add(symbol)
                                blacklist_updated = True
                
                # Between batches, add a delay to avoid hitting rate limits
                if batch_num < total_batches - 1:
                    self.logger.debug(f"Sleeping for {batch_delay} seconds between batches")
                    time.sleep(batch_delay)
            
            # Save the updated blacklist if needed
            if blacklist_updated:
                self._save_blacklist()
            
            # For symbols that weren't validated in this run but were valid before,
            # keep them in the tradable set if they're in the cache
            for symbol in filtered_symbols[max_symbols_to_validate:]:
                if symbol in self._valid_symbols_cache and self._valid_symbols_cache[symbol]:
                    self.logger.debug(f"Keeping previously validated symbol {symbol} from cache")
                    valid_symbols.append(symbol)
            
            self.tradable_symbols = set(valid_symbols)
            self.logger.info(f"Updated tradable symbols: {len(self.tradable_symbols)} available after validation")
            return list(self.tradable_symbols)
        except Exception as e:
            self.logger.error(f"Error updating tradable symbols: {str(e)}", exc_info=True)
            return list(self.tradable_symbols)
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a specific symbol
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            response = self.api.session.get_instruments_info(
                category=self.category,
                symbol=symbol
            )
            
            if response and response.get("retCode") == 0:
                symbols_data = response.get("result", {}).get("list", [])
                if symbols_data:
                    return symbols_data[0]
            
            return {}
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def get_all_symbols_market_data(self) -> Dict:
        """
        Get latest market data for all tradable symbols
        
        Returns:
            Dictionary mapping symbols to their latest market data
        """
        try:
            response = self.api.session.get_tickers(category=self.category)
            if response and response.get("retCode") == 0:
                tickers = response.get("result", {}).get("list", [])
                
                # Convert to dictionary for easy lookup
                market_data = {}
                for ticker in tickers:
                    symbol = ticker.get("symbol")
                    if symbol in self.tradable_symbols:
                        market_data[symbol] = ticker
                
                return market_data
            else:
                self.logger.error(f"Failed to get market data: {response}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def filter_by_volume(self, min_volume: float = 1000000.0) -> List[str]:
        """
        Filter symbols by trading volume
        
        Args:
            min_volume: Minimum 24h volume in USD
            
        Returns:
            List of symbols meeting the volume criteria
        """
        try:
            # Holen wir uns die Marktdaten nur einmal
            market_data = self.get_all_symbols_market_data()
            
            # Beschränke die Ausgabe auf maximal 50 Symbole
            high_volume_symbols = []
            max_symbols = 50
            
            # Sortiere Symbole nach Volumen (absteigend)
            sorted_data = sorted(
                [(symbol, float(data.get("turnover24h", "0"))) for symbol, data in market_data.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Nehme nur Symbole mit ausreichendem Volumen
            for symbol, volume in sorted_data:
                if symbol in self.tradable_symbols and volume >= min_volume:
                    high_volume_symbols.append(symbol)
                    # Begrenze die Anzahl der zurückgegebenen Symbole
                    if len(high_volume_symbols) >= max_symbols:
                        break
            
            self.logger.info(f"Filtered {len(high_volume_symbols)} symbols with volume >= {min_volume}")
            return high_volume_symbols
        except Exception as e:
            self.logger.error(f"Error filtering by volume: {str(e)}", exc_info=True)
            return list(self.tradable_symbols)[:50]  # Begrenze auf 50
    
    def filter_by_volatility(self, min_volatility: float = 0.5, max_volatility: float = 15.0) -> List[str]:
        """
        Filter symbols by price volatility (24h change percentage)
        
        Args:
            min_volatility: Minimum price change percentage (absolute)
            max_volatility: Maximum price change percentage (absolute)
            
        Returns:
            List of symbols meeting the volatility criteria
        """
        try:
            # Hole Marktdaten einmal
            market_data = self.get_all_symbols_market_data()
            
            # Beschränke die Ausgabe auf maximal 50 Symbole
            volatility_filtered = []
            max_symbols = 50
            
            # Nehme nur Symbole aus der tradable_symbols Liste
            for symbol in self.tradable_symbols:
                if len(volatility_filtered) >= max_symbols:
                    break
                    
                data = market_data.get(symbol, {})
                if data:
                    price_24h_change_percent = abs(float(data.get("price24hPcnt", "0")) * 100)
                    if min_volatility <= price_24h_change_percent <= max_volatility:
                        volatility_filtered.append(symbol)
            
            self.logger.info(f"Filtered {len(volatility_filtered)} symbols with volatility between {min_volatility}% and {max_volatility}%")
            return volatility_filtered
        except Exception as e:
            self.logger.error(f"Error filtering by volatility: {str(e)}", exc_info=True)
            return list(self.tradable_symbols)[:50]  # Begrenze auf 50
    
    def get_optimal_trading_symbols(self, max_symbols: int = 10) -> List[str]:
        """
        Get a curated list of optimal trading symbols based on volume and volatility
        
        Args:
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of optimal trading symbols
        """
        try:
            # Update tradable symbols if needed
            if not self.tradable_symbols:
                self.update_tradable_symbols()
            
            # If whitelist is provided, use only those symbols but ensure they're valid
            if self.whitelist:
                self.logger.info(f"Using whitelist with {len(self.whitelist)} symbols")
                
                # Validate the whitelist symbols using our improved validator
                validated_symbols = self.validate_symbols(self.whitelist)
                
                if not validated_symbols:
                    self.logger.warning("None of the whitelist symbols are valid/tradable. Using default optimal symbols.")
                    # Fall back to automatic symbol selection if no whitelist symbols are valid
                else:
                    self.logger.info(f"Found {len(validated_symbols)} valid symbols from whitelist")
                    return validated_symbols[:max_symbols]
            
            # Filter by volume first - prefilter to a larger set of symbols
            volume_filtered = self.filter_by_volume(min_volume=5000000.0)
            if len(volume_filtered) > 30:
                volume_filtered = volume_filtered[:30]
            
            self.logger.info(f"Filtered to {len(volume_filtered)} high volume symbols")
            
            # Validate the volume-filtered symbols
            validated_symbols = self.validate_symbols(volume_filtered)
            self.logger.info(f"After validation: {len(validated_symbols)} valid high-volume symbols")
            
            if len(validated_symbols) < 5:
                # If we have too few validated symbols, try validating more from our tradable symbols
                self.logger.warning("Too few validated symbols after volume filtering, trying more symbols")
                additional_symbols = list(self.tradable_symbols)[:50]  # Try up to 50 more symbols
                additional_validated = self.validate_symbols(additional_symbols)
                validated_symbols.extend(additional_validated)
                # Remove duplicates
                validated_symbols = list(set(validated_symbols))
                self.logger.info(f"Added {len(additional_validated)} more validated symbols, total: {len(validated_symbols)}")
            
            # Then filter by volatility
            volatility_filtered = []
            for symbol in validated_symbols:
                metrics = self.calculate_volatility_metrics(symbol)
                if metrics and metrics.get('volatility_class') in ['low', 'medium', 'high']:
                    volatility_filtered.append(symbol)
                    if len(volatility_filtered) >= max_symbols:
                        break
            
            self.logger.info(f"Selected {len(volatility_filtered)} optimal trading symbols")
            
            # Return up to max_symbols
            return volatility_filtered[:max_symbols]
        except Exception as e:
            self.logger.error(f"Error getting optimal trading symbols: {str(e)}", exc_info=True)
            # In case of error, try to return at least some validated symbols
            try:
                # Take a small subset of symbols and validate them
                potential_symbols = list(self.tradable_symbols)[:20]
                validated_fallback = self.validate_symbols(potential_symbols)
                self.logger.info(f"Returning {len(validated_fallback)} fallback symbols after error")
                return validated_fallback[:max_symbols]
            except:
                # If all else fails, return a limited set of tradable symbols without validation
                return list(self.tradable_symbols)[:max_symbols]
    
    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        try:
            symbols_data = self.api.get_instruments_info()
            symbols = []
            
            if symbols_data:
                for symbol_data in symbols_data:
                    symbol = symbol_data.get("symbol", "")
                    if symbol and symbol not in self.blacklist:
                        symbols.append(symbol)
                        
            self.logger.info(f"Updated tradable symbols: {len(symbols)} available")
            return symbols
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            return []
            
    def add_to_blacklist(self, symbol: str) -> bool:
        """Add a symbol to the blacklist"""
        try:
            if symbol not in self.blacklist:
                self.blacklist.add(symbol)
                
                # Save to file
                with open(self.blacklist_file, 'a') as f:
                    f.write(f"{symbol}\n")
                    
                self.logger.info(f"Added {symbol} to blacklist")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error adding {symbol} to blacklist: {e}")
            return False
            
    def remove_from_blacklist(self, symbol: str) -> bool:
        """Remove a symbol from the blacklist"""
        try:
            if symbol in self.blacklist:
                self.blacklist.remove(symbol)
                
                # Rewrite file
                with open(self.blacklist_file, 'w') as f:
                    for s in sorted(self.blacklist):
                        f.write(f"{s}\n")
                        
                self.logger.info(f"Removed {symbol} from blacklist")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing {symbol} from blacklist: {e}")
            return False
    
    def detect_range(self, symbol: str, interval: str = "15", period: int = 20) -> tuple:
        """
        Detects if a symbol is in a range-bound market
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            period: Lookback period for range detection
            
        Returns:
            Tuple of (is_range, range_low, range_high, range_height_percent)
        """
        try:
            self.logger.debug(f"Detecting range for {symbol} on {interval}m timeframe")
            
            # Get candles
            api_response = self.api.get_kline(symbol=symbol, interval=interval, category=self.category, limit=period + 10)  # Get extra candles for calculation
            
            # Process the ApiResponse object
            if not api_response or not api_response.success:
                self.logger.warning(f"Failed to get kline data for range detection: {api_response.error_message if api_response else 'No response'}")
                return False, 0.0, 0.0, 0.0
                
            # Extract the actual candle data from the response
            candles_data = api_response.data
            
            # For API v5 format, the candles might be nested in a 'list' key
            if isinstance(candles_data, dict) and 'list' in candles_data:
                candles = candles_data.get('list', [])
            else:
                candles = candles_data
            
            if not candles or len(candles) < period:
                self.logger.warning(f"Insufficient data for range detection: {len(candles) if candles else 0} candles, need {period}")
                return False, 0.0, 0.0, 0.0
            
            # Normalize candle data
            df = normalize_candle_data(candles)
            
            if df.empty:
                self.logger.warning("Empty dataframe after normalization")
                return False, 0.0, 0.0, 0.0
            
            # Calculate range parameters
            high_range = df['high'].max()
            low_range = df['low'].min()
            
            # Calculate donchian channel
            df['dc_upper'] = df['high'].rolling(window=period).max()
            df['dc_lower'] = df['low'].rolling(window=period).min()
            
            # Get latest values
            latest = df.iloc[-1]
            upper = latest.get('dc_upper', high_range)
            lower = latest.get('dc_lower', low_range)
            
            # Calculate range height as percentage
            if lower > 0:
                range_height_pct = (upper - lower) / lower * 100
            else:
                range_height_pct = 0
                
            # Range detection criteria
            current_price = latest['close']
            price_position = (current_price - lower) / (upper - lower) if upper > lower else 0.5
            
            # Detect sideways market using range height and price position
            is_range = range_height_pct < 10 and 0.3 <= price_position <= 0.7
            
            # Also check if price is consistently between certain percentiles
            is_in_middle = True
            for i in range(min(10, len(df) - 1)):
                price = df.iloc[-(i+1)]['close']
                if price <= lower * 1.05 or price >= upper * 0.95:
                    is_in_middle = False
                    break
            
            # Combine criteria
            is_range = is_range and is_in_middle
            
            self.logger.debug(f"Range detection for {symbol}: is_range={is_range}, range_height={range_height_pct:.2f}%, " + 
                           f"price_position={price_position:.2f}")
            
            return is_range, float(lower), float(upper), float(range_height_pct)
            
        except Exception as e:
            self.logger.error(f"Error in range detection for {symbol}: {str(e)}", exc_info=True)
            return False, 0.0, 0.0, 0.0
                
    def detect_breakout(self, symbol: str, interval: str = "15", lookback: int = 20) -> tuple:
        """
        Detects if a symbol is experiencing a breakout
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            lookback: Lookback period for breakout detection
            
        Returns:
            Tuple of (is_breakout, breakout_direction, breakout_strength, consolidation_range)
        """
        try:
            self.logger.debug(f"Detecting breakout for {symbol} on {interval}m timeframe")
            
            # Get candles
            api_response = self.api.get_kline(symbol, interval, lookback + 30)  # Get extra candles for calculation
            
            # Process the ApiResponse object
            if not api_response or not api_response.success:
                self.logger.warning(f"Failed to get kline data for breakout detection: {api_response.error_message if api_response else 'No response'}")
                return False, "", 0.0, 0.0
                
            # Extract the actual candle data from the response
            candles_data = api_response.data
            
            # For API v5 format, the candles might be nested in a 'list' key
            if isinstance(candles_data, dict) and 'list' in candles_data:
                candles = candles_data.get('list', [])
            else:
                candles = candles_data
            
            if not candles or len(candles) < lookback + 5:
                self.logger.warning(f"Insufficient data for breakout detection: {len(candles) if candles else 0} candles, need {lookback + 5}")
                return False, "", 0.0, 0.0
            
            # Normalize candle data
            df = normalize_candle_data(candles)
            
            if df.empty:
                self.logger.warning("Empty dataframe after normalization")
                return False, "", 0.0, 0.0
            
            # Calculate consolidation range over lookback period
            consolidation_high = df['high'].iloc[-lookback:-5].max()  # Exclude last 5 candles
            consolidation_low = df['low'].iloc[-lookback:-5].min()    # Exclude last 5 candles
            
            consolidation_range = (consolidation_high - consolidation_low) / consolidation_low * 100 if consolidation_low > 0 else 0
            
            # Get recent prices for breakout detection
            latest_close = df['close'].iloc[-1]
            latest_high = df['high'].iloc[-1]
            latest_low = df['low'].iloc[-1]
            
            # Detect breakout - price exceeds consolidation range
            breakout_up = latest_close > consolidation_high
            breakout_down = latest_close < consolidation_low
            
            # Check volume confirmation - higher than average
            avg_volume = df['volume'].iloc[-lookback:-1].mean()
            latest_volume = df['volume'].iloc[-1]
            
            volume_surge = latest_volume > avg_volume * 1.5
            
            # Calculate breakout strength based on how far price moved beyond the range
            if breakout_up:
                breakout_strength = (latest_close - consolidation_high) / consolidation_high * 100 if consolidation_high > 0 else 0
                breakout_direction = "up"
            elif breakout_down:
                breakout_strength = (consolidation_low - latest_close) / consolidation_low * 100 if consolidation_low > 0 else 0
                breakout_direction = "down"
            else:
                breakout_strength = 0
                breakout_direction = ""
            
            # Final breakout determination - needs both price and volume confirmation
            is_breakout = (breakout_up or breakout_down) and (volume_surge or breakout_strength > 2.0)
            
            if is_breakout:
                self.logger.info(f"Breakout detected for {symbol}: Direction={breakout_direction}, " + 
                              f"Strength={breakout_strength:.2f}%, Volume surge: {volume_surge}, " +
                              f"Consolidation range: {consolidation_range:.2f}%")
            else:
                self.logger.debug(f"No breakout detected for {symbol}")
            
            return is_breakout, breakout_direction, breakout_strength, consolidation_range
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection for {symbol}: {str(e)}", exc_info=True)
            return False, "", 0.0, 0.0
            
    def calculate_volatility_metrics(self, symbol: str, interval: str = "15", lookback: int = 20) -> Dict:
        """
        Calculates volatility metrics for a symbol
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            lookback: Number of periods to consider
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            self.logger.debug(f"Calculating volatility metrics for {symbol} on {interval}m timeframe")
            
            # Get candles for ATR calculation
            api_response = self.api.get_kline(symbol=symbol, interval=interval, category=self.category, limit=lookback + 30)  # Get extra candles for calculation
            
            # Extract candles data from ApiResponse object
            if not api_response or not api_response.success:
                self.logger.warning(f"No candle data available for volatility calculation on {symbol}")
                return {}
                
            # Unwrap the data from the ApiResponse object
            candles = api_response.data
            
            # For API v5 format, the candles might be nested in a 'list' key
            if isinstance(candles, dict) and 'list' in candles:
                candles = candles.get('list', [])
            
            # Check if we have valid candle data
            if not isinstance(candles, list) or len(candles) == 0:
                self.logger.warning(f"Invalid candle data format for {symbol}: {type(candles)}")
                return {}
            
            # Log the structure of the first candle to help debugging API format
            self.logger.debug(f"First candle format for {symbol} volatility check: {candles[0]}")
                
            if len(candles) < lookback:
                self.logger.warning(f"Insufficient candle data for volatility calculation on {symbol}. Got {len(candles)}, need {lookback}")
                return {}
                
            # Normalize candle data
            df = normalize_candle_data(candles)
            
            if df.empty:
                self.logger.error(f"Failed to normalize candle data for {symbol}")
                return {}
            
            # Log columns to help debugging
            self.logger.debug(f"Normalized dataframe columns for {symbol} volatility check: {df.columns.tolist()}")
                
            # Check if we have enough data after normalization
            if len(df) < lookback:
                self.logger.warning(f"Insufficient valid data after normalization for {symbol}. Got {len(df)}, need {lookback}")
                return {}
            
            # Calculate True Range
            df["prev_close"] = df["close"].shift(1)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["prev_close"])
            df["tr3"] = abs(df["low"] - df["prev_close"])
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate ATR (Average True Range)
            df["atr"] = df["tr"].rolling(14).mean()
            
            # Calculate daily volatility (standard deviation of returns)
            df["returns"] = df["close"].pct_change() * 100
            
            # Calculate key metrics
            current_price = df["close"].iloc[0]  # First row as data is sorted in descending order
            atr = df["atr"].dropna().iloc[0] if not df["atr"].dropna().empty else 0
            daily_volatility = df["returns"].std()
            
            # Handle potential null values
            atr = 0 if pd.isna(atr) else atr
            daily_volatility = 0 if pd.isna(daily_volatility) else daily_volatility
            
            # Normalize ATR as percentage of price
            atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
            
            # Estimate hourly and daily price ranges based on ATR
            hourly_range = atr * 0.5  # Estimated hourly range
            daily_range = atr * 2.5   # Estimated daily range
            
            # Calculate market volatility classification
            if atr_percent < 0.5:
                volatility_class = "very_low"
            elif atr_percent < 1.0:
                volatility_class = "low"
            elif atr_percent < 2.0:
                volatility_class = "medium"
            elif atr_percent < 3.5:
                volatility_class = "high"
            else:
                volatility_class = "very_high"
                
            # Return metrics
            result = {
                "atr": float(atr),
                "atr_percent": float(atr_percent),
                "daily_volatility": float(daily_volatility),
                "hourly_range": float(hourly_range),
                "daily_range": float(daily_range),
                "volatility_class": volatility_class
            }
            
            self.logger.debug(f"Volatility metrics for {symbol}: ATR={atr_percent:.2f}%, Class={volatility_class}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics for {symbol}: {str(e)}", exc_info=True)
            return {}
    
    def is_symbol_valid(self, symbol: str, timeout: int = 10) -> tuple:
        """
        Check if a symbol is valid by attempting to fetch its data
        
        Args:
            symbol: Trading pair symbol
            timeout: Timeout in seconds for API calls
            
        Returns:
            Tuple of (is_valid, data_source)
        """
        # Check cache first to avoid repeated API calls
        if symbol in self._valid_symbols_cache:
            return self._valid_symbols_cache[symbol], "cache"
            
        try:
            self.logger.info(f"Validating symbol {symbol}...")
            
            # Step 1: Try to get basic info about the symbol first
            try:
                symbol_info = self.get_symbol_info(symbol)
                if not symbol_info:
                    self.logger.warning(f"Symbol {symbol} not found in instruments info - possibly invalid or delisted")
                    # Don't immediately invalidate, try getting candle data first
                else:
                    status = symbol_info.get("status", "")
                    if status not in ["Trading", "TRADING", "trading", ""]:
                        self.logger.warning(f"Symbol {symbol} status is {status}, not valid for trading")
                        self._valid_symbols_cache[symbol] = False
                        return False, "status"
                    else:
                        self.logger.debug(f"Symbol {symbol} basic info validation successful, status: {status}")
            except Exception as e:
                self.logger.warning(f"Error fetching symbol info for {symbol}: {str(e)}")
                # Continue to next validation step even if this fails
            
            # Step 2: Try to get data from API
            self.logger.debug(f"Attempting to fetch kline data for {symbol} from API")
            
            # Try multiple intervals for validation - some might work when others fail
            intervals_to_try = ["15", "60", "5"]
            valid_with_api = False
            validated_interval = None
            
            # Add dynamic delay between API calls to avoid rate limiting
            # Start with a small delay and increase if needed
            base_delay = 0.5  # seconds
            max_delay = 5.0   # seconds
            current_delay = base_delay
            max_attempts = 3  # attempts per interval
            
            for interval in intervals_to_try:
                limit = 1000  # Only need a few candles to validate
                
                for attempt in range(max_attempts):
                    try:
                        # Add delay to avoid overwhelming the API
                        time.sleep(current_delay)
                        
                        # Use threading for timeout instead of signal.SIGALRM (platform independent)
                        import threading
                        
                        # Variable to store result
                        result_container = {"api_candles": None, "exception": None, "completed": False}
                        
                        # Function to be executed in thread
                        def api_call():
                            try:
                                # Korrigierter Aufruf der get_kline Methode
                                result_container["api_candles"] = self.api.get_kline(
                                    symbol=symbol, 
                                    interval=interval, 
                                    category=self.category, 
                                    limit=limit
                                )
                                result_container["completed"] = True
                            except Exception as e:
                                result_container["exception"] = e
                                result_container["completed"] = True
                        
                        # Create and start thread
                        api_thread = threading.Thread(target=api_call)
                        api_thread.daemon = True  # Make thread daemonic so it exits when main thread exits
                        api_thread.start()
                        
                        # Wait for thread with timeout
                        start_time = time.time()
                        while not result_container["completed"] and time.time() - start_time < timeout:
                            time.sleep(0.1)  # Check every 100ms
                        
                        # Check if thread completed within timeout
                        if not result_container["completed"]:
                            self.logger.warning(f"API call for {symbol} timed out after {timeout} seconds")
                            current_delay = min(current_delay * 2, max_delay)  # Increase delay
                            continue
                        
                        # Check if exception was raised
                        if result_container["exception"]:
                            raise result_container["exception"]
                        
                        # Get result
                        api_candles = result_container["api_candles"]
                        
                        # Verbesserte API-Antwort-Extraktion
                        if hasattr(api_candles, 'data'):
                            # Es ist ein ApiResponse-Objekt
                            candles_data = api_candles.data
                            # Bei ApiResponse-Objekten kann die Liste tiefer verschachtelt sein
                            if isinstance(candles_data, dict) and 'list' in candles_data:
                                candles_data = candles_data.get('list', [])
                        # Manchmal kommt die API-Antwort als direkte dict
                        elif isinstance(api_candles, dict):
                            if 'result' in api_candles and isinstance(api_candles['result'], dict):
                                result_data = api_candles['result']
                                if 'list' in result_data:
                                    candles_data = result_data.get('list', [])
                                else:
                                    candles_data = result_data
                            elif 'list' in api_candles:
                                candles_data = api_candles.get('list', [])
                            else:
                                candles_data = api_candles
                        else:
                            candles_data = api_candles
                        
                        # Debug-Logging zur API-Antwortstruktur hinzufügen
                        if not isinstance(candles_data, list) and candles_data is not None:
                            self.logger.debug(f"API response format for {symbol}: {type(candles_data)}, keys: {candles_data.keys() if hasattr(candles_data, 'keys') else 'no keys'}")
                        
                        # Prüfe, ob genügend Kerzendaten vorhanden sind
                        if candles_data and isinstance(candles_data, list) and len(candles_data) >= 5:
                            # Symbol is valid with API data on this interval
                            valid_with_api = True
                            validated_interval = interval
                            self.logger.info(f"Successfully validated {symbol} with API data ({len(candles_data)} candles on {interval}m interval)")
                            break
                        else:
                            # Bestimme die Anzahl der Kerzen für das Logging
                            candles_count = len(candles_data) if isinstance(candles_data, list) else 0
                            self.logger.warning(f"API returned insufficient data for {symbol} on {interval}m interval ({candles_count} candles)")
                            
                            if attempt < max_attempts - 1:
                                # Increase delay before next attempt
                                current_delay = min(current_delay * 1.5, max_delay)
                                self.logger.debug(f"Retrying {symbol} with increased delay ({current_delay:.2f}s)")
                            elif attempt == max_attempts - 1 and interval == intervals_to_try[-1]:
                                # Wenn dies der letzte Versuch mit dem letzten Interval ist, sollte das Symbol zur Blacklist hinzugefügt werden
                                self.logger.warning(f"All attempts failed for {symbol}, adding to blacklist")
                                self.blacklist.add(symbol)
                                self._valid_symbols_cache[symbol] = False
                                self._save_blacklist()
                                
                    except Exception as e:
                        self.logger.warning(f"Error fetching {interval}m candles for {symbol}: {str(e)}")
                        
                        if attempt < max_attempts - 1:
                            # Increase delay before next attempt
                            current_delay = min(current_delay * 2, max_delay)
                            self.logger.debug(f"Retrying {symbol} with increased delay ({current_delay:.2f}s)")
                        continue
                
                if valid_with_api:
                    break
            
            if valid_with_api:
                self._valid_symbols_cache[symbol] = True
                return True, f"api-{validated_interval}m"
            else:
                self.logger.warning(f"Failed to validate {symbol} with any API interval, trying CSV fallback")
            
            # Step 3: Try to get data from CSV files (fallback)
            valid_with_csv = False
            for interval in intervals_to_try:
                csv_file = os.path.join('cache', f'{symbol}_{interval}_candles.csv')
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) >= 50:  # Require at least 50 rows in CSV
                            # Symbol is valid with CSV data
                            valid_with_csv = True
                            self.logger.info(f"Validated {symbol} with CSV data ({len(df)} rows from {interval}m data)")
                            break
                        else:
                            self.logger.warning(f"CSV file for {symbol} on {interval}m interval has insufficient data ({len(df)} rows)")
                    except Exception as e:
                        self.logger.warning(f"Error reading CSV file for {symbol} on {interval}m interval: {e}")
                else:
                    self.logger.debug(f"No CSV file found for {symbol} on {interval}m interval")
            
            if valid_with_csv:
                self._valid_symbols_cache[symbol] = True
                return True, "csv"
                
            # Step 4: Try with the symbol ticker as a last resort
            try:
                # Add small delay before ticker API call
                time.sleep(base_delay)
                
                ticker_response = self.api.session.get_tickers(
                    category=self.category,
                    symbol=symbol
                )
                
                if ticker_response and ticker_response.get("retCode") == 0:
                    ticker_data = ticker_response.get("result", {}).get("list", [])
                    if ticker_data and len(ticker_data) > 0:
                        # We have valid ticker data for this symbol
                        self.logger.info(f"Validated {symbol} with ticker data")
                        self._valid_symbols_cache[symbol] = True
                        return True, "ticker"
                
                self.logger.warning(f"Could not validate {symbol} with ticker data")
            except Exception as e:
                self.logger.warning(f"Error fetching ticker for {symbol}: {str(e)}")
            
            # Symbol is invalid if we reached here - all validation steps failed
            self.logger.error(f"Symbol {symbol} failed all validation checks")
            self._valid_symbols_cache[symbol] = False
            
            # Füge das Symbol zur Blacklist hinzu, wenn es alle Validierungsversuche nicht besteht
            if symbol not in self.blacklist:
                self.logger.warning(f"Adding {symbol} to blacklist after failed validation")
                self.blacklist.add(symbol)
                self._save_blacklist()
                
            return False, "none"
            
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {str(e)}", exc_info=True)
            self._valid_symbols_cache[symbol] = False
            return False, "error"

    def initialize(self):
        """
        Initialize the market fetcher with all necessary data
        
        This is the main method to call after creating a MarketFetcher instance
        """
        try:
            self.logger.info("Initializing Market Fetcher...")
            
            # Step 1: Load validated symbols from cache if it exists
            self._load_validated_symbols_cache()
            
            # Step 2: Update tradable symbols
            start_time = time.time()
            symbols = self.update_tradable_symbols()
            
            # Log the time taken
            elapsed = time.time() - start_time
            self.logger.info(f"Market Fetcher initialization completed in {elapsed:.2f} seconds")
            self.logger.info(f"Found {len(symbols)} tradable symbols")
            
            # Additional initialization can be done here
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Market Fetcher: {str(e)}", exc_info=True)
            return False
            
    def _load_validated_symbols_cache(self):
        """
        Load previously validated symbols from cache file
        """
        try:
            cache_file = os.path.join('data', 'validated_symbols_cache.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cache_time = cached_data.get('timestamp', 0)
                    cache_age = (datetime.now().timestamp() - cache_time) / 3600  # in hours
                    
                    if cache_age <= 24:  # Only use cache if less than 24 hours old
                        self._valid_symbols_cache = cached_data.get('symbols', {})
                        valid_count = sum(1 for v in self._valid_symbols_cache.values() if v)
                        self.logger.info(f"Loaded {len(self._valid_symbols_cache)} symbols from cache (age: {cache_age:.1f}h, {valid_count} valid)")
                    else:
                        self.logger.info(f"Validated symbols cache too old ({cache_age:.1f}h), will revalidate")
        except Exception as e:
            self.logger.warning(f"Failed to load validated symbols cache: {e}")
            
    def _save_validated_symbols_cache(self):
        """
        Save currently validated symbols to cache file
        """
        try:
            cache_file = os.path.join('data', 'validated_symbols_cache.json')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump({
                    'symbols': self._valid_symbols_cache,
                    'timestamp': datetime.now().timestamp()
                }, f)
                
            valid_count = sum(1 for v in self._valid_symbols_cache.values() if v)
            self.logger.info(f"Saved {len(self._valid_symbols_cache)} validated symbols to cache ({valid_count} valid)")
        except Exception as e:
            self.logger.warning(f"Failed to save validated symbols cache: {e}")
    
    def resume_validation(self, max_symbols: int = 50, batch_size: int = 10):
        """
        Resume validation of symbols that haven't been validated yet
        
        Args:
            max_symbols: Maximum number of symbols to validate
            batch_size: Number of symbols to validate in each batch
        
        Returns:
            List of successfully validated symbols
        """
        try:
            # Get all symbols from API
            self.logger.info("Fetching symbols for resume validation...")
            all_symbols = self.api.get_all_symbols(category=self.category)
            
            # Apply blacklist
            filtered_symbols = [s for s in all_symbols if s not in self.blacklist]
            self.logger.info(f"Found {len(filtered_symbols)} potential symbols after applying blacklist")
            
            # Find unvalidated symbols (not in cache)
            unvalidated = [s for s in filtered_symbols if s not in self._valid_symbols_cache]
            self.logger.info(f"Found {len(unvalidated)} symbols that need validation")
            
            # Limit number of symbols to validate
            symbols_to_validate = unvalidated[:max_symbols]
            
            # Process in batches
            return self._validate_symbols_in_batches(symbols_to_validate, batch_size)
        except Exception as e:
            self.logger.error(f"Error resuming validation: {str(e)}", exc_info=True)
            return []
            
    def _validate_symbols_in_batches(self, symbols_to_validate, batch_size):
        """Helper method to validate symbols in batches"""
        valid_symbols = []
        total_batches = (len(symbols_to_validate) + batch_size - 1) // batch_size
        
        self.logger.info(f"Validating {len(symbols_to_validate)} symbols in {total_batches} batches")
        
        # Setup progress tracking
        batch_delay = 3.0  # seconds between batches
        blacklist_updated = False
        
        for batch_num, i in enumerate(range(0, len(symbols_to_validate), batch_size)):
            batch = symbols_to_validate[i:i+batch_size]
            self.logger.info(f"Processing batch {batch_num+1}/{total_batches} ({len(batch)} symbols)")
            
            for symbol in batch:
                # Try to validate the symbol with increasing timeouts
                for timeout_attempt in range(3):  # Try with increasing timeouts
                    timeout = 10 * (timeout_attempt + 1)  # 10, 20, or 30 seconds
                    
                    try:
                        validation_result, validation_source = self.is_symbol_valid(symbol, timeout=timeout)
                        
                        if validation_result:
                            valid_symbols.append(symbol)
                            self._valid_symbols_cache[symbol] = True
                            self.logger.debug(f"Symbol {symbol} validated successfully using {validation_source}")
                            break  # Break out of timeout retry loop
                        elif timeout_attempt == 2:  # Last attempt failed
                            self.logger.warning(f"Symbol {symbol} validation failed after {timeout_attempt+1} attempts, adding to blacklist")
                            # Add to runtime blacklist
                            self.blacklist.add(symbol)
                            self._valid_symbols_cache[symbol] = False
                            blacklist_updated = True
                    except Exception as e:
                        self.logger.error(f"Error during validation of {symbol}: {str(e)}")
                        if timeout_attempt == 2:  # Last attempt
                            self.logger.warning(f"Symbol {symbol} validation failed with errors, adding to blacklist")
                            self.blacklist.add(symbol)
                            self._valid_symbols_cache[symbol] = False
                            blacklist_updated = True
            
            # Between batches, add a delay to avoid hitting rate limits
            if batch_num < total_batches - 1:
                self.logger.debug(f"Sleeping for {batch_delay} seconds between batches")
                time.sleep(batch_delay)
            
            # Save progress after each batch
            self._save_validated_symbols_cache()
        
        # Save the updated blacklist if needed
        if blacklist_updated:
            self._save_blacklist()
            
        return valid_symbols

    def validate_symbols(self, symbols_list, batch_size=10):
        """
        Validate a list of symbols to make sure they're tradable
        
        Args:
            symbols_list: List of symbols to validate
            batch_size: Number of symbols to validate in each batch
        
        Returns:
            List of valid symbols
        """
        validated_symbols = []
        blacklisted_symbols = []
        
        # Create a record of symbols that consistently cause errors
        error_symbols_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'error_symbols.json')
        
        # Load existing error symbols list if it exists
        error_symbols = {}
        if os.path.exists(error_symbols_file):
            try:
                with open(error_symbols_file, 'r') as f:
                    error_symbols = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading error symbols file: {e}")
        
        # Process symbols in batches to avoid rate limiting
        for i in range(0, len(symbols_list), batch_size):
            batch = symbols_list[i:i+batch_size]
            self.logger.info(f"Validating batch {i//batch_size + 1}/{math.ceil(len(symbols_list)/batch_size)}: {batch}")
            
            for symbol in batch:
                # Skip already blacklisted symbols
                if symbol in blacklisted_symbols:
                    continue
                
                # Skip symbols that have caused errors in the past
                if symbol in error_symbols and error_symbols[symbol].get('error_count', 0) > 3:
                    self.logger.info(f"Skipping {symbol} due to previous API errors")
                    blacklisted_symbols.append(symbol)
                    continue
                
                try:
                    # First, attempt to get instrument info for the symbol
                    instrument_info = self.api.get_instrument_info(symbol)
                    
                    if not instrument_info or not hasattr(instrument_info, 'success') or not instrument_info.success:
                        self.logger.warning(f"Symbol {symbol} not found or not tradable (instrument info check failed)")
                        self._add_to_error_symbols(error_symbols, symbol, "instrument_info_failed")
                        blacklisted_symbols.append(symbol)
                        continue
                    
                    # Then try to get recent kline data (this is a common source of errors)
                    interval = "15"  # 15 minute candles
                    kline_data = self.api.get_kline(symbol, interval, limit=1)
                    
                    if not kline_data or not hasattr(kline_data, 'success') or not kline_data.success:
                        self.logger.warning(f"Symbol {symbol} returned invalid kline data, skipping")
                        self._add_to_error_symbols(error_symbols, symbol, "kline_data_failed")
                        blacklisted_symbols.append(symbol)
                        continue
                    
                    # Check if there's actual data in the kline response
                    if not kline_data.data or len(kline_data.data) == 0:
                        self.logger.warning(f"Symbol {symbol} returned empty kline data, skipping")
                        self._add_to_error_symbols(error_symbols, symbol, "empty_kline_data")
                        blacklisted_symbols.append(symbol)
                        continue
                    
                    # Also try to get tickers to check if price data is available
                    try:
                        ticker_data = self.api.get_tickers(symbol=symbol)
                        
                        if not ticker_data or 'result' not in ticker_data or 'list' not in ticker_data['result'] or not ticker_data['result']['list']:
                            self.logger.warning(f"Symbol {symbol} has no price data in tickers, skipping")
                            self._add_to_error_symbols(error_symbols, symbol, "no_ticker_data")
                            blacklisted_symbols.append(symbol)
                            continue
                    except Exception as ticker_err:
                        self.logger.warning(f"Error fetching ticker for {symbol}: {ticker_err}")
                        self._add_to_error_symbols(error_symbols, symbol, f"ticker_error: {str(ticker_err)}")
                        # Don't necessarily blacklist here as ticker might not be available for all symbols
                    
                    # Symbol passed all validation checks
                    self.logger.info(f"Symbol {symbol} validated successfully")
                    validated_symbols.append(symbol)
                    
                    # Reset error count for this symbol if it previously had errors
                    if symbol in error_symbols:
                        error_symbols[symbol]['error_count'] = 0
                    
                except Exception as e:
                    self.logger.error(f"Error validating symbol {symbol}: {e}")
                    self._add_to_error_symbols(error_symbols, symbol, str(e))
                    blacklisted_symbols.append(symbol)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
        
        # Save updated error symbols list
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(error_symbols_file), exist_ok=True)
            
            with open(error_symbols_file, 'w') as f:
                json.dump(error_symbols, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving error symbols file: {e}")
        
        self.logger.info(f"Validation complete. {len(validated_symbols)} valid symbols, {len(blacklisted_symbols)} blacklisted.")
        return validated_symbols

    def _add_to_error_symbols(self, error_symbols, symbol, error_message):
        """Add a symbol to the error list or update its error count"""
        timestamp = datetime.now().isoformat()
        
        if symbol in error_symbols:
            error_symbols[symbol]['error_count'] += 1
            error_symbols[symbol]['last_error'] = error_message
            error_symbols[symbol]['last_error_time'] = timestamp
        else:
            error_symbols[symbol] = {
                'error_count': 1,
                'first_error_time': timestamp,
                'last_error_time': timestamp,
                'last_error': error_message
            }