import logging
import traceback
import pandas as pd
import numpy as np
import ta
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, SMAIndicator
from typing import Dict, List, Tuple, Optional, Union, Any
from strategy.base_strategy import BaseStrategy
from utils.logger import log_strategy_decision, log_data_load, log_exception, log_api_call
from api.bybit_api import BybitAPI
from utils.data_storage import save_candles_to_csv, load_candles_from_csv

class DonchianChannelStrategy(BaseStrategy):
    """
    Donchian Channel strategy implementation
    """
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialize the Donchian Channel strategy with configuration
        
        Args:
            api: BybitAPI instance
            config: Strategy configuration
        """
        super().__init__(api, config)
        
        # Extract strategy parameters from config
        strategy_params = config.get('strategy', {}).get('parameters', {}).get('donchian_channel', {})
        
        # Donchian Channel parameters
        self.dc_period = strategy_params.get('dc_period', 20)
        self.breakout_confirmation = strategy_params.get('breakout_confirmation', 2)
        self.trailing_exit = strategy_params.get('trailing_exit', True)
        self.atr_multiplier = strategy_params.get('atr_multiplier', 2.0)
        
        # Filter options
        self.use_adx_filter = strategy_params.get('use_adx_filter', True)
        self.adx_period = strategy_params.get('adx_period', 14)
        self.adx_threshold = strategy_params.get('adx_threshold', 25)
        
        # Additional parameters
        self.use_middle_channel = strategy_params.get('use_middle_channel', True)
        self.exit_opposite_band = strategy_params.get('exit_opposite_band', False)
        
        self.logger.info(f"Initialized Donchian Channel strategy with period={self.dc_period}")
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating indicators
        
        Args:
            df: Raw price data
            
        Returns:
            DataFrame with indicators
        """
        # Make sure we have enough data for calculations
        if len(df) < self.dc_period + 50:
            self.logger.warning(f"Not enough data for Donchian Channel calculations. Need at least {self.dc_period + 50} candles, got {len(df)}")
            return df
        
        try:
            # Calculate Donchian Channel
            df['dc_upper'] = df['high'].astype(float).rolling(window=self.dc_period).max()
            df['dc_lower'] = df['low'].astype(float).rolling(window=self.dc_period).min()
            
            # Calculate middle channel if needed
            if self.use_middle_channel:
                df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
            
            # Calculate ATR for volatility-based stops using ta library
            atr_indicator = AverageTrueRange(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), window=14)
            df['atr'] = atr_indicator.average_true_range()
            
            # ADX for trend strength using ta library
            if self.use_adx_filter:
                adx_indicator = ADXIndicator(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), window=self.adx_period)
                df['adx'] = adx_indicator.adx()
            
            # Calculate other technical indicators using ta library
            ema50_indicator = EMAIndicator(df['close'].astype(float), window=50)
            ema200_indicator = EMAIndicator(df['close'].astype(float), window=200)
            df['ema50'] = ema50_indicator.ema_indicator()
            df['ema200'] = ema200_indicator.ema_indicator()
            df['trend'] = np.where(df['ema50'] > df['ema200'], 1, -1)
            
            # Volume indicators using ta library
            volume_ma_indicator = SMAIndicator(df['volume'].astype(float), window=20)
            df['volume_ma'] = volume_ma_indicator.sma_indicator()
            df['volume_ratio'] = df['volume'].astype(float) / df['volume_ma']
            
            # Calculate channel width for potential range detection
            df['dc_width'] = df['dc_upper'] - df['dc_lower']
            if self.use_middle_channel:
                df['dc_width_pct'] = df['dc_width'] / df['dc_middle']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            log_exception(self.logger, e, "Donchian prepare_data", traceback.format_exc())
            return df
    
    def analyze(self, symbol: str, interval: str = "15", limit: int = 100) -> Dict[str, Any]:
        """
        Analyze a symbol and generate trading signals
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            limit: Number of candles to analyze
            
        Returns:
            Dictionary with analysis results and signals
        """
        try:
            self.logger.debug(f"Starting analysis of {symbol} ({interval}m)")
            
            # First, try to load from cache if enabled
            candles = []
            data_source = None
            
            if self.cache_enabled:
                self.logger.debug(f"Attempting to load cached data for {symbol}")
                candles = load_candles_from_csv(symbol, interval, self.cache_dir)
                if candles and len(candles) >= limit:
                    data_source = "cache"
                    log_data_load(
                        self.logger, 
                        source="CSV cache", 
                        symbol=symbol, 
                        timeframe=f"{interval}m", 
                        rows=len(candles), 
                        success=True
                    )
                    self.logger.info(f"Using cached data for {symbol} ({interval}m), {len(candles)} candles")
                else:
                    self.logger.debug(f"Not enough cached data for {symbol}, {len(candles) if candles else 0} candles found, need {limit}")
            
            # If no cached data or not enough data, fetch from API
            if not candles or len(candles) < limit:
                self.logger.info(f"Fetching fresh data for {symbol} ({interval}m)")
                api_response = self.api.get_kline(symbol=symbol, interval=interval, category="linear", limit=limit)
                
                # Process the ApiResponse object
                if api_response and api_response.success:
                    # Extract the actual candle data from the response
                    candles_data = api_response.data
                    
                    # For API v5 format, the candles might be nested in a 'list' key
                    if isinstance(candles_data, dict) and 'list' in candles_data:
                        candles = candles_data.get('list', [])
                    else:
                        candles = candles_data
                    
                    if candles and len(candles) > 0:
                        data_source = "api"
                        log_data_load(
                            self.logger, 
                            source="API", 
                            symbol=symbol, 
                            timeframe=f"{interval}m", 
                            rows=len(candles), 
                            success=True
                        )
                        self.logger.info(f"Downloaded {len(candles)} candles for {symbol} ({interval}m) from API")
                        
                        # Save to cache if enabled
                        if self.cache_enabled:
                            save_candles_to_csv(symbol, interval, candles, self.cache_dir)
                            self.logger.debug(f"Saved {len(candles)} candles to cache for {symbol}")
                    else:
                        log_data_load(
                            self.logger, 
                            source="API", 
                            symbol=symbol, 
                            timeframe=f"{interval}m", 
                            success=False, 
                            error="No data returned"
                        )
                        self.logger.error(f"No API data available for {symbol}")
                        return {"signal": "neutral", "error": "No data available", "data_source": "none"}
                else:
                    error_msg = "API response failed" if api_response else "No API response"
                    if api_response and not api_response.success:
                        error_msg = api_response.error_message
                    
                    log_data_load(
                        self.logger, 
                        source="API", 
                        symbol=symbol, 
                        timeframe=f"{interval}m", 
                        success=False, 
                        error=error_msg
                    )
                    self.logger.error(f"API data fetch failed for {symbol}: {error_msg}")
                    return {"signal": "neutral", "error": error_msg, "data_source": "none"}
                
            if not candles:
                self.logger.error(f"No data available for {symbol} from any source")
                return {"signal": "neutral", "error": "No data", "data_source": "none"}
                
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Ensure columns exist and convert to proper types
            required_columns = ["open", "high", "low", "close", "volume", "timestamp"]
            for col in required_columns:
                if col not in df.columns:
                    error_msg = f"Required column {col} missing from data"
                    self.logger.error(error_msg)
                    return {"signal": "neutral", "error": error_msg, "data_source": data_source}
                    
            # Convert string values to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            # Sort by timestamp (oldest first)
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
            
            # Prepare data by calculating indicators
            df = self.prepare_data(df)
            
            # Generate trading signal
            signal, params = self.generate_signal(df)
            
            # Prepare results
            latest = df.iloc[-1]
            result = {
                "symbol": symbol,
                "signal": signal,
                "price": latest["close"],
                "timestamp": latest["timestamp"],
                "dc_upper": latest.get("dc_upper", 0),
                "dc_lower": latest.get("dc_lower", 0),
                "dc_middle": latest.get("dc_middle", 0) if self.use_middle_channel else 0,
                "atr": latest.get("atr", 0),
                "adx": latest.get("adx", 0) if self.use_adx_filter else 0,
                "data_source": data_source,
                "params": params
            }
            
            # Log the strategy decision
            indicators = {
                "dc_upper": float(latest.get("dc_upper", 0)),
                "dc_lower": float(latest.get("dc_lower", 0)),
                "dc_middle": float(latest.get("dc_middle", 0)) if self.use_middle_channel else 0,
                "atr": float(latest.get("atr", 0)),
                "adx": float(latest.get("adx", 0)) if self.use_adx_filter else 0,
                "ema50": float(latest.get("ema50", 0)),
                "ema200": float(latest.get("ema200", 0))
            }
            
            signals = {
                "upper_breakout": latest["close"] > latest.get("dc_upper", float('inf')),
                "lower_breakout": latest["close"] < latest.get("dc_lower", 0),
                "strong_trend": latest.get("adx", 0) > self.adx_threshold if self.use_adx_filter else False,
                "bullish_trend": latest.get("ema50", 0) > latest.get("ema200", 0),
                "high_volume": latest.get("volume_ratio", 0) > 1.0
            }
            
            reason = params.get("reason", "No reason provided") if params else "No parameters available"
            
            log_strategy_decision(
                self.logger,
                symbol=symbol, 
                timeframe=f"{interval}m",
                decision=signal,
                signals=signals,
                indicators=indicators,
                reason=reason
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error analyzing {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"Donchian Channel analysis for {symbol}", traceback.format_exc())
            return {"signal": "neutral", "error": str(e), "data_source": "error"}
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, Optional[Dict]]:
        """
        Generate trading signal based on prepared data
        
        Args:
            df: Prepared DataFrame with indicators
            
        Returns:
            Tuple of (signal, parameters)
        """
        # Check if we have enough data
        if len(df) < self.dc_period + self.breakout_confirmation:
            return ('neutral', None)
        
        # Get latest candles
        last_candle = df.iloc[-1]
        
        # Initialize signal
        signal = 'neutral'
        params = None
        
        # Upper channel breakout (buy signal)
        upper_breakout = False
        adx_condition = True
        volume_condition = last_candle.get('volume_ratio', 0) > 1.0
        
        # Check for breakout confirmation
        for i in range(1, self.breakout_confirmation + 1):
            if i < len(df):
                if df.iloc[-i]['close'] > df.iloc[-i].get('dc_upper', float('inf')):
                    upper_breakout = True
                    break
        
        # ADX filter
        if self.use_adx_filter and 'adx' in df.columns:
            adx_condition = last_candle['adx'] > self.adx_threshold
            
        # Generate buy signal
        if (upper_breakout and 
            adx_condition and 
            volume_condition and
            last_candle.get('trend', 0) == 1):  # Trend filter
            
            signal = 'buy'
            entry_price = last_candle['close']
            
            # Calculate stop-loss using ATR or a fixed percentage of entry price
            stop_loss = entry_price - (last_candle['atr'] * self.atr_multiplier)
            
            # Take profit based on channel characteristics
            if self.exit_opposite_band:
                take_profit = last_candle.get('dc_lower', entry_price * 0.9)
            else:
                # Risk-reward based take profit
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2)  # 1:2 risk-reward ratio
            
            params = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': 'Upper channel breakout with trend confirmation'
            }
        
        # Lower channel breakout (sell signal)
        lower_breakout = False
        
        # Check for breakout confirmation
        for i in range(1, self.breakout_confirmation + 1):
            if i < len(df):
                if df.iloc[-i]['close'] < df.iloc[-i].get('dc_lower', 0):
                    lower_breakout = True
                    break
                    
        # Generate sell signal
        if (lower_breakout and 
            adx_condition and 
            volume_condition and
            last_candle.get('trend', 0) == -1):  # Trend filter
            
            signal = 'sell'
            entry_price = last_candle['close']
            
            # Calculate stop-loss using ATR or a fixed percentage of entry price
            stop_loss = entry_price + (last_candle.get('atr', entry_price * 0.02) * self.atr_multiplier)
            
            # Take profit based on channel characteristics
            if self.exit_opposite_band:
                take_profit = last_candle.get('dc_upper', entry_price * 1.1)
            else:
                # Risk-reward based take profit
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)  # 1:2 risk-reward ratio
            
            params = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': 'Lower channel breakout with trend confirmation'
            }
        
        return (signal, params) 