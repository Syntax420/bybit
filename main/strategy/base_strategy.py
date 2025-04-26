import logging
import os
import time
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import traceback

from api.bybit_api import BybitAPI

class BaseStrategy(ABC):
    """
    Abstrakte Basisklasse für alle Trading-Strategien.
    
    Diese Klasse definiert die grundlegende Schnittstelle und Funktionalität,
    die von allen konkreten Strategien implementiert werden muss.
    """
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialisiert die Basisstrategie mit API-Zugriff und Konfiguration
        
        Args:
            api: BybitAPI-Instanz für Marktdaten und Order-Ausführung
            config: Konfigurationswörterbuch
        """
        self.api = api
        self.config = config
        self.name = self.__class__.__name__
        
        # Logger mit Strategie-Namen erstellen
        self.logger = logging.getLogger(f"strategy.{self.name.lower()}")
        
        # Default-Werte
        self.cache_dir = config.get("general", {}).get("cache_directory", "cache")
        self.cache_enabled = config.get("general", {}).get("cache_enabled", True)
        self.max_cache_age = config.get("general", {}).get("max_cache_age_minutes", 60)
        
        # Stellen Sie sicher, dass das Cache-Verzeichnis existiert
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.logger.info(f"{self.name} Strategie initialisiert")
        
    def fetch_candles(self, symbol: str, interval: str = "15", limit: int = 200) -> pd.DataFrame:
        """
        Fetch candlestick data and return as a pandas dataframe
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe interval
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Try to use cached data first if available
            if self.cache_enabled:
                cache_file = os.path.join(self.cache_dir, f"{symbol}_{interval}_candles.csv")
                
                # Prüfen, ob eine Cache-Datei existiert und nicht zu alt ist
                if os.path.exists(cache_file):
                    file_age_minutes = (time.time() - os.path.getmtime(cache_file)) / 60
                    
                    if file_age_minutes <= self.max_cache_age:
                        try:
                            df = pd.read_csv(cache_file)
                            df['timestamp'] = pd.to_numeric(df['timestamp'])  # Stelle sicher, dass Timestamp numerisch ist
                            
                            # Ensure all required columns exist with lower case names
                            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                            for col in df.columns:
                                if col.lower() in required_columns and col != col.lower():
                                    df[col.lower()] = df[col]
                            
                            df.attrs['data_source'] = 'cache'
                            self.logger.debug(f"Using cached data for {symbol} ({interval}m), Alter: {file_age_minutes:.1f} Minuten")
                            return df
                        except Exception as e:
                            self.logger.warning(f"Fehler beim Laden aus dem Cache für {symbol}: {str(e)}")
            
            # Fetch fresh data from API
            self.logger.debug(f"Fetching fresh candle data for {symbol} ({interval}m)")
            api_response = self.api.get_kline(symbol=symbol, interval=interval, category="linear", limit=limit)
            
            # Process the ApiResponse object
            if not api_response or not api_response.success:
                self.logger.warning(f"Failed to get kline data: {api_response.error_message if api_response else 'No response'}")
                return pd.DataFrame()
                
            # Extract the actual candle data from the response
            candles_data = api_response.data
            
            # For API v5 format, the candles might be nested in a 'list' key
            if isinstance(candles_data, dict) and 'list' in candles_data:
                candles = candles_data.get('list', [])
            else:
                candles = candles_data
            
            if not candles or len(candles) == 0:
                self.logger.warning(f"No candle data returned for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame and ensure columns have consistent naming
            df = pd.DataFrame(candles)
            
            # Normalize column names - handle both string and numeric columns
            df.columns = [str(col).lower() for col in df.columns]
            
            # Map numeric columns to standard names if needed (for API responses with numeric columns)
            if all(col.isdigit() for col in df.columns):
                # Map numeric columns to standard names
                # Typical order: timestamp, open, high, low, close, volume, turnover
                column_mapping = {
                    '0': 'timestamp', 
                    '1': 'open', 
                    '2': 'high', 
                    '3': 'low', 
                    '4': 'close', 
                    '5': 'volume', 
                    '6': 'turnover'
                }
                df = df.rename(columns=column_mapping)
                self.logger.debug(f"Renamed numeric columns: {df.columns.tolist()}")
            
            # Map expected column names if they're different in the API response
            column_mappings = {
                'start_time': 'timestamp',
                'starttime': 'timestamp',
                'closing_price': 'close',
                'opening_price': 'open',
                'highest_price': 'high',
                'lowest_price': 'low'
            }
            
            for src, dest in column_mappings.items():
                if src in df.columns and dest not in df.columns:
                    df[dest] = df[src]
            
            # Ensure numeric types
            numeric_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                    
            # Save to cache if enabled
            if self.cache_enabled:
                try:
                    df.to_csv(cache_file, index=False)
                    self.logger.debug(f"Candles for {symbol} ({interval}m) saved to cache")
                except Exception as e:
                    self.logger.warning(f"Fehler beim Speichern im Cache für {symbol}: {str(e)}")
            
            df.attrs['data_source'] = 'api'
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    @abstractmethod
    def analyze(self, symbol: str, interval: str = "15", limit: int = 100) -> Dict:
        """
        Abstrakte Methode zur Analyse eines Symbols und Generierung von Handelssignalen
        Muss von allen Unterklassen implementiert werden
        
        Args:
            symbol: Trading-Paar-Symbol
            interval: Zeitrahmen-Intervall
            limit: Anzahl der zu analysierenden Kerzen
            
        Returns:
            Dictionary mit Analyseergebnissen und Signalen
        """
        pass
        
    @abstractmethod
    def get_strategy_parameters(self) -> Dict:
        """
        Abstrakte Methode zur Rückgabe der Strategie-Parameter
        Muss von allen Unterklassen implementiert werden
        
        Returns:
            Dictionary mit Strategie-Parametern
        """
        pass
        
    def calculate_volatility(self, df: pd.DataFrame, periods: int = 14) -> float:
        """
        Berechnet die Volatilität eines Assets basierend auf Kerzendaten
        
        Args:
            df: DataFrame mit Kerzendaten
            periods: Anzahl der zu betrachtenden Perioden
            
        Returns:
            Volatilität als Prozentwert
        """
        try:
            # Standard-Ansatz: Standardabweichung der prozentualen Preisänderung
            if len(df) <= periods:
                # Zu wenige Daten für zuverlässige Berechnung
                return 0.0
                
            # Prozentuale Preisänderungen berechnen
            df['returns'] = df['close'].pct_change() * 100
            
            # Standardabweichung der Renditen über den angegebenen Zeitraum
            volatility = df['returns'].tail(periods).std()
            
            self.logger.debug(f"Volatilität berechnet: {volatility:.2f}% über {periods} Perioden")
            return volatility
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Volatilitätsberechnung: {str(e)}")
            return 0.0
            
    def check_multi_timeframe_confirmation(self, symbol: str, base_interval: str, 
                                         signal_type: str) -> Dict[str, Any]:
        """
        Überprüft, ob ein Signal durch andere Zeitrahmen bestätigt wird
        
        Args:
            symbol: Trading-Paar-Symbol
            base_interval: Basis-Zeitrahmen des Signals
            signal_type: Signaltyp ('buy' oder 'sell')
            
        Returns:
            Dictionary mit Bestätigungsinformationen
        """
        try:
            # Definieren der zu überprüfenden Zeitrahmen (größer als Basis)
            intervals = ["15", "60", "240", "D"]  # 15m, 1h, 4h, 1d
            
            # Nur Zeitrahmen prüfen, die größer als der Basis-Zeitrahmen sind
            base_idx = intervals.index(base_interval) if base_interval in intervals else -1
            if base_idx >= 0:
                check_intervals = intervals[base_idx+1:]
            else:
                check_intervals = intervals[1:]  # Standardmäßig größere Zeitrahmen überprüfen
                
            if not check_intervals:
                return {"confirmed": True, "confirmations": 0, "details": {}}
                
            self.logger.info(f"Überprüfe Multi-Timeframe-Bestätigung für {symbol} ({signal_type}) in Zeitrahmen: {', '.join(check_intervals)}")
            
            confirmations = 0
            details = {}
            
            # Überprüfe jeden Zeitrahmen
            for interval in check_intervals:
                result = self.analyze(symbol, interval)
                interval_signal = result.get("signal", "neutral")
                
                # Signal gilt als bestätigt, wenn es gleich oder neutral ist
                is_confirmed = (interval_signal == signal_type or interval_signal == "neutral")
                if is_confirmed:
                    confirmations += 1
                    
                details[interval] = {
                    "signal": interval_signal,
                    "confirmed": is_confirmed,
                    "indicators": {
                        key: result.get(key, None) for key in ["rsi", "macd", "ema_short", "ema_long"] 
                        if key in result
                    }
                }
                
                self.logger.debug(f"Zeitrahmen {interval}: {interval_signal}, Bestätigt: {is_confirmed}")
                
            # Gesamtergebnis
            confirmation_ratio = confirmations / len(check_intervals) if check_intervals else 1.0
            confirmed = confirmation_ratio >= 0.5  # Mehr als die Hälfte muss übereinstimmen
            
            result = {
                "confirmed": confirmed,
                "confidence": confirmation_ratio,
                "confirmations": confirmations,
                "total_timeframes": len(check_intervals),
                "details": details
            }
            
            self.logger.info(f"Multi-Timeframe-Bestätigung für {symbol} ({signal_type}): "
                          f"{'Bestätigt' if confirmed else 'Nicht bestätigt'} "
                          f"({confirmation_ratio*100:.1f}% Konfidenz)")
                          
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Multi-Timeframe-Überprüfung für {symbol}: {str(e)}")
            return {"confirmed": False, "error": str(e)}