import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from api.bybit_api import BybitAPI
from strategy.base_strategy import BaseStrategy

class RSIMACDStrategy(BaseStrategy):
    """
    RSI und MACD Kombinationsstrategie.
    
    Diese Strategie kombiniert RSI (Relative Strength Index) und MACD (Moving Average Convergence Divergence)
    Indikatoren, um Handelssignale zu erzeugen.
    """
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialisiert die RSI-MACD-Strategie
        
        Args:
            api: BybitAPI-Instanz für Marktdaten
            config: Konfigurationswörterbuch
        """
        super().__init__(api, config)
        
        # Strategie-Parameter aus der Konfiguration laden
        strategy_config = config.get("strategy", {}).get("rsi_macd", {})
        
        # RSI Parameter
        self.rsi_length = strategy_config.get("rsi_length", 14)
        self.rsi_overbought = strategy_config.get("rsi_overbought", 70)
        self.rsi_oversold = strategy_config.get("rsi_oversold", 30)
        
        # MACD Parameter
        self.macd_fast = strategy_config.get("macd_fast", 12)
        self.macd_slow = strategy_config.get("macd_slow", 26)
        self.macd_signal = strategy_config.get("macd_signal", 9)
        
        # EMA Parameter
        self.use_ema_filter = strategy_config.get("use_ema_filter", True)
        self.ema_short_length = strategy_config.get("ema_short", 50)
        self.ema_long_length = strategy_config.get("ema_long", 200)
        
        # Signalparameter
        self.min_rsi_value = strategy_config.get("min_rsi_value", 15)
        self.max_rsi_value = strategy_config.get("max_rsi_value", 85)
        self.macd_threshold = strategy_config.get("macd_threshold", 0.0)
        self.confirmation_needed = strategy_config.get("confirmation_needed", True)
        self.multi_timeframe_check = strategy_config.get("multi_timeframe_check", True)
        
        self.logger.info(f"RSI-MACD Strategie initialisiert: RSI({self.rsi_length}), MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), "
                        f"EMA({self.ema_short_length},{self.ema_long_length})")
    
    def analyze(self, symbol: str, interval: str = "15", limit: int = 100) -> Dict[str, Any]:
        """
        Analysiert ein Symbol mit RSI und MACD Indikatoren und generiert ein Handelssignal
        
        Args:
            symbol: Trading-Paar-Symbol
            interval: Zeitrahmen-Intervall
            limit: Anzahl der zu analysierenden Kerzen
            
        Returns:
            Dictionary mit Analyseergebnissen und Signalen
        """
        try:
            # Kerzendaten abrufen
            df = self.fetch_candles(symbol, interval, limit + self.macd_slow + self.macd_signal)
            
            if df.empty:
                self.logger.warning(f"Keine Kerzendaten für {symbol} verfügbar")
                return {"signal": "neutral", "error": "Keine Daten"}
            
            # Indikatoren berechnen
            self._calculate_rsi(df)
            self._calculate_macd(df)
            
            if self.use_ema_filter:
                self._calculate_ema(df)
            
            # Nur die neuesten Daten für die Analyse verwenden
            latest = df.iloc[-1]
            prior = df.iloc[-2] if len(df) > 1 else latest
            
            # Debugging-Ausgabe
            self.logger.debug(f"{symbol} ({interval}m) - RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.6f}, "
                            f"Signal: {latest['macd_signal']:.6f}, Hist: {latest['macd_hist']:.6f}")
            
            # Signal-Generierung
            signal = self._generate_signal(df)
            
            # Volatilität berechnen
            volatility = self.calculate_volatility(df, periods=self.rsi_length)
            
            # Ergebnis zusammenstellen
            result = {
                "symbol": symbol,
                "interval": interval,
                "signal": signal,
                "rsi": latest['rsi'],
                "prior_rsi": prior['rsi'],
                "macd": latest['macd'],
                "macd_signal": latest['macd_signal'],
                "macd_hist": latest['macd_hist'],
                "volatility": volatility,
                "price": latest['close']
            }
            
            # EMA-Werte hinzufügen, wenn aktiviert
            if self.use_ema_filter:
                result.update({
                    "ema_short": latest[f'ema_{self.ema_short_length}'],
                    "ema_long": latest[f'ema_{self.ema_long_length}'],
                    "is_uptrend": latest[f'ema_{self.ema_short_length}'] > latest[f'ema_{self.ema_long_length}'],
                    "is_downtrend": latest[f'ema_{self.ema_short_length}'] < latest[f'ema_{self.ema_long_length}']
                })
            
            # RSI-/MACD-Zustand hinzufügen
            result.update({
                "is_oversold": latest['rsi'] <= self.rsi_oversold,
                "is_overbought": latest['rsi'] >= self.rsi_overbought,
                "is_bullish": latest['macd'] > latest['macd_signal'],
                "is_bearish": latest['macd'] < latest['macd_signal'],
                "is_cross_up": latest['macd'] > latest['macd_signal'] and prior['macd'] <= prior['macd_signal'],
                "is_cross_down": latest['macd'] < latest['macd_signal'] and prior['macd'] >= prior['macd_signal']
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse von {symbol} mit RSI-MACD: {str(e)}")
            return {"signal": "neutral", "error": str(e)}
            
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """
        Gibt die aktuellen Strategie-Parameter zurück
        
        Returns:
            Dictionary mit allen Strategie-Parametern
        """
        return {
            "name": "RSI-MACD",
            "rsi": {
                "length": self.rsi_length,
                "overbought": self.rsi_overbought,
                "oversold": self.rsi_oversold,
                "min_value": self.min_rsi_value,
                "max_value": self.max_rsi_value
            },
            "macd": {
                "fast": self.macd_fast,
                "slow": self.macd_slow,
                "signal": self.macd_signal,
                "threshold": self.macd_threshold
            },
            "ema": {
                "use_filter": self.use_ema_filter,
                "short": self.ema_short_length,
                "long": self.ema_long_length
            },
            "settings": {
                "confirmation_needed": self.confirmation_needed,
                "multi_timeframe_check": self.multi_timeframe_check
            }
        }
        
    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """
        Berechnet den RSI-Indikator
        
        Args:
            df: DataFrame mit Kerzendaten
        """
        # Preisänderungen berechnen
        delta = df['close'].diff()
        
        # Positive und negative Bewegungen trennen
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Durchschnittliche Gewinne und Verluste berechnen (erste Methode)
        avg_gain = gain.rolling(window=self.rsi_length).mean()
        avg_loss = loss.rolling(window=self.rsi_length).mean()
        
        # Relative Strength berechnen
        rs = avg_gain / avg_loss
        
        # RSI berechnen
        df['rsi'] = 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, df: pd.DataFrame) -> None:
        """
        Berechnet den MACD-Indikator
        
        Args:
            df: DataFrame mit Kerzendaten
        """
        # Exponentielle gleitende Durchschnitte berechnen
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        # MACD-Linie = Schneller EMA - Langsamer EMA
        df['macd'] = ema_fast - ema_slow
        
        # Signallinie = EMA der MACD-Linie
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        
        # Histogramm = MACD - Signallinie
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
    def _calculate_ema(self, df: pd.DataFrame) -> None:
        """
        Berechnet die EMA-Indikatoren für Trendfilter
        
        Args:
            df: DataFrame mit Kerzendaten
        """
        # Kurz- und langfristige EMAs berechnen
        df[f'ema_{self.ema_short_length}'] = df['close'].ewm(span=self.ema_short_length, adjust=False).mean()
        df[f'ema_{self.ema_long_length}'] = df['close'].ewm(span=self.ema_long_length, adjust=False).mean()
        
    def _generate_signal(self, df: pd.DataFrame) -> str:
        """
        Generiert ein Handelssignal basierend auf berechneten Indikatoren
        
        Args:
            df: DataFrame mit Kerzen- und Indikatordaten
            
        Returns:
            Signal-String: 'buy', 'sell' oder 'neutral'
        """
        if df.empty or len(df) < 2:
            return "neutral"
            
        # Die letzten zwei Datenpunkte für Kreuzungsanalyse
        latest = df.iloc[-1]
        prior = df.iloc[-2]
        
        # Standardsignal ist neutral
        signal = "neutral"
        
        # RSI-Bedingungen prüfen
        is_oversold = latest['rsi'] <= self.rsi_oversold
        is_overbought = latest['rsi'] >= self.rsi_overbought
        is_extreme_oversold = latest['rsi'] <= self.min_rsi_value
        is_extreme_overbought = latest['rsi'] >= self.max_rsi_value
        
        # MACD-Bedingungen prüfen
        macd_cross_up = latest['macd'] > latest['macd_signal'] and prior['macd'] <= prior['macd_signal']
        macd_cross_down = latest['macd'] < latest['macd_signal'] and prior['macd'] >= prior['macd_signal']
        is_macd_positive = latest['macd'] > self.macd_threshold
        is_macd_negative = latest['macd'] < -self.macd_threshold
        
        # EMA-Trendfilter prüfen
        ema_trend_filter = True
        if self.use_ema_filter:
            is_uptrend = latest[f'ema_{self.ema_short_length}'] > latest[f'ema_{self.ema_long_length}']
            is_downtrend = latest[f'ema_{self.ema_short_length}'] < latest[f'ema_{self.ema_long_length}']
            ema_trend_filter = is_uptrend if is_oversold else (is_downtrend if is_overbought else True)
        
        # Signal-Logik
        if is_oversold or is_extreme_oversold:
            if (macd_cross_up or is_macd_positive) and ema_trend_filter:
                signal = "buy"
                self.logger.info(f"Kaufsignal generiert: RSI={latest['rsi']:.2f} (überkauft), "
                              f"MACD Kreuzung nach oben, EMA-Filter: {ema_trend_filter}")
        
        elif is_overbought or is_extreme_overbought:
            if (macd_cross_down or is_macd_negative) and ema_trend_filter:
                signal = "sell"
                self.logger.info(f"Verkaufssignal generiert: RSI={latest['rsi']:.2f} (überverkauft), "
                              f"MACD Kreuzung nach unten, EMA-Filter: {ema_trend_filter}")
        
        # Starke Signale bei extremen RSI-Werten
        elif is_macd_positive and macd_cross_up and latest['rsi'] < 45:
            signal = "buy"
            self.logger.info(f"Kaufsignal generiert: RSI={latest['rsi']:.2f}, "
                         f"MACD Kreuzung nach oben und positiv")
        
        elif is_macd_negative and macd_cross_down and latest['rsi'] > 55:
            signal = "sell"
            self.logger.info(f"Verkaufssignal generiert: RSI={latest['rsi']:.2f}, "
                         f"MACD Kreuzung nach unten und negativ")
        
        return signal