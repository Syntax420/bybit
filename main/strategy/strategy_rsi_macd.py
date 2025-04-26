import logging
import traceback
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional

from api.bybit_api import BybitAPI
from utils.logger import log_strategy_decision, log_exception
from strategy.base_strategy import BaseStrategy

class RSIMACDStrategy(BaseStrategy):
    """
    RSI und MACD basierte Trading-Strategie.
    
    Diese Strategie kombiniert den Relative Strength Index (RSI) und Moving Average Convergence/Divergence (MACD),
    sowie Exponential Moving Averages (EMA) zur Signalgenerierung.
    """
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialisiert die RSI-MACD-Strategie
        
        Args:
            api: BybitAPI-Instanz für Marktdaten und Order-Ausführung
            config: Konfigurationswörterbuch
        """
        super().__init__(api, config)
        
        # Lade Strategie-Parameter aus der Konfiguration
        strategy_params = config.get("strategies", {}).get("rsi_macd", {})
        
        # RSI-Parameter
        self.rsi_period = strategy_params.get("rsi_period", 14)
        self.rsi_overbought = strategy_params.get("rsi_overbought", 70)
        self.rsi_oversold = strategy_params.get("rsi_oversold", 30)
        
        # MACD-Parameter
        self.macd_fast_period = strategy_params.get("macd_fast_period", 12)
        self.macd_slow_period = strategy_params.get("macd_slow_period", 26)
        self.macd_signal_period = strategy_params.get("macd_signal_period", 9)
        
        # EMA-Parameter
        self.ema_short_period = strategy_params.get("ema_short_period", 9)
        self.ema_long_period = strategy_params.get("ema_long_period", 21)
        
        self.logger.info(f"RSI-MACD Strategie initialisiert mit Parametern: RSI({self.rsi_period}/{self.rsi_oversold}/{self.rsi_overbought}), " +
                      f"MACD({self.macd_fast_period}/{self.macd_slow_period}/{self.macd_signal_period}), EMA({self.ema_short_period}/{self.ema_long_period})")
        
    def calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Berechnet den Relative Strength Index
        
        Args:
            df: DataFrame mit Preisdaten
            period: RSI-Periodenlänge (optional)
            
        Returns:
            pandas Series mit RSI-Werten
        """
        try:
            if period is None:
                period = self.rsi_period
                
            # Ensure we have the 'close' column
            if 'close' not in df.columns:
                # Try to find an alternative if the column exists with a different case
                for col in df.columns:
                    if col.lower() == 'close':
                        df['close'] = df[col]
                        break
                else:
                    self.logger.error(f"Column 'close' not found in DataFrame. Available columns: {list(df.columns)}")
                    return pd.Series(index=df.index)
                
            # Preisänderungen berechnen
            delta = df['close'].astype(float).diff()
            
            # Auf- und Abwärtsbewegungen aufteilen
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            
            # Durchschnittliche Gewinne und Verluste berechnen
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Relative Strength berechnen
            rs = avg_gain / avg_loss
            
            # RSI berechnen
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Fehler bei der RSI-Berechnung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.Series(index=df.index)
        
    def calculate_macd(self, df: pd.DataFrame, 
                       fast_period: int = None, 
                       slow_period: int = None,
                       signal_period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Berechnet den Moving Average Convergence Divergence
        
        Args:
            df: DataFrame mit Preisdaten
            fast_period: Periode für schnellen EMA
            slow_period: Periode für langsamen EMA
            signal_period: Periode für Signal-Linie
            
        Returns:
            Tuple aus (MACD-Linie, Signal-Linie, Histogramm)
        """
        try:
            if fast_period is None:
                fast_period = self.macd_fast_period
            if slow_period is None:
                slow_period = self.macd_slow_period
            if signal_period is None:
                signal_period = self.macd_signal_period
                
            # Ensure we have the 'close' column
            if 'close' not in df.columns:
                # Try to find an alternative if the column exists with a different case
                for col in df.columns:
                    if col.lower() == 'close':
                        df['close'] = df[col]
                        break
                else:
                    self.logger.error(f"Column 'close' not found in DataFrame. Available columns: {list(df.columns)}")
                    return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)
                
            # Exponential Moving Averages berechnen
            ema_fast = df['close'].astype(float).ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].astype(float).ewm(span=slow_period, adjust=False).mean()
            
            # MACD-Linie berechnen
            macd_line = ema_fast - ema_slow
            
            # Signal-Linie berechnen
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Histogramm berechnen
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"Fehler bei der MACD-Berechnung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)
        
    def calculate_ema(self, df: pd.DataFrame, 
                    short_period: int = None, 
                    long_period: int = None) -> Tuple[pd.Series, pd.Series]:
        """
        Berechnet Exponential Moving Averages
        
        Args:
            df: DataFrame mit Preisdaten
            short_period: Periode für kurzfristigen EMA
            long_period: Periode für langfristigen EMA
            
        Returns:
            Tuple aus (kurzfristiger EMA, langfristiger EMA)
        """
        try:
            if short_period is None:
                short_period = self.ema_short_period
            if long_period is None:
                long_period = self.ema_long_period
                
            # Ensure we have the 'close' column
            if 'close' not in df.columns:
                # Try to find an alternative if the column exists with a different case
                for col in df.columns:
                    if col.lower() == 'close':
                        df['close'] = df[col]
                        break
                else:
                    self.logger.error(f"Column 'close' not found in DataFrame. Available columns: {list(df.columns)}")
                    return pd.Series(index=df.index), pd.Series(index=df.index)
                
            # EMAs berechnen
            ema_short = df['close'].astype(float).ewm(span=short_period, adjust=False).mean()
            ema_long = df['close'].astype(float).ewm(span=long_period, adjust=False).mean()
            
            return ema_short, ema_long
            
        except Exception as e:
            self.logger.error(f"Fehler bei der EMA-Berechnung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.Series(index=df.index), pd.Series(index=df.index)
        
    def analyze(self, symbol: str, interval: str = "15", limit: int = 100) -> Dict:
        """
        Analysiert ein Symbol und generiert Handelssignale
        
        Args:
            symbol: Trading-Paar-Symbol
            interval: Zeitrahmen-Intervall
            limit: Anzahl der zu analysierenden Kerzen
            
        Returns:
            Dictionary mit Analyseergebnissen und Signalen
        """
        try:
            self.logger.debug(f"Starte Analyse von {symbol} ({interval}m)")
            
            # Kerzendaten abrufen mit der Basismethode
            df = self.fetch_candles(symbol, interval, limit)
            
            # Prüfen, ob Daten vorhanden sind
            if df.empty:
                return {"signal": "neutral", "error": "Keine Daten verfügbar", "data_source": "none"}
                
            # Prüfen, ob genügend Kerzen für die Analyse vorhanden sind
            if len(df) < 30:  # Mindestens 30 Kerzen für zuverlässige Indikatoren
                self.logger.warning(f"Nicht genügend Kerzen für zuverlässige Analyse von {symbol}: {len(df)} < 30")
                return {"signal": "neutral", "error": "Unzureichende Daten", "data_source": df.attrs.get('data_source', 'unknown')}
            
            # Berechne technische Indikatoren
            self.logger.debug(f"Berechne technische Indikatoren für {symbol}")
            rsi_series = self.calculate_rsi(df)
            macd_line, signal_line, histogram = self.calculate_macd(df)
            ema_short, ema_long = self.calculate_ema(df)
            
            # Füge berechnete Indikatoren zum DataFrame hinzu
            df['rsi'] = rsi_series
            df['macd_line'] = macd_line
            df['signal_line'] = signal_line
            df['histogram'] = histogram
            df['ema_short'] = ema_short
            df['ema_long'] = ema_long
            
            # Prüfe, ob Indikatoren erfolgreich berechnet wurden
            indicator_columns = ["rsi", "macd_line", "signal_line", "histogram", "ema_short", "ema_long"]
            missing_indicators = [col for col in indicator_columns if col not in df.columns]
            
            if missing_indicators:
                error_msg = f"Fehler bei der Indikatorberechnung: {', '.join(missing_indicators)}"
                self.logger.error(error_msg)
                return {"signal": "neutral", "error": error_msg, "data_source": df.attrs.get('data_source', 'unknown')}
            
            # Hole die neueste Kerze für die Signalgenerierung
            if len(df) >= 2:
                latest = df.iloc[-1]
                previous = df.iloc[-2]
            else:
                latest = df.iloc[-1]
                previous = None
                self.logger.warning(f"Nur eine Kerze verfügbar für {symbol}, Signal kann unzuverlässig sein")
            
            # Generiere Handelssignal
            signal, reason = self.generate_signal(
                latest["rsi"], previous["rsi"] if previous is not None else None,
                latest["macd_line"], previous["macd_line"] if previous is not None else None,
                latest["signal_line"], previous["signal_line"] if previous is not None else None,
                latest["histogram"], previous["histogram"] if previous is not None else None,
                latest["ema_short"], previous["ema_short"] if previous is not None else None,
                latest["ema_long"], previous["ema_long"] if previous is not None else None
            )
            
            # Ergebnisse vorbereiten
            result = {
                "symbol": symbol,
                "signal": signal,
                "price": float(latest["close"]),
                "timestamp": int(latest["timestamp"]),
                "rsi": float(latest["rsi"]),
                "rsi_prev": float(previous["rsi"]) if previous is not None else None,
                "macd_line": float(latest["macd_line"]),
                "macd_line_prev": float(previous["macd_line"]) if previous is not None else None,
                "signal_line": float(latest["signal_line"]),
                "signal_line_prev": float(previous["signal_line"]) if previous is not None else None,
                "histogram": float(latest["histogram"]),
                "histogram_prev": float(previous["histogram"]) if previous is not None else None,
                "ema_short": float(latest["ema_short"]),
                "ema_short_prev": float(previous["ema_short"]) if previous is not None else None,
                "ema_long": float(latest["ema_long"]),
                "ema_long_prev": float(previous["ema_long"]) if previous is not None else None,
                "data_source": df.attrs.get('data_source', 'unknown'),
                "reason": reason
            }
            
            # Protokolliere die Strategieentscheidung
            indicators = {
                "rsi": float(latest["rsi"]),
                "macd_line": float(latest["macd_line"]),
                "signal_line": float(latest["signal_line"]),
                "histogram": float(latest["histogram"]),
                "ema_short": float(latest["ema_short"]),
                "ema_long": float(latest["ema_long"])
            }
            
            signals = {
                "rsi_oversold": latest["rsi"] < self.rsi_oversold,
                "rsi_overbought": latest["rsi"] > self.rsi_overbought,
                "macd_crossover": latest["macd_line"] > latest["signal_line"] and (previous is None or previous["macd_line"] <= previous["signal_line"]),
                "macd_crossunder": latest["macd_line"] < latest["signal_line"] and (previous is None or previous["macd_line"] >= previous["signal_line"]),
                "histogram_increasing": previous is not None and latest["histogram"] > previous["histogram"],
                "histogram_decreasing": previous is not None and latest["histogram"] < previous["histogram"],
                "price_above_ema": latest["close"] > latest["ema_long"],
                "price_below_ema": latest["close"] < latest["ema_long"]
            }
            
            log_strategy_decision(
                self.logger,
                symbol=symbol, 
                timeframe=f"{interval}m",
                decision=signal,
                signals=signals,
                indicators=indicators,
                reason=reason
            )
            
            self.logger.info(f"Strategie {signal.upper()} Signal für {symbol} ({interval}m): {reason}")
            return result
            
        except Exception as e:
            error_msg = f"Fehler bei der Analyse von {symbol}: {str(e)}"
            self.logger.error(error_msg)
            # Stelle sicher, dass der Traceback korrekt verwendet wird
            try:
                log_exception(self.logger, e, f"Strategieanalyse für {symbol}", traceback.format_exc())
            except Exception as log_error:
                self.logger.error(f"Fehler beim Protokollieren einer Ausnahme: {log_error}")
            return {"signal": "neutral", "error": str(e), "data_source": "error"}
    
    def generate_signal(self, 
                       latest_rsi, prev_rsi,
                       latest_macd, prev_macd,
                       latest_signal, prev_signal,
                       latest_histogram, prev_histogram,
                       latest_ema_short, prev_ema_short,
                       latest_ema_long, prev_ema_long) -> Tuple[str, str]:
        """
        Generiert Handelssignal basierend auf technischen Indikatoren
        
        Args:
            latest_rsi: Neueste RSI-Wert
            prev_rsi: Vorheriger RSI-Wert
            latest_macd: Neueste MACD-Linie
            prev_macd: Vorheriger MACD-Linie
            latest_signal: Neueste Signal-Linie
            prev_signal: Vorheriger Signal-Linie
            latest_histogram: Neuestes Histogramm
            prev_histogram: Vorheriges Histogramm
            latest_ema_short: Neueste kurzfristige EMA
            prev_ema_short: Vorherige kurzfristige EMA
            latest_ema_long: Neueste langfristige EMA
            prev_ema_long: Vorherige langfristige EMA
            
        Returns:
            Tuple aus (signal, reason), wobei signal "buy", "sell" oder "neutral" ist
        """
        try:
            # Standard-Signal
            signal = "neutral"
            reason = "Kein klares Signal"
            
            # Indikatorwerte extrahieren
            rsi = latest_rsi
            macd_line = latest_macd
            signal_line = latest_signal
            histogram = latest_histogram
            ema_short = latest_ema_short
            ema_long = latest_ema_long
            
            # Vorherige Werte, falls verfügbar
            prev_macd_line = prev_macd if prev_macd is not None else 0
            prev_signal_line = prev_signal if prev_signal is not None else 0
            prev_histogram = prev_histogram if prev_histogram is not None else 0
            prev_ema_short = prev_ema_short if prev_ema_short is not None else 0
            prev_ema_long = prev_ema_long if prev_ema_long is not None else 0
            
            # KAUF-BEDINGUNGEN
            if (
                # RSI überkauft/überverkauft Bedingung
                rsi < self.rsi_oversold and
                # MACD kreuzt über Signallinie (oder ist kurz davor)
                (macd_line > signal_line or (prev_macd_line > prev_signal_line)) and
                # MACD-Histogramm steigt
                (histogram > 0 or (prev_histogram is not None and histogram > prev_histogram)) and
                # Preis ist über EMA für Trendbestätigung
                ema_short > ema_long
            ):
                signal = "buy"
                reason = f"RSI überverkauft ({rsi:.2f}) mit positivem MACD-Momentum"
                self.logger.info(f"KAUF-Signal generiert: RSI({rsi:.2f}) überverkauft, MACD-Histogramm steigend")
                
            # VERKAUF-BEDINGUNGEN
            elif (
                # RSI überkauft/überverkauft Bedingung
                rsi > self.rsi_overbought and
                # MACD kreuzt unter Signallinie (oder ist kurz davor)
                (macd_line < signal_line or (prev_macd_line < prev_signal_line)) and
                # MACD-Histogramm fällt
                (histogram < 0 or (prev_histogram is not None and histogram < prev_histogram)) and
                # Preis ist unter EMA für Trendbestätigung
                ema_short < ema_long
            ):
                signal = "sell"
                reason = f"RSI überkauft ({rsi:.2f}) mit negativem MACD-Momentum"
                self.logger.info(f"VERKAUF-Signal generiert: RSI({rsi:.2f}) überkauft, MACD-Histogramm fallend")
                
            # ALTERNATIVER KAUF - MACD primär mit RSI-Bestätigung
            elif (
                # MACD kreuzt über Signal
                (prev_macd_line is not None and prev_signal_line is not None and
                prev_macd_line < prev_signal_line and 
                macd_line > signal_line) and
                # RSI ist nicht überkauft
                rsi < 65 and
                # Aufwärtstrend bestätigt durch EMAs
                ema_short > ema_long
            ):
                signal = "buy"
                reason = f"MACD bullisches Crossover mit RSI({rsi:.2f}) nicht überkauft"
                self.logger.info(f"KAUF-Signal generiert: MACD kreuzte über Signallinie, RSI {rsi:.2f}")
                
            # ALTERNATIVER VERKAUF - MACD primär mit RSI-Bestätigung
            elif (
                # MACD kreuzt unter Signal
                (prev_macd_line is not None and prev_signal_line is not None and
                prev_macd_line > prev_signal_line and 
                macd_line < signal_line) and
                # RSI ist nicht überverkauft
                rsi > 35 and
                # Abwärtstrend bestätigt durch EMAs
                ema_short < ema_long
            ):
                signal = "sell"
                reason = f"MACD bärisches Crossover mit RSI({rsi:.2f}) nicht überverkauft"
                self.logger.info(f"VERKAUF-Signal generiert: MACD kreuzte unter Signallinie, RSI {rsi:.2f}")
            
            return signal, reason
        except Exception as e:
            self.logger.error(f"Fehler bei der Signalgenerierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "neutral", f"Fehler bei der Signalgenerierung: {str(e)}"
    
    def get_strategy_parameters(self) -> Dict:
        """
        Liefert die Parameter der RSI-MACD-Strategie
        
        Returns:
            Dictionary mit Strategieparametern
        """
        return {
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "macd_fast_period": self.macd_fast_period,
            "macd_slow_period": self.macd_slow_period,
            "macd_signal_period": self.macd_signal_period,
            "ema_short_period": self.ema_short_period,
            "ema_long_period": self.ema_long_period
        }
