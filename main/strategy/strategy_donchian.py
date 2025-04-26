import logging
import traceback
import pandas as pd
import numpy as np
import ta
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, SMAIndicator
from typing import Dict, List, Tuple, Optional, Union
from strategy.base_strategy import BaseStrategy
from utils.logger import log_strategy_decision, log_data_load, log_exception, log_api_call

class DonchianChannelStrategy(BaseStrategy):
    """
    Donchian Channel Strategie Implementierung
    
    Eine Donchian Channel Strategie basiert auf dem Donchian Channel Indikator, der ein Preisband 
    darstellt, das aus dem höchsten Hoch und dem niedrigsten Tief über einen bestimmten Zeitraum besteht.
    Die Strategie nutzt Ausbrüche aus diesem Kanal als Trading-Signale.
    """
    
    def __init__(self, api, config: Dict):
        """
        Initialisiert die Donchian Channel Strategie mit API-Zugriff und Konfiguration
        
        Args:
            api: BybitAPI-Instanz für Marktdaten und Order-Ausführung
            config: Konfigurationswörterbuch
        """
        super().__init__(api, config)
        
        # Parameter aus Konfiguration extrahieren
        strategy_params = config.get('strategy', {}).get('parameters', {}).get('donchian_channel', {})
        
        # Donchian Channel Parameter
        self.dc_period = strategy_params.get('dc_period', 20)
        self.breakout_confirmation = strategy_params.get('breakout_confirmation', 2)
        self.trailing_exit = strategy_params.get('trailing_exit', True)
        self.atr_multiplier = strategy_params.get('atr_multiplier', 2.0)
        self.risk_reward_ratio = strategy_params.get('risk_reward_ratio', 2.0)
        
        # Filter Optionen
        self.use_adx_filter = strategy_params.get('use_adx_filter', True)
        self.adx_period = strategy_params.get('adx_period', 14)
        self.adx_threshold = strategy_params.get('adx_threshold', 25)
        self.use_volume_filter = strategy_params.get('use_volume_filter', True)
        self.volume_threshold = strategy_params.get('volume_threshold', 1.25)
        
        # Weitere Parameter
        self.use_middle_channel = strategy_params.get('use_middle_channel', True)
        self.exit_opposite_band = strategy_params.get('exit_opposite_band', False)
        self.trend_filter = strategy_params.get('trend_filter', True)
        self.volatility_filter = strategy_params.get('volatility_filter', True)
        self.consolidation_filter = strategy_params.get('consolidation_filter', True)
        self.consolidation_threshold = strategy_params.get('consolidation_threshold', 0.03)
        
        # Cache-Einstellungen
        self.use_cache = self.cache_enabled  # Von der Basisklasse übernehmen
        
        self.logger.info(f"Donchian Channel Strategie initialisiert mit Periode={self.dc_period}, "
                         f"ADX-Filter={self.use_adx_filter}, Volume-Filter={self.use_volume_filter}")
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Daten vor und berechnet technische Indikatoren
        
        Args:
            df: Rohpreisdaten als DataFrame
            
        Returns:
            DataFrame mit berechneten Indikatoren
        """
        # Sicherstellen, dass genügend Daten für die Berechnungen vorhanden sind
        min_required = max(self.dc_period + 10, 50)
        if len(df) < min_required:
            self.logger.warning(f"Nicht genügend Daten für Donchian Channel Berechnungen. "
                               f"Benötige mindestens {min_required} Kerzen, erhalten: {len(df)}")
            return df
        
        try:
            # Donchian Channel berechnen
            df['dc_upper'] = df['high'].astype(float).rolling(window=self.dc_period).max()
            df['dc_lower'] = df['low'].astype(float).rolling(window=self.dc_period).min()
            
            # Mittleren Kanal berechnen (wenn aktiviert)
            if self.use_middle_channel:
                df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
            
            # ATR für volatilitätsbasierte Stops berechnen
            atr_indicator = AverageTrueRange(
                high=df['high'].astype(float),
                low=df['low'].astype(float),
                close=df['close'].astype(float),
                window=14
            )
            df['atr'] = atr_indicator.average_true_range()
            
            # ADX für Trendstärke berechnen
            if self.use_adx_filter:
                adx_indicator = ADXIndicator(
                    high=df['high'].astype(float),
                    low=df['low'].astype(float),
                    close=df['close'].astype(float),
                    window=self.adx_period
                )
                df['adx'] = adx_indicator.adx()
                df['di_plus'] = adx_indicator.adx_pos()
                df['di_minus'] = adx_indicator.adx_neg()
            
            # Gleitende Durchschnitte berechnen
            ema50_indicator = EMAIndicator(df['close'].astype(float), window=50)
            ema200_indicator = EMAIndicator(df['close'].astype(float), window=200)
            df['ema50'] = ema50_indicator.ema_indicator()
            df['ema200'] = ema200_indicator.ema_indicator()
            df['trend'] = np.where(df['ema50'] > df['ema200'], 1, -1)
            
            # Volumenindikatoren berechnen
            volume_ma_indicator = SMAIndicator(df['volume'].astype(float), window=20)
            df['volume_ma'] = volume_ma_indicator.sma_indicator()
            df['volume_ratio'] = df['volume'].astype(float) / df['volume_ma']
            
            # Kanalbreite für Konsolidierungserkennung berechnen
            df['dc_width'] = df['dc_upper'] - df['dc_lower']
            if self.use_middle_channel:
                df['dc_width_pct'] = df['dc_width'] / df['dc_middle']
                
            # Konsolidierungsindikator - wenn die Kanalbreite kleiner als Schwellenwert ist
            if self.consolidation_filter and self.use_middle_channel:
                df['is_consolidating'] = df['dc_width_pct'] < self.consolidation_threshold
            
            # Volatilität (Kanalbreite in Relation zum Durchschnitt)
            df['volatility'] = df['dc_width'] / df['dc_width'].rolling(window=50).mean()
            
            # Entfernung vom oberen/unteren Kanal zum aktuellen Preis in %
            df['upper_distance'] = (df['dc_upper'] - df['close']) / df['close']
            df['lower_distance'] = (df['close'] - df['dc_lower']) / df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenvorbereitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return df
    
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
            
            # Kerzendaten abrufen (nutzt die Methode aus BaseStrategy)
            df = self.fetch_candles(symbol, interval, limit)
            
            if df.empty:
                self.logger.error(f"Keine Daten verfügbar für {symbol}")
                return {"signal": "neutral", "error": "Keine Daten verfügbar"}
                
            # Verarbeite die Daten und berechne Indikatoren
            df = self.prepare_data(df)
            
            # Handelssignal generieren
            signal, params = self.generate_signal(df)
            
            # Neueste Kerze für Ergebnisse verwenden
            latest = df.iloc[-1]
            
            # Ergebnisse vorbereiten
            result = {
                "symbol": symbol,
                "signal": signal,
                "price": float(latest["close"]),
                "timestamp": int(latest["timestamp"]),
                "dc_upper": float(latest.get("dc_upper", 0)),
                "dc_lower": float(latest.get("dc_lower", 0)),
                "dc_middle": float(latest.get("dc_middle", 0)) if self.use_middle_channel else 0,
                "atr": float(latest.get("atr", 0)),
                "adx": float(latest.get("adx", 0)) if self.use_adx_filter else 0,
                "data_source": df.attrs.get('data_source', 'unknown'),
                "params": params
            }
            
            # Signale zum Loggen vorbereiten
            indicators = {
                "dc_upper": float(latest.get("dc_upper", 0)),
                "dc_lower": float(latest.get("dc_lower", 0)),
                "dc_middle": float(latest.get("dc_middle", 0)) if self.use_middle_channel else 0,
                "atr": float(latest.get("atr", 0)),
                "adx": float(latest.get("adx", 0)) if self.use_adx_filter else 0,
                "di_plus": float(latest.get("di_plus", 0)) if self.use_adx_filter else 0,
                "di_minus": float(latest.get("di_minus", 0)) if self.use_adx_filter else 0,
                "ema50": float(latest.get("ema50", 0)),
                "ema200": float(latest.get("ema200", 0)),
                "volume_ratio": float(latest.get("volume_ratio", 0)),
                "dc_width_pct": float(latest.get("dc_width_pct", 0)) if self.use_middle_channel else 0,
                "volatility": float(latest.get("volatility", 0)),
            }
            
            # Signal-Bedingungen zum Loggen
            signals = {
                "upper_breakout": bool(latest["close"] > latest.get("dc_upper", float('inf'))),
                "lower_breakout": bool(latest["close"] < latest.get("dc_lower", 0)),
                "strong_trend": bool(latest.get("adx", 0) > self.adx_threshold if self.use_adx_filter else False),
                "bullish_trend": bool(latest.get("ema50", 0) > latest.get("ema200", 0)),
                "high_volume": bool(latest.get("volume_ratio", 0) > self.volume_threshold if self.use_volume_filter else False),
                "consolidating": bool(latest.get("is_consolidating", False)) if self.consolidation_filter else False
            }
            
            reason = "Keine Begründung verfügbar"
            if params and "reason" in params:
                reason = params["reason"]
                
            # Logger-Eintrag für die Strategie-Entscheidung
            try:
                log_strategy_decision(
                    self.logger,
                    symbol=symbol, 
                    timeframe=f"{interval}m",
                    decision=signal,
                    signals=signals,
                    indicators=indicators,
                    reason=reason
                )
            except Exception as log_error:
                self.logger.error(f"Fehler beim Loggen der Strategie-Entscheidung: {str(log_error)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Fehler bei der Analyse von {symbol}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return {"signal": "neutral", "error": str(e)}
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, Optional[Dict]]:
        """
        Generiert ein Handelssignal basierend auf den vorbereiteten Daten
        
        Args:
            df: Vorbereiteter DataFrame mit Indikatoren
            
        Returns:
            Tupel aus (signal, parameters)
        """
        # Prüfen, ob genügend Daten vorhanden sind
        if len(df) < self.dc_period + self.breakout_confirmation:
            return ('neutral', None)
        
        # Neueste Kerzen extrahieren
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else last_candle
        
        # Signal initialisieren
        signal = 'neutral'
        params = None
        
        # Notwendige Werte vorhanden?
        if 'dc_upper' not in last_candle or 'dc_lower' not in last_candle:
            return ('neutral', {"reason": "Fehlende Donchian Channel Daten"})
        
        # 1. Oberer Kanal Ausbruch (Kaufsignal)
        upper_breakout = self._check_breakout(df, 'upper')
        
        # 2. ADX Filter (Trendstärke)
        adx_condition = True
        if self.use_adx_filter and 'adx' in last_candle:
            adx_condition = last_candle['adx'] > self.adx_threshold
            # Zusätzlich: Prüfe die Richtung der DI-Linien für Bestätigung
            if 'di_plus' in last_candle and 'di_minus' in last_candle:
                adx_bullish = last_candle['di_plus'] > last_candle['di_minus']
                adx_bearish = last_candle['di_minus'] > last_candle['di_plus']
        
        # 3. Volumen Filter
        volume_condition = True
        if self.use_volume_filter and 'volume_ratio' in last_candle:
            volume_condition = last_candle['volume_ratio'] > self.volume_threshold
        
        # 4. Trendfilter (EMA50 > EMA200 für bullischen Trend)
        trend_condition = True
        if self.trend_filter and 'trend' in last_candle:
            trend_bullish = last_candle.get('trend', 0) == 1
            trend_bearish = last_candle.get('trend', 0) == -1
        
        # 5. Volatilitätsfilter
        volatility_condition = True
        if self.volatility_filter and 'volatility' in last_candle:
            volatility_condition = last_candle['volatility'] > 0.8
        
        # 6. Konsolidierungsfilter
        consolidation_condition = True
        if self.consolidation_filter and 'is_consolidating' in last_candle:
            # Nur traden wenn wir NICHT in Konsolidierung sind
            consolidation_condition = not last_candle['is_consolidating']
        
        # Kaufsignal generieren
        if (upper_breakout and 
            adx_condition and 
            volume_condition and
            trend_condition and
            volatility_condition and
            consolidation_condition):
            
            if not self.trend_filter or (self.trend_filter and trend_bullish):
                signal = 'buy'
                entry_price = float(last_candle['close'])
                
                # Stop-Loss basierend auf ATR oder einem festen Prozentsatz des Einstiegspreises berechnen
                atr_value = float(last_candle.get('atr', entry_price * 0.02))
                stop_loss = entry_price - (atr_value * self.atr_multiplier)
                
                # Take Profit basierend auf Kanaleigenschaften
                if self.exit_opposite_band:
                    take_profit = float(last_candle.get('dc_lower', entry_price * 0.9))
                else:
                    # Risiko-Rendite basierter Take Profit
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.risk_reward_ratio)
                
                # Gründe für das Signal sammeln
                reasons = []
                reasons.append("Oberer Kanal Ausbruch")
                if self.trend_filter and trend_bullish:
                    reasons.append("Bullischer Trend (EMA50 > EMA200)")
                if self.use_adx_filter and adx_condition:
                    reasons.append(f"Starker Trend (ADX > {self.adx_threshold})")
                if self.use_volume_filter and volume_condition:
                    reasons.append("Erhöhtes Handelsvolumen")
                
                params = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': " mit ".join(reasons)
                }
        
        # 1. Unterer Kanal Ausbruch (Verkaufssignal)
        lower_breakout = self._check_breakout(df, 'lower')
        
        # Verkaufssignal generieren
        if (lower_breakout and 
            adx_condition and 
            volume_condition and
            trend_condition and
            volatility_condition and
            consolidation_condition):
            
            if not self.trend_filter or (self.trend_filter and trend_bearish):
                signal = 'sell'
                entry_price = float(last_candle['close'])
                
                # Stop-Loss basierend auf ATR oder einem festen Prozentsatz des Einstiegspreises berechnen
                atr_value = float(last_candle.get('atr', entry_price * 0.02))
                stop_loss = entry_price + (atr_value * self.atr_multiplier)
                
                # Take Profit basierend auf Kanaleigenschaften
                if self.exit_opposite_band:
                    take_profit = float(last_candle.get('dc_upper', entry_price * 1.1))
                else:
                    # Risiko-Rendite basierter Take Profit
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.risk_reward_ratio)
                
                # Gründe für das Signal sammeln
                reasons = []
                reasons.append("Unterer Kanal Ausbruch")
                if self.trend_filter and trend_bearish:
                    reasons.append("Bärischer Trend (EMA50 < EMA200)")
                if self.use_adx_filter and adx_condition:
                    reasons.append(f"Starker Trend (ADX > {self.adx_threshold})")
                if self.use_volume_filter and volume_condition:
                    reasons.append("Erhöhtes Handelsvolumen")
                
                params = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': " mit ".join(reasons)
                }
        
        return (signal, params)
        
    def _check_breakout(self, df: pd.DataFrame, breakout_type: str) -> bool:
        """
        Prüft, ob ein Ausbruch aus dem oberen oder unteren Kanal vorliegt
        
        Args:
            df: Preisdaten mit Donchian-Indikatoren
            breakout_type: 'upper' für oberen oder 'lower' für unteren Kanalausbruch
            
        Returns:
            True wenn ein Ausbruch bestätigt wurde, sonst False
        """
        if len(df) < self.breakout_confirmation + 1:
            return False
            
        # Prüfe auf Ausbruchsbestätigung über mehrere Kerzen
        confirmed = False
        
        if breakout_type == 'upper':
            # Für oberen Kanal: Schlusskurs > dc_upper
            for i in range(1, self.breakout_confirmation + 1):
                if df.iloc[-i]['close'] > df.iloc[-i].get('dc_upper', float('inf')):
                    confirmed = True
                    break
        elif breakout_type == 'lower':
            # Für unteren Kanal: Schlusskurs < dc_lower
            for i in range(1, self.breakout_confirmation + 1):
                if df.iloc[-i]['close'] < df.iloc[-i].get('dc_lower', 0):
                    confirmed = True
                    break
                    
        return confirmed
        
    def get_strategy_parameters(self) -> Dict:
        """
        Gibt die Parameter der Strategie zurück
        
        Returns:
            Dictionary mit Strategie-Parametern
        """
        return {
            'name': self.name,
            'dc_period': self.dc_period,
            'breakout_confirmation': self.breakout_confirmation,
            'adx_filter': self.use_adx_filter,
            'adx_threshold': self.adx_threshold,
            'volume_filter': self.use_volume_filter,
            'volume_threshold': self.volume_threshold,
            'trend_filter': self.trend_filter,
            'middle_channel': self.use_middle_channel,
            'trailing_exit': self.trailing_exit,
            'atr_multiplier': self.atr_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio,
            'consolidation_filter': self.consolidation_filter,
            'consolidation_threshold': self.consolidation_threshold
        }