import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

from api.bybit_api import BybitAPI

class RiskManager:
    """
    Risikomanagement-Modul für den Trading Bot.
    
    Verantwortlich für:
    - Berechnung der optimalen Positionsgröße
    - Stop-Loss und Take-Profit-Berechnung
    - Überwachung des Portfolio-Risikos
    - Hebelwirkung und Marge-Kontrolle
    """
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialisiert den Risikomanager
        
        Args:
            api: BybitAPI-Instanz für Konto- und Positionsinformationen
            config: Konfigurationswörterbuch
        """
        self.api = api
        self.config = config
        self.logger = logging.getLogger("risk.manager")
        
        # Parameter aus Konfiguration laden
        self.risk_per_trade_percent = config.get("trading", {}).get("risk_per_trade_percent", 1.0)
        self.max_positions = config.get("trading", {}).get("max_positions", 5)
        self.default_leverage = config.get("trading", {}).get("default_leverage", 3)
        self.take_profit_percent = config.get("trading", {}).get("take_profit_percent", 3.0)
        self.stop_loss_percent = config.get("trading", {}).get("stop_loss_percent", 2.0)
        self.use_trailing_stop = config.get("trading", {}).get("use_trailing_stop", True)
        self.max_portfolio_risk = config.get("trading", {}).get("max_portfolio_risk_percent", 5.0)
        
        # Optimierte Parameter für Pyramiding und Position-Sizing
        self.pyramiding_enabled = config.get("trading", {}).get("pyramiding", {}).get("enabled", False)
        self.max_pyramiding_positions = config.get("trading", {}).get("pyramiding", {}).get("max_positions", 3)
        self.pyramiding_scale_factor = config.get("trading", {}).get("pyramiding", {}).get("scale_factor", 0.7)
        
        # Dynamische Stop-Loss-Anpassung
        self.dynamic_sl_atr_periods = config.get("trading", {}).get("sl_tp", {}).get("atr_periods", 14)
        self.dynamic_sl_atr_multiplier = config.get("trading", {}).get("sl_tp", {}).get("atr_multiplier", 2.0)
        
        # Tracking laufender Positionen
        self.open_positions = {}
        self.position_high_prices = {}
        self.position_low_prices = {}
        
        self.logger.info(f"Risikomanager initialisiert: {self.risk_per_trade_percent}% Risiko pro Trade, "
                         f"max. {self.max_positions} Positionen, SL: {self.stop_loss_percent}%, TP: {self.take_profit_percent}%")

    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, balance: float = None) -> Dict[str, Any]:
        """
        Berechnet die optimale Positionsgröße basierend auf Risikomanagement-Parametern
        
        Args:
            symbol: Trading-Paar-Symbol
            entry_price: Geplanter Einstiegspreis
            stop_loss_price: Geplanter Stop-Loss-Preis
            balance: Verfügbares Kapital (oder None für automatische Abfrage)
            
        Returns:
            Dictionary mit Positionsgrößen-Informationen
        """
        try:
            # Aktuelles Kapital abrufen, falls nicht explizit angegeben
            if balance is None:
                account_info = self.api.get_wallet_balance()
                if account_info and 'result' in account_info:
                    balance = float(account_info['result'].get('USDT', {}).get('available_balance', 0))
                else:
                    self.logger.error("Konnte Kontostand nicht abrufen, verwende Standardwert")
                    balance = 1000.0
            
            # Risikobetrag berechnen (in USDT)
            risk_amount = balance * (self.risk_per_trade_percent / 100)
            
            # Prüfen, ob das Gesamtportfoliorisiko überschritten würde
            portfolio_risk = self._calculate_portfolio_risk()
            remaining_risk = self.max_portfolio_risk - portfolio_risk
            
            if self.risk_per_trade_percent > remaining_risk:
                self.logger.warning(f"Portfoliorisiko würde mit diesem Trade überschritten werden. "
                                  f"Aktuell: {portfolio_risk:.2f}%, Max: {self.max_portfolio_risk}%")
                risk_amount = balance * (remaining_risk / 100)
                adjusted_risk_percent = remaining_risk
            else:
                adjusted_risk_percent = self.risk_per_trade_percent
            
            # Abstand zwischen Einstieg und Stop-Loss berechnen
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit <= 0:
                self.logger.error(f"Ungültiger Stop-Loss für {symbol}: Einstieg {entry_price}, Stop-Loss {stop_loss_price}")
                return {
                    "position_size": 0,
                    "contracts": 0,
                    "risk_amount": 0,
                    "risk_percent": 0,
                    "error": "Ungültiger Stop-Loss"
                }
            
            # Maximale Anzahl von Einheiten, die wir mit unserem Risiko kaufen können
            max_units = risk_amount / risk_per_unit
            
            # Positionsgröße in USD berechnen
            position_size = max_units * entry_price
            
            # Anzahl der Kontrakte (gerundet)
            tick_size = self._get_tick_size(symbol)
            contracts = self._round_to_tick(max_units, tick_size)
            
            # Risiko neu berechnen (basierend auf gerundeter Kontraktzahl)
            actual_risk_amount = contracts * risk_per_unit
            actual_risk_percent = (actual_risk_amount / balance) * 100
            
            result = {
                "symbol": symbol,
                "position_size": position_size,
                "contracts": contracts,
                "risk_amount": actual_risk_amount,
                "risk_percent": actual_risk_percent,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "balance": balance
            }
            
            self.logger.info(f"Positionsgröße für {symbol}: {contracts} Kontrakte "
                           f"({position_size:.2f} USD), Risiko: {actual_risk_amount:.2f} USD ({actual_risk_percent:.2f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Positionsgrößenberechnung für {symbol}: {str(e)}")
            return {
                "position_size": 0,
                "contracts": 0,
                "risk_amount": 0,
                "risk_percent": 0,
                "error": str(e)
            }
            
    def calculate_exit_prices(self, entry_price: float, signal_type: str, 
                             volatility: float = None, symbol: str = None) -> Dict[str, float]:
        """
        Berechnet optimale Stop-Loss- und Take-Profit-Werte
        
        Args:
            entry_price: Einstiegspreis der Position
            signal_type: Signaltyp ('buy' oder 'sell')
            volatility: Optionale Volatilität des Assets (als Prozent)
            symbol: Symbol für dynamische Volatilitätsberechnung
            
        Returns:
            Dictionary mit Exit-Preisen
        """
        try:
            # Standard Stop-Loss und Take-Profit aus Konfiguration
            stop_loss_percent = self.stop_loss_percent
            take_profit_percent = self.take_profit_percent
            
            # Bei hoher Volatilität Stop-Loss anpassen
            if volatility is not None and volatility > 0:
                # Dynamische Anpassung basierend auf Volatilität
                volatility_factor = min(volatility / 2.0, 2.5)  # Begrenze den Faktor
                stop_loss_percent = self.stop_loss_percent * volatility_factor
                
                # Take-Profit entsprechend anpassen
                take_profit_percent = stop_loss_percent * (self.take_profit_percent / self.stop_loss_percent)
                
                self.logger.info(f"Stop-Loss und Take-Profit angepasst für Volatilität ({volatility:.2f}%): "
                               f"SL {stop_loss_percent:.2f}%, TP {take_profit_percent:.2f}%")
            
            # Exit-Preise berechnen
            if signal_type.lower() == 'buy':
                stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
                take_profit_price = entry_price * (1 + take_profit_percent / 100)
            else:  # sell
                stop_loss_price = entry_price * (1 + stop_loss_percent / 100)
                take_profit_price = entry_price * (1 - take_profit_percent / 100)
            
            # Trailing-Stop aktivieren, wenn konfiguriert
            trailing_active = self.use_trailing_stop
            trailing_callback = stop_loss_percent / 2  # Standard: Hälfte des Stop-Loss
            
            result = {
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "trailing_active": trailing_active,
                "trailing_callback": trailing_callback,
                "stop_loss_percent": stop_loss_percent,
                "take_profit_percent": take_profit_percent
            }
            
            self.logger.info(f"Exit-Preise berechnet für {symbol or 'Position'} ({signal_type}): "
                           f"SL {stop_loss_price:.6f}, TP {take_profit_price:.6f}")
                           
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung der Exit-Preise: {str(e)}")
            # Fallback auf einfache Prozentberechnung
            if signal_type.lower() == 'buy':
                sl = entry_price * (1 - self.stop_loss_percent / 100)
                tp = entry_price * (1 + self.take_profit_percent / 100)
            else:
                sl = entry_price * (1 + self.stop_loss_percent / 100)
                tp = entry_price * (1 - self.take_profit_percent / 100)
                
            return {
                "stop_loss": sl,
                "take_profit": tp,
                "trailing_active": self.use_trailing_stop,
                "trailing_callback": self.stop_loss_percent / 2,
                "stop_loss_percent": self.stop_loss_percent,
                "take_profit_percent": self.take_profit_percent
            }
    
    def update_position_tracking(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """
        Aktualisiert die Tracking-Informationen für eine Position
        
        Args:
            symbol: Symbol der Position
            position_data: Positionsdaten
        """
        try:
            self.open_positions[symbol] = position_data
            
            current_price = position_data.get("price", 0)
            if current_price > 0:
                # Höchst-/Tiefstwerte aktualisieren
                if symbol not in self.position_high_prices or current_price > self.position_high_prices[symbol]:
                    self.position_high_prices[symbol] = current_price
                    
                if symbol not in self.position_low_prices or current_price < self.position_low_prices[symbol]:
                    self.position_low_prices[symbol] = current_price
                    
            self.logger.debug(f"Position-Tracking aktualisiert für {symbol}: "
                            f"Höchstwert {self.position_high_prices.get(symbol, 0):.6f}, "
                            f"Tiefstwert {self.position_low_prices.get(symbol, 0):.6f}")
                            
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des Position-Trackings für {symbol}: {str(e)}")
            
    def close_position(self, symbol: str) -> None:
        """
        Bereinigt die Tracking-Daten nach dem Schließen einer Position
        
        Args:
            symbol: Symbol der Position
        """
        try:
            if symbol in self.open_positions:
                del self.open_positions[symbol]
                
            if symbol in self.position_high_prices:
                del self.position_high_prices[symbol]
                
            if symbol in self.position_low_prices:
                del self.position_low_prices[symbol]
                
            self.logger.info(f"Position-Tracking bereinigt für {symbol}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Bereinigen des Position-Trackings für {symbol}: {str(e)}")
            
    def get_account_balance(self) -> float:
        """
        Ruft den aktuellen Kontostand ab (für Paper Trading)
        
        Returns:
            Aktueller Kontostand
        """
        try:
            # Für Paper Trading verwenden wir einen konfigurierten Wert oder Standardwert
            if hasattr(self, 'paper_balance'):
                return self.paper_balance
            
            # Initialen Kontostand aus Konfiguration laden oder Standardwert verwenden
            initial_balance = self.config.get("paper_trading", {}).get("initial_balance", 10000.0)
            self.paper_balance = initial_balance
            self.starting_balance = initial_balance  # Für Performance-Tracking
            
            self.logger.info(f"Paper Trading Kontostand initialisiert: {initial_balance} USDT")
            return initial_balance
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Paper-Trading-Kontostands: {str(e)}")
            return 10000.0  # Fallback-Wert
    
    def set_account_balance(self, new_balance: float) -> None:
        """
        Setzt den aktuellen Kontostand (für Paper Trading)
        
        Args:
            new_balance: Neuer Kontostand
        """
        try:
            self.paper_balance = new_balance
            self.logger.debug(f"Paper Trading Kontostand aktualisiert: {new_balance} USDT")
        except Exception as e:
            self.logger.error(f"Fehler beim Setzen des Paper-Trading-Kontostands: {str(e)}")
            
    def check_portfolio_risk(self) -> Dict[str, Any]:
        """
        Überprüft das aktuelle Portfoliorisiko
        
        Returns:
            Risikobericht (Gesamtrisiko, Anzahl Positionen, etc.)
        """
        try:
            total_risk = self._calculate_portfolio_risk()
            positions_count = len(self.open_positions)
            
            risk_level = "niedrig"
            if total_risk > self.max_portfolio_risk * 0.7:
                risk_level = "hoch"
            elif total_risk > self.max_portfolio_risk * 0.5:
                risk_level = "mittel"
                
            report = {
                "total_risk_percent": total_risk,
                "max_risk_percent": self.max_portfolio_risk,
                "positions_count": positions_count,
                "max_positions": self.max_positions,
                "positions_capacity": self.max_positions - positions_count,
                "risk_level": risk_level,
                "can_open_position": total_risk < self.max_portfolio_risk and positions_count < self.max_positions,
                "positions": list(self.open_positions.keys())
            }
            
            self.logger.info(f"Portfolio-Risikobericht: {total_risk:.2f}% Gesamtrisiko, "
                           f"{positions_count}/{self.max_positions} Positionen, Risikolevel: {risk_level}")
                           
            return report
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Überprüfung des Portfoliorisikos: {str(e)}")
            return {
                "error": str(e),
                "can_open_position": False
            }
    
    def can_open_new_position(self) -> Tuple[bool, Optional[str]]:
        """
        Prüft, ob eine neue Position eröffnet werden kann basierend auf Portfoliorisiko und verfügbarer Kapazität
        
        Returns:
            Tuple mit:
            - Boolean: True, wenn eine neue Position eröffnet werden kann
            - String: Grund, wenn keine Position eröffnet werden kann (oder None)
        """
        try:
            risk_report = self.check_portfolio_risk()
            
            # Prüfen, ob ein Fehler beim Abrufen des Risikoberichts aufgetreten ist
            if "error" in risk_report:
                return False, f"Fehler bei der Risikobewertung: {risk_report['error']}"
                
            # Darf eine neue Position eröffnet werden?
            can_open = risk_report.get("can_open_position", False)
            
            # Wenn nicht, Grund ermitteln
            reason = None
            if not can_open:
                if risk_report.get("positions_count", 0) >= self.max_positions:
                    reason = f"Maximale Anzahl an Positionen erreicht ({self.max_positions})"
                elif risk_report.get("total_risk_percent", 0) >= self.max_portfolio_risk:
                    reason = f"Maximales Portfoliorisiko erreicht ({self.max_portfolio_risk:.1f}%)"
                else:
                    reason = "Unbekannter Risikofaktor verhindert neue Positionen"
                    
            return can_open, reason
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Prüfung, ob eine neue Position eröffnet werden kann: {e}")
            return False, f"Interner Fehler: {str(e)}"
    
    def calculate_dynamic_stop_loss(self, symbol: str, df: pd.DataFrame, 
                                    signal_type: str, entry_price: float) -> float:
        """
        Berechnet einen dynamischen Stop-Loss basierend auf ATR
        
        Args:
            symbol: Trading-Paar-Symbol
            df: DataFrame mit Kerzendaten
            signal_type: 'buy' oder 'sell'
            entry_price: Einstiegspreis
            
        Returns:
            Berechneter Stop-Loss-Preis
        """
        try:
            # ATR (Average True Range) berechnen
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=self.dynamic_sl_atr_periods).mean()
            
            # Letztes ATR extrahieren
            latest_atr = df['atr'].iloc[-1]
            
            # Stop-Loss berechnen
            if signal_type.lower() == 'buy':
                sl_price = entry_price - (latest_atr * self.dynamic_sl_atr_multiplier)
            else:  # Verkauf
                sl_price = entry_price + (latest_atr * self.dynamic_sl_atr_multiplier)
                
            self.logger.info(f"Dynamischer ATR-basierter Stop-Loss für {symbol}: {sl_price:.6f} "
                           f"(ATR: {latest_atr:.6f}, Multiplikator: {self.dynamic_sl_atr_multiplier})")
                           
            return sl_price
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Berechnung des dynamischen Stop-Loss für {symbol}: {str(e)}")
            # Fallback auf prozentbasierten Stop-Loss
            if signal_type.lower() == 'buy':
                return entry_price * (1 - self.stop_loss_percent / 100)
            else:
                return entry_price * (1 + self.stop_loss_percent / 100)
                
    def adjust_trailing_stop(self, symbol: str, current_price: float, 
                            position_type: str, current_stop_loss: float) -> float:
        """
        Passt den Trailing-Stop-Loss an, wenn der Preis sich bewegt hat
        
        Args:
            symbol: Trading-Paar-Symbol
            current_price: Aktueller Preis
            position_type: 'long' oder 'short'
            current_stop_loss: Aktueller Stop-Loss-Preis
            
        Returns:
            Neuer Stop-Loss-Preis (oder alter, wenn keine Anpassung)
        """
        if not self.use_trailing_stop:
            return current_stop_loss
            
        try:
            if position_type.lower() == 'long':
                # Für Long-Positionen: Stop-Loss nach oben anpassen, wenn der Preis steigt
                highest_price = self.position_high_prices.get(symbol, current_price)
                if current_price > highest_price:
                    # Neuer Höchststand - Stop-Loss entsprechend nachziehen
                    trailing_distance = highest_price * (self.stop_loss_percent / 100)
                    new_stop_loss = highest_price - trailing_distance
                    
                    # Nur anpassen, wenn der neue Stop-Loss höher ist
                    if new_stop_loss > current_stop_loss:
                        self.logger.info(f"Trailing-Stop für {symbol} (Long) angepasst: "
                                     f"{current_stop_loss:.6f} -> {new_stop_loss:.6f}")
                        return new_stop_loss
            else:  # Short-Position
                # Für Short-Positionen: Stop-Loss nach unten anpassen, wenn der Preis fällt
                lowest_price = self.position_low_prices.get(symbol, current_price)
                if current_price < lowest_price:
                    # Neuer Tiefststand - Stop-Loss entsprechend nachziehen
                    trailing_distance = lowest_price * (self.stop_loss_percent / 100)
                    new_stop_loss = lowest_price + trailing_distance
                    
                    # Nur anpassen, wenn der neue Stop-Loss niedriger ist
                    if new_stop_loss < current_stop_loss:
                        self.logger.info(f"Trailing-Stop für {symbol} (Short) angepasst: "
                                     f"{current_stop_loss:.6f} -> {new_stop_loss:.6f}")
                        return new_stop_loss
                        
            # Keine Änderung
            return current_stop_loss
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Anpassung des Trailing-Stops für {symbol}: {str(e)}")
            return current_stop_loss
            
    def _calculate_portfolio_risk(self) -> float:
        """
        Berechnet das aktuelle Portfoliorisiko basierend auf offenen Positionen
        
        Returns:
            Gesamtrisiko in Prozent
        """
        total_risk = 0.0
        
        # Summe aller Positionsrisiken
        for symbol, position in self.open_positions.items():
            risk_percent = position.get("risk_percent", 0)
            total_risk += risk_percent
            
        return total_risk
        
    def _get_tick_size(self, symbol: str) -> float:
        """
        Holt die Tick-Größe für ein Symbol (für genaue Rundungen)
        
        Args:
            symbol: Trading-Paar-Symbol
            
        Returns:
            Tick-Größe (oder Standardwert)
        """
        try:
            # Standardwert verwenden, wenn kein Symbol angegeben
            if not symbol:
                return 0.00001
                
            # Versuche, die Instrumenteninformationen abzurufen
            instrument_info = self.api.get_instrument_info(symbol)
            
            if instrument_info and 'result' in instrument_info and 'list' in instrument_info['result']:
                for item in instrument_info['result']['list']:
                    if item.get('symbol') == symbol:
                        return float(item.get('lot_size_filter', {}).get('qty_step', 0.00001))
            
            # Fallback auf sinnvollen Standardwert
            return 0.00001
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Tick-Größe für {symbol}: {str(e)}")
            return 0.00001
            
    def _round_to_tick(self, value: float, tick_size: float) -> float:
        """
        Rundet einen Wert zur nächsten Tick-Größe
        
        Args:
            value: Zu rundender Wert
            tick_size: Tick-Größe
            
        Returns:
            Gerundeter Wert
        """
        if tick_size <= 0:
            return value
            
        return round(value / tick_size) * tick_size