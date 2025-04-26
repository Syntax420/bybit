import logging
from typing import Dict, Any, Optional, List, Type
import importlib
import inspect

from api.bybit_api import BybitAPI
from strategy.base_strategy import BaseStrategy
from strategy.strategy_rsi_macd import RSIMACDStrategy

class StrategyManager:
    """
    Manager für Trading-Strategien.
    
    Bietet eine zentrale Schnittstelle zur Verwaltung und Initialisierung von Trading-Strategien.
    Dient als Factory für die verschiedenen Strategietypen.
    """
    
    # Registrierung verfügbarer Strategien (Name -> Klasse)
    AVAILABLE_STRATEGIES = {
        "rsi_macd": RSIMACDStrategy,
        # Weitere Strategien hier hinzufügen
    }
    
    def __init__(self, api: BybitAPI, config: Dict):
        """
        Initialisiert den Strategie-Manager
        
        Args:
            api: BybitAPI-Instanz für Marktdaten und Order-Ausführung
            config: Konfigurationswörterbuch
        """
        self.api = api
        self.config = config
        self.logger = logging.getLogger("strategy.manager")
        
        # Aktive Strategie und Name aus Konfiguration bestimmen
        self.active_strategy_name = config.get("strategy", {}).get("active_strategy", "rsi_macd")
        self.active_strategy = self._create_strategy(self.active_strategy_name)
        
        if self.active_strategy:
            self.logger.info(f"Strategie-Manager initialisiert mit aktiver Strategie: {self.active_strategy_name}")
        else:
            self.logger.error(f"Fehler beim Initialisieren der aktiven Strategie: {self.active_strategy_name}")
            
    def _create_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Erstellt eine Strategie-Instanz basierend auf dem Namen
        
        Args:
            strategy_name: Name der Strategie
            
        Returns:
            Instanz der angegebenen Strategie oder None bei Fehler
        """
        try:
            if strategy_name not in self.AVAILABLE_STRATEGIES:
                self.logger.error(f"Unbekannte Strategie: {strategy_name}")
                return None
                
            strategy_class = self.AVAILABLE_STRATEGIES[strategy_name]
            strategy = strategy_class(self.api, self.config)
            
            return strategy
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Strategie {strategy_name}: {str(e)}")
            return None
            
    def get_active_strategy(self) -> BaseStrategy:
        """
        Gibt die aktuell aktive Strategie zurück
        
        Returns:
            Die aktive Strategie-Instanz
        """
        return self.active_strategy
        
    def change_strategy(self, strategy_name: str) -> bool:
        """
        Wechselt zu einer anderen Strategie
        
        Args:
            strategy_name: Name der neuen Strategie
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            new_strategy = self._create_strategy(strategy_name)
            if new_strategy:
                self.active_strategy = new_strategy
                self.active_strategy_name = strategy_name
                self.logger.info(f"Strategie gewechselt zu: {strategy_name}")
                return True
            else:
                self.logger.error(f"Strategie konnte nicht gewechselt werden zu: {strategy_name}")
                return False
        except Exception as e:
            self.logger.error(f"Fehler beim Wechseln der Strategie zu {strategy_name}: {str(e)}")
            return False
            
    def get_available_strategies(self) -> List[str]:
        """
        Gibt eine Liste der verfügbaren Strategien zurück
        
        Returns:
            Liste der verfügbaren Strategienamen
        """
        return list(self.AVAILABLE_STRATEGIES.keys())
        
    def discover_strategies(self) -> None:
        """
        Entdeckt und registriert dynamisch alle verfügbaren Strategien im 'strategy'-Paket
        """
        try:
            import pkgutil
            import strategy
            import sys
            
            # Strategiepaket durchsuchen und Module dynamisch laden
            self.logger.info("Suche nach verfügbaren Strategien...")
            
            for _, name, ispkg in pkgutil.iter_modules(strategy.__path__):
                if name.startswith('strategy_'):
                    try:
                        # Modul dynamisch importieren
                        module_name = f"strategy.{name}"
                        module = importlib.import_module(module_name)
                        
                        # Nach Strategieklassen im Modul suchen (die von BaseStrategy erben)
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (inspect.isclass(attr) and 
                                issubclass(attr, BaseStrategy) and 
                                attr != BaseStrategy):
                                
                                # Strategienamen aus Klassenname ableiten (entferne 'Strategy' am Ende)
                                strategy_key = attr.__name__.replace('Strategy', '').lower()
                                
                                # Strategie registrieren
                                self.AVAILABLE_STRATEGIES[strategy_key] = attr
                                self.logger.info(f"Strategie registriert: {strategy_key} -> {attr.__name__}")
                                
                    except Exception as e:
                        self.logger.error(f"Fehler beim Laden der Strategie aus Modul {name}: {str(e)}")
            
            self.logger.info(f"Verfügbare Strategien: {', '.join(self.get_available_strategies())}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Entdecken von Strategien: {str(e)}")
            
    def analyze_symbol(self, symbol: str, interval: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Analysiert ein Symbol mit der aktiven Strategie
        
        Args:
            symbol: Trading-Paar-Symbol
            interval: Zeitrahmen-Intervall (oder None für den Standardwert aus der Konfiguration)
            limit: Anzahl der zu analysierenden Kerzen
            
        Returns:
            Dictionary mit Analyseergebnissen und Signalen
        """
        if not self.active_strategy:
            return {"signal": "neutral", "error": "Keine aktive Strategie"}
            
        # Standardintervall aus der Konfiguration verwenden, wenn nicht angegeben
        if interval is None:
            interval = self.config.get("trading", {}).get("trading_interval", "15").replace("m", "")
            
        return self.active_strategy.analyze(symbol, interval, limit)
        
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """
        Gibt die Parameter der aktiven Strategie zurück
        
        Returns:
            Dictionary mit Strategieparametern
        """
        if not self.active_strategy:
            return {}
            
        return self.active_strategy.get_strategy_parameters()