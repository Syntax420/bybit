import os
import json
import logging
import sys
import re
import hashlib
import copy
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dotenv import load_dotenv

class ConfigValidationError(Exception):
    """Fehler bei der Konfigurationsvalidierung mit Details zu den ungültigen Werten"""
    def __init__(self, message: str, validation_errors: List[str] = None):
        self.validation_errors = validation_errors or []
        error_details = "\n- " + "\n- ".join(self.validation_errors) if self.validation_errors else ""
        super().__init__(f"{message}{error_details}")


class ConfigManager:
    """
    Erweiterte zentrale Verwaltung der Bot-Konfiguration mit:
    - Verbesserter Validierung und Typprüfung
    - Sicherer Behandlung sensibler Daten
    - Unterstützung für verschiedene Konfigurationsquellen
    - Versionierung und Backup
    """
    
    # Default-Einstellungen, die verwendet werden, wenn keine Konfiguration angegeben ist
    DEFAULT_CONFIG = {
        "general": {
            "paper_trading": True,
            "use_cache": True,
            "cache_dir": "cache",
            "max_cache_age_days": 7,
            "backup_config": True,
            "max_config_backups": 5
        },
        "api": {
            "testnet": True,
            "max_retries": 3,
            "retry_delay_seconds": 2,
            "rate_limit_cooldown": 60,  # Wartezeit in Sekunden bei Rate-Limit
            "connection_timeout": 10,    # Timeout für API-Verbindung in Sekunden
            # Sensible Daten werden aus .env oder Umgebungsvariablen geladen
            "api_key": "",  # Nicht in JSON speichern!
            "api_secret": ""  # Nicht in JSON speichern!
        },
        "trading": {
            "trading_interval": "15m",
            "default_leverage": 3,
            "max_positions": 5,
            "risk_per_trade_percent": 1.0,
            "take_profit_percent": 3.0,
            "stop_loss_percent": 2.0,
            "use_trailing_stop": True,
            "min_profit_to_trail": 1.0,
            "trail_percent": 1.0,
            "min_profit_for_breakeven": 3.0,
            "reward_ratio": 1.5,
            "partial_tp_size_1": 30.0,  # Prozentsatz der Position für ersten partiellen TP
            "time_filtering": False,
            "trading_hours_start": 0,
            "trading_hours_end": 23,
            "symbols": {
                "whitelist": [],
                "blacklist": []
            }
        },
        "risk": {
            "max_daily_drawdown_percent": 5.0,
            "max_total_drawdown_percent": 15.0,
            "max_consecutive_losses": 5,
            "reduce_position_after_loss": True,
            "position_reduction_factor": 0.5,  # Nach Verlust Positionsgröße halbieren
            "max_daily_trades": 20,
            "min_account_balance": 100  # Minimale Kontogröße, um weiter zu handeln
        },
        "strategy": {
            "active_strategy": "rsi_macd",
            "parameters": {
                "rsi_macd": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "ema_short": 9,
                    "ema_long": 21
                },
                "donchian_channel": {
                    "window": 20,
                    "atr_period": 14,
                    "atr_multiplier": 2
                }
            }
        },
        "logging": {
            "level": "INFO",
            "save_reports": True,
            "report_dir": "reports",
            "log_dir": "logs",
            "max_log_files": 30,
            "log_api_calls": True,
            "log_trades": True,
            "log_performance": True
        },
        "notifications": {
            "enabled": False,
            "notify_on_trade": True,
            "notify_on_error": True,
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "sender_email": "",
                "receiver_email": ""
                # "smtp_password" wird aus Umgebungsvariablen geladen
            },
            "telegram": {
                "enabled": False,
                "chat_id": ""
                # "bot_token" wird aus Umgebungsvariablen geladen
            }
        }
    }
    
    # Liste sensibler Konfigurationsschlüssel, die nicht in Klartext gespeichert werden
    SENSITIVE_KEYS = [
        "api.api_key", 
        "api.api_secret",
        "notifications.email.smtp_password", 
        "notifications.telegram.bot_token"
    ]
    
    # Validierungsregeln für Konfigurationswerte
    # Format: "pfad.zum.wert": {"type": Typ, "min": Minimum, "max": Maximum, "regex": Regulärer Ausdruck}
    VALIDATION_RULES = {
        "general.paper_trading": {"type": bool},
        "general.max_cache_age_days": {"type": int, "min": 1, "max": 365},
        
        "api.testnet": {"type": bool},
        "api.max_retries": {"type": int, "min": 1, "max": 10},
        "api.retry_delay_seconds": {"type": int, "min": 1, "max": 60},
        "api.api_key": {"type": str, "regex": r"^[A-Za-z0-9-_]*$"},
        "api.api_secret": {"type": str, "regex": r"^[A-Za-z0-9-_]*$"},
        
        "trading.trading_interval": {"type": str, "allowed": ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]},
        "trading.default_leverage": {"type": int, "min": 1, "max": 100},
        "trading.max_positions": {"type": int, "min": 1, "max": 50},
        "trading.risk_per_trade_percent": {"type": float, "min": 0.1, "max": 5.0},
        "trading.take_profit_percent": {"type": float, "min": 0.1},
        "trading.stop_loss_percent": {"type": float, "min": 0.1},
        
        "risk.max_daily_drawdown_percent": {"type": float, "min": 0.1, "max": 100},
        "risk.max_total_drawdown_percent": {"type": float, "min": 0.1, "max": 100},
        "risk.max_consecutive_losses": {"type": int, "min": 1},
        
        "logging.level": {"type": str, "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}
    }
    
    def __init__(self, config_path: str = "config.json", load_env: bool = True):
        """
        Initialisiert den Konfigurationsmanager
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            load_env: Wenn True, werden Umgebungsvariablen aus .env geladen
        """
        self.logger = logging.getLogger("utils.config_manager")
        self.config_path = config_path
        self.env_loaded = False
        self.validation_errors = []
        
        # .env Datei laden, wenn angefordert
        if load_env:
            self.env_loaded = load_dotenv()
            if self.env_loaded:
                self.logger.info(".env Datei erfolgreich geladen")
            else:
                self.logger.warning("Keine .env Datei gefunden oder Fehler beim Laden")
        
        # Konfiguration laden und validieren
        self.config = self._load_config()
        
        # Sensible Daten aus Umgebungsvariablen laden
        self._load_sensitive_data()
        
        # Konfiguration validieren
        self._validate_config()
        
        # Umgebungsvariablen-Überschreibungen anwenden
        self._apply_environment_overrides()
        
        # Bei erfolgreicher Validierung Backup erstellen
        if self.config.get("general", {}).get("backup_config", True):
            self._backup_config()
        
        self.logger.info(f"Konfiguration geladen aus {config_path}")
        
    def get(self, path: str, default: Any = None) -> Any:
        """
        Liest einen Konfigurationswert aus dem angegebenen Pfad
        
        Args:
            path: Punkt-notierter Pfad zum Konfigurationswert (z.B. "trading.max_positions")
            default: Standardwert, falls nicht gefunden
            
        Returns:
            Konfigurationswert oder default, falls nicht gefunden
        """
        parts = path.split(".")
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
        
    def set(self, path: str, value: Any) -> bool:
        """
        Setzt einen Konfigurationswert
        
        Args:
            path: Punkt-notierter Pfad zum Konfigurationswert (z.B. "trading.max_positions")
            value: Neuer Wert
            
        Returns:
            True wenn erfolgreich, sonst False
        """
        parts = path.split(".")
        current = self.config
        
        # Pfad bis zum letzten Element navigieren
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Wert setzen
        last_part = parts[-1]
        
        # Prüfen, ob es sich um einen sensitiven Wert handelt
        is_sensitive = False
        for sensitive_key in self.SENSITIVE_KEYS:
            if path == sensitive_key:
                is_sensitive = True
                break
                
        # Wenn sensitiv und nicht leer, nicht in die Konfiguration speichern, sondern in Umgebungsvariable
        if is_sensitive and value:
            # Konvertieren in Umgebungsvariablenformat: z.B. api.api_key -> BYBIT_API_API_KEY
            env_var = f"BYBIT_{path.upper().replace('.', '_')}"
            os.environ[env_var] = str(value)
            # In der Config nur einen Platzhalter speichern
            current[last_part] = "[SENSITIVE]"
            self.logger.debug(f"Sensitiver Wert für {path} in Umgebungsvariable {env_var} gespeichert")
            return True
            
        # Wert in Konfiguration setzen
        current[last_part] = value
        
        # Validieren
        validation_result = self._validate_single_value(path, value)
        if not validation_result["valid"]:
            self.logger.warning(f"Ungültiger Wert für {path}: {validation_result['error']}")
            return False
            
        return True
        
    def save(self, path: Optional[str] = None, include_sensitive: bool = False) -> bool:
        """
        Speichert die Konfiguration in eine Datei
        
        Args:
            path: Zielpfad (oder self.config_path wenn None)
            include_sensitive: Wenn True, werden auch sensitive Daten gespeichert (nicht empfohlen)
            
        Returns:
            True wenn erfolgreich, sonst False
        """
        save_path = path or self.config_path
        try:
            # Kopie der Konfiguration erstellen, um sensible Daten zu entfernen
            config_to_save = copy.deepcopy(self.config)
            
            # Sensible Daten entfernen, wenn nicht explizit gewünscht
            if not include_sensitive:
                for sensitive_key in self.SENSITIVE_KEYS:
                    parts = sensitive_key.split(".")
                    current = config_to_save
                    last_part = parts[-1]
                    
                    # Zum letzten Element navigieren
                    found = True
                    for part in parts[:-1]:
                        if part in current:
                            current = current[part]
                        else:
                            found = False
                            break
                            
                    # Löschen oder mit Platzhalter ersetzen
                    if found and last_part in current and current[last_part]:
                        current[last_part] = "[SENSITIVE]"
            
            # Konfigurationsverzeichnis erstellen, falls nicht vorhanden
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            # Konfiguration speichern
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
                
            self.logger.info(f"Konfiguration gespeichert in {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            return False
            
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus der Datei
        
        Returns:
            Konfigurationswörterbuch
        """
        # Mit Default-Konfiguration starten
        config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Versuchen, Konfigurationsdatei zu laden
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    
                # Konfiguration rekursiv aktualisieren
                self._update_nested_dict(config, user_config)
                self.logger.info(f"Konfiguration geladen aus {self.config_path}")
            else:
                self.logger.warning(f"Konfigurationsdatei {self.config_path} nicht gefunden, verwende Standardeinstellungen")
                # Konfigurationsdatei mit Defaults anlegen
                os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else '.', exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                self.logger.info(f"Standardkonfiguration gespeichert in {self.config_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            
        return config
        
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Aktualisiert ein verschachteltes Dictionary rekursiv
        
        Args:
            d: Ziel-Dictionary
            u: Quell-Dictionary mit Aktualisierungen
            
        Returns:
            Aktualisiertes Dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _validate_config(self) -> bool:
        """
        Überprüft die gesamte Konfiguration auf Gültigkeit
        
        Returns:
            True wenn gültig, sonst False
        """
        self.validation_errors = []
        valid = True
        
        try:
            # Prüfen, ob erforderliche Verzeichnisse existieren
            self._ensure_directories_exist()
            
            # Validierung aller Werte gemäß Regeln
            for path, rules in self.VALIDATION_RULES.items():
                value = self.get(path)
                result = self._validate_single_value(path, value, rules)
                if not result["valid"]:
                    self.validation_errors.append(f"{path}: {result['error']}")
                    valid = False
            
            # Zusätzliche Business-Logic-Validierungen
            
            # 1. Testnet-Warnung bei Live-Trading
            if not self.config["api"]["testnet"] and self.config["general"]["paper_trading"] is False:
                self.logger.warning("ACHTUNG: Live-Trading am MainNet aktiviert! Stellen Sie sicher, dass dies beabsichtigt ist.")
                
            # 2. Überprüfung, ob aktive Strategie konfiguriert ist
            active_strategy = self.config["strategy"]["active_strategy"]
            if active_strategy not in self.config["strategy"]["parameters"]:
                error_msg = f"Konfiguration für aktive Strategie '{active_strategy}' nicht gefunden"
                self.validation_errors.append(error_msg)
                self.logger.error(error_msg)
                valid = False
                
            # 3. Plausibilitätsprüfung für Risikomanagement
            if self.config["trading"]["risk_per_trade_percent"] > 5:
                self.logger.warning("Risikoparameter ist ungewöhnlich hoch (>5%). Überprüfen Sie Ihre Einstellungen.")
                
            if self.config["trading"]["max_positions"] > 20:
                self.logger.warning("Maximale Positionsanzahl ist sehr hoch (>20). Dies könnte zu Liquiditätsproblemen führen.")
            
            # 4. Überprüfung der API-Schlüssel-Konfiguration im Live-Modus
            if not self.config["general"]["paper_trading"]:
                api_key = self.get("api.api_key")
                api_secret = self.get("api.api_secret")
                
                if not api_key or api_key == "[SENSITIVE]":
                    error_msg = "API-Schlüssel fehlt für Live-Trading"
                    self.validation_errors.append(error_msg)
                    self.logger.error(error_msg)
                    valid = False
                    
                if not api_secret or api_secret == "[SENSITIVE]":
                    error_msg = "API-Secret fehlt für Live-Trading"
                    self.validation_errors.append(error_msg)
                    self.logger.error(error_msg)
                    valid = False
                    
            self.logger.info(f"Konfigurationsvalidierung abgeschlossen: {'Erfolgreich' if valid else 'Fehler gefunden'}")
            
            # Bei schwerwiegenden Fehlern Exception werfen
            if not valid:
                raise ConfigValidationError("Konfiguration enthält ungültige Werte:", self.validation_errors)
                
            return valid
            
        except ConfigValidationError:
            # Diese Exception wird weitergereicht
            raise
        except Exception as e:
            error_msg = f"Fehler bei der Konfigurationsvalidierung: {e}"
            self.validation_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _validate_single_value(self, path: str, value: Any, rules: Dict = None) -> Dict[str, Any]:
        """
        Validiert einen einzelnen Konfigurationswert gegen seine Regeln
        
        Args:
            path: Pfad zum Konfigurationswert
            value: Zu validierender Wert
            rules: Validierungsregeln (Optional)
            
        Returns:
            Dictionary mit Validierungsergebnis {'valid': bool, 'error': str}
        """
        # Wenn keine Regeln explizit übergeben, aus VALIDATION_RULES holen
        if rules is None:
            rules = self.VALIDATION_RULES.get(path)
            
        # Wenn keine Regeln für diesen Pfad definiert sind, als gültig ansehen
        if not rules:
            return {"valid": True, "error": ""}
            
        # Typ überprüfen
        if "type" in rules:
            expected_type = rules["type"]
            
            # Bei sensiblen Daten den Platzhalter akzeptieren
            if value == "[SENSITIVE]" and path in self.SENSITIVE_KEYS:
                return {"valid": True, "error": ""}
                
            # Typprüfung durchführen
            if not isinstance(value, expected_type):
                return {
                    "valid": False, 
                    "error": f"Typ sollte {expected_type.__name__} sein, ist aber {type(value).__name__}"
                }
                
            # Für Strings: Regex-Prüfung
            if expected_type == str and "regex" in rules and rules["regex"] and value:
                pattern = rules["regex"]
                if not re.match(pattern, value):
                    return {
                        "valid": False,
                        "error": f"Wert entspricht nicht dem Muster {pattern}"
                    }
                
            # Für Zahlen: Min/Max-Prüfung
            if expected_type in (int, float):
                if "min" in rules and value < rules["min"]:
                    return {
                        "valid": False,
                        "error": f"Wert sollte mindestens {rules['min']} sein"
                    }
                    
                if "max" in rules and value > rules["max"]:
                    return {
                        "valid": False,
                        "error": f"Wert sollte höchstens {rules['max']} sein"
                    }
                    
            # Erlaubte Werte prüfen
            if "allowed" in rules and value not in rules["allowed"]:
                return {
                    "valid": False,
                    "error": f"Wert muss einer der folgenden sein: {', '.join(rules['allowed'])}"
                }
                
        return {"valid": True, "error": ""}
            
    def _ensure_directories_exist(self) -> None:
        """
        Stellt sicher, dass alle erforderlichen Verzeichnisse existieren
        """
        # Cache-Verzeichnis
        cache_dir = self.config["general"]["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        
        # Logs-Verzeichnis
        log_dir = self.config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        # Reports-Verzeichnis
        report_dir = self.config["logging"]["report_dir"]
        os.makedirs(report_dir, exist_ok=True)
        
        # Backups-Verzeichnis
        backup_dir = os.path.join(os.path.dirname(self.config_path), "config_backups")
        os.makedirs(backup_dir, exist_ok=True)
        
    def _apply_environment_overrides(self) -> None:
        """
        Wendet Überschreibungen aus Umgebungsvariablen an
        
        Ermöglicht die Überschreibung von Konfigurationswerten durch Umgebungsvariablen.
        Format: BYBIT_SECTION_KEY=value (z.B. BYBIT_TRADING_MAX_POSITIONS=3)
        """
        try:
            prefix = "BYBIT_"
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # BYBIT_SECTION_KEY in section und key aufteilen
                    remaining = key[len(prefix):].lower()
                    
                    # Unterstützung für mehrstufige Pfade mit Unterstrichen
                    # z.B. BYBIT_TRADING_SYMBOLS_WHITELIST_0=BTCUSDT
                    path_parts = []
                    current_part = ""
                    array_index = None
                    
                    # Spezialbehandlung für Array-Zugriff
                    if "_" in remaining:
                        parts = remaining.split("_")
                        
                        # Prüfen, ob das letzte Element eine Indexnummer ist
                        if parts[-1].isdigit():
                            array_index = int(parts[-1])
                            remaining = "_".join(parts[:-1])
                    
                    # Punkt-notatierte Pfade erstellen
                    path = remaining.replace("_", ".")
                    
                    # Wert konvertieren
                    typed_value = self._convert_env_value(value)
                    
                    # Pfad in Konfiguration finden und Wert setzen
                    if array_index is not None:
                        # Array-Element anpassen
                        array_path = path
                        current_array = self.get(array_path, [])
                        
                        # Stellen sicher, dass das Array lang genug ist
                        while len(current_array) <= array_index:
                            current_array.append(None)
                            
                        current_array[array_index] = typed_value
                        self.set(array_path, current_array)
                        self.logger.debug(f"Array-Element in {array_path}[{array_index}] auf {typed_value} gesetzt durch Umgebungsvariable {key}")
                    else:
                        # Normaler Wert
                        old_value = self.get(path)
                        self.set(path, typed_value)
                        self.logger.debug(f"Konfigurationswert {path} von {old_value} auf {typed_value} gesetzt durch Umgebungsvariable {key}")
                        
        except Exception as e:
            self.logger.error(f"Fehler beim Anwenden von Umgebungsvariablen-Überschreibungen: {e}")
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Konvertiert einen Umgebungsvariablenwert in den passenden Typ
        
        Args:
            value: Umgebungsvariablenwert als String
            
        Returns:
            Konvertierter Wert im passenden Typ
        """
        # Boolean
        if value.lower() in ['true', 'yes', '1', 'y', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'n', 'off']:
            return False
            
        # Zahlen
        if value.replace('.', '', 1).isdigit():
            # Integer oder Float
            if '.' in value:
                return float(value)
            else:
                return int(value)
                
        # Arrays (kommagetrennt)
        if ',' in value:
            return [self._convert_env_value(item.strip()) for item in value.split(',')]
            
        # Standard: String
        return value
        
    def _load_sensitive_data(self) -> None:
        """
        Lädt sensitive Daten aus Umgebungsvariablen
        """
        try:
            for sensitive_key in self.SENSITIVE_KEYS:
                # Konvertieren in Umgebungsvariablenformat: z.B. api.api_key -> BYBIT_API_API_KEY
                env_var = f"BYBIT_{sensitive_key.upper().replace('.', '_')}"
                
                # Wert aus Umgebungsvariable laden
                if env_var in os.environ and os.environ[env_var]:
                    # Pfad in der Konfiguration finden
                    parts = sensitive_key.split(".")
                    current = self.config
                    
                    # Zum letzten Element navigieren und erstellen, falls nicht vorhanden
                    for i, part in enumerate(parts[:-1]):
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                        
                    # Wert setzen
                    current[parts[-1]] = os.environ[env_var]
                    self.logger.debug(f"Sensitiver Wert für {sensitive_key} aus Umgebungsvariable {env_var} geladen")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden sensitiver Daten: {e}")
            
    def _backup_config(self) -> None:
        """
        Erstellt ein Backup der aktuellen Konfiguration
        """
        try:
            if not self.config.get("general", {}).get("backup_config", True):
                return
                
            # Backup-Verzeichnis erstellen
            backup_dir = os.path.join(os.path.dirname(self.config_path), "config_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Zeitstempel für Backup-Datei
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"config_{timestamp}.json")
            
            # Backup speichern (ohne sensitive Daten)
            self.save(backup_path, include_sensitive=False)
            
            # Alte Backups aufräumen
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Konfigurationsbackups: {e}")
            
    def _cleanup_old_backups(self, backup_dir: str) -> None:
        """
        Räumt alte Konfigurationsbackups auf
        
        Args:
            backup_dir: Verzeichnis mit Backups
        """
        try:
            max_backups = self.config.get("general", {}).get("max_config_backups", 5)
            
            # Alle Backup-Dateien im Verzeichnis finden
            backup_files = [f for f in os.listdir(backup_dir) if f.startswith("config_") and f.endswith(".json")]
            
            # Nach Zeitstempel sortieren (neueste zuerst)
            backup_files.sort(reverse=True)
            
            # Überzählige Backups löschen
            for old_file in backup_files[max_backups:]:
                try:
                    os.remove(os.path.join(backup_dir, old_file))
                    self.logger.debug(f"Altes Konfigurationsbackup gelöscht: {old_file}")
                except Exception as e:
                    self.logger.warning(f"Konnte altes Backup nicht löschen: {old_file} - {e}")
                    
        except Exception as e:
            self.logger.error(f"Fehler beim Aufräumen alter Backups: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Gibt die gesamte Konfiguration zurück (Vorsicht mit sensiblen Daten)
        
        Returns:
            Kopie der Konfiguration
        """
        return copy.deepcopy(self.config)
        
    def reset_to_defaults(self, save: bool = False) -> None:
        """
        Setzt die Konfiguration auf Standardwerte zurück
        
        Args:
            save: Wenn True, wird die zurückgesetzte Konfiguration gespeichert
        """
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.logger.info("Konfiguration auf Standardwerte zurückgesetzt")
        
        if save:
            self.save()
            
    def merge_config(self, other_config: Dict[str, Any], save: bool = False) -> None:
        """
        Führt eine andere Konfiguration mit der aktuellen zusammen
        
        Args:
            other_config: Zu verwendende Konfiguration
            save: Wenn True, wird die zusammengeführte Konfiguration gespeichert
        """
        self._update_nested_dict(self.config, other_config)
        self.logger.info("Konfiguration zusammengeführt")
        
        # Validierung durchführen
        self._validate_config()
        
        if save:
            self.save()