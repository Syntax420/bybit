#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testskript für die verbesserte Donchian Channel Strategie.
Dieses Skript lädt historische Daten und führt Backtest-Analysen durch.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional

from api.bybit_api import BybitAPI
from strategy.strategy_donchian import DonchianChannelStrategy

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/donchian_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    ]
)

logger = logging.getLogger("donchian_test")

def load_config(config_path: str = "config.json") -> Dict:
    """
    Lädt die Konfiguration aus einer JSON-Datei
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        Dictionary mit Konfigurationsparametern
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Konfiguration geladen aus: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
        return {}

def test_single_symbol(api: BybitAPI, strategy: DonchianChannelStrategy, symbol: str, interval: str = "15") -> Dict:
    """
    Testet die Strategie für ein einzelnes Symbol
    
    Args:
        api: BybitAPI-Instanz
        strategy: Donchian Channel Strategie-Instanz
        symbol: Das zu testende Symbol
        interval: Kerzen-Intervall
        
    Returns:
        Analyseergebnis der Strategie
    """
    logger.info(f"Teste Symbol: {symbol} mit Intervall: {interval}m")
    result = strategy.analyze(symbol, interval, 200)
    
    # Ausgabe der Analyseergebnisse
    signal = result.get("signal", "neutral")
    price = result.get("price", 0)
    dc_upper = result.get("dc_upper", 0)
    dc_lower = result.get("dc_lower", 0)
    adx = result.get("adx", 0)
    
    logger.info(f"Ergebnis für {symbol}: Signal={signal}, Preis={price:.8f}, "
               f"Oberer Kanal={dc_upper:.8f}, Unterer Kanal={dc_lower:.8f}, ADX={adx:.2f}")
    
    if signal != "neutral":
        params = result.get("params", {})
        entry = params.get("entry_price", 0)
        sl = params.get("stop_loss", 0)
        tp = params.get("take_profit", 0)
        reason = params.get("reason", "Keine Begründung verfügbar")
        
        logger.info(f"Signal-Details: Entry={entry:.8f}, SL={sl:.8f}, TP={tp:.8f}")
        logger.info(f"Begründung: {reason}")
        
        # Risikoberechnung
        risk_percent = abs(entry - sl) / entry * 100
        reward_percent = abs(tp - entry) / entry * 100
        risk_reward = reward_percent / risk_percent if risk_percent > 0 else 0
        
        logger.info(f"Risiko: {risk_percent:.2f}%, Gewinnpotenzial: {reward_percent:.2f}%, R:R = {risk_reward:.2f}")
    
    return result

def backtest_symbol(api: BybitAPI, strategy: DonchianChannelStrategy, symbol: str, 
                   interval: str = "15", limit: int = 500) -> Dict:
    """
    Führt einen einfachen Backtest für ein Symbol durch
    
    Args:
        api: BybitAPI-Instanz
        strategy: Donchian Channel Strategie-Instanz
        symbol: Das zu testende Symbol
        interval: Kerzen-Intervall
        limit: Anzahl der historischen Kerzen
        
    Returns:
        Backtest-Ergebnisse
    """
    logger.info(f"Starte Backtest für {symbol} ({interval}m) mit {limit} Kerzen")
    
    # Historische Daten laden
    try:
        df = strategy.fetch_candles(symbol, interval, limit)
        
        if df.empty or len(df) < 100:
            logger.error(f"Nicht genügend Daten für Backtest. Benötige mindestens 100 Kerzen, erhalten: {len(df)}")
            return {"symbol": symbol, "success": False, "error": "Nicht genügend Daten"}
            
        # Daten vorbereiten und Indikatoren berechnen
        df = strategy.prepare_data(df)
        
        # Leere Listen für Trades und Signale
        signals = []
        trades = []
        position = None
        
        # Backtest-Logik implementieren
        for i in range(strategy.dc_period + 10, len(df)):
            # Daten bis zum aktuellen Zeitpunkt extrahieren
            current_data = df.iloc[:i+1]
            
            # Signal für aktuelle Kerze generieren
            signal, params = strategy.generate_signal(current_data)
            
            # Signale speichern
            signals.append({
                "timestamp": df.iloc[i]["timestamp"],
                "price": float(df.iloc[i]["close"]),
                "signal": signal,
                "params": params
            })
            
            # Trade-Logik implementieren
            if signal != "neutral" and params:
                entry_price = params["entry_price"]
                stop_loss = params["stop_loss"]
                take_profit = params["take_profit"]
                
                # Wenn wir noch keine offene Position haben, eine neue eröffnen
                if position is None:
                    position = {
                        "type": signal,
                        "entry_time": df.iloc[i]["timestamp"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    }
                    logger.info(f"Position eröffnet: {signal} bei {entry_price:.8f}, SL={stop_loss:.8f}, TP={take_profit:.8f}")
            
            # Wenn wir eine offene Position haben, prüfen, ob SL oder TP erreicht wurde
            if position:
                current_price = float(df.iloc[i]["close"])
                high_price = float(df.iloc[i]["high"])
                low_price = float(df.iloc[i]["low"])
                
                exit_type = None
                exit_price = 0
                
                if position["type"] == "buy":
                    # Prüfen auf Stop Loss (niedriger als SL)
                    if low_price <= position["stop_loss"]:
                        exit_type = "stop_loss"
                        exit_price = position["stop_loss"]
                    # Prüfen auf Take Profit (höher als TP)
                    elif high_price >= position["take_profit"]:
                        exit_type = "take_profit"
                        exit_price = position["take_profit"]
                else:  # sell position
                    # Prüfen auf Stop Loss (höher als SL)
                    if high_price >= position["stop_loss"]:
                        exit_type = "stop_loss"
                        exit_price = position["stop_loss"]
                    # Prüfen auf Take Profit (niedriger als TP)
                    elif low_price <= position["take_profit"]:
                        exit_type = "take_profit"
                        exit_price = position["take_profit"]
                
                # Wenn wir einen Exit haben, Trade abschließen
                if exit_type:
                    profit_loss = 0
                    if position["type"] == "buy":
                        profit_loss = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                    else:
                        profit_loss = (position["entry_price"] - exit_price) / position["entry_price"] * 100
                    
                    trades.append({
                        "type": position["type"],
                        "entry_time": position["entry_time"],
                        "entry_price": position["entry_price"],
                        "exit_time": df.iloc[i]["timestamp"],
                        "exit_price": exit_price,
                        "exit_type": exit_type,
                        "profit_loss": profit_loss
                    })
                    
                    logger.info(f"Position geschlossen: {exit_type} bei {exit_price:.8f}, P/L: {profit_loss:.2f}%")
                    position = None
        
        # Backtest-Ergebnisse berechnen
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t["profit_loss"] > 0])
        losing_trades = len([t for t in trades if t["profit_loss"] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t["profit_loss"] for t in trades if t["profit_loss"] > 0])
        total_loss = sum([t["profit_loss"] for t in trades if t["profit_loss"] <= 0])
        
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        logger.info(f"Backtest abgeschlossen für {symbol}:")
        logger.info(f"Anzahl Trades: {total_trades}, Gewinner: {winning_trades}, Verlierer: {losing_trades}")
        logger.info(f"Gewinnrate: {win_rate:.2%}, Durchschn. Gewinn: {avg_win:.2f}%, Durchschn. Verlust: {avg_loss:.2f}%")
        logger.info(f"Profit-Faktor: {profit_factor:.2f}")
        
        # Ergebnisse zurückgeben
        return {
            "symbol": symbol,
            "success": True,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "trades": trades,
            "signals": signals
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Backtest für {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"symbol": symbol, "success": False, "error": str(e)}

def plot_backtest_results(symbol: str, df: pd.DataFrame, backtest_results: Dict) -> None:
    """
    Visualisiert die Backtest-Ergebnisse
    
    Args:
        symbol: Symbol-Name
        df: DataFrame mit Preisdaten
        backtest_results: Ergebnisse des Backtests
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Prüfen, ob wir gültige Ergebnisse haben
        if not backtest_results.get("success", False) or df.empty:
            logger.error("Keine gültigen Backtest-Ergebnisse zum Plotten verfügbar")
            return
        
        # Trades und Signale auslesen
        trades = backtest_results.get("trades", [])
        
        # Datumskonvertierung für Plotting
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Plot erstellen
        plt.figure(figsize=(14, 10))
        
        # Preis-Chart
        ax1 = plt.subplot(211)
        ax1.plot(df['date'], df['close'], label='Preis', color='black', linewidth=1)
        
        # Donchian Channel
        if 'dc_upper' in df.columns and 'dc_lower' in df.columns:
            ax1.plot(df['date'], df['dc_upper'], label='Oberer Kanal', color='green', linestyle='--', linewidth=0.8)
            ax1.plot(df['date'], df['dc_lower'], label='Unterer Kanal', color='red', linestyle='--', linewidth=0.8)
            
            if 'dc_middle' in df.columns:
                ax1.plot(df['date'], df['dc_middle'], label='Mittlerer Kanal', color='blue', linestyle=':', linewidth=0.8)
        
        # EMAs
        if 'ema50' in df.columns:
            ax1.plot(df['date'], df['ema50'], label='EMA 50', color='orange', linewidth=0.8, alpha=0.7)
        if 'ema200' in df.columns:
            ax1.plot(df['date'], df['ema200'], label='EMA 200', color='purple', linewidth=0.8, alpha=0.7)
        
        # Buy und Sell Signale markieren
        for trade in trades:
            entry_time = pd.to_datetime(trade['entry_time'], unit='ms')
            exit_time = pd.to_datetime(trade['exit_time'], unit='ms')
            
            if trade['type'] == 'buy':
                ax1.scatter(entry_time, trade['entry_price'], color='green', marker='^', s=100, label='_Buy')
                
                if trade['exit_type'] == 'take_profit':
                    ax1.scatter(exit_time, trade['exit_price'], color='blue', marker='o', s=100, label='_TP')
                else:
                    ax1.scatter(exit_time, trade['exit_price'], color='red', marker='o', s=100, label='_SL')
                    
            else:  # sell
                ax1.scatter(entry_time, trade['entry_price'], color='red', marker='v', s=100, label='_Sell')
                
                if trade['exit_type'] == 'take_profit':
                    ax1.scatter(exit_time, trade['exit_price'], color='blue', marker='o', s=100, label='_TP')
                else:
                    ax1.scatter(exit_time, trade['exit_price'], color='red', marker='o', s=100, label='_SL')
        
        # Plot verschönern
        ax1.set_title(f'Donchian Channel Backtest für {symbol}')
        ax1.set_ylabel('Preis')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Zweiter Plot für ADX und Volumen
        ax2 = plt.subplot(212, sharex=ax1)
        
        # ADX plotten
        if 'adx' in df.columns:
            ax2.plot(df['date'], df['adx'], label='ADX', color='purple', linewidth=1)
            ax2.axhline(y=25, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Volumen plotten mit zweiter Y-Achse
        ax3 = ax2.twinx()
        ax3.bar(df['date'], df['volume'], label='Volumen', color='blue', width=0.4, alpha=0.3)
        
        # Plot verschönern
        ax2.set_title('Indikatoren')
        ax2.set_ylabel('ADX')
        ax3.set_ylabel('Volumen')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper right')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        # Plot speichern
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(reports_dir, f'backtest_{symbol}_{timestamp}.png')
        plt.savefig(plot_path)
        
        logger.info(f"Plot gespeichert: {plot_path}")
        
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """
    Hauptfunktion zur Ausführung des Testskripts
    """
    # Konfiguration laden
    config = load_config()
    if not config:
        logger.error("Keine gültige Konfiguration gefunden. Beende Programm.")
        return
    
    # API initialisieren
    api_key = ""
    api_secret = ""
    
    # API-Schlüssel aus Konfiguration oder Umgebungsvariablen laden
    if "api" in config and "key" in config["api"] and "secret" in config["api"]:
        api_key = config["api"]["key"]
        api_secret = config["api"]["secret"]
    else:
        # Versuche, API-Schlüssel aus Datei zu laden
        try:
            with open("API-Schlüssel/API_key.txt", "r") as f:
                api_data = f.read().strip().split("\n")
                if len(api_data) >= 2:
                    api_key = api_data[0].strip()
                    api_secret = api_data[1].strip()
        except Exception as e:
            logger.error(f"Fehler beim Laden der API-Schlüssel: {str(e)}")
    
    if not api_key or not api_secret:
        logger.error("Keine API-Schlüssel gefunden. Bitte in config.json oder API_key.txt definieren.")
        return
    
    # BybitAPI initialisieren
    try:
        api = BybitAPI(api_key, api_secret)
        logger.info("BybitAPI initialisiert")
    except Exception as e:
        logger.error(f"Fehler bei der API-Initialisierung: {str(e)}")
        return
    
    # Donchian Channel Strategie initialisieren
    try:
        strategy = DonchianChannelStrategy(api, config)
        logger.info("Donchian Channel Strategie initialisiert")
    except Exception as e:
        logger.error(f"Fehler bei der Strategie-Initialisierung: {str(e)}")
        return
    
    # Testpaare auswählen (Beispiele)
    test_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "SOLUSDT",
        "MATICUSDT"
    ]
    
    # Eventuell Symbole aus Konfiguration übernehmen
    if "symbols" in config and isinstance(config["symbols"], list):
        config_symbols = [symbol for symbol in config["symbols"] if isinstance(symbol, str)]
        if config_symbols:
            test_symbols = config_symbols
            logger.info(f"Symbole aus Konfiguration übernommen: {test_symbols}")
    
    # Testintervall festlegen
    interval = "15"  # Standard: 15min
    
    # Eventuell Intervall aus Konfiguration übernehmen
    if "strategy" in config and "parameters" in config["strategy"]:
        params = config["strategy"].get("parameters", {})
        interval = params.get("timeframe", interval)
        logger.info(f"Intervall aus Konfiguration übernommen: {interval}")
    
    # Einzelsymbol-Test für aktuelle Signalanalyse
    logger.info("\n========== AKTUELLE SIGNALANALYSE ==========")
    for symbol in test_symbols:
        result = test_single_symbol(api, strategy, symbol, interval)
        print(f"\nSignal für {symbol}: {result.get('signal', 'neutral')}")
        if result.get("signal", "neutral") != "neutral":
            params = result.get("params", {})
            print(f"Einstieg: {params.get('entry_price', 0):.8f}")
            print(f"Stop-Loss: {params.get('stop_loss', 0):.8f}")
            print(f"Take-Profit: {params.get('take_profit', 0):.8f}")
            print(f"Grund: {params.get('reason', 'Keine Begründung verfügbar')}")
    
    # Backtest für ausgewählte Symbole durchführen
    logger.info("\n========== BACKTEST STARTEN ==========")
    
    for symbol in test_symbols:
        # Backtest durchführen
        df = strategy.fetch_candles(symbol, interval, 500)
        if df.empty:
            logger.warning(f"Keine Daten für Symbol {symbol}")
            continue
            
        # Daten vorbereiten
        df = strategy.prepare_data(df)
        
        # Backtest durchführen
        backtest_result = backtest_symbol(api, strategy, symbol, interval, 500)
        
        if backtest_result.get("success", False):
            # Backtest-Ergebnisse plotten
            plot_backtest_results(symbol, df, backtest_result)
    
    logger.info("Testprogramm abgeschlossen.")

if __name__ == "__main__":
    main()