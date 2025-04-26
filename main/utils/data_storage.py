import os
import csv
import json
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time

logger = logging.getLogger("data_storage")

def convert_bybit_candles(candles: List) -> List[Dict]:
    """
    Konvertiert Bybit API Kerzendaten aus dem Array-Format in ein Dictionary-Format
    
    Die Bybit API gibt Kerzendaten als Array zurück in dieser Reihenfolge:
    [timestamp, open, high, low, close, volume, turnover]
    
    Args:
        candles: Liste der Kerzendaten-Arrays von der Bybit API
        
    Returns:
        Liste von Dictionaries mit benannten Feldern
    """
    try:
        if not candles:
            return []
            
        # Define column names based on Bybit API documentation
        columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        
        # Convert each array to a dictionary
        converted_candles = []
        for candle in candles:
            # Make sure we have the right number of elements
            if isinstance(candle, list) and len(candle) >= len(columns):
                # Create dictionary with named fields
                candle_dict = {columns[i]: candle[i] for i in range(len(columns))}
                converted_candles.append(candle_dict)
            elif isinstance(candle, dict):
                # Already a dictionary, ensure it has all required keys
                if all(col in candle for col in columns[:5]):  # At least timestamp, open, high, low, close
                    converted_candles.append(candle)
                # Check for Bybit v5 API format with array data
                elif all(col in candle for col in ["data"]) and isinstance(candle["data"], list):
                    # Extract data from nested structure
                    inner_candle = candle["data"]
                    if len(inner_candle) >= 5:
                        candle_dict = {columns[i]: inner_candle[i] for i in range(min(len(columns), len(inner_candle)))}
                        converted_candles.append(candle_dict)
                # Check for 'o', 'h', 'l', 'c', 't' format
                elif all(col in candle for col in ['o', 'h', 'l', 'c']):
                    candle_dict = {
                        'timestamp': candle.get('t', candle.get('time', '')),
                        'open': candle['o'],
                        'high': candle['h'],
                        'low': candle['l'],
                        'close': candle['c'],
                        'volume': candle.get('v', candle.get('volume', 0))
                    }
                    converted_candles.append(candle_dict)
                else:
                    logger.warning(f"Invalid dictionary format for candle: {candle}")
            else:
                logger.warning(f"Invalid candle format: {candle}")
        
        logger.debug(f"Converted {len(converted_candles)} candles from Bybit Array-Format to Dictionary-Format")
        return converted_candles
    except Exception as e:
        logger.error(f"Error converting Bybit candle data: {str(e)}", exc_info=True)
        return []

def normalize_candle_data(candles: List[Dict], include_volume: bool = True) -> pd.DataFrame:
    """
    Normalisiert und bereinigt Kerzendaten von der API
    
    Args:
        candles: Liste der Kerzendaten aus der API
        include_volume: Ob Volumendaten einbezogen werden sollen
        
    Returns:
        DataFrame mit normalisierten Kerzendaten
    """
    try:
        if not candles:
            logger.warning("No candle data for normalization available")
            return pd.DataFrame()
        
        # Debug the first candle to see its structure
        if candles and len(candles) > 0:
            logger.debug(f"First candle structure: {candles[0]}")
        
        # Check if we need to convert from Bybit array format
        if isinstance(candles, list) and len(candles) > 0:
            if isinstance(candles[0], list):
                logger.debug("Detected: Bybit Array-Format, converting to Dictionary-Format")
                converted_candles = []
                
                # Define column names based on Bybit API documentation
                columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                
                for candle in candles:
                    if len(candle) >= 5:  # At least need timestamp, open, high, low, close
                        # Create dictionary with named fields
                        candle_dict = {}
                        for i, col in enumerate(columns):
                            if i < len(candle):
                                candle_dict[col] = candle[i]
                        converted_candles.append(candle_dict)
                    else:
                        logger.warning(f"Skipping insufficient candle data: {candle}")
                
                candles = converted_candles
                logger.debug(f"Converted {len(converted_candles)} candles from array format")
                
            elif isinstance(candles[0], dict):
                # Check for different dictionary formats
                
                # Format 1: o, h, l, c keys (Bybit v5 API format)
                if all(k in candles[0] for k in ['o', 'h', 'l', 'c']):
                    logger.debug("Detected: Bybit Dictionary-Format with o,h,l,c keys")
                    converted_candles = []
                    for candle in candles:
                        converted = {}
                        if 't' in candle:
                            converted['timestamp'] = candle['t']
                        elif 'time' in candle: 
                            converted['timestamp'] = candle['time']
                        elif 'start_time' in candle:
                            converted['timestamp'] = candle['start_time']
                            
                        converted['open'] = candle['o']
                        converted['high'] = candle['h']
                        converted['low'] = candle['l']
                        converted['close'] = candle['c']
                        
                        if 'v' in candle:
                            converted['volume'] = candle['v']
                        elif 'volume' in candle:
                            converted['volume'] = candle['volume']
                            
                        converted_candles.append(converted)
                    candles = converted_candles
                    logger.debug(f"Converted {len(converted_candles)} candles from o,h,l,c format")
                    
                # Format 2: Candles already have standardized columns
                elif all(k in candles[0] for k in ['open', 'high', 'low', 'close']):
                    logger.debug("Candles already in standard format")
                    # No conversion needed
                    
                # Format 3: Check for other column names that might need mapping
                else:
                    alternative_names = {
                        'open': ['openPrice', 'start', 'startPrice', 'begin', 'beginPrice'],
                        'high': ['highPrice', 'max', 'maxPrice', 'upper', 'upperPrice'],
                        'low': ['lowPrice', 'min', 'minPrice', 'lower', 'lowerPrice'],
                        'close': ['closePrice', 'end', 'endPrice', 'finish', 'finishPrice'],
                        'timestamp': ['time', 'date', 'openTime', 'start_time', 'startTime']
                    }
                    
                    # Check if we need to convert using alternative names
                    conversion_needed = False
                    for target_col in ['open', 'high', 'low', 'close']:
                        if target_col not in candles[0]:
                            for alt_name in alternative_names[target_col]:
                                if alt_name in candles[0]:
                                    conversion_needed = True
                                    break
                            if conversion_needed:
                                break
                    
                    if conversion_needed:
                        logger.debug("Found alternate column names, mapping to standard format")
                        converted_candles = []
                        for candle in candles:
                            converted = {}
                            
                            # Map each column using alternatives
                            for target_col, alt_names in alternative_names.items():
                                if target_col in candle:
                                    converted[target_col] = candle[target_col]
                                else:
                                    for alt_name in alt_names:
                                        if alt_name in candle:
                                            converted[target_col] = candle[alt_name]
                                            break
                            
                            # Only add if we have all required fields
                            if all(k in converted for k in ['open', 'high', 'low', 'close', 'timestamp']):
                                converted_candles.append(converted)
                                
                        candles = converted_candles
                        logger.debug(f"Converted {len(converted_candles)} candles using alternate column names")
            
        # Create DataFrame
        df = pd.DataFrame(candles)
        
        # Debug column names
        logger.debug(f"DataFrame columns before processing: {df.columns.tolist()}")
        
        # If we still don't have the necessary columns, try to extract them from the raw data
        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            logger.warning("Missing required columns, attempting to extract from raw data")
            
            # For each candle, extract required fields using pattern matching
            required_data = []
            for candle in candles:
                # Try to find required values in the data structure
                item = {}
                
                # Handle different possible formats
                if isinstance(candle, dict):
                    # Look for open/high/low/close values
                    for field in ["open", "high", "low", "close", "timestamp"]:
                        # Direct match
                        if field in candle:
                            item[field] = candle[field]
                        # Look for alternative names
                        elif field == "open" and any(k in candle for k in ["o", "Open", "start"]):
                            key = next(k for k in ["o", "Open", "start"] if k in candle)
                            item["open"] = candle[key]
                        elif field == "high" and any(k in candle for k in ["h", "High", "max"]):
                            key = next(k for k in ["h", "High", "max"] if k in candle)
                            item["high"] = candle[key]
                        elif field == "low" and any(k in candle for k in ["l", "Low", "min"]):
                            key = next(k for k in ["l", "Low", "min"] if k in candle)
                            item["low"] = candle[key]
                        elif field == "close" and any(k in candle for k in ["c", "Close", "end"]):
                            key = next(k for k in ["c", "Close", "end"] if k in candle)
                            item["close"] = candle[key]
                        elif field == "timestamp" and any(k in candle for k in ["t", "Time", "time", "start_time"]):
                            key = next(k for k in ["t", "Time", "time", "start_time"] if k in candle)
                            item["timestamp"] = candle[key]
                            
                    # Add volume if available
                    if include_volume:
                        if "volume" in candle:
                            item["volume"] = candle["volume"]
                        elif "v" in candle:
                            item["volume"] = candle["v"]
                            
                # If we have all required fields, add to our data
                if all(field in item for field in ["open", "high", "low", "close"]):
                    required_data.append(item)
                
            # If we extracted data successfully, create a new DataFrame
            if required_data:
                df = pd.DataFrame(required_data)
                logger.debug(f"Extracted data from raw format, new columns: {df.columns.tolist()}")
                
        # Prüfe minimal notwendige Spalten
        required_columns = ["timestamp", "open", "high", "low", "close"]
        if include_volume:
            required_columns.append("volume")
            
        # Prüfe, ob Spalten existieren, falls nicht, versuche alternative Spaltennamen zu finden
        column_mapping = {
            "timestamp": ["timestamp", "time", "openTime", "start_time", "startTime", "t"],
            "open": ["open", "openPrice", "start", "o"],
            "high": ["high", "highPrice", "max", "h"],
            "low": ["low", "lowPrice", "min", "l"],
            "close": ["close", "closePrice", "end", "c"],
            "volume": ["volume", "vol", "quantity", "baseVolume", "v"]
        }
        
        # Spaltenumbenennung anwenden
        new_columns = {}
        for target, alternatives in column_mapping.items():
            if target in required_columns:
                # Finde erste existierende Alternative
                found = False
                for alt in alternatives:
                    if alt in df.columns:
                        new_columns[alt] = target
                        found = True
                        break
                        
                if not found and target != "volume":  # Volume ist optional
                    logger.error(f"Could not find a suitable column for {target}")
                    return pd.DataFrame()
        
        # Spalten umbenennen
        if new_columns:
            df = df.rename(columns=new_columns)
            logger.debug(f"Renamed columns: {new_columns}")
        
        # Debug after renaming
        logger.debug(f"DataFrame columns after renaming: {df.columns.tolist()}")
            
        # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
        if not all(col in df.columns for col in required_columns if col != "volume"):
            missing = [col for col in required_columns if col != "volume" and col not in df.columns]
            logger.error(f"Missing columns after renaming: {missing}")
            logger.debug(f"Available columns: {df.columns.tolist()}")
            
            # Try to recreate the missing columns with default values
            if len(df) > 0:
                # If we have at least one of the required price columns, we can estimate the others
                if any(col in df.columns for col in ["open", "high", "low", "close"]):
                    existing_price_col = next(col for col in ["close", "open", "high", "low"] if col in df.columns)
                    logger.debug(f"Using {existing_price_col} to estimate missing price columns")
                    
                    for missing_col in missing:
                        if missing_col in ["open", "high", "low", "close"]:
                            df[missing_col] = df[existing_price_col]
                            logger.warning(f"Created {missing_col} column using values from {existing_price_col}")
                    
                    # Check again if we have all required columns
                    still_missing = [col for col in required_columns if col != "volume" and col not in df.columns]
                    if still_missing:
                        logger.error(f"Still missing columns after recovery attempt: {still_missing}")
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
            
        # Konvertiere Strings zu numerischen Werten
        numeric_columns = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            numeric_columns.append("volume")
            
        for col in numeric_columns:
            if col in df.columns:
                # First check if it's already numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Check if we have null values after conversion
        null_counts = df[numeric_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values after numeric conversion: {null_counts.to_dict()}")
                
        # Konvertiere Timestamp zu datetime, falls notwendig
        if "timestamp" in df.columns:
            # Prüfe, ob timestamp bereits numerisch ist
            if df["timestamp"].dtype != np.int64 and df["timestamp"].dtype != np.float64:
                # Versuche zu float zu konvertieren (könnte String sein)
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
            
            # Falls timestamp in Millisekunden ist, konvertiere zu Sekunden
            first_ts = df["timestamp"].iloc[0] if not df.empty else 0
            if first_ts > 1e10:  # Wahrscheinlich in Millisekunden
                df["timestamp"] = df["timestamp"] / 1000
                
        # Entferne Zeilen mit NaN-Werten in wichtigen Spalten
        before_dropna = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        after_dropna = len(df)
        
        if before_dropna > after_dropna:
            logger.warning(f"Removed {before_dropna - after_dropna} rows with NaN values")
        
        # Sortiere nach Timestamp (absteigend)
        if "timestamp" in df.columns and not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        
        # Final check to ensure we have all necessary columns
        if df.empty or not all(col in df.columns for col in ["open", "high", "low", "close"]):
            logger.error("Failed to create a valid DataFrame with required columns")
            if not df.empty:
                logger.debug(f"Final columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
        logger.debug(f"Successfully normalized {len(df)} candles")
        return df
        
    except Exception as e:
        logger.error(f"Error normalizing candle data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def save_candles_to_csv(symbol: str, timeframe: str, candles: List[Dict], cache_dir: str = "cache") -> str:
    """
    Speichert Kerzendaten als CSV-Datei im Cache-Verzeichnis
    
    Args:
        symbol: Trading-Symbol (z.B. "BTCUSDT")
        timeframe: Zeitintervall (z.B. "15m")
        candles: Liste der Kerzendaten
        cache_dir: Verzeichnis für den Cache
        
    Returns:
        Pfad zur gespeicherten CSV-Datei oder leer bei Fehler
    """
    try:
        # Validate input data
        if not symbol or not timeframe or not candles:
            logger.error("Invalid input parameters for save_candles_to_csv")
            return ""
            
        # Normalize timeframe format (remove 'm' or 'h' suffix if present)
        if timeframe.lower().endswith('m') or timeframe.lower().endswith('h'):
            timeframe_value = timeframe.lower().rstrip('mh')
            if timeframe.lower().endswith('h'):
                # Convert hours to minutes
                try:
                    timeframe_value = str(int(timeframe_value) * 60)
                except ValueError:
                    logger.warning(f"Invalid timeframe format: {timeframe}, using as is")
                    timeframe_value = timeframe
            timeframe = timeframe_value
        
        # Erstelle Cache-Verzeichnis, falls es noch nicht existiert
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
            
        # Dateiname im Format BTCUSDT_15_candles.csv
        filename = f"{symbol}_{timeframe}_candles.csv"
        filepath = os.path.join(cache_dir, filename)
        
        # Backup existing file if it exists (prevent data loss)
        if os.path.exists(filepath):
            try:
                backup_path = f"{filepath}.bak"
                os.replace(filepath, backup_path)
                logger.debug(f"Created backup of existing file at {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup of {filepath}: {str(e)}")
        
        # Normalisiere die Candle-Daten
        df = normalize_candle_data(candles)
        
        if df.empty:
            logger.error(f"No valid candle data for {symbol} to save")
            return ""
            
        # Make sure all required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} missing from {symbol} candle data")
                return ""
        
        # Verify data integrity before saving
        nan_count = df[required_columns].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in {symbol} data, dropping those rows")
            df = df.dropna(subset=required_columns)
            
            if df.empty:
                logger.error(f"All rows contain NaN values for {symbol}, cannot save")
                return ""
        
        # Try to save with retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Create directory if it does not exist
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                
                # Save to CSV using UTF-8 encoding
                df.to_csv(filepath, index=False, encoding='utf-8')
                
                # Verify the file was saved successfully
                if not os.path.exists(filepath):
                    logger.error(f"Could not save candle data for {symbol} - file does not exist after save")
                    if retry < max_retries - 1:
                        continue
                    return ""
                    
                file_size = os.path.getsize(filepath)
                
                logger.info(f"Candle data for {symbol} ({timeframe}) saved to {filepath}: {len(df)} valid candles, {file_size/1024:.1f} KB")
                return filepath
                
            except Exception as e:
                logger.error(f"Error saving candle data for {symbol} on attempt {retry+1}/{max_retries}: {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying save for {symbol}...")
                    time.sleep(1)  # Kurze Wartezeit
                else:
                    logger.error(f"Failed to save {symbol} data after {max_retries} attempts")
                    return ""
        
        return ""  # Wenn alle Versuche fehlgeschlagen sind
        
    except Exception as e:
        logger.error(f"Error saving candle data for {symbol}: {str(e)}", exc_info=True)
        return ""

def load_candles_from_csv(symbol: str, timeframe: str, cache_dir: str = "cache") -> List[Dict]:
    """
    Lädt gespeicherte Kerzendaten aus einer CSV-Datei
    
    Args:
        symbol: Trading-Symbol (z.B. "BTCUSDT")
        timeframe: Zeitintervall (z.B. "15m")
        cache_dir: Verzeichnis für den Cache
        
    Returns:
        Liste der Kerzendaten oder leere Liste, wenn nicht gefunden
    """
    try:
        # Normalize timeframe format (remove 'm' or 'h' suffix if present)
        if timeframe.lower().endswith('m') or timeframe.lower().endswith('h'):
            timeframe_value = timeframe.lower().rstrip('mh')
            if timeframe.lower().endswith('h'):
                # Convert hours to minutes
                try:
                    timeframe_value = str(int(timeframe_value) * 60)
                except ValueError:
                    logger.warning(f"Invalid timeframe format: {timeframe}, using as is")
                    timeframe_value = timeframe
            timeframe = timeframe_value
            
        # Dateiname im Format BTCUSDT_15_candles.csv
        filename = f"{symbol}_{timeframe}_candles.csv"
        filepath = os.path.join(cache_dir, filename)
        
        # Try alternative formats if the primary one doesn't exist
        if not os.path.exists(filepath):
            # Try with the 'm' suffix
            alt_filepath = os.path.join(cache_dir, f"{symbol}_{timeframe}m_candles.csv")
            if os.path.exists(alt_filepath):
                filepath = alt_filepath
                logger.debug(f"Using alternative file format with 'm' suffix: {filepath}")
            else:
                # Check in legacy format directory - data/ instead of cache/
                legacy_filepath = os.path.join("data", f"{symbol}_{timeframe}m.csv")
                if os.path.exists(legacy_filepath):
                    filepath = legacy_filepath
                    logger.debug(f"Using legacy file format in data/ directory: {filepath}")
                else:
                    logger.warning(f"No candle data for {symbol} ({timeframe}) found")
                    return []
            
        # Attempt to load with retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Lade die CSV-Datei
                df = pd.read_csv(filepath, encoding='utf-8')
                
                # Basic validation check - empty file 
                if df.empty:
                    logger.warning(f"CSV file for {symbol} is empty")
                    return []
                
                # Prüfe minimale Spalten
                required_columns = ["timestamp", "open", "high", "low", "close"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing columns in CSV file for {symbol}: {missing_columns}")
                    logger.debug(f"Available columns: {df.columns.tolist()}")
                    return []
                    
                # Konvertiere numerische Werte
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                # Check for NaN values after conversion
                nan_count = df[required_columns].isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in {symbol} cached data, dropping those rows")
                    df = df.dropna(subset=required_columns)
                
                # Entferne ungültige Einträge
                df = df.dropna(subset=["open", "high", "low", "close"])
                
                if df.empty:
                    logger.warning(f"No valid candle data in CSV file for {symbol} after filtering")
                    return []
                    
                # Ensure timestamp is integer for consistency
                if "timestamp" in df.columns:
                    try:
                        df["timestamp"] = df["timestamp"].astype(float).astype(int)
                    except ValueError:
                        logger.warning(f"Could not convert timestamps to integers for {symbol}, leaving as is")
                    
                # Sortiere nach Zeitstempel (absteigend für neueste zuerst)
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=False)
                    
                # Konvertiere zu Liste von Dictionaries
                candles = df.to_dict("records")
                
                # Check if we have a reasonable number of candles
                if len(candles) < 10:
                    logger.warning(f"Very few candles ({len(candles)}) in cache for {symbol}, data may be incomplete")
                
                logger.info(f"Loaded {len(candles)} candles for {symbol} ({timeframe}) from {filepath}")
                return candles
            except Exception as e:
                logger.warning(f"Error loading candle data for {symbol} on attempt {retry+1}/{max_retries}: {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying load for {symbol}...")
                    time.sleep(1)  # Short wait
                else:
                    logger.error(f"Failed to load {symbol} data after {max_retries} attempts")
                    return []
        
        return []  # If all retries failed
        
    except Exception as e:
        logger.error(f"Error loading candle data for {symbol}: {str(e)}", exc_info=True)
        return []

def save_trade_to_csv(trade_data: Dict, history_dir: str = "trade_history") -> bool:
    """
    Speichert Handelsdaten in einer CSV-Datei
    
    Args:
        trade_data: Handelsdaten (Dictionary)
        history_dir: Verzeichnis für die Handelshistorie
        
    Returns:
        True, wenn erfolgreich, sonst False
    """
    try:
        # Erstelle Verzeichnis, falls es noch nicht existiert
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        # Erstelle den Dateinamen basierend auf dem Jahr und Monat
        current_date = datetime.now()
        year_month = current_date.strftime("%Y-%m")
        filename = f"trades_{year_month}.csv"
        filepath = os.path.join(history_dir, filename)
        
        # Überprüfe, ob die Datei bereits existiert (für Header)
        file_exists = os.path.exists(filepath)
        
        # Öffne die Datei im Append-Modus
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            # Definiere die Felder, die wir speichern wollen
            fieldnames = [
                "timestamp", "symbol", "side", "price", "quantity", 
                "order_type", "status", "pnl", "leverage", "entry_price", 
                "exit_price", "stop_loss", "take_profit", "trade_id"
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Schreibe Header, wenn die Datei neu ist
            if not file_exists:
                writer.writeheader()
                
            # Bereite die Daten vor
            row = {field: trade_data.get(field, "") for field in fieldnames}
            
            # Formatiere Timestamp als lesbares Datum
            if "timestamp" in trade_data:
                timestamp = trade_data["timestamp"]
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                    row["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    
            # Generiere eindeutige Trade-ID, falls nicht vorhanden
            if not row.get("trade_id"):
                row["trade_id"] = f"trade-{int(current_date.timestamp())}"
                
            # Schreibe die Zeile
            writer.writerow(row)
            
        logger.info(f"Trade for {trade_data.get('symbol', 'Unknown')} saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving trade data: {e}")
        return False

def load_trades_from_csv(month: Optional[str] = None, history_dir: str = "trade_history") -> List[Dict]:
    """
    Lädt Handelsdaten aus CSV-Dateien
    
    Args:
        month: Monat im Format "YYYY-MM" (optional, wenn None werden alle geladen)
        history_dir: Verzeichnis für die Handelshistorie
        
    Returns:
        Liste der Handelsdaten
    """
    try:
        # Erstelle Verzeichnis, falls es noch nicht existiert
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            return []
            
        trades = []
        
        # Bestimme die zu ladenden Dateien
        if month:
            files = [f"trades_{month}.csv"]
        else:
            files = [f for f in os.listdir(history_dir) if f.startswith("trades_") and f.endswith(".csv")]
            
        for filename in files:
            filepath = os.path.join(history_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            # Lade die CSV-Datei
            with open(filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(dict(row))
                    
        return trades
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return []
        
def get_trade_statistics(month: Optional[str] = None) -> Dict:
    """
    Berechnet Statistiken für die getätigten Trades
    
    Args:
        month: Monat im Format "YYYY-MM" (optional, wenn None werden alle berücksichtigt)
        
    Returns:
        Dictionary mit Statistiken
    """
    try:
        trades = load_trades_from_csv(month)
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "net_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0
            }
            
        # Zähle die Trades
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
        losing_trades = sum(1 for t in trades if float(t.get("pnl", 0)) < 0)
        
        # Berechne Gewinne und Verluste
        profits = [float(t.get("pnl", 0)) for t in trades if float(t.get("pnl", 0)) > 0]
        losses = [float(t.get("pnl", 0)) for t in trades if float(t.get("pnl", 0)) < 0]
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = total_profit + total_loss
        
        # Bestimme Maxima und Durchschnitte
        max_profit = max(profits) if profits else 0.0
        max_loss = min(losses) if losses else 0.0
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0.0
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }
    except Exception as e:
        logger.error(f"Error calculating trade statistics: {e}")
        return {} 