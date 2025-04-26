import os
import time
import json
import random
import logging
import traceback
import requests
from typing import Dict, List, Optional, TypeVar, Generic, Callable, Any, Union
import hmac
import hashlib
from datetime import datetime, timedelta

# Monkey patch pybit WebSocket for better error handling
import pybit._websocket_stream
original_send_custom_ping = pybit._websocket_stream._WebSocketManager._send_custom_ping

def safer_send_custom_ping(self):
    """A safer version of the WebSocket ping method that handles closed connections gracefully"""
    try:
        if not hasattr(self, 'ws') or self.ws is None:
            return
        
        if hasattr(self, 'exited') and self.exited:
            return
            
        if not hasattr(self.ws, 'sock') or self.ws.sock is None:
            return
            
        if not hasattr(self.ws.sock, 'connected') or not self.ws.sock.connected:
            return
            
        # Only send ping if connection is open and active
        original_send_custom_ping(self)
    except Exception as e:
        # Ignore "Connection is already closed" errors
        if "Connection is already closed" in str(e):
            pass
        else:
            # Log other errors but don't crash
            logging.warning(f"Error in WebSocket ping: {e}")

# Apply the monkey patch
pybit._websocket_stream._WebSocketManager._send_custom_ping = safer_send_custom_ping

# Import pybit
try:
    import pybit
    from pybit.unified_trading import WebSocket
    from pybit.unified_trading import HTTP as Client
except ImportError:
    raise ImportError("PyBit is required. Install with: pip install pybit")

# Lokale Importe
from utils.logger import log_api_call, log_error, log_exception

T = TypeVar('T')

class ApiResponse(Generic[T]):
    """
    Wrapper class for API responses, providing unified error handling and validated results
    """
    def __init__(
        self, 
        success: bool, 
        data: Optional[T] = None, 
        error_code: Optional[int] = None,
        error_message: Optional[str] = None,
        raw_response: Optional[Dict] = None
    ):
        self.success = success
        self.data = data
        self.error_code = error_code
        self.error_message = error_message
        self.raw_response = raw_response
        
    @classmethod
    def success_response(cls, data: T, raw_response: Optional[Dict] = None) -> 'ApiResponse[T]':
        """Create a successful API response"""
        return cls(True, data, raw_response=raw_response)
        
    @classmethod
    def error_response(
        cls, 
        error_message: str, 
        error_code: Optional[int] = None, 
        raw_response: Optional[Dict] = None
    ) -> 'ApiResponse[T]':
        """Create an error API response"""
        return cls(False, None, error_code, error_message, raw_response)
    
    def __bool__(self) -> bool:
        """Allows usage in if-statements for quick success check"""
        return self.success
    
    def __str__(self) -> str:
        if self.success:
            return f"ApiResponse(success=True, data={self.data})"
        return f"ApiResponse(success=False, error_code={self.error_code}, error_message={self.error_message})"
    
    def unwrap(self) -> T:
        """
        Returns the data if successful, otherwise raises an error
        
        Raises:
            Exception: If the response is not a successful response
        """
        if not self.success:
            raise Exception(f"Attempt to unwrap a failed API response: {self.error_message}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """Returns the data if successful, otherwise the default value"""
        if not self.success:
            return default
        return self.data
    
    def unwrap_or_else(self, fallback_fn: Callable[[], T]) -> T:
        """Returns the data if successful, otherwise the result of the fallback function"""
        if not self.success:
            return fallback_fn()
        return self.data


class RateLimiter:
    """
    Token-Bucket-Algorithmus für API-Rate-Limiting
    """
    def __init__(self, rate: float, capacity: int):
        """
        Initialisiere den Rate-Limiter
        
        Args:
            rate: Token-Auffüllrate pro Sekunde
            capacity: Maximale Token-Kapazität
        """
        self.rate = rate  # Token pro Sekunde
        self.capacity = capacity  # Max. Token
        self.tokens = capacity  # Aktuelle Token
        self.last_refresh = time.time()
        self.lock = __import__('threading').Lock()
        
    def _refresh(self) -> None:
        """Aktualisiere verfügbare Token basierend auf der verstrichenen Zeit"""
        now = time.time()
        elapsed = now - self.last_refresh
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refresh = now
        
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Versucht, Token zu akquirieren
        
        Args:
            tokens: Anzahl benötigter Token
            
        Returns:
            True wenn Token akquiriert wurden, False sonst
        """
        with self.lock:
            self._refresh()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_token(self, tokens: int = 1) -> float:
        """
        Warte, bis Token verfügbar sind, und gib die Wartezeit zurück
        
        Args:
            tokens: Anzahl benötigter Token
            
        Returns:
            Wartezeit in Sekunden
        """
        with self.lock:
            self._refresh()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
                
            # Berechne benötigte Wartezeit
            required = tokens - self.tokens
            wait_time = required / self.rate
            
            # Simuliere Warten und Token-Aktualisierung
            self.tokens = 0
            self.last_refresh = time.time() + wait_time
            return wait_time
                

class BybitAPI:
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.logger = logging.getLogger("api")
        
        # Check if first parameter is a config dictionary
        if isinstance(api_key, dict):
            config = api_key
            api_key = config.get("api", {}).get("api_key")
            api_secret = config.get("api", {}).get("api_secret")
            testnet = config.get("api", {}).get("testnet", False)
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limiter = RateLimiter(rate=2.0, capacity=120)  # 2 requests per second average
        self.ws_private = None
        self.ws_public = None
        self.ws_callbacks = {}
        self.user_callbacks = {}
        self.callbacks_by_params = {}
        self.testnet = testnet
        
        # Add time offset tracking to handle server-client time differences
        self.time_offset = 0  # Milliseconds
        
        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        # Get API credentials from environment if not provided
        if not self.api_key or not self.api_secret:
            self.api_key = os.environ.get("BYBIT_API_KEY", "")
            self.api_secret = os.environ.get("BYBIT_API_SECRET", "")
            self.logger.warning("API key or secret not found in constructor, trying environment variables")
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("API key or secret not found in environment variables")
            
        self.logger = logging.getLogger("api.bybit_api")
        self.logger.info(f"Bybit API initialized (testnet: {testnet})")
        
        # Initialize pybit client
        try:
            self.session = Client(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            self.logger.info("Bybit API session created successfully")
        except Exception as e:
            error_msg = f"Error initializing Bybit API: {e}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "BybitAPI initialization", traceback.format_exc())
            self.session = None
        
        self.ws_public = None
        self.ws_private = None
        
        # Track active subscriptions
        self._active_subscriptions = {
            "public": {},
            "private": {}
        }
        
        # Data storage for websocket data
        self._ws_data = {
            "kline": {},
            "orderbook": {},
            "ticker": {},
            "trade": {},
            "position": {},
            "order": {},
            "execution": {}
        }
        
        # WebSocket reconnection settings
        self.ws_reconnect_interval = 30  # seconds
        self.ws_ping_interval = 20  # seconds
        
        # Set up retry parameters
        self.max_retries = 3
        self.retry_delay = 2
        
        # Rate limiting
        # Bybit erlaubt 120 Anfragen pro Minute für normale Endpunkte
        self.rate_limiter = RateLimiter(rate=2.0, capacity=120)  # 2 Anfragen pro Sekunde durchschnittlich
        
        # Cache-Steuerung
        self.cache_enabled = True
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        self.cache_max_age_hours = 12
        self.cache_refresh_threshold_hours = 4
        
        # Stelle sicher, dass Cache-Verzeichnis existiert
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_all_symbols(self, category: str = "linear") -> List[str]:
        """
        Ruft alle verfügbaren Handelssymbole ab
        
        Args:
            category: Kategorie der Symbole (linear für USDT-Futures, usw.)
            
        Returns:
            Liste aller verfügbaren Symbole
        """
        try:
            self.logger.info(f"Hole alle verfügbaren Symbole für Kategorie: {category}")
            
            response = self._api_call_with_retry(
                self.session.get_instruments_info,
                category=category
            )
            
            if not response.success:
                self.logger.error(f"Fehler beim Abrufen der Symbole: {response.error_message}")
                return []
            
            symbols = []
            
            # Extrahiere die Symbole aus der Antwort
            instruments_list = response.data.get("list", [])
            for item in instruments_list:
                symbol = item.get("symbol")
                if symbol:
                    symbols.append(symbol)
            
            self.logger.info(f"{len(symbols)} Symbole erfolgreich abgerufen")
            return symbols
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen aller Symbole: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "get_all_symbols", traceback.format_exc())
            return []

    def initialize(self):
        """
        Initialize API connection and WebSocket
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing Bybit API with Unified V5 endpoints...")
            self.is_running = True  # Set running state flag
            
            # First, synchronize time with server for accurate authentication
            time_sync_success = self._sync_time_with_server()
            if not time_sync_success:
                self.logger.warning("Time synchronization with server failed. API authentication may be unreliable.")
            
            # Validate API credentials if provided
            if self.api_key and self.api_secret:
                self.logger.info("Validating API credentials...")
                # Try a simple authenticated API call
                account_info = self._make_request("GET", "/v5/account/info", {})
                
                if account_info and account_info.get("retCode") == 0:
                    self.logger.info("API credentials validated successfully")
                else:
                    error_msg = account_info.get("retMsg", "Unknown error") if account_info else "Failed to get account info"
                    self.logger.warning(f"API credential validation failed: {error_msg}")
                    self.logger.warning("Continuing with limited functionality. Some API features may not work.")
            else:
                self.logger.warning("No API credentials provided. Only public endpoints will be available.")
            
            # Get server time to confirm connection
            server_time_resp = self.get_server_time()
            
            if not server_time_resp:
                self.logger.error("Failed to connect to Bybit API")
                return False
                
            self.logger.info(f"Successfully connected to Bybit API. Server time: {server_time_resp}")
            
            # Connect WebSocket if API credentials were provided
            if self.api_key and self.api_secret:
                # Make multiple attempts to connect WebSocket
                for attempt in range(3):
                    self.logger.info(f"Attempting to connect WebSocket (attempt {attempt+1}/3)...")
                    ws_success = self.connect_ws()
                    if ws_success:
                        self.logger.info("WebSocket connected successfully")
                        self._setup_ws_monitoring()
                        break
                    else:
                        self.logger.warning(f"WebSocket connection attempt {attempt+1}/3 failed")
                        if attempt < 2:  # Not the last attempt
                            time.sleep(3)  # Wait before retrying
                
                if not hasattr(self, 'ws_private') or not self.ws_private:
                    self.logger.warning("Failed to connect to WebSocket API. Some real-time features will be disabled.")
            else:
                self.logger.warning("No API credentials provided. WebSocket features disabled.")
            
            # Clean up cache
            try:
                self.manage_cache(max_age_days=7, max_cache_size_mb=2000)
            except Exception as cache_err:
                self.logger.warning(f"Cache management error: {cache_err}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Bybit API: {e}")
            log_exception(self.logger, e, "BybitAPI initialization", traceback.format_exc())
            return False

    def connect_ws(self):
        """Connect to the WebSocket API"""
        try:
            self.logger.info("Connecting to WebSocket API...")
            
            # Verify API credentials before attempting to connect
            if not self.api_key or not self.api_secret:
                self.logger.error("Cannot connect to WebSocket: API key or secret missing")
                return False
            
            # First synchronize time with server
            self._sync_time_with_server()
            
            # Create websocket connection with only supported parameters
            try:
                import pybit
                from pybit.unified_trading import WebSocket
                version = getattr(pybit, "__version__", "unknown")
                self.logger.info(f"Using pybit version: {version}")
                
                # Handle any possibly existing connection first
                if hasattr(self, 'ws_private') and self.ws_private:
                    try:
                        # Mark as exited before actually exiting
                        if hasattr(self.ws_private, '_client'):
                            self.ws_private._client.exited = True
                            
                        self.ws_private.exit()
                        self.logger.info("Closed existing WebSocket connection")
                    except Exception as e:
                        self.logger.warning(f"Error closing existing WebSocket: {e}")
                    self.ws_private = None
                    # Add delay for cleanup
                    import time
                    time.sleep(2)
                
                # Prepare authentication timestamp with time offset correction
                expires = int(time.time() * 1000) + self.time_offset + 10000  # 10 seconds expiry
                
                # Create signature
                signature_payload = f"GET/realtime{expires}"
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    signature_payload.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                self.logger.debug(f"WebSocket authentication: timestamp={expires}, signature={signature[:5]}...")
                
                # Try using a more robust error handling mechanism and cleaner instantiation
                try:
                    # Create a unified basic WebSocket connection
                    # Using only the minimal required parameters to avoid compatibility issues
                    self.ws_private = WebSocket(
                        testnet=self.testnet,
                        channel_type="private",
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        # Explicitly use system time offset to avoid auth issues
                        ts_init=expires,
                        sign_init=signature
                    )
                    
                    # Give time for connection to establish
                    import time
                    time.sleep(2)
                    
                except TypeError as type_error:
                    # If we get a TypeError, try with even more minimal parameters
                    error_msg = str(type_error)
                    self.logger.warning(f"WebSocket initialization error: {error_msg}")
                    
                    # Try again with minimal parameters
                    self.ws_private = WebSocket(
                        testnet=self.testnet,
                        channel_type="private",
                        api_key=self.api_key,
                        api_secret=self.api_secret
                    )
                    time.sleep(2)
                
                # Verify WebSocket connection before subscribing
                is_connected = False
                if hasattr(self.ws_private, 'ws') and self.ws_private.ws:
                    try:
                        is_connected = self.ws_private.ws.sock and self.ws_private.ws.sock.connected
                    except:
                        is_connected = False
                
                if not is_connected:
                    self.logger.error("WebSocket connection failed: No active connection")
                    return False
                
                # Add a small delay to ensure connection is fully established
                time.sleep(3)
                
                # Capture potential subscription errors
                subscription_errors = []
                
                # Wrap each subscription in a try-except
                for stream_type in ["position", "execution", "order", "wallet"]:
                    try:
                        if stream_type == "position":
                            self.ws_private.position_stream(callback=self._on_position_update)
                        elif stream_type == "execution":
                            self.ws_private.execution_stream(callback=self._on_execution_update)
                        elif stream_type == "order":
                            self.ws_private.order_stream(callback=self._on_order_update)
                        elif stream_type == "wallet":
                            self.ws_private.wallet_stream(callback=self._on_wallet_update)
                        self.logger.info(f"Successfully subscribed to {stream_type} stream")
                        # Add a small delay between subscriptions
                        time.sleep(0.5)
                    except Exception as sub_err:
                        subscription_errors.append(f"{stream_type} stream: {str(sub_err)}")
                        self.logger.warning(f"Error subscribing to {stream_type} stream: {sub_err}")
                
                if subscription_errors:
                    self.logger.warning(f"Some WebSocket subscriptions failed: {', '.join(subscription_errors)}")
                    # If all subscriptions failed, return False
                    if len(subscription_errors) >= 4:
                        self.logger.error("All WebSocket subscriptions failed")
                        return False
                else:
                    self.logger.info("Successfully subscribed to WebSocket streams")
                    
                return True
                
            except Exception as ws_err:
                self.logger.error(f"Error creating WebSocket connection: {ws_err}")
                if "unexpected keyword argument" in str(ws_err):
                    self.logger.error("Parameter compatibility issue detected with pybit. Try simplifying connection parameters.")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to WebSocket: {e}")
            return False
            
    def _sync_time_with_server(self):
        """Synchronize local time with server time"""
        try:
            self.logger.info("Time synchronizing with server...")
            
            # Use up to 3 retries to ensure successful time sync
            for attempt in range(3):
                try:
                    # Directly use the simpler REST call for time synchronization
                    response = self._make_request("GET", "/v5/market/time", {})
                    
                    if response and "result" in response and "timeSecond" in response["result"]:
                        server_time_ms = int(response["result"]["timeSecond"]) * 1000
                        local_time_ms = int(time.time() * 1000)
                        
                        # Calculate time difference and set offset
                        time_diff = server_time_ms - local_time_ms
                        self.time_offset = time_diff
                        
                        self.logger.info(f"Time synchronized with server. Offset: {time_diff} ms")
                        return True
                    else:
                        self.logger.warning(f"Invalid server time response: {response}")
                        if attempt < 2:  # Not the last attempt
                            time.sleep(1)  # Wait before retrying
                except Exception as e:
                    self.logger.warning(f"Error in time sync attempt {attempt+1}: {e}")
                    if attempt < 2:  # Not the last attempt
                        time.sleep(1)  # Wait before retrying
            
            self.logger.error("Failed to synchronize time with server after multiple attempts")
            return False
        except Exception as e:
            self.logger.error(f"Error synchronizing time with server: {e}")
            return False
    
    def _api_call_with_retry(self, func, max_retries: int = 3, retry_delay: int = 2, *args, **kwargs) -> ApiResponse:
        """
        Call API function with retry logic
        
        Args:
            func: Function to call
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            ApiResponse object containing standardized response information
        """
        endpoint = func.__name__ if hasattr(func, "__name__") else "unknown_endpoint"
        method = "POST" if any(k in kwargs for k in ["side", "orderType", "qty"]) else "GET"
        
        # Log the API call attempt
        self.logger.debug(f"Calling API: {endpoint} with params: {kwargs}")
        
        # Warten auf Rate-Limiting, falls nötig
        wait_time = self.rate_limiter.wait_for_token()
        if wait_time > 0:
            self.logger.debug(f"Rate-Limiting aktiv, warte {wait_time:.2f}s vor API-Aufruf {endpoint}")
            time.sleep(wait_time)
        
        for attempt in range(max_retries):
            try:
                # Record start time for latency measurement
                start_time = time.time()
                response = func(*args, **kwargs)
                # Calculate latency
                latency = (time.time() - start_time) * 1000  # in milliseconds
                
                # Check for successful response - new API format has retCode
                if response and "retCode" in response:
                    if response["retCode"] == 0:
                        # Log successful API call
                        log_api_call(
                            self.logger, 
                            endpoint, 
                            method, 
                            kwargs, 
                            response=response
                        )
                        
                        self.logger.debug(f"API call {endpoint} successful in {latency:.2f}ms")
                        return ApiResponse.success_response(
                            response.get("result", {}),
                            raw_response=response
                        )
                    else:
                        error_msg = response.get("retMsg", "Unknown error")
                        self.logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                        
                        # Prüfen auf Rate-Limiting-Fehler und entsprechend handeln
                        if "too many requests" in error_msg.lower() or response.get("retCode") in [10006, 10016]:
                            retry_delay_with_backoff = retry_delay * (2 ** attempt)  # Exponential backoff
                            self.logger.warning(f"Rate limit exceeded, backing off for {retry_delay_with_backoff}s")
                            time.sleep(retry_delay_with_backoff)
                        elif attempt < max_retries - 1:
                            # Add some jitter to avoid all retries happening at the same time
                            jitter = random.uniform(0, 1)
                            time.sleep(retry_delay + jitter)
                        else:
                            error_log = f"API call {endpoint} failed after {max_retries} attempts: {error_msg}"
                            self.logger.error(error_log)
                            log_api_call(
                                self.logger, 
                                endpoint, 
                                method, 
                                kwargs, 
                                error=error_msg
                            )
                            return ApiResponse.error_response(
                                error_message=error_msg,
                                error_code=response.get("retCode"),
                                raw_response=response
                            )
                # Legacy response format handling
                elif response and isinstance(response, dict):
                    # Log successful API call
                    log_api_call(
                        self.logger, 
                        endpoint, 
                        method, 
                        kwargs, 
                        response=response
                    )
                    
                    self.logger.debug(f"API call {endpoint} successful in {latency:.2f}ms (legacy format)")
                    return ApiResponse.success_response(response, raw_response=response)
                else:
                    error_msg = "Invalid response format"
                    self.logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                    
                    if attempt < max_retries - 1:
                        # Add some jitter to avoid all retries happening at the same time
                        jitter = random.uniform(0, 1)
                        time.sleep(retry_delay + jitter)
                    else:
                        error_log = f"API call {endpoint} failed after {max_retries} attempts: {error_msg}"
                        self.logger.error(error_log)
                        log_api_call(
                            self.logger, 
                            endpoint, 
                            method, 
                            kwargs, 
                            error=error_msg
                        )
                        return ApiResponse.error_response(error_message=error_msg)
            
            except Exception as e:
                self.logger.warning(f"API call exception (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Add some jitter to avoid all retries happening at the same time
                    jitter = random.uniform(0, 1)
                    time.sleep(retry_delay + jitter)
                else:
                    error_log = f"API call {endpoint} failed after {max_retries} attempts with exception: {str(e)}"
                    self.logger.error(error_log)
                    log_exception(
                        self.logger,
                        e,
                        f"API call to {endpoint}",
                        traceback.format_exc()
                    )
                    log_api_call(
                        self.logger, 
                        endpoint, 
                        method, 
                        kwargs, 
                        error=str(e)
                    )
                    return ApiResponse.error_response(error_message=str(e))
                    
        return ApiResponse.error_response(error_message="Max retries exceeded")
    
    def manage_cache(self, max_age_days: int = 7, max_cache_size_mb: int = 1000) -> ApiResponse[dict]:
        """
        Verwaltet Cache-Dateien durch Entfernen alter oder übermäßig großer Dateien
        
        Args:
            max_age_days: Maximales Alter von Cache-Dateien in Tagen
            max_cache_size_mb: Maximale Gesamtgröße des Caches in MB
            
        Returns:
            ApiResponse mit Cache-Statistiken
        """
        try:
            cache_dir = self.cache_dir
            if not os.path.exists(cache_dir):
                self.logger.info("Cache directory does not exist, creating it")
                os.makedirs(cache_dir, exist_ok=True)
                return ApiResponse.success_response({
                    "status": "created",
                    "removed_files": 0,
                    "freed_space_mb": 0,
                    "total_size_mb": 0
                })
            
            self.logger.info(f"Managing cache files (max age: {max_age_days} days, max size: {max_cache_size_mb} MB)")
            
            # Get all cache files
            cache_files = []
            for file in os.listdir(cache_dir):
                if file.endswith('_candles.csv'):
                    file_path = os.path.join(cache_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    file_age = (time.time() - os.path.getmtime(file_path)) / (24 * 3600)  # days
                    cache_files.append({
                        'path': file_path,
                        'size': file_size,
                        'age': file_age,
                        'name': file
                    })
            
            # Remove old files
            removed_files = 0
            freed_space = 0
            for file in cache_files:
                if file['age'] > max_age_days:
                    self.logger.debug(f"Removing old cache file: {file['name']} (age: {file['age']:.1f} days)")
                    os.remove(file['path'])
                    removed_files += 1
                    freed_space += file['size']
            
            # Calculate total cache size
            remaining_files = [f for f in cache_files if f['age'] <= max_age_days]
            total_size = sum(f['size'] for f in remaining_files)
            
            # If still over the limit, remove least recently used files
            if total_size > max_cache_size_mb:
                # Sort by last modified time (oldest first)
                remaining_files.sort(key=lambda x: os.path.getmtime(x['path']))
                
                # Remove files until we're under the limit
                for file in remaining_files:
                    if total_size <= max_cache_size_mb:
                        break
                    self.logger.debug(f"Removing excess cache file: {file['name']} to reduce cache size")
                    os.remove(file['path'])
                    total_size -= file['size']
                    removed_files += 1
                    freed_space += file['size']
            
            self.logger.info(f"Cache cleanup: removed {removed_files} files, freed {freed_space:.2f} MB")
            
            return ApiResponse.success_response({
                "status": "cleaned",
                "removed_files": removed_files,
                "freed_space_mb": round(freed_space, 2),
                "total_size_mb": round(total_size, 2)
            })
        except Exception as e:
            self.logger.error(f"Error managing cache: {e}")
            return ApiResponse.error_response(error_message=str(e))
    
    # Market Data API Calls
    def get_kline(self, symbol: str, interval: str, category: str = "linear", limit: int = 200, start_time: int = None, end_time: int = None) -> ApiResponse[list]:
        """
        Fetches candlestick data (OHLCV) for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT')
            interval: Time interval ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'M', 'W')
            category: Category ('linear', 'spot', 'inverse')
            limit: Maximum number of candles (default 200, max 1000)
            start_time: Start time in Unix-timestamp (ms)
            end_time: End time in Unix-timestamp (ms)
            
        Returns:
            ApiResponse with list of candles [time, open, high, low, close, volume, ...]
        """
        try:
            import threading
            import queue
            import pandas as pd
            
            # Check for cached request
            use_cache = self.cache_enabled and not start_time and not end_time
            cache_file = None
            
            if use_cache:
                # Construct cache filename
                cache_file = os.path.join(self.cache_dir, f"{symbol}_{interval}_candles.csv")
                
                # Check if cache file exists and is not too old
                if os.path.exists(cache_file):
                    file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
                    if file_age_hours < self.cache_refresh_threshold_hours:
                        # Use cache if it's fresh enough
                        self.logger.debug(f"Using cached candle data for {symbol} ({interval}), age: {file_age_hours:.1f}h")
                        # Load data from CSV
                        try:
                            df = pd.read_csv(cache_file, header=0)
                            candles = df.values.tolist()
                            # Ensure consistent format with API response
                            return ApiResponse.success_response({
                                "category": category,
                                "symbol": symbol,
                                "list": candles
                            })
                        except Exception as cache_err:
                            self.logger.warning(f"Error reading cache: {cache_err}, fetching from API")
            
            # Prepare API parameters
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["start"] = start_time
                
            if end_time:
                params["end"] = end_time
            
            # Result queue and thread mechanism for timeout handling
            result_queue = queue.Queue()
            timeout = 10  # 10 seconds timeout
            
            def api_call_thread():
                try:
                    # Use the direct request method rather than session method for better compatibility
                    endpoint = "/v5/market/kline"
                    response = self._make_request("GET", endpoint, params)
                    
                    # Check if the response looks valid
                    if response and isinstance(response, dict) and "retCode" in response:
                        if response["retCode"] == 0 and "result" in response:
                            # Success case - format the response for our ApiResponse wrapper
                            result_data = response["result"]
                            result_queue.put(ApiResponse.success_response(result_data, raw_response=response))
                        else:
                            # Error case
                            error_msg = response.get("retMsg", "Unknown error") 
                            result_queue.put(ApiResponse.error_response(
                                error_message=error_msg,
                                error_code=response.get("retCode"),
                                raw_response=response
                            ))
                    else:
                        # Unexpected response format
                        result_queue.put(ApiResponse.error_response(
                            error_message=f"Invalid response format: {response}",
                            raw_response=response
                        ))
                except Exception as e:
                    result_queue.put(ApiResponse.error_response(f"Thread error: {str(e)}"))
            
            # Start API call in separate thread
            thread = threading.Thread(target=api_call_thread)
            thread.daemon = True
            thread.start()
            
            # Wait up to timeout seconds for a response
            try:
                response = result_queue.get(timeout=timeout)
                self.logger.debug(f"API response for {symbol} ({interval}) received")
            except queue.Empty:
                self.logger.warning(f"Timeout fetching candle data for {symbol} ({interval})")
                return ApiResponse.error_response(f"API timeout after {timeout} seconds")
            
            # If successful and we should cache
            if response.success and use_cache:
                try:
                    # Process data for caching
                    data_list = response.data.get("list", [])
                    
                    # If no data returned, return empty list error
                    if not data_list:
                        self.logger.warning(f"No candle data available for {symbol} ({interval})")
                        return ApiResponse.error_response(f"No candle data for {symbol} available")
                    
                    # Store candle data as DataFrame
                    column_names = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                    
                    # Check data format and convert appropriately
                    if isinstance(data_list[0], list):
                        # Numeric indices - use column_names mapping
                        df = pd.DataFrame(data_list, columns=column_names)
                    elif isinstance(data_list[0], dict):
                        # Dict-like data - extract and create dataframe
                        df = pd.DataFrame(data_list)
                    else:
                        self.logger.warning(f"Unknown data format for {symbol}: {type(data_list[0])}")
                        df = pd.DataFrame(data_list)
                        
                    # Save to cache file
                    df.to_csv(cache_file, index=False)
                    self.logger.debug(f"Candle data for {symbol} ({interval}) saved to cache")
                except Exception as cache_err:
                    self.logger.warning(f"Could not save candle data to cache: {cache_err}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error fetching kline data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_kline for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_orderbook(self, symbol: str, limit: int = 50) -> ApiResponse[dict]:
        """
        Ruft das aktuelle Orderbuch für ein Symbol ab
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            limit: Tiefe des Orderbuches (max 200)
            
        Returns:
            ApiResponse mit Orderbuch-Daten {bids: [...], asks: [...]}
        """
        try:
            response = self._api_call_with_retry(
                self.session.get_orderbook, 
                category="linear", 
                symbol=symbol,
                limit=limit
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen des Orderbooks für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_orderbook for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_tickers(self, symbol=None, category='spot'):
        """
        Get tickers for a specific symbol or all symbols.
        
        Args:
            symbol (str, optional): The trading pair. Defaults to None (all symbols).
            category (str, optional): Product type. Defaults to 'spot'.
            
        Returns:
            dict: The response from the API.
        """
        
        params = {'category': category}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request("GET", "/v5/market/tickers", params)
    
    def get_instrument_info(self, symbol: str = None) -> ApiResponse[list]:
        """
        Ruft Instrument-Informationen für ein Symbol oder alle Symbole ab
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle Symbole.
            
        Returns:
            ApiResponse mit Liste von Instrument-Informationen
        """
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.get_instruments_info,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Instrumenteninformationen für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_instrument_info for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_funding_rate(self, symbol: str, start_time: int = None, end_time: int = None, limit: int = 200) -> ApiResponse[list]:
        """
        Ruft historische Funding-Raten für ein Symbol ab
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            start_time: Optional - Startzeit in Unix-Timestamp (ms)
            end_time: Optional - Endzeit in Unix-Timestamp (ms)
            limit: Maximale Anzahl der Datensätze (default 200)
            
        Returns:
            ApiResponse mit Liste von Funding-Raten
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = start_time
                
            if end_time:
                params["endTime"] = end_time
                
            response = self._api_call_with_retry(
                self.session.get_funding_rate_history,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Funding-Raten für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_funding_rate for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    # Account & Position Methods
    def get_wallet_balance(self, coin: str = "USDT") -> ApiResponse[Dict]:
        """
        Ruft den Kontostand des Wallets ab
        
        Args:
            coin: Währung (z.B. 'USDT', 'BTC')
            
        Returns:
            ApiResponse mit Kontoinformationen
        """
        try:
            params = {
                "accountType": "UNIFIED",  # Aktualisiert von CONTRACT zu UNIFIED für API V5
                "coin": coin
            }
            
            response = self._api_call_with_retry(
                self.session.get_wallet_balance,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen des Kontostands: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "get_wallet_balance", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_positions(self, symbol: str = None, settle_coin: str = "USDT") -> ApiResponse[list]:
        """
        Ruft Informationen zu offenen Positionen ab
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle Positionen.
            settle_coin: Abrechnungswährung (z.B. 'USDT')
            
        Returns:
            ApiResponse mit Liste offener Positionen
        """
        try:
            params = {
                "category": "linear",
                "settleCoin": settle_coin
            }
            
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.get_positions,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Positionen für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_positions for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_leverage(self, symbol: str) -> ApiResponse[Dict]:
        """
        Ruft die aktuelle Hebeleinstellung für ein Symbol ab
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            
        Returns:
            ApiResponse mit Hebeleinstellungen
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            response = self._api_call_with_retry(
                self.session.get_leverage,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen des Hebels für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_leverage for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def set_leverage(self, symbol: str, leverage: int, leverage_side: str = "Both") -> ApiResponse[Dict]:
        """
        Setzt den Hebel für ein Symbol
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            leverage: Hebelwert (1-100, abhängig von Symbol)
            leverage_side: Hebelseite ('Buy', 'Sell', 'Both')
            
        Returns:
            ApiResponse mit Ergebnis der Hebeländerung
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            
            response = self._api_call_with_retry(
                self.session.set_leverage,
                **params
            )
            
            self.logger.info(f"Hebel für {symbol} auf {leverage}x gesetzt")
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Setzen des Hebels für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"set_leverage for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_open_orders(self, symbol: str = None) -> ApiResponse[List]:
        """
        Ruft offene Orders ab
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle offenen Orders.
            
        Returns:
            ApiResponse mit Liste offener Orders
        """
        try:
            params = {
                "category": "linear",
                "orderStatus": "New,PartiallyFilled"
            }
            
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.get_open_orders,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen offener Orders für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_open_orders for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_order_history(self, symbol: str = None, limit: int = 50, order_status: str = "Filled") -> ApiResponse[List]:
        """
        Ruft Order-Historie ab
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle Symbole.
            limit: Maximale Anzahl der Ergebnisse
            order_status: Orderstatus-Filter ('Filled', 'Cancelled', 'Rejected', etc.)
            
        Returns:
            ApiResponse mit Liste historischer Orders
        """
        try:
            params = {
                "category": "linear",
                "limit": limit,
                "orderStatus": order_status
            }
            
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.get_order_history,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Order-Historie für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_order_history for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def get_executions(self, symbol: str = None, limit: int = 50) -> ApiResponse[List]:
        """
        Ruft Ausführungsberichte (getätigte Trades) ab
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle Symbole.
            limit: Maximale Anzahl der Ergebnisse
            
        Returns:
            ApiResponse mit Liste von Ausführungsberichten
        """
        try:
            params = {
                "category": "linear",
                "limit": limit
            }
            
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.get_executions,
                **params
            )
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Abrufen der Ausführungsberichte für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"get_executions for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    # Trading Operations
    def place_order(self, 
                    symbol: str, 
                    side: str, 
                    order_type: str, 
                    qty: float, 
                    price: float = None,
                    time_in_force: str = "GTC",
                    reduce_only: bool = False,
                    close_on_trigger: bool = False,
                    stop_loss: float = None,
                    take_profit: float = None,
                    position_idx: int = 0,
                    order_link_id: str = None,
                    tp_trigger_by: str = "LastPrice",
                    sl_trigger_by: str = "LastPrice",
                    tp_limit_price: float = None,
                    sl_limit_price: float = None) -> ApiResponse[Dict]:
        """
        Platziert eine neue Order
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            side: Handelsrichtung ('Buy' oder 'Sell')
            order_type: Ordertyp ('Limit', 'Market', 'Limit_maker')
            qty: Ordermenge
            price: Orderpreis (erforderlich für Limit-Orders)
            time_in_force: Gültigkeitsdauer der Order ('GTC', 'IOC', 'FOK')
            reduce_only: Wenn True, reduziert die Order nur bestehende Positionen
            close_on_trigger: Wenn True, schließt die Order die Position
            stop_loss: Stop-Loss-Preis
            take_profit: Take-Profit-Preis
            position_idx: Positionsindex (0: One-Way-Mode, 1: Buy, 2: Sell für Hedge-Mode)
            order_link_id: Benutzerdefinierte Order-ID
            tp_trigger_by: Take-Profit-Auslösetyp
            sl_trigger_by: Stop-Loss-Auslösetyp
            tp_limit_price: Limit-Preis für Take-Profit-Markt-Order
            sl_limit_price: Limit-Preis für Stop-Loss-Markt-Order
            
        Returns:
            ApiResponse mit Order-Details
        """
        try:
            # Normalisieren und validieren der Parameter
            side = side.capitalize()
            order_type = order_type.capitalize()
            
            if side not in ["Buy", "Sell"]:
                return ApiResponse.error_response(f"Ungültige Order-Seite: {side}. Muss 'Buy' oder 'Sell' sein.")
                
            if order_type not in ["Limit", "Market", "Limit_maker"]:
                return ApiResponse.error_response(f"Ungültiger Ordertyp: {order_type}. Muss 'Limit', 'Market' oder 'Limit_maker' sein.")
                
            if order_type in ["Limit", "Limit_maker"] and price is None:
                return ApiResponse.error_response(f"Preis muss für Order-Typ {order_type} angegeben werden.")
                
            # Parameter vorbereiten
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": time_in_force,
                "reduceOnly": reduce_only,
                "closeOnTrigger": close_on_trigger,
                "positionIdx": position_idx
            }
            
            # Optionale Parameter
            if price is not None:
                params["price"] = str(price)
                
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
                params["slTriggerBy"] = sl_trigger_by
                if sl_limit_price is not None:
                    params["slOrderPrice"] = str(sl_limit_price)
                
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
                params["tpTriggerBy"] = tp_trigger_by
                if tp_limit_price is not None:
                    params["tpOrderPrice"] = str(tp_limit_price)
                
            if order_link_id is not None:
                params["orderLinkId"] = order_link_id
            else:
                # Generiere einen eindeutigen Order-Link-ID
                now = datetime.now()
                random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
                params["orderLinkId"] = f"{symbol}_{side}_{now.strftime('%Y%m%d%H%M%S')}_{random_suffix}"
            
            # API-Call durchführen
            self.logger.info(f"Platziere {side} {order_type}-Order für {symbol}: {qty} @ {price if price else 'Market'}")
            
            response = self._api_call_with_retry(
                self.session.place_order,
                **params
            )
            
            if response.success:
                order_id = response.data.get("orderId")
                self.logger.info(f"Order für {symbol} erfolgreich platziert: Order-ID {order_id}")
            else:
                self.logger.warning(f"Fehler bei der Order-Platzierung für {symbol}: {response.error_message}")
                
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Platzieren einer Order für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"place_order for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def place_market_order(self, 
                          symbol: str, 
                          side: str, 
                          qty: float,
                          reduce_only: bool = False,
                          stop_loss: float = None,
                          take_profit: float = None,
                          position_idx: int = 0) -> ApiResponse[Dict]:
        """
        Platziert eine Market-Order (Vereinfachungsmethode)
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            side: Handelsrichtung ('Buy' oder 'Sell')
            qty: Ordermenge
            reduce_only: Wenn True, reduziert die Order nur bestehende Positionen
            stop_loss: Stop-Loss-Preis
            take_profit: Take-Profit-Preis
            position_idx: Positionsindex (0: One-Way-Mode, 1: Buy, 2: Sell für Hedge-Mode)
            
        Returns:
            ApiResponse mit Order-Details
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=qty,
            price=None,
            reduce_only=reduce_only,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_idx=position_idx
        )
    
    def place_limit_order(self, 
                         symbol: str, 
                         side: str, 
                         qty: float,
                         price: float,
                         time_in_force: str = "GTC",
                         reduce_only: bool = False,
                         stop_loss: float = None,
                         take_profit: float = None,
                         position_idx: int = 0) -> ApiResponse[Dict]:
        """
        Platziert eine Limit-Order (Vereinfachungsmethode)
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            side: Handelsrichtung ('Buy' oder 'Sell')
            qty: Ordermenge
            price: Orderpreis
            time_in_force: Gültigkeitsdauer der Order ('GTC', 'IOC', 'FOK')
            reduce_only: Wenn True, reduziert die Order nur bestehende Positionen
            stop_loss: Stop-Loss-Preis
            take_profit: Take-Profit-Preis
            position_idx: Positionsindex (0: One-Way-Mode, 1: Buy, 2: Sell für Hedge-Mode)
            
        Returns:
            ApiResponse mit Order-Details
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type="Limit",
            qty=qty,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_idx=position_idx
        )
    
    def modify_order(self,
                    symbol: str,
                    order_id: str = None,
                    order_link_id: str = None,
                    price: float = None,
                    qty: float = None,
                    take_profit: float = None,
                    stop_loss: float = None) -> ApiResponse[Dict]:
        """
        Ändert eine bestehende Order
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            order_id: Bybit Order-ID
            order_link_id: Benutzerdefinierte Order-ID
            price: Neuer Preis
            qty: Neue Menge
            take_profit: Neuer Take-Profit-Preis
            stop_loss: Neuer Stop-Loss-Preis
            
        Returns:
            ApiResponse mit Order-Details
        """
        try:
            if not order_id and not order_link_id:
                return ApiResponse.error_response("Entweder order_id oder order_link_id muss angegeben werden.")
                
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
                
            if order_link_id:
                params["orderLinkId"] = order_link_id
                
            if price is not None:
                params["price"] = str(price)
                
            if qty is not None:
                params["qty"] = str(qty)
                
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
                
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
                
            response = self._api_call_with_retry(
                self.session.amend_order,
                **params
            )
            
            if response.success:
                modified_id = order_id or order_link_id
                self.logger.info(f"Order {modified_id} für {symbol} erfolgreich geändert")
            else:
                self.logger.warning(f"Fehler bei der Order-Änderung für {symbol}: {response.error_message}")
                
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Ändern einer Order für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"modify_order for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def cancel_order(self,
                   symbol: str,
                   order_id: str = None,
                   order_link_id: str = None) -> ApiResponse[Dict]:
        """
        Storniert eine bestehende Order
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            order_id: Bybit Order-ID
            order_link_id: Benutzerdefinierte Order-ID
            
        Returns:
            ApiResponse mit Ergebnis der Stornierung
        """
        try:
            if not order_id and not order_link_id:
                return ApiResponse.error_response("Entweder order_id oder order_link_id muss angegeben werden.")
                
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
                
            if order_link_id:
                params["orderLinkId"] = order_link_id
                
            response = self._api_call_with_retry(
                self.session.cancel_order,
                **params
            )
            
            if response.success:
                cancelled_id = order_id or order_link_id
                self.logger.info(f"Order {cancelled_id} für {symbol} erfolgreich storniert")
            else:
                self.logger.warning(f"Fehler bei der Order-Stornierung für {symbol}: {response.error_message}")
                
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Stornieren einer Order für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"cancel_order for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def cancel_all_orders(self, symbol: str = None) -> ApiResponse[Dict]:
        """
        Storniert alle offenen Orders für ein Symbol oder alle Symbole
        
        Args:
            symbol: Optional - Trading-Paar-Symbol (z.B. 'BTCUSDT'). None für alle Symbole.
            
        Returns:
            ApiResponse mit Ergebnis der Stornierung
        """
        try:
            params = {
                "category": "linear"
            }
            
            if symbol:
                params["symbol"] = symbol
                
            response = self._api_call_with_retry(
                self.session.cancel_all_orders,
                **params
            )
            
            if response.success:
                self.logger.info(f"Alle Orders für {symbol or 'alle Symbole'} erfolgreich storniert")
            else:
                self.logger.warning(f"Fehler bei der Stornierung aller Orders für {symbol or 'alle Symbole'}: {response.error_message}")
                
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Stornieren aller Orders für {symbol or 'alle Symbole'}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"cancel_all_orders for {symbol or 'all'}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def set_trading_stop(self,
                        symbol: str,
                        stop_loss: float = None,
                        take_profit: float = None,
                        tp_trigger_by: str = "LastPrice",
                        sl_trigger_by: str = "LastPrice",
                        tp_size: float = None,
                        sl_size: float = None,
                        position_idx: int = 0,
                        trailing_stop: float = None) -> ApiResponse[Dict]:
        """
        Setzt oder ändert Stop-Loss/Take-Profit für eine offene Position
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            stop_loss: Stop-Loss-Preis
            take_profit: Take-Profit-Preis
            tp_trigger_by: Take-Profit-Auslösetyp ('LastPrice', 'IndexPrice', 'MarkPrice')
            sl_trigger_by: Stop-Loss-Auslösetyp ('LastPrice', 'IndexPrice', 'MarkPrice')
            tp_size: Teilprofitmitnahme-Größe (in Prozent oder Einheiten)
            sl_size: Teilstoppverlust-Größe (in Prozent oder Einheiten)
            position_idx: Positionsindex (0: One-Way-Mode, 1: Buy, 2: Sell für Hedge-Mode)
            trailing_stop: Trailing-Stop-Abstand in Prozent
            
        Returns:
            ApiResponse mit Ergebnis der Änderung
        """
        try:
            # Prüfe, ob mindestens ein Parameter angegeben wurde
            if not any([stop_loss, take_profit, tp_size, sl_size, trailing_stop]):
                return ApiResponse.error_response("Mindestens ein Parameter (stop_loss, take_profit, tp_size, sl_size, trailing_stop) muss angegeben werden.")
                
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": position_idx
            }
            
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
                params["slTriggerBy"] = sl_trigger_by
                
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
                params["tpTriggerBy"] = tp_trigger_by
                
            if tp_size is not None:
                params["tpSize"] = str(tp_size)
                
            if sl_size is not None:
                params["slSize"] = str(sl_size)
                
            if trailing_stop is not None:
                params["trailingStop"] = str(trailing_stop)
                
            response = self._api_call_with_retry(
                self.session.set_trading_stop,
                **params
            )
            
            if response.success:
                sl_msg = f"SL@{stop_loss}" if stop_loss is not None else "unverändert"
                tp_msg = f"TP@{take_profit}" if take_profit is not None else "unverändert"
                ts_msg = f"TS@{trailing_stop}%" if trailing_stop is not None else "unverändert"
                
                self.logger.info(f"Trading-Stop für {symbol} erfolgreich gesetzt: {sl_msg}, {tp_msg}, {ts_msg}")
            else:
                self.logger.warning(f"Fehler beim Setzen des Trading-Stops für {symbol}: {response.error_message}")
                
            return response
            
        except Exception as e:
            error_msg = f"Fehler beim Setzen des Trading-Stops für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"set_trading_stop for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def close_position(self, symbol: str, position_idx: int = 0) -> ApiResponse[Dict]:
        """
        Schließt eine offene Position mit einer Market-Order
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            position_idx: Positionsindex (0: One-Way-Mode, 1: Buy, 2: Sell für Hedge-Mode)
            
        Returns:
            ApiResponse mit Ergebnis der Schließung
        """
        try:
            # Aktuelle Position abrufen
            position_response = self.get_positions(symbol)
            
            if not position_response.success:
                return position_response
                
            positions = position_response.data.get("list", [])
            position = None
            
            # Position mit dem richtigen Index finden
            for pos in positions:
                if pos.get("symbol") == symbol and int(pos.get("positionIdx", 0)) == position_idx:
                    position = pos
                    break
                    
            if not position or float(position.get("size", 0)) == 0:
                return ApiResponse.error_response(f"Keine offene Position für {symbol} mit Index {position_idx} gefunden.")
                
            # Position schließen
            side = "Buy" if position.get("side") == "Sell" else "Sell"  # Gegengesetzte Seite für Schließung
            qty = abs(float(position.get("size", 0)))
            
            return self.place_market_order(
                symbol=symbol,
                side=side,
                qty=qty,
                reduce_only=True,
                position_idx=position_idx
            )
            
        except Exception as e:
            error_msg = f"Fehler beim Schließen der Position für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"close_position for {symbol}", traceback.format_exc())
            return ApiResponse.error_response(error_message=error_msg)
    
    def _make_request(self, method, endpoint, params=None):
        """
        Make a direct REST API request to Bybit with proper error handling and retries
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Request parameters
            
        Returns:
            API response data
        """
        if params is None:
            params = {}
            
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Making API request: {method} {url} with params: {params}")
        
        # Add any required common parameters
        if endpoint.startswith("/v5/market"):
            # Market endpoints don't need authentication
            pass
        else:
            # Potentially authenticated endpoints
            # Add common parameters like timestamp if needed
            pass
            
        # Apply rate limiting
        wait_time = self.rate_limiter.wait_for_token()
        if wait_time > 0:
            self.logger.debug(f"Rate limiting active, waiting {wait_time:.2f}s before API call to {endpoint}")
            time.sleep(wait_time)
            
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Measure API response time
                start_time = time.time()
                
                # Prepare headers
                headers = {}
                
                # Add authentication if needed and credentials are available
                if not endpoint.startswith("/v5/market") and self.api_key and self.api_secret:
                    # Time-based authentication (actual implementation depends on Bybit's requirements)
                    timestamp = str(int(time.time() * 1000) + self.time_offset)
                    
                    # Create signature
                    signature_payload = ""
                    if method == "GET":
                        # For GET requests, we need to include the query parameters in the signature
                        query_string = "&".join([f"{key}={params[key]}" for key in sorted(params.keys())])
                        signature_payload = timestamp + self.api_key + query_string
                    else:
                        # For POST requests, we include the request body in the signature
                        import json
                        signature_payload = timestamp + self.api_key + json.dumps(params)
                        
                    # Generate signature
                    signature = hmac.new(
                        self.api_secret.encode('utf-8'),
                        signature_payload.encode('utf-8'),
                        hashlib.sha256
                    ).hexdigest()
                    
                    # Add authentication headers
                    headers["X-BAPI-API-KEY"] = self.api_key
                    headers["X-BAPI-TIMESTAMP"] = timestamp
                    headers["X-BAPI-SIGN"] = signature
                
                # Make the request
                if method == "GET":
                    response = requests.get(url, params=params, headers=headers)
                elif method == "POST":
                    response = requests.post(url, json=params, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000  # in milliseconds
                
                # Check response
                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.logger.debug(f"API request successful: {method} {endpoint}")
                        return data
                    except ValueError:
                        self.logger.error(f"Invalid JSON response from {endpoint}: {response.text}")
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Retrying API request (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        return None
                elif response.status_code == 429:
                    # Rate limit hit
                    self.logger.warning(f"Rate limit exceeded for {endpoint}, backing off")
                    # Get retry-after header if available
                    retry_after = response.headers.get("Retry-After", retry_delay * (2 ** attempt))
                    retry_after = float(retry_after) if isinstance(retry_after, (int, float, str)) else retry_delay * (2 ** attempt)
                    time.sleep(retry_after)
                    continue
                else:
                    self.logger.error(f"API request failed: {method} {endpoint}, status code: {response.status_code}, response: {response.text}")
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retrying API request (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error on API request to {endpoint}: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retrying API request (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error on API request to {endpoint}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retrying API request (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                return None
                
        # If we get here, all retries failed
        self.logger.error(f"All retries failed for API request to {endpoint}")
        return None
    
    # WebSocket-Methoden
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable) -> bool:
        """
        Abonniert Kerzendaten über WebSocket
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            interval: Kerzen-Intervall ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'M', 'W')
            callback: Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_public:
                self.connect_ws()
                
            if not self.ws_public:
                self.logger.error("WebSocket-Verbindung nicht verfügbar")
                return False
                
            topic = f"kline.{interval}.{symbol}"
            
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    self._ws_data["kline"][f"{symbol}_{interval}"] = data
                    
                    # Benutzerdefinierte Callback aufrufen
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Kline-WebSocket-Callback: {e}")
            
            self.ws_public.kline_stream(
                interval=interval,
                symbol=symbol,
                callback=ws_callback
            )
            
            # Abonnement speichern
            if "kline" not in self._active_subscriptions["public"]:
                self._active_subscriptions["public"]["kline"] = []
                
            self._active_subscriptions["public"]["kline"].append({
                "symbol": symbol,
                "interval": interval,
                "callback": callback
            })
            
            self.logger.info(f"Kerzendaten-WebSocket für {symbol} ({interval}) abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Kerzendaten für {symbol} ({interval}): {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"subscribe_kline for {symbol}", traceback.format_exc())
            return False
    
    def subscribe_orderbook(self, symbol: str, depth: int = 50, callback: Callable = None) -> bool:
        """
        Abonniert Orderbuch-Daten über WebSocket
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            depth: Tiefe des Orderbuchs (1, 50, 200, 500)
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_public:
                self.connect_ws()
                
            if not self.ws_public:
                self.logger.error("WebSocket-Verbindung nicht verfügbar")
                return False
                
            # Standardtiefe auf 50 setzen, wenn ungültig
            if depth not in [1, 50, 200, 500]:
                depth = 50
                
            topic = f"orderbook.{depth}.{symbol}"
            
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    self._ws_data["orderbook"][symbol] = data
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Orderbook-WebSocket-Callback: {e}")
            
            self.ws_public.orderbook_stream(
                depth=depth,
                symbol=symbol,
                callback=ws_callback
            )
            
            # Abonnement speichern
            if "orderbook" not in self._active_subscriptions["public"]:
                self._active_subscriptions["public"]["orderbook"] = []
                
            self._active_subscriptions["public"]["orderbook"].append({
                "symbol": symbol,
                "depth": depth,
                "callback": callback
            })
            
            self.logger.info(f"Orderbuch-WebSocket für {symbol} (Tiefe: {depth}) abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren des Orderbuchs für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"subscribe_orderbook for {symbol}", traceback.format_exc())
            return False
    
    def subscribe_ticker(self, symbol: str, callback: Callable = None) -> bool:
        """
        Abonniert Ticker-Daten über WebSocket
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_public:
                self.connect_ws()
                
            if not self.ws_public:
                self.logger.error("WebSocket-Verbindung nicht verfügbar")
                return False
                
            topic = f"tickers.{symbol}"
            
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    self._ws_data["ticker"][symbol] = data
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Ticker-WebSocket-Callback: {e}")
            
            self.ws_public.ticker_stream(
                symbol=symbol,
                callback=ws_callback
            )
            
            # Abonnement speichern
            if "ticker" not in self._active_subscriptions["public"]:
                self._active_subscriptions["public"]["ticker"] = []
                
            self._active_subscriptions["public"]["ticker"].append({
                "symbol": symbol,
                "callback": callback
            })
            
            self.logger.info(f"Ticker-WebSocket für {symbol} abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren des Tickers für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"subscribe_ticker for {symbol}", traceback.format_exc())
            return False
    
    def subscribe_trade(self, symbol: str, callback: Callable = None) -> bool:
        """
        Abonniert Handelsausführungs-Daten über WebSocket
        
        Args:
            symbol: Trading-Paar-Symbol (z.B. 'BTCUSDT')
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_public:
                self.connect_ws()
                
            if not self.ws_public:
                self.logger.error("WebSocket-Verbindung nicht verfügbar")
                return False
                
            topic = f"publicTrade.{symbol}"
            
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", [])
                    if symbol not in self._ws_data["trade"]:
                        self._ws_data["trade"][symbol] = []
                    
                    # Neue Trades hinzufügen und Liste auf 100 Einträge begrenzen
                    self._ws_data["trade"][symbol] = data + self._ws_data["trade"][symbol]
                    self._ws_data["trade"][symbol] = self._ws_data["trade"][symbol][:100]
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Trade-WebSocket-Callback: {e}")
            
            self.ws_public.trade_stream(
                symbol=symbol,
                callback=ws_callback
            )
            
            # Abonnement speichern
            if "trade" not in self._active_subscriptions["public"]:
                self._active_subscriptions["public"]["trade"] = []
                
            self._active_subscriptions["public"]["trade"].append({
                "symbol": symbol,
                "callback": callback
            })
            
            self.logger.info(f"Trade-WebSocket für {symbol} abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Trades für {symbol}: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, f"subscribe_trade for {symbol}", traceback.format_exc())
            return False
    
    def subscribe_position(self, callback: Callable = None) -> bool:
        """
        Abonniert Positions-Updates über WebSocket
        
        Args:
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_private:
                self.connect_ws()
                
            if not self.ws_private:
                self.logger.error("Private WebSocket-Verbindung nicht verfügbar")
                return False
                
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    
                    # Positions nach Symbol organisieren
                    if "symbol" in data:
                        symbol = data["symbol"]
                        self._ws_data["position"][symbol] = data
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Position-WebSocket-Callback: {e}")
            
            self.ws_private.position_stream(callback=ws_callback)
            
            # Abonnement speichern
            if "position" not in self._active_subscriptions["private"]:
                self._active_subscriptions["private"]["position"] = []
                
            self._active_subscriptions["private"]["position"].append({
                "callback": callback
            })
            
            self.logger.info("Positions-WebSocket abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Positions-Updates: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "subscribe_position", traceback.format_exc())
            return False
    
    def subscribe_order(self, callback: Callable = None) -> bool:
        """
        Abonniert Order-Updates über WebSocket
        
        Args:
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_private:
                self.connect_ws()
                
            if not self.ws_private:
                self.logger.error("Private WebSocket-Verbindung nicht verfügbar")
                return False
                
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    
                    # Orders nach Symbol und ID organisieren
                    if "symbol" in data and "orderId" in data:
                        symbol = data["symbol"]
                        order_id = data["orderId"]
                        
                        if symbol not in self._ws_data["order"]:
                            self._ws_data["order"][symbol] = {}
                            
                        self._ws_data["order"][symbol][order_id] = data
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Order-WebSocket-Callback: {e}")
            
            self.ws_private.order_stream(callback=ws_callback)
            
            # Abonnement speichern
            if "order" not in self._active_subscriptions["private"]:
                self._active_subscriptions["private"]["order"] = []
                
            self._active_subscriptions["private"]["order"].append({
                "callback": callback
            })
            
            self.logger.info("Order-WebSocket abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Order-Updates: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "subscribe_order", traceback.format_exc())
            return False
    
    def subscribe_execution(self, callback: Callable = None) -> bool:
        """
        Abonniert Ausführungs-Updates über WebSocket
        
        Args:
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_private:
                self.connect_ws()
                
            if not self.ws_private:
                self.logger.error("Private WebSocket-Verbindung nicht verfügbar")
                return False
                
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    
                    # Ausführungen nach Symbol organisieren
                    if "symbol" in data:
                        symbol = data["symbol"]
                        
                        if symbol not in self._ws_data["execution"]:
                            self._ws_data["execution"][symbol] = []
                            
                        self._ws_data["execution"][symbol].append(data)
                        # Limit auf die letzten 100 Ausführungen
                        self._ws_data["execution"][symbol] = self._ws_data["execution"][symbol][-100:]
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Execution-WebSocket-Callback: {e}")
            
            self.ws_private.execution_stream(callback=ws_callback)
            
            # Abonnement speichern
            if "execution" not in self._active_subscriptions["private"]:
                self._active_subscriptions["private"]["execution"] = []
                
            self._active_subscriptions["private"]["execution"].append({
                "callback": callback
            })
            
            self.logger.info("Ausführungs-WebSocket abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Ausführungs-Updates: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "subscribe_execution", traceback.format_exc())
            return False
    
    def subscribe_wallet(self, callback: Callable = None) -> bool:
        """
        Abonniert Wallet-Updates über WebSocket
        
        Args:
            callback: Optionale Callback-Funktion für neue Daten
            
        Returns:
            True bei erfolgreicher Anmeldung, sonst False
        """
        try:
            if not self.ws_private:
                self.connect_ws()
                
            if not self.ws_private:
                self.logger.error("Private WebSocket-Verbindung nicht verfügbar")
                return False
                
            def ws_callback(message):
                try:
                    # Daten im Cache aktualisieren
                    data = message.get("data", {})
                    
                    # Benutzerdefinierte Callback aufrufen, falls vorhanden
                    if callback:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Wallet-WebSocket-Callback: {e}")
            
            self.ws_private.wallet_stream(callback=ws_callback)
            
            # Abonnement speichern
            if "wallet" not in self._active_subscriptions["private"]:
                self._active_subscriptions["private"]["wallet"] = []
                
            self._active_subscriptions["private"]["wallet"].append({
                "callback": callback
            })
            
            self.logger.info("Wallet-WebSocket abonniert")
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Abonnieren von Wallet-Updates: {str(e)}"
            self.logger.error(error_msg)
            log_exception(self.logger, e, "subscribe_wallet", traceback.format_exc())
            return False
    
    def get_cached_data(self, data_type: str, symbol: str = None, additional_key: str = None) -> Optional[Any]:
        """
        Ruft gecachte WebSocket-Daten ab
        
        Args:
            data_type: Datentyp ('kline', 'orderbook', 'ticker', 'trade', 'position', 'order', 'execution')
            symbol: Optional - Symbol für die Datenabfrage
            additional_key: Optional - Zusätzlicher Schlüssel (z.B. Intervall für Kerzendaten)
            
        Returns:
            Gecachte Daten oder None, wenn nicht verfügbar
        """
        try:
            if data_type not in self._ws_data:
                return None
                
            if data_type == "kline":
                if not symbol or not additional_key:
                    return None
                key = f"{symbol}_{additional_key}"
                return self._ws_data["kline"].get(key)
            elif data_type in ["orderbook", "ticker"]:
                if not symbol:
                    return None
                return self._ws_data[data_type].get(symbol)
            elif data_type == "trade":
                if not symbol:
                    return None
                return self._ws_data["trade"].get(symbol, [])
            elif data_type == "position":
                if symbol:
                    return self._ws_data["position"].get(symbol)
                return self._ws_data["position"]
            elif data_type == "order":
                if symbol:
                    if additional_key:  # order_id
                        return self._ws_data["order"].get(symbol, {}).get(additional_key)
                    return self._ws_data["order"].get(symbol, {})
                return self._ws_data["order"]
            elif data_type == "execution":
                if symbol:
                    return self._ws_data["execution"].get(symbol, [])
                return self._ws_data["execution"]
            else:
                return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen gecachter Daten: {e}")
            return None

    def _ws_reconnect_handler(self):
        """Main handler for WebSocket reconnection"""
        try:
            # Add a counter for reconnection attempts if it doesn't exist
            if not hasattr(self, 'ws_reconnect_count'):
                self.ws_reconnect_count = 0
                
            # Log the reconnection attempt
            self.logger.info(f"WebSocket reconnection attempt #{self.ws_reconnect_count + 1}")
            
            # Attempt to reconnect
            reconnection_success = self._reconnect_private_ws()
            
            if reconnection_success:
                self.logger.info("WebSocket reconnection successful.")
                # Reset the reconnection counter
                self.ws_reconnect_count = 0
                return True
            else:
                # Increment reconnection counter
                self.ws_reconnect_count += 1
                
                # Log with increasing severity based on number of failures
                if self.ws_reconnect_count <= 3:
                    self.logger.warning(f"WebSocket reconnection failed (attempt {self.ws_reconnect_count}). Will retry.")
                elif self.ws_reconnect_count <= 10:
                    self.logger.error(f"WebSocket reconnection failed after {self.ws_reconnect_count} attempts. Continuing to retry.")
                else:
                    self.logger.critical(f"WebSocket reconnection failed after {self.ws_reconnect_count} attempts. Consider restarting the bot.")
                
                return False
        except Exception as e:
            self.logger.error(f"Error in WebSocket reconnection handler: {e}")
            # Increment reconnection counter
            if hasattr(self, 'ws_reconnect_count'):
                self.ws_reconnect_count += 1
            else:
                self.ws_reconnect_count = 1
                
            return False

    def get_server_time(self):
        """
        Get the current server time from Bybit API
        
        Returns:
            int: Server time in milliseconds if successful, None otherwise
        """
        try:
            # Make a direct REST API call to get server time
            response = self._make_request("GET", "/v5/market/time", {})
            
            # Check if the response is valid
            if not response:
                self.logger.error("Failed to get server time: No response")
                return None
                
            if "retCode" not in response:
                self.logger.error(f"Invalid server time response format: {response}")
                return None
                
            if response["retCode"] != 0:
                error_msg = response.get("retMsg", "Unknown error")
                self.logger.error(f"Failed to get server time: {error_msg}")
                return None
                
            # Extract server time from response
            result = response.get("result", {})
            if not result:
                self.logger.error("No result data in server time response")
                return None
                
            # Extract time in seconds and convert to milliseconds
            time_second = result.get("timeSecond")
            if time_second is None:
                self.logger.error("No timeSecond in server time response")
                return None
                
            # Convert to integer milliseconds
            server_time = int(float(time_second) * 1000)
            
            # Update time offset for future API calls
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            
            if abs(self.time_offset) > 1000:  # More than 1 second difference
                self.logger.warning(f"Large time difference with server: {self.time_offset} ms")
                
            self.logger.debug(f"Server time: {server_time}, Offset: {self.time_offset} ms")
            return server_time
            
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return None
    
    def get_latest_price(self, symbol):
        """
        Get the latest price for a symbol
        
        Args:
            symbol (str): The trading pair symbol (e.g. 'BTCUSDT')
            
        Returns:
            float: Latest price if successful, None otherwise
        """
        try:
            # Check cache first (if available)
            cached_data = self.get_cached_data("ticker", symbol)
            if cached_data is not None:
                self.logger.debug(f"Using cached ticker data for {symbol}")
                return float(cached_data.get("last_price", 0))
                
            # Make direct API request to get ticker data
            params = {"symbol": symbol, "category": "linear"}
            response = self._make_request("GET", "/v5/market/tickers", params)
            
            # Validate response
            if not response or "retCode" not in response:
                self.logger.error(f"Invalid response format for {symbol} ticker")
                return None
                
            if response["retCode"] != 0:
                error_msg = response.get("retMsg", "Unknown error")
                self.logger.error(f"Failed to get ticker for {symbol}: {error_msg}")
                return None
                
            # Extract price from response
            result = response.get("result", {})
            if not result:
                self.logger.error(f"No result data in ticker response for {symbol}")
                return None
                
            # Extract ticker list
            ticker_list = result.get("list", [])
            if not ticker_list:
                self.logger.error(f"No ticker data found for {symbol}")
                return None
                
            # Get first item (should be only one)
            ticker = ticker_list[0]
            
            # Try to extract last price
            last_price = ticker.get("lastPrice")
            if last_price is None:
                self.logger.error(f"No last price found in ticker data for {symbol}")
                return None
                
            # Convert to float and return
            try:
                return float(last_price)
            except (ValueError, TypeError):
                self.logger.error(f"Invalid price format for {symbol}: {last_price}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def close(self):
        """
        Properly close all connections and shut down threads
        Call this method when shutting down the bot to prevent background thread errors
        """
        try:
            self.logger.info("Closing Bybit API connections and stopping threads...")
            self.is_running = False
            
            # Stop WebSocket monitoring first
            if hasattr(self, '_monitoring_thread') and self._monitoring_thread and self._monitoring_thread.is_alive():
                self.logger.debug("Stopping WebSocket monitoring thread")
                # No direct way to stop, but it should respect the is_running flag
            
            # Close WebSocket connections with proper error handling
            if hasattr(self, 'ws_private') and self.ws_private:
                try:
                    self.logger.info("Closing private WebSocket connection...")
                    try:
                        # Mark as exited before actually exiting to prevent ping errors
                        if hasattr(self.ws_private, '_client'):
                            self.ws_private._client.exited = True
                        
                        self.ws_private.exit()
                    except Exception as e:
                        if "Connection is already closed" in str(e):
                            self.logger.info("WebSocket connection was already closed")
                        else:
                            self.logger.warning(f"Error while closing WebSocket: {e}")
                    finally:
                        self.ws_private = None
                except Exception as e:
                    self.logger.error(f"Error closing private WebSocket: {e}")
            
            if hasattr(self, 'ws_public') and self.ws_public:
                try:
                    self.logger.info("Closing public WebSocket connection...")
                    try:
                        # Mark as exited before actually exiting to prevent ping errors
                        if hasattr(self.ws_public, '_client'):
                            self.ws_public._client.exited = True
                            
                        self.ws_public.exit()
                    except Exception as e:
                        if "Connection is already closed" in str(e):
                            self.logger.info("Public WebSocket connection was already closed")
                        else:
                            self.logger.warning(f"Error while closing public WebSocket: {e}")
                    finally:
                        self.ws_public = None
                except Exception as e:
                    self.logger.error(f"Error closing public WebSocket: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("All Bybit API connections closed successfully")
        except Exception as e:
            self.logger.error(f"Error during API shutdown: {e}")

    # Add a new method to handle WebSocket errors
    def _on_websocket_error(self, error):
        """Handle WebSocket errors gracefully"""
        self.logger.error(f"WebSocket error encountered: {error}")
        
        # Check if it's a connection closed exception
        if "WebSocketConnectionClosedException" in str(error) or "Connection is already closed" in str(error):
            self.logger.warning("WebSocket connection was closed. This is expected during reconnection attempts.")
            # Mark the connection as disconnected to trigger the reconnection logic
            if hasattr(self, 'ws_private') and self.ws_private:
                try:
                    # Try to close gracefully if not already closed
                    self.ws_private.exit()
                except:
                    pass
                self.ws_private = None

    # WebSocket callback methods
    def _on_position_update(self, message):
        """Handle position updates from WebSocket"""
        try:
            self.logger.debug(f"Position update received: {message}")
            # Process position data if needed
        except Exception as e:
            self.logger.error(f"Error processing position update: {e}")

    def _on_execution_update(self, message):
        """Handle execution updates from WebSocket"""
        try:
            self.logger.debug(f"Execution update received: {message}")
            # Process execution data if needed
        except Exception as e:
            self.logger.error(f"Error processing execution update: {e}")

    def _on_order_update(self, message):
        """Handle order updates from WebSocket"""
        try:
            self.logger.debug(f"Order update received: {message}")
            # Process order data if needed
        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")

    def _on_wallet_update(self, message):
        """Handle wallet updates from WebSocket"""
        try:
            self.logger.debug(f"Wallet update received: {message}")
            # Process wallet data if needed
        except Exception as e:
            self.logger.error(f"Error processing wallet update: {e}")

    def _setup_ws_monitoring(self):
        """Set up WebSocket connection monitoring to handle disconnects"""
        import threading
        
        def monitor_connection():
            reconnect_attempts = 0
            max_consecutive_attempts = 5
            backoff_time = 30  # Base backoff time in seconds
            
            while True:
                try:
                    # First check if we should be monitoring at all (bot might be shutting down)
                    if not hasattr(self, 'is_running') or not getattr(self, 'is_running', True):
                        self.logger.info("WebSocket monitoring thread stopping as bot is shutting down")
                        break
                    
                    # Check private WebSocket
                    if self.ws_private:
                        # Check if connected - use safer check method
                        try:
                            is_connected = False
                            if hasattr(self.ws_private, 'ws') and self.ws_private.ws:
                                is_connected = self.ws_private.ws.sock and self.ws_private.ws.sock.connected
                        except Exception as e:
                            self.logger.warning(f"Error checking WebSocket connection status: {e}")
                            is_connected = False  # Assume not connected on error
                        
                        if not is_connected:
                            self.logger.warning("Private WebSocket disconnected, attempting to reconnect...")
                            success = self.connect_ws()  # Use main connect method instead of reconnect
                            if success:
                                reconnect_attempts = 0
                                self.logger.info("Successfully reconnected WebSocket")
                            else:
                                reconnect_attempts += 1
                                # Implement exponential backoff
                                current_backoff = min(300, backoff_time * (2 ** (reconnect_attempts - 1)))  # Cap at 5 minutes
                                self.logger.warning(f"Reconnection attempt {reconnect_attempts} failed. Backing off for {current_backoff} seconds before retry.")
                                time.sleep(current_backoff)
                                continue  # Skip to next iteration to try reconnect immediately
                
                    # If too many consecutive failures, log critical error
                    if reconnect_attempts >= max_consecutive_attempts:
                        self.logger.error(f"Failed to reconnect WebSocket after {reconnect_attempts} consecutive attempts. Will continue trying with longer intervals.")
                        # Don't exit the loop, keep trying but log less frequently
                    
                    # Sleep for interval - use standard reconnect interval if reconnections are working
                    time.sleep(self.ws_reconnect_interval)
                except Exception as e:
                    self.logger.error(f"Error in WebSocket monitoring: {e}")
                    time.sleep(self.ws_reconnect_interval)
        
        # Start monitoring in a background thread
        self.is_running = True  # Add flag to control thread lifecycle
        self._monitoring_thread = threading.Thread(target=monitor_connection, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("WebSocket connection monitoring started")

    def test_connectivity(self) -> bool:
        """
        Test API connectivity and credentials
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.logger.info("Testing Bybit API connectivity...")
            
            # Test public endpoint first (server time)
            server_time_response = self._make_request("GET", "/v5/market/time", {})
            
            if not server_time_response or "retCode" not in server_time_response:
                self.logger.error("Failed to connect to Bybit API (public endpoints)")
                return False
                
            if server_time_response.get("retCode") != 0:
                error_msg = server_time_response.get("retMsg", "Unknown error")
                self.logger.error(f"Failed to connect to Bybit API: {error_msg}")
                return False
                
            self.logger.info("Public API endpoint test successful")
            
            # Test authenticated endpoint if credentials are provided
            if self.api_key and self.api_secret:
                self.logger.info("Testing authenticated API endpoint...")
                
                # Test account information endpoint
                account_response = self._make_request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
                
                if not account_response:
                    self.logger.error("Failed to connect to Bybit API (authenticated endpoints)")
                    return False
                    
                if account_response.get("retCode") != 0:
                    error_msg = account_response.get("retMsg", "Unknown error")
                    self.logger.error(f"API authentication failed: {error_msg}")
                    
                    # Check for common authentication errors
                    if "Invalid API key" in error_msg or "api_key not found" in error_msg:
                        self.logger.error("API key is invalid or not found")
                    elif "Invalid sign" in error_msg or "signature" in error_msg.lower():
                        self.logger.error("API signature failed - check API secret and time synchronization")
                    elif "timestamp" in error_msg.lower():
                        self.logger.error("Time synchronization issue detected")
                        # Try to re-sync time
                        self._sync_time_with_server()
                        
                    return False
                    
                self.logger.info("Authenticated API endpoint test successful")
            else:
                self.logger.warning("No API credentials provided, skipping authenticated endpoint test")
                
            return True
        except Exception as e:
            self.logger.error(f"Error testing API connectivity: {e}")
            return False

