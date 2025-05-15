#!/usr/bin/env python
"""
Live Trading Runner for the EMA Intraday Crossover Strategy.

This script runs the strategy in live trading mode, placing actual orders in the market.
Use with caution as it will risk real money.

Usage:
    python run_ema_strategy_live.py --confirm
"""
import os
import sys
import time
import logging
import argparse
import traceback
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from threading import Thread, Lock

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager
from strategy.ema_crossover_strategy import EMAIntraDayStrategy
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager

# Import configuration (with fallback if not found)
try:
    from config.ema_strategy_config import (
        SYMBOLS, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, 
        TIMEFRAME, MAX_POSITION_SIZE, STOP_LOSS_PERCENT,
        LOG_LEVEL, LOG_DIR, LOG_TO_FILE, LOG_TO_CONSOLE,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME, SQUARE_OFF_TIME,
        ORDER_UPDATE_INTERVAL, USE_LIMIT_ORDERS
    )
    config_loaded = True
except ImportError:
    config_loaded = False
    LOG_LEVEL = "INFO"
    LOG_DIR = "data/logs"
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    STOP_LOSS_PERCENT = 0.01
    SYMBOLS = "RELIANCE,HDFCBANK"
    SHORT_EMA_PERIOD = 2
    LONG_EMA_PERIOD = 5
    TIMEFRAME = 5
    MAX_POSITION_SIZE = 1
    MARKET_OPEN_TIME = "09:15:00"
    MARKET_CLOSE_TIME = "15:30:00"
    SQUARE_OFF_TIME = "15:10:00"
    ORDER_UPDATE_INTERVAL = 1
    USE_LIMIT_ORDERS = True


class EMAStrategyLiveTrader:
    """Main class for running the EMA Crossover Strategy in live trading mode."""
    
    def __init__(self, args):
        """
        Initialize the live trader.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.setup_logging()
        
        # Initialize component references
        self.auth = None
        self.kite = None
        self.strategy = None
        self.historical_data_manager = None
        self.realtime_data = None
        self.order_manager = None
        self.risk_manager = None
        
        # Trading parameters from config file
        self.symbols = [s.strip() for s in SYMBOLS.split(',')]
        self.short_ema_period = SHORT_EMA_PERIOD
        self.long_ema_period = LONG_EMA_PERIOD
        self.timeframe = TIMEFRAME
        self.max_position_size = MAX_POSITION_SIZE
        self.stop_loss_percent = STOP_LOSS_PERCENT
        
        # Parse market timing
        self.market_open_time = datetime.strptime(MARKET_OPEN_TIME, "%H:%M:%S").time()
        self.market_close_time = datetime.strptime(MARKET_CLOSE_TIME, "%H:%M:%S").time()
        self.square_off_time = datetime.strptime(SQUARE_OFF_TIME, "%H:%M:%S").time()
        
        # Token mapping
        self.token_map = {}
        self.symbol_token_map = {}
        self.token_symbol_map = {}
        
        # Tick size information for proper order pricing
        self.tick_sizes = {}
        
        # State tracking
        self.is_running = False
        self.last_position_log_time = datetime.min
        self.last_order_check_time = datetime.min
        self.last_tick_time = {}  # To track when we last received a tick for each symbol
        self.last_price = {}  # To track the last price for each symbol
        self.debug_log_interval = 300  # 5 minutes in seconds
        self.trade_allowed = {}  # Dictionary to track if trading is allowed for each symbol
        
        # Candle formation
        self.current_candle = {}  # To track the current candle for each symbol
        self.candle_lock = Lock()  # Lock for thread-safe candle updates
        self.candle_timer_thread = None
        self.last_candle_time = {}  # To track when we last formed a candle
        self.last_trade_candle_time = {}  # To track when we last traded based on a candle
        
        # WebSocket state
        self.connection_check_interval = 60  # Check WebSocket every 60 seconds
        self.last_connection_check = datetime.min
        self.ws_reconnect_attempts = 0
        self.max_ws_reconnect_attempts = 5
        
        # Order tracking
        self.pending_orders = {}  # Symbol -> order_id mapping
        self.order_status = {}    # order_id -> status mapping
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Setup log file
        log_file = os.path.join(LOG_DIR, f"ema_live_trading_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Configure logging
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        
        # Create handlers
        handlers = []
        if LOG_TO_FILE:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
            
        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            handlers.append(console_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def authenticate(self):
        """Authenticate with Zerodha's API."""
        # Load environment variables
        load_dotenv()
        
        # Get API key, secret, and access token from environment variables
        api_key = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        
        if not api_key or not api_secret or not access_token:
            self.logger.error("API key, secret, and access token are required")
            self.logger.error("Please run the authentication script first")
            return False
        
        # Initialize ZerodhaAuth
        try:
            self.logger.info("Initializing Zerodha authentication...")
            self.auth = ZerodhaAuth(api_key, api_secret, access_token)
            
            # Validate connection
            if not self.auth.validate_connection():
                self.logger.error("Connection validation failed. Your access token may be expired.")
                self.logger.error("Please run the authentication script again to get a new access token.")
                return False
                
            self.logger.info("Authentication successful")
            
            # Get KiteConnect instance
            self.kite = self.auth.get_kite()
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def initialize_components(self):
        """Initialize all required components."""
        try:
            # Check if authentication was successful
            if not self.kite:
                self.logger.error("Authentication not completed. Cannot initialize components.")
                return False
            
            self.logger.info("Initializing trading components...")
            
            # Initialize data managers
            self.historical_data_manager = HistoricalDataManager(self.kite)
            self.realtime_data = RealTimeDataManager(
                os.getenv("KITE_API_KEY"), 
                os.getenv("KITE_ACCESS_TOKEN")
            )
            
            # Initialize order and risk managers
            self.order_manager = OrderManager(self.kite)
            self.risk_manager = RiskManager(
                max_risk_per_trade=0.01,  # 1% risk per trade
                stop_loss_pct=self.stop_loss_percent
            )
            
            # Format timeframe string
            timeframe_str = f"{self.timeframe}minute"
            
            # Initialize strategy
            self.strategy = EMAIntraDayStrategy(
                name="EMA_Intraday_Live",
                universe=self.symbols,
                timeframe=timeframe_str,
                short_ema_period=self.short_ema_period,
                long_ema_period=self.long_ema_period,
                max_position_size=self.max_position_size
            )
            
            # Initialize the candle and trading tracking for each symbol
            for symbol in self.symbols:
                self.current_candle[symbol] = None
                self.last_candle_time[symbol] = datetime.min
                self.last_trade_candle_time[symbol] = datetime.min
                self.last_tick_time[symbol] = datetime.min
                self.last_price[symbol] = None
                self.trade_allowed[symbol] = False
            
            self.logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def setup_universe(self):
        """Set up the trading universe and fetch initial data."""
        try:
            # Initialize positions for all symbols
            self.logger.info("Initializing positions for all symbols")
            for symbol in self.symbols:
                self.strategy.update_position(symbol, {"quantity": 0, "average_price": 0})
            
            # Get instrument tokens and tick sizes
            self.logger.info("Fetching instrument tokens and tick sizes for all symbols")
            
            # Fetch all instruments and find our symbols
            instruments = self.kite.instruments("NSE")
            
            # Create mappings for tokens and tick sizes
            for symbol in self.symbols:
                # Find the instrument data for this symbol
                instrument_data = next((inst for inst in instruments if inst["tradingsymbol"] == symbol), None)
                
                if instrument_data:
                    token = instrument_data["instrument_token"]
                    tick_size = instrument_data.get("tick_size", 0.05)  # Default to 0.05 if not found
                    
                    self.token_map[symbol] = token
                    self.tick_sizes[symbol] = tick_size
                    
                    self.logger.info(f"Found {symbol}: Token={token}, Tick Size={tick_size}")
                else:
                    self.logger.warning(f"Could not find instrument data for {symbol}")
            
            # Create reverse token mapping
            self.token_symbol_map = {v: k for k, v in self.token_map.items()}
            
            # Check if we got all tokens
            if len(self.token_map) != len(self.symbols):
                missing_symbols = set(self.symbols) - set(self.token_map.keys())
                if missing_symbols:
                    self.logger.warning(f"Missing tokens for: {missing_symbols}")
            
            tokens = list(self.token_map.values())
            if not tokens:
                self.logger.error("No valid tokens found for any symbols")
                return False
            
            # Fetch historical data
            historical_data = self.fetch_historical_data()
            
            # Initialize strategy with historical data
            for symbol, data in historical_data.items():
                self.strategy.update_historical_data(symbol, data)
                
                # Calculate initial indicators
                indicators = self.strategy.calculate_indicators(data)
                self.strategy.update_indicators(symbol, indicators)
                
                self.logger.info(f"Initialized {symbol} with {len(data)} candles")
                
                # Initialize last price
                if not data.empty:
                    self.last_price[symbol] = data['close'].iloc[-1]
                    self.logger.info(f"Initial price for {symbol}: {self.last_price[symbol]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up universe: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def fetch_historical_data(self):
        """
        Fetch historical data for initialization based on the long EMA period.
        
        Returns:
            Dictionary of dataframes with historical data
        """
        historical_data = {}
        
        # Calculate the number of candles needed based on long EMA period
        # Adding extra candles to ensure we have enough data
        min_candles_needed = self.long_ema_period * 3
        
        # Calculate how many days to fetch based on timeframe and candles needed
        # For 5-minute candles, there are 72 candles in a 6-hour trading day
        candles_per_day = 72  # 6 hours * 12 candles per hour for 5-minute timeframe
        days_to_fetch = max(3, (min_candles_needed // candles_per_day) + 2)  # Minimum 3 days, add buffer
        
        self.logger.info(f"Calculating fetch period: Need {min_candles_needed} candles, fetching {days_to_fetch} days of data")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_to_fetch)
        
        # Convert to string format required by API
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching historical data from {from_date_str} to {to_date_str}")
        
        for symbol in self.symbols:
            try:
                # Get instrument token
                token = self.token_map.get(symbol)
                if not token:
                    self.logger.warning(f"No instrument token found for {symbol}")
                    continue
                    
                # Fetch data - Use the strategy's timeframe
                interval = f"{self.timeframe}minute"
                
                self.logger.info(f"Fetching historical data for {symbol} with token {token} using interval {interval}")
                
                df = self.historical_data_manager.fetch_historical_data(
                    token, from_date_str, to_date_str, interval
                )
                
                if df is not None and not df.empty:
                    historical_data[symbol] = df
                    self.logger.info(f"Fetched {len(df)} {interval} candles for {symbol}")
                    
                    # Log the first 3 and last 3 candles for verification
                    self.logger.info(f"First 3 candles for {symbol}:")
                    for i in range(min(3, len(df))):
                        candle = df.iloc[i]
                        self.logger.info(f"  {df.index[i]}: O={candle['open']}, H={candle['high']}, L={candle['low']}, C={candle['close']}, Vol={candle.get('volume', 'N/A')}")
                    
                    self.logger.info(f"Last 3 candles for {symbol}:")
                    for i in range(max(0, len(df)-3), len(df)):
                        candle = df.iloc[i]
                        self.logger.info(f"  {df.index[i]}: O={candle['open']}, H={candle['high']}, L={candle['low']}, C={candle['close']}, Vol={candle.get('volume', 'N/A')}")
                    
                    # Verify we have enough data for strategy
                    if len(df) < min_candles_needed:
                        self.logger.warning(f"Warning: Got only {len(df)} candles for {symbol}, need at least {min_candles_needed} for reliable signals")
                else:
                    self.logger.warning(f"No data fetched for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                self.logger.error(traceback.format_exc())
        
        return historical_data
    
    def round_to_tick_size(self, symbol, price, is_buy=True):
        """
        Round price to the nearest valid tick size.
        For buy orders, round down to avoid rejection for price too high.
        For sell orders, round up to avoid rejection for price too low.
        
        Args:
            symbol: Trading symbol
            price: The price to round
            is_buy: Whether this is a buy order (True) or sell order (False)
            
        Returns:
            Price rounded to the nearest valid tick size
        """
        if symbol not in self.tick_sizes or not self.tick_sizes[symbol]:
            self.logger.warning(f"No tick size info for {symbol}, using default of 0.05")
            tick_size = 0.05
        else:
            tick_size = self.tick_sizes[symbol]
        
        # Calculate how many ticks
        ticks = price / tick_size
        
        if is_buy:
            # For buy orders, round down to nearest tick
            rounded_ticks = math.floor(ticks)
        else:
            # For sell orders, round up to nearest tick
            rounded_ticks = math.ceil(ticks)
        
        # Calculate rounded price
        rounded_price = rounded_ticks * tick_size
        
        # Ensure we have the right precision (same number of decimal places as tick size)
        tick_decimals = str(tick_size)[::-1].find('.')
        if tick_decimals > 0:
            rounded_price = round(rounded_price, tick_decimals)
        
        if price != rounded_price:
            self.logger.info(f"Rounded price for {symbol} from {price} to {rounded_price} (tick size: {tick_size}, {'buy' if is_buy else 'sell'})")
        
        return rounded_price
    
    def setup_callbacks(self):
        """Setup callbacks for the real-time data manager."""
        # Register callbacks
        self.realtime_data.register_callback('on_tick', self.on_tick)
        self.realtime_data.register_callback('on_order_update', self.on_order_update)
        self.realtime_data.register_callback('on_connect', self.on_connect)
        self.realtime_data.register_callback('on_close', self.on_close)
        self.realtime_data.register_callback('on_error', self.on_error)
        self.realtime_data.register_callback('on_reconnect', self.on_reconnect)
        self.realtime_data.register_callback('on_noreconnect', self.on_noreconnect)
        
        return True
    
    def on_connect(self, response):
        """
        Callback when WebSocket connection is established.
        
        Args:
            response: Connection response
        """
        self.logger.info(f"WebSocket connected: {response}")
        self.ws_reconnect_attempts = 0
        
        # Now that we're connected, subscribe to tokens
        tokens = list(self.token_map.values())
        if tokens:
            self.logger.info(f"Subscribing to {len(tokens)} tokens")
            self.realtime_data.subscribe(tokens, self.token_symbol_map, self.token_map)
        else:
            self.logger.error("No tokens to subscribe to")
    
    def on_close(self, code, reason):
        """
        Callback when WebSocket connection is closed.
        
        Args:
            code: Close code
            reason: Close reason
        """
        self.logger.warning(f"WebSocket connection closed: {code} - {reason}")
        # Connection will be auto-reconnected by the RealTimeDataManager
    
    def on_error(self, code, reason):
        """
        Callback when WebSocket connection encounters an error.
        
        Args:
            code: Error code
            reason: Error reason
        """
        self.logger.error(f"WebSocket error: {code} - {reason}")
        # Connection will be auto-reconnected by the RealTimeDataManager
    
    def on_reconnect(self, attempts_count):
        """
        Callback when WebSocket is attempting to reconnect.
        
        Args:
            attempts_count: Number of reconnection attempts
        """
        self.logger.warning(f"WebSocket reconnecting... Attempt {attempts_count}")
        self.ws_reconnect_attempts = attempts_count
    
    def on_noreconnect(self):
        """Callback when WebSocket has exhausted reconnection attempts."""
        self.logger.error("WebSocket reconnection failed after maximum attempts")
        # Try to manually restart the connection
        self.restart_websocket()
    
    def restart_websocket(self):
        """Try to manually restart the WebSocket connection."""
        try:
            self.logger.info("Manually restarting WebSocket connection...")
            
            # Stop the current connection
            if self.realtime_data:
                self.realtime_data.stop()
                time.sleep(2)  # Wait a bit before reconnecting
            
            # Create a new RealTimeDataManager instance
            self.realtime_data = RealTimeDataManager(
                os.getenv("KITE_API_KEY"), 
                os.getenv("KITE_ACCESS_TOKEN")
            )
            
            # Set up callbacks again
            self.setup_callbacks()
            
            # Start the connection
            if self.realtime_data.start():
                self.logger.info("WebSocket connection restarted successfully")
                return True
            else:
                self.logger.error("Failed to restart WebSocket connection")
                return False
        except Exception as e:
            self.logger.error(f"Error restarting WebSocket connection: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def check_websocket_connection(self):
        """Check WebSocket connection status and reconnect if needed."""
        # Only check every connection_check_interval seconds
        current_time = datetime.now()
        if (current_time - self.last_connection_check).total_seconds() < self.connection_check_interval:
            return True
            
        self.last_connection_check = current_time
        
        # Check if WebSocket is connected
        if not self.realtime_data.is_connected:
            self.logger.warning("WebSocket not connected, attempting to reconnect...")
            return self.restart_websocket()
        
        # Check if we received ticks recently
        stale_symbols = []
        for symbol in self.symbols:
            last_tick = self.last_tick_time.get(symbol, datetime.min)
            # If we haven't received a tick in 5 minutes for any symbol, consider data stale
            if (current_time - last_tick).total_seconds() > 300:  # 5 minutes
                stale_symbols.append(symbol)
        
        if stale_symbols:
            self.logger.warning(f"Tick data is stale for symbols: {stale_symbols}, restarting WebSocket connection...")
            return self.restart_websocket()
        
        return True
    
    def on_tick(self, symbol, tick):
        """
        Callback for tick data.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
        """
        # Ensure symbol is a string, not a token
        if isinstance(symbol, int):
            # If we get a token instead of a symbol, convert it
            symbol = self.token_symbol_map.get(symbol, str(symbol))
        
        # Update last tick time and price
        self.last_tick_time[symbol] = datetime.now()
        
        # Extract last price from tick
        last_price = tick.get('last_price')
        if last_price:
            # Check if price has changed
            prev_price = self.last_price.get(symbol)
            if prev_price is not None and prev_price != last_price:
                self.logger.debug(f"Price change for {symbol}: {prev_price} -> {last_price}")
            
            # Update last price
            self.last_price[symbol] = last_price
        
        try:
            # Check if we need to update the current candle
            self.update_current_candle(symbol, tick)
            
            # Check if we should form a new candle
            self.check_candle_formation(symbol, tick)
            
            # Process tick only if we're allowed to trade
            if self.should_process_tick(symbol):
                # Process tick with strategy
                signal = self.strategy.process_tick(symbol, tick)
                
                # If a signal is generated, process it and place an order
                if signal:
                    self.logger.info(f"Signal generated for {symbol}: {signal}")
                    
                    # Track that we've traded on this candle
                    current_candle_time = self.current_candle[symbol]['timestamp'] if self.current_candle.get(symbol) else None
                    if current_candle_time:
                        self.last_trade_candle_time[symbol] = current_candle_time
                    
                    # Prepare order parameters
                    order_params = self.strategy.prepare_order_parameters(signal, tick)
                    
                    if not order_params:
                        self.logger.warning(f"Could not prepare order parameters for signal: {signal}")
                        return
                    
                    # Apply risk management
                    if 'action' not in signal or signal['action'] not in ['UPDATE']:
                        # Only apply risk management to new orders, not updates
                        risk_adjusted_params = self.risk_manager.process_signal(signal, tick)
                        if risk_adjusted_params:
                            # Merge with original parameters, keeping the strategy's order_type and price
                            order_params.update({
                                k: v for k, v in risk_adjusted_params.items() 
                                if k not in ['order_type', 'price'] and k in order_params
                            })
                            self.logger.info(f"Risk-adjusted order parameters: {order_params}")
                        else:
                            self.logger.warning(f"Signal rejected by risk manager")
                            return
                    
                    # Execute the order based on action type
                    self.execute_order(symbol, signal, order_params)
        
        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
    
    def should_process_tick(self, symbol):
        """
        Determine if we should process the tick for trading.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if we should process the tick, False otherwise
        """
        # Check if it's a trading time
        current_time = datetime.now()
        current_minute = current_time.minute
        
        # Only allow trading at the specified interval
        # For 5-minute timeframe, this would be at minute 0, 5, 10, 15, etc.
        is_trading_minute = current_minute % self.timeframe == 0
        
        # Check if we've already traded on this candle
        current_candle_time = self.current_candle.get(symbol, {}).get('timestamp')
        last_trade_time = self.last_trade_candle_time.get(symbol, datetime.min)
        
        # Only allow one trade per candle
        already_traded_current_candle = (current_candle_time and last_trade_time and 
                                        current_candle_time == last_trade_time)
        
        # Update trade_allowed flag
        if is_trading_minute and not already_traded_current_candle:
            if not self.trade_allowed.get(symbol, False):
                self.logger.info(f"Trading now allowed for {symbol} at {current_time.strftime('%H:%M:%S')}")
                self.trade_allowed[symbol] = True
        else:
            # If trading was previously allowed but now it's not, log it
            if self.trade_allowed.get(symbol, False):
                if already_traded_current_candle:
                    self.logger.info(f"Trading no longer allowed for {symbol} - already traded this candle")
                else:
                    self.logger.info(f"Trading no longer allowed for {symbol} - not a trading minute")
                self.trade_allowed[symbol] = False
        
        return self.trade_allowed.get(symbol, False)
    
    def update_current_candle(self, symbol, tick):
        """
        Update the current candle with tick data.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
        """
        current_time = datetime.now()
        last_price = tick.get('last_price')
        
        if not last_price:
            return
        
        with self.candle_lock:
            # Initialize candle if needed
            if symbol not in self.current_candle or self.current_candle[symbol] is None:
                # Round down to nearest timeframe interval
                candle_time = current_time.replace(
                    minute=(current_time.minute // self.timeframe) * self.timeframe,
                    second=0,
                    microsecond=0
                )
                
                self.current_candle[symbol] = {
                    'timestamp': candle_time,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': tick.get('volume', 0)
                }
                self.logger.info(f"Initialized new candle for {symbol} at {candle_time}: O={last_price}, H={last_price}, L={last_price}, C={last_price}")
            else:
                # Update existing candle
                current_candle = self.current_candle[symbol]
                
                # Update high and low
                if last_price > current_candle['high']:
                    current_candle['high'] = last_price
                
                if last_price < current_candle['low']:
                    current_candle['low'] = last_price
                
                # Always update close price and volume
                current_candle['close'] = last_price
                if 'volume' in tick:
                    current_candle['volume'] = tick['volume']
                
                self.current_candle[symbol] = current_candle
    
    def check_candle_formation(self, symbol, tick):
        """
        Check if we should form a new candle based on time or bar_close flag.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
        """
        current_time = datetime.now()
        
        # Check if tick has bar_close flag
        if 'bar_close' in tick and tick['bar_close']:
            self.logger.info(f"Bar close flag detected for {symbol}")
            self.form_new_candle(symbol, tick)
            return
        
        # Check time-based candle formation
        # Get the timestamp of when the current candle started
        if symbol in self.current_candle and self.current_candle[symbol]:
            candle_start_time = self.current_candle[symbol]['timestamp']
            timeframe_minutes = self.timeframe
            
            # Check if current time is in a new candle period
            next_candle_time = candle_start_time + timedelta(minutes=timeframe_minutes)
            
            if current_time >= next_candle_time:
                self.logger.info(f"Time-based candle closure for {symbol}: {candle_start_time} -> {next_candle_time}")
                self.form_new_candle(symbol, tick)
    
    def form_new_candle(self, symbol, tick):
        """
        Form a new candle and update historical data.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
        """
        with self.candle_lock:
            # Skip if there's no current candle
            if symbol not in self.current_candle or self.current_candle[symbol] is None:
                return
            
            # Get the current candle
            current_candle = self.current_candle[symbol]
            
            self.logger.info(f"Forming new candle for {symbol}: {current_candle}")
            
            # Get historical data
            data = self.strategy.get_historical_data(symbol)
            if data is None:
                self.logger.warning(f"No historical data available for {symbol}")
                return
            
            # Create a new row to append
            new_row = pd.DataFrame([current_candle], index=[current_candle['timestamp']])
            
            # Append to historical data
            updated_data = pd.concat([data, new_row])
            
            # Update strategy with new data
            self.strategy.update_historical_data(symbol, updated_data)
            
            # Calculate indicators
            indicators = self.strategy.calculate_indicators(updated_data)
            self.strategy.update_indicators(symbol, indicators)
            
            # Log the candle formation and indicator values
            self.logger.info(f"New candle added for {symbol}: {current_candle}")
            self.log_indicator_values(symbol, indicators)
            
            # Mark the current candle as processed
            self.last_candle_time[symbol] = current_candle['timestamp']
            
            # Reset current candle
            self.current_candle[symbol] = None
            
            # Reset trade allowed flag to ensure we check again for the new candle
            self.trade_allowed[symbol] = False
    
    def log_indicator_values(self, symbol, indicators):
        """
        Log indicator values for debugging.
        
        Args:
            symbol: Trading symbol
            indicators: Indicator dictionary
        """
        if not indicators or 'ema_short' not in indicators or 'ema_long' not in indicators:
            return
        
        # Get latest values
        ema_short = indicators['ema_short'].iloc[-1] if not indicators['ema_short'].empty else None
        ema_long = indicators['ema_long'].iloc[-1] if not indicators['ema_long'].empty else None
        
        if ema_short is not None and ema_long is not None:
            ema_diff = ema_short - ema_long
            ema_diff_pct = (ema_diff / ema_long) * 100 if ema_long != 0 else 0
            
            self.logger.info(f"{symbol} Indicators (Latest):")
            self.logger.info(f"  Short EMA: {ema_short:.2f}")
            self.logger.info(f"  Long EMA: {ema_long:.2f}")
            self.logger.info(f"  EMA Diff: {ema_diff:.2f} ({ema_diff_pct:.2f}%)")
    
    def execute_order(self, symbol, signal, order_params):
        """
        Execute an order based on the signal and parameters.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
            order_params: Order parameters
        """
        if not order_params:
            return
            
        action = signal.get('action')
        
        # Check if there's already a pending order for this symbol
        if symbol in self.pending_orders:
            pending_order_id = self.pending_orders[symbol]
            pending_status = self.order_status.get(pending_order_id)
            
            # If there's a pending order and it's still open, don't place a new one
            if pending_status in ['OPEN', 'PENDING', 'TRIGGER PENDING']:
                self.logger.warning(f"Skipping new order for {symbol} because there's already a pending order: {pending_order_id}")
                return
        
        try:
            # Round price according to tick size if present in order parameters
            if 'price' in order_params and order_params['price'] is not None:
                is_buy = action == 'BUY'
                order_params['price'] = self.round_to_tick_size(symbol, order_params['price'], is_buy)
            
            if action == 'BUY':
                self.logger.info(f"Placing BUY order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                order_id = self.order_manager.place_order(**order_params)
                self.logger.info(f"Order placed successfully: {order_id}")
                
                # Track the pending order
                self.pending_orders[symbol] = order_id
                self.order_status[order_id] = 'OPEN'
                
                # Register the order with the strategy for tracking
                if hasattr(self.strategy, 'register_order'):
                    self.strategy.register_order(symbol, order_id, 'BUY')
                
            elif action == 'SELL':
                self.logger.info(f"Placing SELL order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                order_id = self.order_manager.place_order(**order_params)
                self.logger.info(f"Order placed successfully: {order_id}")
                
                # Track the pending order
                self.pending_orders[symbol] = order_id
                self.order_status[order_id] = 'OPEN'
                
                # Register the order with the strategy for tracking
                if hasattr(self.strategy, 'register_order'):
                    self.strategy.register_order(symbol, order_id, 'SELL')
                
            elif action == 'UPDATE':
                self.logger.info(f"Updating order for {symbol} - New price: {order_params.get('price')}")
                
                order_id = order_params.pop('order_id')  # Remove order_id from parameters
                
                # Round the modification price according to tick size
                if 'price' in order_params and order_params['price'] is not None:
                    # Get the side from the strategy's active orders
                    side = 'BUY'  # Default to BUY for safety
                    if hasattr(self.strategy, 'active_orders') and symbol in self.strategy.active_orders:
                        side = self.strategy.active_orders[symbol].get('side', 'BUY')
                    
                    is_buy = side == 'BUY'
                    order_params['price'] = self.round_to_tick_size(symbol, order_params['price'], is_buy)
                    
                modified_order_id = self.order_manager.modify_order(
                    order_id=order_id,
                    price=order_params.get('price')
                )
                self.logger.info(f"Order updated successfully: {modified_order_id}")
                
        except Exception as e:
            self.logger.error(f"Error executing {action} order for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
    
    def on_order_update(self, ws, order_data):
        """
        Callback for order updates.
        
        Args:
            ws: WebSocket instance
            order_data: Order update data
        """
        if not order_data:
            return
        
        try:
            order_id = order_data.get('order_id')
            status = order_data.get('status')
            symbol = order_data.get('tradingsymbol')
            
            self.logger.info(f"Order update received: {order_id}, Status: {status}, Symbol: {symbol}")
            
            # Update order status in our tracking
            if order_id:
                self.order_status[order_id] = status
                
                # If order completed or cancelled, remove from pending orders
                if status in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                    for sym, pending_id in list(self.pending_orders.items()):
                        if pending_id == order_id:
                            del self.pending_orders[sym]
                            self.logger.info(f"Removed {sym} from pending orders: {order_id} status is {status}")
            
            # Update strategy with order status
            if symbol and hasattr(self.strategy, 'on_order_update'):
                self.strategy.on_order_update(symbol, order_data)
                
                # If order is filled, update position
                if status == 'COMPLETE':
                    self.logger.info(f"Order {order_id} for {symbol} completed")
                    
                    # Get latest position from Zerodha
                    self.update_positions()
        
        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")
            self.logger.error(traceback.format_exc())
    
    def update_positions(self):
        """Update positions in strategy from Zerodha."""
        try:
            # Get positions from Zerodha
            positions = self.order_manager.get_positions()
            day_positions = positions.get('day', [])
            
            # Update strategy positions
            for position in day_positions:
                symbol = position.get('tradingsymbol')
                if symbol in self.strategy.universe:
                    self.strategy.update_position(symbol, {
                        'quantity': position.get('quantity', 0),
                        'average_price': position.get('average_price', 0)
                    })
                    self.logger.info(f"Updated position for {symbol}: {position.get('quantity')} @ {position.get('average_price')}")
        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            self.logger.error(traceback.format_exc())
    
    def check_pending_orders(self):
        """Check for orders that have been pending too long and cancel them."""
        current_time = datetime.now()
        
        for symbol, order_id in list(self.pending_orders.items()):
            # Get the current status
            status = self.order_status.get(order_id)
            
            # If not in open state, remove from pending
            if status not in ['OPEN', 'PENDING', 'TRIGGER PENDING']:
                del self.pending_orders[symbol]
                continue
            
            # Check if order has been pending for more than 5 minutes
            # Without a timestamp, we'll have to use a crude method to check
            try:
                # Get order history to see when it was placed
                order_history = self.order_manager.get_order_history(order_id)
                if order_history:
                    created_time = order_history[0].get('order_timestamp')
                    if created_time:
                        order_age = (current_time - created_time).total_seconds()
                        
                        # If order has been pending for more than 5 minutes, cancel it
                        if order_age > 300:  # 5 minutes = 300 seconds
                            self.logger.info(f"Cancelling stale order {order_id} for {symbol} (age: {order_age:.1f} seconds)")
                            self.order_manager.cancel_order(order_id=order_id, variety="regular")
                            del self.pending_orders[symbol]
            except Exception as e:
                self.logger.error(f"Error checking order history for {order_id}: {e}")
    
    def log_positions_and_pnl(self):
        """Log current positions, P&L, and indicator values for debugging."""
        try:
            # Get latest positions from Zerodha
            positions = self.order_manager.get_positions()
            day_positions = positions.get('day', [])
            
            # Update strategy positions
            for position in day_positions:
                symbol = position.get('tradingsymbol')
                if symbol in self.strategy.universe:
                    self.strategy.update_position(symbol, {
                        'quantity': position.get('quantity', 0),
                        'average_price': position.get('average_price', 0)
                    })
            
            # Log positions and P&L
            self.logger.info("=== CURRENT POSITIONS AND P&L ===")
            total_pnl = 0
            
            for symbol in self.strategy.universe:
                # Get position from day positions
                position = next((p for p in day_positions if p.get('tradingsymbol') == symbol), None)
                
                if position and position.get('quantity', 0) != 0:
                    quantity = position.get('quantity', 0)
                    avg_price = position.get('average_price', 0)
                    last_price = position.get('last_price', 0)
                    pnl = position.get('pnl', 0)
                    
                    position_type = "LONG" if quantity > 0 else "SHORT" if quantity < 0 else "NONE"
                    
                    self.logger.info(f"{symbol}: {position_type} {abs(quantity)} @ {avg_price:.2f}, Last: {last_price:.2f}, P&L: {pnl:.2f}")
                    total_pnl += pnl
                else:
                    self.logger.info(f"{symbol}: No position")
            
            self.logger.info(f"Total P&L: {total_pnl:.2f}")
            
            # Log indicator values for debugging
            self.logger.info("\n=== INDICATOR VALUES AND CONDITIONS ===")
            
            for symbol in self.strategy.universe:
                # Get historical data
                data = self.strategy.get_historical_data(symbol)
                
                if data is None or len(data) < 2:
                    self.logger.warning(f"{symbol}: Insufficient historical data for indicator logging")
                    continue
                
                # Get latest instrument price
                current_price = self.last_price.get(symbol)
                self.logger.info(f"{symbol} Current Price: {current_price}")
                
                # Log the latest candle
                last_candle = data.iloc[-1] if not data.empty else None
                if last_candle is not None:
                    self.logger.info(f"{symbol} Last Candle:")
                    self.logger.info(f"  Timestamp: {data.index[-1]}")
                    self.logger.info(f"  Open: {last_candle['open']:.2f}")
                    self.logger.info(f"  High: {last_candle['high']:.2f}")
                    self.logger.info(f"  Low: {last_candle['low']:.2f}")
                    self.logger.info(f"  Close: {last_candle['close']:.2f}")
                    if 'volume' in last_candle:
                        self.logger.info(f"  Volume: {last_candle['volume']}")
                
                # Get indicators
                indicators = self.strategy.get_indicators(symbol)
                
                if not indicators or 'ema_short' not in indicators or 'ema_long' not in indicators:
                    self.logger.warning(f"{symbol}: Indicators not available")
                    continue
                
                # Get latest values
                last_close = data['close'].iloc[-1]
                last_short_ema = indicators['ema_short'].iloc[-1]
                last_long_ema = indicators['ema_long'].iloc[-1]
                ema_diff = last_short_ema - last_long_ema
                ema_diff_pct = (ema_diff / last_long_ema) * 100 if last_long_ema != 0 else 0
                
                # Get crossover signal (if any)
                last_signal = indicators['crossover_signal'].iloc[-1] if 'crossover_signal' in indicators else 0
                signal_type = "BULLISH" if last_signal > 0 else "BEARISH" if last_signal < 0 else "NONE"
                
                # Log the indicator values
                self.logger.info(f"{symbol} Indicators (Last Candle):")
                self.logger.info(f"  Close: {last_close:.2f}")
                self.logger.info(f"  Short EMA ({self.short_ema_period}): {last_short_ema:.2f}")
                self.logger.info(f"  Long EMA ({self.long_ema_period}): {last_long_ema:.2f}")
                self.logger.info(f"  EMA Diff: {ema_diff:.2f} ({ema_diff_pct:.2f}%)")
                self.logger.info(f"  Signal: {signal_type}")
                
                # Check for pending orders
                has_pending = symbol in self.pending_orders
                self.logger.info(f"  Pending Orders: {'YES' if has_pending else 'NO'}")
                
                # Check if trading is allowed
                is_trading_allowed = self.trade_allowed.get(symbol, False)
                self.logger.info(f"  Trading Allowed: {'YES' if is_trading_allowed else 'NO'}")
                
                # Log tick size
                tick_size = self.tick_sizes.get(symbol, "Unknown")
                self.logger.info(f"  Tick Size: {tick_size}")
                
                # Check if we've already traded on the current candle
                current_candle_time = self.current_candle.get(symbol, {}).get('timestamp')
                last_trade_time = self.last_trade_candle_time.get(symbol, datetime.min)
                traded_current_candle = (current_candle_time and last_trade_time and 
                                        current_candle_time == last_trade_time)
                
                self.logger.info(f"  Traded on Current Candle: {'YES' if traded_current_candle else 'NO'}")
                
                # Log crossover proximity
                if signal_type == "NONE" and abs(ema_diff_pct) < 0.5:
                    self.logger.info(f"  ALERT: Close to crossover! ({ema_diff_pct:.2f}%)")
                
                self.logger.info("")
            
            # Log time until market close and square-off
            current_time = datetime.now().time()
            market_close = self.market_close_time
            exit_time = self.square_off_time
            
            time_to_exit = None
            if exit_time > current_time:
                exit_td = datetime.combine(datetime.today(), exit_time) - datetime.combine(datetime.today(), current_time)
                time_to_exit = exit_td.total_seconds() / 60  # In minutes
            
            time_to_close = None
            if market_close > current_time:
                close_td = datetime.combine(datetime.today(), market_close) - datetime.combine(datetime.today(), current_time)
                time_to_close = close_td.total_seconds() / 60  # In minutes
            
            self.logger.info("=== TIME INFORMATION ===")
            self.logger.info(f"Current Time: {current_time}")
            self.logger.info(f"Exit Time: {exit_time} ({'N/A' if time_to_exit is None else f'{time_to_exit:.1f} mins remaining'})")
            self.logger.info(f"Market Close: {market_close} ({'N/A' if time_to_close is None else f'{time_to_close:.1f} mins remaining'})")
            self.logger.info("="*40)
            
            # Log WebSocket status
            self.logger.info("=== WEBSOCKET STATUS ===")
            self.logger.info(f"Connected: {self.realtime_data.is_connected}")
            self.logger.info(f"Reconnect Attempts: {self.ws_reconnect_attempts}")
            
            # Log tick status
            self.logger.info("=== TICK STATUS ===")
            current_time = datetime.now()
            for symbol in self.symbols:
                last_tick = self.last_tick_time.get(symbol, datetime.min)
                seconds_since_last_tick = (current_time - last_tick).total_seconds()
                self.logger.info(f"{symbol}: Last tick {seconds_since_last_tick:.1f} seconds ago")
            
        except Exception as e:
            self.logger.error(f"Error logging positions, P&L, and indicators: {e}")
            self.logger.error(traceback.format_exc())
    
    def shutdown(self):
        """Cleanup and shutdown."""
        try:
            # Stop real-time data
            if self.realtime_data:
                self.realtime_data.stop()
            
            # Final position log
            self.log_positions_and_pnl()
            
            self.logger.info("Live trading completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
    
    def start_candle_timer(self):
        """Start a background thread to manage candle formation."""
        if self.candle_timer_thread is not None and self.candle_timer_thread.is_alive():
            self.logger.info("Candle timer thread already running")
            return
        
        self.logger.info("Starting candle timer thread")
        self.candle_timer_thread = Thread(target=self._candle_timer_loop)
        self.candle_timer_thread.daemon = True
        self.candle_timer_thread.start()
    
    def _candle_timer_loop(self):
        """Background thread function to check and form candles at regular intervals."""
        self.logger.info("Candle timer thread started")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                current_minute = current_time.minute
                
                # Check if it's a candle boundary (multiple of timeframe)
                if current_minute % self.timeframe == 0 and current_time.second == 0:
                    self.logger.info(f"Timer: Candle boundary detected at {current_time.strftime('%H:%M:%S')}")
                    
                    # Check if any candles need to be formed
                    for symbol in self.symbols:
                        # Skip if symbol doesn't have a current candle
                        if symbol not in self.current_candle or self.current_candle[symbol] is None:
                            continue
                        
                        # Get the current candle start time
                        candle_start_time = self.current_candle[symbol]['timestamp']
                        
                        # If current candle is from previous interval, form a new one
                        if candle_start_time.minute != current_minute:
                            self.logger.info(f"Timer: Forming candle for {symbol} at boundary: {candle_start_time.minute} -> {current_minute}")
                            
                            # Get last price
                            last_price = self.last_price.get(symbol)
                            if last_price:
                                # Create a minimal tick with the last known price
                                tick = {
                                    'last_price': last_price,
                                    'bar_close': True
                                }
                                
                                # Form new candle
                                self.form_new_candle(symbol, tick)
            
            except Exception as e:
                self.logger.error(f"Error in candle timer thread: {e}")
                self.logger.error(traceback.format_exc())
            
            # Sleep for a short time before checking again (100ms)
            time.sleep(0.1)
        
        self.logger.info("Candle timer thread stopped")
    
    def run(self):
        """Run the live trading strategy."""
        # Print big warning about live trading
        print("\n")
        print("=" * 80)
        print("WARNING: LIVE TRADING MODE ACTIVATED")
        print("This script will place REAL ORDERS with REAL MONEY")
        print("=" * 80)
        print(f"Strategy: EMA Intraday Crossover")
        print(f"Symbols: {self.symbols}")
        print(f"Short EMA: {self.short_ema_period}, Long EMA: {self.long_ema_period}")
        print(f"Timeframe: {self.timeframe} minutes")
        print(f"Max Position Size: {self.max_position_size}")
        print("=" * 80)
        print("\n")
        
        # Confirm if not using --confirm flag
        if not self.args.confirm:
            confirm = input("Are you sure you want to continue with live trading? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                print("Live trading canceled by user")
                return False
        
        self.logger.info("Starting EMA Intraday Crossover Strategy in LIVE TRADING mode")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Short EMA: {self.short_ema_period}, Long EMA: {self.long_ema_period}")
        self.logger.info(f"Timeframe: {self.timeframe} minutes")
        self.logger.info(f"Max Position Size: {self.max_position_size}")
        
        # Setup and initialize
        if not self.authenticate():
            return False
            
        if not self.initialize_components():
            return False
            
        if not self.setup_universe():
            return False
            
        if not self.setup_callbacks():
            return False
        
        # Start real-time data - we'll subscribe to tokens in the on_connect callback
        if not self.realtime_data.start():
            self.logger.error("Failed to start real-time data")
            return False
            
        self.logger.info("WebSocket connection started")
        
        # Start the candle timer thread
        self.is_running = True
        self.start_candle_timer()
        
        self.logger.info("Live trading started")
        
        # Log initial positions and indicators
        self.log_positions_and_pnl()
        
        try:
            # Main loop - run until market close
            while self.is_running:
                current_time = datetime.now()
                current_time_of_day = current_time.time()
                
                # Check WebSocket connection
                self.check_websocket_connection()
                
                # Log current positions, P&L and indicators at regular intervals
                if (current_time - self.last_position_log_time).total_seconds() >= self.debug_log_interval:
                    self.log_positions_and_pnl()
                    self.last_position_log_time = current_time
                
                # Check for stale pending orders every minute
                if (current_time - self.last_order_check_time).total_seconds() >= 60:  # 1 minute
                    self.check_pending_orders()
                    self.last_order_check_time = current_time
                
                time.sleep(0.1)  # Sleep to reduce CPU usage (100ms)
                
                # Check if market closed (after 3:30 PM)
                if current_time_of_day >= self.market_close_time:
                    self.logger.info("Market closed, ending live trading")
                    break
                
                # Check if we should square off all positions (usually 15 minutes before market close)
                if current_time_of_day >= self.square_off_time:
                    # Get positions
                    positions = self.order_manager.get_positions()
                    day_positions = positions.get('day', [])
                    
                    # Check if we have any open positions
                    open_positions = [p for p in day_positions if p.get('quantity', 0) != 0]
                    
                    if open_positions:
                        self.logger.info(f"Square-off time reached ({current_time_of_day}). Closing all positions.")
                        
                        # Square off each position
                        for position in open_positions:
                            symbol = position.get('tradingsymbol')
                            quantity = position.get('quantity', 0)
                            
                            if quantity != 0:
                                # Place market order to close position
                                action = "SELL" if quantity > 0 else "BUY"
                                
                                # Prepare order parameters
                                order_params = {
                                    "symbol": symbol,
                                    "exchange": position.get('exchange', 'NSE'),
                                    "transaction_type": action,
                                    "quantity": abs(quantity),
                                    "product": "MIS",
                                    "order_type": "MARKET",
                                    "tag": "square_off"
                                }
                                
                                try:
                                    self.logger.info(f"Square-off: Placing {action} order for {symbol} - {abs(quantity)} shares")
                                    order_id = self.order_manager.place_order(**order_params)
                                    self.logger.info(f"Square-off order placed successfully: {order_id}")
                                except Exception as e:
                                    self.logger.error(f"Error placing square-off order for {symbol}: {e}")
                        
                        # After placing square-off orders, exit the loop
                        self.logger.info("All square-off orders placed. Exiting.")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("Live trading stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.is_running = False
            self.shutdown()
        
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Trading EMA Intraday Crossover Strategy")
    
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Confirm live trading without additional prompt"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    trader = EMAStrategyLiveTrader(args)
    success = trader.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())