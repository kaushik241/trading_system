#!/usr/bin/env python
"""
Live Trading Runner for the EMA Intraday Crossover Strategy.

This script runs the strategy in live trading mode, placing actual orders in the market.
Use with caution as it will risk real money.

Usage:
    python run_ema_strategy_live.py --symbols RELIANCE,HDFCBANK --short-ema 9 --long-ema 21 --timeframe 5 --max-position 1
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager
from strategy.ema_crossover_strategy import EMAIntraDayStrategy
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager
load_dotenv(override=True)
# Import configuration (with fallback if not found)
try:
    from config.ema_strategy_config import (
        SYMBOLS, SHORT_EMA_PERIOD, LONG_EMA_PERIOD, 
        TIMEFRAME, MAX_POSITION_SIZE, STOP_LOSS_PERCENT,
        LOG_LEVEL, LOG_DIR, LOG_TO_FILE, LOG_TO_CONSOLE,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME, SQUARE_OFF_TIME
    )
    config_loaded = True
except ImportError:
    config_loaded = False
    LOG_LEVEL = "INFO"
    LOG_DIR = "data/logs"
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    STOP_LOSS_PERCENT = 0.01


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
        
        # Trading parameters
        self.symbols = [s.strip() for s in args.symbols.split(',')]
        self.token_map = {}
        self.symbol_token_map = {}
        self.token_symbol_map = {}
        
        # State tracking
        self.is_running = False
        self.last_position_log_time = datetime.min
        self.last_order_check_time = datetime.min
        self.debug_log_interval = 300  # 5 minutes in seconds
        
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
            return False
    
    def initialize_components(self):
        """Initialize all required components."""
        try:
            # Check if authentication was successful
            if not self.kite:
                self.logger.error("Authentication not completed. Cannot initialize components.")
                return False
            
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
                stop_loss_pct=STOP_LOSS_PERCENT if config_loaded else 0.01  # 1% stop loss
            )
            
            # Format timeframe string
            timeframe = f"{self.args.timeframe}minute"
            
            # Initialize strategy
            self.strategy = EMAIntraDayStrategy(
                name="EMA_Intraday_Live",
                universe=self.symbols,
                timeframe=timeframe,
                short_ema_period=self.args.short_ema,
                long_ema_period=self.args.long_ema,
                max_position_size=self.args.max_position
            )
            
            # Set custom debug logging interval if specified
            if hasattr(self.args, 'debug_interval') and self.args.debug_interval > 0:
                self.debug_log_interval = self.args.debug_interval
                self.logger.info(f"Setting debug logging interval to {self.debug_log_interval} seconds")
            
            self.logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def setup_universe(self):
        """Set up the trading universe and fetch initial data."""
        try:
            # Initialize positions for all symbols
            self.logger.info("Initializing positions for all symbols")
            for symbol in self.symbols:
                self.strategy.update_position(symbol, {"quantity": 0, "average_price": 0})
            
            # Get instrument tokens
            self.logger.info("Fetching instrument tokens for all symbols")
            self.token_map = self.historical_data_manager.get_instrument_tokens(self.symbols)
            self.logger.info(f"Retrieved {len(self.token_map)} instrument tokens")
            
            # Check if we got all tokens
            if len(self.token_map) != len(self.symbols):
                missing_symbols = set(self.symbols) - set(self.token_map.keys())
                if missing_symbols:
                    self.logger.warning(f"Missing tokens for: {missing_symbols}")
            
            # Create token mappings
            tokens = list(self.token_map.values())
            self.symbol_token_map = self.token_map
            self.token_symbol_map = {v: k for k, v in self.token_map.items()}
            
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
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up universe: {e}")
            return False
    
    def fetch_historical_data(self, days_back=7):
        """
        Fetch historical data for initialization.
        
        Args:
            days_back: Number of days of historical data to fetch
            
        Returns:
            Dictionary of dataframes with historical data
        """
        historical_data = {}
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
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
                interval = self.strategy.timeframe
                
                df = self.historical_data_manager.fetch_historical_data(
                    token, from_date_str, to_date_str, interval
                )
                
                if df is not None and not df.empty:
                    historical_data[symbol] = df
                    self.logger.info(f"Fetched {len(df)} {interval} candles for {symbol}")
                else:
                    self.logger.warning(f"No data fetched for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        return historical_data
    
    def setup_callbacks(self):
        """Setup callbacks for the real-time data manager."""
        # Register callbacks
        self.realtime_data.register_callback('on_tick', self.on_tick)
        self.realtime_data.register_callback('on_order_update', self.on_order_update)
        self.realtime_data.register_callback('on_connect', self.on_connect)
        
        return True
        
    def on_connect(self, response):
        """
        Callback when WebSocket connection is established.
        
        Args:
            response: Connection response
        """
        self.logger.info(f"WebSocket connected: {response}")
        
        # Now that we're connected, subscribe to tokens
        tokens = list(self.token_map.values())
        if tokens:
            self.logger.info(f"Subscribing to {len(tokens)} tokens")
            self.realtime_data.subscribe(tokens, self.token_symbol_map, self.token_map)
        else:
            self.logger.error("No tokens to subscribe to")
    
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
        
        try:
            # Process tick with strategy
            signal = self.strategy.process_tick(symbol, tick)
            
            # If a signal is generated, process it and place an order
            if signal:
                self.logger.info(f"Signal generated for {symbol}: {signal}")
                
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
        
        try:
            if action == 'BUY':
                self.logger.info(f"Placing BUY order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                order_id = self.order_manager.place_order(**order_params)
                self.logger.info(f"Order placed successfully: {order_id}")
                
                # Register the order with the strategy for tracking
                self.strategy.register_order(symbol, order_id, 'BUY')
                
            elif action == 'SELL':
                self.logger.info(f"Placing SELL order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                order_id = self.order_manager.place_order(**order_params)
                self.logger.info(f"Order placed successfully: {order_id}")
                
                # Register the order with the strategy for tracking
                self.strategy.register_order(symbol, order_id, 'SELL')
                
            elif action == 'UPDATE':
                self.logger.info(f"Updating order for {symbol} - New price: {order_params.get('price')}")
                
                order_id = order_params.pop('order_id')  # Remove order_id from parameters
                modified_order_id = self.order_manager.modify_order(
                    order_id=order_id,
                    price=order_params.get('price')
                )
                self.logger.info(f"Order updated successfully: {modified_order_id}")
                
        except Exception as e:
            self.logger.error(f"Error executing {action} order for {symbol}: {e}")
    
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
            
            # Update strategy with order status
            if symbol and self.strategy:
                self.strategy.on_order_update(symbol, order_data)
                
                # If order is filled, update position
                if status == 'COMPLETE':
                    self.logger.info(f"Order {order_id} for {symbol} completed")
                    
                    # Get latest position from Zerodha
                    self.update_positions()
        
        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")
    
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
    
    def check_pending_orders(self):
        """Check for orders that have been pending too long and cancel them."""
        current_time = datetime.now()
        orders_to_cancel = []
        
        for symbol, order_info in self.strategy.active_orders.items():
            # Skip if not in OPEN state
            if order_info.get('status') != 'OPEN':
                continue
                
            # Check how long the order has been pending
            order_time = order_info.get('timestamp', current_time - timedelta(minutes=30))
            order_age = (current_time - order_time).total_seconds()
            
            # If order has been pending for more than 5 minutes, cancel it
            if order_age > 300:  # 5 minutes = 300 seconds
                order_id = order_info.get('order_id')
                orders_to_cancel.append((symbol, order_id))
        
        # Cancel old pending orders
        for symbol, order_id in orders_to_cancel:
            try:
                self.logger.info(f"Cancelling stale order {order_id} for {symbol}")
                self.order_manager.cancel_order(order_id=order_id, variety="regular")
                
                # Remove from active orders
                if symbol in self.strategy.active_orders:
                    del self.strategy.active_orders[symbol]
                    
            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
    
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
                
                if position:
                    quantity = position.get('quantity', 0)
                    avg_price = position.get('average_price', 0)
                    last_price = position.get('last_price', 0)
                    pnl = position.get('pnl', 0)
                    
                    position_type = "LONG" if quantity > 0 else "SHORT" if quantity < 0 else "NONE"
                    
                    if quantity != 0:
                        self.logger.info(f"{symbol}: {position_type} {abs(quantity)} @ {avg_price:.2f}, Last: {last_price:.2f}, P&L: {pnl:.2f}")
                        total_pnl += pnl
                    else:
                        self.logger.info(f"{symbol}: No position")
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
                self.logger.info(f"  Short EMA ({self.strategy.short_ema_period}): {last_short_ema:.2f}")
                self.logger.info(f"  Long EMA ({self.strategy.long_ema_period}): {last_long_ema:.2f}")
                self.logger.info(f"  EMA Diff: {ema_diff:.2f} ({ema_diff_pct:.2f}%)")
                self.logger.info(f"  Signal: {signal_type}")
                
                # Check for pending orders
                has_pending = self.strategy.has_pending_orders(symbol)
                self.logger.info(f"  Pending Orders: {'YES' if has_pending else 'NO'}")
                
                # Check if symbol is stopped due to stop loss
                is_stopped = symbol in self.strategy.stopped_symbols
                self.logger.info(f"  Stopped (Hit Stop Loss): {'YES' if is_stopped else 'NO'}")
                
                # Print entry price and stop loss if position exists
                entry_price = self.strategy.entry_prices.get(symbol)
                stop_loss = self.strategy.stop_losses.get(symbol)
                
                if entry_price:
                    self.logger.info(f"  Entry Price: {entry_price:.2f}")
                    
                if stop_loss:
                    self.logger.info(f"  Stop Loss: {stop_loss:.2f}")
                    stop_distance_pct = abs((stop_loss - entry_price) / entry_price * 100) if entry_price else 0
                    self.logger.info(f"  Stop Distance: {stop_distance_pct:.2f}%")
                
                # Log crossover proximity
                if signal_type == "NONE" and abs(ema_diff_pct) < 0.5:
                    self.logger.info(f"  ALERT: Close to crossover! ({ema_diff_pct:.2f}%)")
                
                self.logger.info("")
            
            # Log time until market close and square-off
            current_time = datetime.now().time()
            market_close = self.strategy.market_close_time
            exit_time = self.strategy.exit_time
            
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
        
        except Exception as e:
            self.logger.error(f"Error logging positions, P&L, and indicators: {e}")
            import traceback
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
        print(f"Short EMA: {self.args.short_ema}, Long EMA: {self.args.long_ema}")
        print(f"Timeframe: {self.args.timeframe} minutes")
        print(f"Max Position Size: {self.args.max_position}")
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
        self.logger.info(f"Short EMA: {self.args.short_ema}, Long EMA: {self.args.long_ema}")
        self.logger.info(f"Timeframe: {self.args.timeframe} minutes")
        self.logger.info(f"Max Position Size: {self.args.max_position}")
        
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
        self.logger.info("Live trading started")
        self.is_running = True
        
        # Log initial positions and indicators
        self.log_positions_and_pnl()
        
        try:
            # Main loop - run until market close
            while self.is_running:
                current_time = datetime.now()
                
                # Log current positions, P&L and indicators at regular intervals
                if (current_time - self.last_position_log_time).total_seconds() >= self.debug_log_interval:
                    self.log_positions_and_pnl()
                    self.last_position_log_time = current_time
                
                # Check for stale pending orders every minute
                if (current_time - self.last_order_check_time).total_seconds() >= 60:  # 1 minute
                    self.check_pending_orders()
                    self.last_order_check_time = current_time
                
                time.sleep(1)  # Sleep to reduce CPU usage
                
                # Check if market closed (after 3:30 PM)
                current_time_of_day = current_time.time()
                if current_time_of_day >= self.strategy.market_close_time:
                    self.logger.info("Market closed, ending live trading")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Live trading stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.is_running = False
            self.shutdown()
        
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Trading EMA Intraday Crossover Strategy")
    
    # Use config values as defaults if available
    default_symbols = SYMBOLS if config_loaded else "RELIANCE,HDFCBANK"
    default_short_ema = SHORT_EMA_PERIOD if config_loaded else 9
    default_long_ema = LONG_EMA_PERIOD if config_loaded else 21
    default_timeframe = TIMEFRAME if config_loaded else 5
    default_max_position = MAX_POSITION_SIZE if config_loaded else 1
    
    parser.add_argument(
        "--symbols", 
        type=str,
        default=default_symbols,
        help="Comma-separated list of trading symbols"
    )
    
    parser.add_argument(
        "--short-ema", 
        type=int, 
        default=default_short_ema,
        help="Short EMA period"
    )
    
    parser.add_argument(
        "--long-ema", 
        type=int, 
        default=default_long_ema,
        help="Long EMA period"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=int, 
        default=default_timeframe,
        help="Candle timeframe in minutes"
    )
    
    parser.add_argument(
        "--max-position", 
        type=int, 
        default=default_max_position,
        help="Maximum position size"
    )
    
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Confirm live trading without additional prompt"
    )
    
    parser.add_argument(
        "--debug-interval", 
        type=int, 
        default=300,  # 5 minutes
        help="Interval in seconds for logging debug information"
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