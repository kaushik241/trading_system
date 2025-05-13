#!/usr/bin/env python
"""
Test runner for the EMA Intraday Crossover Strategy.

This script runs the strategy in test mode, logging all actions without placing actual orders.
It simulates the strategy behavior with real market data but doesn't execute trades.

Usage:
    python test_ema_strategy.py --symbols RELIANCE,HDFCBANK --short-ema 9 --long-ema 21 --timeframe 5 --max-position 1
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager
from strategy.ema_crossover_strategy import EMAIntraDayStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/logs/test_ema_strategy_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global references
strategy = None
historical_data_manager = None
realtime_data = None
symbol_token_map = {}  # Trading symbol -> instrument token
token_symbol_map = {}  # Instrument token -> trading symbol

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test EMA Intraday Crossover Strategy")
    
    parser.add_argument(
        "--symbols", 
        type=str,
        default="RELIANCE,HDFCBANK",
        help="Comma-separated list of trading symbols"
    )
    
    parser.add_argument(
        "--short-ema", 
        type=int, 
        default=9,
        help="Short EMA period"
    )
    
    parser.add_argument(
        "--long-ema", 
        type=int, 
        default=21,
        help="Long EMA period"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=int, 
        default=5,
        help="Candle timeframe in minutes"
    )
    
    parser.add_argument(
        "--max-position", 
        type=int, 
        default=1,
        help="Maximum position size"
    )
    
    parser.add_argument(
        "--duration", 
        type=int, 
        default=360,  # 6 hours by default
        help="Test duration in minutes"
    )
    
    return parser.parse_args()

def fetch_historical_data(symbols, days_back=7):
    """
    Fetch historical data for initialization.
    
    Args:
        symbols: List of symbols to fetch data for
        days_back: Number of days of historical data to fetch
        
    Returns:
        Dictionary of dataframes with historical data
    """
    global historical_data_manager
    
    historical_data = {}
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    # Convert to string format required by API
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching historical data from {from_date_str} to {to_date_str}")
    
    for symbol in symbols:
        try:
            # Get instrument token
            token = historical_data_manager.instrument_tokens.get(symbol)
            if not token:
                logger.warning(f"No instrument token found for {symbol}")
                continue
                
            # Fetch data - Use the strategy's timeframe
            interval = f"{strategy.timeframe}"
            
            df = historical_data_manager.fetch_historical_data(
                token, from_date_str, to_date_str, interval
            )
            
            if df is not None and not df.empty:
                historical_data[symbol] = df
                logger.info(f"Fetched {len(df)} {interval} candles for {symbol}")
            else:
                logger.warning(f"No data fetched for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return historical_data

def on_tick(symbol, tick):
    """
    Callback for tick data.
    
    This function is called for each tick received from the WebSocket.
    It processes the tick through the strategy and logs the signals generated.
    """
    global strategy, token_symbol_map
    
    # Ensure symbol is a string, not a token
    if isinstance(symbol, int):
        # If we get a token instead of a symbol, convert it
        symbol = token_symbol_map.get(symbol, str(symbol))
    
    # Debug log to see ticks
    logger.debug(f"Received tick for {symbol}: Last price: {tick.get('last_price')}")
    
    try:
        # Process tick with strategy
        signal = strategy.process_tick(symbol, tick)
        
        # If a signal is generated, log it and simulate order placement
        if signal:
            logger.info(f"Signal generated for {symbol}: {signal}")
            
            # In a real scenario, we would execute the order
            # Here we just log it for simulation
            if signal.get('action') == 'BUY':
                logger.info(f"SIMULATION: Placing BUY order for {symbol} - {signal.get('quantity')} shares at {signal.get('price')}")
                
                # Simulate order fill (in a real scenario this would be done by order_manager)
                # For testing, we'll update the position directly
                current_position = strategy.get_position(symbol)
                current_qty = current_position.get('quantity', 0) if current_position else 0
                new_qty = current_qty + signal.get('quantity', 0)
                
                # Update position
                strategy.update_position(symbol, {
                    'quantity': new_qty,
                    'average_price': signal.get('price')
                })
                
                # Update entry price for stop loss calculation
                strategy.entry_prices[symbol] = signal.get('price')
                
                logger.info(f"SIMULATION: Position updated for {symbol} - New quantity: {new_qty}")
                
            elif signal.get('action') == 'SELL':
                logger.info(f"SIMULATION: Placing SELL order for {symbol} - {signal.get('quantity')} shares at {signal.get('price')}")
                
                # Simulate order fill
                current_position = strategy.get_position(symbol)
                current_qty = current_position.get('quantity', 0) if current_position else 0
                new_qty = current_qty - signal.get('quantity', 0)
                
                # Update position
                strategy.update_position(symbol, {
                    'quantity': new_qty,
                    'average_price': signal.get('price')
                })
                
                # Update entry price for stop loss calculation
                if new_qty < 0:  # Only update for short entries, not exits
                    strategy.entry_prices[symbol] = signal.get('price')
                
                logger.info(f"SIMULATION: Position updated for {symbol} - New quantity: {new_qty}")
            
            elif signal.get('action') == 'UPDATE':
                logger.info(f"SIMULATION: Updating order for {symbol} - New price: {signal.get('price')}")
                # In a real scenario, we would modify the order
                # Here we just log it
    
    except Exception as e:
        logger.error(f"Error processing tick for {symbol}: {e}")

def run_strategy_test():
    """Run the EMA strategy in test mode."""
    global strategy, historical_data_manager, realtime_data, symbol_token_map, token_symbol_map
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert symbols string to list
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    logger.info(f"Starting EMA Intraday Crossover Strategy Test")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Short EMA: {args.short_ema}, Long EMA: {args.long_ema}")
    logger.info(f"Timeframe: {args.timeframe} minutes")
    logger.info(f"Max Position Size: {args.max_position}")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key, secret, and access token from environment variables
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    
    if not api_key or not api_secret or not access_token:
        logger.error("API key, secret, and access token are required")
        logger.error("Please run the authentication script first")
        return False
    
    # Initialize ZerodhaAuth
    try:
        auth = ZerodhaAuth(api_key, api_secret, access_token)
        
        # Validate connection
        if not auth.validate_connection():
            logger.error("Connection validation failed. Your access token may be expired.")
            logger.error("Please run the authentication script again to get a new access token.")
            return False
            
        logger.info("Authentication successful")
        
        # Get KiteConnect instance
        kite = auth.get_kite()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False
    
    # Initialize components
    historical_data_manager = HistoricalDataManager(kite)
    realtime_data = RealTimeDataManager(api_key, access_token)
    
    # Format timeframe string
    timeframe = f"{args.timeframe}minute"
    
    # Initialize strategy
    strategy = EMAIntraDayStrategy(
        name="EMA_Intraday_Test",
        universe=symbols,
        timeframe=timeframe,
        short_ema_period=args.short_ema,
        long_ema_period=args.long_ema,
        max_position_size=args.max_position
    )
    
    # Initialize empty positions for all symbols
    logger.info("Initializing empty positions for all symbols")
    for symbol in symbols:
        strategy.update_position(symbol, {"quantity": 0, "average_price": 0})
    
    # Get instrument tokens
    logger.info("Fetching instrument tokens for all symbols")
    token_map = historical_data_manager.get_instrument_tokens(symbols)
    logger.info(f"Retrieved {len(token_map)} instrument tokens")
    
    # Check if we got all tokens
    if len(token_map) != len(symbols):
        logger.warning("Could not find tokens for all symbols")
        missing_symbols = set(symbols) - set(token_map.keys())
        if missing_symbols:
            logger.warning(f"Missing tokens for: {missing_symbols}")
    
    # Create token mappings
    tokens = list(token_map.values())
    symbol_token_map = token_map
    token_symbol_map = {v: k for k, v in token_map.items()}
    
    # Fetch historical data
    historical_data = fetch_historical_data(symbols)
    
    # Initialize strategy with historical data
    for symbol, data in historical_data.items():
        strategy.update_historical_data(symbol, data)
        
        # Calculate initial indicators
        indicators = strategy.calculate_indicators(data)
        strategy.update_indicators(symbol, indicators)
        
        logger.info(f"Initialized {symbol} with {len(data)} candles")
    
    # Register callback for tick data
    realtime_data.register_callback('on_tick', on_tick)
    
    # Subscribe to tokens
    if tokens:
        logger.info(f"Subscribing to {len(tokens)} tokens")
        realtime_data.subscribe(tokens, token_symbol_map, token_map)
    else:
        logger.error("No tokens to subscribe to")
        return False
    
    # Start real-time data
    if not realtime_data.start():
        logger.error("Failed to start real-time data")
        return False
    
    # Calculate end time
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=args.duration)
    
    logger.info(f"Strategy test started at {start_time}")
    logger.info(f"Test will run until {end_time} ({args.duration} minutes)")
    
    try:
        # Main loop - run until duration expires
        while datetime.now() < end_time:
            # Log current positions every 5 minutes
            if datetime.now().minute % 5 == 0 and datetime.now().second < 5:
                log_positions()
                time.sleep(5)  # Sleep to avoid multiple logs in the same minute
            
            time.sleep(1)  # Sleep to reduce CPU usage
            
            # Check if market closed (after 3:30 PM)
            current_time = datetime.now().time()
            if current_time >= strategy.market_close_time:
                logger.info("Market closed, ending test")
                break
    
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
    finally:
        # Stop real-time data
        realtime_data.stop()
        
        # Final position log
        log_positions()
        
        logger.info("Strategy test completed")
    
    return True

def log_positions():
    """Log current positions."""
    global strategy
    
    logger.info("Current positions:")
    for symbol in strategy.universe:
        position = strategy.get_position(symbol)
        position_qty = position.get('quantity', 0) if position else 0
        entry_price = strategy.entry_prices.get(symbol, 'N/A')
        
        if position_qty != 0:
            position_type = "LONG" if position_qty > 0 else "SHORT"
            logger.info(f"  {symbol}: {position_type} {abs(position_qty)} @ {entry_price}")
        else:
            logger.info(f"  {symbol}: No position")

if __name__ == "__main__":
    if not run_strategy_test():
        sys.exit(1)