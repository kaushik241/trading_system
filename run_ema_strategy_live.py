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
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager
from strategy.ema_crossover_strategy import EMAIntraDayStrategy
from execution.order_manager import OrderManager
from risk.risk_manager import RiskManager

# Configure logging
os.makedirs('data/logs', exist_ok=True)
log_file = f'data/logs/ema_live_trading_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global references
strategy = None
historical_data_manager = None
realtime_data = None
order_manager = None
risk_manager = None
symbol_token_map = {}  # Trading symbol -> instrument token
token_symbol_map = {}  # Instrument token -> trading symbol

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Trading EMA Intraday Crossover Strategy")
    
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
        "--confirm", 
        action="store_true",
        help="Confirm live trading without additional prompt"
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
    
    logger.info(f"{datetime.now()} - Fetching historical data from {from_date_str} to {to_date_str}")
    
    for symbol in symbols:
        try:
            # Get instrument token
            token = historical_data_manager.instrument_tokens.get(symbol)
            if not token:
                logger.warning(f"{datetime.now()} - No instrument token found for {symbol}")
                continue
                
            # Fetch data - Use the strategy's timeframe
            interval = f"{strategy.timeframe}"
            
            df = historical_data_manager.fetch_historical_data(
                token, from_date_str, to_date_str, interval
            )
            
            if df is not None and not df.empty:
                historical_data[symbol] = df
                logger.info(f"{datetime.now()} - Fetched {len(df)} {interval} candles for {symbol}")
            else:
                logger.warning(f"{datetime.now()} - No data fetched for {symbol}")
        except Exception as e:
            logger.error(f"{datetime.now()} - Error fetching data for {symbol}: {e}")
    
    return historical_data

def on_tick(symbol, tick):
    """
    Callback for tick data.
    
    This function is called for each tick received from the WebSocket.
    It processes the tick through the strategy and executes any generated signals.
    """
    global strategy, token_symbol_map, order_manager, risk_manager
    
    # Ensure symbol is a string, not a token
    if isinstance(symbol, int):
        # If we get a token instead of a symbol, convert it
        symbol = token_symbol_map.get(symbol, str(symbol))
    
    try:
        # Process tick with strategy
        signal = strategy.process_tick(symbol, tick)
        
        # If a signal is generated, process it and place an order
        if signal:
            logger.info(f"{datetime.now()} - Signal generated for {symbol}: {signal}")
            
            # Prepare order parameters
            order_params = strategy.prepare_order_parameters(signal, tick)
            
            # Apply risk management
            if 'action' not in signal or signal['action'] not in ['UPDATE']:
                # Only apply risk management to new orders, not updates
                risk_adjusted_params = risk_manager.process_signal(signal, tick)
                if risk_adjusted_params:
                    # Merge with original parameters, keeping the strategy's order_type and price
                    order_params.update({
                        k: v for k, v in risk_adjusted_params.items() 
                        if k not in ['order_type', 'price'] and k in order_params
                    })
                    logger.info(f"{datetime.now()} - Risk-adjusted order parameters: {order_params}")
                else:
                    logger.warning(f"{datetime.now()} - Signal rejected by risk manager")
                    return
            
            # Execute the order based on action type
            if signal.get('action') == 'BUY':
                logger.info(f"{datetime.now()} - Placing BUY order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                try:
                    order_id = order_manager.place_order(**order_params)
                    logger.info(f"{datetime.now()} - Order placed successfully: {order_id}")
                    
                    # Register the order with the strategy for tracking
                    strategy.register_order(symbol, order_id, 'BUY')
                    
                except Exception as e:
                    logger.error(f"{datetime.now()} - Error placing BUY order: {e}")
                
            elif signal.get('action') == 'SELL':
                logger.info(f"{datetime.now()} - Placing SELL order for {symbol} - {order_params.get('quantity')} shares at {order_params.get('price')}")
                
                try:
                    order_id = order_manager.place_order(**order_params)
                    logger.info(f"{datetime.now()} - Order placed successfully: {order_id}")
                    
                    # Register the order with the strategy for tracking
                    strategy.register_order(symbol, order_id, 'SELL')
                    
                except Exception as e:
                    logger.error(f"{datetime.now()} - Error placing SELL order: {e}")
            
            elif signal.get('action') == 'UPDATE':
                logger.info(f"{datetime.now()} - Updating order for {symbol} - New price: {order_params.get('price')}")
                
                try:
                    order_id = order_params.pop('order_id')  # Remove order_id from parameters
                    modified_order_id = order_manager.modify_order(
                        order_id=order_id,
                        price=order_params.get('price')
                    )
                    logger.info(f"{datetime.now()} - Order updated successfully: {modified_order_id}")
                    
                except Exception as e:
                    logger.error(f"{datetime.now()} - Error updating order: {e}")
    
    except Exception as e:
        logger.error(f"{datetime.now()} - Error processing tick for {symbol}: {e}")

def on_order_update(ws, order_data):
    """
    Callback for order updates.
    
    This function is called when order status changes.
    It updates the strategy with the new order status.
    """
    global strategy
    
    if not order_data:
        return
    
    try:
        order_id = order_data.get('order_id')
        status = order_data.get('status')
        symbol = order_data.get('tradingsymbol')
        
        logger.info(f"{datetime.now()} - Order update received: {order_id}, Status: {status}, Symbol: {symbol}")
        
        # Update strategy with order status
        if symbol and strategy:
            strategy.on_order_update(symbol, order_data)
            
            # If order is filled, update position
            if status == 'COMPLETE':
                logger.info(f"{datetime.now()} - Order {order_id} for {symbol} completed")
                
                # Get latest position from Zerodha
                update_positions()
    
    except Exception as e:
        logger.error(f"{datetime.now()} - Error processing order update: {e}")

def update_positions():
    """Update positions in strategy from Zerodha."""
    global strategy, order_manager
    
    try:
        # Get positions from Zerodha
        positions = order_manager.get_positions()
        day_positions = positions.get('day', [])
        
        # Update strategy positions
        for position in day_positions:
            symbol = position.get('tradingsymbol')
            if symbol in strategy.universe:
                strategy.update_position(symbol, {
                    'quantity': position.get('quantity', 0),
                    'average_price': position.get('average_price', 0)
                })
                logger.info(f"{datetime.now()} - Updated position for {symbol}: {position.get('quantity')} @ {position.get('average_price')}")
    
    except Exception as e:
        logger.error(f"{datetime.now()} - Error updating positions: {e}")

def run_live_trading():
    """Run the EMA strategy in live trading mode."""
    global strategy, historical_data_manager, realtime_data, order_manager, risk_manager
    global symbol_token_map, token_symbol_map
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert symbols string to list
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Print big warning about live trading
    print("\n")
    print("=" * 80)
    print("WARNING: LIVE TRADING MODE ACTIVATED")
    print("This script will place REAL ORDERS with REAL MONEY")
    print("=" * 80)
    print(f"Strategy: EMA Intraday Crossover")
    print(f"Symbols: {symbols}")
    print(f"Short EMA: {args.short_ema}, Long EMA: {args.long_ema}")
    print(f"Timeframe: {args.timeframe} minutes")
    print(f"Max Position Size: {args.max_position}")
    print("=" * 80)
    print("\n")
    
    # Confirm if not using --confirm flag
    if not args.confirm:
        confirm = input("Are you sure you want to continue with live trading? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Live trading canceled by user")
            return False
    
    logger.info(f"{datetime.now()} - Starting EMA Intraday Crossover Strategy in LIVE TRADING mode")
    logger.info(f"{datetime.now()} - Symbols: {symbols}")
    logger.info(f"{datetime.now()} - Short EMA: {args.short_ema}, Long EMA: {args.long_ema}")
    logger.info(f"{datetime.now()} - Timeframe: {args.timeframe} minutes")
    logger.info(f"{datetime.now()} - Max Position Size: {args.max_position}")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key, secret, and access token from environment variables
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    
    if not api_key or not api_secret or not access_token:
        logger.error(f"{datetime.now()} - API key, secret, and access token are required")
        logger.error(f"{datetime.now()} - Please run the authentication script first")
        return False
    
    # Initialize ZerodhaAuth
    try:
        auth = ZerodhaAuth(api_key, api_secret, access_token)
        
        # Validate connection
        if not auth.validate_connection():
            logger.error(f"{datetime.now()} - Connection validation failed. Your access token may be expired.")
            logger.error(f"{datetime.now()} - Please run the authentication script again to get a new access token.")
            return False
            
        logger.info(f"{datetime.now()} - Authentication successful")
        
        # Get KiteConnect instance
        kite = auth.get_kite()
    except Exception as e:
        logger.error(f"{datetime.now()} - Authentication failed: {e}")
        return False
    
    # Initialize components
    historical_data_manager = HistoricalDataManager(kite)
    realtime_data = RealTimeDataManager(api_key, access_token)
    order_manager = OrderManager(kite)
    risk_manager = RiskManager(max_risk_per_trade=0.01, stop_loss_pct=0.01)  # 1% stop loss
    
    # Format timeframe string
    timeframe = f"{args.timeframe}minute"
    
    # Initialize strategy
    strategy = EMAIntraDayStrategy(
        name="EMA_Intraday_Live",
        universe=symbols,
        timeframe=timeframe,
        short_ema_period=args.short_ema,
        long_ema_period=args.long_ema,
        max_position_size=args.max_position
    )
    
    # Get instrument tokens
    logger.info(f"{datetime.now()} - Fetching instrument tokens for all symbols")
    token_map = historical_data_manager.get_instrument_tokens(symbols)
    logger.info(f"{datetime.now()} - Retrieved {len(token_map)} instrument tokens")
    
    # Check if we got all tokens
    if len(token_map) != len(symbols):
        logger.warning(f"{datetime.now()} - Could not find tokens for all symbols")
        missing_symbols = set(symbols) - set(token_map.keys())
        if missing_symbols:
            logger.warning(f"{datetime.now()} - Missing tokens for: {missing_symbols}")
    
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
        
        logger.info(f"{datetime.now()} - Initialized {symbol} with {len(data)} candles")
    
    # Update initial positions
    update_positions()
    
    # Register callbacks
    realtime_data.register_callback('on_tick', on_tick)
    realtime_data.register_callback('on_order_update', on_order_update)
    
    # Subscribe to tokens
    if tokens:
        logger.info(f"{datetime.now()} - Subscribing to {len(tokens)} tokens")
        realtime_data.subscribe(tokens, token_symbol_map, token_map)
    else:
        logger.error(f"{datetime.now()} - No tokens to subscribe to")
        return False
    
    # Start real-time data
    if not realtime_data.start():
        logger.error(f"{datetime.now()} - Failed to start real-time data")
        return False
    
    logger.info(f"{datetime.now()} - Live trading started")
    
    try:
        # Main loop - run until market close
        while True:
            # Log current positions and P&L every 5 minutes
            if datetime.now().minute % 5 == 0 and datetime.now().second < 5:
                log_positions_and_pnl()
                time.sleep(5)  # Sleep to avoid multiple logs in the same minute
            
            time.sleep(1)  # Sleep to reduce CPU usage
            
            # Check if market closed (after 3:30 PM)
            current_time = datetime.now().time()
            if current_time >= strategy.market_close_time:
                logger.info(f"{datetime.now()} - Market closed, ending live trading")
                break
    
    except KeyboardInterrupt:
        logger.info(f"{datetime.now()} - Live trading stopped by user")
    except Exception as e:
        logger.error(f"{datetime.now()} - Error in main loop: {e}")
    finally:
        # Stop real-time data
        realtime_data.stop()
        
        # Final position log
        log_positions_and_pnl()
        
        logger.info(f"{datetime.now()} - Live trading completed")
    
    return True

def log_positions_and_pnl():
    """Log current positions and P&L."""
    global strategy, order_manager
    
    try:
        # Get latest positions from Zerodha
        positions = order_manager.get_positions()
        day_positions = positions.get('day', [])
        
        # Update strategy positions
        for position in day_positions:
            symbol = position.get('tradingsymbol')
            if symbol in strategy.universe:
                strategy.update_position(symbol, {
                    'quantity': position.get('quantity', 0),
                    'average_price': position.get('average_price', 0)
                })
        
        # Log positions and P&L
        logger.info(f"{datetime.now()} - Current positions and P&L:")
        total_pnl = 0
        
        for symbol in strategy.universe:
            # Get position from day positions
            position = next((p for p in day_positions if p.get('tradingsymbol') == symbol), None)
            
            if position:
                quantity = position.get('quantity', 0)
                avg_price = position.get('average_price', 0)
                last_price = position.get('last_price', 0)
                pnl = position.get('pnl', 0)
                
                position_type = "LONG" if quantity > 0 else "SHORT" if quantity < 0 else "NONE"
                
                if quantity != 0:
                    logger.info(f"{datetime.now()} - {symbol}: {position_type} {abs(quantity)} @ {avg_price:.2f}, Last: {last_price:.2f}, P&L: {pnl:.2f}")
                    total_pnl += pnl
                else:
                    logger.info(f"{datetime.now()} - {symbol}: No position")
            else:
                logger.info(f"{datetime.now()} - {symbol}: No position")
        
        logger.info(f"{datetime.now()} - Total P&L: {total_pnl:.2f}")
    
    except Exception as e:
        logger.error(f"{datetime.now()} - Error logging positions and P&L: {e}")

if __name__ == "__main__":
    if not run_live_trading():
        sys.exit(1)