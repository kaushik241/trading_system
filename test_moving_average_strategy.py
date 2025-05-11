#!/usr/bin/env python
"""
Test script for the Moving Average Strategy.

This script demonstrates how to use the MovingAverageStrategy class
with both historical and real-time data.

Usage:
    python test_moving_average_strategy.py
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager
from strategy.moving_average import MovingAverageStrategy
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global references for callbacks and data
strategy = None
risk_manager = None
order_manager = None
token_symbol_map = {}  # Map from token IDs to symbol names

def fetch_historical_data(symbols, from_date, to_date, interval, historical_data_manager):
    """Fetch historical data for symbols."""
    historical_data = {}
    
    for symbol in symbols:
        try:
            # Get instrument token
            token = historical_data_manager.instrument_tokens.get(symbol)
            if not token:
                logger.warning(f"No instrument token found for {symbol}")
                continue
                
            # Fetch data
            df = historical_data_manager.fetch_historical_data(
                token, from_date, to_date, interval
            )
            
            if df is not None and not df.empty:
                historical_data[symbol] = df
                logger.info(f"Fetched {len(df)} {interval} candles for {symbol}")
                
                # Print first and last few rows for debugging
                print(f"\nSample data for {symbol}:")
                print("First 3 candles:")
                print(df.head(3))
                print("Last 3 candles:")
                print(df.tail(3))
            else:
                logger.warning(f"No data fetched for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return historical_data

def on_tick(symbol, tick):
    """Callback for tick data."""
    global strategy, risk_manager, order_manager, token_symbol_map
    
    # Ensure symbol is a string, not a token
    if isinstance(symbol, int):
        # If we get a token instead of a symbol, convert it
        symbol = token_symbol_map.get(symbol, str(symbol))
    
    # Debug log to see ticks
    logger.debug(f"Received tick for {symbol}: Last price: {tick.get('last_price')}")
    
    # Process tick with strategy
    signals = strategy.generate_signals_realtime(symbol, tick)
    
    # Process signals with risk manager and execute orders
    for signal in signals:
        if not signal:
            continue
            
        # Apply risk management
        order_params = risk_manager.process_signal(signal, tick)
        
        if order_params:
            try:
                # Execute order (real or simulated)
                logger.info(f"Would place order: {order_params}")
                
                # In a real scenario, you would execute through order_manager:
                # order_id = order_manager.place_order(**order_params)
                # logger.info(f"Placed order with ID: {order_id}")
            except Exception as e:
                logger.error(f"Error placing order: {e}")

def test_strategy():
    """Test the Moving Average Strategy."""
    global strategy, risk_manager, order_manager, token_symbol_map
    
    print("\n" + "="*80)
    print("MOVING AVERAGE STRATEGY TEST")
    print("="*80)
    
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
    
    # Initialize components - make sure to initialize before using
    historical_data_manager = HistoricalDataManager(kite)
    realtime_data = RealTimeDataManager(api_key, access_token)
    risk_manager = RiskManager()
    order_manager = OrderManager(kite)
    
    # Initialize strategy with Nifty 50 stocks
    strategy = MovingAverageStrategy(
        name="Nifty50_EMA_Crossover",
        timeframe="5minute",
        fast_period=5,
        slow_period=20,
        is_intraday=True
    )
    strategy.initialize_universe()
    
    # Initialize empty positions for all symbols
    print("\nInitializing empty positions for all symbols...")
    for symbol in strategy.universe:
        strategy.update_position(symbol, {"quantity": 0, "average_price": 0})
    
    # Display strategy parameters
    print("\nStrategy Parameters:")
    print(f"  Name: {strategy.name}")
    print(f"  Universe: {len(strategy.universe)} stocks")
    print(f"  Timeframe: {strategy.timeframe}")
    print(f"  Fast Period: {strategy.fast_period}")
    print(f"  Slow Period: {strategy.slow_period}")
    print(f"  Intraday: {strategy.is_intraday}")
    
    # Get instrument tokens for universe BEFORE using them
    print("\nFetching instrument tokens for all symbols in universe...")
    token_map = historical_data_manager.get_instrument_tokens(strategy.universe)
    print(f"Retrieved {len(token_map)} instrument tokens out of {len(strategy.universe)} symbols")
    
    # Debug: Display the first few tokens that were found
    if len(token_map) > 0:
        print("Sample of instrument tokens:")
        for i, (symbol, token) in enumerate(token_map.items()):
            if i >= 5:  # Only show first 5
                break
            print(f"  {symbol}: {token}")
    
    # Create the token to symbol mapping
    tokens = list(token_map.values())
    symbol_token_map = {v: k for k, v in token_map.items()}
    
    # Make token_symbol_map available globally for the callback
    token_symbol_map = symbol_token_map
    
    # Test 1: Fetch historical data and calculate signals
    print("\nTEST 1: HISTORICAL DATA AND SIGNALS")
    print("-" * 40)
    
    # Fetch historical data for a limited set of symbols (for testing)
    test_symbols = list(token_map.keys())[:3]  # first 3 symbols that we have tokens for
    if not test_symbols:
        logger.warning("No valid symbols with tokens found. Cannot proceed with testing.")
        return False
        
    print(f"Testing with {len(test_symbols)} symbols: {test_symbols}")
    
    # Calculate date range for last 30 days
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"Fetching historical data from {from_date} to {to_date}...")
    
    # Fetch historical data
    historical_data = fetch_historical_data(
        test_symbols, from_date, to_date, strategy.timeframe, historical_data_manager
    )
    
    if not historical_data:
        logger.warning("No historical data retrieved. Cannot continue with signal generation.")
    else:
        # Force a crossover by modifying the last candle to create a signal
        # This is just for testing if signal generation works correctly
        print("\nCreating artificial crossover for testing...")
        for symbol, data in historical_data.items():
            # Pick first symbol to modify
            modify_symbol = symbol
            modify_data = data.copy()
            
            # Calculate indicators
            indicators = strategy.calculate_indicators(modify_data)
            
            # Get the latest prices
            last_close = modify_data['close'].iloc[-1]
            ema_fast_last = indicators['ema_fast'].iloc[-1]
            ema_slow_last = indicators['ema_slow'].iloc[-1]
            
            print(f"Before modification - {modify_symbol}:")
            print(f"  Last Close: {last_close:.2f}")
            print(f"  Fast EMA: {ema_fast_last:.2f}")
            print(f"  Slow EMA: {ema_slow_last:.2f}")
            print(f"  Difference: {(ema_fast_last - ema_slow_last):.2f}")
            
            # Create artificial crossover by modifying the last candle
            if ema_fast_last < ema_slow_last:
                # Create bullish crossover
                print(f"Creating artificial BULLISH crossover...")
                new_close = ema_slow_last * 1.02  # 2% above slow EMA
                modify_data.loc[modify_data.index[-1], 'close'] = new_close
                print(f"  Modified close to: {new_close:.2f} (above slow EMA)")
            else:
                # Create bearish crossover
                print(f"Creating artificial BEARISH crossover...")
                new_close = ema_slow_last * 0.98  # 2% below slow EMA
                modify_data.loc[modify_data.index[-1], 'close'] = new_close
                print(f"  Modified close to: {new_close:.2f} (below slow EMA)")
            
            # Update the data
            historical_data[modify_symbol] = modify_data
            
            # Only modify one symbol for demonstration
            break
        
        # Calculate indicators and generate signals
        print("\nProcessing historical data and generating signals...")
        for symbol, data in historical_data.items():
            print(f"\nAnalyzing {symbol} with {len(data)} candles:")
            
            # Update strategy with historical data
            strategy.update_historical_data(symbol, data)
            
            # Calculate indicators
            indicators = strategy.calculate_indicators(data)
            strategy.update_indicators(symbol, indicators)
            
            # Generate signals (just the last signal)
            signal = strategy.generate_signals(symbol, data)
            
            # Display signal if any
            if signal:
                print(f"\nSignal generated for {symbol}:")
                print(f"  Action: {signal.get('action')}")
                print(f"  Signal Type: {signal.get('signal_type')}")
                print(f"  Price: {signal.get('price')}")
                print(f"  Stop Loss: {signal.get('stop_loss')}")
                
                # Apply risk management
                print("\nApplying risk management...")
                order_params = risk_manager.process_signal(
                    signal, 
                    {"last_price": data['close'].iloc[-1]},
                    {"capital": 100000, "available_margin": 100000, "portfolio_value": 100000, "current_positions": []}
                )
                
                if order_params:
                    print(f"Risk-managed order parameters:")
                    for k, v in order_params.items():
                        print(f"  {k}: {v}")
                else:
                    print("Order rejected by risk management")
            else:
                print(f"\nNo trading signal for {symbol}")
    
    # Test 2: Real-time simulation
    print("\nTEST 2: REAL-TIME SIMULATION")
    print("-" * 40)
    
    # Register callback for tick data
    realtime_data.register_callback('on_tick', on_tick)
    
    # Prepare for real-time simulation
    print("\nPreparing for real-time simulation...")
    
    # Should we run real-time simulation?
    run_realtime = False
    response = input("Would you like to run real-time data simulation? (y/n): ")
    if response.lower() == 'y':
        run_realtime = True
    
    if run_realtime:
        if not tokens:
            logger.warning("No instrument tokens available for WebSocket connection.")
            print("Cannot run real-time simulation without instrument tokens.")
        else:
            print("\nStarting real-time data simulation for 60 seconds...")
            print("This will connect to Zerodha's WebSocket and process live market data.")
            print("Press Ctrl+C to stop earlier.")
            
            try:
                # Subscribe to tokens
                realtime_data.subscribe(tokens, symbol_token_map, token_map)
                
                # Start real-time data
                if not realtime_data.start():
                    logger.error("Failed to start real-time data")
                    return False
                
                # Run for 60 seconds
                time.sleep(60)
                
                # Stop real-time data
                realtime_data.stop()
                
            except KeyboardInterrupt:
                print("\nStopped by user")
                realtime_data.stop()
            except Exception as e:
                logger.error(f"Error in real-time simulation: {e}")
                realtime_data.stop()
    else:
        print("\nSkipping real-time simulation.")
    
    print("\n" + "="*80)
    print("MOVING AVERAGE STRATEGY TEST COMPLETED")
    print("="*80)
    
    return True

if __name__ == "__main__":
    if not test_strategy():
        sys.exit(1)