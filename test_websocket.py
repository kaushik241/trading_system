#!/usr/bin/env python
"""
WebSocket Test Script for Real-Time Data

This script tests the WebSocket connection with Zerodha's streaming API
using the RealTimeDataManager class.

It connects to Zerodha's WebSocket server, subscribes to a few symbols,
and displays real-time market data as it arrives.

Usage:
    python test_websocket.py [--symbols RELIANCE TCS INFY] [--duration 60]
"""
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager
from data.realtime_data import RealTimeDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test WebSocket connection for real-time data")
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        nargs="+", 
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        help="List of trading symbols to subscribe to"
    )
    
    parser.add_argument(
        "--duration", 
        type=int, 
        default=60,
        help="Duration to run the WebSocket test (in seconds)"
    )
    
    return parser.parse_args()

def on_tick(symbol, tick):
    """Callback for tick data."""
    # Get formatted timestamp
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    # Format the output based on the available fields in the tick
    ltp = tick.get('last_price', 'N/A')
    volume = tick.get('volume_traded', 'N/A')
    last_quantity = tick.get('last_traded_quantity', 'N/A')
    
    # Print the tick information
    print(f"[{timestamp}] {symbol}: ₹{ltp} | Vol: {volume} | Last Qty: {last_quantity}")

def test_websocket():
    """Test the WebSocket connection."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("WEBSOCKET TEST FOR REAL-TIME DATA")
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
    
    # Initialize HistoricalDataManager to get instrument tokens
    data_manager = HistoricalDataManager(kite)
    
    # Get instrument tokens for the symbols
    try:
        logger.info(f"Getting instrument tokens for {len(args.symbols)} symbols...")
        token_map = data_manager.get_instrument_tokens(args.symbols)
        
        if not token_map:
            logger.error("No instrument tokens found for the specified symbols")
            return False
            
        # Create the reverse mapping (token to symbol)
        symbol_token_map = {v: k for k, v in token_map.items()}
        tokens = list(token_map.values())
        
        # Print tokens
        for symbol, token in token_map.items():
            logger.info(f"{symbol}: {token}")
        
    except Exception as e:
        logger.error(f"Failed to get instrument tokens: {e}")
        return False
    
    # Initialize RealTimeDataManager
    realtime_manager = RealTimeDataManager(api_key, access_token)
    
    # Register callback for tick data
    realtime_manager.register_callback('on_tick', on_tick)
    
    # Define callbacks for connection events
    def on_connect(response):
        logger.info(f"Connected to WebSocket server: {response}")
    
    def on_close(code, reason):
        logger.warning(f"WebSocket connection closed: {code} - {reason}")
    
    def on_error(code, reason):
        logger.error(f"WebSocket error: {code} - {reason}")
    
    # Register callbacks
    realtime_manager.register_callback('on_connect', on_connect)
    realtime_manager.register_callback('on_close', on_close)
    realtime_manager.register_callback('on_error', on_error)
    
    # Subscribe to tokens
    realtime_manager.subscribe(tokens, symbol_token_map, token_map)
    
    # Start the WebSocket connection
    try:
        logger.info("Starting WebSocket connection...")
        if not realtime_manager.start():
            logger.error("Failed to start WebSocket connection")
            return False
        
        # Main loop
        logger.info(f"WebSocket connection started. Running for {args.duration} seconds...")
        logger.info("Press Ctrl+C to stop earlier.")
        
        try:
            start_time = time.time()
            while time.time() - start_time < args.duration:
                time.sleep(0.1)  # Small sleep to reduce CPU usage
        
        except KeyboardInterrupt:
            logger.info("Test stopped by user")
        
        # Stop the WebSocket connection
        logger.info("Stopping WebSocket connection...")
        realtime_manager.stop()
        
        print("\n" + "="*80)
        print("WEBSOCKET TEST COMPLETED")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return False

if __name__ == "__main__":
    if not test_websocket():
        sys.exit(1)