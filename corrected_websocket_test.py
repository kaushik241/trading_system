#!/usr/bin/env python
"""
Zerodha WebSocket Connection Test Script (Using Trading System Classes)

This script tests the WebSocket connection with Zerodha's streaming API
using our custom RealTimeDataManager class.
"""
import os
import sys
import time
import logging
from dotenv import load_dotenv

# Import our trading system classes
# Note: Adjust the import paths based on your actual project structure
from auth.zerodha_auth import ZerodhaAuth
from data.realtime_data import RealTimeDataManager
from data.historical_data import HistoricalDataManager
load_dotenv(override=True)
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_websocket():
    """Test the WebSocket connection using our trading system classes."""
    print("\n" + "="*80)
    print("ZERODHA WEBSOCKET TEST USING TRADING SYSTEM CLASSES")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key and access token are available
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    
    if not api_key or not api_secret or not access_token:
        logger.error("API key, secret, and access token are required")
        logger.error("Run the authentication test script first to generate them")
        return False
    
    # Initialize ZerodhaAuth
    try:
        auth = ZerodhaAuth(api_key, api_secret, access_token)
        if not auth.validate_connection():
            logger.error("Connection validation failed. Your access token may be expired.")
            logger.error("Run the authentication test script again to get a new access token.")
            return False
        logger.info("ZerodhaAuth initialized and connection validated")
    except Exception as e:
        logger.error(f"Failed to initialize ZerodhaAuth: {e}")
        return False
    
    # Get KiteConnect instance
    kite = auth.get_kite()
    
    # Initialize HistoricalDataManager to get instrument tokens
    try:
        print("\nFetching instrument tokens...")
        historical_data_manager = HistoricalDataManager(kite)
        
        # Define symbols to track
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        
        # Get instrument tokens
        token_map = historical_data_manager.get_instrument_tokens(symbols)
        symbol_token_map = {v: k for k, v in token_map.items()}
        
        tokens = list(token_map.values())
        
        if not tokens:
            logger.error("No instrument tokens found for the specified symbols")
            return False
            
        # Print tokens
        for symbol, token in token_map.items():
            print(f"  {symbol}: {token}")
            
    except Exception as e:
        logger.error(f"Failed to fetch instruments: {e}")
        return False
    
    # Dictionary to store received ticks
    received_ticks = {}
    
    # Custom callback for ticks
    def custom_tick_callback(symbol, tick):
        """Custom callback for tick data."""
        received_ticks[symbol] = tick
        print(f"Tick: {symbol} - LTP: ₹{tick['last_price']:.2f}, Volume: {tick.get('volume_traded', 'N/A')}")
    
    # Initialize RealTimeDataManager
    try:
        print("\nInitializing WebSocket connection...")
        realtime_data = RealTimeDataManager(api_key, access_token)
        
        # Register custom callbacks
        realtime_data.register_callback('on_tick', custom_tick_callback)
        
        # Subscribe to tokens
        realtime_data.subscribe(tokens, symbol_token_map, token_map)
        
        # Start the WebSocket connection
        realtime_data.start()
        
        print("\nWebSocket connected! Receiving market data for 60 seconds...")
        print("Press Ctrl+C to stop earlier")
        
        # Wait for 60 seconds to receive ticks
        try:
            start_time = time.time()
            while time.time() - start_time < 60:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        
        # Stop the WebSocket connection
        realtime_data.stop()
        
        # Print summary
        print("\n" + "="*80)
        print("WEBSOCKET TEST SUMMARY")
        print("="*80)
        print(f"Received market data for {len(received_ticks)} symbols:")
        
        for symbol, tick in received_ticks.items():
            print(f"  {symbol}: ₹{tick['last_price']:.2f}")
        
        print("\nRealTimeDataManager is working correctly!")
        
        return True
        
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return False

if __name__ == "__main__":
    if not test_websocket():
        sys.exit(1)