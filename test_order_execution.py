#!/usr/bin/env python
"""
Test script for OrderManager.

This script tests the OrderManager class with Zerodha's API.
It checks orders, positions, and holdings, but does not place actual orders.

Usage:
    python test_order_execution.py
"""
import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from auth.zerodha_auth import ZerodhaAuth
from execution.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_order_execution():
    """Test the OrderManager with Zerodha's API."""
    print("\n" + "="*80)
    print("ORDER EXECUTION TEST")
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
    
    # Initialize OrderManager
    order_manager = OrderManager(kite)
    
    # Test 1: Get current holdings
    print("\nTEST 1: GET CURRENT HOLDINGS")
    print("-" * 40)
    
    try:
        holdings = order_manager.get_holdings()
        if holdings:
            print(f"You have {len(holdings)} holdings:")
            for holding in holdings:
                print(f"  {holding['tradingsymbol']} ({holding['exchange']}): {holding['quantity']} shares at avg price ₹{holding['average_price']:.2f}")
        else:
            print("You don't have any holdings.")
    except Exception as e:
        logger.error(f"Error getting holdings: {e}")
    
    # Test 2: Get current positions
    print("\nTEST 2: GET CURRENT POSITIONS")
    print("-" * 40)
    
    try:
        positions = order_manager.get_positions()
        day_positions = positions.get('day', [])
        net_positions = positions.get('net', [])
        
        print(f"You have {len(day_positions)} day positions and {len(net_positions)} net positions.")
        
        if net_positions:
            print("\nNet positions:")
            for position in net_positions:
                pnl = position.get('pnl', 0)
                print(f"  {position['tradingsymbol']} ({position['exchange']}): {position['quantity']} shares, P&L: ₹{pnl:.2f}")
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
    
    # Test 3: Get recent orders
    print("\nTEST 3: GET RECENT ORDERS")
    print("-" * 40)
    
    try:
        orders = order_manager.get_orders()
        if orders:
            print(f"You have {len(orders)} recent orders:")
            for order in orders[:5]:  # Display only the first 5 orders
                print(f"  {order['tradingsymbol']} ({order['exchange']}): {order['transaction_type']} {order['quantity']} @ {order.get('price', 'MARKET')}, Status: {order['status']}")
        else:
            print("You don't have any recent orders.")
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
    
    # Test 4: Get recent trades
    print("\nTEST 4: GET RECENT TRADES")
    print("-" * 40)
    
    try:
        trades = order_manager.get_trades()
        if trades:
            print(f"You have {len(trades)} recent trades:")
            for trade in trades[:5]:  # Display only the first 5 trades
                print(f"  {trade['tradingsymbol']} ({trade['exchange']}): {trade['transaction_type']} {trade['quantity']} @ ₹{trade['price']:.2f}")
        else:
            print("You don't have any recent trades.")
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
    
    # Test 5: Display Order Manager capabilities (without placing orders)
    print("\nTEST 5: ORDER MANAGER CAPABILITIES")
    print("-" * 40)
    
    print("The OrderManager can perform the following operations:")
    print("  1. Place orders (market, limit, SL, SL-M)")
    print("  2. Modify existing orders")
    print("  3. Cancel open orders")
    print("  4. Track order status")
    print("  5. Retrieve order history")
    print("  6. Retrieve trades")
    print("  7. Retrieve positions and holdings")
    
    print("\nNOTE: This test script does not place actual orders.")
    print("To place a real order, you can use the OrderManager as follows:")
    print("""
    order_id = order_manager.place_order(
        symbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        product="CNC",
        order_type="MARKET"
    )
    """)
    
    print("\n" + "="*80)
    print("ORDER EXECUTION TEST COMPLETED")
    print("="*80)
    
    return True

if __name__ == "__main__":
    if not test_order_execution():
        sys.exit(1)