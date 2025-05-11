#!/usr/bin/env python
"""
Test script for the RiskManager.

This script demonstrates the functionality of the RiskManager class.

Usage:
    python test_risk_manager.py
"""
import os
import sys
import logging
from pprint import pprint
from dotenv import load_dotenv

# Import our modules
from risk.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_risk_manager():
    """Test the functionality of the RiskManager class."""
    print("\n" + "="*80)
    print("RISK MANAGER TEST")
    print("="*80)
    
    # Create a risk manager with default settings
    risk_manager = RiskManager()
    
    print("\nRisk Manager Parameters:")
    print(f"  Max Position Size: {risk_manager.max_position_size * 100:.1f}% of capital")
    print(f"  Max Risk Per Trade: {risk_manager.max_risk_per_trade * 100:.1f}% of capital")
    print(f"  Stop Loss Percentage: {risk_manager.stop_loss_pct * 100:.1f}%")
    print(f"  Position Sizing Method: {risk_manager.position_sizing_method}")
    print(f"  Max Positions: {risk_manager.max_positions}")
    print(f"  Max Sector Exposure: {risk_manager.max_sector_exposure * 100:.1f}%")
    
    # Test 1: Calculate position size
    print("\nTEST 1: POSITION SIZE CALCULATION")
    print("-" * 40)
    
    capital = 100000  # 1 lakh
    price = 2500  # ₹2,500 per share
    
    position_size = risk_manager.calculate_position_size(
        capital=capital,
        current_price=price
    )
    
    print(f"Position Size Calculation:")
    print(f"  Capital: ₹{capital}")
    print(f"  Current Price: ₹{price}")
    print(f"  Calculated Position Size: {position_size} shares")
    print(f"  Position Value: ₹{position_size * price}")
    print(f"  Percentage of Capital: {(position_size * price) / capital * 100:.2f}%")
    
    # Test 2: Calculate stop loss
    print("\nTEST 2: STOP LOSS CALCULATION")
    print("-" * 40)
    
    entry_price = 2500  # ₹2,500 per share
    
    # Percentage-based stop loss
    stop_loss_pct = risk_manager.calculate_stop_loss(
        entry_price=entry_price,
        is_long=True
    )
    
    print(f"Percentage-Based Stop Loss (Long Position):")
    print(f"  Entry Price: ₹{entry_price}")
    print(f"  Stop Loss Price: ₹{stop_loss_pct}")
    print(f"  Stop Loss Distance: ₹{entry_price - stop_loss_pct}")
    print(f"  Stop Loss Percentage: {(entry_price - stop_loss_pct) / entry_price * 100:.2f}%")
    
    # ATR-based stop loss
    atr = 50  # ₹50 ATR
    stop_loss_atr = risk_manager.calculate_stop_loss(
        entry_price=entry_price,
        is_long=True,
        atr=atr
    )
    
    print(f"\nATR-Based Stop Loss (Long Position):")
    print(f"  Entry Price: ₹{entry_price}")
    print(f"  ATR: ₹{atr}")
    print(f"  Stop Loss Price: ₹{stop_loss_atr}")
    print(f"  Stop Loss Distance: ₹{entry_price - stop_loss_atr}")
    print(f"  Stop Loss Percentage: {(entry_price - stop_loss_atr) / entry_price * 100:.2f}%")
    
    # Test 3: Process a signal
    print("\nTEST 3: PROCESS TRADING SIGNAL")
    print("-" * 40)
    
    # Sample trading signal
    signal = {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "direction": "BUY",
        "product": "CNC",
        "strategy_name": "moving_average_crossover",
        "atr": 50.0,
        "sector": "energy",
        "risk_reward_ratio": 2.0
    }
    
    # Sample tick data
    tick_data = {
        "symbol": "RELIANCE",
        "last_price": 2500.0
    }
    
    # Sample portfolio
    portfolio = {
        "capital": 100000,
        "available_margin": 80000,
        "portfolio_value": 120000,
        "current_positions": [
            {
                "symbol": "TCS",
                "quantity": 5,
                "average_price": 3500,
                "value": 17500,
                "sector": "technology"
            }
        ]
    }
    
    # Process the signal
    order_params = risk_manager.process_signal(
        signal=signal,
        tick_data=tick_data,
        portfolio=portfolio
    )
    
    print("Trading Signal:")
    print(f"  Symbol: {signal['symbol']}")
    print(f"  Direction: {signal['direction']}")
    print(f"  Strategy: {signal['strategy_name']}")
    print(f"  Last Price: ₹{tick_data['last_price']}")
    
    print("\nProcessed Order Parameters:")
    if order_params:
        print(f"  Symbol: {order_params['symbol']}")
        print(f"  Exchange: {order_params['exchange']}")
        print(f"  Transaction Type: {order_params['transaction_type']}")
        print(f"  Quantity: {order_params['quantity']}")
        print(f"  Product: {order_params['product']}")
        print(f"  Order Type: {order_params['order_type']}")
        print(f"  Stop Loss: ₹{order_params['stop_loss']}")
        print(f"  Take Profit: ₹{order_params['take_profit']}")
        print(f"  Risk-Reward Ratio: {(order_params['take_profit'] - tick_data['last_price']) / (tick_data['last_price'] - order_params['stop_loss']):.2f}")
    else:
        print("  Trade rejected by risk management rules")
    
    # Test 4: Different Position Sizing Methods
    print("\nTEST 4: DIFFERENT POSITION SIZING METHODS")
    print("-" * 40)
    
    # Fixed size risk manager
    fixed_size_rm = RiskManager(position_sizing_method="fixed_size")
    
    # Volatility-based risk manager
    volatility_rm = RiskManager(position_sizing_method="volatility")
    
    price = 2500
    capital = 100000
    atr = 50
    
    # Calculate position sizes with different methods
    percent_risk_size = risk_manager.calculate_position_size(
        capital=capital,
        current_price=price
    )
    
    fixed_size = fixed_size_rm.calculate_position_size(
        capital=capital,
        current_price=price
    )
    
    volatility_size = volatility_rm.calculate_position_size(
        capital=capital,
        current_price=price,
        atr=atr
    )
    
    print("Position Sizing Methods Comparison:")
    print(f"  Capital: ₹{capital}")
    print(f"  Stock Price: ₹{price}")
    print(f"  ATR: ₹{atr}")
    print(f"  Percent Risk Method: {percent_risk_size} shares (₹{percent_risk_size * price})")
    print(f"  Fixed Size Method: {fixed_size} shares (₹{fixed_size * price})")
    print(f"  Volatility Method: {volatility_size} shares (₹{volatility_size * price})")
    
    print("\n" + "="*80)
    print("RISK MANAGER TEST COMPLETED")
    print("="*80)
    
    return True

if __name__ == "__main__":
    if not test_risk_manager():
        sys.exit(1)