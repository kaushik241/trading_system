#!/usr/bin/env python
"""
Setup script for the EMA Intraday Crossover Strategy.

This script creates the necessary directory structure and configuration files
for running the EMA Intraday Crossover Strategy.

Usage:
    python setup_ema_strategy.py
"""
import os
import shutil
import argparse
from datetime import datetime

def setup_ema_strategy():
    """Set up the EMA Intraday Crossover Strategy."""
    print("\n" + "="*80)
    print("EMA INTRADAY CROSSOVER STRATEGY SETUP")
    print("="*80)
    
    # Create the necessary directory structure
    dirs = [
        'data/logs',
        'data/historical',
        'config'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    # Make sure the configuration file exists
    config_file = 'config/ema_strategy_config.py'
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        print("Please make sure you have the configuration file in place.")
    else:
        print(f"Configuration file found: {config_file}")
    
    # Check if the strategy file exists
    strategy_file = 'strategy/ema_crossover_strategy.py'
    if not os.path.exists(strategy_file):
        print(f"Strategy file not found: {strategy_file}")
        print("Please make sure the EMA Intraday Crossover Strategy is properly installed.")
    else:
        print(f"Strategy file found: {strategy_file}")
    
    # Check if the runners exist
    test_runner = 'test_ema_strategy.py'
    live_runner = 'run_ema_strategy_live.py'
    
    if not os.path.exists(test_runner):
        print(f"Test runner not found: {test_runner}")
        print("Please make sure the test runner is properly installed.")
    else:
        print(f"Test runner found: {test_runner}")
    
    if not os.path.exists(live_runner):
        print(f"Live runner not found: {live_runner}")
        print("Please make sure the live runner is properly installed.")
    else:
        print(f"Live runner found: {live_runner}")
    
    # Check if authentication is set up
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"Environment file not found: {env_file}")
        print("Please set up your API credentials by running the authentication script.")
    else:
        print(f"Environment file found: {env_file}")
        # Check if API key and token are set
        with open(env_file, 'r') as f:
            env_content = f.read()
            if 'KITE_API_KEY' in env_content and 'KITE_ACCESS_TOKEN' in env_content:
                print("API credentials found in environment file.")
            else:
                print("API credentials not found in environment file.")
                print("Please set up your API credentials by running the authentication script.")
    
    print("\n" + "="*80)
    print("SETUP COMPLETE")
    print("="*80)
    print("\nTo run the strategy in test mode:")
    print(f"python {test_runner} --symbols RELIANCE,HDFCBANK --short-ema 9 --long-ema 21 --timeframe 5")
    print("\nTo run the strategy in live trading mode (USE WITH CAUTION):")
    print(f"python {live_runner} --symbols RELIANCE,HDFCBANK --short-ema 9 --long-ema 21 --timeframe 5 --confirm")
    print("\nMake sure to authenticate first:")
    print("python corrected_auth_test.py")
    print("="*80)

if __name__ == "__main__":
    setup_ema_strategy()