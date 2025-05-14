#!/usr/bin/env python
"""
Configuration file for the EMA Intraday Crossover Strategy.

This file contains the default configuration parameters for the strategy.
You can modify these values directly or override them with command line arguments.
"""

# Trading universe configuration
SYMBOLS = "RELIANCE,HDFCBANK"  # Comma-separated list of symbols

# Strategy parameters
SHORT_EMA_PERIOD = 2  # Period for the shorter EMA
LONG_EMA_PERIOD = 10  # Period for the longer EMA 
TIMEFRAME = 5  # Candle timeframe in minutes

# Risk management parameters
MAX_POSITION_SIZE = 1  # Maximum position size per symbol
STOP_LOSS_PERCENT = 0.02  # 1% stop loss

# Time parameters
MARKET_OPEN_TIME = "09:15:00"  # Market open time
MARKET_CLOSE_TIME = "15:30:00"  # Market close time
SQUARE_OFF_TIME = "15:10:00"  # Time to square off all positions

# Order parameters
USE_LIMIT_ORDERS = True  # Use limit orders at best bid/ask prices
ORDER_UPDATE_INTERVAL = 1  # Interval in seconds to update limit orders

# Logging configuration
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_TO_FILE = True  # Log to file
LOG_TO_CONSOLE = True  # Log to console

# Path configuration
LOG_DIR = "data/logs"  # Directory for log files
DATA_DIR = "data/historical"  # Directory for historical data