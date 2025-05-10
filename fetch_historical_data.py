#!/usr/bin/env python
"""
Script to fetch historical data for backtesting.

This script fetches historical price data for specified symbols
and saves it to CSV files for later use.

Usage:
    python fetch_historical_data.py --symbols RELIANCE TCS INFY --days 365 --interval day --plot
"""
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import our modules
from auth.zerodha_auth import ZerodhaAuth
from data.historical_data import HistoricalDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch and manage historical market data")
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        nargs="+", 
        default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
        help="List of trading symbols"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=365,
        help="Number of days of historical data to fetch"
    )
    
    parser.add_argument(
        "--interval", 
        type=str, 
        default="day",
        choices=["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"],
        help="Candle interval for historical data"
    )
    
    parser.add_argument(
        "--exchange", 
        type=str, 
        default="NSE",
        choices=["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"],
        help="Exchange to fetch data from"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Plot the fetched data"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/historical",
        help="Directory to store historical data"
    )
    
    return parser.parse_args()

def plot_data(data_manager, symbols, data):
    """Plot historical price data."""
    if len(symbols) == 1:
        symbol = symbols[0]
        df = data.get(symbol)
        if df is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['close'], label=f"{symbol} Close Price")
            
            plt.title(f"{symbol} Historical Price Data")
            plt.xlabel("Date")
            plt.ylabel("Price (₹)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        # Plot multiple symbols (normalized to 100)
        plt.figure(figsize=(12, 6))
        
        for symbol, df in data.items():
            if df is not None and not df.empty:
                # Normalize to 100 at the start
                prices = df['close'] / df['close'].iloc[0] * 100
                plt.plot(df.index, prices, label=f"{symbol}")
        
        plt.title("Historical Price Data (Normalized)")
        plt.xlabel("Date")
        plt.ylabel("Price (Normalized to 100)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key, secret, and access token from environment variables
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    
    if not api_key or not api_secret or not access_token:
        logger.error("API key, secret, and access token are required")
        logger.error("Please run the authentication script first")
        sys.exit(1)
    
    # Initialize ZerodhaAuth
    try:
        auth = ZerodhaAuth(api_key, api_secret, access_token)
        
        # Validate connection
        if not auth.validate_connection():
            logger.error("Connection validation failed. Your access token may be expired.")
            logger.error("Please run the authentication script again to get a new access token.")
            sys.exit(1)
            
        logger.info("Authentication successful")
        
        # Get KiteConnect instance
        kite = auth.get_kite()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=args.days)
    
    # Convert to string format required by API
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    # Initialize HistoricalDataManager
    data_manager = HistoricalDataManager(kite, args.data_dir)
    
    # Fetch historical data
    logger.info(f"Fetching {args.days} days of historical data for {len(args.symbols)} symbols...")
    data = data_manager.fetch_multiple_symbols(
        args.symbols, 
        from_date_str, 
        to_date_str, 
        args.interval, 
        args.exchange
    )
    
    # Print summary
    print("\n" + "="*80)
    print("HISTORICAL DATA SUMMARY")
    print("="*80)
    
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} {args.interval}-candles from {df.index.min()} to {df.index.max()}")
    
    # Plot data if requested
    if args.plot:
        plot_data(data_manager, args.symbols, data)
    
    print("\nHistorical data fetched and saved successfully!")

if __name__ == "__main__":
    main()