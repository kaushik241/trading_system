def round_to_tick_size(self, price, is_buy=True):
        """
        Round price to the nearest valid tick size.
        For buy orders, round down to avoid rejection for price too high.
        For sell orders, round up to avoid rejection for price too low.
        
        Args:
            price: The price to round
            is_buy: Whether this is a buy order (True) or sell order (False)
            
        Returns:
            Price rounded to the nearest valid tick size
        """
        if not hasattr(self, 'tick_size') or not self.tick_size:
            return price  # Return unmodified if tick size not available
        
        # Calculate how many ticks
        ticks = price / self.tick_size
        
        if is_buy:
            # For buy orders, round down to nearest tick
            rounded_ticks = math.floor(ticks)
        else:
            # For sell orders, round up to nearest tick
            rounded_ticks = math.ceil(ticks)
        
        # Calculate rounded price
        rounded_price = rounded_ticks * self.tick_size
        
        # Ensure we have the right precision (same number of decimal places as tick size)
        tick_decimals = str(self.tick_size)[::-1].find('.')
        if tick_decimals > 0:
            rounded_price = round(rounded_price, tick_decimals)
        
        logger.debug(f"Rounded price from {price} to {rounded_price} (tick size: {self.tick_size})")
        return rounded_price#!/usr/bin/env python
"""
Simple Limit Order Execution Script

This script:
1. Places a limit buy order for the specified symbol
2. Actively updates the order price to ensure execution
3. Automatically sells the position after 3 minutes

Usage:
    python simple_limit_order.py --symbol RELIANCE --quantity 1 --price 1425
"""
import os
import sys
import time
import logging
import argparse
import threading
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(override=True)
# Import trading system components
from auth.zerodha_auth import ZerodhaAuth
from data.realtime_data import RealTimeDataManager
from data.historical_data import HistoricalDataManager
from execution.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/logs/limit_order_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleLimitOrderExecutor:
    """Simple class to execute a limit order with price updates and timed exit."""
    
    def __init__(self, args):
        """Initialize with command-line arguments."""
        self.args = args
        self.symbol = args.symbol
        self.quantity = args.quantity
        self.limit_price = args.price
        
        # Components
        self.auth = None
        self.kite = None
        self.order_manager = None
        self.realtime_data = None
        self.historical_data_manager = None
        
        # State tracking
        self.token = None
        self.symbol_token_map = {}
        self.token_symbol_map = {}
        self.order_id = None
        self.order_status = None
        self.position_active = False
        self.exit_order_id = None
        self.last_price = None
        self.last_tick_time = datetime.min
        self.trade_complete = False
        
        # Order update tracking
        self.last_order_update_time = datetime.min
        self.order_update_interval = 1  # Update order every 1 second if needed
        
        # Locks for thread safety
        self.order_lock = threading.Lock()
        self.exit_timer = None
    
    def authenticate(self):
        """Authenticate with Zerodha's API."""
        load_dotenv()
        
        api_key = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        
        if not api_key or not api_secret or not access_token:
            logger.error("API key, secret, and access token are required")
            logger.error("Please run the authentication script first")
            return False
        
        try:
            self.auth = ZerodhaAuth(api_key, api_secret, access_token)
            
            if not self.auth.validate_connection():
                logger.error("Connection validation failed. Your access token may be expired.")
                return False
                
            logger.info("Authentication successful")
            self.kite = self.auth.get_kite()
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def initialize(self):
        """Initialize components needed for trading."""
        try:
            if not self.kite:
                logger.error("Authentication not completed")
                return False
            
            # Initialize components
            self.order_manager = OrderManager(self.kite)
            self.historical_data_manager = HistoricalDataManager(self.kite)
            self.realtime_data = RealTimeDataManager(
                os.getenv("KITE_API_KEY"), 
                os.getenv("KITE_ACCESS_TOKEN")
            )
            
            # Get all instruments to fetch tick size information
            logger.info("Fetching instrument details including tick size...")
            instruments = self.kite.instruments("NSE")
            
            # Find our symbol and get its token and tick size
            symbol_data = None
            for instrument in instruments:
                if instrument['tradingsymbol'] == self.symbol:
                    symbol_data = instrument
                    break
            
            if not symbol_data:
                logger.error(f"Could not find instrument data for {self.symbol}")
                return False
            
            self.token = symbol_data['instrument_token']
            self.tick_size = symbol_data.get('tick_size', 0.05)  # Default to 0.05 if not found
            
            logger.info(f"Found {self.symbol} with token {self.token} and tick size {self.tick_size}")
            
            # Create token mappings
            self.symbol_token_map = {self.symbol: self.token}
            self.token_symbol_map = {self.token: self.symbol}
            
            # Setup WebSocket callbacks
            self.realtime_data.register_callback('on_tick', self.on_tick)
            self.realtime_data.register_callback('on_order_update', self.on_order_update)
            self.realtime_data.register_callback('on_connect', self.on_connect)
            
            # Start WebSocket connection
            if not self.realtime_data.start():
                logger.error("Failed to start WebSocket connection")
                return False
            
            logger.info("Initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def on_connect(self, response):
        """Callback when WebSocket connects."""
        logger.info(f"WebSocket connected: {response}")
        
        # Subscribe to the symbol's token
        self.realtime_data.subscribe(
            [self.token], 
            self.token_symbol_map,
            self.symbol_token_map
        )
        
        if self.realtime_data.ticker:
            # Set mode to FULL for detailed market data
            self.realtime_data.set_mode(
                self.realtime_data.ticker.MODE_FULL,
                [self.token]
            )
    
    def on_tick(self, symbol, tick):
        """Process incoming tick data."""
        if isinstance(symbol, int):
            symbol = self.token_symbol_map.get(symbol, str(symbol))
        
        if symbol != self.symbol:
            return
        
        # Update last price and tick time
        self.last_tick_time = datetime.now()
        last_price = tick.get('last_price')
        
        if last_price:
            prev_price = self.last_price
            self.last_price = last_price
            
            if prev_price is not None and prev_price != last_price:
                logger.info(f"Price update for {symbol}: {prev_price} -> {last_price}")
        
        # If we have a pending buy order, check if we need to update the price
        if self.order_id and not self.position_active:
            self.check_order_update(tick)
    
    def check_order_update(self, tick):
        """Check if we need to update the order price."""
        current_time = datetime.now()
        
        # Only update every order_update_interval seconds
        if (current_time - self.last_order_update_time).total_seconds() < self.order_update_interval:
            return
        
        self.last_order_update_time = current_time
        
        with self.order_lock:
            # Skip if order is not in OPEN state
            if self.order_status not in ['OPEN', 'PENDING']:
                return
            
            # Get current bid/ask prices
            depth = tick.get('depth', {})
            bid_price = None
            ask_price = None
            
            if 'buy' in depth and depth['buy'] and 'price' in depth['buy'][0]:
                bid_price = depth['buy'][0]['price']
            
            if 'sell' in depth and depth['sell'] and 'price' in depth['sell'][0]:
                ask_price = depth['sell'][0]['price']
            
            if not bid_price or not ask_price:
                # If depth is not available, use last price
                if self.last_price:
                    # Place slightly above last price for a buy order
                    raw_price = round(self.last_price * 1.0005, 2)  # 0.05% above last price
                    new_price = self.round_to_tick_size(raw_price, is_buy=True)
                    logger.info(f"No market depth available. Using last price +0.05%: {new_price}")
                    self.update_order_price(new_price)
                return
            
            # For a buy order, we want to place it at or slightly above the ask price
            # This increases the chance of execution
            raw_price = round(ask_price * 1.0002, 2)  # 0.02% above ask price
            new_price = self.round_to_tick_size(raw_price, is_buy=True)
            
            # Only update if the price is different from current limit price
            if self.limit_price is None or new_price != self.limit_price:
                logger.info(f"Updating order price: {self.limit_price} -> {new_price}")
                self.update_order_price(new_price)
                
                # If new price is lower than current price, update more aggressively
                if self.limit_price and new_price < self.limit_price:
                    self.order_update_interval = 0.5  # Update more frequently when price is dropping
                else:
                    self.order_update_interval = 1.0  # Normal update interval
    
    def update_order_price(self, new_price):
        """Update the limit order price."""
        try:
            if not self.order_id:
                return
            
            # Store the new price
            self.limit_price = new_price
            
            # Modify the order
            logger.info(f"Modifying order {self.order_id} with parameters: {{'price': {new_price}}}")
            modified_order_id = self.order_manager.modify_order(
                order_id=self.order_id,
                price=new_price
            )
            
            logger.info(f"Order price updated: {new_price}, Order ID: {modified_order_id}")
            
        except Exception as e:
            logger.error(f"Error updating order price: {e}")
            # If we get a tick size error, log additional info to help debug
            if "tick size" in str(e).lower():
                logger.error(f"Tick size error detected. Current tick size: {getattr(self, 'tick_size', 'Unknown')}")
                logger.error(f"Attempted price: {new_price}, Raw float: {float(new_price)}")
                # Try again with a more strictly rounded price
                try:
                    strict_price = float(int(new_price / self.tick_size) * self.tick_size)
                    logger.info(f"Retrying with strictly rounded price: {strict_price}")
                    self.limit_price = strict_price
                    modified_order_id = self.order_manager.modify_order(
                        order_id=self.order_id,
                        price=strict_price
                    )
                    logger.info(f"Order price updated with strict rounding: {strict_price}, Order ID: {modified_order_id}")
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")
    
    def on_order_update(self, ws, order_data):
        """Handle order updates from WebSocket."""
        if not order_data:
            return
        
        order_id = order_data.get('order_id')
        status = order_data.get('status')
        
        # Only process updates for our orders
        if order_id == self.order_id:
            logger.info(f"Order update: {order_id}, Status: {status}")
            self.order_status = status
            
            # If the order is filled, start the exit timer
            if status == 'COMPLETE' and not self.position_active:
                self.position_active = True
                logger.info(f"Buy order filled! Starting 3-minute timer for exit...")
                # Start exit timer
                self.start_exit_timer()
        
        # Handle exit order updates
        elif order_id == self.exit_order_id:
            logger.info(f"Exit order update: {order_id}, Status: {status}")
            
            if status == 'COMPLETE':
                logger.info("Exit order complete! Trade cycle finished.")
                self.trade_complete = True
    
    def place_buy_order(self):
        """Place a limit buy order."""
        try:
            if not self.last_price:
                logger.error("Cannot place order without a price reference")
                return False
            
            # Use last price if no limit price specified
            if self.limit_price is None:
                # Calculate initial limit price (slightly above current price to increase chance of fill)
                raw_price = self.last_price * 1.0005  # 0.05% above current price
                self.limit_price = self.round_to_tick_size(raw_price, is_buy=True)
            else:
                # If limit price was specified, make sure it's properly rounded
                self.limit_price = self.round_to_tick_size(self.limit_price, is_buy=True)
            
            logger.info(f"Placing limit buy order for {self.symbol}: {self.quantity} @ {self.limit_price}")
            
            order_id = self.order_manager.place_order(
                symbol=self.symbol,
                exchange="NSE",
                transaction_type="BUY",
                quantity=self.quantity,
                product="MIS",  # Intraday
                order_type="LIMIT",
                price=self.limit_price
            )
            
            self.order_id = order_id
            self.order_status = "OPEN"
            logger.info(f"Buy order placed with ID: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return False
    
    def start_exit_timer(self):
        """Start a timer to exit the position after 3 minutes."""
        # Cancel any existing timer
        if self.exit_timer:
            self.exit_timer.cancel()
        
        # Create a new timer for 3 minutes
        self.exit_timer = threading.Timer(180.0, self.exit_position)  # 180 seconds = 3 minutes
        self.exit_timer.daemon = True
        self.exit_timer.start()
        
        logger.info(f"Exit timer started. Will sell in 3 minutes at {datetime.now() + timedelta(minutes=3)}")
    
    def exit_position(self):
        """Exit the position by placing a market sell order."""
        try:
            logger.info("Exit timer triggered - placing sell order")
            
            with self.order_lock:
                # Check if we actually have a position
                if not self.position_active:
                    logger.warning("No active position to exit")
                    return
                
                # Place market sell order
                self.exit_order_id = self.order_manager.place_order(
                    symbol=self.symbol,
                    exchange="NSE",
                    transaction_type="SELL",
                    quantity=self.quantity,
                    product="MIS",  # Intraday
                    order_type="MARKET"  # Use market order for exit to ensure execution
                )
                
                logger.info(f"Sell order placed with ID: {self.exit_order_id}")
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            # If there's an error, try again with a limit order
            try:
                logger.info("Retrying exit with a limit order")
                if self.last_price:
                    # Use a limit price slightly below the current market price for quick execution
                    raw_price = self.last_price * 0.9995  # 0.05% below current price
                    limit_price = self.round_to_tick_size(raw_price, is_buy=False)  # for sell order
                    
                    self.exit_order_id = self.order_manager.place_order(
                        symbol=self.symbol,
                        exchange="NSE",
                        transaction_type="SELL",
                        quantity=self.quantity,
                        product="MIS",
                        order_type="LIMIT",
                        price=limit_price
                    )
                    logger.info(f"Sell limit order placed with ID: {self.exit_order_id} at price {limit_price}")
            except Exception as retry_e:
                logger.error(f"Error placing limit sell order: {retry_e}")
                logger.error("CRITICAL: Unable to exit position automatically. Manual intervention required!")
    
    def run(self):
        """Run the simple limit order execution process."""
        try:
            # Initialize
            if not self.authenticate():
                return False
            
            if not self.initialize():
                return False
            
            # Wait for initial price data
            logger.info("Waiting for initial price data...")
            wait_start = datetime.now()
            while self.last_price is None:
                time.sleep(0.1)
                if (datetime.now() - wait_start).total_seconds() > 10:
                    logger.error("Timeout waiting for price data")
                    return False
            
            # Place buy order
            if not self.place_buy_order():
                return False
            
            # Main loop - wait for trade to complete
            logger.info("Entering main loop...")
            while not self.trade_complete:
                # Check for WebSocket health
                if (datetime.now() - self.last_tick_time).total_seconds() > 30:
                    logger.warning("No tick data received in 30 seconds")
                
                time.sleep(0.5)
            
            logger.info("Trade complete!")
            return True
            
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            return False
        finally:
            # Cleanup
            if self.exit_timer:
                self.exit_timer.cancel()
            
            if self.realtime_data:
                self.realtime_data.stop()
            
            logger.info("Execution completed")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Limit Order Execution")
    
    parser.add_argument(
        "--symbol", 
        type=str,
        required=True,
        help="Trading symbol (e.g., RELIANCE)"
    )
    
    parser.add_argument(
        "--quantity", 
        type=int,
        required=True,
        help="Order quantity"
    )
    
    parser.add_argument(
        "--price", 
        type=float,
        default=None,
        help="Limit price (optional - will use current market price if not specified)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print execution plan
    print("\n" + "="*80)
    print("SIMPLE LIMIT ORDER EXECUTION")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Quantity: {args.quantity}")
    print(f"Initial Limit Price: {'Market price' if args.price is None else args.price}")
    print(f"Strategy: Buy with limit order, update price to ensure execution, sell after 3 minutes")
    print("="*80 + "\n")
    
    # Confirm execution
    confirm = input("Do you want to proceed with this trade? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Execution cancelled by user")
        return 1
    
    # Run the executor
    executor = SimpleLimitOrderExecutor(args)
    success = executor.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())