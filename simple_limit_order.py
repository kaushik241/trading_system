#!/usr/bin/env python
"""
Simple Limit Order Execution Script

This script:
1. Places a limit order (BUY or SELL) for the specified symbol
2. Intelligently updates the order price to ensure execution (with modification limits)
3. Automatically executes the opposite side after 3 minutes

Usage:
    python simple_limit_order.py --symbol RELIANCE --quantity 1 --price 1425 --side B
    python simple_limit_order.py --symbol RELIANCE --quantity 1 --price 1425 --side S
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
    """Execute a limit order with intelligent price updates and timed exit."""
    
    def __init__(self, args):
        """Initialize with command-line arguments."""
        self.args = args
        self.symbol = args.symbol
        self.quantity = args.quantity
        self.limit_price = args.price
        self.side = args.side.upper()  # 'B' for BUY first, 'S' for SELL first
        
        # Determine entry and exit sides
        self.entry_side = "BUY" if self.side == "B" else "SELL"
        self.exit_side = "SELL" if self.side == "B" else "BUY"
        
        # Components
        self.auth = None
        self.kite = None
        self.order_manager = None
        self.realtime_data = None
        self.historical_data_manager = None
        
        # State tracking
        self.token = None
        self.tick_size = 0.05  # Default tick size
        self.symbol_token_map = {}
        self.token_symbol_map = {}
        self.order_id = None
        self.order_status = None
        self.position_active = False
        self.position_active_since = None
        self.exit_order_id = None
        self.exit_order_placed = False  # Flag to track if exit order has been placed
        self.last_price = None
        self.last_tick_time = datetime.min
        self.trade_complete = False
        
        # Order update tracking
        self.last_order_update_time = datetime.min
        self.order_update_interval = 5  # Update order every 5 seconds if needed
        self.modification_count = 0  # Track how many times we've modified the order
        self.max_modifications = 4   # Maximum allowed modifications (Zerodha limit)
        
        # Locks for thread safety
        self.order_lock = threading.Lock()
        self.exit_timer = None
        
        # Create data directory if it doesn't exist
        os.makedirs('data/logs', exist_ok=True)
    
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
            
            # Get instrument details including tick size
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
        if self.order_id and not self.position_active and self.order_status in ['OPEN', 'PENDING']:
            self.check_order_update(tick)

    def check_websocket_health(self):
        """Check and attempt to reconnect WebSocket if needed."""
        # Only try reconnecting if we haven't received ticks for more than 30 seconds
        current_time = datetime.now()
        if (current_time - self.last_tick_time).total_seconds() <= 30:
            return True
            
        if not self.realtime_data.is_connected:
            logger.warning("WebSocket disconnected - attempting to reconnect...")
            # Stop the current connection
            self.realtime_data.stop()
            time.sleep(2)  # Brief delay before reconnecting
            
            # Create new connection
            self.realtime_data = RealTimeDataManager(
                os.getenv("KITE_API_KEY"), 
                os.getenv("KITE_ACCESS_TOKEN")
            )
            
            # Register callbacks again
            self.realtime_data.register_callback('on_tick', self.on_tick)
            self.realtime_data.register_callback('on_order_update', self.on_order_update)
            self.realtime_data.register_callback('on_connect', self.on_connect)
            
            # Restart the connection
            if self.realtime_data.start():
                logger.info("WebSocket successfully reconnected")
                return True
            else:
                logger.error("Failed to reconnect WebSocket")
                return False
        
        # If no ticks received for more than 30 seconds but connection still active
        if (current_time - self.last_tick_time).total_seconds() > 30:
            logger.warning(f"No ticks received for {(current_time - self.last_tick_time).total_seconds():.1f} seconds - attempting to restart connection")
            # Try to restart the connection
            return self.restart_websocket()
        
        return True
        
    def restart_websocket(self):
        """Restart the WebSocket connection."""
        try:
            # Stop the current connection
            if self.realtime_data:
                self.realtime_data.stop()
                time.sleep(2)  # Wait before reconnecting
            
            # Create a new connection
            self.realtime_data = RealTimeDataManager(
                os.getenv("KITE_API_KEY"), 
                os.getenv("KITE_ACCESS_TOKEN")
            )
            
            # Register callbacks again
            self.realtime_data.register_callback('on_tick', self.on_tick)
            self.realtime_data.register_callback('on_order_update', self.on_order_update)
            self.realtime_data.register_callback('on_connect', self.on_connect)
            
            # Start the connection
            if self.realtime_data.start():
                logger.info("WebSocket connection restarted successfully")
                return True
            else:
                logger.error("Failed to restart WebSocket connection")
                return False
        except Exception as e:
            logger.error(f"Error restarting WebSocket connection: {e}")
            return False

    
    
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
        if not self.tick_size or self.tick_size <= 0:
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
        return rounded_price
    
    def check_order_update(self, tick):
        """Check if we need to update the order price."""
        current_time = datetime.now()
        
        # Only update every order_update_interval seconds
        if (current_time - self.last_order_update_time).total_seconds() < self.order_update_interval:
            return
        
        # Don't update if we've reached the maximum modifications
        if self.modification_count >= self.max_modifications:
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
                    raw_price = self.last_price * 1.0005  # 0.05% above last price
                    new_price = self.round_to_tick_size(raw_price, is_buy=True)
                    
                    # Only update if price difference is significant (0.2% or more)
                    if not self.limit_price or abs(new_price - self.limit_price) / self.limit_price > 0.002:
                        logger.info(f"No market depth available. Using last price +0.05%: {new_price}")
                        self.update_order_price(new_price)
                return
            
            # For a buy order, we want to place it at or slightly above the ask price
            # to increase the chance of execution
            raw_price = ask_price * 1.0002  # 0.02% above ask price
            new_price = self.round_to_tick_size(raw_price, is_buy=True)
            
            # Only update if the price difference is significant (0.2% or more)
            # to avoid hitting Zerodha's order modification limits
            if self.limit_price is None or abs(new_price - self.limit_price) / self.limit_price > 0.002:
                logger.info(f"Updating order price: {self.limit_price} -> {new_price}")
                self.update_order_price(new_price)
    
    def update_order_price(self, new_price):
        """Update the limit order price."""
        try:
            if not self.order_id or self.modification_count >= self.max_modifications:
                return
            
            # Store the new price
            self.limit_price = new_price
            
            # Modify the order
            logger.info(f"Modifying order {self.order_id} with parameters: {{'price': {new_price}}}")
            modified_order_id = self.order_manager.modify_order(
                order_id=self.order_id,
                price=new_price
            )
            
            # Increment modification counter
            self.modification_count += 1
            
            logger.info(f"Order price updated: {new_price}, Order ID: {modified_order_id} " +
                      f"(Modification {self.modification_count}/{self.max_modifications})")
            
        except Exception as e:
            logger.error(f"Error updating order price: {e}")
            
            # If we hit the modification limit error, stop trying to modify
            if "Maximum allowed order modifications exceeded" in str(e):
                logger.warning("Broker limit reached. No further modifications will be attempted.")
                self.modification_count = self.max_modifications  # Prevent further modifications
    
    def on_order_update(self, ws, order_data):
        """Handle order updates from WebSocket."""
        if not order_data:
            return
        
        try:
            order_id = order_data.get('order_id')
            status = order_data.get('status')
            
            # Only process updates for our orders
            if order_id == self.order_id:
                logger.info(f"Order update received: {order_id}, Status: {status}")
                self.order_status = status
                
                # If the order is filled, start the exit timer
                if status == 'COMPLETE' and not self.position_active:
                    self.position_active = True
                    self.position_active_since = datetime.now()
                    logger.info(f"Buy order filled! Starting 3-minute timer for exit...")
                    
                    # Start exit timer
                    self.start_exit_timer()
            
            # Handle exit order updates
            elif order_id == self.exit_order_id:
                logger.info(f"Exit order update: {order_id}, Status: {status}")
                
                if status == 'COMPLETE':
                    logger.info("Exit order complete! Trade cycle finished.")
                    self.trade_complete = True
        
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
    
    def check_order_status(self):
        """Actively check the order status instead of just relying on WebSocket updates."""
        if not self.order_id or self.position_active:
            return
            
        try:
            # Get order history from the API
            order_history = self.order_manager.get_order_history(self.order_id)
            if order_history:
                latest_status = order_history[-1].get('status')
                
                # Update our status if different
                if latest_status != self.order_status:
                    logger.info(f"Order status updated via API check: {self.order_status} -> {latest_status}")
                    self.order_status = latest_status
                    
                    # If the order is complete and position is not active yet, start the exit timer
                    if latest_status == 'COMPLETE' and not self.position_active:
                        self.position_active = True
                        self.position_active_since = datetime.now()
                        logger.info(f"Buy order filled (detected via API)! Starting 3-minute timer for exit...")
                        self.start_exit_timer()
                        
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
    
    def place_entry_order(self):
        """Place a limit order for entry (BUY or SELL based on side)."""
        try:
            if not self.last_price:
                logger.error("Cannot place order without a price reference")
                return False
            
            # Determine if it's a buy or sell for rounding
            is_buy = self.entry_side == "BUY"
            
            # Use last price if no limit price specified
            if self.limit_price is None:
                if is_buy:
                    # For BUY: slightly above current price to increase fill chance
                    raw_price = self.last_price * 1.0005  # 0.05% above
                else:
                    # For SELL: slightly below current price to increase fill chance
                    raw_price = self.last_price * 0.9995  # 0.05% below
                    
                self.limit_price = self.round_to_tick_size(raw_price, is_buy=is_buy)
            else:
                # If limit price was specified, make sure it's properly rounded
                self.limit_price = self.round_to_tick_size(self.limit_price, is_buy=is_buy)
            
            logger.info(f"Placing limit {self.entry_side} order for {self.symbol}: {self.quantity} @ {self.limit_price}")
            
            order_id = self.order_manager.place_order(
                symbol=self.symbol,
                exchange="NSE",
                transaction_type=self.entry_side,
                quantity=self.quantity,
                product="MIS",  # Intraday
                order_type="LIMIT",
                price=self.limit_price
            )
            
            self.order_id = order_id
            self.order_status = "OPEN"
            logger.info(f"{self.entry_side} order placed with ID: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing {self.entry_side} order: {e}")
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
        
        # Log the expected exit time
        exit_time = datetime.now() + timedelta(minutes=3)
        logger.info(f"Exit timer started. Will sell in 3 minutes at {exit_time.strftime('%H:%M:%S')}")
    
    def exit_position(self):
        """Exit the position by placing an opposite order."""
        # Don't exit if we've already placed an exit order
        if self.exit_order_placed:
            logger.info("Exit order already placed - skipping duplicate exit request")
            return
            
        try:
            logger.info("Exit timer triggered - placing exit order")
            
            with self.order_lock:
                # Check if we actually have a position
                if not self.position_active:
                    logger.warning("No active position to exit")
                    return
                
                # Place market order to exit
                logger.info(f"Placing order: {{" + 
                          f"'tradingsymbol': '{self.symbol}', " + 
                          f"'exchange': 'NSE', " + 
                          f"'transaction_type': '{self.exit_side}', " + 
                          f"'quantity': {self.quantity}, " + 
                          f"'product': 'MIS', " + 
                          f"'order_type': 'MARKET', " + 
                          f"'validity': 'DAY'" + 
                          f"}}")
                
                self.exit_order_id = self.order_manager.place_order(
                    symbol=self.symbol,
                    exchange="NSE",
                    transaction_type=self.exit_side,
                    quantity=self.quantity,
                    product="MIS",  # Intraday
                    order_type="MARKET"  # Use market order for exit to ensure execution
                )
                
                # Mark that we've placed an exit order
                self.exit_order_placed = True
                
                logger.info(f"{self.exit_side} order placed with ID: {self.exit_order_id}")
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            # If there's an error, try again with a limit order
            try:
                logger.info("Retrying exit with a limit order")
                if self.last_price:
                    # Set limit price based on exit side
                    is_buy = self.exit_side == "BUY"
                    
                    if is_buy:
                        # For BUY exit: slightly above current price
                        raw_price = self.last_price * 1.0005
                    else:
                        # For SELL exit: slightly below current price
                        raw_price = self.last_price * 0.9995
                        
                    limit_price = self.round_to_tick_size(raw_price, is_buy=is_buy)
                    
                    self.exit_order_id = self.order_manager.place_order(
                        symbol=self.symbol,
                        exchange="NSE",
                        transaction_type=self.exit_side,
                        quantity=self.quantity,
                        product="MIS",
                        order_type="LIMIT",
                        price=limit_price
                    )
                    
                    # Mark that we've placed an exit order
                    self.exit_order_placed = True
                    
                    logger.info(f"{self.exit_side} limit order placed with ID: {self.exit_order_id} at price {limit_price}")
            except Exception as retry_e:
                logger.error(f"Error placing limit exit order: {retry_e}")
                logger.error("CRITICAL: Unable to exit position automatically. Manual intervention required!")
    
    def check_position_exit(self):
        """Check if we need to force an exit for the position."""
        if not self.position_active or self.trade_complete:
            return
            
        current_time = datetime.now()
        
        # If position has been active for more than 4 minutes, force exit as a failsafe
        if self.position_active_since:
            duration = (current_time - self.position_active_since).total_seconds()
            if duration > 240:  # 4 minutes (3 minutes + 1 minute buffer)
                logger.warning(f"Position active for {duration/60:.1f} minutes - forcing exit")
                self.exit_position()
    
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
            
            # Place entry order
            if not self.place_entry_order():
                return False
            
            # Main loop - wait for trade to complete
            logger.info("Entering main loop...")
            last_status_check = datetime.min
            last_debug_log = datetime.min
            last_websocket_check = datetime.min
            
            while not self.trade_complete:
                current_time = datetime.now()
                
                # Check WebSocket health periodically (every 20 seconds)
                if (current_time - last_websocket_check).total_seconds() > 20:
                    self.check_websocket_health()
                    last_websocket_check = current_time
                
                # Check order status periodically (every 10 seconds)
                if (current_time - last_status_check).total_seconds() > 10:
                    self.check_order_status()
                    self.check_position_exit()
                    last_status_check = current_time
                
                # Log status periodically (every 30 seconds)
                if (current_time - last_debug_log).total_seconds() > 30:
                    position_duration = 0
                    if self.position_active and self.position_active_since:
                        position_duration = (current_time - self.position_active_since).total_seconds() / 60
                        
                    logger.info(f"Status: order={self.order_status}, position_active={self.position_active}, " + 
                            f"position_duration={position_duration:.1f}m, modifications={self.modification_count}")
                    last_debug_log = current_time
                
                time.sleep(0.5)
            
            logger.info("Trade complete!")
            return True
            
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user")
            # Try to exit position if interrupted and not already exited
            if self.position_active and not self.trade_complete and not self.exit_order_placed:
                logger.info("Attempting to exit position before shutting down...")
                self.exit_position()
                # Wait briefly for exit order to process
                time.sleep(5)
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
    
    parser.add_argument(
        "--side", 
        type=str,
        default="B",
        choices=["B", "S", "b", "s"],
        help="Order side: B for BUY first (default), S for SELL first"
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
    
    entry_side = "BUY" if args.side.upper() == "B" else "SELL"
    exit_side = "SELL" if args.side.upper() == "B" else "BUY"
    
    print(f"Side: {entry_side} first, then {exit_side} after 3 minutes")
    print(f"Strategy: Place {entry_side} limit order, update price to ensure execution, {exit_side} after 3 minutes")
    print("="*80 + "\n")
    
    # Confirm execution
    confirm = input("Do you want to proceed with this trade? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Execution cancelled by user")
        return 1
    
    # Create data directories if they don't exist
    os.makedirs('data/logs', exist_ok=True)
    
    # Run the executor
    executor = SimpleLimitOrderExecutor(args)
    success = executor.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())