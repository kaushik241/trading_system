"""
Intraday EMA Crossover Strategy Implementation.

This strategy:
1. Calculates exponential moving averages (EMA) of specified periods
2. Generates a buy signal when the shorter EMA crosses above the longer EMA
3. Generates a sell signal when the shorter EMA crosses below the longer EMA
4. Uses limit orders at best bid/ask prices
5. Implements a 1% stop loss from entry price
6. Exits all positions at 15:14 (3:14 PM)
7. Trades only intraday positions (MIS product type)
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os

import pandas as pd
import numpy as np

from strategy.base_strategy import BaseStrategy

# Configure file-based logging
def setup_logger(name):
    """Configure a logger with file handler for the strategy."""
    logger = logging.getLogger(name)
    
    # Create logs directory if it doesn't exist
    os.makedirs('data/logs', exist_ok=True)
    
    # Create a unique filename for today
    log_file = f'data/logs/ema_crossover_{datetime.now().strftime("%Y%m%d")}.log'
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

# Create strategy logger
logger = setup_logger('ema_crossover')

class EMAIntraDayStrategy(BaseStrategy):
    """
    Intraday EMA Crossover Strategy implementation.
    
    Uses the crossover of short and long EMA periods to generate trading signals.
    Trades only during market hours and squares off all positions before market close.
    """
    
    def __init__(
        self,
        name: str = 'EMA_Intraday_Crossover',
        universe: List[str] = None,
        timeframe: str = '5minute',
        capital: float = 100000,
        short_ema_period: int = 9,
        long_ema_period: int = 21,
        max_position_size: int = 1,
        stop_loss_pct: float = 0.01,  # 1% stop loss
        exchange: str = 'NSE',
    ):
        """
        Initialize the EMA Intraday strategy.
        
        Args:
            name: Strategy name
            universe: List of symbols to trade
            timeframe: Candle interval for strategy (in minutes)
            capital: Initial capital for trading
            short_ema_period: Period for the shorter EMA calculation
            long_ema_period: Period for the longer EMA calculation
            max_position_size: Maximum number of shares per position
            stop_loss_pct: Stop loss percentage from entry price
            exchange: Exchange to trade
        """
        # Always use intraday for this strategy
        super().__init__(
            name=name,
            universe=universe or [],
            timeframe=timeframe,
            capital=capital,
            is_intraday=True,  # Force intraday
            exchange=exchange
        )
        
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        
        # Ensure the periods are valid
        if short_ema_period >= long_ema_period:
            logger.warning(f"{datetime.now()} - Short EMA period ({short_ema_period}) should be less than long EMA period ({long_ema_period})")
        
        # Additional data structures for tracking
        self.last_crossover = {}  # symbol -> {"type": "bullish"/"bearish", "timestamp": datetime}
        self.entry_prices = {}  # symbol -> entry_price
        self.stop_losses = {}  # symbol -> stop_loss_price
        self.order_update_times = {}  # symbol -> last_order_update_time
        self.active_orders = {}  # symbol -> {"order_id": str, "status": str, "side": str}
        self.stopped_symbols = set()  # symbols that hit stop loss and shouldn't be traded again today
        
        # Configure exit time for intraday
        self.exit_time = datetime.strptime("15:14:00", "%H:%M:%S").time()
        
        logger.info(f"{datetime.now()} - Initialized {self.name} with {short_ema_period}/{long_ema_period} EMAs, {timeframe} timeframe")
        logger.info(f"{datetime.now()} - Trading universe: {universe}")
        logger.info(f"{datetime.now()} - Max position size: {max_position_size}, Stop loss: {stop_loss_pct*100}%")
    
    def calculate_ema(self, data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Calculate exponential moving average.
        
        Args:
            data: DataFrame with OHLCV data
            period: EMA period
            column: Column to use for calculation
            
        Returns:
            Series with EMA values
        """
        if len(data) < period:
            logger.warning(f"{datetime.now()} - Not enough data points ({len(data)}) for EMA calculation. Need at least {period}.")
            # Return NaN series of the same length as data
            return pd.Series(index=data.index, data=np.nan)
            
        return data[column].ewm(span=period, adjust=False).mean()
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate strategy-specific indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate EMAs
        indicators['ema_short'] = self.calculate_ema(data, self.short_ema_period)
        indicators['ema_long'] = self.calculate_ema(data, self.long_ema_period)
        
        # Create comparison series 
        indicators['crossover'] = pd.Series(
            index=data.index,
            data=np.where(
                indicators['ema_short'] > indicators['ema_long'], 1,
                np.where(indicators['ema_short'] < indicators['ema_long'], -1, 0)
            )
        )
        
        # Calculate crossover signals (1 for bullish, -1 for bearish, 0 for no crossover)
        indicators['crossover_signal'] = indicators['crossover'].diff().fillna(0)
        
        logger.debug(f"{datetime.now()} - Calculated indicators for {len(data)} candles")
        return indicators
    
    def detect_crossover(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[str]:
        """
        Detect if a crossover occurred in the latest candle.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            indicators: Dictionary of indicator values
            
        Returns:
            'bullish' for bullish crossover, 'bearish' for bearish crossover, None otherwise
        """
        # Ensure we have enough data
        if len(data) < 2:
            logger.warning(f"{datetime.now()} - {symbol}: Not enough data to detect crossover")
            return None
            
        # Get the crossover signal values
        crossover_signal = indicators['crossover_signal']
        
        # Check for crossover in the latest candle
        if crossover_signal.iloc[-1] > 0:  # Short EMA crossed above long EMA
            logger.info(f"{datetime.now()} - {symbol}: BULLISH CROSSOVER - Short EMA crossed above Long EMA")
            return 'bullish'
        elif crossover_signal.iloc[-1] < 0:  # Short EMA crossed below long EMA
            logger.info(f"{datetime.now()} - {symbol}: BEARISH CROSSOVER - Short EMA crossed below Long EMA")
            return 'bearish'
        else:
            logger.debug(f"{datetime.now()} - {symbol}: No crossover detected in the latest candle")
            return None
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss is hit for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if stop loss is hit, False otherwise
        """
        position = self.get_position(symbol)
        if not position or position.get('quantity', 0) == 0:
            return False
            
        entry_price = self.entry_prices.get(symbol)
        if not entry_price:
            logger.warning(f"{datetime.now()} - {symbol}: No entry price found, cannot check stop loss")
            return False
            
        # Check if stop loss is hit based on position direction
        if position.get('quantity', 0) > 0:  # Long position
            stop_price = entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_price:
                logger.info(f"{datetime.now()} - {symbol}: STOP LOSS HIT for LONG position. Entry: {entry_price}, Current: {current_price}, Stop: {stop_price}")
                return True
        elif position.get('quantity', 0) < 0:  # Short position
            stop_price = entry_price * (1 + self.stop_loss_pct)
            if current_price >= stop_price:
                logger.info(f"{datetime.now()} - {symbol}: STOP LOSS HIT for SHORT position. Entry: {entry_price}, Current: {current_price}, Stop: {stop_price}")
                return True
                
        return False
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on the data.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary with entry/exit information
        """
        # Skip stopped symbols (hit stop loss today)
        if symbol in self.stopped_symbols:
            logger.info(f"{datetime.now()} - {symbol}: Symbol hit stop loss today, skipping")
            return {}
            
        # Minimum required data
        if len(data) < self.long_ema_period:
            logger.warning(f"{datetime.now()} - {symbol}: Insufficient data for signal generation. Have {len(data)}, need {self.long_ema_period}")
            return {}
            
        # Get current position
        position = self.get_position(symbol)
        position_qty = position.get('quantity', 0) if position else 0
        
        # Get indicators or calculate if not available
        indicators = self.get_indicators(symbol)
        if not indicators:
            indicators = self.calculate_indicators(data)
            self.update_indicators(symbol, indicators)
        
        # Detect crossover
        crossover = self.detect_crossover(symbol, data, indicators)
        
        # Check time for auto square-off
        current_time = datetime.now().time()
        if self.should_exit_intraday(current_time) and position_qty != 0:
            logger.info(f"{datetime.now()} - {symbol}: Time-based exit at {current_time}")
            
            # Generate square-off signal
            return {
                "symbol": symbol,
                "exchange": self.exchange,
                "action": "SELL" if position_qty > 0 else "BUY",
                "quantity": abs(position_qty),
                "product": "MIS",
                "order_type": "LIMIT",
                "signal_type": "time_exit",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reason": "intraday_exit_time"
            }
        
        # Check stop loss
        if position_qty != 0 and 'close' in data.columns:
            current_price = data['close'].iloc[-1]
            if self.check_stop_loss(symbol, current_price):
                logger.info(f"{datetime.now()} - {symbol}: Stop loss triggered at {current_price}")
                
                # Add to stopped symbols list
                self.stopped_symbols.add(symbol)
                
                # Generate square-off signal
                return {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL" if position_qty > 0 else "BUY",
                    "quantity": abs(position_qty),
                    "product": "MIS",
                    "order_type": "LIMIT",
                    "signal_type": "stop_loss",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "stop_loss_hit"
                }
        
        # Process crossover signals if found
        if crossover:
            # Update last crossover information
            self.last_crossover[symbol] = {
                "type": crossover,
                "timestamp": data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()
            }
            
            last_close = data['close'].iloc[-1]
            
            # Generate signals based on crossover and current position
            if crossover == 'bullish' and position_qty <= 0:
                # If we have a short position, square it off first
                if position_qty < 0:
                    logger.info(f"{datetime.now()} - {symbol}: Squaring off SHORT position before entering LONG")
                    return {
                        "symbol": symbol,
                        "exchange": self.exchange,
                        "action": "BUY",  # Buy to cover short
                        "quantity": abs(position_qty),
                        "product": "MIS",
                        "order_type": "LIMIT",
                        "signal_type": "crossover_exit_short",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "price": last_close
                    }
                
                # Generate buy signal for long entry
                logger.info(f"{datetime.now()} - {symbol}: Generating BUY signal at {last_close}")
                
                # Calculate stop loss price
                stop_loss = last_close * (1 - self.stop_loss_pct)
                
                return {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "BUY",
                    "quantity": self.max_position_size,
                    "product": "MIS",
                    "order_type": "LIMIT",
                    "signal_type": "ema_crossover_bullish",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": last_close,
                    "stop_loss": stop_loss
                }
                
            elif crossover == 'bearish' and position_qty >= 0:
                # If we have a long position, square it off first
                if position_qty > 0:
                    logger.info(f"{datetime.now()} - {symbol}: Squaring off LONG position before entering SHORT")
                    return {
                        "symbol": symbol,
                        "exchange": self.exchange,
                        "action": "SELL",  # Sell to exit long
                        "quantity": position_qty,
                        "product": "MIS",
                        "order_type": "LIMIT",
                        "signal_type": "crossover_exit_long",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "price": last_close
                    }
                
                # Generate sell signal for short entry
                logger.info(f"{datetime.now()} - {symbol}: Generating SELL signal at {last_close}")
                
                # Calculate stop loss price
                stop_loss = last_close * (1 + self.stop_loss_pct)
                
                return {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL",
                    "quantity": self.max_position_size,
                    "product": "MIS",
                    "order_type": "LIMIT",
                    "signal_type": "ema_crossover_bearish",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": last_close,
                    "stop_loss": stop_loss
                }
        
        return {}
    
    def process_tick(self, symbol: str, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a real-time tick and generate trading signal if needed.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
            
        Returns:
            Signal dictionary if a signal is generated, None otherwise
        """
        # Skip stopped symbols
        if symbol in self.stopped_symbols:
            return None
            
        # Check if there are pending orders - avoid generating conflicting signals
        if self.has_pending_orders(symbol):
            order_id = self.active_orders[symbol].get('order_id')
            logger.debug(f"{datetime.now()} - {symbol}: Pending order {order_id}, skipping signal generation")
            
            # Still check if we need to update order price
            current_time = datetime.now()
            last_update = self.order_update_times.get(symbol, datetime.min)
            
            # Update limit orders every second if they're still pending
            if (current_time - last_update).total_seconds() > 1:
                self.order_update_times[symbol] = current_time
                
                # Get current bid/ask
                bid_price = tick.get('depth', {}).get('buy', [{}])[0].get('price')
                ask_price = tick.get('depth', {}).get('sell', [{}])[0].get('price')
                
                if not bid_price or not ask_price:
                    logger.warning(f"{datetime.now()} - {symbol}: Missing depth information in tick")
                    return None
                    
                # Update price based on order side
                side = self.active_orders[symbol].get('side')
                order_id = self.active_orders[symbol].get('order_id')
                
                if side == 'BUY':
                    new_price = ask_price  # Use ask price for buy
                    logger.info(f"{datetime.now()} - {symbol}: Updating BUY order price to {new_price}")
                    
                    return {
                        "symbol": symbol,
                        "exchange": self.exchange,
                        "action": "UPDATE",
                        "order_id": order_id,
                        "order_type": "LIMIT",
                        "price": new_price,
                        "signal_type": "order_update"
                    }
                elif side == 'SELL':
                    new_price = bid_price  # Use bid price for sell
                    logger.info(f"{datetime.now()} - {symbol}: Updating SELL order price to {new_price}")
                    
                    return {
                        "symbol": symbol,
                        "exchange": self.exchange,
                        "action": "UPDATE",
                        "order_id": order_id,
                        "order_type": "LIMIT",
                        "price": new_price,
                        "signal_type": "order_update"
                    }
            
            return None
            
        # Get historical data
        data = self.get_historical_data(symbol)
        if data is None or len(data) < self.long_ema_period:
            logger.warning(f"{datetime.now()} - {symbol}: Insufficient historical data to process tick")
            return None
        
        # Check for stop loss
        current_price = tick.get('last_price')
        if current_price and self.check_stop_loss(symbol, current_price):
            # Get current position
            position = self.get_position(symbol)
            position_qty = position.get('quantity', 0) if position else 0
            
            if position_qty != 0:
                # Add to stopped symbols list
                self.stopped_symbols.add(symbol)
                
                logger.info(f"{datetime.now()} - {symbol}: Stop loss triggered at {current_price}")
                
                # Generate square-off signal
                return {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL" if position_qty > 0 else "BUY",
                    "quantity": abs(position_qty),
                    "product": "MIS",
                    "order_type": "LIMIT",
                    "signal_type": "stop_loss",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "stop_loss_hit"
                }
        
        # Check time for auto square-off
        current_time = datetime.now().time()
        if self.should_exit_intraday(current_time):
            # Get current position
            position = self.get_position(symbol)
            position_qty = position.get('quantity', 0) if position else 0
            
            if position_qty != 0:
                logger.info(f"{datetime.now()} - {symbol}: Time-based exit at {current_time}")
                
                # Generate square-off signal
                return {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL" if position_qty > 0 else "BUY",
                    "quantity": abs(position_qty),
                    "product": "MIS",
                    "order_type": "LIMIT",
                    "signal_type": "time_exit",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "intraday_exit_time"
                }
        
        # Check if tick contains a new candle closure
        if 'bar_close' in tick and tick['bar_close']:
            # Process bar close
            new_bar = {
                "timestamp": tick.get("timestamp", datetime.now()),
                "open": tick.get("open", data['close'].iloc[-1]),
                "high": tick.get("high", data['close'].iloc[-1]),
                "low": tick.get("low", data['close'].iloc[-1]),
                "close": tick.get("close", tick.get("last_price", data['close'].iloc[-1])),
                "volume": tick.get("volume", 0)
            }
            
            # Append new bar to historical data
            if 'timestamp' in new_bar and isinstance(new_bar['timestamp'], datetime):
                new_row = pd.DataFrame([new_bar], index=[new_bar['timestamp']])
            else:
                new_row = pd.DataFrame([new_bar], index=[datetime.now()])
            
            data = pd.concat([data, new_row])
            self.update_historical_data(symbol, data)
            
            logger.info(f"{datetime.now()} - {symbol}: New candle processed, recalculating indicators")
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            self.update_indicators(symbol, indicators)
            
            # Generate signals
            return self.generate_signals(symbol, data)
        
        return None
    
    def on_order_update(self, symbol: str, order_update: Dict[str, Any]) -> None:
        """
        Handle order updates for the strategy.
        
        Args:
            symbol: Trading symbol
            order_update: Order update information
        """
        if not order_update:
            return
            
        order_id = order_update.get('order_id')
        status = order_update.get('status')
        
        logger.info(f"{datetime.now()} - {symbol}: Order update - ID: {order_id}, Status: {status}")
        
        # Update active orders tracking
        if symbol in self.active_orders and self.active_orders[symbol].get('order_id') == order_id:
            self.active_orders[symbol]['status'] = status
            
            # If order is filled, update position and entry price
            if status == 'COMPLETE':
                logger.info(f"{datetime.now()} - {symbol}: Order {order_id} completed")
                
                # Update entry price for stop loss calculation
                filled_price = order_update.get('average_price')
                if filled_price:
                    self.entry_prices[symbol] = filled_price
                    
                    # Calculate and store stop loss
                    side = self.active_orders[symbol].get('side')
                    if side == 'BUY':
                        stop_loss = filled_price * (1 - self.stop_loss_pct)
                        logger.info(f"{datetime.now()} - {symbol}: Long position entered at {filled_price}, stop loss set at {stop_loss}")
                    elif side == 'SELL':
                        stop_loss = filled_price * (1 + self.stop_loss_pct)
                        logger.info(f"{datetime.now()} - {symbol}: Short position entered at {filled_price}, stop loss set at {stop_loss}")
                        
                    self.stop_losses[symbol] = stop_loss
                
                # Clear active order
                if status in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                    del self.active_orders[symbol]
        
    def register_order(self, symbol: str, order_id: str, side: str) -> None:
        """
        Register a new order for tracking.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            side: Order side (BUY/SELL)
        """
        logger.info(f"{datetime.now()} - {symbol}: Registering new {side} order: {order_id}")
        
        self.active_orders[symbol] = {
            'order_id': order_id,
            'status': 'OPEN',
            'side': side,
            'timestamp': datetime.now()
        }
        
        # Initialize order update time
        self.order_update_times[symbol] = datetime.now()
    
    def has_pending_orders(self, symbol: str) -> bool:
        """
        Check if a symbol has pending orders.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if the symbol has pending orders, False otherwise
        """
        if symbol not in self.active_orders:
            return False
            
        order_status = self.active_orders[symbol].get('status')
        return order_status in ['OPEN', 'PENDING', 'TRIGGER PENDING']
    
    def get_market_price_for_order(self, symbol: str, tick: Dict[str, Any], is_buy: bool) -> Optional[float]:
        """
        Get the appropriate market price for a limit order.
        
        Args:
            symbol: Trading symbol
            tick: Tick data
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Price to use for the limit order, or None if price not available
        """
        try:
            depth = tick.get('depth', {})
            
            if is_buy:
                # For buy orders, use the ask price (best selling price)
                asks = depth.get('sell', [])
                if asks and 'price' in asks[0]:
                    price = asks[0]['price']
                    logger.info(f"{datetime.now()} - {symbol}: Using ask price for BUY: {price}")
                    return price
            else:
                # For sell orders, use the bid price (best buying price)
                bids = depth.get('buy', [])
                if bids and 'price' in bids[0]:
                    price = bids[0]['price']
                    logger.info(f"{datetime.now()} - {symbol}: Using bid price for SELL: {price}")
                    return price
            
            # If depth is not available, use last price
            price = tick.get('last_price')
            if price:
                logger.warning(f"{datetime.now()} - {symbol}: Depth not available, using last price: {price}")
                return price
            else:
                logger.error(f"{datetime.now()} - {symbol}: No valid price found in tick data")
                return None
                
        except Exception as e:
            logger.error(f"{datetime.now()} - {symbol}: Error getting market price: {e}")
            return tick.get('last_price')  # Fallback to last price
    
    def prepare_order_parameters(self, signal: Dict[str, Any], tick: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Prepare order parameters from a signal.
        
        Args:
            signal: Signal dictionary
            tick: Current market tick data (optional)
            
        Returns:
            Dictionary with order parameters or None if parameters cannot be prepared
        """
        try:
            symbol = signal.get('symbol')
            action = signal.get('action')
            
            if not symbol or not action:
                logger.error(f"{datetime.now()} - Missing required fields in signal: {signal}")
                return None
            
            # Basic order parameters
            order_params = {
                "symbol": symbol,
                "exchange": signal.get("exchange", self.exchange),
                "transaction_type": action,
                "quantity": signal.get("quantity", self.max_position_size),
                "product": "MIS",  # Always MIS for intraday
                "order_type": signal.get("order_type", "LIMIT"),
                "tag": f"{self.name}_{signal.get('signal_type', 'signal')}"
            }
            
            # If this is an update to an existing order
            if action == 'UPDATE':
                if 'order_id' not in signal:
                    logger.error(f"{datetime.now()} - {symbol}: Missing order_id for UPDATE signal")
                    return None
                    
                order_params['order_id'] = signal.get('order_id')
                
            # Set price based on tick data for LIMIT orders
            if order_params['order_type'] == 'LIMIT':
                if 'price' in signal and signal['price']:
                    order_params['price'] = signal['price']
                elif tick:
                    is_buy = action == 'BUY'
                    price = self.get_market_price_for_order(symbol, tick, is_buy)
                    if price:
                        order_params['price'] = price
                    else:
                        logger.error(f"{datetime.now()} - {symbol}: Unable to determine price for order")
                        return None
                else:
                    logger.error(f"{datetime.now()} - {symbol}: No price in signal and no tick data for LIMIT order")
                    return None
                
            # Remove None values
            order_params = {k: v for k, v in order_params.items() if v is not None}
            
            logger.info(f"{datetime.now()} - {symbol}: Prepared order parameters: {order_params}")
            return order_params
            
        except Exception as e:
            logger.error(f"{datetime.now()} - Error preparing order parameters: {e}")
            logger.error(f"{datetime.now()} - Signal: {signal}")
            return None