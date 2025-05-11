"""
Order execution module for the trading system.

This module handles:
- Order placement
- Order tracking
- Order modification
- Order cancellation
"""
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderManager:
    """Manages order placement and tracking."""
    
    def __init__(self, kite):
        """
        Initialize the order manager.
        
        Args:
            kite: An authenticated KiteConnect instance
        """
        self.kite = kite
        self.order_history_cache = {}  # Cache for order history
    
    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        variety: str = "regular",
        validity: str = "DAY",
        disclosed_quantity: Optional[int] = None,
        squareoff: Optional[float] = None,
        stoploss: Optional[float] = None,
        trailing_stoploss: Optional[float] = None,
        tag: Optional[str] = None
    ) -> str:
        """
        Place an order with Zerodha.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE, etc.)
            transaction_type: BUY or SELL
            quantity: Order quantity
            product: Product code (MIS, CNC, NRML, etc.)
            order_type: Order type (MARKET, LIMIT, SL, SL-M)
            price: Order price (required for LIMIT orders)
            trigger_price: Trigger price (required for SL, SL-M orders)
            variety: Order variety (regular, amo, bo, co, iceberg)
            validity: Order validity (DAY, IOC, TTL)
            disclosed_quantity: Disclosed quantity for iceberg orders
            squareoff: Square off price for bracket orders
            stoploss: Stoploss price for bracket orders  
            trailing_stoploss: Trailing stoploss for bracket orders
            tag: Optional tag for the order
            
        Returns:
            Order ID if successful
            
        Raises:
            Exception: If the order placement fails
        """
        try:
            # Prepare the order parameters
            params = {
                "tradingsymbol": symbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": product,
                "order_type": order_type,
                "validity": validity,
            }
            
            # Add optional parameters if provided
            if price is not None:
                params["price"] = price
                
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
                
            if disclosed_quantity is not None:
                params["disclosed_quantity"] = disclosed_quantity
                
            if squareoff is not None:
                params["squareoff"] = squareoff
                
            if stoploss is not None:
                params["stoploss"] = stoploss
                
            if trailing_stoploss is not None:
                params["trailing_stoploss"] = trailing_stoploss
                
            if tag is not None:
                params["tag"] = tag
            
            # Log the order details
            logger.info(f"Placing order: {params}")
            
            # Place the order
            order_id = self.kite.place_order(variety=variety, **params)
            
            logger.info(f"Order placed successfully with order_id: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def modify_order(
        self,
        order_id: str,
        variety: str = "regular",
        parent_order_id: Optional[str] = None,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        order_type: Optional[str] = None,
        trigger_price: Optional[float] = None,
        validity: Optional[str] = None,
        disclosed_quantity: Optional[int] = None
    ) -> str:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            variety: Order variety (regular, amo, bo, co)
            parent_order_id: Parent order ID (required for CO and BO orders)
            quantity: New order quantity
            price: New order price
            order_type: New order type
            trigger_price: New trigger price
            validity: New validity
            disclosed_quantity: New disclosed quantity
            
        Returns:
            Order ID if successful
            
        Raises:
            Exception: If the order modification fails
        """
        try:
            # Prepare the parameters
            params = {}
            
            if parent_order_id is not None:
                params["parent_order_id"] = parent_order_id
                
            if quantity is not None:
                params["quantity"] = quantity
                
            if price is not None:
                params["price"] = price
                
            if order_type is not None:
                params["order_type"] = order_type
                
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
                
            if validity is not None:
                params["validity"] = validity
                
            if disclosed_quantity is not None:
                params["disclosed_quantity"] = disclosed_quantity
            
            logger.info(f"Modifying order {order_id} with parameters: {params}")
            
            # Modify the order
            order_id = self.kite.modify_order(
                variety=variety,
                order_id=order_id,
                **params
            )
            
            logger.info(f"Order modified successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            raise
    
    def cancel_order(
        self,
        order_id: str,
        variety: str = "regular",
        parent_order_id: Optional[str] = None
    ) -> str:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            variety: Order variety (regular, amo, bo, co)
            parent_order_id: Parent order ID (required for CO and BO orders)
            
        Returns:
            Order ID if successful
            
        Raises:
            Exception: If the order cancellation fails
        """
        try:
            params = {}
            if parent_order_id is not None:
                params["parent_order_id"] = parent_order_id
                
            logger.info(f"Cancelling order: {order_id}")
            
            # Cancel the order
            order_id = self.kite.cancel_order(
                variety=variety,
                order_id=order_id,
                **params
            )
            
            logger.info(f"Order cancelled successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders placed for the day.
        
        Returns:
            List of order objects
        """
        try:
            orders = self.kite.orders()
            logger.debug(f"Retrieved {len(orders)} orders")
            return orders
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise
    
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            List of order history objects
        """
        try:
            history = self.kite.order_history(order_id=order_id)
            logger.debug(f"Retrieved history for order {order_id}: {len(history)} records")
            
            # Cache the order history
            self.order_history_cache[order_id] = history
            
            return history
        except Exception as e:
            logger.error(f"Error fetching history for order {order_id}: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> str:
        """
        Get the current status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Order status string
        """
        try:
            # Try to get from cache first
            if order_id in self.order_history_cache:
                return self.order_history_cache[order_id][-1]["status"]
            
            # Otherwise fetch from API
            history = self.get_order_history(order_id)
            if history:
                return history[-1]["status"]
            else:
                return "UNKNOWN"
        except Exception as e:
            logger.error(f"Error fetching status for order {order_id}: {e}")
            raise
    
    def is_order_complete(self, order_id: str) -> bool:
        """
        Check if an order is complete.
        
        Args:
            order_id: ID of the order
            
        Returns:
            True if the order is complete, False otherwise
        """
        status = self.get_order_status(order_id)
        return status == "COMPLETE"
    
    def is_order_open(self, order_id: str) -> bool:
        """
        Check if an order is still open.
        
        Args:
            order_id: ID of the order
            
        Returns:
            True if the order is still open, False otherwise
        """
        status = self.get_order_status(order_id)
        return status in ["OPEN", "PENDING", "TRIGGER PENDING"]
    
    def get_trades(self, order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trades for the day or for a specific order.
        
        Args:
            order_id: Optional order ID to filter trades
            
        Returns:
            List of trade objects
        """
        try:
            if order_id:
                trades = self.kite.order_trades(order_id=order_id)
                logger.debug(f"Retrieved {len(trades)} trades for order {order_id}")
            else:
                trades = self.kite.trades()
                logger.debug(f"Retrieved {len(trades)} trades")
                
            return trades
        except Exception as e:
            order_info = f" for order {order_id}" if order_id else ""
            logger.error(f"Error fetching trades{order_info}: {e}")
            raise
    
    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get current positions.
        
        Returns:
            Dictionary with day and net positions
        """
        try:
            positions = self.kite.positions()
            logger.debug(f"Retrieved positions: {len(positions.get('day', []))} day positions, {len(positions.get('net', []))} net positions")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get holdings.
        
        Returns:
            List of holding objects
        """
        try:
            holdings = self.kite.holdings()
            logger.debug(f"Retrieved {len(holdings)} holdings")
            return holdings
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            raise