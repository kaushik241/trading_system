"""
Tests for the order execution module of the trading system.
"""
import unittest
from unittest.mock import MagicMock, patch

from execution.order_manager import OrderManager

class TestOrderManager(unittest.TestCase):
    """Tests for the OrderManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock KiteConnect instance
        self.mock_kite = MagicMock()
        
        # Create an instance of OrderManager with the mock
        self.order_manager = OrderManager(self.mock_kite)
        
        # Sample order data for testing
        self.sample_order = {
            "order_id": "test_order_123",
            "status": "COMPLETE",
            "tradingsymbol": "RELIANCE",
            "exchange": "NSE",
            "transaction_type": "BUY",
            "quantity": 1,
            "product": "CNC",
            "order_type": "MARKET",
            "price": 2500,
            "trigger_price": 0,
            "average_price": 2500,
            "filled_quantity": 1,
            "pending_quantity": 0,
            "cancelled_quantity": 0,
            "disclosed_quantity": 0,
            "validity": "DAY",
            "variety": "regular",
            "placed_by": "test_user",
        }
        
        # Sample trades data
        self.sample_trades = [
            {
                "trade_id": "trade_123",
                "order_id": "test_order_123",
                "tradingsymbol": "RELIANCE",
                "exchange": "NSE",
                "transaction_type": "BUY",
                "quantity": 1,
                "price": 2500,
                "product": "CNC",
                "fill_timestamp": "2023-01-01 09:15:00",
            }
        ]
        
        # Sample positions data
        self.sample_positions = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "product": "CNC",
                    "quantity": 1,
                    "average_price": 2500,
                    "last_price": 2550,
                    "unrealized": 50,
                    "realized": 0,
                    "pnl": 50,
                }
            ],
            "net": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "product": "CNC",
                    "quantity": 1,
                    "average_price": 2500,
                    "last_price": 2550,
                    "unrealized": 50,
                    "realized": 0,
                    "pnl": 50,
                }
            ]
        }
        
        # Sample holdings data
        self.sample_holdings = [
            {
                "tradingsymbol": "RELIANCE",
                "exchange": "NSE",
                "quantity": 1,
                "average_price": 2500,
                "last_price": 2550,
                "pnl": 50,
            }
        ]
    
    def test_place_order(self):
        """Test placing an order."""
        # Set up mock return value for kite.place_order
        self.mock_kite.place_order.return_value = "test_order_123"
        
        # Call the method
        order_id = self.order_manager.place_order(
            symbol="RELIANCE",
            exchange="NSE",
            transaction_type="BUY",
            quantity=1,
            product="CNC",
            order_type="MARKET"
        )
        
        # Check if the mock was called correctly
        self.mock_kite.place_order.assert_called_once()
        
        # Check the returned order_id
        self.assertEqual(order_id, "test_order_123")
    
    def test_place_order_with_additional_params(self):
        """Test placing an order with additional parameters."""
        # Set up mock return value for kite.place_order
        self.mock_kite.place_order.return_value = "test_order_456"
        
        # Call the method with additional parameters
        order_id = self.order_manager.place_order(
            symbol="INFY",
            exchange="NSE",
            transaction_type="BUY",
            quantity=2,
            product="MIS",
            order_type="LIMIT",
            price=1500,
            trigger_price=0,
            variety="regular",
            validity="DAY",
            disclosed_quantity=1,
            tag="test_tag"
        )
        
        # Check if the mock was called correctly
        self.mock_kite.place_order.assert_called_once()
        
        # Get the arguments passed to place_order
        args, kwargs = self.mock_kite.place_order.call_args
        
        # Check that the additional parameters were passed correctly
        self.assertEqual(kwargs["tradingsymbol"], "INFY")
        self.assertEqual(kwargs["exchange"], "NSE")
        self.assertEqual(kwargs["transaction_type"], "BUY")
        self.assertEqual(kwargs["quantity"], 2)
        self.assertEqual(kwargs["product"], "MIS")
        self.assertEqual(kwargs["order_type"], "LIMIT")
        self.assertEqual(kwargs["price"], 1500)
        self.assertEqual(kwargs["validity"], "DAY")
        self.assertEqual(kwargs["disclosed_quantity"], 1)
        self.assertEqual(kwargs["tag"], "test_tag")
        
        # Check the returned order_id
        self.assertEqual(order_id, "test_order_456")
    
    def test_modify_order(self):
        """Test modifying an order."""
        # Set up mock return value for kite.modify_order
        self.mock_kite.modify_order.return_value = "test_order_123"
        
        # Call the method
        order_id = self.order_manager.modify_order(
            order_id="test_order_123",
            quantity=2,
            price=2600
        )
        
        # Check if the mock was called correctly
        self.mock_kite.modify_order.assert_called_once()
        
        # Get the arguments passed to modify_order
        args, kwargs = self.mock_kite.modify_order.call_args
        
        # Check that the parameters were passed correctly
        self.assertEqual(kwargs["order_id"], "test_order_123")
        self.assertEqual(kwargs["quantity"], 2)
        self.assertEqual(kwargs["price"], 2600)
        
        # Check the returned order_id
        self.assertEqual(order_id, "test_order_123")
    
    def test_cancel_order(self):
        """Test cancelling an order."""
        # Set up mock return value for kite.cancel_order
        self.mock_kite.cancel_order.return_value = "test_order_123"
        
        # Call the method
        order_id = self.order_manager.cancel_order(
            order_id="test_order_123",
            variety="regular"
        )
        
        # Check if the mock was called correctly
        self.mock_kite.cancel_order.assert_called_once()
        
        # Get the arguments passed to cancel_order
        args, kwargs = self.mock_kite.cancel_order.call_args
        
        # Check that the parameters were passed correctly
        self.assertEqual(kwargs["order_id"], "test_order_123")
        self.assertEqual(kwargs["variety"], "regular")
        
        # Check the returned order_id
        self.assertEqual(order_id, "test_order_123")
    
    def test_get_orders(self):
        """Test getting all orders."""
        # Set up mock return value for kite.orders
        self.mock_kite.orders.return_value = [self.sample_order]
        
        # Call the method
        orders = self.order_manager.get_orders()
        
        # Check if the mock was called correctly
        self.mock_kite.orders.assert_called_once()
        
        # Check the result
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0], self.sample_order)
    
    def test_get_order_history(self):
        """Test getting the history of an order."""
        # Set up mock return value for kite.order_history
        self.mock_kite.order_history.return_value = [self.sample_order]
        
        # Call the method
        history = self.order_manager.get_order_history(order_id="test_order_123")
        
        # Check if the mock was called correctly
        self.mock_kite.order_history.assert_called_once_with(order_id="test_order_123")
        
        # Check the result
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], self.sample_order)
        
        # Check if it was cached
        self.assertEqual(self.order_manager.order_history_cache["test_order_123"], [self.sample_order])
    
    def test_get_order_status_from_api(self):
        """Test getting the status of an order from API."""
        # Set up mock return value for kite.order_history
        self.mock_kite.order_history.return_value = [self.sample_order]
        
        # Call the method
        status = self.order_manager.get_order_status(order_id="test_order_123")
        
        # Check if the mock was called correctly
        self.mock_kite.order_history.assert_called_once_with(order_id="test_order_123")
        
        # Check the result
        self.assertEqual(status, "COMPLETE")
    
    def test_get_order_status_from_cache(self):
        """Test getting the status of an order from cache."""
        # Populate the cache
        self.order_manager.order_history_cache["test_order_123"] = [self.sample_order]
        
        # Call the method
        status = self.order_manager.get_order_status(order_id="test_order_123")
        
        # Check that order_history was not called (used cache instead)
        self.mock_kite.order_history.assert_not_called()
        
        # Check the result
        self.assertEqual(status, "COMPLETE")
    
    def test_is_order_complete(self):
        """Test checking if an order is complete."""
        # Populate the cache
        self.order_manager.order_history_cache["test_order_123"] = [self.sample_order]
        
        # Call the method
        is_complete = self.order_manager.is_order_complete(order_id="test_order_123")
        
        # Check the result
        self.assertTrue(is_complete)
    
    def test_is_order_open(self):
        """Test checking if an order is still open."""
        # Populate the cache with an open order
        open_order = dict(self.sample_order)
        open_order["status"] = "OPEN"
        self.order_manager.order_history_cache["test_order_456"] = [open_order]
        
        # Call the method
        is_open = self.order_manager.is_order_open(order_id="test_order_456")
        
        # Check the result
        self.assertTrue(is_open)
        
        # Check with a completed order
        self.order_manager.order_history_cache["test_order_123"] = [self.sample_order]
        is_open = self.order_manager.is_order_open(order_id="test_order_123")
        self.assertFalse(is_open)
    
    def test_get_trades_for_all_orders(self):
        """Test getting trades for all orders."""
        # Set up mock return value for kite.trades
        self.mock_kite.trades.return_value = self.sample_trades
        
        # Call the method
        trades = self.order_manager.get_trades()
        
        # Check if the mock was called correctly
        self.mock_kite.trades.assert_called_once()
        
        # Check the result
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0], self.sample_trades[0])
    
    def test_get_trades_for_specific_order(self):
        """Test getting trades for a specific order."""
        # Set up mock return value for kite.order_trades
        self.mock_kite.order_trades.return_value = self.sample_trades
        
        # Call the method
        trades = self.order_manager.get_trades(order_id="test_order_123")
        
        # Check if the mock was called correctly
        self.mock_kite.order_trades.assert_called_once_with(order_id="test_order_123")
        
        # Check the result
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0], self.sample_trades[0])
    
    def test_get_positions(self):
        """Test getting positions."""
        # Set up mock return value for kite.positions
        self.mock_kite.positions.return_value = self.sample_positions
        
        # Call the method
        positions = self.order_manager.get_positions()
        
        # Check if the mock was called correctly
        self.mock_kite.positions.assert_called_once()
        
        # Check the result
        self.assertEqual(positions, self.sample_positions)
        self.assertEqual(len(positions["day"]), 1)
        self.assertEqual(len(positions["net"]), 1)
    
    def test_get_holdings(self):
        """Test getting holdings."""
        # Set up mock return value for kite.holdings
        self.mock_kite.holdings.return_value = self.sample_holdings
        
        # Call the method
        holdings = self.order_manager.get_holdings()
        
        # Check if the mock was called correctly
        self.mock_kite.holdings.assert_called_once()
        
        # Check the result
        self.assertEqual(holdings, self.sample_holdings)
        self.assertEqual(len(holdings), 1)
    
    def test_place_order_exception(self):
        """Test exception handling when placing an order."""
        # Set up mock to raise an exception
        self.mock_kite.place_order.side_effect = Exception("API error")
        
        # Call the method and expect an exception
        with self.assertRaises(Exception):
            self.order_manager.place_order(
                symbol="RELIANCE",
                exchange="NSE",
                transaction_type="BUY",
                quantity=1,
                product="CNC",
                order_type="MARKET"
            )

if __name__ == '__main__':
    unittest.main()