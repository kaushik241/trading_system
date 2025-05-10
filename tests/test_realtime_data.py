"""
Tests for the real-time data module of the trading system.
"""
import unittest
from unittest.mock import MagicMock, patch

from data.realtime_data import RealTimeDataManager

class TestRealTimeDataManager(unittest.TestCase):
    """Tests for the RealTimeDataManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Mock API credentials
        self.api_key = "test_api_key"
        self.access_token = "test_access_token"
        
        # Create an instance of RealTimeDataManager
        self.realtime_manager = RealTimeDataManager(self.api_key, self.access_token)
        
        # Set up mock for KiteTicker
        self.mock_ticker = MagicMock()
        
        # Define test data
        self.test_tokens = [12345, 67890]
        self.test_token_symbol_map = {12345: "RELIANCE", 67890: "TCS"}
        self.test_symbol_token_map = {"RELIANCE": 12345, "TCS": 67890}
        self.test_tick = {
            'instrument_token': 12345,
            'last_price': 2500.0,
            'volume': 1000,
            'depth': {'buy': [], 'sell': []},
            'timestamp': '2023-05-01 09:15:00'
        }
    
    def test_initialization(self):
        """Test initialization of RealTimeDataManager."""
        # Check if instance variables are initialized correctly
        self.assertEqual(self.realtime_manager.api_key, self.api_key)
        self.assertEqual(self.realtime_manager.access_token, self.access_token)
        self.assertIsNone(self.realtime_manager.ticker)
        self.assertFalse(self.realtime_manager.is_connected)
        self.assertEqual(self.realtime_manager.subscribed_tokens, [])
        self.assertEqual(self.realtime_manager.token_symbol_map, {})
        self.assertEqual(self.realtime_manager.symbol_token_map, {})
        self.assertEqual(self.realtime_manager.latest_ticks, {})
        
        # Check if callbacks are initialized to None
        for callback_type in ['on_tick', 'on_connect', 'on_close', 'on_error', 'on_reconnect', 'on_noreconnect', 'on_order_update']:
            self.assertIsNone(self.realtime_manager.callbacks[callback_type])
    
    @patch('data.realtime_data.KiteTicker')
    def test_start(self, mock_kite_ticker):
        """Test starting the WebSocket connection."""
        # Set up mock KiteTicker instance
        mock_kite_ticker.return_value = self.mock_ticker
        
        # Call the method
        result = self.realtime_manager.start()
        
        # Check if KiteTicker was initialized correctly
        mock_kite_ticker.assert_called_once_with(self.api_key, self.access_token)
        
        # Check if callbacks were set up
        self.assertIsNotNone(self.mock_ticker.on_ticks)
        self.assertIsNotNone(self.mock_ticker.on_connect)
        self.assertIsNotNone(self.mock_ticker.on_close)
        self.assertIsNotNone(self.mock_ticker.on_error)
        self.assertIsNotNone(self.mock_ticker.on_reconnect)
        self.assertIsNotNone(self.mock_ticker.on_noreconnect)
        self.assertIsNotNone(self.mock_ticker.on_order_update)
        
        # Check if connect was called
        self.mock_ticker.connect.assert_called_once_with(threaded=True)
        
        # Check return value
        self.assertTrue(result)
    
    def test_stop(self):
        """Test stopping the WebSocket connection."""
        # Set up mock ticker
        self.realtime_manager.ticker = self.mock_ticker
        self.realtime_manager.is_connected = True
        
        # Call the method
        result = self.realtime_manager.stop()
        
        # Check if close was called
        self.mock_ticker.close.assert_called_once()
        
        # Check if instance variables were updated
        self.assertIsNone(self.realtime_manager.ticker)
        self.assertFalse(self.realtime_manager.is_connected)
        
        # Check return value
        self.assertTrue(result)
    
    def test_subscribe(self):
        """Test subscribing to instrument tokens."""
        # Set up mock ticker
        self.realtime_manager.ticker = self.mock_ticker
        self.realtime_manager.is_connected = True
        
        # Call the method
        result = self.realtime_manager.subscribe(
            self.test_tokens,
            self.test_token_symbol_map,
            self.test_symbol_token_map
        )
        
        # Check if instance variables were updated
        self.assertEqual(self.realtime_manager.subscribed_tokens, self.test_tokens)
        self.assertEqual(self.realtime_manager.token_symbol_map, self.test_token_symbol_map)
        self.assertEqual(self.realtime_manager.symbol_token_map, self.test_symbol_token_map)
        
        # Check if subscribe was called
        self.mock_ticker.subscribe.assert_called_once_with(self.test_tokens)
        
        # Check if set_mode was called
        self.mock_ticker.set_mode.assert_called_once()
        
        # Check return value
        self.assertTrue(result)
    
    def test_unsubscribe(self):
        """Test unsubscribing from instrument tokens."""
        # Set up mock ticker and instance variables
        self.realtime_manager.ticker = self.mock_ticker
        self.realtime_manager.is_connected = True
        self.realtime_manager.subscribed_tokens = self.test_tokens
        self.realtime_manager.token_symbol_map = self.test_token_symbol_map
        self.realtime_manager.latest_ticks = {"RELIANCE": self.test_tick}
        
        # Call the method
        result = self.realtime_manager.unsubscribe([12345])
        
        # Check if unsubscribe was called
        self.mock_ticker.unsubscribe.assert_called_once_with([12345])
        
        # Check if instance variables were updated
        self.assertEqual(self.realtime_manager.subscribed_tokens, [67890])
        
        # Check if latest_ticks was updated
        self.assertNotIn("RELIANCE", self.realtime_manager.latest_ticks)
        
        # Check return value
        self.assertTrue(result)
    
    def test_set_mode(self):
        """Test setting mode for instrument tokens."""
        # Set up mock ticker
        self.realtime_manager.ticker = self.mock_ticker
        self.realtime_manager.is_connected = True
        
        # Define modes
        self.mock_ticker.MODE_LTP = "ltp"
        self.mock_ticker.MODE_QUOTE = "quote"
        self.mock_ticker.MODE_FULL = "full"
        
        # Call the method
        result = self.realtime_manager.set_mode("full", self.test_tokens)
        
        # Check if set_mode was called
        self.mock_ticker.set_mode.assert_called_once_with("full", self.test_tokens)
        
        # Check return value
        self.assertTrue(result)
    
    def test_get_last_price(self):
        """Test getting last price for a symbol."""
        # Set up instance variables
        self.realtime_manager.latest_ticks = {"RELIANCE": self.test_tick}
        
        # Call the method
        price = self.realtime_manager.get_last_price("RELIANCE")
        
        # Check return value
        self.assertEqual(price, 2500.0)
        
        # Test for non-existent symbol
        price = self.realtime_manager.get_last_price("NONEXISTENT")
        self.assertIsNone(price)
    
    def test_get_last_tick(self):
        """Test getting last tick for a symbol."""
        # Set up instance variables
        self.realtime_manager.latest_ticks = {"RELIANCE": self.test_tick}
        
        # Call the method
        tick = self.realtime_manager.get_last_tick("RELIANCE")
        
        # Check return value
        self.assertEqual(tick, self.test_tick)
        
        # Test for non-existent symbol
        tick = self.realtime_manager.get_last_tick("NONEXISTENT")
        self.assertIsNone(tick)
    
    def test_register_callback(self):
        """Test registering a callback function."""
        # Define a callback function
        def test_callback():
            pass
        
        # Call the method
        result = self.realtime_manager.register_callback('on_tick', test_callback)
        
        # Check if callback was registered
        self.assertEqual(self.realtime_manager.callbacks['on_tick'], test_callback)
        
        # Check return value
        self.assertTrue(result)
        
        # Test for invalid event type
        result = self.realtime_manager.register_callback('invalid_event', test_callback)
        self.assertFalse(result)
    
    def test_on_ticks(self):
        """Test _on_ticks callback."""
        # Set up instance variables
        self.realtime_manager.token_symbol_map = self.test_token_symbol_map
        
        # Define a mock callback
        mock_callback = MagicMock()
        self.realtime_manager.callbacks['on_tick'] = mock_callback
        
        # Call the method
        ws = MagicMock()
        ticks = [self.test_tick]
        self.realtime_manager._on_ticks(ws, ticks)
        
        # Check if latest_ticks was updated
        self.assertEqual(self.realtime_manager.latest_ticks["RELIANCE"], self.test_tick)
        
        # Check if callback was called
        mock_callback.assert_called_once_with("RELIANCE", self.test_tick)
    
    def test_on_connect(self):
        """Test _on_connect callback."""
        # Set up instance variables
        self.realtime_manager.ticker = self.mock_ticker
        self.realtime_manager.subscribed_tokens = self.test_tokens
        
        # Define a mock callback
        mock_callback = MagicMock()
        self.realtime_manager.callbacks['on_connect'] = mock_callback
        
        # Call the method
        ws = MagicMock()
        response = {'status': 'success'}
        self.realtime_manager._on_connect(ws, response)
        
        # Check if is_connected was updated
        self.assertTrue(self.realtime_manager.is_connected)
        
        # Check if subscribe was called
        self.mock_ticker.subscribe.assert_called_once_with(self.test_tokens)
        
        # Check if set_mode was called
        self.mock_ticker.set_mode.assert_called_once()
        
        # Check if callback was called
        mock_callback.assert_called_once_with(response)

if __name__ == '__main__':
    unittest.main()