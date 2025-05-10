"""
Tests for the data module of the trading system.
"""
import os
import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from data.historical_data import HistoricalDataManager

class TestHistoricalDataManager(unittest.TestCase):
    """Tests for the HistoricalDataManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock KiteConnect instance
        self.mock_kite = MagicMock()
        
        # Set up a test data directory
        self.test_data_dir = "tests/test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create an instance of HistoricalDataManager with mock KiteConnect
        self.data_manager = HistoricalDataManager(self.mock_kite, self.test_data_dir)
        
        # Sample data for testing
        self.sample_data = [
            {
                "date": "2023-01-01 09:15:00",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000
            },
            {
                "date": "2023-01-02 09:15:00",
                "open": 102.0,
                "high": 107.0,
                "low": 101.0,
                "close": 106.0,
                "volume": 1200
            },
            {
                "date": "2023-01-03 09:15:00",
                "open": 106.0,
                "high": 110.0,
                "low": 105.0,
                "close": 108.0,
                "volume": 1500
            }
        ]
        
        # Sample instruments data
        self.sample_instruments = [
            {
                "instrument_token": 12345,
                "exchange_token": 123,
                "tradingsymbol": "RELIANCE",
                "name": "RELIANCE INDUSTRIES LTD",
                "exchange": "NSE"
            },
            {
                "instrument_token": 67890,
                "exchange_token": 678,
                "tradingsymbol": "TCS",
                "name": "TATA CONSULTANCY SERVICES LTD",
                "exchange": "NSE"
            }
        ]
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove test files
        for filename in os.listdir(self.test_data_dir):
            if filename.endswith(".csv"):
                os.remove(os.path.join(self.test_data_dir, filename))
        
        # Try to remove test directory
        try:
            os.rmdir(self.test_data_dir)
        except OSError:
            pass  # Directory not empty or doesn't exist
    
    def test_initialization(self):
        """Test initialization of HistoricalDataManager."""
        # Check if the data directory is created
        self.assertTrue(os.path.exists(self.test_data_dir))
        
        # Check if the instrument_tokens attribute is initialized as an empty dict
        self.assertEqual(self.data_manager.instrument_tokens, {})
    
    def test_get_instrument_tokens(self):
        """Test getting instrument tokens."""
        # Set up mock return value for kite.instruments
        self.mock_kite.instruments.return_value = self.sample_instruments
        
        # Call the method
        tokens = self.data_manager.get_instrument_tokens(["RELIANCE", "TCS"])
        
        # Check if the mock was called correctly
        self.mock_kite.instruments.assert_called_once_with("NSE")
        
        # Check the result
        self.assertEqual(tokens, {"RELIANCE": 12345, "TCS": 67890})
        
        # Check if instrument_tokens attribute is updated
        self.assertEqual(self.data_manager.instrument_tokens, {"RELIANCE": 12345, "TCS": 67890})
    
    def test_fetch_historical_data(self):
        """Test fetching historical data."""
        # Set up mock return value for kite.historical_data
        self.mock_kite.historical_data.return_value = self.sample_data
        
        # Call the method
        from_date = "2023-01-01"
        to_date = "2023-01-03"
        df = self.data_manager.fetch_historical_data(12345, from_date, to_date)
        
        # Check if the mock was called correctly
        self.mock_kite.historical_data.assert_called_once_with(
            instrument_token=12345,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        
        # Check the result
        self.assertEqual(len(df), 3)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
    
    def test_save_and_load_data(self):
        """Test saving and loading data."""
        # Create a test DataFrame
        df = pd.DataFrame(self.sample_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save the data
        symbol = "TESTDATA"
        self.data_manager.save_data(symbol, df)
        
        # Check if the file was created
        filename = os.path.join(self.test_data_dir, f"{symbol}.csv")
        self.assertTrue(os.path.exists(filename))
        
        # Load the data
        loaded_df = self.data_manager.load_data(symbol)
        
        # Check if the loaded data matches the original
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_fetch_multiple_symbols(self):
        """Test fetching data for multiple symbols."""
        # Set up mock return values
        self.mock_kite.instruments.return_value = self.sample_instruments
        self.mock_kite.historical_data.return_value = self.sample_data
        
        # Call the method
        from_date = "2023-01-01"
        to_date = "2023-01-03"
        data = self.data_manager.fetch_multiple_symbols(["RELIANCE", "TCS"], from_date, to_date)
        
        # Check if the mocks were called correctly
        self.mock_kite.instruments.assert_called_once_with("NSE")
        self.assertEqual(self.mock_kite.historical_data.call_count, 2)
        
        # Check the result
        self.assertEqual(len(data), 2)
        self.assertIn("RELIANCE", data)
        self.assertIn("TCS", data)
        
        # Check if data was saved
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "RELIANCE.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "TCS.csv")))
    
    def test_load_multiple_symbols(self):
        """Test loading data for multiple symbols."""
        # Create and save test DataFrames
        df = pd.DataFrame(self.sample_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        symbols = ["SYMBOL1", "SYMBOL2"]
        for symbol in symbols:
            self.data_manager.save_data(symbol, df)
        
        # Load the data
        loaded_data = self.data_manager.load_multiple_symbols(symbols)
        
        # Check the result
        self.assertEqual(len(loaded_data), 2)
        self.assertIn("SYMBOL1", loaded_data)
        self.assertIn("SYMBOL2", loaded_data)
        
        # Check if the loaded data matches the original
        for symbol in symbols:
            pd.testing.assert_frame_equal(df, loaded_data[symbol])
    
    def test_get_latest_data(self):
        """Test getting the latest data for a symbol."""
        # Create a test DataFrame with 100 days of data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()  # Oldest to newest
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "open": 100 + i,
                "high": 105 + i,
                "low": 99 + i,
                "close": 102 + i,
                "volume": 1000 + i * 10
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Save the data
        symbol = "TESTLATEST"
        self.data_manager.save_data(symbol, df)
        
        # Get the latest 30 days of data
        latest_df = self.data_manager.get_latest_data(symbol, days=30)
        
        # Check the result
        self.assertEqual(len(latest_df), 30)
        self.assertEqual(latest_df.iloc[-1]['close'], 102 + 99)  # Last day's close
        self.assertEqual(latest_df.iloc[0]['close'], 102 + 70)  # First day of the 30 days

if __name__ == '__main__':
    unittest.main()