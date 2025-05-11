"""
Tests for the risk management module of the trading system.
"""
import unittest
from unittest.mock import MagicMock, patch
import math

from risk.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Tests for the RiskManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create default risk manager
        self.risk_manager = RiskManager()
        
        # Create a risk manager with custom settings
        self.custom_risk_manager = RiskManager(
            max_position_size=0.10,      # 10% of capital
            max_risk_per_trade=0.02,     # 2% risk per trade
            stop_loss_pct=0.05,          # 5% stop loss
            max_positions=5,             # Maximum 5 positions
            max_correlated_positions=2,  # Maximum 2 correlated positions
            max_sector_exposure=0.30,    # Maximum 30% sector exposure
            max_drawdown=0.15,           # Maximum 15% drawdown
            position_sizing_method="fixed_size"  # Use fixed size method
        )
        
        # Sample signal data
        self.sample_signal = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "direction": "BUY",
            "product": "CNC",
            "strategy_name": "moving_average_crossover",
            "atr": 50.0,
            "sector": "energy",
            "correlated_symbols": ["ONGC", "IOC"],
            "current_drawdown": 0.05,
            "required_margin": 10000,
            "risk_reward_ratio": 2.0
        }
        
        # Sample tick data
        self.sample_tick = {
            "symbol": "RELIANCE",
            "last_price": 2500.0,
            "volume": 1000,
            "depth": {
                "buy": [{"price": 2499.0, "quantity": 100}],
                "sell": [{"price": 2501.0, "quantity": 100}]
            }
        }
        
        # Sample portfolio data
        self.sample_portfolio = {
            "capital": 100000,
            "available_margin": 80000,
            "portfolio_value": 120000,
            "current_positions": [
                {
                    "symbol": "TCS",
                    "quantity": 5,
                    "average_price": 3500,
                    "value": 17500,
                    "sector": "technology"
                },
                {
                    "symbol": "INFY",
                    "quantity": 10,
                    "average_price": 1500,
                    "value": 15000,
                    "sector": "technology"
                }
            ]
        }
    
    def test_initialization(self):
        """Test initialization of RiskManager."""
        # Check default values
        self.assertEqual(self.risk_manager.max_position_size, 0.05)
        self.assertEqual(self.risk_manager.max_risk_per_trade, 0.01)
        self.assertEqual(self.risk_manager.stop_loss_pct, 0.02)
        self.assertEqual(self.risk_manager.position_sizing_method, "percent_risk")
        
        # Check custom values
        self.assertEqual(self.custom_risk_manager.max_position_size, 0.10)
        self.assertEqual(self.custom_risk_manager.max_risk_per_trade, 0.02)
        self.assertEqual(self.custom_risk_manager.stop_loss_pct, 0.05)
        self.assertEqual(self.custom_risk_manager.position_sizing_method, "fixed_size")
    
    def test_calculate_stop_loss_long(self):
        """Test calculating stop loss for long position."""
        # Test percentage-based stop loss
        entry_price = 1000
        expected_stop_loss = 980  # 1000 - (1000 * 0.02)
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, is_long=True)
        self.assertEqual(stop_loss, expected_stop_loss)
        
        # Test ATR-based stop loss
        atr = 20
        expected_stop_loss = 960  # 1000 - (20 * 2)
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, is_long=True, atr=atr)
        self.assertEqual(stop_loss, expected_stop_loss)
    
    def test_calculate_stop_loss_short(self):
        """Test calculating stop loss for short position."""
        # Test percentage-based stop loss
        entry_price = 1000
        expected_stop_loss = 1020  # 1000 + (1000 * 0.02)
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, is_long=False)
        self.assertEqual(stop_loss, expected_stop_loss)
        
        # Test ATR-based stop loss
        atr = 20
        expected_stop_loss = 1040  # 1000 + (20 * 2)
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, is_long=False, atr=atr)
        self.assertEqual(stop_loss, expected_stop_loss)
    
    def test_calculate_position_size_percent_risk(self):
        """Test calculating position size using percent risk method."""
        capital = 100000
        current_price = 1000
        stop_loss_price = 980
        
        # Expected calculation:
        # Risk per share = 1000 - 980 = 20
        # Max risk amount = 100000 * 0.01 = 1000
        # Position size = 1000 / 20 = 50 shares
        # Max position size = (100000 * 0.05) / 1000 = 5 shares
        # Result should be min(50, 5) = 5 shares
        
        expected_position_size = 5
        
        position_size = self.risk_manager.calculate_position_size(
            capital=capital,
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )
        
        self.assertEqual(position_size, expected_position_size)
    
    def test_calculate_position_size_fixed_size(self):
        """Test calculating position size using fixed size method."""
        capital = 100000
        current_price = 1000
        
        # Expected calculation:
        # Position size = (100000 * 0.10) / 1000 = 10 shares
        
        expected_position_size = 10
        
        position_size = self.custom_risk_manager.calculate_position_size(
            capital=capital,
            current_price=current_price
        )
        
        self.assertEqual(position_size, expected_position_size)
    
    def test_calculate_position_size_volatility(self):
        """Test calculating position size using volatility method."""
        # Create a risk manager with volatility method
        volatility_risk_manager = RiskManager(position_sizing_method="volatility")
        
        capital = 100000
        current_price = 1000
        atr = 20
        
        # Expected calculation:
        # Max risk amount = 100000 * 0.01 = 1000
        # Position size = 1000 / (20 * 2) = 25 shares
        # Max position size = (100000 * 0.05) / 1000 = 5 shares
        # Result should be min(25, 5) = 5 shares
        
        expected_position_size = 5
        
        position_size = volatility_risk_manager.calculate_position_size(
            capital=capital,
            current_price=current_price,
            atr=atr
        )
        
        self.assertEqual(position_size, expected_position_size)
    
    def test_should_trade_max_positions(self):
        """Test should_trade with maximum positions."""
        # Create a risk manager with max_positions=2
        risk_manager = RiskManager(max_positions=2)
        
        # Portfolio already has 2 positions
        result = risk_manager.should_trade(
            signal=self.sample_signal,
            current_positions=self.sample_portfolio["current_positions"],
            available_margin=self.sample_portfolio["available_margin"],
            portfolio_value=self.sample_portfolio["portfolio_value"]
        )
        
        # Should return False since we're at max positions
        self.assertFalse(result)
    
    def test_should_trade_sector_exposure(self):
        """Test should_trade with sector exposure limit."""
        # Create a risk manager with max_sector_exposure=0.10 (10%)
        risk_manager = RiskManager(max_sector_exposure=0.10)
        
        # Current technology sector exposure: (17500 + 15000) / 120000 = 0.27 (27%)
        
        # Signal for another technology stock
        tech_signal = dict(self.sample_signal)
        tech_signal["sector"] = "technology"
        
        result = risk_manager.should_trade(
            signal=tech_signal,
            current_positions=self.sample_portfolio["current_positions"],
            available_margin=self.sample_portfolio["available_margin"],
            portfolio_value=self.sample_portfolio["portfolio_value"]
        )
        
        # Should return False due to sector exposure
        self.assertFalse(result)
        
        # Signal for a different sector should be allowed
        energy_signal = dict(self.sample_signal)
        energy_signal["sector"] = "energy"
        
        result = risk_manager.should_trade(
            signal=energy_signal,
            current_positions=self.sample_portfolio["current_positions"],
            available_margin=self.sample_portfolio["available_margin"],
            portfolio_value=self.sample_portfolio["portfolio_value"]
        )
        
        # Should return True for different sector
        self.assertTrue(result)
    
    def test_should_trade_margin_check(self):
        """Test should_trade with margin check."""
        # Create signal requiring more margin than available
        high_margin_signal = dict(self.sample_signal)
        high_margin_signal["required_margin"] = 100000  # More than available (80000)
        
        result = self.risk_manager.should_trade(
            signal=high_margin_signal,
            current_positions=self.sample_portfolio["current_positions"],
            available_margin=self.sample_portfolio["available_margin"],
            portfolio_value=self.sample_portfolio["portfolio_value"]
        )
        
        # Should return False due to insufficient margin
        self.assertFalse(result)
    
    def test_adjust_position_size(self):
        """Test adjusting position size based on existing exposure."""
        # Create a portfolio with high exposure
        # Total exposure: 80000 out of 100000 (80%)
        high_exposure_portfolio = {
            "current_positions": [
                {
                    "symbol": "TCS",
                    "quantity": 10,
                    "average_price": 4000,
                    "value": 40000,
                    "sector": "technology"
                },
                {
                    "symbol": "INFY",
                    "quantity": 20,
                    "average_price": 2000,
                    "value": 40000,
                    "sector": "technology"
                }
            ],
            "portfolio_value": 100000
        }
        
        # Maximum position size would be 5% of 100000 = 5000
        # Remaining exposure: 20000 out of 100000 (20%)
        # Scaling factor: 0.20 / 0.05 = 4
        # Original quantity: 10
        # Adjusted quantity should be 10 (since 4 > 1)
        
        result = self.risk_manager.adjust_position_size(
            symbol="RELIANCE",
            quantity=10,
            current_positions=high_exposure_portfolio["current_positions"],
            portfolio_value=high_exposure_portfolio["portfolio_value"]
        )
        
        # Should return original quantity since there's enough room
        self.assertEqual(result, 10)
        
        # Even higher exposure (95%)
        very_high_exposure_portfolio = {
            "current_positions": [
                {
                    "symbol": "TCS",
                    "quantity": 10,
                    "average_price": 4000,
                    "value": 40000,
                    "sector": "technology"
                },
                {
                    "symbol": "INFY",
                    "quantity": 20,
                    "average_price": 2000,
                    "value": 40000,
                    "sector": "technology"
                },
                {
                    "symbol": "HDFC",
                    "quantity": 5,
                    "average_price": 3000,
                    "value": 15000,
                    "sector": "finance"
                }
            ],
            "portfolio_value": 100000
        }
        
        # Remaining exposure: 5000 out of 100000 (5%)
        # Scaling factor: 0.05 / 0.05 = 1
        # Original quantity: 10
        # Adjusted quantity should be 10 since there's just enough room
        
        result = self.risk_manager.adjust_position_size(
            symbol="RELIANCE",
            quantity=10,
            current_positions=very_high_exposure_portfolio["current_positions"],
            portfolio_value=very_high_exposure_portfolio["portfolio_value"]
        )
        
        # Should adjust the quantity
        self.assertEqual(result, 10)
    
    def test_process_signal(self):
        """Test processing a strategy signal."""
        # Process a signal
        order_params = self.risk_manager.process_signal(
            signal=self.sample_signal,
            tick_data=self.sample_tick,
            portfolio=self.sample_portfolio
        )
        
        # Check the order parameters
        self.assertIsNotNone(order_params)
        self.assertEqual(order_params["symbol"], "RELIANCE")
        self.assertEqual(order_params["exchange"], "NSE")
        self.assertEqual(order_params["transaction_type"], "BUY")
        self.assertEqual(order_params["product"], "CNC")
        self.assertEqual(order_params["order_type"], "MARKET")
        self.assertEqual(order_params["tag"], "moving_average_crossover")
        
        # Check that stop loss and take profit are included
        self.assertIn("stop_loss", order_params)
        self.assertIn("take_profit", order_params)
        
        # Check that position size is calculated
        self.assertGreater(order_params["quantity"], 0)

if __name__ == '__main__':
    unittest.main()