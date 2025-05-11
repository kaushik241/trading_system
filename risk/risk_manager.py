"""
Risk management module for the trading system.

This module handles:
- Position sizing
- Stop loss calculation
- Risk per trade management
- Portfolio risk control
"""
import logging
import math
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk for trading positions."""
    
    def __init__(
        self,
        max_position_size: float = 0.05,  # 5% of capital per position
        max_risk_per_trade: float = 0.01,  # 1% risk per trade
        stop_loss_pct: float = 0.02,      # 2% stop loss
        max_positions: int = 10,          # Maximum number of open positions
        max_correlated_positions: int = 3, # Maximum correlated positions
        max_sector_exposure: float = 0.20, # Maximum 20% exposure per sector
        max_drawdown: float = 0.10,       # Maximum 10% drawdown
        position_sizing_method: str = "percent_risk" # Method for sizing positions
    ):
        """
        Initialize the risk manager.
        
        Args:
            max_position_size: Maximum position size as a fraction of capital
            max_risk_per_trade: Maximum risk per trade as a fraction of capital
            stop_loss_pct: Default stop loss percentage
            max_positions: Maximum number of open positions
            max_correlated_positions: Maximum number of correlated positions
            max_sector_exposure: Maximum exposure to any one sector
            max_drawdown: Maximum drawdown allowed
            position_sizing_method: Method for calculating position size
                (percent_risk, fixed_size, volatility)
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        self.max_correlated_positions = max_correlated_positions
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown = max_drawdown
        self.position_sizing_method = position_sizing_method
        
    def calculate_position_size(
        self,
        capital: float,
        current_price: float,
        stop_loss_price: Optional[float] = None,
        atr: Optional[float] = None,
        symbol: Optional[str] = None,
        is_long: bool = True
    ) -> int:
        """
        Calculate the position size based on risk parameters.
        
        Args:
            capital: Available capital
            current_price: Current price of the asset
            stop_loss_price: Stop loss price (optional)
            atr: Average True Range for volatility-based sizing (optional)
            symbol: Trading symbol (optional)
            is_long: Whether the position is long (True) or short (False)
            
        Returns:
            Position size in number of shares/units
        """
        # Calculate stop loss price if not provided
        if stop_loss_price is None:
            stop_loss_price = self.calculate_stop_loss(current_price, is_long)
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk per share: {risk_per_share}. Using default stop loss.")
            # Default to stop_loss_pct if the risk_per_share is invalid
            risk_per_share = current_price * self.stop_loss_pct
        
        # Calculate the max amount to risk on this trade
        max_risk_amount = capital * self.max_risk_per_trade
        
        # Calculate position size based on selected method
        if self.position_sizing_method == "percent_risk":
            # Risk a fixed percentage of capital
            position_size = max_risk_amount / risk_per_share
            
        elif self.position_sizing_method == "fixed_size":
            # Use a fixed percentage of capital
            position_size = (capital * self.max_position_size) / current_price
            
        elif self.position_sizing_method == "volatility":
            # Use ATR for volatility-based sizing
            if atr is None or atr <= 0:
                logger.warning("Invalid ATR for volatility-based position sizing. Using percent risk method.")
                position_size = max_risk_amount / risk_per_share
            else:
                # Use a multiple of ATR for position sizing
                atr_multiple = 2.0  # Risk 2x ATR per trade
                position_size = max_risk_amount / (atr * atr_multiple)
        else:
            logger.warning(f"Unknown position sizing method: {self.position_sizing_method}. Using percent risk.")
            position_size = max_risk_amount / risk_per_share
        
        # Calculate max position size based on percentage of capital
        max_position_shares = (capital * self.max_position_size) / current_price
        
        # Take the smaller of the two values
        position_size = min(position_size, max_position_shares)
        
        # Round down to whole number of shares
        position_size = math.floor(position_size)
        
        # Ensure position size is at least 1
        position_size = max(1, position_size)
        
        logger.info(f"Calculated position size: {position_size} shares at {current_price} per share")
        return position_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool = True,
        atr: Optional[float] = None,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss price based on entry price and direction.
        
        Args:
            entry_price: Entry price of the position
            is_long: Whether the position is long (True) or short (False)
            atr: Average True Range for volatility-based stop loss (optional)
            multiplier: Multiplier for ATR-based stop loss
            
        Returns:
            Stop loss price
        """
        if atr is not None and atr > 0:
            # Calculate ATR-based stop loss
            stop_distance = atr * multiplier
        else:
            # Calculate percentage-based stop loss
            stop_distance = entry_price * self.stop_loss_pct
        
        # Calculate stop loss based on position direction
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # Round to 2 decimal places
        stop_loss = round(stop_loss, 2)
        
        logger.info(f"Calculated stop loss: {stop_loss} for {'long' if is_long else 'short'} position at {entry_price}")
        return stop_loss
    
    def should_trade(
        self,
        signal: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
        available_margin: float,
        portfolio_value: float
    ) -> bool:
        """
        Determine if a trade should be placed based on risk constraints.
        
        Args:
            signal: Signal data from strategy
            current_positions: List of current open positions
            available_margin: Available margin for trading
            portfolio_value: Total portfolio value
            
        Returns:
            True if trade should be executed, False otherwise
        """
        # Check number of open positions
        if len(current_positions) >= self.max_positions:
            logger.info(f"Maximum positions ({self.max_positions}) reached. Not placing new trade.")
            return False
        
        # Extract symbol and sector info from signal
        symbol = signal.get("symbol")
        sector = signal.get("sector", "unknown")
        
        # Check sector exposure
        sector_exposure = sum(
            p.get("value", 0) for p in current_positions 
            if p.get("sector") == sector
        ) / portfolio_value if portfolio_value > 0 else 0
        
        if sector_exposure >= self.max_sector_exposure:
            logger.info(f"Maximum sector exposure ({self.max_sector_exposure*100}%) reached for {sector}. Not placing new trade.")
            return False
        
        # Check for correlated positions
        correlated_symbols = signal.get("correlated_symbols", [])
        correlated_positions = [
            p for p in current_positions 
            if p.get("symbol") in correlated_symbols
        ]
        
        if len(correlated_positions) >= self.max_correlated_positions:
            logger.info(f"Maximum correlated positions ({self.max_correlated_positions}) reached. Not placing new trade.")
            return False
        
        # Check current drawdown
        current_drawdown = signal.get("current_drawdown", 0)
        if current_drawdown >= self.max_drawdown:
            logger.info(f"Maximum drawdown ({self.max_drawdown*100}%) reached. Not placing new trade.")
            return False
        
        # Check if we have sufficient margin
        required_margin = signal.get("required_margin", 0)
        if required_margin > available_margin:
            logger.info(f"Insufficient margin available. Required: {required_margin}, Available: {available_margin}")
            return False
        
        # All checks passed
        return True
    
    def adjust_position_size(
        self,
        symbol: str,
        quantity: int,
        current_positions: List[Dict[str, Any]],
        portfolio_value: float
    ) -> int:
        """
        Adjust position size based on existing exposure.
        
        Args:
            symbol: Trading symbol
            quantity: Calculated position size
            current_positions: List of current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Adjusted position size
        """
        # Find existing position in the same symbol
        existing_position = next(
            (p for p in current_positions if p.get("symbol") == symbol),
            None
        )
        
        if existing_position:
            # Factor in existing exposure
            total_exposure = sum(p.get("value", 0) for p in current_positions)
            current_exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # If already over max exposure, don't add more
            if current_exposure_pct >= 1.0:
                logger.warning(f"Already at maximum portfolio exposure. Reducing position size.")
                return 0
            
            # Calculate remaining room for exposure
            remaining_exposure_pct = 1.0 - current_exposure_pct
            
            # Scale quantity if needed
            if remaining_exposure_pct < self.max_position_size:
                scale_factor = remaining_exposure_pct / self.max_position_size
                adjusted_quantity = int(quantity * scale_factor)
                logger.info(f"Adjusting position size from {quantity} to {adjusted_quantity} due to existing exposure")
                return adjusted_quantity
        
        # No adjustment needed
        return quantity
    
    def process_signal(
        self,
        signal: Dict[str, Any],
        tick_data: Dict[str, Any],
        portfolio: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a strategy signal and apply risk management rules.
        
        Args:
            signal: Signal from strategy
            tick_data: Current market data
            portfolio: Portfolio information (optional)
            
        Returns:
            Order parameters dict or None if trade should not be executed
        """
        # Default portfolio data if not provided
        if portfolio is None:
            portfolio = {
                "capital": 100000,
                "current_positions": [],
                "available_margin": 100000,
                "portfolio_value": 100000
            }
        
        symbol = signal.get("symbol")
        direction = signal.get("direction", "BUY")  # Default to BUY
        is_long = direction == "BUY"
        
        # Get current price from tick data
        current_price = tick_data.get("last_price")
        if current_price is None:
            logger.error(f"Missing current price in tick data for {symbol}")
            return None
        
        # Check risk management rules
        if not self.should_trade(
            signal, 
            portfolio.get("current_positions", []),
            portfolio.get("available_margin", 0),
            portfolio.get("portfolio_value", 0)
        ):
            logger.info(f"Trade for {symbol} rejected by risk management rules")
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(
            capital=portfolio.get("capital", 100000),
            current_price=current_price,
            atr=signal.get("atr"),
            symbol=symbol,
            is_long=is_long
        )
        
        # Adjust for existing positions
        quantity = self.adjust_position_size(
            symbol=symbol,
            quantity=quantity,
            current_positions=portfolio.get("current_positions", []),
            portfolio_value=portfolio.get("portfolio_value", 0)
        )
        
        if quantity <= 0:
            logger.info(f"Position size for {symbol} adjusted to zero. Not placing trade.")
            return None
        
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(
            entry_price=current_price,
            is_long=is_long,
            atr=signal.get("atr")
        )
        
        # Calculate take profit (if applicable)
        risk_reward_ratio = signal.get("risk_reward_ratio", 2.0)
        price_diff = abs(current_price - stop_loss)
        take_profit = current_price + (price_diff * risk_reward_ratio) if is_long else current_price - (price_diff * risk_reward_ratio)
        
        # Prepare order parameters
        order_params = {
            "symbol": symbol,
            "exchange": signal.get("exchange", "NSE"),
            "transaction_type": direction,
            "quantity": quantity,
            "product": signal.get("product", "CNC"),
            "order_type": "MARKET",
            "price": None,  # For market order
            "trigger_price": None,
            "variety": "regular"
        }
        
        # Add a tag to identify the strategy
        if "strategy_name" in signal:
            order_params["tag"] = signal["strategy_name"]
            
        logger.info(f"Processed signal for {symbol}. Order parameters: {order_params}")
        
        # Include stop loss and take profit in the return for reference
        # (these won't be included in the actual order, but may be used for subsequent orders)
        order_params["stop_loss"] = stop_loss
        order_params["take_profit"] = take_profit
        
        return order_params