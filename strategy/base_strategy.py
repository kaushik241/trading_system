"""
Base strategy class for the trading system.

This module defines the interface that all trading strategies must implement.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(
        self,
        name: str = 'BaseStrategy',
        universe: List[str] = None,
        timeframe: str = '5minute',  # Available: 1minute, 5minute, 15minute, 30minute, 60minute, day
        capital: float = 100000,
        is_intraday: bool = True,
        exchange: str = 'NSE',
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            universe: List of symbols to trade
            timeframe: Candle interval for strategy
            capital: Initial capital for trading
            is_intraday: Whether to use intraday rules
            exchange: Exchange to trade
        """
        self.name = name
        self.universe = universe or []
        self.timeframe = timeframe
        self.capital = capital
        self.is_intraday = is_intraday
        self.exchange = exchange
        
        # Store current positions
        self.positions = {}  # symbol -> position_info
        
        # Store historical data
        self.historical_data = {}  # symbol -> DataFrame
        
        # Store latest indicators
        self.indicators = {}  # symbol -> indicator_values
        
        # Signal and order tracking
        self.signals = {}  # symbol -> latest_signal
        self.pending_orders = {}  # order_id -> order_info
        
        # Time control for intraday strategies
        self.market_open_time = time(9, 15)  # 9:15 AM
        self.market_close_time = time(15, 30)  # 3:30 PM
        self.exit_time = time(15, 14)  # 3:14 PM for intraday exit
        
        logger.info(f"Initialized {self.name} strategy for {len(self.universe)} symbols")
    
    def add_symbols(self, symbols: List[str]) -> None:
        """
        Add symbols to the trading universe.
        
        Args:
            symbols: List of symbols to add
        """
        for symbol in symbols:
            if symbol not in self.universe:
                self.universe.append(symbol)
        logger.info(f"Added {len(symbols)} symbols to {self.name} universe")
    
    def remove_symbols(self, symbols: List[str]) -> None:
        """
        Remove symbols from the trading universe.
        
        Args:
            symbols: List of symbols to remove
        """
        for symbol in symbols:
            if symbol in self.universe:
                self.universe.remove(symbol)
        logger.info(f"Removed {len(symbols)} symbols from {self.name} universe")
    
    def set_universe(self, universe: List[str]) -> None:
        """
        Set the trading universe.
        
        Args:
            universe: List of symbols to trade
        """
        self.universe = universe.copy()
        logger.info(f"Set {self.name} universe to {len(self.universe)} symbols")
    
    def update_historical_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with historical data
        """
        self.historical_data[symbol] = data
        logger.debug(f"Updated historical data for {symbol} with {len(data)} candles")
    
    def get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with historical data, or None if not available
        """
        return self.historical_data.get(symbol)
    
    def update_position(self, symbol: str, position_info: Dict[str, Any]) -> None:
        """
        Update position information for a symbol.
        
        Args:
            symbol: Trading symbol
            position_info: Dictionary with position information
        """
        self.positions[symbol] = position_info
        logger.debug(f"Updated position for {symbol}: {position_info}")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position information, or None if not found
        """
        return self.positions.get(symbol)
    
    def is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if the market is open.
        
        Args:
            current_time: Current time (default: now)
            
        Returns:
            True if market is open, False otherwise
        """
        if current_time is None:
            current_time = datetime.now().time()
        elif isinstance(current_time, datetime):
            current_time = current_time.time()
        
        return self.market_open_time <= current_time < self.market_close_time
    
    def should_exit_intraday(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if intraday positions should be exited.
        
        Args:
            current_time: Current time (default: now)
            
        Returns:
            True if positions should be exited, False otherwise
        """
        if not self.is_intraday:
            return False
            
        if current_time is None:
            current_time = datetime.now().time()
        elif isinstance(current_time, datetime):
            current_time = current_time.time()
        
        return current_time >= self.exit_time
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate strategy-specific indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        pass
    
    @abstractmethod
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on the data.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            Signal dictionary with entry/exit information
        """
        pass
    
    @abstractmethod
    def process_tick(self, symbol: str, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a real-time tick and generate trading signal if needed.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
            
        Returns:
            Signal dictionary if a signal is generated, None otherwise
        """
        pass
    
    def update_indicators(self, symbol: str, indicators: Dict[str, Any]) -> None:
        """
        Update indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of indicator values
        """
        self.indicators[symbol] = indicators
        logger.debug(f"Updated indicators for {symbol}")
    
    def get_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicator values, or None if not found
        """
        return self.indicators.get(symbol)
    
    def update_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """
        Update the latest signal for a symbol.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
        """
        self.signals[symbol] = signal
        logger.info(f"Generated signal for {symbol}: {signal}")
    
    def get_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest signal for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Signal dictionary, or None if not found
        """
        return self.signals.get(symbol)
    
    def prepare_order_parameters(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare order parameters from a signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Dictionary with order parameters
        """
        # Basic order parameters
        order_params = {
            "symbol": signal.get("symbol"),
            "exchange": signal.get("exchange", self.exchange),
            "transaction_type": signal.get("action"),
            "quantity": signal.get("quantity", 0),
            "product": "MIS" if self.is_intraday else "CNC",
            "order_type": signal.get("order_type", "MARKET"),
            "price": signal.get("price"),
            "trigger_price": signal.get("trigger_price"),
            "tag": f"{self.name}_{signal.get('signal_type', 'signal')}"
        }
        
        # Remove None values
        order_params = {k: v for k, v in order_params.items() if v is not None}
        
        return order_params
    
    def backtest(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data (symbol -> DataFrame)
            initial_capital: Initial capital for backtest
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            
        Returns:
            Dictionary with backtest results
        """
        # Placeholder for backtest results
        results = {
            "strategy_name": self.name,
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trades": [],
            "equity_curve": pd.DataFrame()
        }
        
        logger.info(f"Backtesting not fully implemented in base class.")
        return results
    
    def on_bar_close(self, symbol: str, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a bar close and generate trading signal if needed.
        
        Args:
            symbol: Trading symbol
            bar: Bar data dictionary
            
        Returns:
            Signal dictionary if a signal is generated, None otherwise
        """
        # Get historical data
        data = self.get_historical_data(symbol)
        if data is None:
            logger.warning(f"No historical data available for {symbol}")
            return None
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        self.update_indicators(symbol, indicators)
        
        # Generate signals
        signal = self.generate_signals(symbol, data)
        if signal:
            self.update_signal(symbol, signal)
            return signal
        
        return None
    
    def generate_signals_realtime(self, symbol: str, tick: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on real-time data.
        
        Args:
            symbol: Trading symbol
            tick: Tick data dictionary
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Check if market is open
        current_time = datetime.now()
        if not self.is_market_open(current_time):
            logger.debug(f"Market is closed. No signals generated for {symbol}")
            return signals
        
        # Check if intraday positions should be exited
        if self.should_exit_intraday(current_time) and self.is_intraday:
            position = self.get_position(symbol)
            if position and position.get("quantity", 0) != 0:
                # Generate exit signal
                exit_signal = {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL" if position.get("quantity", 0) > 0 else "BUY",
                    "quantity": abs(position.get("quantity", 0)),
                    "signal_type": "time_exit",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "intraday_exit_time"
                }
                signals.append(exit_signal)
                logger.info(f"Generated time-based exit signal for {symbol}")
        
        # Process tick to generate signals
        signal = self.process_tick(symbol, tick)
        if signal:
            signals.append(signal)
        
        return signals