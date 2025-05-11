"""
Moving average crossover strategy implementation.

This strategy:
1. Calculates exponential moving averages (EMA) of specified periods
2. Generates a buy signal when the faster EMA crosses above the slower EMA
3. Generates a sell signal when the faster EMA crosses below the slower EMA
4. Exits positions at 15:14 for intraday trading
"""
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """Moving average crossover strategy."""
    
    def __init__(
        self,
        name: str = 'MovingAverageCrossover',
        universe: List[str] = None,
        timeframe: str = '5minute',
        capital: float = 100000,
        fast_period: int = 5,
        slow_period: int = 20,
        is_intraday: bool = True,
        exchange: str = 'NSE',
        atr_period: int = 14  # ATR period for stop loss calculation
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            universe: List of symbols to trade
            timeframe: Candle interval for strategy
            capital: Initial capital for trading
            fast_period: Period of the fast moving average
            slow_period: Period of the slow moving average
            is_intraday: Whether to use intraday rules
            exchange: Exchange to trade
            atr_period: Period for ATR calculation
        """
        super().__init__(
            name=name,
            universe=universe or [],
            timeframe=timeframe,
            capital=capital,
            is_intraday=is_intraday,
            exchange=exchange
        )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        
        # Ensure the periods are valid
        if fast_period >= slow_period:
            logger.warning(f"Fast period ({fast_period}) should be less than slow period ({slow_period})")
        
        # Additional data structures for tracking crossovers
        self.last_crossover = {}  # symbol -> {"type": "bullish"/"bearish", "timestamp": datetime}
        
        logger.info(f"Initialized {self.name} with {fast_period}/{slow_period} EMAs")
    
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
        return data[column].ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
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
        indicators['ema_fast'] = self.calculate_ema(data, self.fast_period)
        indicators['ema_slow'] = self.calculate_ema(data, self.slow_period)
        
        # Calculate ATR for stop loss determination
        indicators['atr'] = self.calculate_atr(data, self.atr_period)
        
        # Create comparison series (not numpy array)
        indicators['crossover'] = pd.Series(
            index=data.index,
            data=np.where(
                indicators['ema_fast'] > indicators['ema_slow'], 1,
                np.where(indicators['ema_fast'] < indicators['ema_slow'], -1, 0)
            )
        )
        
        # Calculate crossover signals (1 for bullish, -1 for bearish, 0 for no crossover)
        indicators['crossover_signal'] = indicators['crossover'].diff().fillna(0)
        
        # Print some debug info
        print(f"Calculated indicators for {len(data)} candles")
        print(f"  Last 5 candles EMA comparison:")
        for i in range(1, min(6, len(data)) + 1):
            idx = -i
            print(f"    {data.index[idx]} - Fast: {indicators['ema_fast'].iloc[idx]:.2f}, " 
                  f"Slow: {indicators['ema_slow'].iloc[idx]:.2f}, "
                  f"Diff: {(indicators['ema_fast'].iloc[idx] - indicators['ema_slow'].iloc[idx]):.2f}, "
                  f"Crossover: {indicators['crossover'].iloc[idx]}, "
                  f"Signal: {indicators['crossover_signal'].iloc[idx]}")
        
        return indicators
    
    def detect_crossover(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[str]:
        """
        Detect if a crossover occurred in the latest candle.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: Dictionary of indicator values
            
        Returns:
            'bullish' for bullish crossover, 'bearish' for bearish crossover, None otherwise
        """
        # Ensure we have enough data
        if len(data) < 2:
            print("  Not enough data to detect crossover")
            return None
            
        # Get the crossover signal values
        crossover_signal = indicators['crossover_signal']
        
        # Debug info
        print(f"Checking for crossover in last candle:")
        last_index = crossover_signal.index[-1]
        last_value = crossover_signal.iloc[-1]
        print(f"  Last candle at {last_index} has signal value: {last_value}")
        
        # Check for crossover in the latest candle
        if last_value > 0:  # Fast EMA crossed above slow EMA
            print(f"  BULLISH CROSSOVER DETECTED! Fast EMA crossed above Slow EMA")
            return 'bullish'
        elif last_value < 0:  # Fast EMA crossed below slow EMA
            print(f"  BEARISH CROSSOVER DETECTED! Fast EMA crossed below Slow EMA")
            return 'bearish'
        else:
            print(f"  No crossover detected in the latest candle")
            return None
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals based on the data.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary with entry/exit information
        """
        if len(data) < self.slow_period:
            print(f"Insufficient data for {symbol}: {len(data)} candles")
            return {}
            
        # Get current position
        position = self.get_position(symbol)
        position_qty = position.get('quantity', 0) if position else 0
        
        print(f"Current position for {symbol}: {position_qty} shares")
        
        # Get indicators or calculate if not available
        indicators = self.get_indicators(symbol)
        if not indicators:
            print(f"No indicators found for {symbol}, calculating now...")
            indicators = self.calculate_indicators(data)
            self.update_indicators(symbol, indicators)
        
        # Detect crossover
        print(f"Detecting crossover for {symbol}...")
        crossover = self.detect_crossover(data, indicators)
        
        if crossover:
            # Update last crossover information
            self.last_crossover[symbol] = {
                "type": crossover,
                "timestamp": data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()
            }
            
            print(f"Crossover detected for {symbol}: {crossover}")
            
            # Generate signals based on crossover and current position
            if crossover == 'bullish' and position_qty <= 0:
                # Generate buy signal
                print(f"Generating BUY signal for {symbol}")
                last_close = data['close'].iloc[-1]
                last_atr = indicators['atr'].iloc[-1]
                
                # Calculate stop loss based on ATR
                stop_loss = last_close - (last_atr * 2)
                
                signal = {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "BUY",
                    "quantity": 1,  # Will be sized by risk manager
                    "signal_type": "ema_crossover_bullish",
                    "timestamp": data.index[-1].strftime("%Y-%m-%d %H:%M:%S") if isinstance(data.index[-1], datetime) else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": last_close,
                    "stop_loss": stop_loss,
                    "indicators": {
                        "ema_fast": indicators['ema_fast'].iloc[-1],
                        "ema_slow": indicators['ema_slow'].iloc[-1],
                        "atr": last_atr
                    }
                }
                print(f"BUY signal generated for {symbol}: {signal}")
                return signal
                
            elif crossover == 'bearish' and position_qty >= 0:
                # Generate sell signal
                print(f"Generating SELL signal for {symbol}")
                last_close = data['close'].iloc[-1]
                last_atr = indicators['atr'].iloc[-1]
                
                # Calculate stop loss based on ATR
                stop_loss = last_close + (last_atr * 2)
                
                signal = {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "action": "SELL",
                    "quantity": abs(position_qty) if position_qty > 0 else 1,  # Exit existing or size by risk
                    "signal_type": "ema_crossover_bearish",
                    "timestamp": data.index[-1].strftime("%Y-%m-%d %H:%M:%S") if isinstance(data.index[-1], datetime) else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "price": last_close,
                    "stop_loss": stop_loss,
                    "indicators": {
                        "ema_fast": indicators['ema_fast'].iloc[-1],
                        "ema_slow": indicators['ema_slow'].iloc[-1],
                        "atr": last_atr
                    }
                }
                print(f"SELL signal generated for {symbol}: {signal}")
                return signal
        
        print(f"No signal generated for {symbol}")
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
        # Get historical data
        data = self.get_historical_data(symbol)
        if data is None or len(data) < self.slow_period:
            logger.warning(f"Insufficient historical data for {symbol} to process tick")
            return None
        
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
            
            # Process the new bar
            return self.on_bar_close(symbol, new_bar)
        
        return None
    
    def get_nifty50_stocks(self) -> List[str]:
        """Get the list of Nifty 50 stocks."""
        # This is a simplified list. In reality, you would get this dynamically.
        # You could fetch this from an API or a database.
        nifty50 = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
            "HDFC", "ITC", "KOTAKBANK", "LT", "HINDUNILVR", 
            "SBIN", "BAJFINANCE", "AXISBANK", "BHARTIARTL", "ASIANPAINT", 
            "MARUTI", "HCLTECH", "TITAN", "SUNPHARMA", "NESTLEIND", 
            "ULTRACEMCO", "TATAMOTORS", "WIPRO", "ADANIENT", "POWERGRID", 
            "JSWSTEEL", "BAJAJFINSV", "NTPC", "TATASTEEL", "M&M", 
            "TECHM", "BAJAJ-AUTO", "ADANIPORTS", "HINDALCO", "GRASIM", 
            "HDFCLIFE", "COALINDIA", "DIVISLAB", "ONGC", "DRREDDY", 
            "SBILIFE", "EICHERMOT", "INDUSINDBK", "TATACONSUM", "BRITANNIA", 
            "CIPLA", "UPL", "HEROMOTOCO", "APOLLOHOSP", "BPCL"
        ]
        return nifty50
    
    def initialize_universe(self) -> None:
        """Initialize the trading universe with Nifty 50 stocks."""
        nifty50 = self.get_nifty50_stocks()
        self.set_universe(nifty50)
        logger.info(f"Initialized universe with {len(nifty50)} Nifty 50 stocks")