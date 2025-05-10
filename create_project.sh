#!/bin/bash

# Create main package directory and its subdirectories
mkdir -p auth data strategy risk execution utils
mkdir -p notebooks
mkdir -p data/{historical,results,logs}
mkdir -p tests

# Create root level files
touch .env
cat > requirements.txt << EOF
python-dateutil
pandas
numpy
matplotlib
pytest
websocket-client
requests
jupyter
talib
pytz
EOF

cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="trading_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dateutil",
        "pandas",
        "numpy",
        "matplotlib",
    ],
)
EOF

cat > README.md << EOF
# Trading System

An automated trading system for Indian markets using Zerodha's Kite API.

## Setup

1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Configure your API keys in \`.env\`
4. Run authentication: \`python authenticate.py\`

## Components

- Backtesting
- Paper Trading
- Live Trading
- Strategy Optimization
EOF

touch authenticate.py
touch run_backtest.py
touch run_optimization.py
touch run_paper_trading.py
touch run_live_trading.py

# Create __init__.py in current directory
cat > __init__.py << EOF
"""Trading system package."""
__version__ = "0.1.0"
EOF

touch config.py
touch system.py

# Create module files
for module in auth data strategy risk execution utils; do
    cat > $module/__init__.py << EOF
"""$module module for the trading system."""
EOF
done

# Create specific module files
touch auth/zerodha_auth.py
touch data/historical_data.py
touch data/realtime_data.py
touch strategy/base_strategy.py
touch strategy/moving_average.py
touch strategy/rsi_strategy.py
touch risk/risk_manager.py
touch execution/order_manager.py
touch utils/logger.py
touch utils/helpers.py

# Create notebook files
touch notebooks/01_setup.ipynb
touch notebooks/02_data_analysis.ipynb
touch notebooks/03_backtest.ipynb
touch notebooks/04_optimization.ipynb
touch notebooks/05_paper_trading.ipynb

# Create test files
cat > tests/__init__.py << EOF
"""Test package for trading system."""
EOF

touch tests/test_auth.py
touch tests/test_data.py
touch tests/test_strategy.py
touch tests/test_risk.py
touch tests/test_execution.py
touch tests/test_system.py

echo "Project structure created successfully!"