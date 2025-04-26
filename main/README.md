# Bybit Trading Bot

A customizable cryptocurrency trading bot for the Bybit exchange, supporting paper trading and live trading modes, with multiple strategy options.

## Features

- Paper trading mode for strategy testing without real money
- Live trading with real funds (use with caution!)
- Support for multiple technical analysis strategies
- Risk management controls
- Performance tracking and analysis
- Trailing stop-losses
- Multi-timeframe analysis
- Automatic symbol selection based on volume and volatility

## Installation

1. Clone or download the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure the `config.json` file with your Bybit API credentials and settings

## Configuration

Edit the `config.json` file to customize the bot's behavior:

- API credentials
- Trading parameters
- Risk management settings
- Strategy selection

## Usage

### Starting the Bot

Run the main script to start the bot:

```bash
python main.py
```

You'll be prompted to choose between paper trading or live trading mode.

### Command-Line Options

```bash
# Start in paper trading mode (non-interactive)
python main.py --paper

# Start in live trading mode (non-interactive, requires confirmation)
python main.py --live

# Use a custom config file
python main.py --config custom_config.json

# Add a symbol to the blacklist
python main.py --add-blacklist BTCUSDT

# Remove a symbol from the blacklist
python main.py --remove-blacklist BTCUSDT
```

### Testing API Connection

To test your Bybit API connection:

```bash
python test_api_connection.py
```

## Supported Strategies

- RSI-MACD (default)
- Donchian Channel

Configure the active strategy in `config.json`:

```json
"strategy": {
    "active": "rsi_macd",
    "parameters": {
        "rsi_macd": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30
            // more parameters...
        }
    }
}
```

## Paper Trading vs. Live Trading

### Paper Trading Mode

- No real trades are executed
- Uses testnet for data but simulates trades
- Great for testing strategies
- Collects data for strategy analysis

### Live Trading Mode

- Real trades with real money
- Uses your API credentials to execute actual trades
- Requires proper risk management settings
- Use with caution!

## Risk Management

Configure risk parameters in `config.json`:

```json
"risk_management": {
    "max_consecutive_losses": 3,
    "max_daily_loss_percent": 5.0,
    "max_open_positions": 5
}
```

## Performance Analysis

The bot logs trade data and performance metrics that can be analyzed using the built-in log analyzer.

## Disclaimer

Trading cryptocurrencies involves significant risk. This bot is provided for educational and research purposes only. Use at your own risk.

## License

MIT 