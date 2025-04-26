# Log Analyzer Module

## Overview

The Log Analyzer is a comprehensive tool for analyzing and visualizing trading bot logs. It parses different types of logs (error, strategy, API, trade, and performance) to generate insightful statistics and reports.

## Features

- **Error Log Analysis**: Identifies common error patterns, most frequent errors, and critical issues
- **Strategy Log Analysis**: Tracks strategy signal patterns, buy/sell ratios, and most active symbols
- **API Log Analysis**: Monitors API call success rates, most used endpoints, and common failures
- **Trade Log Analysis**: Analyzes trading activity, volumes, and performance by symbol
- **Performance Analysis**: Tracks account balance, PnL, and trading success rates
- **Visualization**: Generates performance charts and HTML reports
- **Comprehensive Reporting**: Creates detailed reports combining all analytics

## Usage

### Command Line Interface

The log analyzer can be run as a command-line tool:

```bash
python log_analyzer_cli.py --type all --days 7 --output both --format html
```

Arguments:
- `--type` or `-t`: Type of log to analyze (error, strategy, api, trade, performance, all)
- `--days` or `-d`: Number of days to look back (default: 1)
- `--output` or `-o`: Where to output results (console, file, both)
- `--format` or `-f`: Output format (text, html)
- `--log-dir`: Directory containing log files (default: logs)

### Within Python Code

You can also use the log analyzer programmatically:

```python
from utils.log_analyzer import LogAnalyzer

# Create analyzer
analyzer = LogAnalyzer(log_dir="logs")

# Generate a comprehensive report
report = analyzer.generate_comprehensive_report(days_back=7)
print(report)

# Analyze specific log types
error_stats = analyzer.analyze_error_log(days_back=1)
trade_stats = analyzer.analyze_trade_log(days_back=7)
api_stats = analyzer.analyze_api_log(days_back=3)

# Generate and save HTML report with charts
report_path = analyzer.save_comprehensive_report(days_back=7, include_charts=True)
```

### Quick Analysis Function

For quick analysis, use the convenience function:

```python
from utils.log_analyzer import analyze_logs

# Analyze all logs and print to console
analyze_logs(log_type="all", days_back=1, print_to_console=True)

# Analyze strategy logs and log to system log
analyze_logs(log_type="strategy", days_back=7, print_to_console=False)
```

## Log Files

The analyzer processes the following log files:

- `error.log`: Error messages and exceptions
- `critical.log`: Critical errors and system issues
- `strategy.log`: Strategy signals and decisions
- `api.log`: API calls, responses, and errors
- `trades.log`: Executed trades and orders
- `performance.log`: Account performance metrics

## Requirements

- Python 3.6+
- pandas
- matplotlib
- NumPy

## Integration

The Log Analyzer is designed to work with the trading bot's logging system. All logs are expected to be in JSON format for structured analysis. The module can be scheduled to run periodically or on-demand for monitoring trading bot performance.

## Output Examples

### Text Report Sample
```
================ TRADING BOT ANALYTICS REPORT ================
Generated on: 10.08.2023 - 15:30 Uhr
Period: Last 7 days
===========================================================

I. PERFORMANCE METRICS
=======================
Performance Analysis (Stand: 10.08.2023 - 15:30 Uhr, last 7 days):
- Current Balance: $10245.78
- Starting Balance: $10000.00
- Overall P&L: $245.78 (+2.46%)
- Overall Win Rate: 58.3%
...
```

### HTML Report
The HTML report includes the same information as the text report but also embeds performance charts for visual analysis. 