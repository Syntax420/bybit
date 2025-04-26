import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
from utils.data_storage import load_trades_from_csv, get_trade_statistics

def generate_statistics_report(month: Optional[str] = None, history_dir: str = "trade_history"):
    """
    Generate and display a statistics report for trades
    
    Args:
        month: Month in YYYY-MM format (None for all)
        history_dir: Directory for trade history
    """
    # Get trade statistics
    stats = get_trade_statistics(month)
    
    # No trades found
    if stats.get("total_trades", 0) == 0:
        print("No trades found for the specified period")
        return
    
    # Print report header
    period = f"Period: {month}" if month else "All time"
    print("\n" + "=" * 50)
    print(f"TRADE PERFORMANCE REPORT - {period}")
    print("=" * 50)
    
    # Print general statistics
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Winning Trades: {stats['winning_trades']} ({stats['win_rate']:.2f}%)")
    print(f"Losing Trades: {stats['losing_trades']} ({100 - stats['win_rate']:.2f}%)")
    print("-" * 50)
    
    # Print profit/loss statistics
    print(f"Total Profit: {stats['total_profit']:.2f} USDT")
    print(f"Total Loss: {stats['total_loss']:.2f} USDT")
    print(f"Net Profit: {stats['net_profit']:.2f} USDT")
    print("-" * 50)
    
    # Print additional metrics
    print(f"Average Profit per Winning Trade: {stats['avg_profit']:.2f} USDT")
    print(f"Average Loss per Losing Trade: {stats['avg_loss']:.2f} USDT")
    print(f"Max Profit Trade: {stats['max_profit']:.2f} USDT")
    print(f"Max Loss Trade: {stats['max_loss']:.2f} USDT")
    print("=" * 50)

def plot_performance(month: Optional[str] = None, history_dir: str = "trade_history", save_path: Optional[str] = None):
    """
    Plot trading performance
    
    Args:
        month: Month in YYYY-MM format (None for all)
        history_dir: Directory for trade history
        save_path: Path to save the plot (if None, display instead)
    """
    # Load trades
    trades = load_trades_from_csv(month, history_dir)
    
    if not trades:
        print("No trades found for the specified period")
        return
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(trades)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Convert numeric columns
    numeric_cols = ['price', 'quantity', 'pnl', 'leverage', 'entry_price', 'exit_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative PnL
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cumulative_pnl'], 'b-')
    plt.title('Cumulative Profit/Loss Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USDT)')
    plt.grid(True)
    
    # Plot individual trade PnLs
    plt.subplot(2, 1, 2)
    colors = ['g' if x >= 0 else 'r' for x in df['pnl']]
    plt.bar(df['timestamp'], df['pnl'], color=colors)
    plt.title('Individual Trade Profit/Loss')
    plt.xlabel('Date')
    plt.ylabel('PnL (USDT)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def generate_trade_report(trade_id: str, history_dir: str = "trade_history"):
    """
    Generate detailed report for a specific trade
    
    Args:
        trade_id: The ID of the trade
        history_dir: Directory for trade history
    """
    # Load all trades
    trades = load_trades_from_csv(None, history_dir)
    
    # Find the trade
    trade = None
    for t in trades:
        if t.get('trade_id') == trade_id:
            trade = t
            break
    
    if not trade:
        print(f"Trade with ID {trade_id} not found")
        return
    
    # Print trade details
    print("\n" + "=" * 50)
    print(f"DETAILED TRADE REPORT - ID: {trade_id}")
    print("=" * 50)
    
    # General information
    print(f"Symbol: {trade.get('symbol', 'N/A')}")
    print(f"Side: {trade.get('side', 'N/A')}")
    print(f"Date: {trade.get('timestamp', 'N/A')}")
    print(f"Status: {trade.get('status', 'N/A')}")
    print("-" * 50)
    
    # Price information
    print(f"Entry Price: {trade.get('entry_price', 'N/A')}")
    print(f"Exit Price: {trade.get('exit_price', 'N/A')}")
    print(f"Stop Loss: {trade.get('stop_loss', 'N/A')}")
    print(f"Take Profit: {trade.get('take_profit', 'N/A')}")
    print("-" * 50)
    
    # Position information
    print(f"Quantity: {trade.get('quantity', 'N/A')}")
    print(f"Leverage: {trade.get('leverage', 'N/A')}x")
    print("-" * 50)
    
    # Result information
    print(f"PnL: {trade.get('pnl', 'N/A')} USDT")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot Report Generator')
    parser.add_argument('--month', type=str, help='Month to analyze in YYYY-MM format')
    parser.add_argument('--stats', action='store_true', help='Generate statistics report')
    parser.add_argument('--plot', action='store_true', help='Generate performance plot')
    parser.add_argument('--trade-id', type=str, help='Generate report for specific trade ID')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    
    args = parser.parse_args()
    
    # Generate appropriate reports
    if args.stats:
        generate_statistics_report(args.month)
    
    if args.plot:
        plot_performance(args.month, save_path=args.save_plot)
    
    if args.trade_id:
        generate_trade_report(args.trade_id)
    
    # If no specific report is requested, generate all
    if not (args.stats or args.plot or args.trade_id):
        generate_statistics_report(args.month)
        plot_performance(args.month) 