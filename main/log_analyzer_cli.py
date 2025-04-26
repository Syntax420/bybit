#!/usr/bin/env python
"""
Log Analyzer CLI - Command line interface for analyzing trading bot logs
"""

import sys
import argparse
import logging
from utils.log_analyzer import LogAnalyzer, analyze_logs
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('log_analyzer_cli')

def main():
    """Main function for the log analyzer CLI"""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Analyze trading bot logs')
    
    # Add arguments
    parser.add_argument(
        '--type', '-t',
        choices=['error', 'strategy', 'api', 'trade', 'performance', 'all'],
        default='all',
        help='Type of log to analyze (default: all)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=1,
        help='Number of days to look back (default: 1)'
    )
    
    parser.add_argument(
        '--output', '-o',
        choices=['console', 'file', 'both'],
        default='console',
        help='Where to output the analysis (default: console)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'html'],
        default='text',
        help='Output format (default: text, html includes charts)'
    )
    
    parser.add_argument(
        '--log-dir',
        default='logs',
        help='Directory containing log files (default: logs)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = LogAnalyzer(log_dir=args.log_dir)
        
        # Print to console
        if args.output in ['console', 'both']:
            logger.info(f"Analyzing {args.type} logs for the past {args.days} day(s)...")
            
            if args.type == 'all':
                summary = analyzer.generate_comprehensive_report(days_back=args.days)
            elif args.type == 'error':
                summary = analyzer.generate_summary(days_back=args.days)
            elif args.type == 'strategy':
                summary = analyzer.generate_strategy_summary(days_back=args.days)
            elif args.type == 'api':
                summary = analyzer.generate_api_summary(days_back=args.days)
            elif args.type == 'trade':
                summary = analyzer.generate_trade_summary(days_back=args.days)
            elif args.type == 'performance':
                summary = analyzer.generate_performance_summary(days_back=args.days)
            
            print("\n" + summary + "\n")
            
        # Save to file
        if args.output in ['file', 'both']:
            logger.info(f"Saving {args.type} logs analysis to file...")
            
            # Create reports directory if it doesn't exist
            reports_dir = 'reports'
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            if args.format == 'html' and args.type in ['all', 'performance']:
                # Save HTML report with charts
                report_path = analyzer.save_comprehensive_report(
                    days_back=args.days, 
                    include_charts=True
                )
                logger.info(f"Report saved to: {report_path}")
            else:
                # Get the summary
                if args.type == 'all':
                    summary = analyzer.generate_comprehensive_report(days_back=args.days)
                elif args.type == 'error':
                    summary = analyzer.generate_summary(days_back=args.days)
                elif args.type == 'strategy':
                    summary = analyzer.generate_strategy_summary(days_back=args.days)
                elif args.type == 'api':
                    summary = analyzer.generate_api_summary(days_back=args.days)
                elif args.type == 'trade':
                    summary = analyzer.generate_trade_summary(days_back=args.days)
                elif args.type == 'performance':
                    summary = analyzer.generate_performance_summary(days_back=args.days)
                
                # Save to file
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"{reports_dir}/log_analysis_{args.type}_{timestamp}.txt"
                
                with open(report_file, 'w') as f:
                    f.write(summary)
                
                logger.info(f"Report saved to: {report_file}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 