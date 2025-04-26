#!/usr/bin/env python
"""
Test script for the enhanced log analyzer module
"""

import os
import sys
import logging
import argparse
from utils.log_analyzer import LogAnalyzer, analyze_logs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_log_analyzer')

def test_error_analysis():
    """Test error log analysis functionality"""
    logger.info("Testing error log analysis...")
    
    analyzer = LogAnalyzer()
    summary = analyzer.generate_summary(days_back=1)
    
    print("\n=== ERROR LOG ANALYSIS ===")
    print(summary)
    
    return True

def test_strategy_analysis():
    """Test strategy log analysis functionality"""
    logger.info("Testing strategy log analysis...")
    
    analyzer = LogAnalyzer()
    summary = analyzer.generate_strategy_summary(days_back=1)
    
    print("\n=== STRATEGY LOG ANALYSIS ===")
    print(summary)
    
    return True

def test_api_analysis():
    """Test API log analysis functionality"""
    logger.info("Testing API log analysis...")
    
    analyzer = LogAnalyzer()
    summary = analyzer.generate_api_summary(days_back=1)
    
    print("\n=== API LOG ANALYSIS ===")
    print(summary)
    
    return True

def test_trade_analysis():
    """Test trade log analysis functionality"""
    logger.info("Testing trade log analysis...")
    
    analyzer = LogAnalyzer()
    summary = analyzer.generate_trade_summary(days_back=7)
    
    print("\n=== TRADE LOG ANALYSIS ===")
    print(summary)
    
    return True

def test_performance_analysis():
    """Test performance log analysis functionality"""
    logger.info("Testing performance log analysis...")
    
    analyzer = LogAnalyzer()
    summary = analyzer.generate_performance_summary(days_back=30)
    
    print("\n=== PERFORMANCE LOG ANALYSIS ===")
    print(summary)
    
    return True

def test_comprehensive_report():
    """Test comprehensive report generation"""
    logger.info("Testing comprehensive report generation...")
    
    analyzer = LogAnalyzer()
    
    # Generate text report
    summary = analyzer.generate_comprehensive_report(days_back=7)
    
    print("\n=== COMPREHENSIVE REPORT ===")
    print(summary)
    
    # Generate HTML report with charts
    try:
        report_path = analyzer.save_comprehensive_report(days_back=7, include_charts=True)
        print(f"\nHTML report saved to: {report_path}")
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
    
    return True

def main():
    """Main function for testing the log analyzer"""
    parser = argparse.ArgumentParser(description='Test the enhanced log analyzer module')
    
    parser.add_argument(
        '--test', '-t',
        choices=['error', 'strategy', 'api', 'trade', 'performance', 'comprehensive', 'all'],
        default='all',
        help='Which test to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_funcs = {
        'error': test_error_analysis,
        'strategy': test_strategy_analysis,
        'api': test_api_analysis,
        'trade': test_trade_analysis,
        'performance': test_performance_analysis,
        'comprehensive': test_comprehensive_report
    }
    
    # Run selected tests
    if args.test == 'all':
        print("\n=== RUNNING ALL TESTS ===\n")
        for test_name, test_func in test_funcs.items():
            print(f"\n--- Running {test_name} test ---")
            try:
                test_func()
                print(f"✓ {test_name} test completed successfully")
            except Exception as e:
                print(f"✗ {test_name} test failed: {str(e)}")
    else:
        # Run single test
        test_func = test_funcs.get(args.test)
        if test_func:
            try:
                test_func()
                print(f"\n✓ {args.test} test completed successfully")
            except Exception as e:
                print(f"\n✗ {args.test} test failed: {str(e)}")
        else:
            print(f"Unknown test: {args.test}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 