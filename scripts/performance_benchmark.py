#!/usr/bin/env python3
"""
Performance Benchmark and Load Testing Script
Tests the HFT Neurosymbolic AI System under various loads and measures latency
"""

import asyncio
import time
import json
import statistics
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance testing for HFT Neurosymbolic AI System"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results = []
        self.timing_data = {}
        
    async def single_request_timing(self, symbol: str, timeframe: str, strategy: str) -> Dict[str, Any]:
        """Measure timing for a single request with detailed breakdown"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Start timing
                request_start = time.time()
                
                # Make the request
                response = await client.post(
                    f"{self.api_base_url}/api/v1/trading/signal",
                    json={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy
                    }
                )
                
                request_end = time.time()
                request_duration = request_end - request_start
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract timing information from response
                    signal_time = data.get("signal", {}).get("signal_time_ms", 0) / 1000.0
                    ai_time = data.get("ai_prediction", {}).get("prediction_time_ms", 0) / 1000.0
                    symbolic_time = data.get("symbolic_analysis", {}).get("reasoning_time_ms", 0) / 1000.0
                    
                    total_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "total_time": total_time,
                        "request_duration": request_duration,
                        "signal_time": signal_time,
                        "ai_time": ai_time,
                        "symbolic_time": symbolic_time,
                        "confidence": data.get("confidence", 0),
                        "action": data.get("signal", {}).get("action", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "error": f"HTTP {response.status_code}",
                        "total_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "error": str(e),
                "total_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def load_test(self, symbols: List[str], timeframes: List[str], 
                       strategies: List[str], concurrent_requests: int = 10) -> List[Dict[str, Any]]:
        """Run load test with concurrent requests"""
        logger.info(f"Starting load test: {len(symbols)} symbols × {len(timeframes)} timeframes × {len(strategies)} strategies")
        logger.info(f"Concurrent requests: {concurrent_requests}")
        
        # Generate all combinations
        test_cases = []
        for symbol in symbols:
            for timeframe in timeframes:
                for strategy in strategies:
                    test_cases.append((symbol, timeframe, strategy))
        
        logger.info(f"Total test cases: {len(test_cases)}")
        
        # Run concurrent requests
        results = []
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def run_single_test(symbol, timeframe, strategy):
            async with semaphore:
                return await self.single_request_timing(symbol, timeframe, strategy)
        
        # Execute all tests
        tasks = [run_single_test(symbol, timeframe, strategy) for symbol, timeframe, strategy in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to list
        valid_results = [r for r in results if isinstance(r, dict)]
        self.results.extend(valid_results)
        
        logger.info(f"Completed load test: {len(valid_results)} successful requests")
        return valid_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze performance results and generate statistics"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame(self.results)
        successful_results = df[df['success'] == True]
        
        if successful_results.empty:
            return {"error": "No successful results to analyze"}
        
        # Calculate statistics
        stats = {
            "total_requests": len(df),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(df),
            "timing_stats": {
                "total_time": {
                    "mean": successful_results['total_time'].mean(),
                    "median": successful_results['total_time'].median(),
                    "std": successful_results['total_time'].std(),
                    "min": successful_results['total_time'].min(),
                    "max": successful_results['total_time'].max(),
                    "p95": successful_results['total_time'].quantile(0.95),
                    "p99": successful_results['total_time'].quantile(0.99)
                },
                "ai_time": {
                    "mean": successful_results['ai_time'].mean(),
                    "median": successful_results['ai_time'].median(),
                    "std": successful_results['ai_time'].std(),
                    "p95": successful_results['ai_time'].quantile(0.95)
                },
                "symbolic_time": {
                    "mean": successful_results['symbolic_time'].mean(),
                    "median": successful_results['symbolic_time'].median(),
                    "std": successful_results['symbolic_time'].std(),
                    "p95": successful_results['symbolic_time'].quantile(0.95)
                },
                "signal_time": {
                    "mean": successful_results['signal_time'].mean(),
                    "median": successful_results['signal_time'].median(),
                    "std": successful_results['signal_time'].std(),
                    "p95": successful_results['signal_time'].quantile(0.95)
                }
            },
            "throughput": {
                "requests_per_second": len(successful_results) / successful_results['total_time'].sum(),
                "avg_response_time": successful_results['total_time'].mean()
            },
            "by_strategy": {},
            "by_symbol": {},
            "by_timeframe": {}
        }
        
        # Analyze by strategy
        for strategy in successful_results['strategy'].unique():
            strategy_data = successful_results[successful_results['strategy'] == strategy]
            stats["by_strategy"][strategy] = {
                "count": len(strategy_data),
                "avg_total_time": strategy_data['total_time'].mean(),
                "avg_ai_time": strategy_data['ai_time'].mean(),
                "avg_symbolic_time": strategy_data['symbolic_time'].mean(),
                "p95_total_time": strategy_data['total_time'].quantile(0.95)
            }
        
        # Analyze by symbol
        for symbol in successful_results['symbol'].unique():
            symbol_data = successful_results[successful_results['symbol'] == symbol]
            stats["by_symbol"][symbol] = {
                "count": len(symbol_data),
                "avg_total_time": symbol_data['total_time'].mean(),
                "avg_ai_time": symbol_data['ai_time'].mean(),
                "avg_symbolic_time": symbol_data['symbolic_time'].mean(),
                "p95_total_time": symbol_data['total_time'].quantile(0.95)
            }
        
        # Analyze by timeframe
        for timeframe in successful_results['timeframe'].unique():
            timeframe_data = successful_results[successful_results['timeframe'] == timeframe]
            stats["by_timeframe"][timeframe] = {
                "count": len(timeframe_data),
                "avg_total_time": timeframe_data['total_time'].mean(),
                "avg_ai_time": timeframe_data['ai_time'].mean(),
                "avg_symbolic_time": timeframe_data['symbolic_time'].mean(),
                "p95_total_time": timeframe_data['total_time'].quantile(0.95)
            }
        
        return stats
    
    def generate_plots(self, output_dir: str = "benchmark_plots"):
        """Generate performance visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        successful_results = df[df['success'] == True]
        
        if successful_results.empty:
            logger.warning("No successful results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HFT Neurosymbolic AI Performance Benchmark', fontsize=16)
        
        # 1. Response time distribution
        axes[0, 0].hist(successful_results['total_time'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Total Response Time Distribution')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(successful_results['total_time'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # 2. AI vs Symbolic time scatter
        axes[0, 1].scatter(successful_results['ai_time'], successful_results['symbolic_time'], alpha=0.6)
        axes[0, 1].set_title('AI Time vs Symbolic Time')
        axes[0, 1].set_xlabel('AI Time (seconds)')
        axes[0, 1].set_ylabel('Symbolic Time (seconds)')
        
        # 3. Response time by strategy
        strategy_times = [successful_results[successful_results['strategy'] == s]['total_time'].values 
                         for s in successful_results['strategy'].unique()]
        axes[0, 2].boxplot(strategy_times, labels=successful_results['strategy'].unique())
        axes[0, 2].set_title('Response Time by Strategy')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Response time by symbol
        symbol_times = [successful_results[successful_results['symbol'] == s]['total_time'].values 
                       for s in successful_results['symbol'].unique()]
        axes[1, 0].boxplot(symbol_times, labels=successful_results['symbol'].unique())
        axes[1, 0].set_title('Response Time by Symbol')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Component timing breakdown
        component_times = ['ai_time', 'symbolic_time', 'signal_time']
        component_means = [successful_results[col].mean() for col in component_times]
        axes[1, 1].bar(component_times, component_means, color=['lightcoral', 'lightgreen', 'lightblue'])
        axes[1, 1].set_title('Average Component Timing')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Throughput over time
        successful_results['timestamp'] = pd.to_datetime(successful_results['timestamp'])
        successful_results = successful_results.sort_values('timestamp')
        successful_results['cumulative_requests'] = range(1, len(successful_results) + 1)
        successful_results['elapsed_time'] = (successful_results['timestamp'] - successful_results['timestamp'].iloc[0]).dt.total_seconds()
        successful_results['throughput'] = successful_results['cumulative_requests'] / successful_results['elapsed_time']
        
        axes[1, 2].plot(successful_results['elapsed_time'], successful_results['throughput'])
        axes[1, 2].set_title('Throughput Over Time')
        axes[1, 2].set_xlabel('Elapsed Time (seconds)')
        axes[1, 2].set_ylabel('Requests per Second')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_benchmark.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {output_dir}/performance_benchmark.png")
    
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save raw results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save analysis
        analysis = self.analyze_results()
        analysis_filename = filename.replace('.json', '_analysis.json')
        with open(analysis_filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        logger.info(f"Analysis saved to {analysis_filename}")
        
        return filename, analysis_filename

async def run_comprehensive_benchmark():
    """Run a comprehensive performance benchmark"""
    benchmark = PerformanceBenchmark()
    
    # Test configurations
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]
    timeframes = ["daily", "hourly", "minute"]
    strategies = ["neurosymbolic", "rule_only", "momentum", "mean_reversion"]
    
    logger.info("Starting comprehensive performance benchmark...")
    
    # Test 1: Single request timing
    logger.info("Test 1: Single request timing")
    single_result = await benchmark.single_request_timing("AAPL", "daily", "neurosymbolic")
    logger.info(f"Single request result: {single_result}")
    
    # Test 2: Light load test
    logger.info("Test 2: Light load test (5 concurrent requests)")
    await benchmark.load_test(symbols[:3], timeframes[:2], strategies[:2], concurrent_requests=5)
    
    # Test 3: Medium load test
    logger.info("Test 3: Medium load test (10 concurrent requests)")
    await benchmark.load_test(symbols, timeframes, strategies[:2], concurrent_requests=10)
    
    # Test 4: Heavy load test
    logger.info("Test 4: Heavy load test (20 concurrent requests)")
    await benchmark.load_test(symbols, timeframes, strategies, concurrent_requests=20)
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = benchmark.analyze_results()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Requests: {analysis['total_requests']}")
    print(f"Successful Requests: {analysis['successful_requests']}")
    print(f"Success Rate: {analysis['success_rate']:.2%}")
    print(f"Average Response Time: {analysis['timing_stats']['total_time']['mean']:.3f}s")
    print(f"95th Percentile: {analysis['timing_stats']['total_time']['p95']:.3f}s")
    print(f"99th Percentile: {analysis['timing_stats']['total_time']['p99']:.3f}s")
    print(f"Throughput: {analysis['throughput']['requests_per_second']:.2f} req/s")
    
    print("\nComponent Timing:")
    print(f"  AI Engine: {analysis['timing_stats']['ai_time']['mean']:.3f}s (p95: {analysis['timing_stats']['ai_time']['p95']:.3f}s)")
    print(f"  Symbolic Reasoner: {analysis['timing_stats']['symbolic_time']['mean']:.3f}s (p95: {analysis['timing_stats']['symbolic_time']['p95']:.3f}s)")
    print(f"  Trading Engine: {analysis['timing_stats']['signal_time']['mean']:.3f}s (p95: {analysis['timing_stats']['signal_time']['p95']:.3f}s)")
    
    # Generate plots
    benchmark.generate_plots()
    
    # Save results
    benchmark.save_results()
    
    return analysis

if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
