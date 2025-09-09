#!/usr/bin/env python3
"""
Quick Load Test Script for HFT Neurosymbolic AI System
Simple script to test system performance under load
"""

import asyncio
import time
import httpx
import json
from datetime import datetime
from typing import List, Dict, Any

async def quick_load_test(api_url: str = "http://localhost:8000", 
                         num_requests: int = 50, 
                         concurrent: int = 10):
    """Run a quick load test"""
    
    print(f"🚀 Starting quick load test: {num_requests} requests, {concurrent} concurrent")
    print(f"API URL: {api_url}")
    
    # Test symbols and strategies
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
    strategies = ["neurosymbolic", "rule_only"]
    timeframes = ["daily", "hourly"]
    
    results = []
    start_time = time.time()
    
    async def make_request(session, symbol, strategy, timeframe):
        """Make a single request"""
        try:
            request_start = time.time()
            response = await session.post(
                f"{api_url}/api/v1/trading/signal",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy
                },
                timeout=10.0
            )
            request_time = time.time() - request_start
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "symbol": symbol,
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "total_time": request_time,
                    "ai_time": data.get("ai_prediction", {}).get("prediction_time_ms", 0) / 1000.0,
                    "symbolic_time": data.get("symbolic_analysis", {}).get("reasoning_time_ms", 0) / 1000.0,
                    "confidence": data.get("confidence", 0),
                    "action": data.get("signal", {}).get("action", "unknown")
                }
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "error": f"HTTP {response.status_code}",
                    "total_time": request_time
                }
        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "strategy": strategy,
                "timeframe": timeframe,
                "error": str(e),
                "total_time": time.time() - request_start
            }
    
    # Generate test cases
    test_cases = []
    for i in range(num_requests):
        symbol = symbols[i % len(symbols)]
        strategy = strategies[i % len(strategies)]
        timeframe = timeframes[i % len(timeframes)]
        test_cases.append((symbol, strategy, timeframe))
    
    # Run concurrent requests
    semaphore = asyncio.Semaphore(concurrent)
    
    async def run_with_semaphore(session, symbol, strategy, timeframe):
        async with semaphore:
            return await make_request(session, symbol, strategy, timeframe)
    
    async with httpx.AsyncClient() as client:
        tasks = [run_with_semaphore(client, symbol, strategy, timeframe) 
                for symbol, strategy, timeframe in test_cases]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    if successful:
        total_times = [r["total_time"] for r in successful]
        ai_times = [r["ai_time"] for r in successful]
        symbolic_times = [r["symbolic_time"] for r in successful]
        
        print("\n" + "="*50)
        print("QUICK LOAD TEST RESULTS")
        print("="*50)
        print(f"Total Requests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success Rate: {len(successful)/len(results):.1%}")
        print(f"Total Test Time: {total_time:.2f}s")
        print(f"Throughput: {len(successful)/total_time:.1f} req/s")
        
        print(f"\nResponse Times:")
        print(f"  Average: {sum(total_times)/len(total_times):.3f}s")
        print(f"  Min: {min(total_times):.3f}s")
        print(f"  Max: {max(total_times):.3f}s")
        print(f"  P95: {sorted(total_times)[int(len(total_times)*0.95)]:.3f}s")
        
        print(f"\nComponent Times:")
        print(f"  AI Engine: {sum(ai_times)/len(ai_times):.3f}s avg")
        print(f"  Symbolic Reasoner: {sum(symbolic_times)/len(symbolic_times):.3f}s avg")
        
        # Strategy breakdown
        print(f"\nBy Strategy:")
        for strategy in strategies:
            strategy_results = [r for r in successful if r["strategy"] == strategy]
            if strategy_results:
                avg_time = sum(r["total_time"] for r in strategy_results) / len(strategy_results)
                print(f"  {strategy}: {avg_time:.3f}s avg ({len(strategy_results)} requests)")
        
        # Symbol breakdown
        print(f"\nBy Symbol:")
        for symbol in symbols:
            symbol_results = [r for r in successful if r["symbol"] == symbol]
            if symbol_results:
                avg_time = sum(r["total_time"] for r in symbol_results) / len(symbol_results)
                print(f"  {symbol}: {avg_time:.3f}s avg ({len(symbol_results)} requests)")
    
    if failed:
        print(f"\nFailed Requests:")
        error_counts = {}
        for result in failed:
            error = result.get("error", "Unknown")
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"  {error}: {count} times")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_load_test_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            "test_config": {
                "num_requests": num_requests,
                "concurrent": concurrent,
                "api_url": api_url
            },
            "results": results,
            "summary": {
                "total_time": total_time,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful)/len(results) if results else 0
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return results

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_requests = 50
    concurrent = 10
    api_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        num_requests = int(sys.argv[1])
    if len(sys.argv) > 2:
        concurrent = int(sys.argv[2])
    if len(sys.argv) > 3:
        api_url = sys.argv[3]
    
    asyncio.run(quick_load_test(api_url, num_requests, concurrent))
