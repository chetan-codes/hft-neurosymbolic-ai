#!/usr/bin/env python3
"""
Comprehensive load testing framework for performance validation
"""

import asyncio
import time
import httpx
import json
import statistics
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
import concurrent.futures
from enum import Enum

class LoadLevel(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    PEAK = "peak"

@dataclass
class LoadTestResult:
    load_level: LoadLevel
    target_rps: int
    achieved_rps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float

class LoadTestingFramework:
    def __init__(self, 
                 neurosymbolic_url="http://localhost:8000",
                 jena_url="http://localhost:3030/hft_jena"):
        self.neurosymbolic_url = neurosymbolic_url
        self.jena_url = jena_url
        self.results = {}
        
    async def run_load_test(self, 
                           load_level: LoadLevel, 
                           duration_seconds: int = 60,
                           concurrent_users: int = 10) -> LoadTestResult:
        """Run load test for a specific load level"""
        
        # Define load levels
        load_configs = {
            LoadLevel.LIGHT: {"target_rps": 10, "concurrent_users": 5},
            LoadLevel.MEDIUM: {"target_rps": 50, "concurrent_users": 20},
            LoadLevel.HEAVY: {"target_rps": 100, "concurrent_users": 50},
            LoadLevel.PEAK: {"target_rps": 200, "concurrent_users": 100}
        }
        
        config = load_configs[load_level]
        target_rps = config["target_rps"]
        concurrent_users = min(concurrent_users, config["concurrent_users"])
        
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {load_level.value.upper()}")
        print(f"Target RPS: {target_rps}")
        print(f"Concurrent Users: {concurrent_users}")
        print(f"Duration: {duration_seconds}s")
        print(f"{'='*60}")
        
        # Run tests for both systems
        neurosymbolic_result = await self._run_system_load_test(
            "neurosymbolic", target_rps, duration_seconds, concurrent_users
        )
        
        jena_result = await self._run_system_load_test(
            "jena", target_rps, duration_seconds, concurrent_users
        )
        
        # Store results
        self.results[load_level] = {
            "neurosymbolic_ai": neurosymbolic_result,
            "jena_fuseki": jena_result,
            "improvement_factor": jena_result.achieved_rps / neurosymbolic_result.achieved_rps if neurosymbolic_result.achieved_rps > 0 else 0
        }
        
        return neurosymbolic_result, jena_result
    
    async def _run_system_load_test(self, 
                                   system: str, 
                                   target_rps: int, 
                                   duration_seconds: int,
                                   concurrent_users: int) -> LoadTestResult:
        """Run load test for a specific system"""
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Metrics collection
        latencies = []
        successes = 0
        failures = 0
        total_requests = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request():
            async with semaphore:
                request_start = time.time()
                success = False
                
                try:
                    if system == "neurosymbolic":
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{self.neurosymbolic_url}/api/v1/trading/signal",
                                json={
                                    "symbol": f"SYMBOL_{total_requests % 5}",
                                    "timeframe": "1m",
                                    "strategy": "neurosymbolic"
                                },
                                timeout=10.0
                            )
                            success = response.status_code == 200
                    
                    elif system == "jena":
                        sparql_query = """
                        PREFIX hft: <http://hft.example.org/>
                        SELECT ?symbol ?price WHERE {
                            ?company hft:symbol ?symbol .
                            ?company hft:price ?price .
                        } LIMIT 10
                        """
                        
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{self.jena_url}/query",
                                data={"query": sparql_query, "output": "json"},
                                headers={"Content-Type": "application/x-www-form-urlencoded"},
                                timeout=10.0
                            )
                            success = response.status_code == 200
                
                except Exception as e:
                    success = False
                
                latency_ms = (time.time() - request_start) * 1000
                return latency_ms, success
        
        # Main load testing loop
        tasks = []
        request_interval = 1.0 / target_rps if target_rps > 0 else 0.1
        
        while time.time() < end_time:
            # Create batch of requests
            batch_size = min(target_rps, 10)  # Limit batch size
            batch_tasks = []
            
            for _ in range(batch_size):
                if time.time() >= end_time:
                    break
                
                task = asyncio.create_task(make_request())
                batch_tasks.append(task)
                total_requests += 1
            
            if batch_tasks:
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, tuple):
                        latency, success = result
                        latencies.append(latency)
                        if success:
                            successes += 1
                        else:
                            failures += 1
                    else:
                        failures += 1
                        latencies.append(1000.0)  # Penalty for failed requests
            
            # Control request rate
            await asyncio.sleep(request_interval)
        
        # Calculate final metrics
        actual_duration = time.time() - start_time
        achieved_rps = total_requests / actual_duration if actual_duration > 0 else 0
        success_rate = successes / total_requests if total_requests > 0 else 0
        error_rate = failures / total_requests if total_requests > 0 else 0
        
        # Calculate latency percentiles
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        result = LoadTestResult(
            load_level=LoadLevel(system),  # This will be overridden
            target_rps=target_rps,
            achieved_rps=achieved_rps,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            success_rate=success_rate,
            error_rate=error_rate,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            duration_seconds=actual_duration
        )
        
        print(f"{system.upper()}: {achieved_rps:.1f} RPS, {avg_latency:.1f}ms avg, {success_rate:.1%} success")
        
        return result
    
    async def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing across all levels"""
        print("Starting comprehensive load testing...")
        
        # Test all load levels
        load_levels = [LoadLevel.LIGHT, LoadLevel.MEDIUM, LoadLevel.HEAVY, LoadLevel.PEAK]
        durations = [30, 60, 90, 120]  # Different durations for different levels
        
        for i, load_level in enumerate(load_levels):
            duration = durations[i]
            await self.run_load_test(load_level, duration)
            
            # Brief pause between tests
            await asyncio.sleep(10)
        
        # Generate comprehensive report
        report = self.generate_load_test_report()
        
        # Save results
        with open("load_test_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualizations
        self.create_load_test_charts()
        
        return report
    
    def generate_load_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report"""
        
        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_load_levels": len(self.results),
                "systems_tested": ["neurosymbolic_ai", "jena_fuseki"]
            },
            "load_test_results": {},
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Process results for each load level
        for load_level, results in self.results.items():
            ns_result = results["neurosymbolic_ai"]
            jena_result = results["jena_fuseki"]
            improvement = results["improvement_factor"]
            
            report["load_test_results"][load_level.value] = {
                "neurosymbolic_ai": {
                    "achieved_rps": ns_result.achieved_rps,
                    "avg_latency_ms": ns_result.avg_latency_ms,
                    "p95_latency_ms": ns_result.p95_latency_ms,
                    "p99_latency_ms": ns_result.p99_latency_ms,
                    "success_rate": ns_result.success_rate,
                    "total_requests": ns_result.total_requests
                },
                "jena_fuseki": {
                    "achieved_rps": jena_result.achieved_rps,
                    "avg_latency_ms": jena_result.avg_latency_ms,
                    "p95_latency_ms": jena_result.p95_latency_ms,
                    "p99_latency_ms": jena_result.p99_latency_ms,
                    "success_rate": jena_result.success_rate,
                    "total_requests": jena_result.total_requests
                },
                "improvement_factor": improvement
            }
        
        # Calculate overall performance comparison
        all_ns_rps = [results["neurosymbolic_ai"].achieved_rps for results in self.results.values()]
        all_jena_rps = [results["jena_fuseki"].achieved_rps for results in self.results.values()]
        
        report["performance_comparison"] = {
            "avg_improvement_factor": np.mean([results["improvement_factor"] for results in self.results.values()]),
            "max_improvement_factor": max([results["improvement_factor"] for results in self.results.values()]),
            "min_improvement_factor": min([results["improvement_factor"] for results in self.results.values()]),
            "neurosymbolic_avg_rps": np.mean(all_ns_rps),
            "jena_avg_rps": np.mean(all_jena_rps)
        }
        
        # Generate recommendations
        if report["performance_comparison"]["avg_improvement_factor"] > 2.0:
            report["recommendations"].append("Neurosymbolic AI shows significant performance advantages")
        
        if any(results["neurosymbolic_ai"].success_rate > 0.95 for results in self.results.values()):
            report["recommendations"].append("Neurosymbolic AI maintains high reliability under load")
        
        if any(results["neurosymbolic_ai"].p95_latency_ms < 100 for results in self.results.values()):
            report["recommendations"].append("Neurosymbolic AI meets low-latency requirements")
        
        return report
    
    def create_load_test_charts(self, output_dir="load_test_charts"):
        """Create load test visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        load_levels = list(self.results.keys())
        load_names = [level.value for level in load_levels]
        
        ns_rps = [self.results[level]["neurosymbolic_ai"].achieved_rps for level in load_levels]
        jena_rps = [self.results[level]["jena_fuseki"].achieved_rps for level in load_levels]
        
        ns_latency = [self.results[level]["neurosymbolic_ai"].avg_latency_ms for level in load_levels]
        jena_latency = [self.results[level]["jena_fuseki"].avg_latency_ms for level in load_levels]
        
        ns_success = [self.results[level]["neurosymbolic_ai"].success_rate for level in load_levels]
        jena_success = [self.results[level]["jena_fuseki"].success_rate for level in load_levels]
        
        # Create comprehensive chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: RPS Comparison
        x = np.arange(len(load_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ns_rps, width, label='Neurosymbolic AI', color='green', alpha=0.7)
        axes[0, 0].bar(x + width/2, jena_rps, width, label='Jena Fuseki', color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Load Level')
        axes[0, 0].set_ylabel('Achieved RPS')
        axes[0, 0].set_title('Throughput Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(load_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Latency Comparison
        axes[0, 1].bar(x - width/2, ns_latency, width, label='Neurosymbolic AI', color='green', alpha=0.7)
        axes[0, 1].bar(x + width/2, jena_latency, width, label='Jena Fuseki', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Load Level')
        axes[0, 1].set_ylabel('Average Latency (ms)')
        axes[0, 1].set_title('Latency Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(load_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Success Rate Comparison
        axes[1, 0].bar(x - width/2, ns_success, width, label='Neurosymbolic AI', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, jena_success, width, label='Jena Fuseki', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Load Level')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(load_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Improvement Factor
        improvement_factors = [self.results[level]["improvement_factor"] for level in load_levels]
        axes[1, 1].bar(x, improvement_factors, color='blue', alpha=0.7)
        axes[1, 1].set_xlabel('Load Level')
        axes[1, 1].set_ylabel('Improvement Factor (x)')
        axes[1, 1].set_title('Performance Improvement Factor')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(load_names)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (ns, jena) in enumerate(zip(ns_rps, jena_rps)):
            axes[0, 0].text(i, max(ns, jena) + 1, f'{ns:.1f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + width, max(ns, jena) + 1, f'{jena:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/load_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Load test charts saved to {output_dir}/load_test_comparison.png")

async def main():
    """Run comprehensive load testing"""
    framework = LoadTestingFramework()
    
    try:
        report = await framework.run_comprehensive_load_test()
        
        print("\n" + "="*60)
        print("LOAD TESTING COMPLETE")
        print("="*60)
        print(f"Average Improvement: {report['performance_comparison']['avg_improvement_factor']:.2f}x")
        print(f"Max Improvement: {report['performance_comparison']['max_improvement_factor']:.2f}x")
        print("Detailed report saved to load_test_results.json")
        
    except Exception as e:
        print(f"Load testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
