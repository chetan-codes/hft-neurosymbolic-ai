#!/usr/bin/env python3
"""
Real-time performance monitor for continuous benchmarking
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
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
import threading
import queue

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    system: str
    latency_ms: float
    success: bool
    error_message: str = None

class RealTimePerformanceMonitor:
    def __init__(self, 
                 neurosymbolic_url="http://localhost:8000",
                 jena_url="http://localhost:3030/hft_jena",
                 monitoring_interval=1.0):
        self.neurosymbolic_url = neurosymbolic_url
        self.jena_url = jena_url
        self.monitoring_interval = monitoring_interval
        self.metrics_queue = queue.Queue()
        self.running = False
        self.metrics_history = []
        
    async def start_monitoring(self, duration_minutes=10):
        """Start real-time performance monitoring"""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting real-time monitoring for {duration_minutes} minutes...")
        
        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_neurosymbolic()),
            asyncio.create_task(self._monitor_jena()),
            asyncio.create_task(self._process_metrics()),
            asyncio.create_task(self._generate_realtime_report())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.running = False
            logger.info("Monitoring stopped")
    
    async def _monitor_neurosymbolic(self):
        """Monitor Neurosymbolic AI system performance"""
        while self.running:
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.neurosymbolic_url}/api/v1/trading/signal",
                        json={
                            "symbol": "AAPL",
                            "timeframe": "1m",
                            "strategy": "neurosymbolic"
                        },
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        success = True
                    else:
                        error_msg = f"HTTP {response.status_code}"
                        
            except Exception as e:
                error_msg = str(e)
            
            latency_ms = (time.time() - start_time) * 1000
            
            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                system="neurosymbolic",
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg
            )
            
            self.metrics_queue.put(metric)
            await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_jena(self):
        """Monitor Jena Fuseki performance"""
        sparql_query = """
        PREFIX hft: <http://hft.example.org/>
        SELECT ?symbol ?price WHERE {
            ?company hft:symbol ?symbol .
            ?company hft:price ?price .
        } LIMIT 10
        """
        
        while self.running:
            start_time = time.time()
            success = False
            error_msg = None
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.jena_url}/query",
                        data={"query": sparql_query, "output": "json"},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        success = True
                    else:
                        error_msg = f"HTTP {response.status_code}"
                        
            except Exception as e:
                error_msg = str(e)
            
            latency_ms = (time.time() - start_time) * 1000
            
            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                system="jena",
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg
            )
            
            self.metrics_queue.put(metric)
            await asyncio.sleep(self.monitoring_interval)
    
    async def _process_metrics(self):
        """Process metrics from queue"""
        while self.running:
            try:
                # Get metric from queue with timeout
                metric = self.metrics_queue.get(timeout=1.0)
                self.metrics_history.append(metric)
                
                # Keep only last 1000 metrics to prevent memory issues
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing metrics: {e}")
    
    async def _generate_realtime_report(self):
        """Generate real-time performance report every 30 seconds"""
        while self.running:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            if len(self.metrics_history) < 10:
                continue
            
            # Filter recent metrics (last 5 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=5)
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                continue
            
            # Separate by system
            neurosymbolic_metrics = [m for m in recent_metrics if m.system == "neurosymbolic"]
            jena_metrics = [m for m in recent_metrics if m.system == "jena"]
            
            if not neurosymbolic_metrics or not jena_metrics:
                continue
            
            # Calculate statistics
            ns_latencies = [m.latency_ms for m in neurosymbolic_metrics if m.success]
            jena_latencies = [m.latency_ms for m in jena_metrics if m.success]
            
            if not ns_latencies or not jena_latencies:
                continue
            
            # Calculate performance metrics
            ns_mean = np.mean(ns_latencies)
            jena_mean = np.mean(jena_latencies)
            improvement_factor = jena_mean / ns_mean if ns_mean > 0 else 0
            
            ns_success_rate = len(ns_latencies) / len(neurosymbolic_metrics)
            jena_success_rate = len(jena_latencies) / len(jena_metrics)
            
            # Print real-time stats
            print(f"\n{'='*60}")
            print(f"REAL-TIME PERFORMANCE REPORT - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Neurosymbolic AI: {ns_mean:.2f}ms avg, {ns_success_rate:.1%} success")
            print(f"Jena Fuseki:     {jena_mean:.2f}ms avg, {jena_success_rate:.1%} success")
            print(f"Improvement:     {improvement_factor:.2f}x faster")
            print(f"Sample size:     {len(ns_latencies)} vs {len(jena_latencies)}")
            print(f"{'='*60}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Separate by system
        neurosymbolic_metrics = [m for m in self.metrics_history if m.system == "neurosymbolic"]
        jena_metrics = [m for m in self.metrics_history if m.system == "jena"]
        
        # Filter successful requests
        ns_successful = [m for m in neurosymbolic_metrics if m.success]
        jena_successful = [m for m in jena_metrics if m.success]
        
        if not ns_successful or not jena_successful:
            return {"error": "Insufficient successful requests"}
        
        # Calculate statistics
        ns_latencies = [m.latency_ms for m in ns_successful]
        jena_latencies = [m.latency_ms for m in jena_successful]
        
        report = {
            "test_summary": {
                "start_time": self.metrics_history[0].timestamp.isoformat(),
                "end_time": self.metrics_history[-1].timestamp.isoformat(),
                "total_metrics": len(self.metrics_history),
                "monitoring_duration_minutes": (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp).total_seconds() / 60
            },
            "neurosymbolic_ai": {
                "total_requests": len(neurosymbolic_metrics),
                "successful_requests": len(ns_successful),
                "success_rate": len(ns_successful) / len(neurosymbolic_metrics),
                "mean_latency_ms": np.mean(ns_latencies),
                "median_latency_ms": np.median(ns_latencies),
                "p95_latency_ms": np.percentile(ns_latencies, 95),
                "p99_latency_ms": np.percentile(ns_latencies, 99),
                "std_latency_ms": np.std(ns_latencies),
                "min_latency_ms": np.min(ns_latencies),
                "max_latency_ms": np.max(ns_latencies)
            },
            "jena_fuseki": {
                "total_requests": len(jena_metrics),
                "successful_requests": len(jena_successful),
                "success_rate": len(jena_successful) / len(jena_metrics),
                "mean_latency_ms": np.mean(jena_latencies),
                "median_latency_ms": np.median(jena_latencies),
                "p95_latency_ms": np.percentile(jena_latencies, 95),
                "p99_latency_ms": np.percentile(jena_latencies, 99),
                "std_latency_ms": np.std(jena_latencies),
                "min_latency_ms": np.min(jena_latencies),
                "max_latency_ms": np.max(jena_latencies)
            }
        }
        
        # Calculate improvements
        ns_mean = report["neurosymbolic_ai"]["mean_latency_ms"]
        jena_mean = report["jena_fuseki"]["mean_latency_ms"]
        
        report["performance_comparison"] = {
            "latency_improvement_factor": jena_mean / ns_mean if ns_mean > 0 else 0,
            "latency_improvement_percent": ((jena_mean - ns_mean) / jena_mean * 100) if jena_mean > 0 else 0,
            "throughput_improvement_factor": report["neurosymbolic_ai"]["success_rate"] / report["jena_fuseki"]["success_rate"] if report["jena_fuseki"]["success_rate"] > 0 else 0
        }
        
        return report
    
    def create_performance_charts(self, output_dir="performance_charts"):
        """Create performance visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate metrics by system
        neurosymbolic_metrics = [m for m in self.metrics_history if m.system == "neurosymbolic" and m.success]
        jena_metrics = [m for m in self.metrics_history if m.system == "jena" and m.success]
        
        if not neurosymbolic_metrics or not jena_metrics:
            print("Insufficient data for charts")
            return
        
        # Create latency over time chart
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Latency over time
        plt.subplot(2, 2, 1)
        ns_times = [m.timestamp for m in neurosymbolic_metrics]
        ns_latencies = [m.latency_ms for m in neurosymbolic_metrics]
        jena_times = [m.timestamp for m in jena_metrics]
        jena_latencies = [m.latency_ms for m in jena_metrics]
        
        plt.plot(ns_times, ns_latencies, 'g-', alpha=0.7, label='Neurosymbolic AI')
        plt.plot(jena_times, jena_latencies, 'r-', alpha=0.7, label='Jena Fuseki')
        plt.title('Latency Over Time')
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 2: Latency distribution
        plt.subplot(2, 2, 2)
        plt.hist(ns_latencies, bins=20, alpha=0.7, label='Neurosymbolic AI', color='green')
        plt.hist(jena_latencies, bins=20, alpha=0.7, label='Jena Fuseki', color='red')
        plt.title('Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 3: Success rate over time
        plt.subplot(2, 2, 3)
        # Calculate success rate in 5-minute windows
        window_minutes = 5
        windows = []
        ns_success_rates = []
        jena_success_rates = []
        
        start_time = min(ns_times + jena_times)
        end_time = max(ns_times + jena_times)
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + timedelta(minutes=window_minutes)
            
            # Count successful requests in window
            ns_window = [m for m in neurosymbolic_metrics if current_time <= m.timestamp < window_end]
            jena_window = [m for m in jena_metrics if current_time <= m.timestamp < window_end]
            
            if ns_window:
                ns_success_rate = len([m for m in ns_window if m.success]) / len(ns_window)
                ns_success_rates.append(ns_success_rate)
            else:
                ns_success_rates.append(0)
            
            if jena_window:
                jena_success_rate = len([m for m in jena_window if m.success]) / len(jena_window)
                jena_success_rates.append(jena_success_rate)
            else:
                jena_success_rates.append(0)
            
            windows.append(current_time)
            current_time = window_end
        
        plt.plot(windows, ns_success_rates, 'g-', label='Neurosymbolic AI')
        plt.plot(windows, jena_success_rates, 'r-', label='Jena Fuseki')
        plt.title('Success Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 4: Performance comparison
        plt.subplot(2, 2, 4)
        systems = ['Neurosymbolic AI', 'Jena Fuseki']
        mean_latencies = [np.mean(ns_latencies), np.mean(jena_latencies)]
        colors = ['green', 'red']
        
        bars = plt.bar(systems, mean_latencies, color=colors, alpha=0.7)
        plt.title('Average Latency Comparison')
        plt.ylabel('Latency (ms)')
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/realtime_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance charts saved to {output_dir}/realtime_performance.png")

async def main():
    """Run real-time performance monitoring"""
    monitor = RealTimePerformanceMonitor()
    
    try:
        # Start monitoring for 10 minutes
        await monitor.start_monitoring(duration_minutes=10)
        
        # Generate final report
        report = monitor.generate_final_report()
        
        # Save report
        with open("realtime_performance_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create charts
        monitor.create_performance_charts()
        
        print("\n" + "="*60)
        print("REAL-TIME MONITORING COMPLETE")
        print("="*60)
        print(f"Latency Improvement: {report['performance_comparison']['latency_improvement_factor']:.2f}x")
        print(f"Success Rate Improvement: {report['performance_comparison']['throughput_improvement_factor']:.2f}x")
        print("Detailed report saved to realtime_performance_report.json")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Monitoring error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
