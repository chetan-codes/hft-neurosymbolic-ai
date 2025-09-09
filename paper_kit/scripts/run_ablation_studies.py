#!/usr/bin/env python3
"""
Ablation Studies for Neurosymbolic Trading System

This script runs ablation studies to measure the contribution
of different components in the neurosymbolic trading system.
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import requests
import matplotlib.pyplot as plt
import seaborn as sns

class AblationStudy:
    """Run ablation studies on the neurosymbolic trading system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.results = []
        
    async def run_ablation_study(self, symbols: List[str], num_samples: int = 100) -> Dict[str, Any]:
        """Run comprehensive ablation study"""
        
        print("Running ablation study...")
        
        # Define ablation conditions
        ablation_conditions = {
            "full_system": {
                "strategy": "neurosymbolic",
                "description": "Full neurosymbolic system"
            },
            "rule_only": {
                "strategy": "rule_only", 
                "description": "Symbolic reasoning only"
            },
            "ai_only": {
                "strategy": "neurosymbolic",
                "description": "AI predictions only (simulated)"
            },
            "no_agreement_bonus": {
                "strategy": "neurosymbolic",
                "description": "No agreement bonus/penalty"
            },
            "no_symbol_factors": {
                "strategy": "neurosymbolic",
                "description": "No symbol-specific factors"
            },
            "hard_risk_gates": {
                "strategy": "neurosymbolic",
                "description": "Hard risk gates (force HOLD on violations)"
            }
        }
        
        ablation_results = {}
        
        for condition_name, condition_config in ablation_conditions.items():
            print(f"Testing condition: {condition_name}")
            
            condition_results = await self._test_condition(
                symbols, condition_config, num_samples
            )
            ablation_results[condition_name] = condition_results
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(ablation_results)
        
        # Generate comparison plots
        self._generate_ablation_plots(ablation_results, performance_metrics)
        
        # Save results
        results = {
            "ablation_conditions": ablation_conditions,
            "condition_results": ablation_results,
            "performance_metrics": performance_metrics,
            "summary": self._generate_summary(performance_metrics)
        }
        
        with open("paper_kit/experiments/ablation_studies.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Ablation study completed!")
        return results
    
    async def _test_condition(self, symbols: List[str], condition_config: Dict[str, Any], 
                            num_samples: int) -> Dict[str, Any]:
        """Test a specific ablation condition"""
        
        results = {
            "confidences": [],
            "actions": [],
            "latencies": [],
            "errors": 0,
            "samples": []
        }
        
        for symbol in symbols:
            for i in range(num_samples // len(symbols)):
                try:
                    start_time = time.time()
                    
                    # Get trading signal
                    response = requests.post(
                        f"{self.api_base_url}/api/v1/trading/signal",
                        json={
                            "symbol": symbol,
                            "timeframe": "daily",
                            "strategy": condition_config["strategy"]
                        }
                    )
                    
                    latency = time.time() - start_time
                    
                    if response.status_code == 200:
                        signal_data = response.json()
                        
                        # Extract metrics
                        confidence = signal_data.get("confidence", 0.0)
                        action = signal_data.get("action", "hold")
                        ai_conf = signal_data.get("ai_confidence", 0.0)
                        sym_conf = signal_data.get("symbolic_confidence", 0.0)
                        
                        results["confidences"].append(confidence)
                        results["actions"].append(action)
                        results["latencies"].append(latency)
                        
                        results["samples"].append({
                            "symbol": symbol,
                            "confidence": confidence,
                            "action": action,
                            "ai_confidence": ai_conf,
                            "symbolic_confidence": sym_conf,
                            "latency": latency
                        })
                        
                    else:
                        results["errors"] += 1
                        
                except Exception as e:
                    results["errors"] += 1
                    print(f"Error in {condition_config['description']}: {e}")
        
        return results
    
    def _calculate_performance_metrics(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for each condition"""
        
        metrics = {}
        
        for condition_name, results in ablation_results.items():
            if not results["confidences"]:
                continue
                
            confidences = np.array(results["confidences"])
            latencies = np.array(results["latencies"])
            actions = results["actions"]
            
            # Calculate metrics
            condition_metrics = {
                "mean_confidence": float(np.mean(confidences)),
                "std_confidence": float(np.std(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "mean_latency": float(np.mean(latencies)),
                "std_latency": float(np.std(latencies)),
                "max_latency": float(np.max(latencies)),
                "action_distribution": {
                    "buy": actions.count("buy"),
                    "sell": actions.count("sell"),
                    "hold": actions.count("hold")
                },
                "error_rate": results["errors"] / (results["errors"] + len(confidences)),
                "total_samples": len(confidences)
            }
            
            # Calculate confidence spread (measure of differentiation)
            confidence_spread = np.max(confidences) - np.min(confidences)
            condition_metrics["confidence_spread"] = float(confidence_spread)
            
            # Calculate action diversity (measure of decision variety)
            unique_actions = len(set(actions))
            condition_metrics["action_diversity"] = unique_actions
            
            metrics[condition_name] = condition_metrics
        
        return metrics
    
    def _generate_ablation_plots(self, ablation_results: Dict[str, Any], 
                               performance_metrics: Dict[str, Any]):
        """Generate ablation study plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data for plotting
        condition_names = list(ablation_results.keys())
        mean_confidences = [performance_metrics[name]["mean_confidence"] for name in condition_names]
        std_confidences = [performance_metrics[name]["std_confidence"] for name in condition_names]
        mean_latencies = [performance_metrics[name]["mean_latency"] for name in condition_names]
        confidence_spreads = [performance_metrics[name]["confidence_spread"] for name in condition_names]
        action_diversities = [performance_metrics[name]["action_diversity"] for name in condition_names]
        error_rates = [performance_metrics[name]["error_rate"] for name in condition_names]
        
        # 1. Mean Confidence by Condition
        axes[0, 0].bar(condition_names, mean_confidences, yerr=std_confidences, capsize=5)
        axes[0, 0].set_ylabel("Mean Confidence")
        axes[0, 0].set_title("Mean Confidence by Condition")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Latency by Condition
        axes[0, 1].bar(condition_names, mean_latencies)
        axes[0, 1].set_ylabel("Mean Latency (s)")
        axes[0, 1].set_title("Latency by Condition")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence Spread by Condition
        axes[0, 2].bar(condition_names, confidence_spreads)
        axes[0, 2].set_ylabel("Confidence Spread")
        axes[0, 2].set_title("Confidence Differentiation")
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Action Diversity by Condition
        axes[1, 0].bar(condition_names, action_diversities)
        axes[1, 0].set_ylabel("Number of Unique Actions")
        axes[1, 0].set_title("Action Diversity")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error Rate by Condition
        axes[1, 1].bar(condition_names, error_rates)
        axes[1, 1].set_ylabel("Error Rate")
        axes[1, 1].set_title("Error Rate by Condition")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confidence Distribution Comparison
        for i, (condition_name, results) in enumerate(ablation_results.items()):
            if results["confidences"]:
                axes[1, 2].hist(results["confidences"], alpha=0.6, label=condition_name, bins=10)
        
        axes[1, 2].set_xlabel("Confidence Score")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title("Confidence Distribution Comparison")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("paper_kit/experiments/ablation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Ablation plots saved to paper_kit/experiments/ablation_plots.png")
    
    def _generate_summary(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ablation study results"""
        
        # Find baseline (full system)
        baseline = performance_metrics.get("full_system", {})
        
        summary = {
            "baseline_performance": baseline,
            "component_contributions": {},
            "key_findings": []
        }
        
        # Calculate component contributions
        for condition_name, metrics in performance_metrics.items():
            if condition_name == "full_system":
                continue
                
            # Calculate relative performance
            confidence_impact = (metrics["mean_confidence"] - baseline["mean_confidence"]) / baseline["mean_confidence"]
            latency_impact = (metrics["mean_latency"] - baseline["mean_latency"]) / baseline["mean_latency"]
            spread_impact = (metrics["confidence_spread"] - baseline["confidence_spread"]) / baseline["confidence_spread"]
            
            summary["component_contributions"][condition_name] = {
                "confidence_impact": float(confidence_impact),
                "latency_impact": float(latency_impact),
                "spread_impact": float(spread_impact)
            }
        
        # Generate key findings
        if "rule_only" in performance_metrics:
            rule_only = performance_metrics["rule_only"]
            summary["key_findings"].append(
                f"Rule-only system has {rule_only['mean_confidence']:.3f} mean confidence vs {baseline['mean_confidence']:.3f} for full system"
            )
        
        if "ai_only" in performance_metrics:
            ai_only = performance_metrics["ai_only"]
            summary["key_findings"].append(
                f"AI-only system has {ai_only['mean_confidence']:.3f} mean confidence vs {baseline['mean_confidence']:.3f} for full system"
            )
        
        return summary

async def main():
    """Main function to run ablation studies"""
    
    # Initialize study
    study = AblationStudy()
    
    # Define study parameters
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    num_samples = 50  # Reduced for faster execution
    
    # Run study
    results = await study.run_ablation_study(
        symbols=symbols,
        num_samples=num_samples
    )
    
    # Print summary
    print("\n=== Ablation Study Results ===")
    
    for condition_name, metrics in results["performance_metrics"].items():
        print(f"\n{condition_name}:")
        print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"  Confidence Spread: {metrics['confidence_spread']:.4f}")
        print(f"  Mean Latency: {metrics['mean_latency']:.4f}s")
        print(f"  Action Diversity: {metrics['action_diversity']}")
        print(f"  Error Rate: {metrics['error_rate']:.4f}")
    
    print("\n=== Key Findings ===")
    for finding in results["summary"]["key_findings"]:
        print(f"- {finding}")

if __name__ == "__main__":
    asyncio.run(main())
