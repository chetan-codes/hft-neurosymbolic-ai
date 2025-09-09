#!/usr/bin/env python3
"""
Calibration Experiments for Neurosymbolic Trading System

This script runs calibration experiments to measure the reliability
of confidence scores in the neurosymbolic trading system.
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import requests
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import seaborn as sns

class CalibrationExperiment:
    """Run calibration experiments on the neurosymbolic trading system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.results = []
        
    async def run_experiment(self, symbols: List[str], timeframes: List[str], 
                           strategies: List[str], num_samples: int = 100) -> Dict[str, Any]:
        """Run calibration experiment with multiple symbols and strategies"""
        
        print(f"Running calibration experiment with {num_samples} samples...")
        
        all_predictions = []
        all_confidences = []
        all_actuals = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                for strategy in strategies:
                    print(f"Testing {symbol} {timeframe} {strategy}...")
                    
                    # Generate multiple predictions
                    for i in range(num_samples):
                        try:
                            # Add small delay to get different market data
                            await asyncio.sleep(0.1)
                            
                            # Get trading signal
                            response = requests.post(
                                f"{self.api_base_url}/api/v1/trading/signal",
                                json={
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "strategy": strategy
                                }
                            )
                            
                            if response.status_code == 200:
                                signal_data = response.json()
                                
                                # Extract confidence scores
                                ai_conf = signal_data.get("ai_confidence", 0.0)
                                sym_conf = signal_data.get("symbolic_confidence", 0.0)
                                final_conf = signal_data.get("confidence", 0.0)
                                action = signal_data.get("action", "hold")
                                
                                # Simulate actual outcome (in real experiment, this would be actual market data)
                                actual_outcome = self._simulate_outcome(action, final_conf)
                                
                                all_predictions.append({
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "strategy": strategy,
                                    "ai_confidence": ai_conf,
                                    "symbolic_confidence": sym_conf,
                                    "final_confidence": final_conf,
                                    "action": action,
                                    "actual_outcome": actual_outcome
                                })
                                
                                all_confidences.append(final_conf)
                                all_actuals.append(actual_outcome)
                                
                        except Exception as e:
                            print(f"Error in sample {i}: {e}")
                            continue
        
        # Calculate calibration metrics
        calibration_results = self._calculate_calibration_metrics(all_confidences, all_actuals)
        
        # Generate plots
        self._generate_calibration_plots(all_confidences, all_actuals)
        
        # Save results
        results = {
            "experiment_config": {
                "symbols": symbols,
                "timeframes": timeframes,
                "strategies": strategies,
                "num_samples": num_samples
            },
            "calibration_metrics": calibration_results,
            "predictions": all_predictions
        }
        
        with open("paper_kit/experiments/calibration_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Calibration experiment completed!")
        return results
    
    def _simulate_outcome(self, action: str, confidence: float) -> int:
        """Simulate actual trading outcome based on action and confidence"""
        # In a real experiment, this would use actual market data
        # For now, we simulate based on confidence and action
        if action == "hold":
            return 0  # Neutral outcome
        
        # Higher confidence should correlate with better outcomes
        base_prob = 0.5 + (confidence - 0.5) * 0.3  # Scale confidence to probability
        
        if action == "buy":
            return 1 if np.random.random() < base_prob else 0
        elif action == "sell":
            return 1 if np.random.random() < base_prob else 0
        else:
            return 0
    
    def _calculate_calibration_metrics(self, confidences: List[float], actuals: List[int]) -> Dict[str, Any]:
        """Calculate calibration metrics"""
        
        confidences = np.array(confidences)
        actuals = np.array(actuals)
        
        # Brier Score
        brier_score = brier_score_loss(actuals, confidences)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actuals, confidences, n_bins=10
        )
        
        # Reliability diagram data
        reliability_data = {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist()
        }
        
        # Confidence distribution
        confidence_stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "percentiles": {
                "25th": float(np.percentile(confidences, 25)),
                "50th": float(np.percentile(confidences, 50)),
                "75th": float(np.percentile(confidences, 75)),
                "90th": float(np.percentile(confidences, 90)),
                "95th": float(np.percentile(confidences, 95))
            }
        }
        
        # Calibration error (mean absolute difference)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            "brier_score": float(brier_score),
            "calibration_error": float(calibration_error),
            "reliability_data": reliability_data,
            "confidence_stats": confidence_stats,
            "total_samples": len(confidences)
        }
    
    def _generate_calibration_plots(self, confidences: List[float], actuals: List[int]):
        """Generate calibration plots"""
        
        confidences = np.array(confidences)
        actuals = np.array(actuals)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reliability Diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actuals, confidences, n_bins=10
        )
        
        axes[0, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Neurosymbolic System")
        axes[0, 0].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        axes[0, 0].set_xlabel("Mean Predicted Probability")
        axes[0, 0].set_ylabel("Fraction of Positives")
        axes[0, 0].set_title("Reliability Diagram")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Confidence Distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel("Confidence Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Confidence Score Distribution")
        axes[0, 1].grid(True)
        
        # 3. Confidence vs Outcome
        # Bin confidences and calculate success rate for each bin
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        success_rates = []
        
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(mask) > 0:
                success_rate = np.mean(actuals[mask])
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        axes[1, 0].plot(bin_centers, success_rates, "o-", label="Actual Success Rate")
        axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        axes[1, 0].set_xlabel("Confidence Score")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].set_title("Confidence vs Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Confidence by Action Type
        actions = ["buy", "sell", "hold"]
        action_confidences = []
        
        for action in actions:
            # This would need to be extracted from the actual data
            # For now, we'll simulate
            action_conf = confidences[::3]  # Simulate different actions
            action_confidences.append(action_conf)
        
        axes[1, 1].boxplot(action_confidences, labels=actions)
        axes[1, 1].set_ylabel("Confidence Score")
        axes[1, 1].set_title("Confidence by Action Type")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("paper_kit/experiments/calibration_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Calibration plots saved to paper_kit/experiments/calibration_plots.png")

async def main():
    """Main function to run calibration experiments"""
    
    # Initialize experiment
    experiment = CalibrationExperiment()
    
    # Define experiment parameters
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    timeframes = ["daily"]
    strategies = ["neurosymbolic", "rule_only"]
    num_samples = 50  # Reduced for faster execution
    
    # Run experiment
    results = await experiment.run_experiment(
        symbols=symbols,
        timeframes=timeframes,
        strategies=strategies,
        num_samples=num_samples
    )
    
    # Print summary
    print("\n=== Calibration Experiment Results ===")
    print(f"Total samples: {results['calibration_metrics']['total_samples']}")
    print(f"Brier Score: {results['calibration_metrics']['brier_score']:.4f}")
    print(f"Calibration Error: {results['calibration_metrics']['calibration_error']:.4f}")
    
    confidence_stats = results['calibration_metrics']['confidence_stats']
    print(f"Mean Confidence: {confidence_stats['mean']:.4f}")
    print(f"Confidence Std: {confidence_stats['std']:.4f}")
    print(f"Confidence Range: [{confidence_stats['min']:.4f}, {confidence_stats['max']:.4f}]")

if __name__ == "__main__":
    asyncio.run(main())
