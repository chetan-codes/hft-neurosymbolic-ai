#!/usr/bin/env python3
"""
Quick Calibration Analysis Script
Simple script to analyze confidence calibration in the HFT Neurosymbolic AI System
"""

import asyncio
import time
import httpx
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

async def quick_calibration_test(api_url: str = "http://localhost:8000", 
                                num_samples: int = 100):
    """Run a quick calibration test"""
    
    print(f"🔍 Starting quick calibration test: {num_samples} samples")
    print(f"API URL: {api_url}")
    
    # Test symbols and strategies
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
    strategies = ["neurosymbolic", "rule_only"]
    
    results = []
    
    async def collect_sample(symbol, strategy):
        """Collect a single calibration sample"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{api_url}/api/v1/trading/signal",
                    json={
                        "symbol": symbol,
                        "timeframe": "daily",
                        "strategy": strategy
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract confidence scores
                    ai_confidence = data.get("ai_prediction", {}).get("ensemble", {}).get("confidence", 0)
                    symbolic_confidence = data.get("symbolic_analysis", {}).get("analysis", {}).get("trading_recommendation", {}).get("confidence", 0)
                    combined_confidence = data.get("confidence", 0)
                    
                    # Extract action
                    action = data.get("signal", {}).get("action", "wait")
                    
                    # Simulate ground truth (in real implementation, this would come from actual market outcomes)
                    # For now, we'll simulate based on confidence and some randomness
                    ground_truth_prob = (ai_confidence + symbolic_confidence) / 2
                    if action == "buy":
                        ground_truth_prob += 0.1
                    elif action == "sell":
                        ground_truth_prob -= 0.1
                    
                    ground_truth_prob = np.clip(ground_truth_prob + np.random.normal(0, 0.1), 0, 1)
                    ground_truth = 1 if np.random.random() < ground_truth_prob else 0
                    
                    return {
                        "symbol": symbol,
                        "strategy": strategy,
                        "ai_confidence": ai_confidence,
                        "symbolic_confidence": symbolic_confidence,
                        "combined_confidence": combined_confidence,
                        "action": action,
                        "ground_truth": ground_truth
                    }
                else:
                    return None
        except Exception as e:
            print(f"Error collecting sample for {symbol}-{strategy}: {e}")
            return None
    
    # Collect samples
    print("Collecting samples...")
    tasks = []
    for i in range(num_samples):
        symbol = symbols[i % len(symbols)]
        strategy = strategies[i % len(strategies)]
        tasks.append(collect_sample(symbol, strategy))
    
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]
    
    if not results:
        print("❌ No valid samples collected")
        return
    
    print(f"✅ Collected {len(results)} valid samples")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate calibration metrics
    print("\n" + "="*50)
    print("CALIBRATION METRICS")
    print("="*50)
    
    # AI Confidence
    ai_data = df[df['ai_confidence'] > 0]
    if len(ai_data) > 10:
        ai_brier = brier_score_loss(ai_data['ground_truth'], ai_data['ai_confidence'])
        print(f"AI Confidence Brier Score: {ai_brier:.4f}")
    
    # Symbolic Confidence
    symbolic_data = df[df['symbolic_confidence'] > 0]
    if len(symbolic_data) > 10:
        symbolic_brier = brier_score_loss(symbolic_data['ground_truth'], symbolic_data['symbolic_confidence'])
        print(f"Symbolic Confidence Brier Score: {symbolic_brier:.4f}")
    
    # Combined Confidence
    combined_data = df[df['combined_confidence'] > 0]
    if len(combined_data) > 10:
        combined_brier = brier_score_loss(combined_data['ground_truth'], combined_data['combined_confidence'])
        print(f"Combined Confidence Brier Score: {combined_brier:.4f}")
    
    # Strategy comparison
    print(f"\nBy Strategy:")
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        if len(strategy_data) > 10:
            strategy_brier = brier_score_loss(strategy_data['ground_truth'], strategy_data['combined_confidence'])
            print(f"  {strategy}: {strategy_brier:.4f}")
    
    # Symbol comparison
    print(f"\nBy Symbol:")
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        if len(symbol_data) > 10:
            symbol_brier = brier_score_loss(symbol_data['ground_truth'], symbol_data['combined_confidence'])
            print(f"  {symbol}: {symbol_brier:.4f}")
    
    # Generate calibration plot
    print("\nGenerating calibration plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    if len(combined_data) > 10:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            combined_data['ground_truth'], combined_data['combined_confidence'], n_bins=10
        )
        axes[0].plot(mean_predicted_value, fraction_of_positives, 'o-', label='Combined Confidence')
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Curve')
        axes[0].legend()
        axes[0].grid(True)
    
    # Confidence distribution
    axes[1].hist(combined_data['combined_confidence'], bins=20, alpha=0.7, label='Combined')
    if len(ai_data) > 0:
        axes[1].hist(ai_data['ai_confidence'], bins=20, alpha=0.7, label='AI')
    if len(symbolic_data) > 0:
        axes[1].hist(symbolic_data['symbolic_confidence'], bins=20, alpha=0.7, label='Symbolic')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"calibration_plot_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Calibration plot saved to: {plot_filename}")
    
    # Save results
    results_filename = f"calibration_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump({
            "samples": results,
            "metrics": {
                "ai_brier": ai_brier if len(ai_data) > 10 else None,
                "symbolic_brier": symbolic_brier if len(symbolic_data) > 10 else None,
                "combined_brier": combined_brier if len(combined_data) > 10 else None
            },
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_filename}")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_samples = 100
    api_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        api_url = sys.argv[2]
    
    asyncio.run(quick_calibration_test(api_url, num_samples))
