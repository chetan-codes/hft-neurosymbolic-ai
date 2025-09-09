#!/usr/bin/env python3
"""
Confidence Analysis for Neurosymbolic Trading System

This script analyzes confidence score variations across different
symbols, timeframes, and market conditions.
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
from scipy import stats

class ConfidenceAnalysis:
    """Analyze confidence score variations in the neurosymbolic trading system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.results = []
        
    async def run_confidence_analysis(self, symbols: List[str], timeframes: List[str], 
                                    strategies: List[str], num_samples: int = 100) -> Dict[str, Any]:
        """Run comprehensive confidence analysis"""
        
        print("Running confidence analysis...")
        
        all_data = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                for strategy in strategies:
                    print(f"Analyzing {symbol} {timeframe} {strategy}...")
                    
                    symbol_data = await self._analyze_symbol_confidence(
                        symbol, timeframe, strategy, num_samples
                    )
                    all_data.extend(symbol_data)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_data)
        
        # Perform various analyses
        symbol_analysis = self._analyze_symbol_variations(df)
        timeframe_analysis = self._analyze_timeframe_variations(df)
        strategy_analysis = self._analyze_strategy_variations(df)
        correlation_analysis = self._analyze_correlations(df)
        
        # Generate visualizations
        self._generate_confidence_plots(df)
        
        # Save results
        results = {
            "raw_data": all_data,
            "symbol_analysis": symbol_analysis,
            "timeframe_analysis": timeframe_analysis,
            "strategy_analysis": strategy_analysis,
            "correlation_analysis": correlation_analysis,
            "summary": self._generate_summary(df)
        }
        
        with open("paper_kit/experiments/confidence_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Confidence analysis completed!")
        return results
    
    async def _analyze_symbol_confidence(self, symbol: str, timeframe: str, 
                                       strategy: str, num_samples: int) -> List[Dict[str, Any]]:
        """Analyze confidence for a specific symbol"""
        
        symbol_data = []
        
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
                    
                    # Extract confidence data
                    data_point = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "ai_confidence": signal_data.get("ai_confidence", 0.0),
                        "symbolic_confidence": signal_data.get("symbolic_confidence", 0.0),
                        "final_confidence": signal_data.get("confidence", 0.0),
                        "action": signal_data.get("action", "hold"),
                        "timestamp": signal_data.get("timestamp", ""),
                        "sample_id": i
                    }
                    
                    symbol_data.append(data_point)
                    
            except Exception as e:
                print(f"Error in sample {i} for {symbol}: {e}")
                continue
        
        return symbol_data
    
    def _analyze_symbol_variations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence variations across symbols"""
        
        symbol_stats = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            stats = {
                "mean_ai_confidence": float(symbol_data['ai_confidence'].mean()),
                "std_ai_confidence": float(symbol_data['ai_confidence'].std()),
                "mean_symbolic_confidence": float(symbol_data['symbolic_confidence'].mean()),
                "std_symbolic_confidence": float(symbol_data['symbolic_confidence'].std()),
                "mean_final_confidence": float(symbol_data['final_confidence'].mean()),
                "std_final_confidence": float(symbol_data['final_confidence'].std()),
                "confidence_range": float(symbol_data['final_confidence'].max() - symbol_data['final_confidence'].min()),
                "action_distribution": symbol_data['action'].value_counts().to_dict(),
                "sample_count": len(symbol_data)
            }
            
            symbol_stats[symbol] = stats
        
        # Calculate symbol differentiation metrics
        final_confidences = [stats["mean_final_confidence"] for stats in symbol_stats.values()]
        symbol_differentiation = {
            "confidence_variance": float(np.var(final_confidences)),
            "confidence_std": float(np.std(final_confidences)),
            "confidence_range": float(max(final_confidences) - min(final_confidences))
        }
        
        return {
            "symbol_stats": symbol_stats,
            "differentiation_metrics": symbol_differentiation
        }
    
    def _analyze_timeframe_variations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence variations across timeframes"""
        
        timeframe_stats = {}
        
        for timeframe in df['timeframe'].unique():
            timeframe_data = df[df['timeframe'] == timeframe]
            
            stats = {
                "mean_ai_confidence": float(timeframe_data['ai_confidence'].mean()),
                "std_ai_confidence": float(timeframe_data['ai_confidence'].std()),
                "mean_symbolic_confidence": float(timeframe_data['symbolic_confidence'].mean()),
                "std_symbolic_confidence": float(timeframe_data['symbolic_confidence'].std()),
                "mean_final_confidence": float(timeframe_data['final_confidence'].mean()),
                "std_final_confidence": float(timeframe_data['final_confidence'].std()),
                "sample_count": len(timeframe_data)
            }
            
            timeframe_stats[timeframe] = stats
        
        return timeframe_stats
    
    def _analyze_strategy_variations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence variations across strategies"""
        
        strategy_stats = {}
        
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            
            stats = {
                "mean_ai_confidence": float(strategy_data['ai_confidence'].mean()),
                "std_ai_confidence": float(strategy_data['ai_confidence'].std()),
                "mean_symbolic_confidence": float(strategy_data['symbolic_confidence'].mean()),
                "std_symbolic_confidence": float(strategy_data['symbolic_confidence'].std()),
                "mean_final_confidence": float(strategy_data['final_confidence'].mean()),
                "std_final_confidence": float(strategy_data['final_confidence'].std()),
                "confidence_range": float(strategy_data['final_confidence'].max() - strategy_data['final_confidence'].min()),
                "action_distribution": strategy_data['action'].value_counts().to_dict(),
                "sample_count": len(strategy_data)
            }
            
            strategy_stats[strategy] = stats
        
        return strategy_stats
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different confidence measures"""
        
        # Calculate correlations
        correlations = {
            "ai_symbolic_correlation": float(df['ai_confidence'].corr(df['symbolic_confidence'])),
            "ai_final_correlation": float(df['ai_confidence'].corr(df['final_confidence'])),
            "symbolic_final_correlation": float(df['symbolic_confidence'].corr(df['final_confidence']))
        }
        
        # Calculate confidence factor contributions
        ai_contribution = float(df['ai_confidence'].std() / df['final_confidence'].std())
        symbolic_contribution = float(df['symbolic_confidence'].std() / df['final_confidence'].std())
        
        return {
            "correlations": correlations,
            "factor_contributions": {
                "ai_contribution": ai_contribution,
                "symbolic_contribution": symbolic_contribution
            }
        }
    
    def _generate_confidence_plots(self, df: pd.DataFrame):
        """Generate confidence analysis plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confidence by Symbol
        symbol_confidences = df.groupby('symbol')['final_confidence'].agg(['mean', 'std']).reset_index()
        axes[0, 0].bar(symbol_confidences['symbol'], symbol_confidences['mean'], 
                      yerr=symbol_confidences['std'], capsize=5)
        axes[0, 0].set_ylabel("Mean Final Confidence")
        axes[0, 0].set_title("Confidence by Symbol")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. AI vs Symbolic Confidence Scatter
        axes[0, 1].scatter(df['ai_confidence'], df['symbolic_confidence'], alpha=0.6)
        axes[0, 1].set_xlabel("AI Confidence")
        axes[0, 1].set_ylabel("Symbolic Confidence")
        axes[0, 1].set_title("AI vs Symbolic Confidence")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(df['ai_confidence'], df['symbolic_confidence'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df['ai_confidence'], p(df['ai_confidence']), "r--", alpha=0.8)
        
        # 3. Confidence Distribution by Strategy
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            axes[0, 2].hist(strategy_data['final_confidence'], alpha=0.6, label=strategy, bins=15)
        
        axes[0, 2].set_xlabel("Final Confidence")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("Confidence Distribution by Strategy")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confidence Range by Symbol
        symbol_ranges = df.groupby('symbol')['final_confidence'].agg(['min', 'max']).reset_index()
        symbol_ranges['range'] = symbol_ranges['max'] - symbol_ranges['min']
        axes[1, 0].bar(symbol_ranges['symbol'], symbol_ranges['range'])
        axes[1, 0].set_ylabel("Confidence Range")
        axes[1, 0].set_title("Confidence Range by Symbol")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Action Distribution by Symbol
        action_counts = df.groupby(['symbol', 'action']).size().unstack(fill_value=0)
        action_counts.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Action Distribution by Symbol")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title="Action")
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confidence vs Action
        action_confidences = df.groupby('action')['final_confidence'].agg(['mean', 'std']).reset_index()
        axes[1, 2].bar(action_confidences['action'], action_confidences['mean'], 
                      yerr=action_confidences['std'], capsize=5)
        axes[1, 2].set_ylabel("Mean Final Confidence")
        axes[1, 2].set_title("Confidence by Action")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("paper_kit/experiments/confidence_analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Confidence analysis plots saved to paper_kit/experiments/confidence_analysis_plots.png")
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of confidence analysis"""
        
        summary = {
            "overall_stats": {
                "total_samples": len(df),
                "unique_symbols": df['symbol'].nunique(),
                "unique_strategies": df['strategy'].nunique(),
                "unique_timeframes": df['timeframe'].nunique()
            },
            "confidence_stats": {
                "mean_ai_confidence": float(df['ai_confidence'].mean()),
                "std_ai_confidence": float(df['ai_confidence'].std()),
                "mean_symbolic_confidence": float(df['symbolic_confidence'].mean()),
                "std_symbolic_confidence": float(df['symbolic_confidence'].std()),
                "mean_final_confidence": float(df['final_confidence'].mean()),
                "std_final_confidence": float(df['final_confidence'].std()),
                "final_confidence_range": float(df['final_confidence'].max() - df['final_confidence'].min())
            },
            "correlations": {
                "ai_symbolic": float(df['ai_confidence'].corr(df['symbolic_confidence'])),
                "ai_final": float(df['ai_confidence'].corr(df['final_confidence'])),
                "symbolic_final": float(df['symbolic_confidence'].corr(df['final_confidence']))
            },
            "action_distribution": df['action'].value_counts().to_dict()
        }
        
        return summary

async def main():
    """Main function to run confidence analysis"""
    
    # Initialize analysis
    analysis = ConfidenceAnalysis()
    
    # Define analysis parameters
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    timeframes = ["daily"]
    strategies = ["neurosymbolic", "rule_only"]
    num_samples = 50  # Reduced for faster execution
    
    # Run analysis
    results = await analysis.run_confidence_analysis(
        symbols=symbols,
        timeframes=timeframes,
        strategies=strategies,
        num_samples=num_samples
    )
    
    # Print summary
    print("\n=== Confidence Analysis Results ===")
    
    overall = results["summary"]["overall_stats"]
    print(f"Total samples: {overall['total_samples']}")
    print(f"Unique symbols: {overall['unique_symbols']}")
    print(f"Unique strategies: {overall['unique_strategies']}")
    
    confidence_stats = results["summary"]["confidence_stats"]
    print(f"\nConfidence Statistics:")
    print(f"  Mean AI Confidence: {confidence_stats['mean_ai_confidence']:.4f}")
    print(f"  Mean Symbolic Confidence: {confidence_stats['mean_symbolic_confidence']:.4f}")
    print(f"  Mean Final Confidence: {confidence_stats['mean_final_confidence']:.4f}")
    print(f"  Final Confidence Range: {confidence_stats['final_confidence_range']:.4f}")
    
    correlations = results["summary"]["correlations"]
    print(f"\nCorrelations:")
    print(f"  AI-Symbolic: {correlations['ai_symbolic']:.4f}")
    print(f"  AI-Final: {correlations['ai_final']:.4f}")
    print(f"  Symbolic-Final: {correlations['symbolic_final']:.4f}")
    
    action_dist = results["summary"]["action_distribution"]
    print(f"\nAction Distribution:")
    for action, count in action_dist.items():
        print(f"  {action}: {count}")

if __name__ == "__main__":
    asyncio.run(main())
