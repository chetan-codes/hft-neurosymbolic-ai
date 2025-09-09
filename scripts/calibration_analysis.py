#!/usr/bin/env python3
"""
Calibration Analysis and Ablation Studies for HFT Neurosymbolic AI System
Analyzes confidence calibration, performs ablation studies, and generates research plots
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import httpx
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationAnalyzer:
    """Comprehensive calibration analysis for neurosymbolic AI system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results = []
        self.calibration_data = []
        
    async def collect_calibration_data(self, symbols: List[str], strategies: List[str], 
                                     num_samples: int = 100) -> List[Dict[str, Any]]:
        """Collect data for calibration analysis"""
        logger.info(f"Collecting calibration data: {len(symbols)} symbols × {len(strategies)} strategies × {num_samples} samples")
        
        calibration_data = []
        
        for symbol in symbols:
            for strategy in strategies:
                logger.info(f"Collecting data for {symbol} with {strategy} strategy")
                
                for i in range(num_samples):
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.post(
                                f"{self.api_base_url}/api/v1/trading/signal",
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
                                
                                # Extract action and other metadata
                                action = data.get("signal", {}).get("action", "wait")
                                regime = data.get("symbolic_analysis", {}).get("analysis", {}).get("market_regime", {}).get("regime", "unknown")
                                technical_signal = data.get("symbolic_analysis", {}).get("analysis", {}).get("technical_signals", {}).get("signal", "wait")
                                
                                # Simulate ground truth (in real implementation, this would come from actual market outcomes)
                                # For now, we'll simulate based on confidence and some randomness
                                ground_truth = self._simulate_ground_truth(ai_confidence, symbolic_confidence, action)
                                
                                calibration_data.append({
                                    "symbol": symbol,
                                    "strategy": strategy,
                                    "ai_confidence": ai_confidence,
                                    "symbolic_confidence": symbolic_confidence,
                                    "combined_confidence": combined_confidence,
                                    "action": action,
                                    "regime": regime,
                                    "technical_signal": technical_signal,
                                    "ground_truth": ground_truth,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                    except Exception as e:
                        logger.warning(f"Failed to collect data for {symbol}-{strategy}: {e}")
                        continue
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
        
        self.calibration_data = calibration_data
        logger.info(f"Collected {len(calibration_data)} calibration samples")
        return calibration_data
    
    def _simulate_ground_truth(self, ai_confidence: float, symbolic_confidence: float, action: str) -> int:
        """Simulate ground truth for calibration analysis"""
        # In a real system, this would be based on actual market outcomes
        # For now, we'll simulate based on confidence and some randomness
        
        # Higher confidence should correlate with better outcomes
        base_probability = (ai_confidence + symbolic_confidence) / 2
        
        # Add some noise and action-specific bias
        if action == "buy":
            base_probability += 0.1
        elif action == "sell":
            base_probability -= 0.1
        
        # Add random noise
        noise = np.random.normal(0, 0.1)
        final_probability = np.clip(base_probability + noise, 0, 1)
        
        # Convert to binary outcome
        return 1 if np.random.random() < final_probability else 0
    
    def analyze_calibration(self) -> Dict[str, Any]:
        """Analyze confidence calibration"""
        if not self.calibration_data:
            return {"error": "No calibration data available"}
        
        df = pd.DataFrame(self.calibration_data)
        
        # Separate data by confidence type
        ai_data = df[df['ai_confidence'] > 0]
        symbolic_data = df[df['symbolic_confidence'] > 0]
        combined_data = df[df['combined_confidence'] > 0]
        
        calibration_results = {}
        
        # AI Confidence Calibration
        if len(ai_data) > 10:
            ai_calibration = self._calculate_calibration_metrics(
                ai_data['ai_confidence'].values,
                ai_data['ground_truth'].values,
                "AI Confidence"
            )
            calibration_results["ai_confidence"] = ai_calibration
        
        # Symbolic Confidence Calibration
        if len(symbolic_data) > 10:
            symbolic_calibration = self._calculate_calibration_metrics(
                symbolic_data['symbolic_confidence'].values,
                symbolic_data['ground_truth'].values,
                "Symbolic Confidence"
            )
            calibration_results["symbolic_confidence"] = symbolic_calibration
        
        # Combined Confidence Calibration
        if len(combined_data) > 10:
            combined_calibration = self._calculate_calibration_metrics(
                combined_data['combined_confidence'].values,
                combined_data['ground_truth'].values,
                "Combined Confidence"
            )
            calibration_results["combined_confidence"] = combined_calibration
        
        # Strategy-wise calibration
        strategy_calibration = {}
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            if len(strategy_data) > 10:
                strategy_calibration[strategy] = self._calculate_calibration_metrics(
                    strategy_data['combined_confidence'].values,
                    strategy_data['ground_truth'].values,
                    f"Strategy: {strategy}"
                )
        calibration_results["by_strategy"] = strategy_calibration
        
        # Symbol-wise calibration
        symbol_calibration = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) > 10:
                symbol_calibration[symbol] = self._calculate_calibration_metrics(
                    symbol_data['combined_confidence'].values,
                    symbol_data['ground_truth'].values,
                    f"Symbol: {symbol}"
                )
        calibration_results["by_symbol"] = symbol_calibration
        
        return calibration_results
    
    def _calculate_calibration_metrics(self, confidence_scores: np.ndarray, 
                                     ground_truth: np.ndarray, name: str) -> Dict[str, Any]:
        """Calculate calibration metrics for a set of confidence scores"""
        try:
            # Brier Score
            brier_score = brier_score_loss(ground_truth, confidence_scores)
            
            # Log Loss
            log_loss_score = log_loss(ground_truth, confidence_scores)
            
            # Reliability Diagram data
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ground_truth, confidence_scores, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(confidence_scores, ground_truth, n_bins=10)
            
            # Maximum Calibration Error (MCE)
            mce = self._calculate_mce(confidence_scores, ground_truth, n_bins=10)
            
            # Isotonic Calibration
            isotonic_model = IsotonicRegression(out_of_bounds='clip')
            isotonic_calibrated = isotonic_model.fit_transform(confidence_scores, ground_truth)
            isotonic_brier = brier_score_loss(ground_truth, isotonic_calibrated)
            
            # Platt Scaling
            platt_model = LogisticRegression()
            platt_model.fit(confidence_scores.reshape(-1, 1), ground_truth)
            platt_calibrated = platt_model.predict_proba(confidence_scores.reshape(-1, 1))[:, 1]
            platt_brier = brier_score_loss(ground_truth, platt_calibrated)
            
            return {
                "name": name,
                "brier_score": brier_score,
                "log_loss": log_loss_score,
                "ece": ece,
                "mce": mce,
                "isotonic_brier": isotonic_brier,
                "platt_brier": platt_brier,
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
                "sample_size": len(confidence_scores),
                "mean_confidence": np.mean(confidence_scores),
                "std_confidence": np.std(confidence_scores)
            }
            
        except Exception as e:
            logger.error(f"Calibration calculation failed for {name}: {e}")
            return {"error": str(e)}
    
    def _calculate_ece(self, confidence_scores: np.ndarray, ground_truth: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, confidence_scores: np.ndarray, ground_truth: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def generate_calibration_plots(self, output_dir: str = "calibration_plots"):
        """Generate calibration plots and visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.calibration_data:
            logger.warning("No calibration data to plot")
            return
        
        df = pd.DataFrame(self.calibration_data)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Reliability Diagrams
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confidence Calibration Analysis', fontsize=16)
        
        # AI Confidence
        if len(df[df['ai_confidence'] > 0]) > 10:
            ai_data = df[df['ai_confidence'] > 0]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ai_data['ground_truth'], ai_data['ai_confidence'], n_bins=10
            )
            axes[0, 0].plot(mean_predicted_value, fraction_of_positives, 'o-', label='AI Confidence')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[0, 0].set_xlabel('Mean Predicted Probability')
            axes[0, 0].set_ylabel('Fraction of Positives')
            axes[0, 0].set_title('AI Confidence Calibration')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Symbolic Confidence
        if len(df[df['symbolic_confidence'] > 0]) > 10:
            symbolic_data = df[df['symbolic_confidence'] > 0]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                symbolic_data['ground_truth'], symbolic_data['symbolic_confidence'], n_bins=10
            )
            axes[0, 1].plot(mean_predicted_value, fraction_of_positives, 'o-', label='Symbolic Confidence')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[0, 1].set_xlabel('Mean Predicted Probability')
            axes[0, 1].set_ylabel('Fraction of Positives')
            axes[0, 1].set_title('Symbolic Confidence Calibration')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Combined Confidence
        if len(df[df['combined_confidence'] > 0]) > 10:
            combined_data = df[df['combined_confidence'] > 0]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                combined_data['ground_truth'], combined_data['combined_confidence'], n_bins=10
            )
            axes[1, 0].plot(mean_predicted_value, fraction_of_positives, 'o-', label='Combined Confidence')
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[1, 0].set_xlabel('Mean Predicted Probability')
            axes[1, 0].set_ylabel('Fraction of Positives')
            axes[1, 0].set_title('Combined Confidence Calibration')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Confidence Distribution
        axes[1, 1].hist(df['combined_confidence'], bins=20, alpha=0.7, label='Combined')
        if len(df[df['ai_confidence'] > 0]) > 0:
            axes[1, 1].hist(df[df['ai_confidence'] > 0]['ai_confidence'], bins=20, alpha=0.7, label='AI')
        if len(df[df['symbolic_confidence'] > 0]) > 0:
            axes[1, 1].hist(df[df['symbolic_confidence'] > 0]['symbolic_confidence'], bins=20, alpha=0.7, label='Symbolic')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Confidence Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/calibration_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Strategy Comparison
        if len(df['strategy'].unique()) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Brier Score by Strategy
            strategy_brier = []
            strategy_names = []
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                if len(strategy_data) > 10:
                    brier = brier_score_loss(strategy_data['ground_truth'], strategy_data['combined_confidence'])
                    strategy_brier.append(brier)
                    strategy_names.append(strategy)
            
            axes[0].bar(strategy_names, strategy_brier)
            axes[0].set_title('Brier Score by Strategy')
            axes[0].set_ylabel('Brier Score (lower is better)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Confidence vs Accuracy by Strategy
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                if len(strategy_data) > 10:
                    axes[1].scatter(strategy_data['combined_confidence'], 
                                  strategy_data['ground_truth'], 
                                  alpha=0.6, label=strategy)
            
            axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Ground Truth')
            axes[1].set_title('Confidence vs Accuracy by Strategy')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Symbol Analysis
        if len(df['symbol'].unique()) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Brier Score by Symbol
            symbol_brier = []
            symbol_names = []
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                if len(symbol_data) > 10:
                    brier = brier_score_loss(symbol_data['ground_truth'], symbol_data['combined_confidence'])
                    symbol_brier.append(brier)
                    symbol_names.append(symbol)
            
            axes[0].bar(symbol_names, symbol_brier)
            axes[0].set_title('Brier Score by Symbol')
            axes[0].set_ylabel('Brier Score (lower is better)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Confidence Distribution by Symbol
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                if len(symbol_data) > 10:
                    axes[1].hist(symbol_data['combined_confidence'], bins=10, alpha=0.7, label=symbol)
            
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Confidence Distribution by Symbol')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/symbol_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Calibration plots saved to {output_dir}/")
    
    def perform_ablation_study(self) -> Dict[str, Any]:
        """Perform ablation study on different components"""
        if not self.calibration_data:
            return {"error": "No calibration data available"}
        
        df = pd.DataFrame(self.calibration_data)
        ablation_results = {}
        
        # 1. AI vs Symbolic vs Combined
        components = ['ai_confidence', 'symbolic_confidence', 'combined_confidence']
        component_results = {}
        
        for component in components:
            component_data = df[df[component] > 0]
            if len(component_data) > 10:
                brier = brier_score_loss(component_data['ground_truth'], component_data[component])
                ece = self._calculate_ece(component_data[component].values, component_data['ground_truth'].values)
                component_results[component] = {
                    "brier_score": brier,
                    "ece": ece,
                    "sample_size": len(component_data)
                }
        
        ablation_results["component_ablation"] = component_results
        
        # 2. Strategy Ablation
        strategy_results = {}
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            if len(strategy_data) > 10:
                brier = brier_score_loss(strategy_data['ground_truth'], strategy_data['combined_confidence'])
                ece = self._calculate_ece(strategy_data['combined_confidence'].values, strategy_data['ground_truth'].values)
                strategy_results[strategy] = {
                    "brier_score": brier,
                    "ece": ece,
                    "sample_size": len(strategy_data)
                }
        
        ablation_results["strategy_ablation"] = strategy_results
        
        # 3. Symbol Ablation
        symbol_results = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) > 10:
                brier = brier_score_loss(symbol_data['ground_truth'], symbol_data['combined_confidence'])
                ece = self._calculate_ece(symbol_data['combined_confidence'].values, symbol_data['ground_truth'].values)
                symbol_results[symbol] = {
                    "brier_score": brier,
                    "ece": ece,
                    "sample_size": len(symbol_data)
                }
        
        ablation_results["symbol_ablation"] = symbol_results
        
        return ablation_results
    
    def save_results(self, filename: str = None):
        """Save calibration analysis results"""
        if filename is None:
            filename = f"calibration_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Collect all results
        results = {
            "calibration_data": self.calibration_data,
            "calibration_analysis": self.analyze_calibration(),
            "ablation_study": self.perform_ablation_study(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Calibration analysis results saved to {filename}")
        return filename

async def run_calibration_analysis():
    """Run comprehensive calibration analysis"""
    analyzer = CalibrationAnalyzer()
    
    # Test configurations
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]
    strategies = ["neurosymbolic", "rule_only", "momentum", "mean_reversion"]
    
    logger.info("Starting calibration analysis...")
    
    # Collect calibration data
    await analyzer.collect_calibration_data(symbols, strategies, num_samples=50)
    
    # Analyze calibration
    calibration_results = analyzer.analyze_calibration()
    
    # Print summary
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS SUMMARY")
    print("="*60)
    
    for component, metrics in calibration_results.items():
        if isinstance(metrics, dict) and "brier_score" in metrics:
            print(f"\n{component}:")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  Sample Size: {metrics['sample_size']}")
    
    # Generate plots
    analyzer.generate_calibration_plots()
    
    # Save results
    analyzer.save_results()
    
    return calibration_results

if __name__ == "__main__":
    asyncio.run(run_calibration_analysis())
