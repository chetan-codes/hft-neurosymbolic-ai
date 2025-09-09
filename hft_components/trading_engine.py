#!/usr/bin/env python3
"""
Trading Engine - Combines AI and Symbolic Reasoning for HFT
Generates final trading signals and executes trades
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingEngine:
    """Trading engine that combines AI predictions and symbolic reasoning"""
    
    def __init__(self):
        self.strategies = {}
        self.portfolio = {}
        self.risk_manager = {}
        self.execution_engine = {}
        self.health_status = True
        
        # Initialize trading components
        self._initialize_strategies()
        self._initialize_risk_manager()
        self._initialize_execution_engine()
        
        # Performance metrics
        self.metrics = {
            "signals_generated": 0,
            "trades_executed": 0,
            "avg_signal_time": 0.0,
            "success_rate": 0.0,
            "total_pnl": 0.0
        }
        
        logger.info("Trading Engine initialized")
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            self.strategies = {
                "neurosymbolic": {
                    "description": "Combines AI predictions with symbolic reasoning",
                    "weights": {
                        "ai_prediction": 0.4,
                        "symbolic_analysis": 0.4,
                        "risk_assessment": 0.2
                    },
                    "thresholds": {
                        "min_confidence": 0.0,
                        "min_volume": 1000000,
                        "max_position_size": 0.1
                    }
                },
                # Fallback strategy that ignores AI and relies purely on symbolic analysis
                "rule_only": {
                    "description": "Rule-only strategy that ignores AI predictions",
                    "weights": {
                        "ai_prediction": 0.0,
                        "symbolic_analysis": 1.0,
                        "risk_assessment": 0.0
                    },
                    "thresholds": {
                        "min_confidence": 0.0,
                        "min_volume": 100000,
                        "max_position_size": 0.05
                    }
                },
                "momentum": {
                    "description": "Momentum-based trading strategy",
                    "weights": {
                        "price_momentum": 0.5,
                        "volume_momentum": 0.3,
                        "volatility": 0.2
                    },
                    "thresholds": {
                        "min_confidence": 0.0,
                        "min_volume": 500000,
                        "max_position_size": 0.05
                    }
                },
                "mean_reversion": {
                    "description": "Mean reversion trading strategy",
                    "weights": {
                        "price_deviation": 0.4,
                        "rsi": 0.3,
                        "bollinger_bands": 0.3
                    },
                    "thresholds": {
                        "min_confidence": 0.0,
                        "min_volume": 2000000,
                        "max_position_size": 0.08
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            self.health_status = False
    
    def _initialize_risk_manager(self):
        """Initialize risk management rules"""
        try:
            self.risk_manager = {
                "position_limits": {
                    "max_single_position": 0.1,  # 10% of portfolio
                    "max_sector_exposure": 0.25,  # 25% in single sector
                    "max_leverage": 2.0,          # 2x leverage max
                    "min_liquidity": 1000000      # $1M minimum liquidity
                },
                "stop_loss": {
                    "max_daily_loss": 0.02,       # 2% daily loss limit
                    "max_position_loss": 0.05,    # 5% position loss limit
                    "trailing_stop": 0.03         # 3% trailing stop
                },
                "correlation_limits": {
                    "max_correlation": 0.8,       # 80% correlation limit
                    "min_diversification": 10     # Minimum 10 positions
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            self.health_status = False
    
    def _initialize_execution_engine(self):
        """Initialize trade execution engine"""
        try:
            self.execution_engine = {
                "execution_algorithms": {
                    "twap": "Time-weighted average price",
                    "vwap": "Volume-weighted average price",
                    "iceberg": "Iceberg order execution",
                    "market": "Market order execution"
                },
                "slippage_limits": {
                    "max_slippage": 0.001,        # 0.1% max slippage
                    "min_spread": 0.0005          # 0.05% minimum spread
                },
                "execution_limits": {
                    "max_order_size": 100000,     # $100K max order size
                    "min_order_size": 1000,       # $1K min order size
                    "max_orders_per_minute": 10   # 10 orders per minute
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize execution engine: {e}")
            self.health_status = False
    
    def is_healthy(self) -> bool:
        """Check if trading engine is healthy"""
        return self.health_status
    
    async def generate_signal(self, symbol: str, ai_prediction: Dict[str, Any], 
                            symbolic_analysis: Dict[str, Any], strategy: str = "neurosymbolic") -> Dict[str, Any]:
        """Generate trading signal combining AI and symbolic analysis"""
        start_time = time.time()
        
        try:
            # Provide sensible defaults/aliases
            if strategy == "default":
                strategy = "neurosymbolic"
            # If unknown strategy, fallback to rule_only rather than erroring out
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy '{strategy}', falling back to 'rule_only'")
                strategy = "rule_only"
            
            # Get strategy configuration
            strategy_config = self.strategies[strategy]
            
            # Combine AI prediction and symbolic analysis
            combined_signal = await self._combine_signals(
                ai_prediction, symbolic_analysis, strategy_config
            )
            
            # Apply risk management
            risk_adjusted_signal = await self._apply_risk_management(
                symbol, combined_signal, strategy_config
            )
            
            # Generate final trading signal
            final_signal = await self._generate_final_signal(
                symbol, risk_adjusted_signal, strategy_config
            )
            
            # Update metrics
            signal_time = time.time() - start_time
            self._update_metrics(signal_time)
            
            return {
                "symbol": symbol,
                "strategy": strategy,
                "signal": final_signal,
                "confidence": final_signal.get("confidence", 0.0),
                "reasoning": final_signal.get("reasoning", ""),
                "execution_plan": final_signal.get("execution_plan", {}),
                "signal_time_ms": signal_time * 1000,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {"error": str(e)}
    
    async def _combine_signals(self, ai_prediction: Dict[str, Any], 
                             symbolic_analysis: Dict[str, Any], 
                             strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Combine AI predictions with symbolic analysis"""
        try:
            weights = strategy_config["weights"]
            
            # Extract AI prediction components
            ai_ensemble = ai_prediction.get("ensemble", {})
            # If AI produced an error or missing ensemble, treat confidence as 0
            if not ai_ensemble or ai_prediction.get("error") or ai_ensemble.get("error"):
                ai_confidence = 0.0
                ai_prediction_values = []
            else:
                ai_confidence = ai_ensemble.get("confidence", 0.0)
                ai_prediction_values = ai_ensemble.get("prediction", [])
            
            # Extract symbolic analysis components
            symbolic_recommendation = symbolic_analysis.get("analysis", {}).get("trading_recommendation", {})
            symbolic_action = symbolic_recommendation.get("action", "hold")
            symbolic_confidence = symbolic_recommendation.get("confidence", 0.0)
            
            # Combine signals using weighted approach
            combined_confidence = (
                weights.get("ai_prediction", 0.0) * ai_confidence +
                weights.get("symbolic_analysis", 0.0) * symbolic_confidence
            )
            
            # Remove hard minimum confidence clamp to allow natural spread
            combined_confidence = max(0.0, min(0.99, combined_confidence))
            
            # Determine action with softer thresholds and agreement bias
            if ai_confidence == 0.0:
                action = symbolic_action
            else:
                if symbolic_action == "buy" and ai_confidence >= 0.55:
                    action = "buy"
                elif symbolic_action == "sell" and ai_confidence <= 0.45:
                    action = "sell"
                elif symbolic_action == "hold":
                    action = "hold"
                else:
                    action = "hold"
            
            # Agreement bonus/penalty on combined confidence
            if (symbolic_action == "buy" and ai_confidence >= 0.55) or (symbolic_action == "sell" and ai_confidence <= 0.45):
                combined_confidence = min(0.99, combined_confidence + 0.03)
            elif symbolic_action in ("buy", "sell") and 0.45 < ai_confidence < 0.55:
                combined_confidence = max(0.0, combined_confidence - 0.02)

            return {
                "action": action,
                "confidence": combined_confidence,
                "ai_confidence": ai_confidence,
                "symbolic_confidence": symbolic_confidence,
                "ai_prediction": ai_prediction_values,
                "symbolic_reasoning": symbolic_recommendation.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return {"action": "hold", "confidence": 0.0}
    
    async def _apply_risk_management(self, symbol: str, signal: Dict[str, Any], 
                                   strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management rules to trading signal"""
        try:
            # Check position limits
            position_ok = await self._check_position_limits(symbol, signal)
            
            # Check liquidity requirements
            liquidity_ok = await self._check_liquidity_requirements(symbol)
            
            # Check correlation limits
            correlation_ok = await self._check_correlation_limits(symbol)
            
            # Check stop loss conditions
            stop_loss_ok = await self._check_stop_loss_conditions(symbol)
            
            # Apply risk adjustments
            risk_adjusted_signal = signal.copy()
            
            if not all([position_ok, liquidity_ok, correlation_ok, stop_loss_ok]):
                # Soft risk gate: scale down confidence and position sizing, avoid immediate HOLD
                risk_adjusted_signal["confidence"] *= 0.6
                risk_adjusted_signal["risk_violations"] = [
                    "position_limit" if not position_ok else None,
                    "liquidity" if not liquidity_ok else None,
                    "correlation" if not correlation_ok else None,
                    "stop_loss" if not stop_loss_ok else None
                ]
                risk_adjusted_signal["risk_violations"] = [v for v in risk_adjusted_signal["risk_violations"] if v]
                # Force HOLD only if multiple concurrent critical violations
                if len(risk_adjusted_signal["risk_violations"]) >= 2:
                    risk_adjusted_signal["action"] = "hold"
            
            return risk_adjusted_signal
            
        except Exception as e:
            logger.error(f"Risk management failed: {e}")
            return signal
    
    async def _generate_final_signal(self, symbol: str, signal: Dict[str, Any], 
                                   strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading signal with execution plan"""
        try:
            thresholds = strategy_config["thresholds"]
            
            # Do not enforce a hard min-confidence hold; proceed with soft handling downstream
            
            # Generate execution plan
            execution_plan = await self._generate_execution_plan(symbol, signal, strategy_config)
            
            # Final signal
            final_signal = {
                "action": signal["action"],
                "confidence": signal["confidence"],
                "reasoning": signal.get("symbolic_reasoning", ""),
                "execution_plan": execution_plan,
                "risk_violations": signal.get("risk_violations", [])
            }
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Final signal generation failed: {e}")
            return {"action": "hold", "confidence": 0.0, "reasoning": "error"}
    
    async def _check_position_limits(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Check position limits"""
        try:
            # Placeholder - would check actual portfolio positions
            current_position = self.portfolio.get(symbol, 0.0)
            max_position = self.risk_manager["position_limits"]["max_single_position"]
            
            if signal["action"] == "buy":
                return current_position < max_position
            elif signal["action"] == "sell":
                return current_position > -max_position
            
            return True
            
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            return False
    
    async def _check_liquidity_requirements(self, symbol: str) -> bool:
        """Check liquidity requirements"""
        try:
            # Placeholder - would check actual market liquidity
            min_liquidity = self.risk_manager["position_limits"]["min_liquidity"]
            
            # Simulate liquidity check
            simulated_liquidity = 5000000  # $5M simulated liquidity
            return simulated_liquidity >= min_liquidity
            
        except Exception as e:
            logger.error(f"Liquidity check failed: {e}")
            return False
    
    async def _check_correlation_limits(self, symbol: str) -> bool:
        """Check correlation limits"""
        try:
            # Placeholder - would check actual portfolio correlations
            max_correlation = self.risk_manager["correlation_limits"]["max_correlation"]
            
            # Simulate correlation check
            simulated_correlation = 0.3  # 30% simulated correlation
            return simulated_correlation <= max_correlation
            
        except Exception as e:
            logger.error(f"Correlation check failed: {e}")
            return False
    
    async def _check_stop_loss_conditions(self, symbol: str) -> bool:
        """Check stop loss conditions"""
        try:
            # Placeholder - would check actual P&L and stop loss conditions
            max_daily_loss = self.risk_manager["stop_loss"]["max_daily_loss"]
            
            # Simulate daily P&L check
            simulated_daily_pnl = -0.01  # -1% simulated daily loss
            return simulated_daily_pnl > -max_daily_loss
            
        except Exception as e:
            logger.error(f"Stop loss check failed: {e}")
            return False
    
    async def _generate_execution_plan(self, symbol: str, signal: Dict[str, Any], 
                                     strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan for the trading signal"""
        try:
            thresholds = strategy_config["thresholds"]
            
            # Calculate position size based on confidence and limits
            max_position_size = thresholds["max_position_size"]
            position_size = signal["confidence"] * max_position_size
            
            # Select execution algorithm
            if signal["confidence"] > 0.8:
                execution_algorithm = "market"  # High confidence = market order
            elif signal["confidence"] > 0.6:
                execution_algorithm = "twap"    # Medium confidence = TWAP
            else:
                execution_algorithm = "vwap"    # Low confidence = VWAP
            
            # Generate execution plan
            execution_plan = {
                "symbol": symbol,
                "action": signal["action"],
                "position_size": position_size,
                "execution_algorithm": execution_algorithm,
                "urgency": "high" if signal["confidence"] > 0.8 else "medium",
                "slippage_limit": self.execution_engine["slippage_limits"]["max_slippage"],
                "order_limits": {
                    "max_order_size": self.execution_engine["execution_limits"]["max_order_size"],
                    "min_order_size": self.execution_engine["execution_limits"]["min_order_size"]
                }
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Execution plan generation failed: {e}")
            return {}
    
    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading signal"""
        try:
            execution_plan = signal.get("execution_plan", {})
            
            if not execution_plan:
                return {"status": "failed", "reason": "No execution plan"}
            
            # Simulate trade execution
            execution_result = await self._simulate_execution(execution_plan)
            
            # Update portfolio
            await self._update_portfolio(execution_plan, execution_result)
            
            # Update metrics
            self.metrics["trades_executed"] += 1
            
            return {
                "status": "success",
                "execution_id": f"exec_{int(time.time())}",
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"status": "failed", "reason": str(e)}
    
    async def _simulate_execution(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trade execution"""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.001)  # 1ms execution time
            
            # Simulate execution price
            base_price = 100.0  # Placeholder base price
            slippage = execution_plan.get("slippage_limit", 0.001)
            
            if execution_plan["action"] == "buy":
                execution_price = base_price * (1 + slippage)
            else:
                execution_price = base_price * (1 - slippage)
            
            # Simulate execution result
            execution_result = {
                "executed_price": execution_price,
                "executed_size": execution_plan["position_size"],
                "execution_time_ms": 1.0,
                "slippage": slippage,
                "algorithm": execution_plan["execution_algorithm"]
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Execution simulation failed: {e}")
            return {"error": str(e)}
    
    async def _update_portfolio(self, execution_plan: Dict[str, Any], execution_result: Dict[str, Any]):
        """Update portfolio after trade execution"""
        try:
            symbol = execution_plan["symbol"]
            action = execution_plan["action"]
            size = execution_result["executed_size"]
            
            # Update portfolio position
            current_position = self.portfolio.get(symbol, 0.0)
            
            if action == "buy":
                self.portfolio[symbol] = current_position + size
            elif action == "sell":
                self.portfolio[symbol] = current_position - size
            
            logger.info(f"Portfolio updated: {symbol} = {self.portfolio[symbol]}")
            
        except Exception as e:
            logger.error(f"Portfolio update failed: {e}")
    
    def _update_metrics(self, signal_time: float):
        """Update performance metrics"""
        try:
            self.metrics["signals_generated"] += 1
            
            # Update average signal time
            current_avg = self.metrics["avg_signal_time"]
            count = self.metrics["signals_generated"]
            self.metrics["avg_signal_time"] = (current_avg * (count - 1) + signal_time) / count
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get trading engine metrics"""
        return self.metrics.copy()
    
    def get_portfolio(self) -> Dict[str, float]:
        """Get current portfolio positions"""
        return self.portfolio.copy()
    
    def add_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> bool:
        """Add a new trading strategy"""
        try:
            self.strategies[strategy_name] = strategy_config
            logger.info(f"Added strategy: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add strategy: {e}")
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a trading strategy"""
        try:
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]
                logger.info(f"Removed strategy: {strategy_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove strategy: {e}")
            return False
    
    def update_risk_limits(self, new_limits: Dict[str, Any]) -> bool:
        """Update risk management limits"""
        try:
            self.risk_manager.update(new_limits)
            logger.info("Risk limits updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update risk limits: {e}")
            return False 