#!/usr/bin/env python3
"""
Symbolic Reasoner - Logical Reasoning for HFT
Uses MiniKanren for compliance checking and symbolic analysis
Integrates with Rule Loader for dynamic rule management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import numpy as np

# MiniKanren imports
from kanren import run, var, eq, conde, Relation, facts, fact
from kanren.core import lall, lany, condeseq

# Import rule loader
from .rule_loader import RuleLoader

logger = logging.getLogger(__name__)

class SymbolicReasoner:
    """Symbolic reasoning engine for HFT compliance and analysis"""
    
    def __init__(self, rule_loader: Optional[RuleLoader] = None):
        self.rules = {}
        self.facts = {}
        self.compliance_rules = {}
        self.risk_limits = {}
        self.health_status = True
        
        # Rule loader integration
        self.rule_loader = rule_loader
        self.active_rule_pack = None
        
        # Initialize reasoning rules
        self._initialize_rules()
        self._initialize_compliance_rules()
        self._initialize_risk_limits()
        
        # Load default rule pack if available
        if self.rule_loader:
            self._load_default_rules()
        
        # Performance metrics
        self.metrics = {
            "reasoning_sessions": 0,
            "avg_reasoning_time": 0.0,
            "compliance_checks": 0,
            "risk_assessments": 0,
            "rule_pack_usage": 0,
            "dynamic_rule_evaluations": 0
        }
        
        # Reasoning trace tracking
        self.reasoning_traces = []
        self.current_trace = None
        self.trace_enabled = True
        
        logger.info("Symbolic Reasoner initialized with rule loader integration")
    
    def _initialize_rules(self):
        """Initialize symbolic reasoning rules"""
        try:
            # Define relations
            self.price_trend = Relation('price_trend')
            self.volume_pattern = Relation('volume_pattern')
            self.technical_signal = Relation('technical_signal')
            self.market_regime = Relation('market_regime')
            self.risk_level = Relation('risk_level')
            
            # Define facts and rules
            self._define_trading_rules()
            self._define_market_rules()
            self._define_risk_rules()
            
        except Exception as e:
            logger.error(f"Failed to initialize rules: {e}")
            self.health_status = False
    
    def _define_trading_rules(self):
        """Define trading-related symbolic rules"""
        try:
            # Price trend rules
            facts(self.price_trend,
                ('bullish', 'strong_uptrend'),
                ('bearish', 'strong_downtrend'),
                ('neutral', 'sideways_market'),
                ('volatile', 'high_volatility')
            )
            
            # Volume pattern rules
            facts(self.volume_pattern,
                ('high', 'above_average'),
                ('low', 'below_average'),
                ('normal', 'average_volume'),
                ('spike', 'volume_spike')
            )
            
            # Technical signal rules
            facts(self.technical_signal,
                ('buy', 'strong_buy_signal'),
                ('sell', 'strong_sell_signal'),
                ('hold', 'neutral_signal'),
                ('wait', 'wait_for_confirmation')
            )
            
        except Exception as e:
            logger.error(f"Failed to define trading rules: {e}")
    
    def _define_market_rules(self):
        """Define market regime rules"""
        try:
            # Market regime facts
            facts(self.market_regime,
                ('trending', 'clear_direction'),
                ('ranging', 'sideways_movement'),
                ('volatile', 'high_uncertainty'),
                ('calm', 'low_volatility')
            )
            
        except Exception as e:
            logger.error(f"Failed to define market rules: {e}")
    
    def _define_risk_rules(self):
        """Define risk assessment rules"""
        try:
            # Risk level facts
            facts(self.risk_level,
                ('low', 'acceptable_risk'),
                ('medium', 'moderate_risk'),
                ('high', 'high_risk'),
                ('extreme', 'extreme_risk')
            )
            
        except Exception as e:
            logger.error(f"Failed to define risk rules: {e}")
    
    def _initialize_compliance_rules(self):
        """Initialize compliance checking rules"""
        try:
            self.compliance_rules = {
                "position_limits": {
                    "max_position_size": 0.1,  # 10% of portfolio
                    "max_daily_loss": 0.02,    # 2% daily loss limit
                    "max_concentration": 0.25   # 25% in single asset
                },
                "trading_hours": {
                    "start": "09:30",
                    "end": "16:00",
                    "timezone": "America/New_York"
                },
                "restricted_securities": [
                    "penny_stocks",
                    "illiquid_securities",
                    "high_risk_derivatives"
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize compliance rules: {e}")
    
    def _initialize_risk_limits(self):
        """Initialize risk management limits"""
        try:
            self.risk_limits = {
                "var_limit": 0.02,        # 2% Value at Risk limit
                "volatility_limit": 0.3,  # 30% volatility limit
                "correlation_limit": 0.8,  # 80% correlation limit
                "liquidity_minimum": 1000000,  # $1M minimum liquidity
                "max_drawdown": 0.15      # 15% maximum drawdown
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize risk limits: {e}")
    
    def is_healthy(self) -> bool:
        """Check if symbolic reasoner is healthy"""
        return self.health_status
    
    async def analyze(self, market_data: Dict[str, Any], ai_prediction: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """Perform symbolic analysis of market data and AI predictions"""
        start_time = time.time()
        
        try:
            # Start reasoning trace
            session_id = f"analysis_{int(time.time())}"
            trace_id = self.start_reasoning_trace(session_id, market_data)
            
            self.add_reasoning_step("analysis_start", "Starting symbolic analysis of market data")
            
            # Try to use rule pack evaluation first, fall back to hardcoded rules
            if self.active_rule_pack and self.rule_loader:
                self.add_reasoning_step("rule_evaluation", f"Using rule pack: {self.active_rule_pack}")
                
                # Evaluate market regime with detailed logging
                regime_result = self.evaluate_market_regime_rules(market_data)
                if "evaluated_regimes" in regime_result:
                    for regime_eval in regime_result["evaluated_regimes"]:
                        self.add_rule_evaluation(
                            rule_name=regime_eval["regime_name"],
                            rule_type="regime",
                            input_data={"volatility": market_data.get("volatility", 0), "trend_strength": market_data.get("trend_strength", 0)},
                            output={"regime": regime_eval["regime_name"], "match_score": regime_eval["match_score"]},
                            match_score=regime_eval["match_score"],
                            applied=regime_eval["regime_name"] == regime_result["regime"]
                        )
                
                # Evaluate technical signals with detailed logging
                signal_result = self.evaluate_technical_signal_rules(market_data)
                if "evaluated_rules" in signal_result:
                    for rule_eval in signal_result["evaluated_rules"]:
                        self.add_rule_evaluation(
                            rule_name=rule_eval["rule_id"],
                            rule_type="signal",
                            input_data={"ma_short": market_data.get("ma_short", 0), "ma_long": market_data.get("ma_long", 0), "rsi": market_data.get("rsi", 50)},
                            output={"signal": rule_eval["signal"], "match_score": rule_eval["match_score"]},
                            match_score=rule_eval["match_score"],
                            applied=rule_eval["rule_id"] == signal_result.get("rule_id")
                        )
                
                analysis_results = {
                    "market_regime": regime_result,
                    "technical_signals": signal_result,
                    "risk_assessment": await self._assess_risk(market_data, ai_prediction),
                    "compliance_check": await self._check_compliance(market_data, ai_prediction),
                    "trading_recommendation": await self._generate_trading_recommendation(market_data, ai_prediction, symbol)
                }
            else:
                self.add_reasoning_step("rule_evaluation", "Using hardcoded rules (fallback)")
                # Fall back to original hardcoded analysis
                analysis_results = {
                    "market_regime": await self._analyze_market_regime(market_data),
                    "technical_signals": await self._analyze_technical_signals(market_data, symbol),
                    "risk_assessment": await self._assess_risk(market_data, ai_prediction),
                    "compliance_check": await self._check_compliance(market_data, ai_prediction),
                    "trading_recommendation": await self._generate_trading_recommendation(market_data, ai_prediction, symbol)
                }
            
            # Add final decision
            final_decision = {
                "market_regime": analysis_results.get("market_regime", {}).get("regime", "unknown"),
                "technical_signal": analysis_results.get("technical_signals", {}).get("signal", "wait"),
                "risk_level": analysis_results.get("risk_assessment", {}).get("risk_level", "unknown"),
                "compliance_status": analysis_results.get("compliance_check", {}).get("status", "unknown"),
                "recommendation": analysis_results.get("trading_recommendation", {}).get("action", "hold")
            }
            
            self.add_decision("symbolic_analysis", final_decision["recommendation"], 
                            analysis_results.get("technical_signals", {}).get("confidence", 0.0),
                            [f"Market regime: {final_decision['market_regime']}", 
                             f"Technical signal: {final_decision['technical_signal']}",
                             f"Risk level: {final_decision['risk_level']}"])
            
            # Update metrics
            reasoning_time = time.time() - start_time
            self._update_metrics(reasoning_time)
            
            # End reasoning trace
            performance_metrics = {"reasoning_time_ms": reasoning_time * 1000}
            self.end_reasoning_trace(final_decision, performance_metrics)
            
            return {
                "analysis": analysis_results,
                "reasoning_time_ms": reasoning_time * 1000,
                "timestamp": datetime.now().isoformat(),
                "rule_pack_used": self.active_rule_pack if self.active_rule_pack else "hardcoded",
                "reasoning_trace_id": trace_id
            }
            
        except Exception as e:
            logger.error(f"Symbolic analysis failed: {e}")
            if self.current_trace:
                self.add_reasoning_step("error", f"Symbolic analysis failed: {e}")
                self.end_reasoning_trace({"error": str(e)})
            return {"error": str(e)}
    
    async def _analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market regime using symbolic reasoning"""
        try:
            # Extract price and volume data
            price_data = self._extract_price_data(market_data)
            volume_data = self._extract_volume_data(market_data)
            
            if not price_data or len(price_data) < 20:
                return {"regime": "unknown", "confidence": 0.0}
            
            # Calculate market characteristics
            price_changes = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]
            volatility = self._calculate_volatility(price_changes)
            trend_strength = self._calculate_trend_strength(price_data)
            
            # Symbolic reasoning for market regime
            regime = self._symbolic_regime_classification(volatility, trend_strength, volume_data)
            
            return {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "confidence": self._calculate_regime_confidence(volatility, trend_strength)
            }
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return {"regime": "unknown", "confidence": 0.0}
    
    async def _analyze_technical_signals(self, market_data: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """Analyze technical signals using symbolic reasoning"""
        try:
            price_data = self._extract_price_data(market_data)
            
            if not price_data or len(price_data) < 20:
                return {"signal": "wait", "confidence": 0.0}
            
            # Calculate technical indicators
            ma_short = self._calculate_moving_average(price_data, 5)
            ma_long = self._calculate_moving_average(price_data, 20)
            rsi = self._calculate_rsi(price_data)
            
            # Symbolic reasoning for technical signals
            signal = self._symbolic_signal_generation(ma_short, ma_long, rsi, price_data)
            
            return {
                "signal": signal,
                "ma_crossover": ma_short > ma_long,
                "rsi_level": rsi,
                "confidence": self._calculate_signal_confidence(ma_short, ma_long, rsi, symbol)
            }
            
        except Exception as e:
            logger.error(f"Technical signal analysis failed: {e}")
            return {"signal": "wait", "confidence": 0.0}
    
    async def _assess_risk(self, market_data: Dict[str, Any], ai_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk using symbolic reasoning"""
        try:
            price_data = self._extract_price_data(market_data)
            
            if not price_data:
                return {"risk_level": "unknown", "confidence": 0.0}
            
            # Calculate risk metrics
            volatility = self._calculate_volatility(price_data)
            var_95 = self._calculate_var(price_data, 0.95)
            max_drawdown = self._calculate_max_drawdown(price_data)
            
            # Check against risk limits
            risk_violations = []
            if volatility > self.risk_limits["volatility_limit"]:
                risk_violations.append("high_volatility")
            if var_95 > self.risk_limits["var_limit"]:
                risk_violations.append("high_var")
            if max_drawdown > self.risk_limits["max_drawdown"]:
                risk_violations.append("high_drawdown")
            
            # Symbolic risk assessment
            risk_level = self._symbolic_risk_assessment(volatility, var_95, max_drawdown, risk_violations)
            
            return {
                "risk_level": risk_level,
                "volatility": volatility,
                "var_95": var_95,
                "max_drawdown": max_drawdown,
                "violations": risk_violations,
                "confidence": self._calculate_risk_confidence(volatility, var_95, max_drawdown)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"risk_level": "unknown", "confidence": 0.0}
    
    async def _check_compliance(self, market_data: Dict[str, Any], ai_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with trading rules"""
        try:
            current_time = datetime.now()
            
            # Check trading hours
            trading_hours_ok = self._check_trading_hours(current_time)
            
            # Check position limits (would need portfolio data)
            position_limits_ok = True  # Placeholder
            
            # Check for restricted securities
            restricted_securities_ok = self._check_restricted_securities(market_data)
            
            # Overall compliance
            compliance_ok = trading_hours_ok and position_limits_ok and restricted_securities_ok
            
            return {
                "compliant": compliance_ok,
                "trading_hours": trading_hours_ok,
                "position_limits": position_limits_ok,
                "restricted_securities": restricted_securities_ok,
                "violations": [] if compliance_ok else ["compliance_violation"]
            }
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return {"compliant": False, "violations": ["compliance_check_error"]}
    
    async def _generate_trading_recommendation(self, market_data: Dict[str, Any], ai_prediction: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """Generate trading recommendation using symbolic reasoning"""
        try:
            # Get analysis results
            market_regime = await self._analyze_market_regime(market_data)
            technical_signals = await self._analyze_technical_signals(market_data, symbol)
            risk_assessment = await self._assess_risk(market_data, ai_prediction)
            compliance_check = await self._check_compliance(market_data, ai_prediction)
            
            # Symbolic reasoning for recommendation
            recommendation = self._symbolic_recommendation_generation(
                market_regime, technical_signals, risk_assessment, compliance_check, ai_prediction
            )
            
            return {
                "action": recommendation["action"],
                "confidence": recommendation["confidence"],
                "reasoning": recommendation["reasoning"],
                "conditions": recommendation["conditions"]
            }
            
        except Exception as e:
            logger.error(f"Trading recommendation generation failed: {e}")
            return {"action": "hold", "confidence": 0.0, "reasoning": "error"}
    
    def _extract_price_data(self, market_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract price data from market data"""
        try:
            # Try different data sources
            if "neo4j" in market_data and market_data["neo4j"]:
                return [float(record["price"]) for record in market_data["neo4j"] if "price" in record and record["price"]]
            elif "dgraph" in market_data and market_data["dgraph"]:
                data = market_data["dgraph"].get("market_data", [])
                # Extract close prices from Dgraph RDF format
                close_prices = []
                for item in data:
                    if (isinstance(item, dict) and 
                        "predicate" in item and 
                        "closePrice" in item["predicate"] and 
                        "object" in item):
                        try:
                            close_prices.append(float(item["object"]))
                        except (ValueError, TypeError):
                            continue
                return close_prices if close_prices else None
            elif "jena" in market_data and market_data["jena"]:
                results = market_data["jena"].get("results", {}).get("bindings", [])
                return [float(binding["price"]["value"]) for binding in results if "price" in binding and binding["price"]["value"]]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract price data: {e}")
            return None
    
    def _extract_volume_data(self, market_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract volume data from market data"""
        try:
            # Similar to price data extraction
            if "neo4j" in market_data and market_data["neo4j"]:
                return [float(record["volume"]) for record in market_data["neo4j"] if "volume" in record and record["volume"]]
            elif "dgraph" in market_data and market_data["dgraph"]:
                data = market_data["dgraph"].get("market_data", [])
                # Extract volume from Dgraph RDF format
                volumes = []
                for item in data:
                    if (isinstance(item, dict) and 
                        "predicate" in item and 
                        "volume" in item["predicate"] and 
                        "object" in item):
                        try:
                            volumes.append(float(item["object"]))
                        except (ValueError, TypeError):
                            continue
                return volumes if volumes else None
            elif "jena" in market_data and market_data["jena"]:
                results = market_data["jena"].get("results", {}).get("bindings", [])
                return [float(binding["volume"]["value"]) for binding in results if "volume" in binding and binding["volume"]["value"]]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract volume data: {e}")
            return None
    
    def _calculate_volatility(self, data: List[float]) -> float:
        """Calculate volatility (standard deviation)"""
        try:
            if len(data) < 2:
                return 0.0
            return float(np.std(data))
        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, data: List[float]) -> float:
        """Calculate trend strength using linear regression"""
        try:
            if len(data) < 10:
                return 0.0
            
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            return slope  # Return signed slope to preserve direction
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_moving_average(self, data: List[float], window: int) -> float:
        """Calculate moving average"""
        try:
            if len(data) < window:
                return data[-1] if data else 0.0
            return float(np.mean(data[-window:]))
        except Exception as e:
            logger.error(f"Moving average calculation failed: {e}")
            return 0.0
    
    def _calculate_rsi(self, data: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(data) < period + 1:
                return 50.0
            
            deltas = np.diff(data)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0
    
    def _calculate_var(self, data: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        try:
            if len(data) < 2:
                return 0.0
            
            returns = np.diff(data) / data[:-1]
            var = np.percentile(returns, (1 - confidence) * 100)
            return abs(float(var))
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, data: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(data) < 2:
                return 0.0
            
            peak = data[0]
            max_dd = 0.0
            
            for price in data:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                max_dd = max(max_dd, dd)
            
            return float(max_dd)
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    def _symbolic_regime_classification(self, volatility: float, trend_strength: float, volume_data: Optional[List[float]]) -> str:
        """Classify market regime using symbolic reasoning"""
        try:
            # Define symbolic rules for regime classification
            if volatility > 0.05:  # High volatility
                if trend_strength > 0.01:  # Strong trend
                    return "trending"
                else:
                    return "volatile"
            else:  # Low volatility
                if trend_strength > 0.005:  # Moderate trend
                    return "trending"
                else:
                    return "ranging"
                    
        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return "unknown"
    
    def _symbolic_signal_generation(self, ma_short: float, ma_long: float, rsi: float, price_data: List[float]) -> str:
        """Generate technical signals using symbolic reasoning"""
        try:
            current_price = price_data[-1] if price_data else 0.0
            
            # Define symbolic rules
            if ma_short > ma_long and rsi < 70:  # Golden cross, not overbought
                return "buy"
            elif ma_short < ma_long and rsi > 30:  # Death cross, not oversold
                return "sell"
            elif rsi > 70:  # Overbought
                return "sell"
            elif rsi < 30:  # Oversold
                return "buy"
            else:
                return "hold"
                
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return "wait"
    
    def _symbolic_risk_assessment(self, volatility: float, var_95: float, max_dd: float, violations: List[str]) -> str:
        """Assess risk level using symbolic reasoning"""
        try:
            # Count violations
            violation_count = len(violations)
            
            if violation_count >= 2 or max_dd > 0.2:
                return "extreme"
            elif violation_count >= 1 or volatility > 0.1:
                return "high"
            elif volatility > 0.05:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return "unknown"
    
    def _check_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        try:
            # Simple check - would need timezone handling for production
            hour = current_time.hour
            minute = current_time.minute
            current_minutes = hour * 60 + minute
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_start = 9 * 60 + 30  # 9:30 AM
            market_end = 16 * 60        # 4:00 PM
            
            return market_start <= current_minutes <= market_end
            
        except Exception as e:
            logger.error(f"Trading hours check failed: {e}")
            return False
    
    def _check_restricted_securities(self, market_data: Dict[str, Any]) -> bool:
        """Check if securities are restricted"""
        try:
            # Placeholder - would check against restricted securities list
            return True
        except Exception as e:
            logger.error(f"Restricted securities check failed: {e}")
            return False
    
    def _symbolic_recommendation_generation(self, market_regime: Dict, technical_signals: Dict, 
                                          risk_assessment: Dict, compliance_check: Dict, 
                                          ai_prediction: Dict) -> Dict[str, Any]:
        """Generate trading recommendation using symbolic reasoning"""
        try:
            # Extract key information
            regime = market_regime.get("regime", "unknown")
            signal = technical_signals.get("signal", "wait")
            risk_level = risk_assessment.get("risk_level", "unknown")
            compliant = compliance_check.get("compliant", False)
            
            # Symbolic reasoning rules
            if not compliant:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "reasoning": "compliance_violation",
                    "conditions": ["compliance_check_failed"]
                }
            
            if risk_level in ["high", "extreme"]:
                return {
                    "action": "hold",
                    "confidence": 0.8,
                    "reasoning": "high_risk_environment",
                    "conditions": ["risk_limit_exceeded"]
                }
            
            # Combine signals
            if signal == "buy" and regime in ["trending", "ranging"]:
                return {
                    "action": "buy",
                    "confidence": 0.7,
                    "reasoning": "positive_technical_signals",
                    "conditions": ["buy_signal", "favorable_regime"]
                }
            elif signal == "sell" and regime in ["trending", "ranging"]:
                return {
                    "action": "sell",
                    "confidence": 0.7,
                    "reasoning": "negative_technical_signals",
                    "conditions": ["sell_signal", "favorable_regime"]
                }
            else:
                return {
                    "action": "hold",
                    "confidence": 0.6,
                    "reasoning": "wait_for_better_conditions",
                    "conditions": ["neutral_signals"]
                }
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {"action": "hold", "confidence": 0.0, "reasoning": "error"}
    
    def _calculate_regime_confidence(self, volatility: float, trend_strength: float) -> float:
        """Calculate confidence in market regime classification"""
        try:
            # Higher confidence for clearer patterns
            volatility_factor = 1.0 - min(volatility * 10, 1.0)
            trend_factor = min(trend_strength * 100, 1.0)
            
            confidence = (volatility_factor + trend_factor) / 2
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Regime confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_signal_confidence(self, ma_short: float, ma_long: float, rsi: float, symbol: str = None) -> float:
        """Calculate confidence in technical signals with symbol-specific factors"""
        try:
            # Factor 1: Moving average crossover strength
            ma_diff = abs(ma_short - ma_long) / max(ma_long, 1e-8)
            ma_confidence = min(0.95, ma_diff * 10)  # Scale up the difference
            
            # Factor 2: RSI extremity (more extreme = higher confidence)
            rsi_extreme = abs(rsi - 50) / 50
            rsi_confidence = min(0.95, rsi_extreme)
            
            # Factor 3: Signal alignment (when MA and RSI agree)
            signal_alignment = 0.0
            if (ma_short > ma_long and rsi > 50) or (ma_short < ma_long and rsi < 50):
                signal_alignment = 0.3  # Bonus for aligned signals
            
            # Factor 4: RSI extreme conditions (oversold/overbought)
            rsi_extreme_bonus = 0.0
            if rsi < 30 or rsi > 70:
                rsi_extreme_bonus = 0.2  # Bonus for extreme RSI
            
            # Factor 5: Symbol-specific confidence adjustment
            symbol_factor = self._calculate_symbol_signal_factor(symbol, ma_short, ma_long, rsi)
            
            # Combine factors
            confidence = (
                0.32 * ma_confidence +
                0.22 * rsi_confidence +
                0.15 * signal_alignment +
                0.10 * rsi_extreme_bonus +
                0.12 * symbol_factor
            )
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Signal confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_symbol_signal_factor(self, symbol: str, ma_short: float, ma_long: float, rsi: float) -> float:
        """Calculate symbol-specific signal confidence factor"""
        try:
            if not symbol:
                return 0.5
            
            # Create symbol-specific variations
            symbol_hash = hash(symbol) % 1000
            
            # Base factor from symbol characteristics
            base_factor = 0.4 + (symbol_hash / 1000.0) * 0.3  # Range: 0.4 to 0.7
            
            # Adjust based on symbol type (tech stocks vs traditional)
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:  # Tech stocks
                base_factor += 0.1  # Tech stocks tend to have clearer signals
            elif symbol in ['JNJ', 'PG', 'KO', 'WMT']:  # Defensive stocks
                base_factor -= 0.05  # More stable, less clear signals
            
            # Adjust based on MA relationship
            # magnitude of MA separation contributes smoothly
            sep = abs(ma_short - ma_long) / max(ma_long, 1e-8)
            base_factor += max(-0.05, min(0.15, sep * 0.5))
            
            # Adjust based on RSI level
            if rsi > 70 or rsi < 30:  # Extreme RSI
                base_factor += 0.08
            elif 40 < rsi < 60:  # Neutral RSI
                base_factor -= 0.04
            
            return max(0.1, min(0.95, base_factor))
            
        except Exception as e:
            logger.error(f"Symbol signal factor calculation failed: {e}")
            return 0.5
    
    def _calculate_risk_confidence(self, volatility: float, var_95: float, max_dd: float) -> float:
        """Calculate confidence in risk assessment"""
        try:
            # Higher confidence for more extreme risk metrics
            vol_factor = min(volatility * 10, 1.0)
            var_factor = min(var_95 * 50, 1.0)
            dd_factor = min(max_dd * 5, 1.0)
            
            confidence = (vol_factor + var_factor + dd_factor) / 3
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Risk confidence calculation failed: {e}")
            return 0.5
    
    def _update_metrics(self, reasoning_time: float):
        """Update performance metrics"""
        try:
            self.metrics["reasoning_sessions"] += 1
            
            # Update average reasoning time
            current_avg = self.metrics["avg_reasoning_time"]
            count = self.metrics["reasoning_sessions"]
            self.metrics["avg_reasoning_time"] = (current_avg * (count - 1) + reasoning_time) / count
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get symbolic reasoner metrics"""
        return self.metrics.copy()
    
    def add_rule(self, rule_name: str, rule_definition: Dict[str, Any]) -> bool:
        """Add a new symbolic rule"""
        try:
            self.rules[rule_name] = rule_definition
            logger.info(f"Added rule: {rule_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add rule: {e}")
            return False
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a symbolic rule"""
        try:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed rule: {rule_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove rule: {e}")
            return False
    
    def _load_default_rules(self):
        """Load default rule pack if available"""
        try:
            if self.rule_loader:
                available_packs = self.rule_loader.list_rule_packs()
                if available_packs:
                    # Load the first available rule pack
                    default_pack = available_packs[0]
                    self.load_rule_pack(default_pack)
                    logger.info(f"Loaded default rule pack: {default_pack}")
        except Exception as e:
            logger.warning(f"Failed to load default rules: {e}")
    
    def load_rule_pack(self, rule_pack_name: str) -> bool:
        """Load a rule pack from the rule loader"""
        try:
            if not self.rule_loader:
                logger.warning("No rule loader available")
                return False
            
            rule_pack = self.rule_loader.get_rule_pack(rule_pack_name)
            if rule_pack:
                self.active_rule_pack = rule_pack_name
                self.metrics["rule_pack_usage"] += 1
                logger.info(f"Loaded rule pack: {rule_pack_name}")
                return True
            else:
                logger.error(f"Rule pack not found: {rule_pack_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load rule pack {rule_pack_name}: {e}")
            return False
    
    def get_active_rule_pack(self) -> Optional[str]:
        """Get the name of the currently active rule pack"""
        return self.active_rule_pack
    
    def list_available_rule_packs(self) -> List[str]:
        """List available rule packs"""
        if self.rule_loader:
            return self.rule_loader.list_rule_packs()
        return []
    
    def evaluate_market_regime_rules(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market regime using loaded rule packs"""
        try:
            if not self.active_rule_pack or not self.rule_loader:
                return {"regime": "unknown", "confidence": 0.0, "rule_source": "hardcoded"}
            
            self.metrics["dynamic_rule_evaluations"] += 1
            
            # Get market regime rules from active rule pack
            regime_rules = self.rule_loader.get_market_regime_rules(self.active_rule_pack)
            
            # Extract market characteristics
            price_data = self._extract_price_data(market_data)
            volume_data = self._extract_volume_data(market_data)
            
            if not price_data or len(price_data) < 20:
                return {"regime": "unknown", "confidence": 0.0, "rule_source": "rule_pack"}
            
            # Calculate market characteristics
            price_changes = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]
            volatility = self._calculate_volatility(price_changes)
            trend_strength = self._calculate_trend_strength(price_data)
            
            # Evaluate against rule pack regimes
            regime_result = self._evaluate_regime_rules(regime_rules, volatility, trend_strength, volume_data)
            
            return {
                **regime_result,
                "rule_source": "rule_pack",
                "rule_pack": self.active_rule_pack
            }
            
        except Exception as e:
            logger.error(f"Market regime rule evaluation failed: {e}")
            return {"regime": "unknown", "confidence": 0.0, "rule_source": "error"}
    
    def evaluate_technical_signal_rules(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate technical signals using loaded rule packs"""
        try:
            if not self.active_rule_pack or not self.rule_loader:
                return {"signal": "wait", "confidence": 0.0, "rule_source": "hardcoded"}
            
            self.metrics["dynamic_rule_evaluations"] += 1
            
            # Get technical signal rules from active rule pack
            signal_rules = self.rule_loader.get_technical_signal_rules(self.active_rule_pack)
            
            # Extract market data
            price_data = self._extract_price_data(market_data)
            
            if not price_data or len(price_data) < 20:
                return {"signal": "wait", "confidence": 0.0, "rule_source": "rule_pack"}
            
            # Calculate technical indicators
            ma_short = self._calculate_moving_average(price_data, 5)
            ma_long = self._calculate_moving_average(price_data, 20)
            rsi = self._calculate_rsi(price_data)
            
            # Evaluate against rule pack signals
            signal_result = self._evaluate_signal_rules(signal_rules, ma_short, ma_long, rsi, price_data)
            
            return {
                **signal_result,
                "rule_source": "rule_pack",
                "rule_pack": self.active_rule_pack
            }
            
        except Exception as e:
            logger.error(f"Technical signal rule evaluation failed: {e}")
            return {"signal": "wait", "confidence": 0.0, "rule_source": "error"}
    
    def _evaluate_regime_rules(self, regime_rules: Dict[str, Any], volatility: float, trend_strength: float, volume_data: List[float]) -> Dict[str, Any]:
        """Evaluate market regime against rule pack rules"""
        try:
            best_regime = None
            best_confidence = 0.0
            best_match_score = 0.0
            
            # Track all evaluated regimes for logging
            evaluated_regimes = []
            
            for regime_name, regime_data in regime_rules.items():
                if 'conditions' not in regime_data:
                    continue
                
                # Calculate match score for this regime
                match_score = self._calculate_regime_match_score(regime_data['conditions'], volatility, trend_strength, volume_data)
                
                # Log regime evaluation
                regime_eval = {
                    "regime_name": regime_name,
                    "match_score": match_score,
                    "confidence_threshold": regime_data.get('confidence_threshold', 0.5),
                    "conditions": regime_data.get('conditions', [])
                }
                evaluated_regimes.append(regime_eval)
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_regime = regime_name
                    best_confidence = regime_data.get('confidence_threshold', 0.5)
            
            # Log all regime evaluations for traceability
            logger.info(f"Regime rule evaluation completed. Evaluated {len(evaluated_regimes)} regimes")
            for regime_eval in evaluated_regimes:
                logger.debug(f"Regime {regime_eval['regime_name']}: match_score={regime_eval['match_score']:.3f}, "
                           f"confidence_threshold={regime_eval['confidence_threshold']:.3f}")
            
            if best_regime:
                # Calculate confidence based on match score and rule confidence
                calculated_confidence = max(0.1, min(0.95, best_match_score * best_confidence))
                
                # Log the selected regime
                logger.info(f"Selected regime: {best_regime} (match_score={best_match_score:.3f}, "
                           f"confidence={calculated_confidence:.3f})")
                
                return {
                    "regime": best_regime,
                    "confidence": calculated_confidence,
                    "match_score": best_match_score,
                    "rule_confidence": best_confidence,
                    "evaluated_regimes": evaluated_regimes  # Include all evaluated regimes for traceability
                }
            else:
                logger.info("No matching regimes found, returning unknown regime")
                return {
                    "regime": "unknown", 
                    "confidence": 0.0, 
                    "match_score": 0.0,
                    "evaluated_regimes": evaluated_regimes
                }
                
        except Exception as e:
            logger.error(f"Regime rule evaluation failed: {e}")
            return {"regime": "unknown", "confidence": 0.0, "match_score": 0.0}
    
    def _evaluate_signal_rules(self, signal_rules: Dict[str, Any], ma_short: float, ma_long: float, rsi: float, price_data: List[float]) -> Dict[str, Any]:
        """Evaluate technical signals against rule pack rules"""
        try:
            best_signal = None
            best_confidence = 0.0
            best_match_score = 0.0
            best_priority = 0
            best_rule_id = None
            best_signal_name = None
            best_rule_name = None
            
            # Track all evaluated rules for logging
            evaluated_rules = []
            
            for signal_name, signal_data in signal_rules.items():
                if 'rules' not in signal_data:
                    continue
                
                for rule_name, rule_data in signal_data['rules'].items():
                    # Create unique rule ID
                    rule_id = f"{signal_name}.{rule_name}"
                    
                    # Calculate match score for this rule
                    match_score = self._calculate_signal_match_score(rule_data, ma_short, ma_long, rsi, price_data)
                    
                    # Log rule evaluation
                    rule_eval = {
                        "rule_id": rule_id,
                        "signal_name": signal_name,
                        "rule_name": rule_name,
                        "match_score": match_score,
                        "condition": rule_data.get('condition', ''),
                        "signal": rule_data.get('signal', 'wait'),
                        "rule_confidence": rule_data.get('confidence', 0.5)
                    }
                    evaluated_rules.append(rule_eval)
                    
                    if match_score > 0:  # Only consider rules that match
                        # Calculate priority: RSI extreme conditions get higher priority
                        priority = 0
                        if 'rsi < 30' in rule_data.get('condition', '') and rsi < 30:
                            priority = 3  # Highest priority for oversold
                        elif 'rsi > 70' in rule_data.get('condition', '') and rsi > 70:
                            priority = 3  # Highest priority for overbought
                        elif 'ma_20 > ma_50' in rule_data.get('condition', '') or 'ma_20 < ma_50' in rule_data.get('condition', ''):
                            priority = 1  # Lower priority for MA crossovers
                        else:
                            priority = 2  # Medium priority for other signals
                        
                        # Select rule with higher priority, or higher match score if priorities are equal
                        if priority > best_priority or (priority == best_priority and match_score > best_match_score):
                            best_priority = priority
                            best_match_score = match_score
                            best_signal = rule_data.get('signal', 'wait')
                            best_confidence = rule_data.get('confidence', 0.5)
                            best_rule_id = rule_id
                            best_signal_name = signal_name
                            best_rule_name = rule_name
            
            # Log all rule evaluations for traceability
            logger.info(f"Signal rule evaluation completed. Evaluated {len(evaluated_rules)} rules")
            for rule_eval in evaluated_rules:
                logger.debug(f"Rule {rule_eval['rule_id']}: match_score={rule_eval['match_score']:.3f}, "
                           f"signal={rule_eval['signal']}, confidence={rule_eval['rule_confidence']:.3f}")
            
            if best_signal:
                # Calculate confidence based on match score and rule confidence
                calculated_confidence = max(0.1, min(0.95, best_match_score * best_confidence))
                
                # Log the selected rule
                logger.info(f"Selected rule: {best_rule_id} (signal={best_signal}, "
                           f"match_score={best_match_score:.3f}, priority={best_priority}, "
                           f"confidence={calculated_confidence:.3f})")
                
                return {
                    "signal": best_signal,
                    "confidence": calculated_confidence,
                    "match_score": best_match_score,
                    "rule_confidence": best_confidence,
                    "priority": best_priority,
                    "rule_id": best_rule_id,
                    "signal_name": best_signal_name,
                    "rule_name": best_rule_name,
                    "evaluated_rules": evaluated_rules  # Include all evaluated rules for traceability
                }
            else:
                logger.info("No matching rules found, returning wait signal")
                return {
                    "signal": "wait", 
                    "confidence": 0.0, 
                    "match_score": 0.0,
                    "rule_id": None,
                    "signal_name": None,
                    "rule_name": None,
                    "evaluated_rules": evaluated_rules
                }
                
        except Exception as e:
            logger.error(f"Signal rule evaluation failed: {e}")
            return {"signal": "wait", "confidence": 0.0, "match_score": 0.0}
    
    def _calculate_regime_match_score(self, conditions: List[Dict[str, Any]], volatility: float, trend_strength: float, volume_data: List[float]) -> float:
        """Calculate how well market data matches regime conditions"""
        try:
            match_score = 0.0
            total_conditions = len(conditions)
            
            if total_conditions == 0:
                return 0.0
            
            for condition in conditions:
                # Handle the rule pack structure: each condition is a dict with one key-value pair
                for condition_type, condition_value in condition.items():
                    if condition_type == 'price_momentum':
                        if condition_value == 'positive' and trend_strength > 0.1:
                            match_score += 1.0
                        elif condition_value == 'negative' and trend_strength < -0.1:
                            match_score += 1.0
                        elif condition_value == 'neutral' and abs(trend_strength) <= 0.1:
                            match_score += 1.0
                    
                    elif condition_type == 'volatility':
                        if condition_value == 'moderate_to_high' and 0.1 < volatility < 0.5:
                            match_score += 1.0
                        elif condition_value == 'high' and volatility > 0.3:
                            match_score += 1.0
                        elif condition_value == 'low' and volatility < 0.1:
                            match_score += 1.0
                        elif condition_value == 'moderate' and 0.05 < volatility < 0.2:
                            match_score += 1.0
                    
                    elif condition_type == 'volume_trend':
                        if volume_data and len(volume_data) > 10:
                            recent_volume = sum(volume_data[-10:]) / 10
                            earlier_volume = sum(volume_data[-20:-10]) / 10
                            volume_trend = recent_volume / earlier_volume if earlier_volume > 0 else 1.0
                            
                            if condition_value == 'increasing' and volume_trend > 1.1:
                                match_score += 1.0
                            elif condition_value == 'decreasing' and volume_trend < 0.9:
                                match_score += 1.0
                            elif condition_value == 'stable' and 0.9 <= volume_trend <= 1.1:
                                match_score += 1.0
                    
                    elif condition_type == 'moving_averages':
                        # This would need MA data, for now give partial credit
                        if condition_value in ['bullish_crossover', 'bearish_crossover']:
                            match_score += 0.5
            
            return match_score / total_conditions if total_conditions > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Regime match score calculation failed: {e}")
            return 0.0
    
    def _calculate_signal_match_score(self, rule_data: Dict[str, Any], ma_short: float, ma_long: float, rsi: float, price_data: List[float]) -> float:
        """Calculate how well market data matches signal rule conditions"""
        try:
            match_score = 0.0
            
            if 'condition' in rule_data:
                condition = rule_data['condition']
                
                # Moving average crossover
                if 'ma_20 > ma_50' in condition:
                    if ma_short > ma_long:  # Using 5-day as short MA
                        match_score += 1.0
                
                if 'ma_20 < ma_50' in condition:
                    if ma_short < ma_long:
                        match_score += 1.0
                
                # RSI conditions
                if 'rsi < 30' in condition and rsi < 30:
                    match_score += 1.0
                
                if 'rsi > 70' in condition and rsi > 70:
                    match_score += 1.0
                
                # Volume conditions
                if 'volume > avg_volume * 2' in condition:
                    if price_data and len(price_data) > 20:
                        recent_volume = price_data[-1] if hasattr(price_data[-1], 'volume') else 1.0
                        avg_volume = sum(price_data[-20:]) / 20 if hasattr(price_data[0], 'volume') else 1.0
                        if recent_volume > avg_volume * 2:
                            match_score += 1.0
            
            return match_score
            
        except Exception as e:
            logger.error(f"Signal match score calculation failed: {e}")
            return 0.0
    
    # ===== REASONING TRACE METHODS =====
    
    def start_reasoning_trace(self, session_id: str, market_data: Dict[str, Any]) -> str:
        """Start a new reasoning trace session"""
        try:
            trace_id = f"trace_{session_id}_{int(time.time())}"
            self.current_trace = {
                "trace_id": trace_id,
                "session_id": session_id,
                "start_time": datetime.now().isoformat(),
                "market_data_summary": self._summarize_market_data(market_data),
                "reasoning_steps": [],
                "rule_evaluations": [],
                "decisions": [],
                "performance_metrics": {}
            }
            logger.info(f"Started reasoning trace: {trace_id}")
            return trace_id
        except Exception as e:
            logger.error(f"Failed to start reasoning trace: {e}")
            return None
    
    def add_reasoning_step(self, step_type: str, description: str, data: Dict[str, Any] = None, 
                          rule_applied: str = None, confidence: float = None):
        """Add a reasoning step to the current trace"""
        try:
            if self.current_trace and self.trace_enabled:
                step = {
                    "timestamp": datetime.now().isoformat(),
                    "step_type": step_type,
                    "description": description,
                    "data": data or {},
                    "rule_applied": rule_applied,
                    "confidence": confidence
                }
                self.current_trace["reasoning_steps"].append(step)
                logger.debug(f"Added reasoning step: {step_type} - {description}")
        except Exception as e:
            logger.error(f"Failed to add reasoning step: {e}")
    
    def add_rule_evaluation(self, rule_name: str, rule_type: str, input_data: Dict[str, Any], 
                           output: Dict[str, Any], match_score: float, applied: bool):
        """Add a rule evaluation to the current trace"""
        try:
            if self.current_trace and self.trace_enabled:
                evaluation = {
                    "timestamp": datetime.now().isoformat(),
                    "rule_name": rule_name,
                    "rule_type": rule_type,
                    "input_data": input_data,
                    "output": output,
                    "match_score": match_score,
                    "applied": applied
                }
                self.current_trace["rule_evaluations"].append(evaluation)
                logger.debug(f"Added rule evaluation: {rule_name} (applied: {applied})")
        except Exception as e:
            logger.error(f"Failed to add rule evaluation: {e}")
    
    def add_decision(self, decision_type: str, decision: str, confidence: float, 
                    reasoning: List[str], alternatives: List[str] = None):
        """Add a decision to the current trace"""
        try:
            if self.current_trace and self.trace_enabled:
                decision_record = {
                    "timestamp": datetime.now().isoformat(),
                    "decision_type": decision_type,
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "alternatives": alternatives or []
                }
                self.current_trace["decisions"].append(decision_record)
                logger.debug(f"Added decision: {decision_type} - {decision}")
        except Exception as e:
            logger.error(f"Failed to add decision: {e}")
    
    def end_reasoning_trace(self, final_decision: Dict[str, Any], performance_metrics: Dict[str, Any] = None):
        """End the current reasoning trace and store it"""
        try:
            if self.current_trace:
                self.current_trace["end_time"] = datetime.now().isoformat()
                self.current_trace["final_decision"] = final_decision
                self.current_trace["performance_metrics"] = performance_metrics or {}
                
                # Calculate trace duration
                start_time = datetime.fromisoformat(self.current_trace["start_time"])
                end_time = datetime.fromisoformat(self.current_trace["end_time"])
                duration = (end_time - start_time).total_seconds()
                self.current_trace["duration_seconds"] = duration
                
                # Store the completed trace
                self.reasoning_traces.append(self.current_trace)
                
                # Update metrics
                self.metrics["reasoning_sessions"] += 1
                if self.metrics["avg_reasoning_time"] == 0:
                    self.metrics["avg_reasoning_time"] = duration
                else:
                    self.metrics["avg_reasoning_time"] = (self.metrics["avg_reasoning_time"] + duration) / 2
                
                logger.info(f"Completed reasoning trace: {self.current_trace['trace_id']} (duration: {duration:.3f}s)")
                
                # Clear current trace
                trace_id = self.current_trace["trace_id"]
                self.current_trace = None
                return trace_id
        except Exception as e:
            logger.error(f"Failed to end reasoning trace: {e}")
            return None
    
    def get_reasoning_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific reasoning trace by ID"""
        try:
            for trace in self.reasoning_traces:
                if trace["trace_id"] == trace_id:
                    return trace
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve reasoning trace: {e}")
            return None
    
    def export_reasoning_traces(self, format: str = "json", limit: int = None) -> str:
        """Export reasoning traces in specified format"""
        try:
            traces_to_export = self.reasoning_traces
            if limit:
                traces_to_export = traces_to_export[-limit:]
            
            if format.lower() == "json":
                return json.dumps(traces_to_export, indent=2, default=str)
            elif format.lower() == "yaml":
                import yaml
                return yaml.dump(traces_to_export, default_flow_style=False)
            elif format.lower() == "dot":
                return self._export_to_dot(traces_to_export)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Failed to export reasoning traces: {e}")
            return None
    
    def _export_to_dot(self, traces_to_export: Dict[str, Any]) -> str:
        """Export reasoning traces to DOT format for Graphviz"""
        try:
            dot_content = []
            dot_content.append("digraph ReasoningTrace {")
            dot_content.append("  rankdir=TB;")
            dot_content.append("  node [shape=box, style=filled];")
            dot_content.append("  edge [color=gray];")
            
            for trace_id, trace in traces_to_export.items():
                # Create subgraph for each trace
                dot_content.append(f"  subgraph cluster_{trace_id.replace('-', '_')} {{")
                dot_content.append(f"    label=\"{trace_id}\";")
                dot_content.append(f"    style=filled;")
                dot_content.append(f"    fillcolor=lightgray;")
                
                # Add nodes for each step
                for i, step in enumerate(trace.get("steps", [])):
                    step_id = f"{trace_id.replace('-', '_')}_step_{i}"
                    step_type = step.get("type", "unknown")
                    step_desc = step.get("description", "").replace('"', '\\"')
                    
                    # Color nodes by type
                    color = "lightblue"
                    if step_type == "rule_evaluation":
                        color = "lightgreen"
                    elif step_type == "decision":
                        color = "lightcoral"
                    elif step_type == "error":
                        color = "lightpink"
                    
                    dot_content.append(f"    {step_id} [label=\"{step_type}: {step_desc}\", fillcolor={color}];")
                
                # Add edges between steps
                for i in range(len(trace.get("steps", [])) - 1):
                    current_id = f"{trace_id.replace('-', '_')}_step_{i}"
                    next_id = f"{trace_id.replace('-', '_')}_step_{i+1}"
                    dot_content.append(f"    {current_id} -> {next_id};")
                
                # Add final decision node if present
                if "final_decision" in trace:
                    decision_id = f"{trace_id.replace('-', '_')}_decision"
                    decision = trace["final_decision"]
                    action = decision.get("action", "unknown")
                    confidence = decision.get("confidence", 0.0)
                    dot_content.append(f"    {decision_id} [label=\"Decision: {action} (conf: {confidence:.2f})\", fillcolor=gold];")
                    
                    if trace.get("steps"):
                        last_step_id = f"{trace_id.replace('-', '_')}_step_{len(trace['steps'])-1}"
                        dot_content.append(f"    {last_step_id} -> {decision_id};")
                
                dot_content.append("  }")
            
            dot_content.append("}")
            
            return "\n".join(dot_content)
            
        except Exception as e:
            logger.error(f"Failed to export to DOT format: {e}")
            return f"// Error generating DOT: {e}"
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get a summary of all reasoning traces"""
        try:
            if not self.reasoning_traces:
                return {"message": "No reasoning traces available"}
            
            total_traces = len(self.reasoning_traces)
            avg_duration = sum(t.get("duration_seconds", 0) for t in self.reasoning_traces) / total_traces
            
            # Count decision types
            decision_counts = {}
            for trace in self.reasoning_traces:
                for decision in trace.get("decisions", []):
                    decision_type = decision.get("decision_type", "unknown")
                    decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            
            # Count rule usage
            rule_usage = {}
            for trace in self.reasoning_traces:
                for evaluation in trace.get("rule_evaluations", []):
                    rule_name = evaluation.get("rule_name", "unknown")
                    rule_usage[rule_name] = rule_usage.get(rule_name, 0) + 1
            
            return {
                "total_traces": total_traces,
                "avg_duration_seconds": avg_duration,
                "decision_distribution": decision_counts,
                "rule_usage": rule_usage,
                "recent_traces": [t["trace_id"] for t in self.reasoning_traces[-5:]]
            }
        except Exception as e:
            logger.error(f"Failed to get reasoning summary: {e}")
            return {"error": str(e)}
    
    def _summarize_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of market data for trace tracking"""
        try:
            summary = {}
            
            # Summarize Dgraph data
            if "dgraph" in market_data and market_data["dgraph"]:
                dgraph_data = market_data["dgraph"].get("market_data", [])
                summary["dgraph"] = {
                    "total_records": len(dgraph_data),
                    "symbols": list(set([str(item.get("subject", "")).split("/")[-1].split("_")[0] 
                                       for item in dgraph_data if "subject" in item]))
                }
            
            # Summarize other data sources
            for source in ["neo4j", "jena"]:
                if source in market_data and market_data[source]:
                    summary[source] = {"total_records": len(market_data[source])}
            
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize market data: {e}")
            return {"error": str(e)}
    
    def clear_reasoning_traces(self, older_than_days: int = None):
        """Clear old reasoning traces"""
        try:
            if older_than_days is None:
                self.reasoning_traces.clear()
                logger.info("Cleared all reasoning traces")
            else:
                cutoff_time = datetime.now() - timedelta(days=older_than_days)
                self.reasoning_traces = [
                    trace for trace in self.reasoning_traces
                    if datetime.fromisoformat(trace["start_time"]) > cutoff_time
                ]
                logger.info(f"Cleared reasoning traces older than {older_than_days} days")
        except Exception as e:
            logger.error(f"Failed to clear reasoning traces: {e}") 