#!/usr/bin/env python3
"""
RDF Reasoning Engine - Server-side inference for complex rules
Handles complex reasoning that benefits from RDF/SPARQL inference capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import httpx
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.plugins.sparql import prepareQuery
import numpy as np

logger = logging.getLogger(__name__)

class RDFReasoningEngine:
    """RDF-based reasoning engine for complex rule inference"""
    
    def __init__(self, jena_endpoint: str = "http://localhost:3030/dataset/sparql"):
        self.jena_endpoint = jena_endpoint
        self.health_status = True
        
        # Define namespaces
        self.HFT = Namespace("http://hft.neurosymbolic.ai/ontology#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.OWL = Namespace("http://www.w3.org/2002/07/owl#")
        self.XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
        
        # Complex rule patterns that benefit from RDF inference
        self.complex_rules = {
            "market_regime_inference": self._market_regime_inference_rules(),
            "risk_correlation_rules": self._risk_correlation_rules(),
            "temporal_pattern_rules": self._temporal_pattern_rules(),
            "cross_asset_rules": self._cross_asset_rules()
        }
        
        # Performance metrics
        self.metrics = {
            "rdf_queries_executed": 0,
            "inference_rules_applied": 0,
            "avg_inference_time_ms": 0.0,
            "total_inference_time": 0.0
        }
    
    def is_healthy(self) -> bool:
        """Check if RDF reasoning engine is healthy"""
        return self.health_status
    
    async def perform_rdf_inference(self, market_data: Dict[str, Any], 
                                  rule_type: str, symbol: str = None) -> Dict[str, Any]:
        """Perform RDF-based inference for complex rules"""
        start_time = time.time()
        
        try:
            if rule_type not in self.complex_rules:
                return {"error": f"Unknown rule type: {rule_type}"}
            
            # Convert market data to RDF
            rdf_data = await self._convert_to_rdf(market_data, symbol)
            
            # Execute RDF inference
            inference_result = await self._execute_rdf_inference(rdf_data, rule_type)
            
            # Update metrics
            inference_time = time.time() - start_time
            self._update_metrics(inference_time)
            
            return {
                "rule_type": rule_type,
                "inference_result": inference_result,
                "inference_time_ms": inference_time * 1000,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            }
            
        except Exception as e:
            logger.error(f"RDF inference failed: {e}")
            return {"error": str(e)}
    
    async def _convert_to_rdf(self, market_data: Dict[str, Any], symbol: str = None) -> Graph:
        """Convert market data to RDF format"""
        g = Graph()
        
        # Bind namespaces
        g.bind("hft", self.HFT)
        g.bind("rdf", self.RDF)
        g.bind("rdfs", self.RDFS)
        g.bind("owl", self.OWL)
        g.bind("xsd", self.XSD)
        
        # Create subject URI
        timestamp = datetime.now().isoformat()
        subject_uri = self.HFT[f"MarketData_{symbol}_{timestamp.replace(':', '-')}"]
        
        # Add market data as RDF triples
        g.add((subject_uri, self.RDF.type, self.HFT.MarketData))
        g.add((subject_uri, self.HFT.symbol, Literal(symbol or "UNKNOWN")))
        g.add((subject_uri, self.HFT.timestamp, Literal(timestamp, datatype=self.XSD.dateTime)))
        
        # Add price data
        if "neo4j" in market_data and market_data["neo4j"]:
            for i, record in enumerate(market_data["neo4j"][:10]):  # Limit to recent data
                if "price" in record and record["price"]:
                    price_uri = self.HFT[f"PricePoint_{symbol}_{i}"]
                    g.add((price_uri, self.RDF.type, self.HFT.PricePoint))
                    g.add((price_uri, self.HFT.price, Literal(float(record["price"]), datatype=self.XSD.float)))
                    g.add((price_uri, self.HFT.sequence, Literal(i, datatype=self.XSD.integer)))
                    g.add((subject_uri, self.HFT.hasPricePoint, price_uri))
        
        # Add volume data
        if "neo4j" in market_data and market_data["neo4j"]:
            for i, record in enumerate(market_data["neo4j"][:10]):
                if "volume" in record and record["volume"]:
                    volume_uri = self.HFT[f"VolumePoint_{symbol}_{i}"]
                    g.add((volume_uri, self.RDF.type, self.HFT.VolumePoint))
                    g.add((volume_uri, self.HFT.volume, Literal(float(record["volume"]), datatype=self.XSD.float)))
                    g.add((volume_uri, self.HFT.sequence, Literal(i, datatype=self.XSD.integer)))
                    g.add((subject_uri, self.HFT.hasVolumePoint, volume_uri))
        
        # Add technical indicators
        if "ma_short" in market_data:
            g.add((subject_uri, self.HFT.maShort, Literal(float(market_data["ma_short"]), datatype=self.XSD.float)))
        if "ma_long" in market_data:
            g.add((subject_uri, self.HFT.maLong, Literal(float(market_data["ma_long"]), datatype=self.XSD.float)))
        if "rsi" in market_data:
            g.add((subject_uri, self.HFT.rsi, Literal(float(market_data["rsi"]), datatype=self.XSD.float)))
        if "volatility" in market_data:
            g.add((subject_uri, self.HFT.volatility, Literal(float(market_data["volatility"]), datatype=self.XSD.float)))
        
        return g
    
    async def _execute_rdf_inference(self, rdf_data: Graph, rule_type: str) -> Dict[str, Any]:
        """Execute RDF inference using SPARQL queries"""
        try:
            # Get the appropriate SPARQL query for the rule type
            query = self.complex_rules[rule_type]
            
            # Execute SPARQL query against Jena
            results = await self._execute_sparql_query(query, rdf_data)
            
            # Process results
            inference_result = self._process_inference_results(results, rule_type)
            
            return inference_result
            
        except Exception as e:
            logger.error(f"RDF inference execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_sparql_query(self, query: str, rdf_data: Graph) -> List[Dict[str, Any]]:
        """Execute SPARQL query against Jena endpoint"""
        try:
            # For now, execute against local RDF data
            # In production, this would query the Jena endpoint
            results = rdf_data.query(query)
            
            # Convert results to list of dictionaries
            result_list = []
            for row in results:
                result_dict = {}
                for var in row:
                    result_dict[str(var)] = str(row[var])
                result_list.append(result_dict)
            
            self.metrics["rdf_queries_executed"] += 1
            return result_list
            
        except Exception as e:
            logger.error(f"SPARQL query execution failed: {e}")
            return []
    
    def _process_inference_results(self, results: List[Dict[str, Any]], rule_type: str) -> Dict[str, Any]:
        """Process SPARQL query results into inference conclusions"""
        if not results:
            return {"conclusion": "no_inference", "confidence": 0.0}
        
        if rule_type == "market_regime_inference":
            return self._process_market_regime_inference(results)
        elif rule_type == "risk_correlation_rules":
            return self._process_risk_correlation_inference(results)
        elif rule_type == "temporal_pattern_rules":
            return self._process_temporal_pattern_inference(results)
        elif rule_type == "cross_asset_rules":
            return self._process_cross_asset_inference(results)
        else:
            return {"conclusion": "unknown_rule_type", "confidence": 0.0}
    
    def _process_market_regime_inference(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process market regime inference results"""
        if not results:
            return {"regime": "unknown", "confidence": 0.0, "reasoning": "no_data"}
        
        # Analyze price patterns and volatility
        regimes = []
        confidences = []
        
        for result in results:
            if "regime" in result:
                regimes.append(result["regime"])
            if "confidence" in result:
                confidences.append(float(result["confidence"]))
        
        # Determine dominant regime
        if regimes:
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            dominant_regime = max(regime_counts, key=regime_counts.get)
            confidence = np.mean(confidences) if confidences else 0.5
            
            return {
                "regime": dominant_regime,
                "confidence": confidence,
                "reasoning": f"RDF inference based on {len(results)} patterns",
                "pattern_count": len(results)
            }
        
        return {"regime": "unknown", "confidence": 0.0, "reasoning": "no_patterns_detected"}
    
    def _process_risk_correlation_inference(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process risk correlation inference results"""
        if not results:
            return {"risk_level": "unknown", "confidence": 0.0}
        
        # Analyze risk correlations
        risk_factors = []
        for result in results:
            if "riskFactor" in result:
                risk_factors.append(result["riskFactor"])
        
        # Determine overall risk level
        if risk_factors:
            high_risk_count = sum(1 for factor in risk_factors if "high" in factor.lower())
            total_factors = len(risk_factors)
            
            if high_risk_count / total_factors > 0.6:
                risk_level = "high"
                confidence = 0.8
            elif high_risk_count / total_factors > 0.3:
                risk_level = "medium"
                confidence = 0.6
            else:
                risk_level = "low"
                confidence = 0.7
            
            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "factor_count": total_factors
            }
        
        return {"risk_level": "unknown", "confidence": 0.0}
    
    def _process_temporal_pattern_inference(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process temporal pattern inference results"""
        if not results:
            return {"pattern": "none", "confidence": 0.0}
        
        # Analyze temporal patterns
        patterns = []
        for result in results:
            if "pattern" in result:
                patterns.append(result["pattern"])
        
        if patterns:
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            dominant_pattern = max(pattern_counts, key=pattern_counts.get)
            confidence = len(patterns) / 10.0  # Normalize by expected pattern count
            
            return {
                "pattern": dominant_pattern,
                "confidence": min(confidence, 1.0),
                "pattern_count": len(patterns),
                "all_patterns": patterns
            }
        
        return {"pattern": "none", "confidence": 0.0}
    
    def _process_cross_asset_inference(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process cross-asset inference results"""
        if not results:
            return {"correlation": "none", "confidence": 0.0}
        
        # Analyze cross-asset correlations
        correlations = []
        for result in results:
            if "correlation" in result:
                correlations.append(float(result["correlation"]))
        
        if correlations:
            avg_correlation = np.mean(correlations)
            confidence = min(len(correlations) / 5.0, 1.0)  # Normalize by expected correlation count
            
            if avg_correlation > 0.7:
                correlation_level = "strong_positive"
            elif avg_correlation > 0.3:
                correlation_level = "moderate_positive"
            elif avg_correlation < -0.7:
                correlation_level = "strong_negative"
            elif avg_correlation < -0.3:
                correlation_level = "moderate_negative"
            else:
                correlation_level = "weak"
            
            return {
                "correlation": correlation_level,
                "correlation_value": avg_correlation,
                "confidence": confidence,
                "correlation_count": len(correlations)
            }
        
        return {"correlation": "none", "confidence": 0.0}
    
    def _market_regime_inference_rules(self) -> str:
        """SPARQL query for market regime inference"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?regime ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:maShort ?maShort .
            ?marketData hft:maLong ?maLong .
            
            BIND(IF(?volatility > 0.05, "high_volatility", "low_volatility") AS ?volatilityRegime) .
            BIND(IF(?maShort > ?maLong, "trending_bull", "trending_bear") AS ?trendRegime) .
            
            BIND(IF(?volatilityRegime = "high_volatility" && ?trendRegime = "trending_bull", 
                   "sideways_volatile", 
                   IF(?volatilityRegime = "low_volatility" && ?trendRegime = "trending_bull",
                      "trending_bull",
                      IF(?volatilityRegime = "low_volatility" && ?trendRegime = "trending_bear",
                         "trending_bear",
                         "unknown"))) AS ?regime) .
            
            BIND(0.8 AS ?confidence) .
        }
        """
    
    def _risk_correlation_rules(self) -> str:
        """SPARQL query for risk correlation analysis"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?riskFactor ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:rsi ?rsi .
            
            BIND(IF(?volatility > 0.08, "high_volatility_risk", "normal_volatility") AS ?volatilityRisk) .
            BIND(IF(?rsi > 80, "overbought_risk", IF(?rsi < 20, "oversold_risk", "normal_rsi")) AS ?rsiRisk) .
            
            BIND(IF(?volatilityRisk = "high_volatility_risk" || ?rsiRisk = "overbought_risk" || ?rsiRisk = "oversold_risk",
                   CONCAT(?volatilityRisk, "_", ?rsiRisk), "low_risk") AS ?riskFactor) .
            
            BIND(0.7 AS ?confidence) .
        }
        """
    
    def _temporal_pattern_rules(self) -> str:
        """SPARQL query for temporal pattern analysis"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?pattern ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:hasPricePoint ?pricePoint .
            ?pricePoint hft:price ?price .
            ?pricePoint hft:sequence ?sequence .
            
            # Look for price patterns
            BIND(IF(?sequence > 0, 
                   IF(?price > (SELECT ?prevPrice WHERE { ?prevPricePoint hft:sequence (?sequence - 1) . ?prevPricePoint hft:price ?prevPrice }), 
                      "uptrend", "downtrend"), 
                   "unknown") AS ?trendPattern) .
            
            BIND(IF(?trendPattern = "uptrend", "bullish_pattern", 
                   IF(?trendPattern = "downtrend", "bearish_pattern", "sideways_pattern")) AS ?pattern) .
            
            BIND(0.6 AS ?confidence) .
        }
        """
    
    def _cross_asset_rules(self) -> str:
        """SPARQL query for cross-asset correlation analysis"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?correlation ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:symbol ?symbol .
            ?marketData hft:volatility ?volatility .
            
            # Simulate cross-asset correlation (in real implementation, this would query multiple assets)
            BIND(IF(?symbol = "AAPL", 0.8, 
                   IF(?symbol = "TSLA", 0.6, 
                      IF(?symbol = "MSFT", 0.7, 0.5))) AS ?correlation) .
            
            BIND(0.7 AS ?confidence) .
        }
        """
    
    def _update_metrics(self, inference_time: float):
        """Update performance metrics"""
        self.metrics["inference_rules_applied"] += 1
        self.metrics["total_inference_time"] += inference_time
        self.metrics["avg_inference_time_ms"] = (
            self.metrics["total_inference_time"] / self.metrics["inference_rules_applied"]
        ) * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear performance metrics"""
        self.metrics = {
            "rdf_queries_executed": 0,
            "inference_rules_applied": 0,
            "avg_inference_time_ms": 0.0,
            "total_inference_time": 0.0
        }
