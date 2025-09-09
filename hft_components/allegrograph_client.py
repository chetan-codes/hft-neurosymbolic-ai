#!/usr/bin/env python3
"""
AllegroGraph Client - Advanced RDF reasoning and SPARQL queries
Provides advanced reasoning capabilities while maintaining Neo4j for low-latency traversals
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import httpx
import numpy as np
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.plugins.sparql import prepareQuery

logger = logging.getLogger(__name__)

class AllegroGraphClient:
    """AllegroGraph client for advanced RDF reasoning and SPARQL queries"""
    
    def __init__(self, 
                 ag_endpoint: str = "http://localhost:10035",
                 ag_username: str = "admin",
                 ag_password: str = "admin",
                 ag_catalog: str = "root",
                 ag_repository: str = "hft_reasoning"):
        self.ag_endpoint = ag_endpoint
        self.ag_username = ag_username
        self.ag_password = ag_password
        self.ag_catalog = ag_catalog
        self.ag_repository = ag_repository
        self.health_status = True
        
        # Define namespaces
        self.HFT = Namespace("http://hft.neurosymbolic.ai/ontology#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.OWL = Namespace("http://www.w3.org/2002/07/owl#")
        self.XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
        self.TIME = Namespace("http://www.w3.org/2006/time#")
        self.FOAF = Namespace("http://xmlns.com/foaf/0.1/")
        
        # Advanced reasoning rules
        self.reasoning_rules = {
            "market_regime_ontology": self._market_regime_ontology(),
            "risk_propagation_rules": self._risk_propagation_rules(),
            "temporal_reasoning_rules": self._temporal_reasoning_rules(),
            "cross_asset_correlation_rules": self._cross_asset_correlation_rules(),
            "compliance_inference_rules": self._compliance_inference_rules()
        }
        
        # Performance metrics
        self.metrics = {
            "sparql_queries_executed": 0,
            "reasoning_rules_applied": 0,
            "avg_query_time_ms": 0.0,
            "total_query_time": 0.0,
            "inference_assertions_generated": 0
        }
    
    def is_healthy(self) -> bool:
        """Check if AllegroGraph client is healthy"""
        return self.health_status
    
    async def test_connection(self) -> bool:
        """Test connection to AllegroGraph"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test basic connectivity
                response = await client.get(
                    f"{self.ag_endpoint}/catalogs/{self.ag_catalog}/repositories/{self.ag_repository}",
                    auth=(self.ag_username, self.ag_password)
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"AllegroGraph connection test failed: {e}")
            return False
    
    async def store_market_data(self, market_data: Dict[str, Any], symbol: str = None) -> bool:
        """Store market data in AllegroGraph for reasoning"""
        try:
            # Convert market data to RDF
            rdf_graph = await self._convert_market_data_to_rdf(market_data, symbol)
            
            # Serialize to Turtle format
            turtle_data = rdf_graph.serialize(format='turtle')
            
            # Store in AllegroGraph
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ag_endpoint}/catalogs/{self.ag_catalog}/repositories/{self.ag_repository}/statements",
                    content=turtle_data,
                    headers={"Content-Type": "text/turtle"},
                    auth=(self.ag_username, self.ag_password)
                )
                
                if response.status_code in [200, 201, 204]:
                    logger.info(f"Successfully stored market data for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to store market data: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to store market data in AllegroGraph: {e}")
            return False
    
    async def perform_advanced_reasoning(self, reasoning_type: str, 
                                       market_data: Dict[str, Any], 
                                       symbol: str = None) -> Dict[str, Any]:
        """Perform advanced reasoning using AllegroGraph"""
        start_time = time.time()
        
        try:
            if reasoning_type not in self.reasoning_rules:
                return {"error": f"Unknown reasoning type: {reasoning_type}"}
            
            # Store market data first
            await self.store_market_data(market_data, symbol)
            
            # Execute reasoning query
            query = self.reasoning_rules[reasoning_type]
            results = await self._execute_sparql_query(query)
            
            # Process results
            reasoning_result = self._process_reasoning_results(results, reasoning_type)
            
            # Update metrics
            query_time = time.time() - start_time
            self._update_metrics(query_time)
            
            return {
                "reasoning_type": reasoning_type,
                "results": reasoning_result,
                "query_time_ms": query_time * 1000,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            }
            
        except Exception as e:
            logger.error(f"Advanced reasoning failed: {e}")
            return {"error": str(e)}
    
    async def _convert_market_data_to_rdf(self, market_data: Dict[str, Any], symbol: str = None) -> Graph:
        """Convert market data to RDF format for AllegroGraph"""
        g = Graph()
        
        # Bind namespaces
        g.bind("hft", self.HFT)
        g.bind("rdf", self.RDF)
        g.bind("rdfs", self.RDFS)
        g.bind("owl", self.OWL)
        g.bind("xsd", self.XSD)
        g.bind("time", self.TIME)
        g.bind("foaf", self.FOAF)
        
        # Create subject URI
        timestamp = datetime.now().isoformat()
        subject_uri = self.HFT[f"MarketData_{symbol}_{timestamp.replace(':', '-')}"]
        
        # Add market data as RDF triples
        g.add((subject_uri, self.RDF.type, self.HFT.MarketData))
        g.add((subject_uri, self.HFT.symbol, Literal(symbol or "UNKNOWN")))
        g.add((subject_uri, self.HFT.timestamp, Literal(timestamp, datatype=self.XSD.dateTime)))
        
        # Add price data with temporal ordering
        if "neo4j" in market_data and market_data["neo4j"]:
            for i, record in enumerate(market_data["neo4j"][:20]):  # Limit to recent data
                if "price" in record and record["price"]:
                    price_uri = self.HFT[f"PricePoint_{symbol}_{i}"]
                    g.add((price_uri, self.RDF.type, self.HFT.PricePoint))
                    g.add((price_uri, self.HFT.price, Literal(float(record["price"]), datatype=self.XSD.float)))
                    g.add((price_uri, self.HFT.sequence, Literal(i, datatype=self.XSD.integer)))
                    g.add((price_uri, self.TIME.hasTime, Literal(timestamp, datatype=self.XSD.dateTime)))
                    g.add((subject_uri, self.HFT.hasPricePoint, price_uri))
        
        # Add technical indicators
        if "ma_short" in market_data:
            g.add((subject_uri, self.HFT.maShort, Literal(float(market_data["ma_short"]), datatype=self.XSD.float)))
        if "ma_long" in market_data:
            g.add((subject_uri, self.HFT.maLong, Literal(float(market_data["ma_long"]), datatype=self.XSD.float)))
        if "rsi" in market_data:
            g.add((subject_uri, self.HFT.rsi, Literal(float(market_data["rsi"]), datatype=self.XSD.float)))
        if "volatility" in market_data:
            g.add((subject_uri, self.HFT.volatility, Literal(float(market_data["volatility"]), datatype=self.XSD.float)))
        
        # Add market regime classification
        if "regime" in market_data:
            g.add((subject_uri, self.HFT.marketRegime, Literal(market_data["regime"])))
        
        # Add risk level
        if "risk_level" in market_data:
            g.add((subject_uri, self.HFT.riskLevel, Literal(market_data["risk_level"])))
        
        return g
    
    async def _execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against AllegroGraph"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ag_endpoint}/catalogs/{self.ag_catalog}/repositories/{self.ag_repository}/sparql",
                    data={"query": query},
                    auth=(self.ag_username, self.ag_password)
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    results = result_data.get("results", {}).get("bindings", [])
                    self.metrics["sparql_queries_executed"] += 1
                    return results
                else:
                    logger.error(f"SPARQL query failed: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"SPARQL query execution failed: {e}")
            return []
    
    def _process_reasoning_results(self, results: List[Dict[str, Any]], reasoning_type: str) -> Dict[str, Any]:
        """Process SPARQL query results into reasoning conclusions"""
        if not results:
            return {"conclusion": "no_inference", "confidence": 0.0}
        
        if reasoning_type == "market_regime_ontology":
            return self._process_market_regime_ontology(results)
        elif reasoning_type == "risk_propagation_rules":
            return self._process_risk_propagation(results)
        elif reasoning_type == "temporal_reasoning_rules":
            return self._process_temporal_reasoning(results)
        elif reasoning_type == "cross_asset_correlation_rules":
            return self._process_cross_asset_correlation(results)
        elif reasoning_type == "compliance_inference_rules":
            return self._process_compliance_inference(results)
        else:
            return {"conclusion": "unknown_reasoning_type", "confidence": 0.0}
    
    def _process_market_regime_ontology(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process market regime ontology reasoning results"""
        if not results:
            return {"regime": "unknown", "confidence": 0.0, "reasoning": "no_data"}
        
        # Analyze regime classifications
        regimes = []
        confidences = []
        reasoning_chains = []
        
        for result in results:
            if "regime" in result:
                regimes.append(result["regime"]["value"])
            if "confidence" in result:
                confidences.append(float(result["confidence"]["value"]))
            if "reasoning" in result:
                reasoning_chains.append(result["reasoning"]["value"])
        
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
                "reasoning_chains": reasoning_chains,
                "pattern_count": len(results),
                "reasoning_type": "ontology_based"
            }
        
        return {"regime": "unknown", "confidence": 0.0, "reasoning": "no_patterns_detected"}
    
    def _process_risk_propagation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process risk propagation reasoning results"""
        if not results:
            return {"risk_level": "unknown", "confidence": 0.0}
        
        # Analyze risk propagation
        risk_factors = []
        propagation_paths = []
        
        for result in results:
            if "riskFactor" in result:
                risk_factors.append(result["riskFactor"]["value"])
            if "propagationPath" in result:
                propagation_paths.append(result["propagationPath"]["value"])
        
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
                "propagation_paths": propagation_paths,
                "factor_count": total_factors
            }
        
        return {"risk_level": "unknown", "confidence": 0.0}
    
    def _process_temporal_reasoning(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process temporal reasoning results"""
        if not results:
            return {"pattern": "none", "confidence": 0.0}
        
        # Analyze temporal patterns
        patterns = []
        temporal_relations = []
        
        for result in results:
            if "pattern" in result:
                patterns.append(result["pattern"]["value"])
            if "temporalRelation" in result:
                temporal_relations.append(result["temporalRelation"]["value"])
        
        if patterns:
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            dominant_pattern = max(pattern_counts, key=pattern_counts.get)
            confidence = len(patterns) / 10.0  # Normalize by expected pattern count
            
            return {
                "pattern": dominant_pattern,
                "confidence": min(confidence, 1.0),
                "temporal_relations": temporal_relations,
                "pattern_count": len(patterns)
            }
        
        return {"pattern": "none", "confidence": 0.0}
    
    def _process_cross_asset_correlation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process cross-asset correlation reasoning results"""
        if not results:
            return {"correlation": "none", "confidence": 0.0}
        
        # Analyze cross-asset correlations
        correlations = []
        correlation_types = []
        
        for result in results:
            if "correlation" in result:
                correlations.append(float(result["correlation"]["value"]))
            if "correlationType" in result:
                correlation_types.append(result["correlationType"]["value"])
        
        if correlations:
            avg_correlation = np.mean(correlations)
            confidence = min(len(correlations) / 5.0, 1.0)
            
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
                "correlation_types": correlation_types,
                "confidence": confidence,
                "correlation_count": len(correlations)
            }
        
        return {"correlation": "none", "confidence": 0.0}
    
    def _process_compliance_inference(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process compliance inference results"""
        if not results:
            return {"compliant": False, "confidence": 0.0}
        
        # Analyze compliance rules
        compliance_status = []
        violations = []
        
        for result in results:
            if "compliant" in result:
                compliance_status.append(result["compliant"]["value"].lower() == "true")
            if "violation" in result:
                violations.append(result["violation"]["value"])
        
        if compliance_status:
            compliant_count = sum(compliance_status)
            total_rules = len(compliance_status)
            
            is_compliant = compliant_count / total_rules > 0.8
            confidence = compliant_count / total_rules
            
            return {
                "compliant": is_compliant,
                "confidence": confidence,
                "violations": violations,
                "rule_count": total_rules
            }
        
        return {"compliant": False, "confidence": 0.0}
    
    def _market_regime_ontology(self) -> str:
        """SPARQL query for market regime ontology reasoning"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX time: <http://www.w3.org/2006/time#>
        
        SELECT ?regime ?confidence ?reasoning WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:maShort ?maShort .
            ?marketData hft:maLong ?maLong .
            ?marketData hft:rsi ?rsi .
            
            # Ontology-based reasoning
            BIND(IF(?volatility > 0.05, "high_volatility", "low_volatility") AS ?volatilityClass) .
            BIND(IF(?maShort > ?maLong, "trending_bull", "trending_bear") AS ?trendClass) .
            BIND(IF(?rsi > 70, "overbought", IF(?rsi < 30, "oversold", "neutral")) AS ?rsiClass) .
            
            # Complex regime classification using ontology rules
            BIND(IF(?volatilityClass = "high_volatility" && ?trendClass = "trending_bull" && ?rsiClass = "overbought",
                   "sideways_volatile_bullish",
                   IF(?volatilityClass = "high_volatility" && ?trendClass = "trending_bear" && ?rsiClass = "oversold",
                      "sideways_volatile_bearish",
                      IF(?volatilityClass = "low_volatility" && ?trendClass = "trending_bull",
                         "trending_bull",
                         IF(?volatilityClass = "low_volatility" && ?trendClass = "trending_bear",
                            "trending_bear",
                            "unknown")))) AS ?regime) .
            
            BIND(0.9 AS ?confidence) .
            BIND(CONCAT("Ontology reasoning: ", ?volatilityClass, " + ", ?trendClass, " + ", ?rsiClass) AS ?reasoning) .
        }
        """
    
    def _risk_propagation_rules(self) -> str:
        """SPARQL query for risk propagation analysis"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?riskFactor ?propagationPath ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:rsi ?rsi .
            ?marketData hft:maShort ?maShort .
            ?marketData hft:maLong ?maLong .
            
            # Risk propagation analysis
            BIND(IF(?volatility > 0.08, "high_volatility_risk", "normal_volatility") AS ?volatilityRisk) .
            BIND(IF(?rsi > 80, "overbought_risk", IF(?rsi < 20, "oversold_risk", "normal_rsi")) AS ?rsiRisk) .
            BIND(IF(ABS(?maShort - ?maLong) / ?maLong > 0.05, "divergence_risk", "normal_ma") AS ?maRisk) .
            
            # Risk propagation paths
            BIND(IF(?volatilityRisk = "high_volatility_risk" && ?rsiRisk = "overbought_risk",
                   "volatility_to_rsi_propagation",
                   IF(?rsiRisk = "oversold_risk" && ?maRisk = "divergence_risk",
                      "rsi_to_ma_propagation",
                      "no_propagation")) AS ?propagationPath) .
            
            BIND(CONCAT(?volatilityRisk, "_", ?rsiRisk, "_", ?maRisk) AS ?riskFactor) .
            BIND(0.8 AS ?confidence) .
        }
        """
    
    def _temporal_reasoning_rules(self) -> str:
        """SPARQL query for temporal reasoning"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?pattern ?temporalRelation ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:hasPricePoint ?pricePoint .
            ?pricePoint hft:price ?price .
            ?pricePoint hft:sequence ?sequence .
            ?pricePoint time:hasTime ?timestamp .
            
            # Temporal pattern analysis
            BIND(IF(?sequence > 0, 
                   IF(?price > (SELECT ?prevPrice WHERE { 
                       ?prevPricePoint hft:sequence (?sequence - 1) . 
                       ?prevPricePoint hft:price ?prevPrice 
                   }), 
                      "uptrend", "downtrend"), 
                   "unknown") AS ?trendPattern) .
            
            # Temporal relations
            BIND(IF(?trendPattern = "uptrend", "before_after_positive",
                   IF(?trendPattern = "downtrend", "before_after_negative", "temporal_unknown")) AS ?temporalRelation) .
            
            BIND(IF(?trendPattern = "uptrend", "bullish_temporal_pattern", 
                   IF(?trendPattern = "downtrend", "bearish_temporal_pattern", "sideways_temporal_pattern")) AS ?pattern) .
            
            BIND(0.7 AS ?confidence) .
        }
        """
    
    def _cross_asset_correlation_rules(self) -> str:
        """SPARQL query for cross-asset correlation analysis"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?correlation ?correlationType ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:symbol ?symbol .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:rsi ?rsi .
            
            # Cross-asset correlation simulation (in real implementation, this would query multiple assets)
            BIND(IF(?symbol = "AAPL", 0.8, 
                   IF(?symbol = "TSLA", 0.6, 
                      IF(?symbol = "MSFT", 0.7, 
                         IF(?symbol = "GOOGL", 0.75, 0.5)))) AS ?correlation) .
            
            BIND(IF(?correlation > 0.7, "strong_positive",
                   IF(?correlation > 0.3, "moderate_positive",
                      IF(?correlation < -0.7, "strong_negative",
                         IF(?correlation < -0.3, "moderate_negative", "weak")))) AS ?correlationType) .
            
            BIND(0.8 AS ?confidence) .
        }
        """
    
    def _compliance_inference_rules(self) -> str:
        """SPARQL query for compliance inference"""
        return """
        PREFIX hft: <http://hft.neurosymbolic.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?compliant ?violation ?confidence WHERE {
            ?marketData rdf:type hft:MarketData .
            ?marketData hft:volatility ?volatility .
            ?marketData hft:rsi ?rsi .
            ?marketData hft:timestamp ?timestamp .
            
            # Compliance rules
            BIND(IF(?volatility < 0.1, "true", "false") AS ?volatilityCompliant) .
            BIND(IF(?rsi > 20 && ?rsi < 80, "true", "false") AS ?rsiCompliant) .
            
            # Time-based compliance (simplified)
            BIND(IF(CONTAINS(?timestamp, "T09:") || CONTAINS(?timestamp, "T10:") || 
                   CONTAINS(?timestamp, "T11:") || CONTAINS(?timestamp, "T14:") || 
                   CONTAINS(?timestamp, "T15:"), "true", "false") AS ?timeCompliant) .
            
            # Overall compliance
            BIND(IF(?volatilityCompliant = "true" && ?rsiCompliant = "true" && ?timeCompliant = "true",
                   "true", "false") AS ?compliant) .
            
            BIND(IF(?compliant = "false", 
                   CONCAT("Violations: ", 
                          IF(?volatilityCompliant = "false", "volatility ", ""),
                          IF(?rsiCompliant = "false", "rsi ", ""),
                          IF(?timeCompliant = "false", "time ", "")), 
                   "none") AS ?violation) .
            
            BIND(0.9 AS ?confidence) .
        }
        """
    
    def _update_metrics(self, query_time: float):
        """Update performance metrics"""
        self.metrics["reasoning_rules_applied"] += 1
        self.metrics["total_query_time"] += query_time
        self.metrics["avg_query_time_ms"] = (
            self.metrics["total_query_time"] / self.metrics["reasoning_rules_applied"]
        ) * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear performance metrics"""
        self.metrics = {
            "sparql_queries_executed": 0,
            "reasoning_rules_applied": 0,
            "avg_query_time_ms": 0.0,
            "total_query_time": 0.0,
            "inference_assertions_generated": 0
        }
