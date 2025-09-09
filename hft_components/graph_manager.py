#!/usr/bin/env python3
"""
Graph Manager - Handles all graph database operations
Integrates Dgraph, Neo4j, and Apache Jena for HFT data
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# Graph database clients
import pydgraph
from neo4j import GraphDatabase
import redis
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages multiple graph databases for HFT data"""
    
    def __init__(self):
        self.dgraph_client = None
        self.neo4j_driver = None
        self.redis_client = None
        self.jena_sparql = None
        self.health_status = {
            "dgraph": False,
            "neo4j": False,
            "redis": False,
            "jena": False
        }
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connections to all graph databases"""
        try:
            # Dgraph connection
            self.dgraph_client = pydgraph.DgraphClient(
                pydgraph.DgraphClientStub("dgraph:9080")
            )
            self.health_status["dgraph"] = True
            logger.info("Dgraph connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Dgraph: {e}")
        
        try:
            # Neo4j connection
            self.neo4j_driver = GraphDatabase.driver(
                "neo4j://neo4j:7687",
                auth=("neo4j", "hft_password_2025")
            )
            self.health_status["neo4j"] = True
            logger.info("Neo4j connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
        
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host="redis",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.health_status["redis"] = True
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
        
        try:
            # Jena SPARQL connection
            self.jena_sparql = SPARQLWrapper("http://jena:3030/hft_dataset/query")
            self.jena_sparql.setReturnFormat(JSON)
            self.health_status["jena"] = True
            logger.info("Jena SPARQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Jena: {e}")
    
    def is_healthy(self) -> bool:
        """Check if all graph databases are healthy"""
        return all(self.health_status.values())
    
    async def load_rdf_data(self, rdf_file: str) -> Dict[str, Any]:
        """Load RDF data into all graph databases"""
        results = {
            "dgraph": {"status": "failed", "triples": 0},
            "neo4j": {"status": "failed", "triples": 0},
            "jena": {"status": "failed", "triples": 0}
        }
        
        try:
            # Load RDF graph
            g = Graph()
            g.parse(rdf_file, format="turtle")
            total_triples = len(g)
            
            logger.info(f"Loading {total_triples} triples from {rdf_file}")
            
            # Load into Dgraph
            if self.health_status["dgraph"]:
                results["dgraph"] = await self._load_to_dgraph(g)
            
            # Load into Neo4j
            if self.health_status["neo4j"]:
                results["neo4j"] = await self._load_to_neo4j(g)
            
            # Load into Jena
            if self.health_status["jena"]:
                results["jena"] = await self._load_to_jena(g)
            
            # Cache metadata in Redis
            if self.health_status["redis"]:
                self._cache_metadata(rdf_file, total_triples, results)
            
            logger.info(f"RDF data loading completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load RDF data: {e}")
            return results
    
    async def _load_to_dgraph(self, g: Graph) -> Dict[str, Any]:
        """Load RDF data into Dgraph"""
        try:
            # Convert RDF to Dgraph format
            dgraph_data = self._convert_rdf_to_dgraph(g)
            
            # Create transaction and mutate
            txn = self.dgraph_client.txn()
            try:
                response = txn.mutate(set_obj=dgraph_data)
                txn.commit()
                
                return {
                    "status": "success",
                    "triples": len(dgraph_data),
                    "uids": response.uids
                }
            finally:
                txn.discard()
                
        except Exception as e:
            logger.error(f"Failed to load to Dgraph: {e}")
            return {"status": "failed", "triples": 0, "error": str(e)}
    
    async def _load_to_neo4j(self, g: Graph) -> Dict[str, Any]:
        """Load RDF data into Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                # Convert RDF to Cypher
                cypher_queries = self._convert_rdf_to_cypher(g)
                
                triple_count = 0
                for query in cypher_queries:
                    result = session.run(query)
                    triple_count += result.consume().counters.relationships_created
                
                return {
                    "status": "success",
                    "triples": triple_count
                }
                
        except Exception as e:
            logger.error(f"Failed to load to Neo4j: {e}")
            return {"status": "failed", "triples": 0, "error": str(e)}
    
    async def _load_to_jena(self, g: Graph) -> Dict[str, Any]:
        """Load RDF data into Jena"""
        try:
            # Convert to SPARQL UPDATE
            sparql_updates = self._convert_rdf_to_sparql_update(g)
            
            triple_count = 0
            for update in sparql_updates:
                self.jena_sparql.setQuery(update)
                self.jena_sparql.setMethod('POST')
                result = self.jena_sparql.query()
                triple_count += 1  # Simplified count
            
            return {
                "status": "success",
                "triples": triple_count
            }
            
        except Exception as e:
            logger.error(f"Failed to load to Jena: {e}")
            return {"status": "failed", "triples": 0, "error": str(e)}
    
    def _convert_rdf_to_dgraph(self, g: Graph) -> List[Dict]:
        """Convert RDF graph to Dgraph format"""
        dgraph_data = []
        
        for s, p, o in g:
            # Create Dgraph node
            node = {
                "uid": f"_:{hash(s)}",
                "dgraph.type": "StockData",
                "subject": str(s),
                "predicate": str(p),
                "object": str(o) if isinstance(o, URIRef) else str(o),
                "object_type": "uri" if isinstance(o, URIRef) else "literal"
            }
            dgraph_data.append(node)
        
        return dgraph_data
    
    def _convert_rdf_to_cypher(self, g: Graph) -> List[str]:
        """Convert RDF graph to Cypher queries"""
        cypher_queries = []
        
        # Create constraints and indexes
        cypher_queries.append("""
        CREATE CONSTRAINT stock_symbol IF NOT EXISTS
        FOR (s:Stock) REQUIRE s.symbol IS UNIQUE
        """)
        
        # Convert triples to Cypher
        for s, p, o in g:
            if isinstance(s, URIRef) and "company" in str(s):
                # Company node
                symbol = str(s).split("/")[-1]
                cypher_queries.append(f"""
                MERGE (s:Stock {{symbol: '{symbol}'}})
                """)
            elif isinstance(s, URIRef) and "price" in str(s):
                # Price observation
                parts = str(s).split("/")[-1].split("_")
                if len(parts) >= 3:
                    symbol = parts[0]
                    date = f"{parts[1]}-{parts[2]}-{parts[3]}"
                    predicate = str(p).split("/")[-1]
                    value = str(o)
                    
                    cypher_queries.append(f"""
                    MATCH (s:Stock {{symbol: '{symbol}'}})
                    MERGE (p:PriceObservation {{date: '{date}', symbol: '{symbol}'}})
                    MERGE (s)-[:HAS_PRICE]->(p)
                    SET p.{predicate} = '{value}'
                    """)
        
        return cypher_queries
    
    def _convert_rdf_to_sparql_update(self, g: Graph) -> List[str]:
        """Convert RDF graph to SPARQL UPDATE queries"""
        updates = []
        
        # Clear existing data
        updates.append("DELETE { ?s ?p ?o } WHERE { ?s ?p ?o }")
        
        # Insert new data
        for s, p, o in g:
            update = f"""
            INSERT {{
                <{s}> <{p}> {f"<{o}>" if isinstance(o, URIRef) else f'"{o}"'}
            }}
            """
            updates.append(update)
        
        return updates
    
    def _cache_metadata(self, rdf_file: str, triple_count: int, results: Dict):
        """Cache metadata in Redis"""
        try:
            metadata = {
                "file": rdf_file,
                "triple_count": triple_count,
                "load_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                f"rdf_metadata:{rdf_file}",
                3600,  # 1 hour TTL
                json.dumps(metadata)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache metadata: {e}")
    
    async def get_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market data for a symbol from graph databases"""
        try:
            # Try Redis cache first
            if self.health_status["redis"]:
                cached_data = self.redis_client.get(f"market_data:{symbol}:{timeframe}")
                if cached_data:
                    return json.loads(cached_data)
            
            # Query from graph databases
            market_data = {}
            
            # Query Dgraph
            if self.health_status["dgraph"]:
                dgraph_data = await self._query_dgraph_market_data(symbol, timeframe)
                market_data["dgraph"] = dgraph_data
            
            # Query Neo4j
            if self.health_status["neo4j"]:
                neo4j_data = await self._query_neo4j_market_data(symbol, timeframe)
                market_data["neo4j"] = neo4j_data
            
            # Query Jena
            if self.health_status["jena"]:
                jena_data = await self._query_jena_market_data(symbol, timeframe)
                market_data["jena"] = jena_data
            
            # Cache in Redis
            if self.health_status["redis"]:
                self.redis_client.setex(
                    f"market_data:{symbol}:{timeframe}",
                    300,  # 5 minutes TTL
                    json.dumps(market_data)
                )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {}
    
    async def _query_dgraph_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Query market data from Dgraph"""
        try:
            # Query for data with the symbol in the subject
            query = f"""
            {{
                market_data(func: has(predicate)) @filter(regexp(subject, /{symbol}/)) {{
                    subject
                    predicate
                    object
                }}
            }}
            """
            
            txn = self.dgraph_client.txn(read_only=True)
            try:
                response = txn.query(query)
                # Decode bytes response to JSON
                import json
                return json.loads(response.json)
            finally:
                txn.discard()
                
        except Exception as e:
            logger.error(f"Failed to query Dgraph: {e}")
            return {}
    
    async def _query_neo4j_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Query market data from Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (s:Stock {symbol: $symbol})-[:HAS_PRICE]->(p:PriceObservation)
                RETURN s.symbol as symbol, p.date as date, p.closePrice as price, p.volume as volume
                ORDER BY p.date DESC
                LIMIT 100
                """
                
                result = session.run(query, symbol=symbol)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to query Neo4j: {e}")
            return {}
    
    async def _query_jena_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Query market data from Jena"""
        try:
            # Use parameterized query to avoid injection issues
            sparql_query = """
            PREFIX stock: <http://example.org/stock/>
            PREFIX company: <http://example.org/company/>
            
            SELECT ?date ?price ?volume
            WHERE {
                ?price_obs stock:forCompany ?company .
                ?price_obs stock:closePrice ?price .
                ?price_obs stock:volume ?volume .
                ?price_obs stock:observationDate ?date .
                FILTER(STRENDS(STR(?company), ?symbol))
            }
            ORDER BY DESC(?date)
            LIMIT 100
            """
            
            # Set query with parameters
            self.jena_sparql.setQuery(sparql_query)
            self.jena_sparql.setLiteral('symbol', symbol)
            
            result = self.jena_sparql.query()
            return result.convert()
            
        except Exception as e:
            logger.error(f"Failed to query Jena: {e}")
            return {}
    
    async def query(self, query: str, format: str = "json") -> Dict[str, Any]:
        """Execute query across all graph databases"""
        results = {}
        
        try:
            # Determine query type and route to appropriate database
            if "SELECT" in query.upper() and "WHERE" in query.upper():
                # SPARQL query
                if self.health_status["jena"]:
                    results["jena"] = await self._execute_sparql_query(query)
                if self.health_status["dgraph"]:
                    results["dgraph"] = await self._execute_dgraph_query(query)
            elif "MATCH" in query.upper():
                # Cypher query
                if self.health_status["neo4j"]:
                    results["neo4j"] = await self._execute_cypher_query(query)
            else:
                # Try all databases
                if self.health_status["dgraph"]:
                    results["dgraph"] = await self._execute_dgraph_query(query)
                if self.health_status["neo4j"]:
                    results["neo4j"] = await self._execute_cypher_query(query)
                if self.health_status["jena"]:
                    results["jena"] = await self._execute_sparql_query(query)
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_sparql_query(self, query: str) -> Dict[str, Any]:
        """Execute SPARQL query on Jena"""
        try:
            self.jena_sparql.setQuery(query)
            result = self.jena_sparql.query()
            return result.convert()
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return {"error": str(e)}
    
    async def _execute_cypher_query(self, query: str) -> Dict[str, Any]:
        """Execute Cypher query on Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return {"error": str(e)}
    
    async def _execute_dgraph_query(self, query: str) -> Dict[str, Any]:
        """Execute GraphQL query on Dgraph"""
        try:
            txn = self.dgraph_client.txn(read_only=True)
            try:
                response = txn.query(query)
                return response.json
            finally:
                txn.discard()
        except Exception as e:
            logger.error(f"Dgraph query failed: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all graph databases"""
        return self.health_status.copy()
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.neo4j_driver:
                self.neo4j_driver.close()
            if self.redis_client:
                self.redis_client.close()
            logger.info("Graph manager connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}") 