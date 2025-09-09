#!/usr/bin/env python3
"""
RDF to Property Graph Mappers for HFT Neurosymbolic AI System
Converts between RDF triples and property graph formats
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
from abc import ABC, abstractmethod

try:
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, XSD
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

from hft_components.structured_logging import logger, with_correlation_id

logger = logging.getLogger(__name__)

class GraphMapper(ABC):
    """Abstract base class for graph mappers"""
    
    @abstractmethod
    def to_property_graph(self, rdf_data: Any) -> Dict[str, Any]:
        """Convert RDF data to property graph format"""
        pass
    
    @abstractmethod
    def to_rdf(self, property_graph: Dict[str, Any]) -> Any:
        """Convert property graph data to RDF format"""
        pass

class RDFToNeo4jMapper(GraphMapper):
    """Mapper for converting RDF to Neo4j property graph format"""
    
    def __init__(self):
        self.name = "rdf_to_neo4j"
        self.available = RDFLIB_AVAILABLE
        
    def to_property_graph(self, rdf_data: Union[Graph, List[Dict]]) -> Dict[str, Any]:
        """Convert RDF data to Neo4j property graph format"""
        if not self.available:
            return {"error": "rdflib not available"}
        
        try:
            with with_correlation_id():
                logger.info("Converting RDF to Neo4j property graph")
                
                if isinstance(rdf_data, Graph):
                    return self._convert_rdf_graph_to_neo4j(rdf_data)
                elif isinstance(rdf_data, list):
                    return self._convert_rdf_triples_to_neo4j(rdf_data)
                else:
                    return {"error": "Unsupported RDF data format"}
                    
        except Exception as e:
            logger.error("Failed to convert RDF to Neo4j", error=str(e))
            return {"error": str(e)}
    
    def _convert_rdf_graph_to_neo4j(self, rdf_graph: Graph) -> Dict[str, Any]:
        """Convert RDF Graph to Neo4j format"""
        nodes = {}
        relationships = []
        
        # Process all triples
        for subject, predicate, obj in rdf_graph:
            subj_id = self._get_node_id(subject)
            obj_id = self._get_node_id(obj)
            
            # Create subject node
            if subj_id not in nodes:
                nodes[subj_id] = self._create_neo4j_node(subject)
            
            # Create object node
            if obj_id not in nodes:
                nodes[obj_id] = self._create_neo4j_node(obj)
            
            # Create relationship
            relationship = {
                "type": self._get_predicate_name(predicate),
                "start_node": subj_id,
                "end_node": obj_id,
                "properties": {
                    "predicate_uri": str(predicate),
                    "created_at": datetime.now().isoformat()
                }
            }
            relationships.append(relationship)
        
        return {
            "nodes": list(nodes.values()),
            "relationships": relationships,
            "metadata": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "conversion_timestamp": datetime.now().isoformat()
            }
        }
    
    def _convert_rdf_triples_to_neo4j(self, triples: List[Dict]) -> Dict[str, Any]:
        """Convert RDF triples list to Neo4j format"""
        nodes = {}
        relationships = []
        
        for triple in triples:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            
            subj_id = self._get_node_id_from_string(subject)
            obj_id = self._get_node_id_from_string(obj)
            
            # Create subject node
            if subj_id not in nodes:
                nodes[subj_id] = {
                    "id": subj_id,
                    "labels": self._infer_labels(subject),
                    "properties": {
                        "uri": subject,
                        "type": "resource"
                    }
                }
            
            # Create object node
            if obj_id not in nodes:
                nodes[obj_id] = {
                    "id": obj_id,
                    "labels": self._infer_labels(obj),
                    "properties": {
                        "uri": obj,
                        "type": "literal" if self._is_literal(obj) else "resource"
                    }
                }
            
            # Create relationship
            relationship = {
                "type": self._get_predicate_name_from_string(predicate),
                "start_node": subj_id,
                "end_node": obj_id,
                "properties": {
                    "predicate_uri": predicate,
                    "created_at": datetime.now().isoformat()
                }
            }
            relationships.append(relationship)
        
        return {
            "nodes": list(nodes.values()),
            "relationships": relationships,
            "metadata": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "conversion_timestamp": datetime.now().isoformat()
            }
        }
    
    def _get_node_id(self, rdf_term) -> str:
        """Generate node ID from RDF term"""
        if isinstance(rdf_term, URIRef):
            return f"uri_{hash(str(rdf_term))}"
        elif isinstance(rdf_term, Literal):
            return f"literal_{hash(str(rdf_term))}"
        elif isinstance(rdf_term, BNode):
            return f"bnode_{str(rdf_term)}"
        else:
            return f"unknown_{hash(str(rdf_term))}"
    
    def _get_node_id_from_string(self, uri: str) -> str:
        """Generate node ID from URI string"""
        return f"uri_{hash(uri)}"
    
    def _create_neo4j_node(self, rdf_term) -> Dict[str, Any]:
        """Create Neo4j node from RDF term"""
        node_id = self._get_node_id(rdf_term)
        
        if isinstance(rdf_term, URIRef):
            return {
                "id": node_id,
                "labels": self._infer_labels_from_uri(str(rdf_term)),
                "properties": {
                    "uri": str(rdf_term),
                    "type": "resource",
                    "local_name": str(rdf_term).split("/")[-1].split("#")[-1]
                }
            }
        elif isinstance(rdf_term, Literal):
            return {
                "id": node_id,
                "labels": ["Literal"],
                "properties": {
                    "value": str(rdf_term),
                    "type": "literal",
                    "datatype": str(rdf_term.datatype) if rdf_term.datatype else "string"
                }
            }
        elif isinstance(rdf_term, BNode):
            return {
                "id": node_id,
                "labels": ["BlankNode"],
                "properties": {
                    "bnode_id": str(rdf_term),
                    "type": "blank_node"
                }
            }
        else:
            return {
                "id": node_id,
                "labels": ["Unknown"],
                "properties": {
                    "value": str(rdf_term),
                    "type": "unknown"
                }
            }
    
    def _infer_labels(self, rdf_term) -> List[str]:
        """Infer Neo4j labels from RDF term"""
        if isinstance(rdf_term, URIRef):
            return self._infer_labels_from_uri(str(rdf_term))
        elif isinstance(rdf_term, Literal):
            return ["Literal"]
        elif isinstance(rdf_term, BNode):
            return ["BlankNode"]
        else:
            return ["Unknown"]
    
    def _infer_labels_from_uri(self, uri: str) -> List[str]:
        """Infer labels from URI"""
        labels = ["Resource"]
        
        # Add domain-specific labels
        if "company" in uri.lower():
            labels.append("Company")
        elif "stock" in uri.lower():
            labels.append("Stock")
        elif "price" in uri.lower():
            labels.append("Price")
        elif "market" in uri.lower():
            labels.append("Market")
        elif "trading" in uri.lower():
            labels.append("Trading")
        
        # Add type labels based on URI structure
        if "observation" in uri.lower():
            labels.append("Observation")
        elif "data" in uri.lower():
            labels.append("Data")
        
        return labels
    
    def _get_predicate_name(self, predicate) -> str:
        """Get predicate name from RDF predicate"""
        if isinstance(predicate, URIRef):
            return self._get_predicate_name_from_string(str(predicate))
        else:
            return "RELATES_TO"
    
    def _get_predicate_name_from_string(self, predicate_uri: str) -> str:
        """Get predicate name from URI string"""
        # Extract local name
        local_name = predicate_uri.split("/")[-1].split("#")[-1]
        
        # Map common predicates
        predicate_map = {
            "closePrice": "HAS_CLOSE_PRICE",
            "openPrice": "HAS_OPEN_PRICE",
            "highPrice": "HAS_HIGH_PRICE",
            "lowPrice": "HAS_LOW_PRICE",
            "volume": "HAS_VOLUME",
            "observationDate": "OBSERVED_ON",
            "forCompany": "FOR_COMPANY",
            "symbol": "HAS_SYMBOL",
            "name": "HAS_NAME",
            "sector": "IN_SECTOR",
            "industry": "IN_INDUSTRY"
        }
        
        return predicate_map.get(local_name, local_name.upper())
    
    def _is_literal(self, obj: str) -> bool:
        """Check if object is a literal value"""
        # Simple heuristic: if it's not a URI, it's likely a literal
        return not (obj.startswith("http://") or obj.startswith("https://") or obj.startswith("urn:"))
    
    def to_rdf(self, property_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Neo4j property graph to RDF format"""
        if not self.available:
            return {"error": "rdflib not available"}
        
        try:
            with with_correlation_id():
                logger.info("Converting Neo4j property graph to RDF")
                
                rdf_graph = Graph()
                nodes = property_graph.get("nodes", [])
                relationships = property_graph.get("relationships", [])
                
                # Create namespace
                ns = Namespace("http://example.org/hft/")
                
                # Process nodes and relationships
                for relationship in relationships:
                    start_node_id = relationship["start_node"]
                    end_node_id = relationship["end_node"]
                    predicate_name = relationship["type"]
                    
                    # Find nodes
                    start_node = next((n for n in nodes if n["id"] == start_node_id), None)
                    end_node = next((n for n in nodes if n["id"] == end_node_id), None)
                    
                    if start_node and end_node:
                        # Create URIs
                        start_uri = URIRef(start_node["properties"].get("uri", f"http://example.org/node/{start_node_id}"))
                        end_uri = URIRef(end_node["properties"].get("uri", f"http://example.org/node/{end_node_id}"))
                        predicate_uri = URIRef(f"http://example.org/hft/{predicate_name.lower()}")
                        
                        # Add triple
                        rdf_graph.add((start_uri, predicate_uri, end_uri))
                
                # Convert to triples format
                triples = []
                for subject, predicate, obj in rdf_graph:
                    triples.append({
                        "subject": str(subject),
                        "predicate": str(predicate),
                        "object": str(obj)
                    })
                
                return {
                    "triples": triples,
                    "metadata": {
                        "total_triples": len(triples),
                        "conversion_timestamp": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error("Failed to convert Neo4j to RDF", error=str(e))
            return {"error": str(e)}

class RDFToDgraphMapper(GraphMapper):
    """Mapper for converting RDF to Dgraph format"""
    
    def __init__(self):
        self.name = "rdf_to_dgraph"
        self.available = True  # No external dependencies
    
    def to_property_graph(self, rdf_data: Union[Graph, List[Dict]]) -> Dict[str, Any]:
        """Convert RDF data to Dgraph format"""
        try:
            with with_correlation_id():
                logger.info("Converting RDF to Dgraph format")
                
                if isinstance(rdf_data, list):
                    return self._convert_rdf_triples_to_dgraph(rdf_data)
                else:
                    return {"error": "Only RDF triples list supported for Dgraph"}
                    
        except Exception as e:
            logger.error("Failed to convert RDF to Dgraph", error=str(e))
            return {"error": str(e)}
    
    def _convert_rdf_triples_to_dgraph(self, triples: List[Dict]) -> Dict[str, Any]:
        """Convert RDF triples to Dgraph format"""
        dgraph_data = []
        
        for triple in triples:
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            
            # Create Dgraph node
            node = {
                "uid": f"_:{hash(subject)}",
                "dgraph.type": "StockData",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "object_type": "literal" if self._is_literal(obj) else "uri"
            }
            dgraph_data.append(node)
        
        return {
            "data": dgraph_data,
            "metadata": {
                "total_nodes": len(dgraph_data),
                "conversion_timestamp": datetime.now().isoformat()
            }
        }
    
    def _is_literal(self, obj: str) -> bool:
        """Check if object is a literal value"""
        return not (obj.startswith("http://") or obj.startswith("https://") or obj.startswith("urn:"))
    
    def to_rdf(self, dgraph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Dgraph data to RDF format"""
        try:
            with with_correlation_id():
                logger.info("Converting Dgraph to RDF format")
                
                data = dgraph_data.get("data", [])
                triples = []
                
                for node in data:
                    triple = {
                        "subject": node.get("subject", ""),
                        "predicate": node.get("predicate", ""),
                        "object": node.get("object", "")
                    }
                    triples.append(triple)
                
                return {
                    "triples": triples,
                    "metadata": {
                        "total_triples": len(triples),
                        "conversion_timestamp": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error("Failed to convert Dgraph to RDF", error=str(e))
            return {"error": str(e)}

class GraphMapperManager:
    """Manager for all graph mappers"""
    
    def __init__(self):
        self.mappers = {
            "neo4j": RDFToNeo4jMapper(),
            "dgraph": RDFToDgraphMapper()
        }
        
        available_mappers = [name for name, mapper in self.mappers.items() if mapper.available]
        logger.info("Graph mappers initialized", available_mappers=available_mappers)
    
    def get_available_mappers(self) -> List[str]:
        """Get list of available mappers"""
        return [name for name, mapper in self.mappers.items() if mapper.available]
    
    def convert_to_property_graph(
        self, 
        rdf_data: Any, 
        target_format: str = "neo4j"
    ) -> Dict[str, Any]:
        """Convert RDF data to property graph format"""
        
        if target_format not in self.mappers:
            return {"error": f"Unknown target format: {target_format}"}
        
        mapper = self.mappers[target_format]
        
        if not mapper.available:
            return {"error": f"Mapper for {target_format} is not available"}
        
        try:
            with with_correlation_id():
                start_time = time.time()
                
                result = mapper.to_property_graph(rdf_data)
                
                duration = time.time() - start_time
                logger.info("Graph conversion completed", 
                           target_format=target_format, 
                           duration_ms=duration * 1000)
                
                return result
                
        except Exception as e:
            logger.error("Graph conversion failed", target_format=target_format, error=str(e))
            return {"error": str(e)}
    
    def convert_to_rdf(
        self, 
        property_graph: Dict[str, Any], 
        source_format: str = "neo4j"
    ) -> Dict[str, Any]:
        """Convert property graph data to RDF format"""
        
        if source_format not in self.mappers:
            return {"error": f"Unknown source format: {source_format}"}
        
        mapper = self.mappers[source_format]
        
        if not mapper.available:
            return {"error": f"Mapper for {source_format} is not available"}
        
        try:
            with with_correlation_id():
                start_time = time.time()
                
                result = mapper.to_rdf(property_graph)
                
                duration = time.time() - start_time
                logger.info("Graph conversion completed", 
                           source_format=source_format, 
                           duration_ms=duration * 1000)
                
                return result
                
        except Exception as e:
            logger.error("Graph conversion failed", source_format=source_format, error=str(e))
            return {"error": str(e)}
    
    def convert_between_formats(
        self, 
        data: Any, 
        source_format: str, 
        target_format: str
    ) -> Dict[str, Any]:
        """Convert between different graph formats"""
        
        if source_format == target_format:
            return {"error": "Source and target formats are the same"}
        
        try:
            with with_correlation_id():
                logger.info("Converting between graph formats", 
                           source_format=source_format, 
                           target_format=target_format)
                
                # First convert to RDF
                rdf_result = self.convert_to_rdf(data, source_format)
                if "error" in rdf_result:
                    return rdf_result
                
                # Then convert to target format
                final_result = self.convert_to_property_graph(rdf_result, target_format)
                
                return final_result
                
        except Exception as e:
            logger.error("Format conversion failed", 
                        source_format=source_format, 
                        target_format=target_format, 
                        error=str(e))
            return {"error": str(e)}
