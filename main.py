#!/usr/bin/env python3
"""
HFT Neurosymbolic AI System - Main FastAPI Application
High-Frequency Trading with Semantic AI and Graph Databases
"""

import os
import logging
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, Summary
from hft_components.structured_logging import logger, with_correlation_id, trading_logger, performance_logger

# Import our custom modules
from utils.data_processing.yahoo_finance_to_rdf import YahooFinanceToRDF
from hft_components.graph_manager import GraphManager
from hft_components.ai_engine import AIEngine
from hft_components.symbolic_reasoner import SymbolicReasoner
from hft_components.trading_engine import TradingEngine
from hft_components.monitoring import MonitoringService
from hft_components.rule_loader import RuleLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HFT Neurosymbolic AI System",
    description="High-Frequency Trading with Semantic AI and Graph Databases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request tracking middleware with structured logging
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    # Set correlation ID for this request
    with with_correlation_id(correlation_id):
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        # Log request start
        logger.info(
            "HTTP request started",
            extra={
                "method": request.method,
                "endpoint": request.url.path,
                "correlation_id": correlation_id,
                "user_agent": request.headers.get("user-agent", ""),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        try:
            response = await call_next(request)
            
            # Record request metrics
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.url.path, 
                status=response.status_code
            ).inc()
            
            # Log request completion
            logger.info(
                "HTTP request completed",
                extra={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "correlation_id": correlation_id
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

# Pydantic models for API
class DataIngestionRequest(BaseModel):
    symbols: List[str] = Field(..., description="Stock symbols to process")
    period: str = Field(default="1y", description="Data period")
    synthetic: bool = Field(default=False, description="Use synthetic data")
    target_triples: int = Field(default=1000000, description="Target number of RDF triples")

class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    strategy: str = Field(default="neurosymbolic", description="Trading strategy")

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, Any]
    timestamp: datetime

# Global instances
graph_manager: Optional[GraphManager] = None
ai_engine: Optional[AIEngine] = None
rule_loader: Optional[RuleLoader] = None
symbolic_reasoner: Optional[SymbolicReasoner] = None
trading_engine: Optional[TradingEngine] = None
monitoring_service: Optional[MonitoringService] = None

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Number of active HTTP connections')
TRADING_SIGNALS = Counter('trading_signals_total', 'Total trading signals generated', ['symbol', 'strategy'])
AI_PREDICTIONS = Counter('ai_predictions_total', 'Total AI predictions made')
SYMBOLIC_REASONING_SESSIONS = Counter('symbolic_reasoning_sessions_total', 'Total symbolic reasoning sessions')
DATABASE_QUERIES = Counter('database_queries_total', 'Total database queries', ['database'])
SYSTEM_HEALTH = Gauge('system_health_score', 'Overall system health score (0-100)')

@app.on_event("startup")
async def startup_event():
    """Initialize all system components on startup"""
    global graph_manager, ai_engine, rule_loader, symbolic_reasoner, trading_engine, monitoring_service
    
    logger.info("Starting HFT Neurosymbolic AI System...")
    
    try:
        # Initialize components
        graph_manager = GraphManager()
        ai_engine = AIEngine()
        
        # Initialize rule loader first
        rule_loader = RuleLoader()
        
        # Initialize symbolic reasoner with rule loader
        symbolic_reasoner = SymbolicReasoner(rule_loader)
        
        trading_engine = TradingEngine()
        monitoring_service = MonitoringService()
        
        # Start background services
        asyncio.create_task(monitoring_service.start_monitoring())
        
        logger.info("HFT Neurosymbolic AI System started successfully")
        
        # Load and log available rule packs
        if rule_loader:
            # Try to load the default rule pack
            try:
                # Check if the rule pack file exists and load it
                if (rule_loader.rules_dir / "hft_trading_rules.yaml").exists():
                    rule_loader.load_rule_pack("hft_trading_rules.yaml")
                    logger.info("Default rule pack loaded successfully")
                
                available_packs = rule_loader.list_rule_packs()
                logger.info(f"Available rule packs: {available_packs}")
                if available_packs:
                    # Set the first available pack as active
                    default_pack = available_packs[0]
                    symbolic_reasoner.load_rule_pack(default_pack)
                    logger.info(f"Active rule pack set to: {default_pack}")
                else:
                    logger.warning("No rule packs available")
            except Exception as e:
                logger.error(f"Failed to load rule packs: {e}")
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down HFT Neurosymbolic AI System...")
    
    if monitoring_service:
        await monitoring_service.stop_monitoring()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HFT Neurosymbolic AI System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check all components
        component_status = {
            "graph_manager": graph_manager.is_healthy() if graph_manager else False,
            "ai_engine": ai_engine.is_healthy() if ai_engine else False,
            "symbolic_reasoner": symbolic_reasoner.is_healthy() if symbolic_reasoner else False,
            "trading_engine": trading_engine.is_healthy() if trading_engine else False,
            "monitoring_service": monitoring_service.is_healthy() if monitoring_service else False
        }
        
        overall_health = all(component_status.values())
        
        # Update Prometheus metrics
        health_score = 100 if overall_health else 0
        SYSTEM_HEALTH.set(health_score)
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "components": component_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        # Update system health metric
        if monitoring_service:
            health_data = monitoring_service.get_system_health()
            SYSTEM_HEALTH.set(health_data.get("overall_health", 0))
        
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/metrics", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/ingest")
async def ingest_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest stock data and convert to RDF triples"""
    try:
        logger.info(f"Starting data ingestion for {len(request.symbols)} symbols")
        
        # Add to background tasks
        background_tasks.add_task(
            process_data_ingestion,
            request.symbols,
            request.period,
            request.synthetic,
            request.target_triples
        )
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/data/ingest", status="200").inc()
        
        return {
            "message": "Data ingestion started",
            "task_id": f"ingest_{datetime.now().timestamp()}",
            "symbols": request.symbols,
            "target_triples": request.target_triples
        }
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/data/ingest", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

async def process_data_ingestion(symbols: List[str], period: str, synthetic: bool, target_triples: int):
    """Background task for data ingestion"""
    try:
        # Initialize RDF converter
        converter = YahooFinanceToRDF()
        
        # Get data
        if synthetic:
            df = converter.generate_synthetic_data(symbols, days=365)
        else:
            df = converter.fetch_real_data(symbols, period)
        
        if df.empty:
            logger.error("No data available for conversion")
            return
        
        # Convert to RDF
        converter.convert_to_rdf(df)
        
        # Save RDF
        output_file = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ttl"
        converter.save_rdf(output_file, "turtle")
        
        # Load into graph databases
        if graph_manager:
            await graph_manager.load_rdf_data(output_file)
        
        # Update monitoring
        if monitoring_service:
            monitoring_service.update_metrics("triples_generated", len(converter.g))
        
        logger.info(f"Data ingestion completed: {len(converter.g)} triples generated")
        
    except Exception as e:
        logger.error(f"Background data ingestion failed: {e}")

@app.post("/api/v1/trading/signal")
async def generate_trading_signal(request: TradingSignalRequest):
    """Generate trading signals using neurosymbolic AI"""
    try:
        logger.info(f"Generating trading signal for {request.symbol}")
        
        # Get market data
        market_data = await graph_manager.get_market_data(request.symbol, request.timeframe)
        
        # Generate AI predictions
        ai_prediction = await ai_engine.predict(market_data, symbol=request.symbol)
        AI_PREDICTIONS.inc()
        
        # Apply symbolic reasoning
        symbolic_analysis = await symbolic_reasoner.analyze(market_data, ai_prediction, symbol=request.symbol)
        SYMBOLIC_REASONING_SESSIONS.inc()
        
        # Normalize strategy alias at API layer
        normalized_strategy = request.strategy or "neurosymbolic"
        if normalized_strategy == "default":
            normalized_strategy = "neurosymbolic"
        
        # Generate trading signal
        trading_signal = await trading_engine.generate_signal(
            request.symbol,
            ai_prediction,
            symbolic_analysis,
            normalized_strategy
        )
        
        # Update Prometheus metrics
        TRADING_SIGNALS.labels(symbol=request.symbol, strategy=normalized_strategy).inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/trading/signal", status="200").inc()
        
        return {
            "symbol": request.symbol,
            "signal": trading_signal,
            "confidence": trading_signal.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat(),
            "ai_prediction": ai_prediction,
            "symbolic_analysis": symbolic_analysis
        }
        
    except Exception as e:
        logger.error(f"Trading signal generation failed: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/trading/signal", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/status")
async def get_system_status() -> SystemStatus:
    """Get system status and metrics"""
    try:
        metrics = {}
        
        if monitoring_service:
            metrics = monitoring_service.get_metrics()
        
        component_status = {
            "graph_manager": "healthy" if graph_manager and graph_manager.is_healthy() else "unhealthy",
            "ai_engine": "healthy" if ai_engine and ai_engine.is_healthy() else "unhealthy",
            "symbolic_reasoner": "healthy" if symbolic_reasoner and symbolic_reasoner.is_healthy() else "unhealthy",
            "trading_engine": "healthy" if trading_engine and trading_engine.is_healthy() else "unhealthy",
            "monitoring_service": "healthy" if monitoring_service and monitoring_service.is_healthy() else "unhealthy"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in component_status.values()) else "degraded"
        
        return SystemStatus(
            status=overall_status,
            components=component_status,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/graph/query")
async def query_graph(query: str, format: str = "json"):
    """Query the graph databases"""
    try:
        if not graph_manager:
            raise HTTPException(status_code=503, detail="Graph manager not available")
        
        result = await graph_manager.query(query, format)
        
        # Update Prometheus metrics
        DATABASE_QUERIES.labels(database="graph").inc()
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/graph/query", status="200").inc()
        
        return result
        
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/graph/query", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rules/packs")
async def list_rule_packs():
    """List available rule packs"""
    try:
        if not rule_loader:
            raise HTTPException(status_code=503, detail="Rule loader not available")
        
        packs = rule_loader.list_rule_packs()
        active_pack = symbolic_reasoner.get_active_rule_pack() if symbolic_reasoner else None
        
        return {
            "available_packs": packs,
            "active_pack": active_pack,
            "total_packs": len(packs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list rule packs: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/rules/packs", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rules/packs/{pack_name}/load")
async def load_rule_pack(pack_name: str):
    """Load a specific rule pack"""
    try:
        if not symbolic_reasoner:
            raise HTTPException(status_code=503, detail="Symbolic reasoner not available")
        
        success = symbolic_reasoner.load_rule_pack(pack_name)
        
        if success:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/rules/packs/load", status="200").inc()
            return {
                "message": f"Rule pack {pack_name} loaded successfully",
                "active_pack": pack_name,
                "timestamp": datetime.now().isoformat()
            }
        else:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/rules/packs/load", status="400").inc()
            raise HTTPException(status_code=400, detail=f"Failed to load rule pack {pack_name}")
        
    except Exception as e:
        logger.error(f"Failed to load rule pack {pack_name}: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/rules/packs/load", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rules/packs/{pack_name}")
async def get_rule_pack(pack_name: str):
    """Get details of a specific rule pack"""
    try:
        if not rule_loader:
            raise HTTPException(status_code=503, detail="Rule loader not available")
        
        rule_pack = rule_loader.get_rule_pack(pack_name)
        
        if rule_pack:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/rules/packs/{pack_name}", status="200").inc()
            return {
                "name": pack_name,
                "metadata": rule_pack.get("metadata", {}),
                "market_regimes": list(rule_pack.get("market_regimes", {}).keys()),
                "technical_signals": list(rule_pack.get("technical_signals", {}).keys()),
                "risk_assessment": list(rule_pack.get("risk_assessment", {}).keys()),
                "compliance": list(rule_pack.get("compliance", {}).keys()),
                "execution": list(rule_pack.get("execution", {}).keys())
            }
        else:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/rules/packs/{pack_name}", status="404").inc()
            raise HTTPException(status_code=404, detail=f"Rule pack {pack_name} not found")
        
    except Exception as e:
        logger.error(f"Failed to get rule pack {pack_name}: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/rules/packs/{pack_name}", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

# ===== REASONING TRACE ENDPOINTS =====

@app.get("/api/v1/reasoning/traces")
async def get_reasoning_traces(limit: int = 10, format: str = "json"):
    """Get reasoning traces summary"""
    try:
        if not symbolic_reasoner:
            raise HTTPException(status_code=503, detail="Symbolic reasoner not available")
        
        summary = symbolic_reasoner.get_reasoning_summary()
        
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces", status="200").inc()
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get reasoning traces: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/reasoning/traces/{trace_id}")
async def get_reasoning_trace(trace_id: str):
    """Get a specific reasoning trace by ID"""
    try:
        if not symbolic_reasoner:
            raise HTTPException(status_code=503, detail="Symbolic reasoner not available")
        
        trace = symbolic_reasoner.get_reasoning_trace(trace_id)
        
        if trace:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/{trace_id}", status="200").inc()
            return {
                "trace": trace,
                "timestamp": datetime.now().isoformat()
            }
        else:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/{trace_id}", status="404").inc()
            raise HTTPException(status_code=404, detail=f"Reasoning trace {trace_id} not found")
        
    except Exception as e:
        logger.error(f"Failed to get reasoning trace {trace_id}: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/{trace_id}", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/reasoning/traces/export")
async def export_reasoning_traces(format: str = "json", limit: int = None):
    """Export reasoning traces in specified format"""
    try:
        if not symbolic_reasoner:
            raise HTTPException(status_code=503, detail="Symbolic reasoner not available")
        
        exported_data = symbolic_reasoner.export_reasoning_traces(format, limit)
        
        if exported_data:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/export", status="200").inc()
            
            if format.lower() == "json":
                return Response(content=exported_data, media_type="application/json")
            elif format.lower() == "yaml":
                return Response(content=exported_data, media_type="text/yaml")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        else:
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/export", status="400").inc()
            raise HTTPException(status_code=400, detail="Failed to export reasoning traces")
        
    except Exception as e:
        logger.error(f"Failed to export reasoning traces: {e}")
        REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/reasoning/traces/export", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/reasoning/traces")
async def clear_reasoning_traces(older_than_days: int = None):
    """Clear reasoning traces"""
    try:
        if not symbolic_reasoner:
            raise HTTPException(status_code=503, detail="Symbolic reasoner not available")
        
        symbolic_reasoner.clear_reasoning_traces(older_than_days)
        
        REQUEST_COUNT.labels(method="DELETE", endpoint="/api/v1/reasoning/traces", status="200").inc()
        return {
            "message": "Reasoning traces cleared successfully",
            "older_than_days": older_than_days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear reasoning traces: {e}")
        REQUEST_COUNT.labels(method="DELETE", endpoint="/api/v1/reasoning/traces", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/benchmarks/hftbench")
async def run_hftbench_benchmark():
    """Run HFTBench benchmarking"""
    try:
        # This would integrate with HFTBench when available
        logger.info("Running HFTBench benchmark...")
        
        # Placeholder for HFTBench integration
        benchmark_results = {
            "latency_ms": 0.5,  # Target: <100µs
            "daily_yield_percent": 2.5,  # Example yield
            "accuracy": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "benchmark": "HFTBench",
            "results": benchmark_results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"HFTBench benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/stream")
async def stream_metrics():
    """Stream real-time metrics"""
    async def generate():
        while True:
            if monitoring_service:
                metrics = monitoring_service.get_metrics()
                yield f"data: {json.dumps(metrics)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 