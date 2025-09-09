# HFT Neurosymbolic AI System - Main.py File Analysis

## Overview
The `main.py` file is the core FastAPI application for the HFT (High-Frequency Trading) Neurosymbolic AI System. It serves as the main entry point and API gateway, orchestrating various AI and graph database components to provide trading signal generation and market analysis capabilities.

## Architecture & Design Patterns

### 1. **FastAPI Application Structure**
```python
app = FastAPI(
    title="HFT Neurosymbolic AI System",
    description="High-Frequency Trading with Semantic AI and Graph Databases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```
- **Modern Web Framework**: Uses FastAPI for high-performance async API development
- **Auto-Documentation**: Automatically generates OpenAPI/Swagger documentation
- **Type Safety**: Leverages Python type hints for better development experience

### 2. **Middleware Architecture**
The application implements a sophisticated middleware system for request tracking and observability:

```python
@app.middleware("http")
async def track_requests(request: Request, call_next):
    # Request tracking with correlation IDs
    # Prometheus metrics collection
    # Structured logging
```

**Key Features:**
- **Correlation ID Tracking**: Each request gets a unique UUID for tracing
- **Performance Metrics**: Tracks request latency and throughput
- **Structured Logging**: Uses `structlog` for enhanced logging capabilities
- **Prometheus Integration**: Real-time metrics collection for monitoring

### 3. **Dependency Injection System**
FastAPI's powerful dependency injection system is used throughout:

```python
# Global component instances
graph_manager: Optional[GraphManager] = None
ai_engine: Optional[AIEngine] = None
rule_loader: Optional[RuleLoader] = None
symbolic_reasoner: Optional[SymbolicReasoner] = None
trading_engine: Optional[TradingEngine] = None
monitoring_service: Optional[MonitoringService] = None
```

**Benefits:**
- **Loose Coupling**: Components can be easily swapped or mocked
- **Testability**: Dependencies can be overridden for testing
- **Resource Management**: Proper initialization and cleanup of resources

## Core Components Integration

### 1. **AI Engine Integration**
```python
# Generate AI predictions
ai_prediction = await ai_engine.predict(market_data, symbol=request.symbol)
```
- **Neural Networks**: LSTM and Transformer models for market prediction
- **Ensemble Methods**: Combines multiple AI models for robust predictions
- **Confidence Scoring**: Dynamic confidence calculation based on market conditions

### 2. **Symbolic Reasoning Engine**
```python
# Apply symbolic reasoning
symbolic_analysis = await symbolic_reasoner.analyze(market_data, ai_prediction, symbol=request.symbol)
```
- **Rule-Based Logic**: Uses MiniKanren for symbolic reasoning
- **Market Regime Detection**: Identifies market conditions (bull, bear, sideways)
- **Technical Signal Analysis**: Evaluates technical indicators and patterns
- **Reasoning Traces**: Tracks decision-making process for explainability

### 3. **Graph Database Management**
```python
# Get market data from graph databases
market_data = await graph_manager.get_market_data(request.symbol, request.timeframe)
```
- **Multi-Database Support**: Dgraph, Neo4j, Apache Jena Fuseki
- **RDF Data Processing**: Handles semantic market data
- **Query Optimization**: Efficient data retrieval for real-time trading

### 4. **Trading Engine**
```python
# Generate trading signal
trading_signal = await trading_engine.generate_signal(
    request.symbol,
    ai_prediction,
    symbolic_analysis,
    normalized_strategy
)
```
- **Strategy Management**: Multiple trading strategies (neurosymbolic, rule_only)
- **Signal Combination**: Merges AI predictions with symbolic reasoning
- **Risk Assessment**: Evaluates trade risk and compliance

## API Endpoints Analysis

### 1. **Core Trading Endpoints**

#### `/api/v1/trading/signal` (POST)
**Purpose**: Generate trading signals using neurosymbolic AI
```python
class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    strategy: str = Field(default="neurosymbolic", description="Trading strategy")
```

**Process Flow:**
1. **Data Retrieval**: Fetch market data from graph databases
2. **AI Prediction**: Generate neural network predictions
3. **Symbolic Analysis**: Apply rule-based reasoning
4. **Signal Generation**: Combine AI and symbolic insights
5. **Response**: Return trading recommendation with confidence scores

### 2. **Data Management Endpoints**

#### `/api/v1/data/ingest` (POST)
**Purpose**: Ingest and process market data
```python
class DataIngestionRequest(BaseModel):
    symbols: List[str] = Field(..., description="Stock symbols to process")
    period: str = Field(default="1y", description="Data period")
    synthetic: bool = Field(default=False, description="Use synthetic data")
    target_triples: int = Field(default=1000000, description="Target number of RDF triples")
```

**Features:**
- **Background Processing**: Uses FastAPI's `BackgroundTasks` for async data processing
- **RDF Conversion**: Converts market data to semantic RDF format
- **Multi-Source Support**: Yahoo Finance, CCXT, alternative data sources

### 3. **System Monitoring Endpoints**

#### `/health` (GET)
**Purpose**: System health check
```python
component_status = {
    "graph_manager": graph_manager.is_healthy() if graph_manager else False,
    "ai_engine": ai_engine.is_healthy() if ai_engine else False,
    "symbolic_reasoner": symbolic_reasoner.is_healthy() if symbolic_reasoner else False,
    "trading_engine": trading_engine.is_healthy() if trading_engine else False,
    "monitoring_service": monitoring_service.is_healthy() if monitoring_service else False
}
```

#### `/metrics` (GET)
**Purpose**: Prometheus metrics endpoint
- **Request Metrics**: HTTP request counts, latency, active connections
- **Trading Metrics**: Signal generation counts, AI prediction counts
- **System Metrics**: Health scores, database query counts

### 4. **Rule Management Endpoints**

#### `/api/v1/rules/packs` (GET)
**Purpose**: List available rule packs
- **Dynamic Rule Loading**: Load trading rules at runtime
- **Rule Validation**: JSON schema validation for rule packs
- **Active Rule Management**: Switch between different rule sets

#### `/api/v1/rules/packs/{pack_name}/load` (POST)
**Purpose**: Load specific rule pack
- **Hot Swapping**: Change rules without restarting the system
- **Validation**: Ensure rule pack integrity before loading

### 5. **Reasoning Trace Endpoints**

#### `/api/v1/reasoning/traces` (GET)
**Purpose**: Get reasoning traces for explainability
- **Decision Tracking**: Track how trading decisions are made
- **Export Formats**: JSON, YAML, DOT (Graphviz) formats
- **Trace Management**: View, export, and clear reasoning traces

## Advanced FastAPI Features Used

### 1. **Background Tasks**
```python
async def process_data_ingestion(symbols: List[str], period: str, synthetic: bool, target_triples: int):
    # Background processing for data ingestion
    background_tasks.add_task(process_data_ingestion, ...)
```
- **Async Processing**: Non-blocking data processing
- **Resource Management**: Proper cleanup of background tasks
- **Error Handling**: Robust error handling for background operations

### 2. **Pydantic Models**
```python
class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    strategy: str = Field(default="neurosymbolic", description="Trading strategy")
```
- **Data Validation**: Automatic request/response validation
- **Type Safety**: Compile-time type checking
- **Documentation**: Auto-generated API documentation
- **Serialization**: Automatic JSON serialization/deserialization

### 3. **CORS Middleware**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- **Cross-Origin Support**: Enable web frontend integration
- **Security**: Configurable CORS policies
- **Development**: Easy frontend-backend integration

### 4. **Request/Response Models**
```python
class SystemStatus(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, Any]
    timestamp: datetime
```
- **Structured Responses**: Consistent API response format
- **Type Hints**: Better IDE support and documentation
- **Validation**: Automatic response validation

## Prometheus Metrics Integration

### 1. **Custom Metrics**
```python
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Number of active HTTP connections')
TRADING_SIGNALS = Counter('trading_signals_total', 'Total trading signals generated', ['symbol', 'strategy'])
AI_PREDICTIONS = Counter('ai_predictions_total', 'Total AI predictions made')
```

### 2. **Metrics Collection**
- **Request Tracking**: HTTP method, endpoint, status code, latency
- **Business Metrics**: Trading signals, AI predictions, reasoning sessions
- **System Metrics**: Health scores, database queries, active connections
- **Real-time Monitoring**: Live metrics streaming endpoint

## Error Handling & Logging

### 1. **Structured Logging**
```python
from hft_components.structured_logging import logger, with_correlation_id, trading_logger, performance_logger
```
- **Correlation IDs**: Track requests across components
- **Structured Data**: JSON-formatted logs for better parsing
- **Multiple Loggers**: Specialized loggers for different concerns

### 2. **Exception Handling**
```python
try:
    # Business logic
    return result
except Exception as e:
    logger.error(f"Operation failed: {e}")
    REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/trading/signal", status="500").inc()
    raise HTTPException(status_code=500, detail=str(e))
```
- **Graceful Degradation**: Proper error responses
- **Metrics Integration**: Error tracking in Prometheus
- **Logging**: Comprehensive error logging

## Startup & Shutdown Lifecycle

### 1. **Application Startup**
```python
@app.on_event("startup")
async def startup_event():
    # Initialize all system components
    # Load rule packs
    # Start background services
    # Configure monitoring
```

### 2. **Application Shutdown**
```python
@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources
    # Stop background services
    # Close database connections
```

## Security Considerations

### 1. **Input Validation**
- **Pydantic Models**: Automatic request validation
- **Type Safety**: Compile-time type checking
- **Sanitization**: Input sanitization through validation

### 2. **Error Information**
- **Controlled Exposure**: Limited error details in responses
- **Logging**: Comprehensive error logging for debugging
- **Monitoring**: Error tracking and alerting

## Performance Optimizations

### 1. **Async/Await Pattern**
- **Non-blocking I/O**: All database and external API calls are async
- **Concurrent Processing**: Multiple requests processed simultaneously
- **Resource Efficiency**: Better resource utilization

### 2. **Caching Strategy**
- **Component Caching**: Global component instances
- **Database Caching**: Redis integration for caching
- **Response Caching**: Potential for response caching

### 3. **Background Processing**
- **Non-blocking Operations**: Heavy operations in background
- **Resource Management**: Proper cleanup of background tasks
- **Error Handling**: Robust error handling for background operations

## Integration Points

### 1. **External Services**
- **Graph Databases**: Dgraph, Neo4j, Apache Jena Fuseki
- **Data Sources**: Yahoo Finance, CCXT exchanges
- **Monitoring**: Prometheus, Grafana
- **Caching**: Redis

### 2. **Internal Components**
- **AI Engine**: Neural network predictions
- **Symbolic Reasoner**: Rule-based analysis
- **Trading Engine**: Signal generation
- **Graph Manager**: Database operations
- **Monitoring Service**: System health tracking

## Development & Maintenance

### 1. **Code Organization**
- **Modular Design**: Clear separation of concerns
- **Dependency Injection**: Loose coupling between components
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Inline documentation and docstrings

### 2. **Testing Support**
- **Dependency Overrides**: Easy mocking for testing
- **Health Checks**: Built-in health monitoring
- **Metrics**: Performance and usage metrics
- **Logging**: Comprehensive logging for debugging

## Conclusion

The `main.py` file represents a sophisticated, production-ready FastAPI application that demonstrates advanced patterns and best practices:

1. **Modern Architecture**: Uses FastAPI's advanced features effectively
2. **Observability**: Comprehensive monitoring and logging
3. **Scalability**: Async processing and background tasks
4. **Maintainability**: Clean code organization and dependency injection
5. **Performance**: Optimized for high-frequency trading requirements
6. **Reliability**: Robust error handling and health monitoring

This application serves as the central orchestrator for a complex AI-driven trading system, integrating multiple databases, AI models, and reasoning engines to provide real-time trading signals with explainable decision-making processes.
