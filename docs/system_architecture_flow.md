# HFT Neurosymbolic AI System - Architecture Flow Diagram

## System Overview Flow

```mermaid
graph TB
    %% External Data Sources
    subgraph "Data Sources"
        YF[yfinance API]
        CCXT[CCXT Crypto APIs]
        ALT[Alternative Data]
    end
    
    %% Data Ingestion Layer
    subgraph "Data Ingestion"
        ADAPTERS[Market Data Adapters]
        LOADERS[Data Loaders]
        RDF[RDF Data Generator]
    end
    
    %% Graph Databases
    subgraph "Graph Storage Layer"
        DGRAPH[(Dgraph<br/>RDF Triples)]
        NEO4J[(Neo4j<br/>Property Graph)]
        JENA[(Apache Jena<br/>SPARQL Endpoint)]
        REDIS[(Redis<br/>Cache & Sessions)]
    end
    
    %% Core AI Components
    subgraph "Neurosymbolic AI Core"
        GM[Graph Manager<br/>Data Orchestration]
        AI[AI Engine<br/>LSTM + Transformer]
        SR[Symbolic Reasoner<br/>Rule-based Logic]
        RL[Rule Loader<br/>YAML Rule Packs]
    end
    
    %% Trading Engine
    subgraph "Trading Engine"
        TE[Trading Engine<br/>Signal Generation]
        STRATEGIES[Strategy Configs<br/>neurosymbolic, momentum,<br/>mean_reversion, rule_only]
    end
    
    %% API Layer
    subgraph "API Layer"
        FASTAPI[FastAPI Server<br/>Port 8001]
        ENDPOINTS[Trading Endpoints<br/>/api/v1/trading/signal<br/>/api/v1/reasoning/traces<br/>/api/v1/rules/packs]
    end
    
    %% Dashboard
    subgraph "User Interface"
        STREAMLIT[Streamlit Dashboard<br/>Port 8501]
        TRACE_VIEWER[Reasoning Trace Viewer]
        SIGNAL_PANEL[Trading Signal Panel]
    end
    
    %% Monitoring
    subgraph "Monitoring & Observability"
        PROMETHEUS[Prometheus Metrics]
        GRAFANA[Grafana Dashboards]
        LOGS[Structured Logging]
    end
    
    %% Data Flow Connections
    YF --> ADAPTERS
    CCXT --> ADAPTERS
    ALT --> ADAPTERS
    
    ADAPTERS --> LOADERS
    LOADERS --> RDF
    RDF --> DGRAPH
    RDF --> NEO4J
    RDF --> JENA
    
    DGRAPH --> GM
    NEO4J --> GM
    JENA --> GM
    REDIS --> GM
    
    GM --> AI
    GM --> SR
    RL --> SR
    
    AI --> TE
    SR --> TE
    STRATEGIES --> TE
    
    TE --> FASTAPI
    FASTAPI --> ENDPOINTS
    
    ENDPOINTS --> STREAMLIT
    STREAMLIT --> TRACE_VIEWER
    STREAMLIT --> SIGNAL_PANEL
    
    FASTAPI --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    FASTAPI --> LOGS
    
    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef storage fill:#f3e5f5
    classDef aiCore fill:#e8f5e8
    classDef trading fill:#fff3e0
    classDef api fill:#fce4ec
    classDef ui fill:#f1f8e9
    classDef monitoring fill:#fff8e1
    
    class YF,CCXT,ALT dataSource
    class DGRAPH,NEO4J,JENA,REDIS storage
    class GM,AI,SR,RL aiCore
    class TE,STRATEGIES trading
    class FASTAPI,ENDPOINTS api
    class STREAMLIT,TRACE_VIEWER,SIGNAL_PANEL ui
    class PROMETHEUS,GRAFANA,LOGS monitoring
```

## Detailed Component Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant GraphManager
    participant AIEngine
    participant SymbolicReasoner
    participant TradingEngine
    participant Dgraph
    participant Neo4j
    
    User->>Streamlit: Request Trading Signal
    Streamlit->>FastAPI: POST /api/v1/trading/signal
    FastAPI->>GraphManager: get_market_data(symbol, timeframe)
    
    GraphManager->>Dgraph: Query RDF triples
    Dgraph-->>GraphManager: Market data (prices, volume)
    GraphManager->>Neo4j: Query property graph
    Neo4j-->>GraphManager: Market data (prices, volume)
    GraphManager-->>FastAPI: Combined market data
    
    FastAPI->>AIEngine: predict(market_data)
    AIEngine->>AIEngine: Process features (8 dims)
    AIEngine->>AIEngine: LSTM prediction
    AIEngine->>AIEngine: Ensemble combination
    AIEngine-->>FastAPI: AI predictions + confidence
    
    FastAPI->>SymbolicReasoner: analyze(market_data, ai_prediction)
    SymbolicReasoner->>SymbolicReasoner: Load rule pack
    SymbolicReasoner->>SymbolicReasoner: Evaluate market regime rules
    SymbolicReasoner->>SymbolicReasoner: Evaluate technical signal rules
    SymbolicReasoner->>SymbolicReasoner: Risk assessment
    SymbolicReasoner->>SymbolicReasoner: Compliance check
    SymbolicReasoner->>SymbolicReasoner: Generate reasoning trace
    SymbolicReasoner-->>FastAPI: Symbolic analysis + trace
    
    FastAPI->>TradingEngine: generate_signal(symbol, ai_pred, symbolic_analysis, strategy)
    TradingEngine->>TradingEngine: Combine AI + symbolic (weighted)
    TradingEngine->>TradingEngine: Apply strategy config
    TradingEngine->>TradingEngine: Generate execution plan
    TradingEngine-->>FastAPI: Trading signal + execution plan
    
    FastAPI-->>Streamlit: Complete response
    Streamlit-->>User: Display signal + reasoning trace
```

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A[Market Data<br/>AAPL, TSLA, etc.]
    end
    
    subgraph "Processing Pipeline"
        B[Feature Engineering<br/>8 dimensions]
        C[AI Prediction<br/>LSTM Ensemble]
        D[Rule Evaluation<br/>YAML Rules]
        E[Signal Combination<br/>Weighted Fusion]
    end
    
    subgraph "Output Layer"
        F[Trading Signal<br/>BUY/SELL/HOLD]
        G[Execution Plan<br/>Position Size, Algorithm]
        H[Reasoning Trace<br/>Decision Path]
    end
    
    A --> B
    B --> C
    A --> D
    C --> E
    D --> E
    E --> F
    E --> G
    D --> H
    
    classDef input fill:#e3f2fd
    classDef process fill:#e8f5e8
    classDef output fill:#fff3e0
    
    class A input
    class B,C,D,E process
    class F,G,H output
```

## Technology Stack

```mermaid
graph TB
    subgraph "Frontend"
        ST[Streamlit Dashboard]
    end
    
    subgraph "Backend"
        FA[FastAPI]
        ASYNC[AsyncIO]
    end
    
    subgraph "AI/ML"
        PT[PyTorch]
        LSTM[LSTM Networks]
        TRANS[Transformers]
        SK[Scikit-learn]
    end
    
    subgraph "Symbolic Reasoning"
        RULES[YAML Rule Packs]
        MINIKANREN[MiniKanren]
        UNIFY[Unification]
        Z3[Z3 Solver]
    end
    
    subgraph "Graph Databases"
        DG[Dgraph]
        N4J[Neo4j]
        JENA[Apache Jena]
        RD[Redis]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        LOG[Structured Logging]
    end
    
    subgraph "Infrastructure"
        DOCKER[Docker Compose]
        CONDA[Conda Environment]
    end
    
    ST --> FA
    FA --> ASYNC
    FA --> PT
    FA --> RULES
    FA --> DG
    FA --> PROM
    
    PT --> LSTM
    PT --> TRANS
    PT --> SK
    
    RULES --> MINIKANREN
    RULES --> UNIFY
    RULES --> Z3
    
    DG --> N4J
    DG --> JENA
    DG --> RD
    
    PROM --> GRAF
    PROM --> LOG
    
    DOCKER --> CONDA
```

## Key Features & Capabilities

### ✅ **Implemented Features**
- **Neurosymbolic AI Pipeline**: AI predictions + symbolic reasoning
- **Rule-based Trading**: YAML rule packs with validation
- **Multi-database Support**: Dgraph, Neo4j, Jena, Redis
- **Reasoning Traces**: Complete decision audit trail
- **API Endpoints**: RESTful trading signal generation
- **Docker Orchestration**: Multi-container deployment
- **Monitoring**: Prometheus metrics + Grafana dashboards

### 🔄 **In Progress**
- **Streamlit Dashboard**: Reasoning trace viewer
- **Data Adapters**: yfinance, ccxt integration
- **Neo4j Stabilization**: Memory optimization

### 📋 **Planned Features**
- **Advanced AI Models**: Transformer training
- **Real-time Data**: WebSocket streaming
- **Backtesting**: Historical performance analysis
- **Risk Management**: Position sizing algorithms
- **Compliance**: Regulatory reporting

## System Requirements

- **Python**: 3.10+
- **Memory**: 8GB+ RAM (Neo4j optimized for 256M-512M)
- **Storage**: 10GB+ for data and models
- **Network**: Ports 8001 (FastAPI), 8501 (Streamlit), 8081 (Hasura)
- **Dependencies**: Docker, Conda, PyTorch, Graph databases
