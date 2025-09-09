# HFT Neurosymbolic AI System

A comprehensive High-Frequency Trading (HFT) system that combines semantic AI and graph databases for explainable, low-latency trading decisions.

## 🎯 Overview

This system integrates multiple cutting-edge technologies to create a neurosymbolic AI platform for HFT:

- **Graph Databases**: Dgraph (RDF), Neo4j (Property Graph), Apache Jena (SPARQL)
- **AI/ML**: PyTorch with LSTM and Transformer models
- **Symbolic Reasoning**: MiniKanren for logical compliance checking
- **High-Performance**: PuppyGraph for <1ms queries, targeting <100µs latency
- **Monitoring**: Prometheus + Grafana for real-time metrics
- **Containerized**: Full Docker deployment for scalability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Streamlit UI   │    │   Monitoring    │
│   (Port 8000)   │    │   (Port 8501)   │    │  (Port 9090)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              HFT Core Engine                    │
         │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
         │  │ AI Engine   │ │ Symbolic    │ │ Trading     │ │
         │  │ (PyTorch)   │ │ Reasoner    │ │ Engine      │ │
         │  │             │ │ (MiniKanren)│ │             │ │
         │  └─────────────┘ └─────────────┘ └─────────────┘ │
         └─────────────────────────────────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Graph Databases                    │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
         │  │ Dgraph  │ │ Neo4j   │ │ Jena    │ │ Redis   │ │
         │  │ (RDF)   │ │ (Prop)  │ │(SPARQL) │ │(Cache)  │ │
         │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
         └─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space**
- **macOS/Linux** (tested on macOS M2)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hft_neurosymbolic
   ```

2. **Run the setup script**:
   ```bash
   ./setup_and_start.sh setup
   ```

3. **Access the system**:
   - **HFT Application**: http://localhost:8000
   - **Streamlit Dashboard**: http://localhost:8501
   - **Neo4j Browser**: http://localhost:7474
   - **Grafana**: http://localhost:3000

### Default Credentials

- **Neo4j**: `neo4j` / `hft_password_2025`
- **Grafana**: `admin` / `hft_admin_2025`

## 📊 System Components

### 1. Data Ingestion & RDF Conversion

The system can ingest Yahoo Finance data (or synthetic data) and convert it to ~1M RDF triples:

```bash
# Generate synthetic data and convert to RDF
docker-compose exec hft_app python yahoo_finance_to_rdf.py \
    --synthetic \
    --symbols AAPL GOOGL MSFT AMZN TSLA \
    --days 365 \
    --output stock_data.ttl
```

### 2. Graph Database Integration

- **Dgraph**: Scalable RDF storage (billions of triples)
- **Neo4j**: Property graph with GDS algorithms
- **Apache Jena**: SPARQL endpoint for semantic queries
- **Redis**: High-speed caching and session management

### 3. AI Engine (PyTorch)

- **LSTM Models**: Time series prediction with attention
- **Transformer Models**: Advanced sequence modeling
- **Ensemble Methods**: Combined predictions for accuracy
- **Real-time Inference**: <1ms prediction latency

### 4. Symbolic Reasoner (MiniKanren)

- **Compliance Checking**: Trading rule validation
- **Risk Assessment**: Portfolio risk analysis
- **Market Regime Detection**: Symbolic market classification
- **Explainable Decisions**: Logical reasoning traces

### 5. Trading Engine

- **Signal Generation**: Combines AI + symbolic analysis
- **Risk Management**: Position limits, stop-loss, correlation checks
- **Execution Planning**: TWAP, VWAP, market order algorithms
- **Portfolio Management**: Real-time position tracking

## 🔧 API Endpoints

### Data Ingestion
```bash
POST /api/v1/data/ingest
{
    "symbols": ["AAPL", "GOOGL"],
    "period": "1y",
    "synthetic": false,
    "target_triples": 1000000
}
```

### Trading Signals
```bash
POST /api/v1/trading/signal
{
    "symbol": "AAPL",
    "timeframe": "1h",
    "strategy": "neurosymbolic"
}
```

### System Status
```bash
GET /api/v1/system/status
GET /api/v1/graph/query?query=SELECT...
GET /api/v1/benchmarks/hftbench
```

## 📈 Performance Targets

- **Latency**: <100µs (target), <1ms (current)
- **Throughput**: 10,000+ requests/second
- **Scalability**: 10-100B RDF triples
- **Accuracy**: >85% prediction accuracy
- **Uptime**: 99.9% availability

## 🛠️ Development

### Project Structure
```
hft_neurosymbolic/
├── main.py                 # FastAPI application
├── yahoo_finance_to_rdf.py # Data ingestion
├── hft_components/         # Core components
│   ├── graph_manager.py    # Database management
│   ├── ai_engine.py        # PyTorch models
│   ├── symbolic_reasoner.py # MiniKanren logic
│   ├── trading_engine.py   # Trading logic
│   └── monitoring.py       # System monitoring
├── config/                 # Configuration files
├── data/                   # Data storage
├── models/                 # Trained models
├── logs/                   # System logs
└── docker-compose.yml      # Container orchestration
```

### Adding New Components

1. **Create component** in `hft_components/`
2. **Add to main.py** initialization
3. **Update Dockerfile** if new dependencies
4. **Add monitoring** metrics
5. **Test integration**

### Custom Strategies

```python
# Add new trading strategy
strategy_config = {
    "description": "Custom strategy",
    "weights": {
        "ai_prediction": 0.5,
        "symbolic_analysis": 0.3,
        "risk_assessment": 0.2
    },
    "thresholds": {
        "min_confidence": 0.7,
        "max_position_size": 0.1
    }
}

trading_engine.add_strategy("custom", strategy_config)
```

## 🔍 Monitoring & Observability

### Metrics Dashboard
- **System Health**: CPU, memory, disk usage
- **Performance**: Latency, throughput, error rates
- **Trading**: Signals, trades, P&L, success rates
- **AI**: Predictions, accuracy, model performance
- **Databases**: Query performance, connection health

### Alerts
- High CPU/memory usage
- Database connectivity issues
- High latency (>100ms)
- Error rate spikes (>5%)
- Risk limit violations

## 🧪 Testing & Validation

### Unit Tests
```bash
docker-compose exec hft_app python -m pytest tests/
```

### Integration Tests
```bash
# Test data ingestion
docker-compose exec hft_app python -m pytest tests/test_data_ingestion.py

# Test trading signals
docker-compose exec hft_app python -m pytest tests/test_trading.py
```

### Performance Tests
```bash
# Load testing
docker-compose exec hft_app python benchmarks/load_test.py

# Latency testing
docker-compose exec hft_app python benchmarks/latency_test.py
```

## 🚀 Deployment

### Production Setup

1. **Environment Variables**:
   ```bash
   export HFT_ENVIRONMENT=production
   export HFT_LOG_LEVEL=INFO
   export HFT_MAX_WORKERS=4
   ```

2. **Resource Limits**:
   ```yaml
   # docker-compose.yml
   services:
     hft_app:
       deploy:
         resources:
           limits:
             memory: 8G
             cpus: '4.0'
   ```

3. **Scaling**:
   ```bash
   docker-compose up -d --scale hft_app=3
   ```

### Cloud Deployment

- **AWS**: ECS/EKS with RDS, ElastiCache
- **GCP**: GKE with Cloud SQL, Memorystore
- **Azure**: AKS with Azure Database, Redis Cache

## 🔒 Security

### Authentication
- JWT tokens for API access
- Role-based access control
- API key management

### Data Protection
- Encrypted data at rest
- TLS for data in transit
- Regular security audits

### Compliance
- Trading rule validation
- Audit trail logging
- Regulatory reporting

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **System Architecture**: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Development Guide**: [DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **Deployment Guide**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [Wiki](wiki)
- **Email**: support@hft-neurosymbolic.ai

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Basic system setup
- ✅ Data ingestion pipeline
- ✅ AI model integration
- ✅ Symbolic reasoning
- ✅ Trading engine

### Phase 2 (Next)
- 🔄 HFTBench integration
- 🔄 Advanced ML models
- 🔄 Real-time data feeds
- 🔄 Advanced risk management

### Phase 3 (Future)
- 📋 FPGA acceleration
- 📋 Quantum computing integration
- 📋 Multi-asset support
- 📋 Global deployment

---

**Built with ❤️ for the future of HFT** 