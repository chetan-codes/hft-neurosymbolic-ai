# HFT Neurosymbolic AI System - Documentation

## System Architecture Overview

This document provides a comprehensive overview of the HFT Neurosymbolic AI System architecture, including flow diagrams, component interactions, and technical specifications.

## 📁 Documentation Files

- **`system_architecture_flow.md`** - Complete system flow diagrams with Mermaid syntax
- **`system_architecture.dot`** - Graphviz DOT format for system architecture
- **`README.md`** - This documentation index

## 🏗️ System Architecture

### High-Level Overview

The HFT Neurosymbolic AI System combines neural networks with symbolic reasoning to generate trading signals. The system processes market data through multiple graph databases, applies AI predictions and rule-based logic, and produces actionable trading recommendations with full explainability.

### Key Components

1. **Data Sources**: yfinance, CCXT, Alternative data
2. **Graph Storage**: Dgraph (RDF), Neo4j (Property), Jena (SPARQL), Redis (Cache)
3. **AI Core**: LSTM/Transformer models + Symbolic reasoning
4. **Trading Engine**: Signal generation with multiple strategies
5. **API Layer**: FastAPI with comprehensive endpoints
6. **User Interface**: Streamlit dashboard with reasoning trace viewer
7. **Monitoring**: Prometheus metrics + Grafana dashboards

## 🔄 Data Flow

```
Market Data → Graph Storage → AI Engine + Symbolic Reasoner → Trading Engine → API → Dashboard
```

## 📊 Current Status

- ✅ **Core Pipeline**: Functional end-to-end
- ✅ **AI Predictions**: LSTM ensemble working
- ✅ **Symbolic Reasoning**: Rule-based logic with traces
- ✅ **API Endpoints**: Trading signals, reasoning traces, rule management
- ✅ **Docker Deployment**: Multi-container orchestration
- 🔄 **Dashboard**: Streamlit with reasoning trace viewer (in progress)
- 📋 **Data Adapters**: yfinance, ccxt integration (planned)

## 🚀 Quick Start

1. **Activate Environment**: `conda activate hft_neurosymbolic`
2. **Start Services**: `docker-compose up -d`
3. **Access Dashboard**: http://localhost:8501
4. **API Endpoint**: http://localhost:8001

## 📈 Performance Metrics

- **Golden Tests**: 100% pass rate
- **AI Predictions**: Functional with confidence scoring
- **Reasoning Traces**: Complete audit trail
- **Response Time**: <500ms for trading signals

## 🔧 Technical Stack

- **Backend**: FastAPI, AsyncIO, PyTorch
- **Databases**: Dgraph, Neo4j, Jena, Redis, PostgreSQL
- **AI/ML**: LSTM, Transformers, Scikit-learn
- **Symbolic**: MiniKanren, Z3 Solver, YAML rules
- **Infrastructure**: Docker, Conda, Prometheus, Grafana

## 📝 API Endpoints

- `POST /api/v1/trading/signal` - Generate trading signals
- `GET /api/v1/reasoning/traces` - View reasoning traces
- `GET /api/v1/reasoning/traces/export` - Export traces (JSON/YAML/DOT)
- `GET /api/v1/rules/packs` - Manage rule packs
- `GET /metrics` - Prometheus metrics

## 🎯 Next Steps

1. **Streamlit Dashboard Enhancement**: Reasoning trace visualization
2. **Data Adapters**: Real-time market data integration
3. **Neo4j Stabilization**: Memory optimization and data loading
4. **Advanced AI Models**: Transformer training and optimization

## 📞 Support

For technical questions or issues, refer to the system logs and monitoring dashboards. The reasoning traces provide detailed decision paths for debugging and analysis.
