# HFT Neurosymbolic AI System - Project Structure

This document describes the organized project structure and import patterns for the HFT Neurosymbolic AI System.

## 📁 Directory Structure

```
hft_neurosymbolic/
├── README.md                          # Main project documentation
├── ROADMAP.md                         # Development roadmap
├── PROJECT_STRUCTURE.md               # This file
├── CREDENTIALS_README.md              # Credentials management guide
├── credentials.env                    # Actual credentials (gitignored)
├── credentials.env.template           # Credentials template
├── .gitignore                         # Git ignore rules
├── docker-compose.yml                 # Docker orchestration
├── Dockerfile                         # Docker image definition
├── main.py                           # FastAPI main application
├── dashboard.py                      # Streamlit dashboard
│
├── hft_components/                   # Core system components
│   ├── __init__.py
│   ├── ai_engine.py                  # AI prediction engine
│   ├── data_adapters.py              # Data source adapters
│   ├── graph_manager.py              # Graph database manager
│   ├── monitoring.py                 # System monitoring
│   ├── rdf_mappers.py                # RDF mapping utilities
│   ├── rule_loader.py                # Rule management
│   ├── structured_logging.py         # Logging configuration
│   ├── symbolic_reasoner.py          # Symbolic reasoning engine
│   └── trading_engine.py             # Trading signal generation
│
├── config/                           # Configuration files
│   ├── fuseki-config.ttl             # Apache Jena configuration
│   ├── hft_trading_rules.yaml        # Trading rules
│   ├── rules_schema.yaml             # Rule schema definition
│   ├── template_rules.yaml           # Rule templates
│   ├── prometheus.yml                # Prometheus configuration
│   └── grafana/                      # Grafana dashboards
│       ├── dashboards/
│       └── datasources/
│
├── data/                             # Data storage
│   ├── __init__.py
│   ├── rdf/                          # RDF data files
│   │   ├── *.ttl                     # Turtle RDF files
│   │   └── *.xml                     # RDF/XML files
│   ├── json/                         # JSON data files
│   │   └── *.json                    # Market data JSON
│   └── xml/                          # XML data files
│       └── *.xml                     # Other XML data
│
├── utils/                            # Utility modules
│   ├── __init__.py
│   ├── requirements_rdf.txt          # RDF-specific requirements
│   ├── sitecustomize.py              # Python compatibility shim
│   ├── data_processing/              # Data processing utilities
│   │   └── yahoo_finance_to_rdf.py   # Yahoo Finance to RDF converter
│   ├── testing/                      # Testing utilities
│   │   ├── setup_verification_docker.py
│   │   └── debug_rules.py
│   └── deployment/                   # Deployment scripts
│       └── setup_and_start.sh
│
├── examples/                         # Example usage
│   ├── __init__.py
│   ├── basic/                        # Basic examples
│   │   └── example_usage.py          # Basic usage examples
│   └── advanced/                     # Advanced examples
│
├── notebooks/                        # Jupyter notebooks
│   ├── __init__.py
│   ├── analysis/                     # Data analysis notebooks
│   └── experiments/                  # Experiment notebooks
│
├── tests/                            # Test suites
│   ├── test_neurosymbolic_rules.py   # Rule evaluation tests
│   └── test_rule_loader.py           # Rule loader tests
│
├── scripts/                          # Standalone scripts
│   └── load_neo4j_test_data.py       # Neo4j data loading
│
├── docs/                             # Documentation
│   ├── README.md                     # Documentation index
│   ├── README_RDF.md                 # RDF-specific documentation
│   ├── system_architecture_flow.md   # Architecture diagrams
│   └── system_architecture.dot       # Graphviz architecture
│
├── benchmarks/                       # Performance benchmarks
├── models/                           # ML model storage
└── logs/                             # Application logs
```

## 🔗 Import Patterns

### Core Components
All core components are in the `hft_components/` package and use relative imports:

```python
# Within hft_components/
from .rule_loader import RuleLoader
from .symbolic_reasoner import SymbolicReasoner
from .ai_engine import AIEngine
```

### Main Application
The main application imports from both core components and utilities:

```python
# In main.py
from hft_components.graph_manager import GraphManager
from hft_components.ai_engine import AIEngine
from utils.data_processing.yahoo_finance_to_rdf import YahooFinanceToRDF
```

### Examples and Scripts
Examples and scripts use absolute imports with path manipulation:

```python
# In examples/ or scripts/
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hft_components.rule_loader import RuleLoader
from utils.data_processing.yahoo_finance_to_rdf import YahooFinanceToRDF
```

### Tests
Tests use path manipulation to access the parent directory:

```python
# In tests/
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_components.rule_loader import RuleLoader
from hft_components.symbolic_reasoner import SymbolicReasoner
```

## 📦 Package Structure

### `hft_components/` - Core System
- **Purpose**: Contains all core business logic and system components
- **Imports**: Uses relative imports within the package
- **Dependencies**: Minimal external dependencies, focused on core functionality

### `utils/` - Utilities
- **Purpose**: Contains utility functions, data processing tools, and helper scripts
- **Imports**: Can import from `hft_components/` when needed
- **Dependencies**: May have additional dependencies for specific utilities

### `data/` - Data Storage
- **Purpose**: Organized storage for different types of data files
- **Structure**: Separated by data format (RDF, JSON, XML)
- **Access**: Read-only access from other components

### `config/` - Configuration
- **Purpose**: All configuration files and schemas
- **Types**: YAML, TTL, and other configuration formats
- **Access**: Read by components during initialization

### `examples/` - Usage Examples
- **Purpose**: Demonstrates how to use the system
- **Structure**: Separated by complexity (basic/advanced)
- **Imports**: Uses absolute imports with path manipulation

### `tests/` - Test Suites
- **Purpose**: Unit tests and integration tests
- **Structure**: One file per major component
- **Imports**: Uses path manipulation to access parent directory

## 🚀 Usage Guidelines

### Adding New Components
1. **Core Logic**: Add to `hft_components/` with relative imports
2. **Utilities**: Add to `utils/` with appropriate subdirectory
3. **Examples**: Add to `examples/` with proper import paths
4. **Tests**: Add to `tests/` with path manipulation

### Import Best Practices
1. **Use relative imports** within packages
2. **Use absolute imports** for cross-package dependencies
3. **Add path manipulation** for scripts and examples
4. **Keep imports clean** and organized at the top of files

### File Organization
1. **Group related files** in appropriate directories
2. **Use descriptive names** for files and directories
3. **Add `__init__.py`** to make directories Python packages
4. **Document structure** in this file

## 🔧 Maintenance

### Updating Imports
When moving files, update imports in:
1. The moved file itself
2. Files that import from the moved file
3. Documentation that references the file
4. Docker files that copy the file

### Adding Dependencies
1. **Core components**: Add to main `requirements.txt`
2. **Utilities**: Add to specific utility requirements
3. **Examples**: Add to example-specific requirements
4. **Tests**: Add to test requirements

### Documentation Updates
1. Update this file when changing structure
2. Update README files with new paths
3. Update example code with correct imports
4. Update Docker files with new paths

---

This structure provides a clean, organized, and maintainable codebase for the HFT Neurosymbolic AI System.
