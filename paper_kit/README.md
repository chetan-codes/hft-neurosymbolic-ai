# HFT Neurosymbolic AI System - Paper Kit

This paper kit provides all the necessary components to understand, reproduce, and extend the research presented in our neurosymbolic trading system paper.

## Overview

Our system combines neural network predictions with symbolic reasoning to create explainable, high-frequency trading decisions. The key innovation is the real-time fusion of AI confidence with rule-based market analysis, producing traceable trading signals with competitive latency.

## Paper Kit Contents

### 1. Core System Components
- **Rule Pack**: `hft_trading_rules_v1.yaml` - Production trading rules with market regimes, technical signals, and risk management
- **Fusion Formulas**: Mathematical definitions of confidence calculation and signal combination
- **Architecture Diagrams**: System flow and component interaction diagrams
- **Trace Examples**: Sample reasoning traces showing decision paths

### 2. Experimental Data
- **Calibration Results**: Confidence calibration plots and reliability metrics
- **Ablation Studies**: Performance comparisons with rule-only and ML-only baselines
- **Latency Benchmarks**: Real-time performance measurements across components
- **Confidence Analysis**: Symbol-specific and market-condition-specific confidence variations

### 3. Reproducibility Package
- **Docker Environment**: Complete containerized system for reproducible experiments
- **Test Data**: Sample market data for validation and testing
- **Evaluation Scripts**: Automated testing and benchmarking tools
- **Configuration Files**: All system parameters and hyperparameters

## Key Research Contributions

### 1. Neurosymbolic Fusion Architecture
- **Real-time Integration**: Seamless combination of neural predictions and symbolic rules
- **Confidence Composition**: Multi-factor confidence calculation with symbol-specific adjustments
- **Traceable Decisions**: Every output includes a complete reasoning trace

### 2. Strength-based Symbolic Reasoning
- **Dynamic Rule Evaluation**: Rules produce continuous confidence scores based on evidence strength
- **Market Regime Detection**: Multi-factor regime classification with volatility and trend analysis
- **Technical Signal Generation**: Priority-based signal evaluation with RSI and MA crossovers

### 3. Soft Risk Management
- **Adaptive Risk Gates**: Risk violations scale confidence rather than forcing HOLD
- **Multi-factor Risk Assessment**: Volatility, VaR, and drawdown-based risk evaluation
- **Compliance Integration**: Real-time compliance checking with trading hours and position limits

### 4. Explainable AI Integration
- **Reasoning Traces**: Complete decision paths with rule applications and confidence factors
- **Visualization Tools**: DOT graph export for decision tree visualization
- **Performance Metrics**: Comprehensive logging and monitoring of system behavior

## Mathematical Formulations

### Confidence Composition Formula

**AI Confidence:**
```
C_ai = 0.22×S + 0.18×M + 0.14×Q + 0.14×T + 0.10×Sy + 0.10×V + 0.06×D + 0.06×Sl
```
Where:
- S = Stability factor (prediction consistency)
- M = Magnitude factor (prediction strength)
- Q = Data quality (sequence length)
- T = Trend consistency
- Sy = Symbol-specific factor
- V = Volatility percentile factor
- D = Direction consistency
- Sl = Trend slope factor

**Symbolic Confidence:**
```
C_sym = 0.32×MA + 0.22×RSI + 0.15×Align + 0.10×Extreme + 0.12×Symbol
```
Where:
- MA = Moving average crossover strength
- RSI = RSI extremity factor
- Align = Signal alignment bonus
- Extreme = RSI extreme conditions bonus
- Symbol = Symbol-specific factor

**Combined Confidence:**
```
C_combined = w_ai×C_ai + w_sym×C_sym + Agreement_Bonus
```
Where:
- w_ai = AI weight (0.4 for neurosymbolic strategy)
- w_sym = Symbolic weight (0.4 for neurosymbolic strategy)
- Agreement_Bonus = +0.03 for aligned signals, -0.02 for conflicting signals

### Rule Evaluation Formula

**Regime Match Score:**
```
Score = Σ(condition_matches) / total_conditions
```
Where each condition is evaluated against market characteristics (volatility, trend strength, volume trends).

**Signal Match Score:**
```
Score = Σ(rule_matches) × priority_weight
```
Where priority weights favor RSI extreme conditions (priority=3) over MA crossovers (priority=1).

## Usage Instructions

### 1. System Setup
```bash
# Clone the repository
git clone <repository-url>
cd hft_neurosymbolic

# Start the Docker environment
docker-compose up -d

# Verify system health
curl http://localhost:8001/health
```

### 2. Running Experiments
```bash
# Generate trading signals
curl -X POST "http://localhost:8001/api/v1/trading/signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","timeframe":"daily","strategy":"neurosymbolic"}'

# Export reasoning traces
curl "http://localhost:8001/api/v1/reasoning/traces/export?format=json"

# View system metrics
curl "http://localhost:8001/metrics"
```

### 3. Reproducing Results
```bash
# Run calibration experiments
python scripts/run_calibration_experiments.py

# Generate ablation study results
python scripts/run_ablation_studies.py

# Create confidence analysis plots
python scripts/generate_confidence_analysis.py
```

## File Structure

```
paper_kit/
├── README.md                           # This file
├── rule_pack/
│   └── hft_trading_rules_v1.yaml      # Production rule pack
├── formulas/
│   ├── confidence_composition.md       # Mathematical formulas
│   └── rule_evaluation.md             # Rule evaluation logic
├── diagrams/
│   ├── system_architecture.dot        # System architecture diagram
│   └── fusion_flow.md                 # Fusion process flow
├── traces/
│   ├── sample_traces.json             # Example reasoning traces
│   └── trace_visualization.dot        # DOT visualization examples
├── experiments/
│   ├── calibration_results.json       # Calibration experiment results
│   ├── ablation_studies.json         # Ablation study results
│   └── latency_benchmarks.json       # Performance benchmarks
├── scripts/
│   ├── run_calibration_experiments.py # Calibration experiments
│   ├── run_ablation_studies.py       # Ablation studies
│   └── generate_confidence_analysis.py # Confidence analysis
└── data/
    ├── sample_market_data.json        # Sample market data
    └── test_cases.json               # Test cases for validation
```

## Citation

If you use this system in your research, please cite:

```bibtex
@article{neurosymbolic_hft_2025,
  title={Real-time Neurosymbolic Trading: Combining Neural Predictions with Symbolic Reasoning for Explainable High-Frequency Trading},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025},
  volume={[Volume]},
  number={[Number]},
  pages={[Pages]},
  publisher={[Publisher]}
}
```

## Contact

For questions about this paper kit or the neurosymbolic trading system, please contact [your-email@domain.com].

## License

This work is licensed under [License Type]. See LICENSE file for details.
