# Confidence Composition Formulas

This document provides the mathematical formulations for confidence calculation in the neurosymbolic trading system.

## AI Engine Confidence Calculation

The AI confidence is calculated using multiple factors to ensure symbol-specific and market-condition-specific variations:

### Formula
```
C_ai = 0.22×S + 0.18×M + 0.14×Q + 0.14×T + 0.10×Sy + 0.10×V + 0.06×D + 0.06×Sl
```

### Factor Definitions

#### 1. Stability Factor (S)
```
S = max(0.1, min(0.95, 1.0 - (prediction_std / (feature_std + 1e-8))))
```
- Measures prediction consistency
- Higher values indicate more stable predictions
- Range: [0.1, 0.95]

#### 2. Magnitude Factor (M)
```
M = max(0.1, min(0.95, prediction_magnitude * 10))
```
- Measures prediction strength
- Higher values indicate stronger predictions
- Range: [0.1, 0.95]

#### 3. Data Quality Factor (Q)
```
Q = min(1.0, len(features) / 50.0)
```
- Normalizes data quantity to 50 data points
- More data points increase confidence
- Range: [0.0, 1.0]

#### 4. Trend Consistency Factor (T)
```
T = max(0.1, min(0.95, 1.0 - (std(diff(prediction)) / (mean(abs(prediction)) + 1e-8))))
```
- Measures prediction direction consistency
- Higher values indicate trending predictions
- Range: [0.1, 0.95]

#### 5. Symbol-Specific Factor (Sy)
```
Sy = 0.3 + (symbol_hash / 1000.0) * 0.4 + price_adjustment + prediction_adjustment
```
- Creates symbol-specific variations
- Includes price level and prediction characteristics
- Range: [0.1, 0.95]

#### 6. Volatility Confidence Factor (V)
```
V = volatility_percentile_factor
```
- Maps volatility percentile to confidence inversely
- High volatility percentile (90th) → 0.3
- Median volatility → 0.6
- Low volatility percentile (20th) → 0.85

#### 7. Direction Consistency Factor (D)
```
D = max(positive_count, negative_count) / total_count
```
- Measures prediction direction consistency
- Higher values for consistent directional predictions
- Range: [0.0, 1.0]

#### 8. Trend Slope Factor (Sl)
```
Sl = max(0.1, min(0.95, abs(slope) / (mean(prices) + 1e-8) * 500))
```
- Measures trend strength from price slope
- Stronger trends increase confidence
- Range: [0.1, 0.95]

## Symbolic Reasoner Confidence Calculation

The symbolic confidence is calculated based on technical indicators and rule strength:

### Formula
```
C_sym = 0.32×MA + 0.22×RSI + 0.15×Align + 0.10×Extreme + 0.12×Symbol
```

### Factor Definitions

#### 1. Moving Average Crossover Strength (MA)
```
MA = min(0.95, abs(ma_short - ma_long) / max(ma_long, 1e-8) * 10)
```
- Measures MA separation strength
- Larger separations indicate stronger signals
- Range: [0.0, 0.95]

#### 2. RSI Extremity Factor (RSI)
```
RSI = min(0.95, abs(rsi - 50) / 50)
```
- Measures RSI distance from neutral (50)
- More extreme RSI values increase confidence
- Range: [0.0, 0.95]

#### 3. Signal Alignment Bonus (Align)
```
Align = 0.3 if (ma_short > ma_long AND rsi > 50) OR (ma_short < ma_long AND rsi < 50) else 0.0
```
- Bonus for aligned MA and RSI signals
- Binary factor for signal agreement
- Value: 0.0 or 0.3

#### 4. RSI Extreme Conditions Bonus (Extreme)
```
Extreme = 0.2 if rsi < 30 OR rsi > 70 else 0.0
```
- Bonus for oversold/overbought conditions
- Binary factor for extreme RSI
- Value: 0.0 or 0.2

#### 5. Symbol-Specific Factor (Symbol)
```
Symbol = base_factor + ma_adjustment + rsi_adjustment
```
- Symbol-specific confidence adjustments
- Includes MA relationship and RSI level adjustments
- Range: [0.1, 0.95]

## Combined Confidence Calculation

The final confidence combines AI and symbolic components with agreement bonuses:

### Formula
```
C_combined = w_ai×C_ai + w_sym×C_sym + Agreement_Bonus
```

### Parameters

#### Weights (for neurosymbolic strategy)
- `w_ai = 0.4` (AI prediction weight)
- `w_sym = 0.4` (Symbolic analysis weight)
- `w_risk = 0.2` (Risk assessment weight)

#### Agreement Bonus/Penalty
```
Agreement_Bonus = +0.03 if signals align else -0.02 if signals conflict else 0.0
```

#### Signal Alignment Logic
- **Aligned**: (symbolic_action == "buy" AND ai_confidence >= 0.55) OR (symbolic_action == "sell" AND ai_confidence <= 0.45)
- **Conflicting**: symbolic_action in ("buy", "sell") AND 0.45 < ai_confidence < 0.55
- **Neutral**: Otherwise

## Rule Evaluation Formulas

### Regime Match Score
```
Score = Σ(condition_matches) / total_conditions
```

#### Condition Matching Logic
- **Price Momentum**: trend_strength > 0.1 (positive), < -0.1 (negative), abs <= 0.1 (neutral)
- **Volatility**: 0.1 < volatility < 0.5 (moderate_to_high), > 0.3 (high), < 0.1 (low)
- **Volume Trend**: recent_volume / earlier_volume > 1.1 (increasing), < 0.9 (decreasing), 0.9-1.1 (stable)

### Signal Match Score
```
Score = Σ(rule_matches) × priority_weight
```

#### Priority Weights
- **RSI Extreme Conditions**: priority = 3 (highest)
- **Other Signals**: priority = 2 (medium)
- **MA Crossovers**: priority = 1 (lowest)

#### Rule Matching Logic
- **MA Crossover**: ma_short > ma_long (golden cross), ma_short < ma_long (death cross)
- **RSI Conditions**: rsi < 30 (oversold), rsi > 70 (overbought)
- **Volume Conditions**: volume > avg_volume * 2 (volume spike)

## Confidence Bounds and Normalization

All confidence values are bounded to the range [0.1, 0.95] to:
- Prevent overconfidence (max 0.95)
- Avoid zero confidence (min 0.1)
- Maintain meaningful differentiation between signals

## Implementation Notes

1. **Symbol Hashing**: Uses `hash(symbol) % 1000` for deterministic but varied symbol-specific factors
2. **Numerical Stability**: All divisions include `1e-8` epsilon to prevent division by zero
3. **Feature Scaling**: AI features are normalized using MinMaxScaler before prediction
4. **Ensemble Confidence**: Multi-model predictions use weighted averaging with agreement factors
5. **Risk Adjustment**: Risk violations scale confidence by 0.6 and force HOLD only for 2+ violations
