# Rule Evaluation Logic

This document details the symbolic rule evaluation process in the neurosymbolic trading system.

## Rule Pack Structure

The system uses YAML-based rule packs with the following structure:

```yaml
metadata:
  name: "hft_trading_rules_v1"
  version: "1.0.0"
  description: "Production trading rules for HFT neurosymbolic AI system"

market_regimes:
  regime_name:
    name: "Human Readable Name"
    description: "Description of market conditions"
    conditions:
      - condition_type: "condition_value"
    confidence_threshold: 0.7
    actions:
      - action: "buy/sell/hold"
        confidence: 0.8
        risk_level: "moderate"

technical_signals:
  signal_category:
    name: "Signal Category Name"
    rules:
      rule_name:
        condition: "mathematical_condition"
        signal: "bullish/bearish/neutral"
        confidence: 0.75
        confirmation_required: true/false
```

## Market Regime Evaluation

### Process Flow
1. Extract market characteristics from data
2. Calculate volatility and trend strength
3. Evaluate against each regime's conditions
4. Select regime with highest match score
5. Calculate confidence based on match score and rule confidence

### Market Characteristics Extraction

#### Price Data Extraction
```python
def _extract_price_data(market_data):
    # Try multiple data sources in order of preference
    if "dgraph" in market_data:
        # Extract from RDF triples
        close_prices = []
        for item in market_data["dgraph"]["market_data"]:
            if "closePrice" in item["predicate"]:
                close_prices.append(float(item["object"]))
        return close_prices
    # ... other sources
```

#### Volume Data Extraction
```python
def _extract_volume_data(market_data):
    # Similar to price extraction but for volume
    volumes = []
    for item in market_data["dgraph"]["market_data"]:
        if "volume" in item["predicate"]:
            volumes.append(float(item["object"]))
    return volumes
```

### Regime Match Score Calculation

```python
def _calculate_regime_match_score(conditions, volatility, trend_strength, volume_data):
    match_score = 0.0
    total_conditions = len(conditions)
    
    for condition in conditions:
        for condition_type, condition_value in condition.items():
            if condition_type == 'price_momentum':
                if condition_value == 'positive' and trend_strength > 0.1:
                    match_score += 1.0
                elif condition_value == 'negative' and trend_strength < -0.1:
                    match_score += 1.0
                elif condition_value == 'neutral' and abs(trend_strength) <= 0.1:
                    match_score += 1.0
            
            elif condition_type == 'volatility':
                if condition_value == 'moderate_to_high' and 0.1 < volatility < 0.5:
                    match_score += 1.0
                elif condition_value == 'high' and volatility > 0.3:
                    match_score += 1.0
                elif condition_value == 'low' and volatility < 0.1:
                    match_score += 1.0
            
            elif condition_type == 'volume_trend':
                if volume_data and len(volume_data) > 10:
                    recent_volume = sum(volume_data[-10:]) / 10
                    earlier_volume = sum(volume_data[-20:-10]) / 10
                    volume_trend = recent_volume / earlier_volume
                    
                    if condition_value == 'increasing' and volume_trend > 1.1:
                        match_score += 1.0
                    elif condition_value == 'decreasing' and volume_trend < 0.9:
                        match_score += 1.0
    
    return match_score / total_conditions if total_conditions > 0 else 0.0
```

### Regime Confidence Calculation

```python
def _evaluate_regime_rules(regime_rules, volatility, trend_strength, volume_data):
    best_regime = None
    best_confidence = 0.0
    best_match_score = 0.0
    
    for regime_name, regime_data in regime_rules.items():
        match_score = _calculate_regime_match_score(
            regime_data['conditions'], 
            volatility, 
            trend_strength, 
            volume_data
        )
        
        if match_score > best_match_score:
            best_match_score = match_score
            best_regime = regime_name
            best_confidence = regime_data.get('confidence_threshold', 0.5)
    
    if best_regime:
        calculated_confidence = max(0.1, min(0.95, best_match_score * best_confidence))
        return {
            "regime": best_regime,
            "confidence": calculated_confidence,
            "match_score": best_match_score,
            "rule_confidence": best_confidence
        }
    else:
        return {"regime": "unknown", "confidence": 0.0, "match_score": 0.0}
```

## Technical Signal Evaluation

### Process Flow
1. Calculate technical indicators (MA, RSI, etc.)
2. Evaluate against each signal rule
3. Apply priority system for rule selection
4. Calculate confidence based on match score and rule confidence

### Technical Indicators Calculation

#### Moving Averages
```python
def _calculate_moving_average(data, window):
    if len(data) < window:
        return data[-1] if data else 0.0
    return float(np.mean(data[-window:]))

# Calculate short and long MAs
ma_short = _calculate_moving_average(price_data, 5)
ma_long = _calculate_moving_average(price_data, 20)
```

#### RSI Calculation
```python
def _calculate_rsi(data, period=14):
    if len(data) < period + 1:
        return 50.0
    
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)
```

### Signal Match Score Calculation

```python
def _calculate_signal_match_score(rule_data, ma_short, ma_long, rsi, price_data):
    match_score = 0.0
    
    if 'condition' in rule_data:
        condition = rule_data['condition']
        
        # Moving average crossover
        if 'ma_20 > ma_50' in condition:
            if ma_short > ma_long:  # Using 5-day as short MA
                match_score += 1.0
        
        if 'ma_20 < ma_50' in condition:
            if ma_short < ma_long:
                match_score += 1.0
        
        # RSI conditions
        if 'rsi < 30' in condition and rsi < 30:
            match_score += 1.0
        
        if 'rsi > 70' in condition and rsi > 70:
            match_score += 1.0
        
        # Volume conditions
        if 'volume > avg_volume * 2' in condition:
            if price_data and len(price_data) > 20:
                recent_volume = price_data[-1] if hasattr(price_data[-1], 'volume') else 1.0
                avg_volume = sum(price_data[-20:]) / 20 if hasattr(price_data[0], 'volume') else 1.0
                if recent_volume > avg_volume * 2:
                    match_score += 1.0
    
    return match_score
```

### Priority-Based Signal Selection

```python
def _evaluate_signal_rules(signal_rules, ma_short, ma_long, rsi, price_data):
    best_signal = None
    best_confidence = 0.0
    best_match_score = 0.0
    best_priority = 0
    
    for signal_name, signal_data in signal_rules.items():
        if 'rules' not in signal_data:
            continue
        
        for rule_name, rule_data in signal_data['rules'].items():
            match_score = _calculate_signal_match_score(rule_data, ma_short, ma_long, rsi, price_data)
            
            if match_score > 0:  # Only consider rules that match
                # Calculate priority: RSI extreme conditions get higher priority
                priority = 0
                if 'rsi < 30' in rule_data.get('condition', '') and rsi < 30:
                    priority = 3  # Highest priority for oversold
                elif 'rsi > 70' in rule_data.get('condition', '') and rsi > 70:
                    priority = 3  # Highest priority for overbought
                elif 'ma_20 > ma_50' in rule_data.get('condition', '') or 'ma_20 < ma_50' in rule_data.get('condition', ''):
                    priority = 1  # Lower priority for MA crossovers
                else:
                    priority = 2  # Medium priority for other signals
                
                # Select rule with higher priority, or higher match score if priorities are equal
                if priority > best_priority or (priority == best_priority and match_score > best_match_score):
                    best_priority = priority
                    best_match_score = match_score
                    best_signal = rule_data.get('signal', 'wait')
                    best_confidence = rule_data.get('confidence', 0.5)
    
    if best_signal:
        calculated_confidence = max(0.1, min(0.95, best_match_score * best_confidence))
        return {
            "signal": best_signal,
            "confidence": calculated_confidence,
            "match_score": best_match_score,
            "rule_confidence": best_confidence,
            "priority": best_priority
        }
    else:
        return {"signal": "wait", "confidence": 0.0, "match_score": 0.0}
```

## Rule Pack Loading and Validation

### Rule Pack Loading
```python
def load_rule_pack(self, rule_pack_name: str) -> bool:
    if not self.rule_loader:
        logger.warning("No rule loader available")
        return False
    
    rule_pack = self.rule_loader.get_rule_pack(rule_pack_name)
    if rule_pack:
        self.active_rule_pack = rule_pack_name
        self.metrics["rule_pack_usage"] += 1
        logger.info(f"Loaded rule pack: {rule_pack_name}")
        return True
    else:
        logger.error(f"Rule pack not found: {rule_pack_name}")
        return False
```

### Rule Validation
The system validates rule packs against a JSON schema to ensure:
- Required fields are present
- Data types are correct
- Value ranges are within acceptable bounds
- Rule structure is valid

## Performance Optimizations

1. **Caching**: Rule evaluation results are cached for repeated queries
2. **Early Termination**: Evaluation stops when a high-confidence match is found
3. **Parallel Processing**: Multiple rules can be evaluated concurrently
4. **Lazy Loading**: Rules are loaded only when needed

## Error Handling

The system includes comprehensive error handling for:
- Missing or invalid rule data
- Calculation errors in technical indicators
- Rule pack loading failures
- Data extraction errors

All errors are logged and the system gracefully falls back to default behavior or returns "unknown" results.
