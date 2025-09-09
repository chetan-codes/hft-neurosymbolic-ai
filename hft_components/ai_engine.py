#!/usr/bin/env python3
"""
AI Engine - Neural Network Predictions for HFT
Uses PyTorch for price prediction and pattern recognition
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    """LSTM Neural Network for HFT price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerPredictor(nn.Module):
    """Transformer model for HFT prediction"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int, dropout: float = 0.1):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take the last output and project to output size
        x = x[:, -1, :]
        x = self.output_projection(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AIEngine:
    """AI Engine for HFT predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 100  # longer lookback to increase symbol differentiation
        self.prediction_horizon = 5  # Predict next 5 time steps
        
        # Model configurations (temporarily only LSTM enabled; input_size matches feature columns = 8)
        self.model_configs = {
            "lstm": {
                "input_size": 8,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": self.prediction_horizon
            }
        }
        
        # Initialize models
        self._initialize_models()
        
        # Performance metrics
        self.metrics = {
            "predictions_made": 0,
            "avg_prediction_time": 0.0,
            "model_accuracy": {}
        }
        
        logger.info(f"AI Engine initialized on device: {self.device}")
    
    def _initialize_models(self):
        """Initialize neural network models"""
        try:
            for model_type, config in self.model_configs.items():
                if model_type == "lstm":
                    model = LSTMPredictor(**config)
                else:
                    continue
                
                model.to(self.device)
                model.eval()
                self.models[model_type] = model
                
                # Initialize scaler
                self.scalers[model_type] = MinMaxScaler()
                
                logger.info(f"Initialized {model_type} model")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def is_healthy(self) -> bool:
        """Check if AI engine is healthy"""
        return len(self.models) > 0 and all(model is not None for model in self.models.values())
    
    async def predict(self, market_data: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """Generate predictions from market data"""
        start_time = time.time()
        timing_breakdown = {}
        
        try:
            predictions = {}
            
            # Process market data
            data_processing_start = time.time()
            processed_data = self._process_market_data(market_data)
            timing_breakdown["data_processing_ms"] = (time.time() - data_processing_start) * 1000
            
            if processed_data is None:
                return {"error": "Invalid market data"}
            
            # Generate predictions for each model
            model_timings = {}
            for model_type, model in self.models.items():
                try:
                    model_start = time.time()
                    prediction = await self._generate_prediction(model_type, model, processed_data, symbol)
                    model_timings[f"{model_type}_ms"] = (time.time() - model_start) * 1000
                    predictions[model_type] = prediction
                except Exception as e:
                    logger.error(f"Prediction failed for {model_type}: {e}")
                    predictions[model_type] = {"error": str(e)}
                    model_timings[f"{model_type}_ms"] = 0
            
            # Ensemble prediction
            ensemble_start = time.time()
            ensemble_prediction = self._ensemble_predictions(predictions)
            timing_breakdown["ensemble_ms"] = (time.time() - ensemble_start) * 1000
            
            # Update metrics
            prediction_time = time.time() - start_time
            self._update_metrics(prediction_time)
            
            # Add detailed timing information
            timing_breakdown.update(model_timings)
            timing_breakdown["total_ms"] = prediction_time * 1000
            
            return {
                "predictions": predictions,
                "ensemble": ensemble_prediction,
                "prediction_time_ms": prediction_time * 1000,
                "timing_breakdown": timing_breakdown,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def _process_market_data(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Process market data for model input"""
        try:
            # Extract price data from different sources
            price_data = []
            
            # Try Neo4j data first
            if "neo4j" in market_data and market_data["neo4j"]:
                for record in market_data["neo4j"]:
                    if "price" in record and record["price"]:
                        price_data.append(float(record["price"]))
            
            # Try Dgraph data
            elif "dgraph" in market_data and market_data["dgraph"]:
                data = market_data["dgraph"].get("market_data", [])
                for item in data:
                    # Look for close price records
                    if "closePrice" in str(item.get("predicate", "")) and item.get("object"):
                        try:
                            price_data.append(float(item["object"]))
                        except (ValueError, TypeError):
                            continue
            
            # Try Jena data
            elif "jena" in market_data and market_data["jena"]:
                results = market_data["jena"].get("results", {}).get("bindings", [])
                for binding in results:
                    if "price" in binding and binding["price"]["value"]:
                        price_data.append(float(binding["price"]["value"]))
            
            if len(price_data) < self.sequence_length:
                logger.warning(f"Insufficient data: {len(price_data)} < {self.sequence_length}")
                return None
            
            # Convert to numpy array and create features
            price_array = np.array(price_data[-self.sequence_length:])
            
            # Create technical indicators
            features = self._create_features(price_array)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to process market data: {e}")
            return None
    
    def _create_features(self, price_array: np.ndarray) -> np.ndarray:
        """Create technical indicators as features"""
        try:
            features = []
            
            # Price features
            features.append(price_array)  # Raw prices
            
            # Moving averages
            ma_5 = np.convolve(price_array, np.ones(5)/5, mode='valid')
            ma_10 = np.convolve(price_array, np.ones(10)/10, mode='valid')
            ma_20 = np.convolve(price_array, np.ones(20)/20, mode='valid')
            
            # Pad moving averages
            ma_5 = np.pad(ma_5, (4, 0), mode='edge')
            ma_10 = np.pad(ma_10, (9, 0), mode='edge')
            ma_20 = np.pad(ma_20, (19, 0), mode='edge')
            
            features.extend([ma_5, ma_10, ma_20])
            
            # Price changes
            price_changes = np.diff(price_array, prepend=price_array[0])
            features.append(price_changes)
            
            # Volatility (rolling standard deviation)
            volatility = np.array([np.std(price_array[max(0, i-10):i+1]) for i in range(len(price_array))])
            features.append(volatility)
            
            # RSI-like indicator
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
            avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi)
            
            # Volume (placeholder - would come from actual data)
            volume = np.ones_like(price_array) * 1000000
            features.append(volume)
            
            # Combine all features
            feature_matrix = np.column_stack(features)
            
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            return np.zeros((len(price_array), 10))
    
    async def _generate_prediction(self, model_type: str, model: nn.Module, features: np.ndarray, symbol: str = None) -> Dict[str, Any]:
        """Generate prediction using a specific model"""
        try:
            # Scale features
            scaler = self.scalers[model_type]
            scaled_features = scaler.fit_transform(features)
            
            # Convert to tensor
            x = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                prediction = model(x)
                prediction = prediction.cpu().numpy().flatten()
            
            # Calculate confidence based on model uncertainty and symbol characteristics
            confidence = self._calculate_confidence(prediction, features, symbol)
            
            return {
                "prediction": prediction.tolist(),
                "confidence": confidence,
                "model_type": model_type
            }
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {"error": str(e)}
    
    def _calculate_confidence(self, prediction: np.ndarray, features: np.ndarray, symbol: str = None) -> float:
        """Calculate prediction confidence based on multiple factors including symbol-specific characteristics"""
        try:
            # Factor 1: Prediction stability (lower std = higher confidence)
            prediction_std = np.std(prediction)
            feature_std = np.std(features[:, 0])  # Price standard deviation
            stability_factor = max(0.1, min(0.95, 1.0 - (prediction_std / (feature_std + 1e-8))))
            
            # Factor 2: Prediction magnitude (stronger predictions = higher confidence)
            prediction_magnitude = np.mean(np.abs(prediction))
            magnitude_factor = max(0.1, min(0.95, prediction_magnitude * 10))
            
            # Factor 3: Feature quality (more data points = higher confidence)
            data_quality = min(1.0, len(features) / 50.0)  # Normalize to 50 data points
            
            # Factor 4: Prediction consistency (trending predictions = higher confidence)
            if len(prediction) > 1:
                trend_consistency = 1.0 - (np.std(np.diff(prediction)) / (np.mean(np.abs(prediction)) + 1e-8))
                trend_factor = max(0.1, min(0.95, trend_consistency))
            else:
                trend_factor = 0.5
            
            # Factor 5: Symbol-specific characteristics
            symbol_factor = self._calculate_symbol_confidence_factor(symbol, features, prediction)
            
            # Factor 6: Market volatility impact (higher volatility percentile = lower confidence)
            volatility_factor = self._calculate_volatility_confidence_factor(features)
            
            # Factor 7: Prediction direction consistency
            direction_factor = self._calculate_direction_confidence_factor(prediction)

            # Factor 8: Trend slope magnitude (stronger normalized slope = higher confidence)
            prices = features[:, 0]
            try:
                x = np.arange(len(prices))
                slope = np.polyfit(x, prices, 1)[0]
                norm_slope = abs(slope) / (np.mean(prices) + 1e-8)
                slope_factor = max(0.1, min(0.95, norm_slope * 500))  # scale to reasonable range
            except Exception:
                slope_factor = 0.5
            
            # Combine factors with weights
            confidence = (
                0.22 * stability_factor +
                0.18 * magnitude_factor +
                0.14 * data_quality +
                0.14 * trend_factor +
                0.10 * symbol_factor +
                0.10 * volatility_factor +
                0.06 * direction_factor +
                0.06 * slope_factor
            )
            
            return float(max(0.1, min(0.95, confidence)))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_symbol_confidence_factor(self, symbol: str, features: np.ndarray, prediction: np.ndarray) -> float:
        """Calculate symbol-specific confidence factor"""
        try:
            if not symbol:
                return 0.5
            
            # Different symbols have different characteristics
            symbol_hash = hash(symbol) % 1000  # Create deterministic but varied hash
            
            # Base factor from symbol characteristics
            base_factor = 0.3 + (symbol_hash / 1000.0) * 0.4  # Range: 0.3 to 0.7
            
            # Adjust based on price level (higher price stocks tend to be more stable)
            if len(features) > 0:
                avg_price = np.mean(features[:, 0])
                if avg_price > 200:  # High-priced stocks
                    base_factor += 0.1
                elif avg_price < 50:  # Low-priced stocks
                    base_factor -= 0.1
            
            # Adjust based on prediction characteristics
            if len(prediction) > 0:
                pred_range = np.max(prediction) - np.min(prediction)
                if pred_range > 0.1:  # High volatility predictions
                    base_factor -= 0.1
                elif pred_range < 0.01:  # Low volatility predictions
                    base_factor += 0.1
            
            return max(0.1, min(0.95, base_factor))
            
        except Exception as e:
            logger.error(f"Symbol confidence factor calculation failed: {e}")
            return 0.5
    
    def _calculate_volatility_confidence_factor(self, features: np.ndarray) -> float:
        """Calculate confidence factor based on market volatility"""
        try:
            if len(features) < 10:
                return 0.5
            
            prices = features[:, 0]
            returns = np.diff(prices) / (prices[:-1] + 1e-8)
            # rolling std series
            win = min(20, max(5, len(returns)))
            roll = np.array([np.std(returns[max(0, i-win+1):i+1]) for i in range(len(returns))])
            current = roll[-1] if len(roll) else 0.0
            # percentile within window
            pct = 50.0
            if len(roll) > 5:
                pct = (np.sum(roll <= current) / len(roll)) * 100.0
            # map percentile to confidence inversely
            # high vol percentile (e.g., 90th) -> 0.3, median -> ~0.6, low -> 0.85
            if pct >= 85:
                return 0.3
            if pct >= 60:
                return 0.5
            if pct >= 40:
                return 0.6
            if pct >= 20:
                return 0.75
            return 0.85
                
        except Exception as e:
            logger.error(f"Volatility confidence factor calculation failed: {e}")
            return 0.5
    
    def _calculate_direction_confidence_factor(self, prediction: np.ndarray) -> float:
        """Calculate confidence based on prediction direction consistency"""
        try:
            if len(prediction) < 2:
                return 0.5
            
            # Check if predictions are trending in one direction
            positive_count = np.sum(prediction > 0)
            negative_count = np.sum(prediction < 0)
            total_count = len(prediction)
            
            # Higher confidence for consistent direction
            direction_consistency = max(positive_count, negative_count) / total_count
            
            if direction_consistency > 0.8:  # Very consistent direction
                return 0.9
            elif direction_consistency > 0.6:  # Moderately consistent
                return 0.7
            else:  # Mixed signals
                return 0.4
                
        except Exception as e:
            logger.error(f"Direction confidence factor calculation failed: {e}")
            return 0.5
    
    def _ensemble_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions from multiple models"""
        try:
            valid_predictions = []
            weights = []
            confidences = []
            
            for model_type, pred in predictions.items():
                if "error" not in pred and "prediction" in pred:
                    valid_predictions.append(pred["prediction"])
                    confidence = pred.get("confidence", 0.5)
                    weights.append(confidence)
                    confidences.append(confidence)
            
            if not valid_predictions:
                return {"error": "No valid predictions"}
            
            # Weighted ensemble
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            ensemble_pred = np.average(valid_predictions, axis=0, weights=weights)
            
            # Calculate ensemble confidence based on:
            # 1. Average individual model confidence
            # 2. Agreement between models (lower variance = higher confidence)
            # 3. Number of models (more models = higher confidence)
            avg_confidence = np.mean(confidences)
            
            # Model agreement factor
            if len(valid_predictions) > 1:
                pred_array = np.array(valid_predictions)
                agreement_factor = 1.0 - (np.std(pred_array) / (np.mean(np.abs(pred_array)) + 1e-8))
                agreement_factor = max(0.1, min(0.95, agreement_factor))
            else:
                agreement_factor = 0.7  # Single model gets moderate agreement score
            
            # Model count factor
            model_count_factor = min(1.0, len(valid_predictions) / 3.0)  # Normalize to 3 models
            
            # Combine factors
            ensemble_confidence = (
                0.5 * avg_confidence +
                0.3 * agreement_factor +
                0.2 * model_count_factor
            )
            
            return {
                "prediction": ensemble_pred.tolist(),
                "confidence": float(max(0.1, min(0.95, ensemble_confidence))),
                "model_count": len(valid_predictions),
                "individual_confidences": confidences
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {"error": str(e)}
    
    def _update_metrics(self, prediction_time: float):
        """Update performance metrics"""
        try:
            self.metrics["predictions_made"] += 1
            
            # Update average prediction time
            current_avg = self.metrics["avg_prediction_time"]
            count = self.metrics["predictions_made"]
            self.metrics["avg_prediction_time"] = (current_avg * (count - 1) + prediction_time) / count
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    async def train_model(self, training_data: Dict[str, Any], model_type: str = "lstm") -> Dict[str, Any]:
        """Train a model with new data"""
        try:
            if model_type not in self.models:
                return {"error": f"Model type {model_type} not found"}
            
            model = self.models[model_type]
            model.train()
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X is None or y is None:
                return {"error": "Invalid training data"}
            
            # Training parameters
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            epochs = 50
            batch_size = 32
            
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            # Update model accuracy
            self.metrics["model_accuracy"][model_type] = {
                "final_loss": train_losses[-1],
                "training_curves": train_losses,
                "last_updated": datetime.now().isoformat()
            }
            
            model.eval()
            
            return {
                "status": "success",
                "model_type": model_type,
                "final_loss": train_losses[-1],
                "epochs": epochs
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}
    
    def _prepare_training_data(self, training_data: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare training data for model training"""
        try:
            # Extract price sequences
            sequences = []
            targets = []
            
            # This would be implemented based on actual training data format
            # For now, return None to indicate no training data
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None, None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get AI engine metrics"""
        return self.metrics.copy()
    
    def save_model(self, model_type: str, filepath: str) -> bool:
        """Save a trained model"""
        try:
            if model_type not in self.models:
                return False
            
            model = self.models[model_type]
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scalers[model_type],
                'config': self.model_configs[model_type]
            }, filepath)
            
            logger.info(f"Model {model_type} saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_type: str, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if model_type not in self.models:
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            model = self.models[model_type]
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.scalers[model_type] = checkpoint['scaler']
            
            model.eval()
            
            logger.info(f"Model {model_type} loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 