#!/usr/bin/env python3
"""
Calibration Engine - Rule-level calibration and per-symbol priors
Implements adaptive confidence calibration for improved accuracy
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import pickle
import os

logger = logging.getLogger(__name__)

class CalibrationEngine:
    """Advanced calibration engine for rule-level and per-symbol confidence adjustment"""
    
    def __init__(self, calibration_data_dir: str = "calibration_data"):
        self.calibration_data_dir = calibration_data_dir
        self.health_status = True
        
        # Create calibration data directory
        os.makedirs(calibration_data_dir, exist_ok=True)
        
        # Calibration models
        self.rule_calibrators = {}  # rule_id -> calibrator
        self.symbol_calibrators = {}  # symbol -> calibrator
        self.global_calibrator = None
        
        # Calibration data storage
        self.calibration_data = {
            "rule_data": {},  # rule_id -> historical data
            "symbol_data": {},  # symbol -> historical data
            "global_data": []  # all data for global calibration
        }
        
        # Calibration parameters
        self.calibration_params = {
            "min_samples": 50,  # Minimum samples for calibration
            "calibration_method": "isotonic",  # "isotonic" or "platt"
            "update_frequency": 100,  # Update every N samples
            "confidence_threshold": 0.1,  # Minimum confidence for calibration
            "max_calibration_age_days": 30  # Recalibrate if older than this
        }
        
        # Performance metrics
        self.metrics = {
            "calibration_updates": 0,
            "rule_calibrations": 0,
            "symbol_calibrations": 0,
            "global_calibrations": 0,
            "avg_calibration_improvement": 0.0,
            "total_samples_processed": 0
        }
        
        # Load existing calibration models
        self._load_calibration_models()
    
    def is_healthy(self) -> bool:
        """Check if calibration engine is healthy"""
        return self.health_status
    
    async def calibrate_confidence(self, 
                                 confidence: float, 
                                 rule_id: str = None, 
                                 symbol: str = None,
                                 ground_truth: float = None) -> float:
        """Calibrate confidence score using rule-level and symbol-level calibrators"""
        try:
            calibrated_confidence = confidence
            
            # Apply rule-level calibration
            if rule_id and rule_id in self.rule_calibrators:
                rule_calibrator = self.rule_calibrators[rule_id]
                if hasattr(rule_calibrator, 'predict_proba'):
                    calibrated_confidence = rule_calibrator.predict_proba([[confidence]])[0][1]
                else:
                    calibrated_confidence = rule_calibrator.predict([[confidence]])[0]
            
            # Apply symbol-level calibration
            if symbol and symbol in self.symbol_calibrators:
                symbol_calibrator = self.symbol_calibrators[symbol]
                if hasattr(symbol_calibrator, 'predict_proba'):
                    calibrated_confidence = symbol_calibrator.predict_proba([[calibrated_confidence]])[0][1]
                else:
                    calibrated_confidence = symbol_calibrator.predict([[calibrated_confidence]])[0]
            
            # Apply global calibration if available
            if self.global_calibrator:
                if hasattr(self.global_calibrator, 'predict_proba'):
                    calibrated_confidence = self.global_calibrator.predict_proba([[calibrated_confidence]])[0][1]
                else:
                    calibrated_confidence = self.global_calibrator.predict([[calibrated_confidence]])[0]
            
            # Ensure confidence is in valid range
            calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
            
            # Store calibration data for future updates
            if ground_truth is not None:
                await self._store_calibration_data(confidence, calibrated_confidence, ground_truth, rule_id, symbol)
            
            return calibrated_confidence
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return confidence  # Return original confidence if calibration fails
    
    async def update_calibration(self, 
                               rule_id: str = None, 
                               symbol: str = None,
                               force_update: bool = False) -> bool:
        """Update calibration models with recent data"""
        try:
            updated = False
            
            # Update rule-level calibration
            if rule_id and rule_id in self.calibration_data["rule_data"]:
                rule_data = self.calibration_data["rule_data"][rule_id]
                if len(rule_data) >= self.calibration_params["min_samples"] or force_update:
                    success = await self._update_rule_calibrator(rule_id, rule_data)
                    if success:
                        updated = True
                        self.metrics["rule_calibrations"] += 1
            
            # Update symbol-level calibration
            if symbol and symbol in self.calibration_data["symbol_data"]:
                symbol_data = self.calibration_data["symbol_data"][symbol]
                if len(symbol_data) >= self.calibration_params["min_samples"] or force_update:
                    success = await self._update_symbol_calibrator(symbol, symbol_data)
                    if success:
                        updated = True
                        self.metrics["symbol_calibrations"] += 1
            
            # Update global calibration
            if len(self.calibration_data["global_data"]) >= self.calibration_params["min_samples"] or force_update:
                success = await self._update_global_calibrator(self.calibration_data["global_data"])
                if success:
                    updated = True
                    self.metrics["global_calibrations"] += 1
            
            if updated:
                self.metrics["calibration_updates"] += 1
                await self._save_calibration_models()
            
            return updated
            
        except Exception as e:
            logger.error(f"Calibration update failed: {e}")
            return False
    
    async def _store_calibration_data(self, 
                                    original_confidence: float,
                                    calibrated_confidence: float,
                                    ground_truth: float,
                                    rule_id: str = None,
                                    symbol: str = None):
        """Store calibration data for future model updates"""
        try:
            data_point = {
                "timestamp": datetime.now().isoformat(),
                "original_confidence": original_confidence,
                "calibrated_confidence": calibrated_confidence,
                "ground_truth": ground_truth,
                "rule_id": rule_id,
                "symbol": symbol
            }
            
            # Store in global data
            self.calibration_data["global_data"].append(data_point)
            
            # Store in rule-specific data
            if rule_id:
                if rule_id not in self.calibration_data["rule_data"]:
                    self.calibration_data["rule_data"][rule_id] = []
                self.calibration_data["rule_data"][rule_id].append(data_point)
            
            # Store in symbol-specific data
            if symbol:
                if symbol not in self.calibration_data["symbol_data"]:
                    self.calibration_data["symbol_data"][symbol] = []
                self.calibration_data["symbol_data"][symbol].append(data_point)
            
            self.metrics["total_samples_processed"] += 1
            
            # Trigger calibration update if enough samples
            if self.metrics["total_samples_processed"] % self.calibration_params["update_frequency"] == 0:
                await self.update_calibration(rule_id, symbol)
            
        except Exception as e:
            logger.error(f"Failed to store calibration data: {e}")
    
    async def _update_rule_calibrator(self, rule_id: str, rule_data: List[Dict[str, Any]]) -> bool:
        """Update rule-specific calibration model"""
        try:
            if len(rule_data) < self.calibration_params["min_samples"]:
                return False
            
            # Prepare data
            X = np.array([[d["original_confidence"]] for d in rule_data])
            y = np.array([d["ground_truth"] for d in rule_data])
            
            # Create calibrator
            if self.calibration_params["calibration_method"] == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:  # Platt scaling
                base_estimator = LogisticRegression()
                calibrator = CalibratedClassifierCV(base_estimator, method='sigmoid', cv=3)
            
            # Fit calibrator
            calibrator.fit(X, y)
            
            # Store calibrator
            self.rule_calibrators[rule_id] = calibrator
            
            # Calculate improvement
            original_brier = brier_score_loss(y, X.flatten())
            calibrated_predictions = calibrator.predict(X)
            calibrated_brier = brier_score_loss(y, calibrated_predictions)
            improvement = original_brier - calibrated_brier
            
            self.metrics["avg_calibration_improvement"] = (
                (self.metrics["avg_calibration_improvement"] * (self.metrics["rule_calibrations"] - 1) + improvement) /
                self.metrics["rule_calibrations"]
            )
            
            logger.info(f"Updated rule calibrator for {rule_id}: Brier improvement = {improvement:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update rule calibrator for {rule_id}: {e}")
            return False
    
    async def _update_symbol_calibrator(self, symbol: str, symbol_data: List[Dict[str, Any]]) -> bool:
        """Update symbol-specific calibration model"""
        try:
            if len(symbol_data) < self.calibration_params["min_samples"]:
                return False
            
            # Prepare data
            X = np.array([[d["original_confidence"]] for d in symbol_data])
            y = np.array([d["ground_truth"] for d in symbol_data])
            
            # Create calibrator
            if self.calibration_params["calibration_method"] == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:  # Platt scaling
                base_estimator = LogisticRegression()
                calibrator = CalibratedClassifierCV(base_estimator, method='sigmoid', cv=3)
            
            # Fit calibrator
            calibrator.fit(X, y)
            
            # Store calibrator
            self.symbol_calibrators[symbol] = calibrator
            
            # Calculate improvement
            original_brier = brier_score_loss(y, X.flatten())
            calibrated_predictions = calibrator.predict(X)
            calibrated_brier = brier_score_loss(y, calibrated_predictions)
            improvement = original_brier - calibrated_brier
            
            logger.info(f"Updated symbol calibrator for {symbol}: Brier improvement = {improvement:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update symbol calibrator for {symbol}: {e}")
            return False
    
    async def _update_global_calibrator(self, global_data: List[Dict[str, Any]]) -> bool:
        """Update global calibration model"""
        try:
            if len(global_data) < self.calibration_params["min_samples"]:
                return False
            
            # Prepare data
            X = np.array([[d["original_confidence"]] for d in global_data])
            y = np.array([d["ground_truth"] for d in global_data])
            
            # Create calibrator
            if self.calibration_params["calibration_method"] == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:  # Platt scaling
                base_estimator = LogisticRegression()
                calibrator = CalibratedClassifierCV(base_estimator, method='sigmoid', cv=3)
            
            # Fit calibrator
            calibrator.fit(X, y)
            
            # Store calibrator
            self.global_calibrator = calibrator
            
            # Calculate improvement
            original_brier = brier_score_loss(y, X.flatten())
            calibrated_predictions = calibrator.predict(X)
            calibrated_brier = brier_score_loss(y, calibrated_predictions)
            improvement = original_brier - calibrated_brier
            
            logger.info(f"Updated global calibrator: Brier improvement = {improvement:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update global calibrator: {e}")
            return False
    
    def _load_calibration_models(self):
        """Load existing calibration models from disk"""
        try:
            # Load rule calibrators
            rule_calibrators_file = os.path.join(self.calibration_data_dir, "rule_calibrators.pkl")
            if os.path.exists(rule_calibrators_file):
                with open(rule_calibrators_file, 'rb') as f:
                    self.rule_calibrators = pickle.load(f)
                logger.info(f"Loaded {len(self.rule_calibrators)} rule calibrators")
            
            # Load symbol calibrators
            symbol_calibrators_file = os.path.join(self.calibration_data_dir, "symbol_calibrators.pkl")
            if os.path.exists(symbol_calibrators_file):
                with open(symbol_calibrators_file, 'rb') as f:
                    self.symbol_calibrators = pickle.load(f)
                logger.info(f"Loaded {len(self.symbol_calibrators)} symbol calibrators")
            
            # Load global calibrator
            global_calibrator_file = os.path.join(self.calibration_data_dir, "global_calibrator.pkl")
            if os.path.exists(global_calibrator_file):
                with open(global_calibrator_file, 'rb') as f:
                    self.global_calibrator = pickle.load(f)
                logger.info("Loaded global calibrator")
            
            # Load calibration data
            calibration_data_file = os.path.join(self.calibration_data_dir, "calibration_data.json")
            if os.path.exists(calibration_data_file):
                with open(calibration_data_file, 'r') as f:
                    self.calibration_data = json.load(f)
                logger.info("Loaded calibration data")
            
        except Exception as e:
            logger.error(f"Failed to load calibration models: {e}")
    
    async def _save_calibration_models(self):
        """Save calibration models to disk"""
        try:
            # Save rule calibrators
            rule_calibrators_file = os.path.join(self.calibration_data_dir, "rule_calibrators.pkl")
            with open(rule_calibrators_file, 'wb') as f:
                pickle.dump(self.rule_calibrators, f)
            
            # Save symbol calibrators
            symbol_calibrators_file = os.path.join(self.calibration_data_dir, "symbol_calibrators.pkl")
            with open(symbol_calibrators_file, 'wb') as f:
                pickle.dump(self.symbol_calibrators, f)
            
            # Save global calibrator
            if self.global_calibrator:
                global_calibrator_file = os.path.join(self.calibration_data_dir, "global_calibrator.pkl")
                with open(global_calibrator_file, 'wb') as f:
                    pickle.dump(self.global_calibrator, f)
            
            # Save calibration data
            calibration_data_file = os.path.join(self.calibration_data_dir, "calibration_data.json")
            with open(calibration_data_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            
            logger.info("Saved calibration models and data")
            
        except Exception as e:
            logger.error(f"Failed to save calibration models: {e}")
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration status and metrics"""
        return {
            "rule_calibrators": len(self.rule_calibrators),
            "symbol_calibrators": len(self.symbol_calibrators),
            "global_calibrator": self.global_calibrator is not None,
            "total_samples": self.metrics["total_samples_processed"],
            "calibration_updates": self.metrics["calibration_updates"],
            "avg_improvement": self.metrics["avg_calibration_improvement"],
            "calibration_params": self.calibration_params
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def clear_calibration_data(self, older_than_days: int = None):
        """Clear old calibration data"""
        try:
            if older_than_days is None:
                self.calibration_data = {"rule_data": {}, "symbol_data": {}, "global_data": []}
                self.rule_calibrators = {}
                self.symbol_calibrators = {}
                self.global_calibrator = None
                logger.info("Cleared all calibration data")
            else:
                cutoff_time = datetime.now() - timedelta(days=older_than_days)
                
                # Clear old data from global data
                self.calibration_data["global_data"] = [
                    d for d in self.calibration_data["global_data"]
                    if datetime.fromisoformat(d["timestamp"]) > cutoff_time
                ]
                
                # Clear old data from rule data
                for rule_id in self.calibration_data["rule_data"]:
                    self.calibration_data["rule_data"][rule_id] = [
                        d for d in self.calibration_data["rule_data"][rule_id]
                        if datetime.fromisoformat(d["timestamp"]) > cutoff_time
                    ]
                
                # Clear old data from symbol data
                for symbol in self.calibration_data["symbol_data"]:
                    self.calibration_data["symbol_data"][symbol] = [
                        d for d in self.calibration_data["symbol_data"][symbol]
                        if datetime.fromisoformat(d["timestamp"]) > cutoff_time
                    ]
                
                logger.info(f"Cleared calibration data older than {older_than_days} days")
            
        except Exception as e:
            logger.error(f"Failed to clear calibration data: {e}")
