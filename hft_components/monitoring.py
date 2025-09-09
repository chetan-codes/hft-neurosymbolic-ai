#!/usr/bin/env python3
"""
Monitoring Service - System Performance and Health Monitoring
Tracks metrics, performance, and health for HFT neurosymbolic AI system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import psutil
import threading

logger = logging.getLogger(__name__)

class MonitoringService:
    """Monitoring service for HFT neurosymbolic AI system"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.health_status = True
        self.monitoring_task = None
        self.stop_monitoring_flag = False
        
        # Initialize monitoring components
        self._initialize_metrics()
        self._initialize_alerts()
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": 80.0,      # 80% CPU usage
            "memory_usage": 85.0,   # 85% memory usage
            "disk_usage": 90.0,     # 90% disk usage
            "latency_ms": 100.0,    # 100ms latency
            "error_rate": 0.05,     # 5% error rate
            "response_time_ms": 50.0 # 50ms response time
        }
        
        logger.info("Monitoring Service initialized")
    
    def _initialize_metrics(self):
        """Initialize system metrics"""
        try:
            self.metrics = {
                "system": {
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "disk_usage": 0.0,
                    "network_io": {"bytes_sent": 0, "bytes_recv": 0},
                    "uptime": 0.0
                },
                "performance": {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "avg_response_time_ms": 0.0,
                    "requests_per_second": 0.0
                },
                "trading": {
                    "signals_generated": 0,
                    "trades_executed": 0,
                    "success_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_signal_time_ms": 0.0
                },
                "ai": {
                    "predictions_made": 0,
                    "avg_prediction_time_ms": 0.0,
                    "model_accuracy": {},
                    "ensemble_confidence": 0.0
                },
                "symbolic": {
                    "reasoning_sessions": 0,
                    "avg_reasoning_time_ms": 0.0,
                    "compliance_checks": 0,
                    "risk_assessments": 0
                },
                "databases": {
                    "dgraph_health": True,
                    "neo4j_health": True,
                    "redis_health": True,
                    "jena_health": True,
                    "total_triples": 0,
                    "query_count": 0
                },
                "latency": {
                    "data_ingestion_ms": 0.0,
                    "signal_generation_ms": 0.0,
                    "trade_execution_ms": 0.0,
                    "database_query_ms": 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            self.health_status = False
    
    def _initialize_alerts(self):
        """Initialize alert system"""
        try:
            self.alerts = []
            
            # Alert levels
            self.alert_levels = {
                "info": "INFO",
                "warning": "WARNING", 
                "error": "ERROR",
                "critical": "CRITICAL"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize alerts: {e}")
    
    def is_healthy(self) -> bool:
        """Check if monitoring service is healthy"""
        return self.health_status
    
    async def start_monitoring(self):
        """Start the monitoring service"""
        try:
            if self.monitoring_task is None:
                self.stop_monitoring_flag = False
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                logger.info("Monitoring service started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.health_status = False
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        try:
            self.stop_monitoring_flag = True
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            logger.info("Monitoring service stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while not self.stop_monitoring_flag:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Log metrics periodically
                await self._log_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5 second monitoring interval
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
            self.health_status = False
    
    async def _update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            self.metrics["system"]["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["system"]["memory_usage"] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics["system"]["disk_usage"] = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            self.metrics["system"]["network_io"]["bytes_sent"] = network.bytes_sent
            self.metrics["system"]["network_io"]["bytes_recv"] = network.bytes_recv
            
            # Uptime
            self.metrics["system"]["uptime"] = time.time() - psutil.boot_time()
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    async def _check_alerts(self):
        """Check for system alerts"""
        try:
            # Check CPU usage
            if self.metrics["system"]["cpu_usage"] > self.thresholds["cpu_usage"]:
                await self._create_alert(
                    "HIGH_CPU_USAGE",
                    "warning",
                    f"CPU usage is {self.metrics['system']['cpu_usage']:.1f}%"
                )
            
            # Check memory usage
            if self.metrics["system"]["memory_usage"] > self.thresholds["memory_usage"]:
                await self._create_alert(
                    "HIGH_MEMORY_USAGE",
                    "warning",
                    f"Memory usage is {self.metrics['system']['memory_usage']:.1f}%"
                )
            
            # Check disk usage
            if self.metrics["system"]["disk_usage"] > self.thresholds["disk_usage"]:
                await self._create_alert(
                    "HIGH_DISK_USAGE",
                    "warning",
                    f"Disk usage is {self.metrics['system']['disk_usage']:.1f}%"
                )
            
            # Check latency
            if self.metrics["latency"]["signal_generation_ms"] > self.thresholds["latency_ms"]:
                await self._create_alert(
                    "HIGH_LATENCY",
                    "error",
                    f"Signal generation latency is {self.metrics['latency']['signal_generation_ms']:.1f}ms"
                )
            
            # Check error rate
            total_requests = self.metrics["performance"]["total_requests"]
            failed_requests = self.metrics["performance"]["failed_requests"]
            if total_requests > 0:
                error_rate = failed_requests / total_requests
                if error_rate > self.thresholds["error_rate"]:
                    await self._create_alert(
                        "HIGH_ERROR_RATE",
                        "error",
                        f"Error rate is {error_rate:.2%}"
                    )
            
            # Check database health
            if not self.metrics["databases"]["dgraph_health"]:
                await self._create_alert("DGRAPH_UNHEALTHY", "critical", "Dgraph database is unhealthy")
            
            if not self.metrics["databases"]["neo4j_health"]:
                await self._create_alert("NEO4J_UNHEALTHY", "critical", "Neo4j database is unhealthy")
            
            if not self.metrics["databases"]["redis_health"]:
                await self._create_alert("REDIS_UNHEALTHY", "critical", "Redis cache is unhealthy")
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    async def _create_alert(self, alert_type: str, level: str, message: str):
        """Create a new alert"""
        try:
            alert = {
                "id": f"alert_{int(time.time())}",
                "type": alert_type,
                "level": level,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            }
            
            self.alerts.append(alert)
            
            # Log alert
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, f"ALERT: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _log_metrics(self):
        """Log metrics periodically"""
        try:
            # Log every 60 seconds
            current_time = time.time()
            if hasattr(self, '_last_log_time') and current_time - self._last_log_time < 60:
                return
            
            self._last_log_time = current_time
            
            # Log key metrics
            logger.info(f"System Metrics - CPU: {self.metrics['system']['cpu_usage']:.1f}%, "
                       f"Memory: {self.metrics['system']['memory_usage']:.1f}%, "
                       f"Signals: {self.metrics['trading']['signals_generated']}, "
                       f"Trades: {self.metrics['trading']['trades_executed']}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def update_metrics(self, metric_path: str, value: Any):
        """Update a specific metric"""
        try:
            # Navigate to the metric path
            keys = metric_path.split('.')
            current = self.metrics
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Update the final metric
            current[keys[-1]] = value
            
        except Exception as e:
            logger.error(f"Failed to update metric {metric_path}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        return self.metrics.copy()
    
    def get_alerts(self, level: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        try:
            filtered_alerts = self.alerts
            
            if level is not None:
                filtered_alerts = [alert for alert in filtered_alerts if alert["level"] == level]
            
            if acknowledged is not None:
                filtered_alerts = [alert for alert in filtered_alerts if alert["acknowledged"] == acknowledged]
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["acknowledged"] = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def clear_old_alerts(self, max_age_hours: int = 24):
        """Clear old alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
            ]
            
            logger.info(f"Cleared alerts older than {max_age_hours} hours")
            
        except Exception as e:
            logger.error(f"Failed to clear old alerts: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Calculate health scores
            cpu_health = max(0, 100 - self.metrics["system"]["cpu_usage"])
            memory_health = max(0, 100 - self.metrics["system"]["memory_usage"])
            disk_health = max(0, 100 - self.metrics["system"]["disk_usage"])
            
            # Database health
            db_health = sum([
                self.metrics["databases"]["dgraph_health"],
                self.metrics["databases"]["neo4j_health"],
                self.metrics["databases"]["redis_health"],
                self.metrics["databases"]["jena_health"]
            ]) / 4 * 100
            
            # Overall health score
            overall_health = (cpu_health + memory_health + disk_health + db_health) / 4
            
            # Determine health status
            if overall_health >= 90:
                status = "excellent"
            elif overall_health >= 75:
                status = "good"
            elif overall_health >= 50:
                status = "fair"
            else:
                status = "poor"
            
            return {
                "status": status,
                "overall_health": overall_health,
                "components": {
                    "cpu": cpu_health,
                    "memory": memory_health,
                    "disk": disk_health,
                    "databases": db_health
                },
                "active_alerts": len([alert for alert in self.alerts if not alert["acknowledged"]]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "status": "unknown",
                "overall_health": 0.0,
                "components": {},
                "active_alerts": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            total_requests = self.metrics["performance"]["total_requests"]
            successful_requests = self.metrics["performance"]["successful_requests"]
            
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "requests": {
                    "total": total_requests,
                    "successful": successful_requests,
                    "failed": self.metrics["performance"]["failed_requests"],
                    "success_rate": success_rate
                },
                "trading": {
                    "signals_generated": self.metrics["trading"]["signals_generated"],
                    "trades_executed": self.metrics["trading"]["trades_executed"],
                    "success_rate": self.metrics["trading"]["success_rate"],
                    "total_pnl": self.metrics["trading"]["total_pnl"]
                },
                "ai": {
                    "predictions_made": self.metrics["ai"]["predictions_made"],
                    "avg_prediction_time_ms": self.metrics["ai"]["avg_prediction_time_ms"],
                    "ensemble_confidence": self.metrics["ai"]["ensemble_confidence"]
                },
                "latency": {
                    "avg_response_time_ms": self.metrics["performance"]["avg_response_time_ms"],
                    "signal_generation_ms": self.metrics["latency"]["signal_generation_ms"],
                    "trade_execution_ms": self.metrics["latency"]["trade_execution_ms"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def reset_metrics(self):
        """Reset all metrics to zero"""
        try:
            self._initialize_metrics()
            logger.info("Metrics reset")
        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON file"""
        try:
            export_data = {
                "metrics": self.metrics,
                "alerts": self.alerts,
                "health": self.get_system_health(),
                "performance": self.get_performance_summary(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def import_metrics(self, filepath: str) -> bool:
        """Import metrics from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            if "metrics" in import_data:
                self.metrics = import_data["metrics"]
            
            if "alerts" in import_data:
                self.alerts = import_data["alerts"]
            
            logger.info(f"Metrics imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import metrics: {e}")
            return False 