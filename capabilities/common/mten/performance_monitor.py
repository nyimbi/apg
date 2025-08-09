#!/usr/bin/env python3
"""
Performance Monitoring and Alerting System for MTen Multi-Tenant Management

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Real-time performance monitoring, alerting, and automated optimization system
for production environments with comprehensive metrics collection and analysis.
"""

import asyncio
import json
import time
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import aiohttp
import psutil
import threading
from collections import defaultdict, deque
import statistics


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_HIT_RATE = "cache_hit_rate"
    ACTIVE_USERS = "active_users"
    TENANT_COUNT = "tenant_count"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class AlertRule:
    """Performance alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    window_minutes: int = 5
    min_occurrences: int = 1
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Collects system and application performance metrics"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)  # Store last 10k metrics
        self.collection_interval = 10  # seconds
        self.running = False
        self._collection_task = None
    
    async def start(self):
        """Start metrics collection"""
        self.running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("📊 Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("📊 Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now(UTC)
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.CPU_USAGE,
            value=cpu_percent,
            timestamp=timestamp,
            tags={"source": "system"}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024
        
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.MEMORY_USAGE,
            value=memory_percent,
            timestamp=timestamp,
            tags={"source": "system", "unit": "percent"}
        ))
        
        self.metrics_buffer.append(PerformanceMetric(
            name="memory_usage_mb",
            value=memory_mb,
            timestamp=timestamp,
            tags={"source": "system", "unit": "megabytes"}
        ))
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.metrics_buffer.append(PerformanceMetric(
                name="disk_read_bytes",
                value=disk_io.read_bytes,
                timestamp=timestamp,
                tags={"source": "system"}
            ))
            
            self.metrics_buffer.append(PerformanceMetric(
                name="disk_write_bytes", 
                value=disk_io.write_bytes,
                timestamp=timestamp,
                tags={"source": "system"}
            ))
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            self.metrics_buffer.append(PerformanceMetric(
                name="network_bytes_sent",
                value=network_io.bytes_sent,
                timestamp=timestamp,
                tags={"source": "system"}
            ))
            
            self.metrics_buffer.append(PerformanceMetric(
                name="network_bytes_recv",
                value=network_io.bytes_recv,
                timestamp=timestamp,
                tags={"source": "system"}
            ))
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        timestamp = datetime.now(UTC)
        
        # Simulate application metrics collection
        # In real implementation, these would come from the actual application
        
        # Response time simulation (would be from actual API calls)
        response_time_ms = 45.0 + (time.time() % 10) * 5  # Simulate varying response times
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.RESPONSE_TIME,
            value=response_time_ms,
            timestamp=timestamp,
            tags={"endpoint": "/api/v1/tenants", "method": "GET"}
        ))
        
        # Throughput simulation
        throughput_rps = 850 + (time.time() % 20) * 10  # Simulate varying throughput
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.THROUGHPUT,
            value=throughput_rps,
            timestamp=timestamp,
            tags={"service": "mten-api"}
        ))
        
        # Error rate simulation
        error_rate = 0.001 + (time.time() % 100) * 0.00001  # Very low error rate
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.ERROR_RATE,
            value=error_rate,
            timestamp=timestamp,
            tags={"service": "mten-api"}
        ))
        
        # Active users simulation
        active_users = 150 + int((time.time() % 60) * 2)  # Simulate user activity
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.ACTIVE_USERS,
            value=active_users,
            timestamp=timestamp,
            tags={"service": "mten-api"}
        ))
        
        # Database connections simulation
        db_connections = 45 + int((time.time() % 30))
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.DATABASE_CONNECTIONS,
            value=db_connections,
            timestamp=timestamp,
            tags={"database": "postgresql", "pool": "main"}
        ))
        
        # Cache hit rate simulation
        cache_hit_rate = 0.92 + (time.time() % 10) * 0.005  # High cache hit rate
        self.metrics_buffer.append(PerformanceMetric(
            name=MetricType.CACHE_HIT_RATE,
            value=cache_hit_rate,
            timestamp=timestamp,
            tags={"cache": "redis", "layer": "application"}
        ))
    
    def get_metrics(self, 
                   metric_name: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   tags: Optional[Dict[str, str]] = None,
                   limit: int = 1000) -> List[PerformanceMetric]:
        """Get filtered metrics from buffer"""
        
        filtered_metrics = list(self.metrics_buffer)
        
        # Apply filters
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]
        
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        if tags:
            filtered_metrics = [
                m for m in filtered_metrics 
                if all(m.tags.get(k) == v for k, v in tags.items())
            ]
        
        # Sort by timestamp (newest first) and limit
        filtered_metrics.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_metrics[:limit]


class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.notification_handlers: Dict[str, Callable] = {}
        self.running = False
        self._alert_task = None
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"🚨 Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"🗑️ Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, channel: str, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers[channel] = handler
        logger.info(f"📢 Added notification handler: {channel}")
    
    async def start(self):
        """Start alert monitoring"""
        self.running = True
        self._alert_task = asyncio.create_task(self._alert_loop())
        logger.info("🚨 Alert monitoring started")
    
    async def stop(self):
        """Stop alert monitoring"""
        self.running = False
        if self._alert_task:
            self._alert_task.cancel()
        logger.info("🚨 Alert monitoring stopped")
    
    async def _alert_loop(self):
        """Main alert evaluation loop"""
        while self.running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(30)  # Check alerts every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        # Get recent metrics for the rule
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(minutes=rule.window_minutes)
        
        metrics = self.metrics_collector.get_metrics(
            metric_name=rule.metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        if len(metrics) < rule.min_occurrences:
            return  # Not enough data points
        
        # Calculate aggregated value (use latest value for now)
        if not metrics:
            return
        
        latest_metric = metrics[0]  # Metrics are sorted by timestamp desc
        metric_value = latest_metric.value
        
        # Check condition
        condition_met = False
        if rule.condition == "gt":
            condition_met = metric_value > rule.threshold
        elif rule.condition == "lt":
            condition_met = metric_value < rule.threshold
        elif rule.condition == "eq":
            condition_met = abs(metric_value - rule.threshold) < 0.001
        elif rule.condition == "ne":
            condition_met = abs(metric_value - rule.threshold) >= 0.001
        
        # Handle alert state
        if condition_met:
            if rule.name not in self.active_alerts:
                # New alert
                alert = PerformanceAlert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {metric_value:.2f})",
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    timestamp=datetime.now(UTC)
                )
                
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                await self._send_alert_notification(alert, rule)
                logger.warning(f"🚨 ALERT TRIGGERED: {alert.message}")
        
        else:
            # Check if alert should be resolved
            if rule.name in self.active_alerts:
                alert = self.active_alerts[rule.name]
                alert.resolved = True
                alert.resolved_at = datetime.now(UTC)
                
                del self.active_alerts[rule.name]
                
                await self._send_resolution_notification(alert, rule)
                logger.info(f"✅ ALERT RESOLVED: {rule.name}")
    
    async def _send_alert_notification(self, alert: PerformanceAlert, rule: AlertRule):
        """Send alert notification through configured channels"""
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](alert, "triggered")
                except Exception as e:
                    logger.error(f"Notification error ({channel}): {e}")
    
    async def _send_resolution_notification(self, alert: PerformanceAlert, rule: AlertRule):
        """Send alert resolution notification"""
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](alert, "resolved")
                except Exception as e:
                    logger.error(f"Resolution notification error ({channel}): {e}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[PerformanceAlert]:
        """Get alert history"""
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:limit]


class PerformanceAnalyzer:
    """Analyzes performance metrics and provides insights"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def analyze_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=hours)
        
        # Get metrics for analysis
        response_time_metrics = self.metrics_collector.get_metrics(
            metric_name=MetricType.RESPONSE_TIME,
            start_time=start_time,
            end_time=end_time
        )
        
        throughput_metrics = self.metrics_collector.get_metrics(
            metric_name=MetricType.THROUGHPUT,
            start_time=start_time,
            end_time=end_time
        )
        
        error_rate_metrics = self.metrics_collector.get_metrics(
            metric_name=MetricType.ERROR_RATE,
            start_time=start_time,
            end_time=end_time
        )
        
        cpu_metrics = self.metrics_collector.get_metrics(
            metric_name=MetricType.CPU_USAGE,
            start_time=start_time,
            end_time=end_time
        )
        
        memory_metrics = self.metrics_collector.get_metrics(
            metric_name=MetricType.MEMORY_USAGE,
            start_time=start_time,
            end_time=end_time
        )
        
        # Analyze trends
        analysis = {
            "period": f"{hours} hours",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics_analysis": {}
        }
        
        # Analyze response time
        if response_time_metrics:
            values = [m.value for m in response_time_metrics]
            analysis["metrics_analysis"]["response_time"] = {
                "current": values[0] if values else 0,
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "trend": self._calculate_trend(values)
            }
        
        # Analyze throughput
        if throughput_metrics:
            values = [m.value for m in throughput_metrics]
            analysis["metrics_analysis"]["throughput"] = {
                "current": values[0] if values else 0,
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "trend": self._calculate_trend(values)
            }
        
        # Analyze error rate
        if error_rate_metrics:
            values = [m.value for m in error_rate_metrics]
            analysis["metrics_analysis"]["error_rate"] = {
                "current": values[0] if values else 0,
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "trend": self._calculate_trend(values)
            }
        
        # Analyze system metrics
        if cpu_metrics:
            values = [m.value for m in cpu_metrics]
            analysis["metrics_analysis"]["cpu_usage"] = {
                "current": values[0] if values else 0,
                "average": statistics.mean(values),
                "max": max(values),
                "trend": self._calculate_trend(values)
            }
        
        if memory_metrics:
            values = [m.value for m in memory_metrics]
            analysis["metrics_analysis"]["memory_usage"] = {
                "current": values[0] if values else 0,
                "average": statistics.mean(values),
                "max": max(values),
                "trend": self._calculate_trend(values)
            }
        
        # Generate insights
        analysis["insights"] = await self._generate_insights(analysis["metrics_analysis"])
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation using first and last values
        first_third = values[-len(values)//3:] if len(values) >= 3 else [values[-1]]
        last_third = values[:len(values)//3] if len(values) >= 3 else [values[0]]
        
        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)
        
        change_percent = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if abs(change_percent) < 5:
            return "stable"
        elif change_percent > 0:
            return "increasing"
        else:
            return "decreasing"
    
    async def _generate_insights(self, metrics_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance insights from metrics analysis"""
        insights = []
        
        # Response time insights
        if "response_time" in metrics_analysis:
            rt_data = metrics_analysis["response_time"]
            if rt_data["current"] > 100:
                insights.append("🐌 Response time is above optimal threshold (>100ms)")
            if rt_data["trend"] == "increasing":
                insights.append("📈 Response time is trending upward - investigate potential bottlenecks")
            if rt_data["std_dev"] > 20:
                insights.append("📊 High response time variability detected - check system stability")
        
        # Throughput insights
        if "throughput" in metrics_analysis:
            tp_data = metrics_analysis["throughput"]
            if tp_data["trend"] == "decreasing":
                insights.append("📉 Throughput is decreasing - may indicate capacity constraints")
            if tp_data["current"] < tp_data["average"] * 0.8:
                insights.append("⚠️ Current throughput significantly below average")
        
        # Error rate insights
        if "error_rate" in metrics_analysis:
            er_data = metrics_analysis["error_rate"]
            if er_data["current"] > 0.01:  # 1%
                insights.append("🚨 Error rate above 1% - immediate attention required")
            if er_data["trend"] == "increasing":
                insights.append("📈 Error rate is increasing - check application logs")
        
        # System resource insights
        if "cpu_usage" in metrics_analysis:
            cpu_data = metrics_analysis["cpu_usage"]
            if cpu_data["current"] > 80:
                insights.append("🔥 High CPU usage detected - consider scaling")
            if cpu_data["max"] > 95:
                insights.append("⚠️ CPU usage peaked above 95% - investigate resource-intensive operations")
        
        if "memory_usage" in metrics_analysis:
            mem_data = metrics_analysis["memory_usage"]
            if mem_data["current"] > 85:
                insights.append("🧠 High memory usage detected - check for memory leaks")
            if mem_data["trend"] == "increasing":
                insights.append("📈 Memory usage trending upward - potential memory leak")
        
        # Cross-metric insights
        if ("response_time" in metrics_analysis and 
            "cpu_usage" in metrics_analysis and
            metrics_analysis["response_time"]["trend"] == "increasing" and
            metrics_analysis["cpu_usage"]["current"] > 70):
            insights.append("🔗 Response time increase correlates with high CPU usage")
        
        if not insights:
            insights.append("✅ All metrics are within normal ranges")
        
        return insights
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(minutes=5)  # Last 5 minutes
        
        # Get latest metrics
        latest_metrics = {}
        for metric_type in MetricType:
            metrics = self.metrics_collector.get_metrics(
                metric_name=metric_type,
                start_time=start_time,
                end_time=end_time,
                limit=1
            )
            if metrics:
                latest_metrics[metric_type] = metrics[0].value
        
        # Determine overall health status
        health_status = "healthy"
        health_score = 100
        
        # Check critical thresholds
        if latest_metrics.get(MetricType.RESPONSE_TIME, 0) > 150:
            health_status = "degraded"
            health_score -= 20
        
        if latest_metrics.get(MetricType.ERROR_RATE, 0) > 0.02:  # 2%
            health_status = "critical"
            health_score -= 30
        
        if latest_metrics.get(MetricType.CPU_USAGE, 0) > 90:
            health_status = "critical"
            health_score -= 25
        
        if latest_metrics.get(MetricType.MEMORY_USAGE, 0) > 90:
            health_status = "degraded" if health_status == "healthy" else health_status
            health_score -= 15
        
        return {
            "timestamp": end_time.isoformat(),
            "health_status": health_status,
            "health_score": max(0, health_score),
            "current_metrics": latest_metrics,
            "key_indicators": {
                "response_time_ok": latest_metrics.get(MetricType.RESPONSE_TIME, 0) <= 100,
                "error_rate_ok": latest_metrics.get(MetricType.ERROR_RATE, 0) <= 0.01,
                "cpu_usage_ok": latest_metrics.get(MetricType.CPU_USAGE, 0) <= 80,
                "memory_usage_ok": latest_metrics.get(MetricType.MEMORY_USAGE, 0) <= 85,
                "throughput_ok": latest_metrics.get(MetricType.THROUGHPUT, 0) >= 500
            }
        }


class AutoOptimizer:
    """Automated performance optimization engine"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.optimization_rules = []
        self.optimization_history = []
        self.enabled = True
    
    async def add_optimization_rule(self, rule: Dict[str, Any]):
        """Add an automated optimization rule"""
        self.optimization_rules.append(rule)
        logger.info(f"🔧 Added optimization rule: {rule['name']}")
    
    async def run_optimization_cycle(self):
        """Run automated optimization cycle"""
        if not self.enabled:
            return
        
        logger.info("🔧 Starting optimization cycle")
        
        # Get current performance summary
        analyzer = PerformanceAnalyzer(self.metrics_collector)
        summary = await analyzer.get_performance_summary()
        
        optimizations_applied = []
        
        # Check each optimization rule
        for rule in self.optimization_rules:
            try:
                if await self._should_apply_optimization(rule, summary):
                    result = await self._apply_optimization(rule)
                    if result:
                        optimizations_applied.append(rule['name'])
                        
                        # Record optimization
                        self.optimization_history.append({
                            "timestamp": datetime.now(UTC),
                            "rule_name": rule['name'],
                            "trigger_metrics": summary['current_metrics'],
                            "action": rule['action'],
                            "success": True
                        })
                        
            except Exception as e:
                logger.error(f"Optimization rule error ({rule['name']}): {e}")
        
        if optimizations_applied:
            logger.info(f"🔧 Applied optimizations: {', '.join(optimizations_applied)}")
        else:
            logger.info("🔧 No optimizations needed")
    
    async def _should_apply_optimization(self, rule: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        """Check if optimization rule should be applied"""
        conditions = rule.get("conditions", {})
        
        for metric_name, condition in conditions.items():
            current_value = summary["current_metrics"].get(metric_name, 0)
            
            if condition["operator"] == "gt" and current_value <= condition["threshold"]:
                return False
            elif condition["operator"] == "lt" and current_value >= condition["threshold"]:
                return False
            elif condition["operator"] == "eq" and abs(current_value - condition["threshold"]) > 0.001:
                return False
        
        # Check cooldown period
        last_application = None
        for history in reversed(self.optimization_history):
            if history["rule_name"] == rule["name"]:
                last_application = history["timestamp"]
                break
        
        if last_application:
            cooldown_minutes = rule.get("cooldown_minutes", 30)
            if datetime.now(UTC) - last_application < timedelta(minutes=cooldown_minutes):
                return False
        
        return True
    
    async def _apply_optimization(self, rule: Dict[str, Any]) -> bool:
        """Apply optimization action"""
        action = rule["action"]
        
        logger.info(f"🔧 Applying optimization: {rule['name']} - {action['type']}")
        
        # Simulate optimization actions
        if action["type"] == "scale_up":
            logger.info(f"📈 Scaling up: {action.get('component', 'application')} by {action.get('factor', 1.5)}x")
            return True
            
        elif action["type"] == "clear_cache":
            logger.info("🗑️ Clearing application cache")
            return True
            
        elif action["type"] == "restart_service":
            logger.info(f"🔄 Restarting service: {action.get('service', 'unknown')}")
            return True
            
        elif action["type"] == "adjust_connection_pool":
            logger.info(f"🔗 Adjusting connection pool size to {action.get('size', 50)}")
            return True
            
        elif action["type"] == "optimize_gc":
            logger.info("🗑️ Triggering garbage collection optimization")
            return True
        
        return False


# Notification Handlers

async def email_notification_handler(alert: PerformanceAlert, action: str):
    """Email notification handler"""
    logger.info(f"📧 EMAIL NOTIFICATION: {action.upper()} - {alert.message}")
    # In real implementation, would send actual email


async def slack_notification_handler(alert: PerformanceAlert, action: str):
    """Slack notification handler"""
    logger.info(f"💬 SLACK NOTIFICATION: {action.upper()} - {alert.message}")
    # In real implementation, would send to Slack webhook


async def pagerduty_notification_handler(alert: PerformanceAlert, action: str):
    """PagerDuty notification handler"""
    if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
        logger.info(f"📟 PAGERDUTY NOTIFICATION: {action.upper()} - {alert.message}")
        # In real implementation, would trigger PagerDuty incident


async def webhook_notification_handler(alert: PerformanceAlert, action: str):
    """Generic webhook notification handler"""
    logger.info(f"🪝 WEBHOOK NOTIFICATION: {action.upper()} - {alert.message}")
    # In real implementation, would POST to webhook URL


class PerformanceMonitor:
    """Main performance monitoring orchestrator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.auto_optimizer = AutoOptimizer(self.metrics_collector, self.alert_manager)
        self.running = False
    
    async def start(self):
        """Start the performance monitoring system"""
        logger.info("🚀 Starting MTen Performance Monitoring System")
        
        # Start components
        await self.metrics_collector.start()
        await self.alert_manager.start()
        
        # Configure default alert rules
        await self._setup_default_alert_rules()
        
        # Configure notification handlers
        await self._setup_notification_handlers()
        
        # Configure optimization rules
        await self._setup_optimization_rules()
        
        self.running = True
        logger.info("✅ Performance monitoring system started")
    
    async def stop(self):
        """Stop the performance monitoring system"""
        logger.info("🛑 Stopping performance monitoring system")
        
        await self.metrics_collector.stop()
        await self.alert_manager.stop()
        
        self.running = False
        logger.info("✅ Performance monitoring system stopped")
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name=MetricType.RESPONSE_TIME,
                condition="gt",
                threshold=150.0,  # 150ms
                severity=AlertSeverity.HIGH,
                window_minutes=5,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="critical_response_time",
                metric_name=MetricType.RESPONSE_TIME,
                condition="gt",
                threshold=300.0,  # 300ms
                severity=AlertSeverity.CRITICAL,
                window_minutes=2,
                notification_channels=["email", "slack", "pagerduty"]
            ),
            AlertRule(
                name="high_error_rate",
                metric_name=MetricType.ERROR_RATE,
                condition="gt",
                threshold=0.02,  # 2%
                severity=AlertSeverity.HIGH,
                window_minutes=3,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="critical_error_rate",
                metric_name=MetricType.ERROR_RATE,
                condition="gt",
                threshold=0.05,  # 5%
                severity=AlertSeverity.CRITICAL,
                window_minutes=2,
                notification_channels=["email", "slack", "pagerduty"]
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name=MetricType.CPU_USAGE,
                condition="gt",
                threshold=85.0,  # 85%
                severity=AlertSeverity.MEDIUM,
                window_minutes=10,
                notification_channels=["slack"]
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name=MetricType.CPU_USAGE,
                condition="gt",
                threshold=95.0,  # 95%
                severity=AlertSeverity.CRITICAL,
                window_minutes=5,
                notification_channels=["email", "slack", "pagerduty"]
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name=MetricType.MEMORY_USAGE,
                condition="gt",
                threshold=90.0,  # 90%
                severity=AlertSeverity.HIGH,
                window_minutes=10,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="low_throughput",
                metric_name=MetricType.THROUGHPUT,
                condition="lt",
                threshold=500.0,  # 500 RPS
                severity=AlertSeverity.MEDIUM,
                window_minutes=15,
                notification_channels=["slack"]
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    async def _setup_notification_handlers(self):
        """Setup notification handlers"""
        self.alert_manager.add_notification_handler("email", email_notification_handler)
        self.alert_manager.add_notification_handler("slack", slack_notification_handler)
        self.alert_manager.add_notification_handler("pagerduty", pagerduty_notification_handler)
        self.alert_manager.add_notification_handler("webhook", webhook_notification_handler)
    
    async def _setup_optimization_rules(self):
        """Setup automated optimization rules"""
        optimization_rules = [
            {
                "name": "high_response_time_cache_clear",
                "conditions": {
                    MetricType.RESPONSE_TIME: {"operator": "gt", "threshold": 200.0},
                    MetricType.CACHE_HIT_RATE: {"operator": "lt", "threshold": 0.8}
                },
                "action": {"type": "clear_cache"},
                "cooldown_minutes": 30
            },
            {
                "name": "high_cpu_gc_optimize",
                "conditions": {
                    MetricType.CPU_USAGE: {"operator": "gt", "threshold": 90.0}
                },
                "action": {"type": "optimize_gc"},
                "cooldown_minutes": 15
            },
            {
                "name": "db_connection_pool_adjust",
                "conditions": {
                    MetricType.DATABASE_CONNECTIONS: {"operator": "gt", "threshold": 80}
                },
                "action": {"type": "adjust_connection_pool", "size": 100},
                "cooldown_minutes": 60
            }
        ]
        
        for rule in optimization_rules:
            await self.auto_optimizer.add_optimization_rule(rule)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return await self.performance_analyzer.get_performance_summary()
    
    async def get_performance_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        return await self.performance_analyzer.analyze_performance_trends(hours)
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        alerts = self.alert_manager.get_active_alerts()
        return [
            {
                "rule_name": alert.rule_name,
                "severity": alert.severity,
                "message": alert.message,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in alerts
        ]
    
    async def run_optimization_cycle(self):
        """Manually trigger optimization cycle"""
        await self.auto_optimizer.run_optimization_cycle()


async def main():
    """Main entry point for performance monitoring system"""
    monitor = PerformanceMonitor()
    
    try:
        await monitor.start()
        
        logger.info("📊 Performance monitoring is running...")
        logger.info("🔧 Automated optimization is enabled")
        
        # Run for demonstration
        for i in range(10):
            await asyncio.sleep(30)  # Wait 30 seconds
            
            # Get and display health status
            health = await monitor.get_health_status()
            logger.info(f"Health Status: {health['health_status']} (Score: {health['health_score']})")
            
            # Check for active alerts
            active_alerts = await monitor.get_active_alerts()
            if active_alerts:
                logger.info(f"Active Alerts: {len(active_alerts)}")
            
            # Run optimization cycle every 2 minutes
            if i % 4 == 0:  # Every 4th iteration (2 minutes)
                await monitor.run_optimization_cycle()
        
        # Display final analysis
        analysis = await monitor.get_performance_analysis(hours=1)
        logger.info("📈 Performance Analysis Summary:")
        for insight in analysis.get("insights", []):
            logger.info(f"  {insight}")
        
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())