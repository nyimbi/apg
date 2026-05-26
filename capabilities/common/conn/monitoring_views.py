"""
APG Connection Management Monitoring Views
Flask-AppBuilder views for monitoring and metrics visualization

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from flask import request, jsonify, render_template
from flask_appbuilder import BaseView, expose, has_access
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from .monitoring import (
    global_metrics_collector, global_health_checker,
    get_health_status, get_metrics, HealthStatus
)
from .service_bridge import with_service_bridge


class MonitoringDashboardView(BaseView):
    """Main monitoring dashboard view"""

    route_base = "/monitoring"
    default_view = "dashboard"

    @expose("/")
    @expose("/dashboard")
    @has_access
    async def dashboard(self):
        """Main monitoring dashboard"""
        try:
            # Get current metrics and health status
            metrics_data = get_metrics()
            health_data = await get_health_status()

            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(metrics_data)

            # Get recent performance data
            performance_data = self._get_performance_trends()

            return self.render_template(
                "monitoring_dashboard.html",
                metrics=metrics_data,
                health=health_data,
                summary=summary_stats,
                performance=performance_data
            )
        except Exception as e:
            return self.render_template(
                "monitoring_dashboard.html",
                error=f"Failed to load monitoring data: {str(e)}"
            )

    @expose("/health")
    @has_access
    async def health_status(self):
        """Detailed health status page"""
        try:
            health_data = await get_health_status()
            health_history = self._get_health_history()

            return self.render_template(
                "health_status.html",
                health=health_data,
                history=health_history
            )
        except Exception as e:
            return self.render_template(
                "health_status.html",
                error=f"Failed to load health data: {str(e)}"
            )

    @expose("/metrics")
    @has_access
    def metrics_detail(self):
        """Detailed metrics visualization"""
        try:
            metrics_data = get_metrics()
            metric_trends = self._get_metric_trends()

            return self.render_template(
                "metrics_detail.html",
                metrics=metrics_data,
                trends=metric_trends
            )
        except Exception as e:
            return self.render_template(
                "metrics_detail.html",
                error=f"Failed to load metrics data: {str(e)}"
            )

    @expose("/alerts")
    @has_access
    def alerts_dashboard(self):
        """Alerts and notifications dashboard"""
        try:
            alerts = self._get_active_alerts()
            alert_history = self._get_alert_history()

            return self.render_template(
                "alerts_dashboard.html",
                alerts=alerts,
                history=alert_history
            )
        except Exception as e:
            return self.render_template(
                "alerts_dashboard.html",
                error=f"Failed to load alerts data: {str(e)}"
            )

    # API Endpoints for real-time data

    @expose("/api/health")
    @has_access
    async def api_health(self):
        """API endpoint for health status"""
        try:
            health_data = await get_health_status()
            return jsonify(health_data)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500

    @expose("/api/metrics")
    @has_access
    def api_metrics(self):
        """API endpoint for metrics data"""
        try:
            metrics_data = get_metrics()
            return jsonify(metrics_data)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500

    @expose("/api/metrics/prometheus")
    @has_access
    def api_prometheus_metrics(self):
        """Prometheus-compatible metrics endpoint"""
        try:
            metrics = get_metrics()
            prometheus_format = self._convert_to_prometheus_format(metrics)

            return prometheus_format, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        except Exception as e:
            return f"# ERROR: {str(e)}", 500, {'Content-Type': 'text/plain; charset=utf-8'}

    @expose("/api/performance")
    @has_access
    @with_service_bridge
    def api_performance(self, service_bridge=None):
        """API endpoint for performance data"""
        try:
            # Get performance data from service bridge
            performance_data = {
                'connections': {
                    'total': 0,
                    'active': 0,
                    'failed': 0
                },
                'flows': {
                    'total': 0,
                    'running': 0,
                    'completed': 0,
                    'failed': 0
                },
                'throughput': {
                    'records_per_second': 0,
                    'bytes_per_second': 0
                },
                'latency': {
                    'avg_ms': 0,
                    'p95_ms': 0,
                    'p99_ms': 0
                }
            }

            if service_bridge and hasattr(service_bridge, 'connection_manager'):
                # Get real performance data
                connections = service_bridge.connection_manager.connections
                flows = service_bridge.connection_manager.flows

                performance_data['connections'] = {
                    'total': len(connections),
                    'active': len([c for c in connections.values() if c.status.value == 'active']),
                    'failed': len([c for c in connections.values() if c.status.value == 'error'])
                }

                performance_data['flows'] = {
                    'total': len(flows),
                    'running': len([f for f in flows.values() if f.status.value == 'running']),
                    'completed': len([f for f in flows.values() if f.status.value == 'completed']),
                    'failed': len([f for f in flows.values() if f.status.value == 'failed'])
                }

            return jsonify(performance_data)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500

    @expose("/api/alerts")
    @has_access
    def api_alerts(self):
        """API endpoint for alerts data"""
        try:
            alerts = self._get_active_alerts()
            return jsonify({
                'alerts': alerts,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500

    # Helper methods

    def _calculate_summary_stats(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics data"""
        counters = metrics_data.get('counters', {})
        gauges = metrics_data.get('gauges', {})

        return {
            'total_requests': counters.get('requests_total', 0),
            'error_rate': self._calculate_error_rate(counters),
            'active_connections': gauges.get('active_connections', 0),
            'active_flows': gauges.get('active_flows', 0),
            'system_health': self._get_system_health_score(),
            'uptime_hours': gauges.get('uptime_seconds', 0) / 3600,
            'memory_usage_mb': gauges.get('system_memory_mb', 0),
            'cpu_usage_percent': gauges.get('system_cpu_percent', 0)
        }

    def _calculate_error_rate(self, counters: Dict[str, Any]) -> float:
        """Calculate error rate from counter metrics"""
        total_requests = counters.get('requests_total', 0)
        error_requests = counters.get('errors', 0)

        if total_requests == 0:
            return 0.0

        return (error_requests / total_requests) * 100

    def _get_system_health_score(self) -> int:
        """Calculate system health score (0-100)"""
        try:
            # Mock implementation - in reality would analyze various health metrics
            health_history = getattr(global_health_checker, 'health_history', [])
            if not health_history:
                return 100

            latest_health = health_history[-1]['results']
            healthy_components = sum(1 for result in latest_health.values()
                                  if result.status == HealthStatus.HEALTHY)
            total_components = len(latest_health)

            if total_components == 0:
                return 100

            return int((healthy_components / total_components) * 100)
        except Exception:
            return 85  # Default healthy score

    def _get_performance_trends(self) -> Dict[str, List]:
        """Get performance trend data for charts"""
        # Mock trend data - in reality would be calculated from historical metrics
        now = datetime.now(timezone.utc)
        timestamps = [(now - timedelta(minutes=i*5)).isoformat() for i in range(12, 0, -1)]

        return {
            'timestamps': timestamps,
            'request_rate': [45, 52, 48, 61, 55, 58, 63, 59, 67, 64, 70, 68],
            'error_rate': [2.1, 1.8, 2.3, 1.5, 1.9, 2.0, 1.7, 2.2, 1.6, 1.8, 1.4, 1.9],
            'response_time': [120, 135, 118, 142, 128, 131, 125, 139, 122, 127, 133, 129],
            'cpu_usage': [65, 68, 62, 71, 67, 69, 64, 73, 66, 70, 68, 72],
            'memory_usage': [78, 79, 77, 81, 80, 82, 79, 83, 81, 84, 82, 85]
        }

    def _get_health_history(self) -> List[Dict[str, Any]]:
        """Get health check history"""
        history = getattr(global_health_checker, 'health_history', [])
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'overall_status': self._calculate_overall_status(entry['results']),
                'component_count': len(entry['results']),
                'healthy_count': sum(1 for r in entry['results'].values()
                                   if r.status == HealthStatus.HEALTHY)
            }
            for entry in history[-20:]  # Last 20 health checks
        ]

    def _calculate_overall_status(self, results: Dict) -> str:
        """Calculate overall status from component results"""
        statuses = [result.status for result in results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY.value
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED.value
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY.value
        else:
            return HealthStatus.UNKNOWN.value

    def _get_metric_trends(self) -> Dict[str, Any]:
        """Get metric trend data"""
        # Mock implementation - would analyze historical metrics
        return {
            'connection_trends': {
                'created_per_hour': [5, 7, 6, 8, 5, 9, 7, 6, 8, 7, 9, 8],
                'failed_per_hour': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
            },
            'flow_trends': {
                'executed_per_hour': [25, 32, 28, 35, 30, 38, 33, 29, 36, 31, 39, 34],
                'avg_duration_minutes': [2.5, 2.8, 2.3, 2.9, 2.6, 2.7, 2.4, 2.8, 2.5, 2.6, 2.7, 2.6]
            },
            'data_trends': {
                'processed_gb_per_hour': [1.2, 1.8, 1.5, 2.1, 1.7, 2.3, 1.9, 1.6, 2.0, 1.8, 2.2, 2.0]
            }
        }

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        # Mock alerts - in reality would check actual alert conditions
        alerts = []

        # Check for high error rate
        metrics = get_metrics()
        error_rate = self._calculate_error_rate(metrics.get('counters', {}))
        if error_rate > 5.0:
            alerts.append({
                'id': 'high_error_rate',
                'severity': 'warning',
                'title': 'High Error Rate',
                'message': f'Error rate is {error_rate:.1f}% (threshold: 5.0%)',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'component': 'overall'
            })

        # Check for high memory usage
        gauges = metrics.get('gauges', {})
        memory_mb = gauges.get('system_memory_mb', 0)
        if memory_mb > 1000:  # > 1GB
            alerts.append({
                'id': 'high_memory',
                'severity': 'warning',
                'title': 'High Memory Usage',
                'message': f'Memory usage is {memory_mb:.0f}MB (threshold: 1000MB)',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'component': 'system'
            })

        return alerts

    def _get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history"""
        # Mock alert history
        return [
            {
                'id': 'connection_timeout',
                'severity': 'error',
                'title': 'Connection Timeout',
                'message': 'Database connection timed out after 30 seconds',
                'timestamp': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                'resolved': True,
                'component': 'database'
            },
            {
                'id': 'high_cpu',
                'severity': 'warning',
                'title': 'High CPU Usage',
                'message': 'CPU usage exceeded 80% for 5 minutes',
                'timestamp': (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
                'resolved': True,
                'component': 'system'
            }
        ]

    def _convert_to_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Convert metrics to Prometheus format"""
        lines = []

        # Add counters
        for name, value in metrics.get('counters', {}).items():
            lines.append(f"# HELP apg_{name} Total count of {name}")
            lines.append(f"# TYPE apg_{name} counter")
            lines.append(f"apg_{name} {value}")
            lines.append("")

        # Add gauges
        for name, value in metrics.get('gauges', {}).items():
            lines.append(f"# HELP apg_{name} Current value of {name}")
            lines.append(f"# TYPE apg_{name} gauge")
            lines.append(f"apg_{name} {value}")
            lines.append("")

        # Add histograms
        for name, data in metrics.get('timers', {}).items():
            lines.append(f"# HELP apg_{name} Histogram of {name}")
            lines.append(f"# TYPE apg_{name} histogram")
            lines.append(f"apg_{name}_count {data.get('count', 0)}")
            lines.append(f"apg_{name}_sum {data.get('avg', 0) * data.get('count', 0)}")
            lines.append("")

        return '\n'.join(lines)


class LogsView(BaseView):
    """Logs viewing interface"""

    route_base = "/logs"

    @expose("/")
    @has_access
    def logs_dashboard(self):
        """Logs dashboard"""
        try:
            log_data = self._get_recent_logs()
            log_stats = self._get_log_statistics()

            return self.render_template(
                "logs_dashboard.html",
                logs=log_data,
                stats=log_stats
            )
        except Exception as e:
            return self.render_template(
                "logs_dashboard.html",
                error=f"Failed to load log data: {str(e)}"
            )

    @expose("/api/logs")
    @has_access
    def api_logs(self):
        """API endpoint for recent logs"""
        try:
            level = request.args.get('level', 'INFO')
            limit = int(request.args.get('limit', 100))

            logs = self._get_recent_logs(level, limit)
            return jsonify({
                'logs': logs,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500

    def _get_recent_logs(self, level: str = 'INFO', limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        # Mock log data - in reality would read from log files or log aggregation system
        import random

        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        components = ['connection_manager', 'flow_executor', 'lineage_engine', 'service_bridge']

        logs = []
        for i in range(limit):
            timestamp = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 1440))
            logs.append({
                'timestamp': timestamp.isoformat(),
                'level': random.choice(log_levels),
                'component': random.choice(components),
                'message': f"Mock log message {i + 1}",
                'request_id': f"req-{random.randint(10000, 99999)}"
            })

        # Filter by level if specified
        if level != 'ALL':
            level_priority = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
            min_priority = level_priority.get(level, 1)
            logs = [log for log in logs if level_priority.get(log['level'], 1) >= min_priority]

        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

    def _get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        return {
            'total_logs_today': 15420,
            'errors_today': 23,
            'warnings_today': 156,
            'avg_logs_per_hour': 643,
            'top_components': [
                {'name': 'connection_manager', 'count': 5420},
                {'name': 'flow_executor', 'count': 4280},
                {'name': 'lineage_engine', 'count': 3150},
                {'name': 'service_bridge', 'count': 2570}
            ]
        }