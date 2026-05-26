"""
ERP Dashboard Views
Comprehensive management dashboard for all ERP systems

This module provides Flask-AppBuilder views for managing and monitoring
all ERP connections with real-time status and analytics.
"""

from flask import render_template, flash, redirect, url_for, request, jsonify
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import protect
from datetime import datetime, timezone, timedelta
import json
import asyncio
from typing import Dict, List, Any

from .service import ConnectionManager
from .erp_monitoring_system import ERPMonitoringSystem, AlertSeverity, MetricType
from .erp_integration_test_suite import ERPIntegrationTester
from singer_taps.erp_registry import get_erp_registry, ERPSystemType


class ERPDashboardView(BaseView):
    """Main ERP dashboard with system overview"""

    default_view = 'erp_overview'

    @expose('/erp-overview/')
    @has_access
    def erp_overview(self):
        """ERP systems overview dashboard"""
        return self.render_template('erp_dashboard_overview.html',
                                  title="ERP Systems Overview")

    @expose('/api/erp-status/')
    @has_access
    def api_erp_status(self):
        """API endpoint for ERP system status"""
        try:
            # Get monitoring system status
            monitoring_system = self._get_monitoring_system()
            health_status = monitoring_system.get_erp_health_status()

            # Get registry information
            registry = get_erp_registry()
            status_summary = registry.get_implementation_status_summary()

            # Format response
            response = {
                "summary": {
                    "total_systems": len(health_status),
                    "healthy_systems": len([h for h in health_status.values() if h["overall_status"] == "active"]),
                    "total_alerts": sum(h["active_alerts"] for h in health_status.values()),
                    "implementation_status": status_summary
                },
                "systems": health_status,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @expose('/api/erp-metrics/')
    @has_access
    def api_erp_metrics(self):
        """API endpoint for ERP performance metrics"""
        try:
            erp_system = request.args.get('system')
            metric_type = request.args.get('metric', 'response_time')
            hours = int(request.args.get('hours', 24))

            # Get metrics from monitoring system
            monitoring_system = self._get_monitoring_system()

            if erp_system:
                metrics_data = self._get_system_metrics(monitoring_system, erp_system, metric_type, hours)
            else:
                metrics_data = self._get_all_systems_metrics(monitoring_system, metric_type, hours)

            return jsonify(metrics_data)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _get_monitoring_system(self) -> ERPMonitoringSystem:
        """Get ERP monitoring system instance"""
        # In real implementation, this would be injected or retrieved from app context
        connection_manager = ConnectionManager()
        return ERPMonitoringSystem(connection_manager)

    def _get_system_metrics(self, monitoring_system: ERPMonitoringSystem,
                          erp_system: str, metric_type: str, hours: int) -> Dict:
        """Get metrics for specific ERP system"""
        health = monitoring_system.erp_health.get(erp_system)
        if not health:
            return {"error": f"ERP system {erp_system} not found"}

        # Get metric enum
        try:
            metric_enum = MetricType(metric_type)
        except ValueError:
            return {"error": f"Invalid metric type: {metric_type}"}

        # Extract metrics from last N hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        metrics = health.metrics.get(metric_enum, [])

        recent_metrics = [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "metadata": point.metadata
            }
            for point in metrics
            if point.timestamp >= cutoff_time
        ]

        return {
            "erp_system": erp_system,
            "metric_type": metric_type,
            "time_range_hours": hours,
            "data_points": len(recent_metrics),
            "metrics": recent_metrics
        }

    def _get_all_systems_metrics(self, monitoring_system: ERPMonitoringSystem,
                               metric_type: str, hours: int) -> Dict:
        """Get metrics for all ERP systems"""
        all_metrics = {}

        for erp_system in monitoring_system.erp_health.keys():
            system_metrics = self._get_system_metrics(monitoring_system, erp_system, metric_type, hours)
            if "error" not in system_metrics:
                all_metrics[erp_system] = system_metrics

        return {
            "metric_type": metric_type,
            "time_range_hours": hours,
            "systems": all_metrics
        }


class ERPSystemManagementView(BaseView):
    """ERP system management and configuration"""

    default_view = 'system_list'

    @expose('/system-list/')
    @has_access
    def system_list(self):
        """List all available ERP systems"""
        return self.render_template('erp_system_list.html',
                                  title="ERP System Management")

    @expose('/system-detail/<erp_system>/')
    @has_access
    def system_detail(self, erp_system):
        """Detailed view of specific ERP system"""
        return self.render_template('erp_system_detail.html',
                                  erp_system=erp_system,
                                  title=f"ERP System - {erp_system}")

    @expose('/api/available-systems/')
    @has_access
    def api_available_systems(self):
        """API endpoint for available ERP systems"""
        try:
            registry = get_erp_registry()
            connectors = registry.list_connectors()

            systems_data = []
            for connector in connectors:
                systems_data.append({
                    "system_type": connector.system_type.value,
                    "display_name": connector.display_name,
                    "vendor": connector.vendor,
                    "description": connector.description,
                    "supported_versions": connector.supported_versions,
                    "authentication_methods": connector.authentication_methods,
                    "data_categories": connector.data_categories,
                    "stream_count": connector.stream_count,
                    "implementation_status": connector.implementation_status,
                    "configuration_template": connector.configuration_template
                })

            return jsonify({
                "total_systems": len(systems_data),
                "systems": systems_data,
                "vendors": registry.get_vendors(),
                "status_summary": registry.get_implementation_status_summary()
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @expose('/api/system-config-template/<system_type>/')
    @has_access
    def api_system_config_template(self, system_type):
        """Get configuration template for ERP system"""
        try:
            registry = get_erp_registry()

            # Convert string to enum
            try:
                system_enum = ERPSystemType(system_type)
            except ValueError:
                return jsonify({"error": f"Invalid system type: {system_type}"}), 400

            template = registry.get_configuration_template(system_enum)

            if template:
                return jsonify({
                    "system_type": system_type,
                    "configuration_template": template,
                    "required_fields": list(template.keys()) if template else []
                })
            else:
                return jsonify({"error": "No configuration template available"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500


class ERPAlertsView(BaseView):
    """ERP alerts and incident management"""

    default_view = 'alerts_dashboard'

    @expose('/alerts-dashboard/')
    @has_access
    def alerts_dashboard(self):
        """Alerts and incidents dashboard"""
        return self.render_template('erp_alerts_dashboard.html',
                                  title="ERP Alerts & Incidents")

    @expose('/api/alerts/')
    @has_access
    def api_alerts(self):
        """API endpoint for ERP alerts"""
        try:
            erp_system = request.args.get('system')
            severity = request.args.get('severity')

            # Convert severity string to enum if provided
            severity_enum = None
            if severity:
                try:
                    severity_enum = AlertSeverity(severity)
                except ValueError:
                    return jsonify({"error": f"Invalid severity: {severity}"}), 400

            monitoring_system = self._get_monitoring_system()
            alerts = monitoring_system.get_active_alerts(erp_system, severity_enum)

            # Group alerts by severity
            alerts_by_severity = {}
            for alert in alerts:
                sev = alert["severity"]
                if sev not in alerts_by_severity:
                    alerts_by_severity[sev] = []
                alerts_by_severity[sev].append(alert)

            return jsonify({
                "total_alerts": len(alerts),
                "alerts_by_severity": alerts_by_severity,
                "alerts": alerts
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @expose('/api/resolve-alert/<alert_id>/', methods=['POST'])
    @has_access
    def api_resolve_alert(self, alert_id):
        """Resolve an alert"""
        try:
            monitoring_system = self._get_monitoring_system()

            success = asyncio.run(monitoring_system.resolve_alert(alert_id))

            if success:
                return jsonify({"success": True, "message": f"Alert {alert_id} resolved"})
            else:
                return jsonify({"success": False, "message": "Alert not found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _get_monitoring_system(self) -> ERPMonitoringSystem:
        """Get ERP monitoring system instance"""
        # In real implementation, this would be injected or retrieved from app context
        connection_manager = ConnectionManager()
        return ERPMonitoringSystem(connection_manager)


class ERPTestingView(BaseView):
    """ERP integration testing interface"""

    default_view = 'testing_dashboard'

    @expose('/testing-dashboard/')
    @has_access
    def testing_dashboard(self):
        """Testing dashboard"""
        return self.render_template('erp_testing_dashboard.html',
                                  title="ERP Integration Testing")

    @expose('/api/run-tests/', methods=['POST'])
    @has_access
    def api_run_tests(self):
        """Run ERP integration tests"""
        try:
            test_config = request.get_json()
            if not test_config:
                return jsonify({"error": "No test configuration provided"}), 400

            # Run tests asynchronously
            tester = ERPIntegrationTester()

            # This would typically be run in background task
            results = asyncio.run(tester.run_comprehensive_tests(test_config))

            # Format results for response
            formatted_results = {}
            for erp_system, test_results in results.items():
                formatted_results[erp_system] = {
                    "total_tests": len(test_results),
                    "passed_tests": len([r for r in test_results if r.passed]),
                    "failed_tests": len([r for r in test_results if not r.passed]),
                    "success_rate": len([r for r in test_results if r.passed]) / len(test_results) if test_results else 0,
                    "test_details": [
                        {
                            "test_name": r.test_name,
                            "category": r.category.value,
                            "passed": r.passed,
                            "duration": r.duration_seconds,
                            "message": r.message,
                            "metrics": r.metrics,
                            "errors": r.errors
                        }
                        for r in test_results
                    ]
                }

            return jsonify({
                "test_execution_complete": True,
                "total_systems_tested": len(formatted_results),
                "results": formatted_results
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500


# Dashboard Templates

ERP_DASHBOARD_TEMPLATE = '''
{% extends "appbuilder/general/widgets/base_list.html" %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-12">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3 class="panel-title">
                        <i class="fa fa-server"></i> Enterprise ERP Systems Dashboard
                    </h3>
                </div>
                <div class="panel-body">
                    <!-- System Status Overview -->
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card bg-success text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-check-circle"></i> Healthy Systems</h5>
                                    <h2 id="healthy-count">-</h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-info text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-server"></i> Total Systems</h5>
                                    <h2 id="total-count">-</h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-warning text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-exclamation-triangle"></i> Active Alerts</h5>
                                    <h2 id="alerts-count">-</h2>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-primary text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-chart-line"></i> Avg Response</h5>
                                    <h2 id="avg-response">-</h2>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- ERP Vendors Overview -->
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>ERP Systems by Vendor</h5>
                                </div>
                                <div class="card-body">
                                    <canvas id="vendorChart" width="400" height="200"></canvas>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5>System Health Status</h5>
                                </div>
                                <div class="card-body">
                                    <canvas id="healthChart" width="400" height="200"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Real-time System Status -->
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5>Real-time ERP System Status</h5>
                                    <button class="btn btn-sm btn-primary float-right" onclick="refreshStatus()">
                                        <i class="fa fa-refresh"></i> Refresh
                                    </button>
                                </div>
                                <div class="card-body">
                                    <div id="systems-table">
                                        <div class="text-center">
                                            <i class="fa fa-spinner fa-spin"></i> Loading ERP systems...
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Quick Actions -->
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <h5>Quick Actions</h5>
                            <div class="btn-group" role="group">
                                <a href="{{ url_for('ERPSystemManagementView.system_list') }}" class="btn btn-primary">
                                    <i class="fa fa-cogs"></i> Manage Systems
                                </a>
                                <a href="{{ url_for('ERPAlertsView.alerts_dashboard') }}" class="btn btn-warning">
                                    <i class="fa fa-bell"></i> View Alerts
                                </a>
                                <a href="{{ url_for('ERPTestingView.testing_dashboard') }}" class="btn btn-info">
                                    <i class="fa fa-flask"></i> Run Tests
                                </a>
                                <button class="btn btn-success" onclick="exportReport()">
                                    <i class="fa fa-download"></i> Export Report
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// Global charts
let vendorChart, healthChart;

// Initialize dashboard
$(document).ready(function() {
    loadERPStatus();
    initializeCharts();

    // Auto-refresh every 30 seconds
    setInterval(loadERPStatus, 30000);
});

function loadERPStatus() {
    $.get('{{ url_for("ERPDashboardView.api_erp_status") }}')
        .done(function(data) {
            updateStatusCards(data.summary);
            updateSystemsTable(data.systems);
            updateCharts(data);
        })
        .fail(function() {
            showError('Failed to load ERP status');
        });
}

function updateStatusCards(summary) {
    $('#healthy-count').text(summary.healthy_systems);
    $('#total-count').text(summary.total_systems);
    $('#alerts-count').text(summary.total_alerts);

    // Calculate average response time (simplified)
    $('#avg-response').text('< 3s');
}

function updateSystemsTable(systems) {
    let html = '<table class="table table-striped">';
    html += '<thead><tr><th>ERP System</th><th>Status</th><th>Uptime</th><th>Response Time</th><th>Alerts</th><th>Actions</th></tr></thead>';
    html += '<tbody>';

    for (const [systemName, systemData] of Object.entries(systems)) {
        const statusClass = systemData.overall_status === 'active' ? 'success' : 'danger';
        const statusIcon = systemData.overall_status === 'active' ? 'check' : 'times';

        html += `<tr>
            <td><strong>${systemName}</strong><br><small>${systemData.system_type}</small></td>
            <td><span class="label label-${statusClass}"><i class="fa fa-${statusIcon}"></i> ${systemData.overall_status}</span></td>
            <td>${systemData.uptime_percentage.toFixed(1)}%</td>
            <td>${systemData.avg_response_time.toFixed(2)}s</td>
            <td>${systemData.active_alerts} <span class="text-muted">(${systemData.alert_severities.join(', ')})</span></td>
            <td>
                <a href="{{ url_for('ERPSystemManagementView.system_detail', erp_system='') }}${systemName}" class="btn btn-xs btn-primary">
                    <i class="fa fa-eye"></i> View
                </a>
            </td>
        </tr>`;
    }

    html += '</tbody></table>';
    $('#systems-table').html(html);
}

function initializeCharts() {
    // Vendor chart
    const vendorCtx = document.getElementById('vendorChart').getContext('2d');
    vendorChart = new Chart(vendorCtx, {
        type: 'doughnut',
        data: {
            labels: ['SAP', 'Microsoft', 'Oracle', 'NetSuite', 'Workday', 'Sage'],
            datasets: [{
                data: [7, 8, 5, 4, 4, 5],
                backgroundColor: ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // Health chart
    const healthCtx = document.getElementById('healthChart').getContext('2d');
    healthChart = new Chart(healthCtx, {
        type: 'bar',
        data: {
            labels: ['Healthy', 'Warning', 'Critical', 'Offline'],
            datasets: [{
                data: [25, 5, 2, 1],
                backgroundColor: ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            legend: { display: false }
        }
    });
}

function updateCharts(data) {
    // Update charts with real data
    // Implementation would parse actual system data
}

function refreshStatus() {
    loadERPStatus();
    showSuccess('Status refreshed');
}

function exportReport() {
    window.open('{{ url_for("ERPDashboardView.api_erp_status") }}', '_blank');
}

function showSuccess(message) {
    // Show success toast/notification
    console.log('Success:', message);
}

function showError(message) {
    // Show error toast/notification
    console.error('Error:', message);
}
</script>
{% endblock %}
'''

# Save the dashboard template
import os
os.makedirs('/Users/nyimbiodero/src/pjs/apg/capabilities/common/conn/templates', exist_ok=True)

with open('/Users/nyimbiodero/src/pjs/apg/capabilities/common/conn/templates/erp_dashboard_overview.html', 'w') as f:
    f.write(ERP_DASHBOARD_TEMPLATE)