"""Focused tests for executable CONN monitoring runtime state."""

from capabilities.common.conn import service
from capabilities.common.conn.models import Connection, ConnectionStatus, ConnectionType
from capabilities.common.conn.monitoring import MetricsCollector
from capabilities.common.conn.service import ConnectionManager


def test_metrics_collector_tracks_active_connections_and_flows():
	collector = MetricsCollector(enable_otel=False)

	collector.register_active_connection("conn-b")
	collector.register_active_connection("conn-a")
	collector.register_active_flow("flow-1")

	assert collector.get_active_connections() == ["conn-a", "conn-b"]
	assert collector.get_active_flows() == ["flow-1"]
	assert collector.gauges["active_connections"] == 2
	assert collector.gauges["active_flows"] == 1

	collector.unregister_active_connection("conn-a")
	collector.unregister_active_flow("flow-1")

	assert collector.get_active_connections() == ["conn-b"]
	assert collector.get_active_flows() == []
	assert collector.gauges["active_connections"] == 1
	assert collector.gauges["active_flows"] == 0


def test_connection_manager_syncs_active_connection_metric(monkeypatch):
	collector = MetricsCollector(enable_otel=False)
	monkeypatch.setattr(service, "global_metrics_collector", collector)
	manager = ConnectionManager(audit_enabled=False)
	connection = Connection(
		id="conn-1",
		tenant_id="tenant-a",
		name="Primary API",
		connection_type=ConnectionType.API,
		status=ConnectionStatus.ACTIVE
	)

	manager._sync_connection_monitoring(connection)
	assert collector.get_active_connections() == ["conn-1"]

	connection.status = ConnectionStatus.INACTIVE
	manager._sync_connection_monitoring(connection)
	assert collector.get_active_connections() == []
