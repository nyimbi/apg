"""Focused tests for executable data-quality view state."""

from datetime import datetime, timezone

from capabilities.common.conn.data_quality import (
	DataQualityDimension,
	DataQualityIssue,
	DataQualityLevel,
	DataQualityMetrics,
	IssueSeverity,
	IssueType,
	global_data_quality_monitor,
)
from capabilities.common.conn.data_quality_views import (
	DataQualityConnectionView,
	DataQualityDashboardView,
)


def setup_function():
	global_data_quality_monitor.quality_history.clear()


def _metrics(connection_id: str, score: float, level: DataQualityLevel) -> DataQualityMetrics:
	issue = DataQualityIssue(
		issue_type=IssueType.MISSING_VALUE,
		severity=IssueSeverity.HIGH,
		dimension=DataQualityDimension.COMPLETENESS,
		description="Missing values detected",
		field_name="email",
	)
	return DataQualityMetrics(
		total_records=10,
		valid_records=8,
		completeness_score=80,
		accuracy_score=100,
		consistency_score=100,
		validity_score=90,
		uniqueness_score=100,
		timeliness_score=100,
		integrity_score=100,
		overall_score=score,
		quality_level=level,
		issues=[issue],
		profiling_stats={
			"connection_id": connection_id,
			"connection_name": f"Connection {connection_id}",
		},
		assessment_timestamp=datetime.now(timezone.utc),
	)


def test_data_quality_dashboard_uses_monitor_history():
	view = DataQualityDashboardView()
	global_data_quality_monitor.quality_history.append(
		_metrics("conn-a", 82.5, DataQualityLevel.GOOD)
	)

	stats = view._get_connections_quality_stats()
	distribution = view._get_quality_level_distribution()
	top_issues = view._get_top_quality_issues()

	assert stats == [{
		"connection_id": "conn-a",
		"name": "Connection conn-a",
		"quality_score": 82.5,
		"last_assessed": stats[0]["last_assessed"],
	}]
	assert distribution["good"] == 1
	assert top_issues[0] == {
		"issue_type": "missing_value",
		"count": 1,
		"severity": "high",
	}


def test_data_quality_view_extracts_embedded_sample_records():
	view = DataQualityDashboardView()
	connection = type(
		"ConnectionRecord",
		(),
		{
			"meta_data": {"sample_records": [{"id": 1, "email": "a@example.com"}]},
			"tap_config": {},
			"target_config": {},
		},
	)()

	assert view._get_assessment_sample_data("conn-a", connection) == [
		{"id": 1, "email": "a@example.com"}
	]


def test_data_quality_connection_view_metrics_come_from_history():
	view = DataQualityConnectionView()
	global_data_quality_monitor.quality_history.append(
		_metrics("conn-a", 66.0, DataQualityLevel.FAIR)
	)

	metrics = view._quality_metrics_for_connection("conn-a")

	assert metrics["overall_score"] == 66.0
	assert metrics["quality_level"] == "fair"
	assert metrics["issues_count"] == 1
	assert metrics["trends"]["recent_scores"] == [66.0]
