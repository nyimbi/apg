"""Focused tests for executable ML insights view state."""

from capabilities.common.conn.ml_insights import AnalysisType
from capabilities.common.conn.ml_insights_views import ANALYSIS_JOBS, MLInsightsDashboardView


def setup_function():
	ANALYSIS_JOBS.clear()


def test_ml_insights_view_executes_and_stores_analysis_job():
	view = MLInsightsDashboardView()

	job = view._execute_analysis_job(
		connection_id="conn-runtime",
		analysis_types=[AnalysisType.DATA_PROFILING],
		sample_size=120
	)

	assert job["status"] == "completed"
	assert job["progress"] == 100
	assert job["sample_size"] == 120
	assert job["insights_generated"] >= 1
	assert ANALYSIS_JOBS[job["job_id"]] == job
	assert job["insights"][0]["analysis_type"] == "data_profiling"


def test_ml_insights_view_stats_come_from_stored_jobs():
	view = MLInsightsDashboardView()
	job = view._execute_analysis_job(
		connection_id="conn-runtime",
		analysis_types=[AnalysisType.DATA_PROFILING],
		sample_size=120
	)

	stats = view._get_analysis_type_stats()
	recent = view._get_recent_insights()
	top_connections = view._get_top_analyzed_connections()

	assert stats["data_profiling"] == job["insights_generated"]
	assert recent[0]["insight_id"] == job["insights"][0]["insight_id"]
	assert top_connections[0]["name"] == "conn-runtime"
	assert top_connections[0]["insights_count"] == job["insights_generated"]


def test_ml_insights_view_uses_embedded_connection_sample_records():
	view = MLInsightsDashboardView()
	connection = type(
		"ConnectionRecord",
		(),
		{
			"meta_data": {
				"sample_records": [
					{"id": 1, "category": "alpha", "value": 10},
					{"id": 2, "category": "beta", "value": 20},
				]
			},
			"tap_config": {},
			"target_config": {},
		}
	)()

	sample = view._connection_sample_records(connection)

	assert sample == [
		{"id": 1, "category": "alpha", "value": 10},
		{"id": 2, "category": "beta", "value": 20},
	]
