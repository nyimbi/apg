"""Service tests for bia_tsa Time Series Analytics."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import TimeSeriesService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_register_stream():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","Temp Sensor","mqtt","1m","u1","sensor://device-1"))
	assert s["state"] == "active" and s["protocol"] == "mqtt"

def test_list_streams_scoped():
	svc = TimeSeriesService()
	_run(svc.register_stream("t1","S1","mqtt","1m","u1","src://1"))
	_run(svc.register_stream("t2","S2","kafka","5m","u2","src://2"))
	assert len(_run(svc.list_streams("t1"))) == 1

def test_pause_and_resume_stream():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	p = _run(svc.pause_stream("t1",s["id"]))
	assert p["state"] == "paused"
	r = _run(svc.resume_stream("t1",s["id"]))
	assert r["state"] == "active"

def test_ingest_data():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	result = _run(svc.ingest_data("t1",s["id"],[{"ts":"2026-01-01T00:00:00","v":42.0},{"ts":"2026-01-01T00:01:00","v":43.1}]))
	assert result["points_ingested"] == 2
	s2 = _run(svc.get_stream("t1",s["id"]))
	assert s2["point_count"] == 2

def test_paused_stream_blocks_ingest():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	_run(svc.pause_stream("t1",s["id"]))
	try:
		_run(svc.ingest_data("t1",s["id"],[{"ts":"now","v":1.0}]))
		assert False, "Should raise"
	except ValueError:
		pass

def test_configure_anomaly_detection():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	ac = _run(svc.configure_anomaly_detection("t1",s["id"],"Spike Detector","zscore","u1"))
	assert ac["method"] == "zscore"

def test_detect_anomaly():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	ac = _run(svc.configure_anomaly_detection("t1",s["id"],"AD","zscore","u1"))
	ev = _run(svc.detect_anomaly("t1",s["id"],ac["id"],999.9,0.97))
	assert ev["severity"] == "high"

def test_run_decomposition():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1d","u1","src://1"))
	d = _run(svc.run_decomposition("t1",s["id"],["trend","seasonality","residual"]))
	assert len(d["trend_data"]) > 0 and len(d["seasonality_data"]) > 0

def test_create_forecast():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1d","u1","src://1"))
	f = _run(svc.create_forecast("t1",s["id"],"prophet",30,"u1"))
	assert len(f["forecast_data"]) == 30

def test_create_window():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	w = _run(svc.create_window("t1",s["id"],"5min avg","tumbling",300,"avg","u1"))
	assert w["window_type"] == "tumbling"

def test_fill_gaps():
	svc = TimeSeriesService()
	s = _run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	r = _run(svc.fill_gaps("t1",s["id"],"forward_fill"))
	assert r["status"] == "completed"

def test_stats():
	svc = TimeSeriesService()
	_run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	stats = _run(svc.get_stats("t1"))
	assert stats["stream_count"] == 1

def test_audit_events():
	svc = TimeSeriesService()
	_run(svc.register_stream("t1","S","mqtt","1m","u1","src://1"))
	assert len(_run(svc.get_audit_events("t1"))) >= 1
