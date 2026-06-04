"""Service tests for bia_pda Predictive Analytics."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import PredictiveAnalyticsService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_train_model():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","Sales Forecast","prophet","u1","ds1"))
	assert m["model_type"] == "prophet" and m["state"] == "trained"

def test_deploy_model():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M1","arima","u1","ds1"))
	d = _run(svc.deploy_model("t1",m["id"]))
	assert d["state"] == "deployed"

def test_deprecate_model():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M1","arima","u1","ds1"))
	dep = _run(svc.deprecate_model("t1",m["id"]))
	assert dep["state"] == "deprecated"

def test_list_models_scoped():
	svc = PredictiveAnalyticsService()
	_run(svc.train_model("t1","M1","prophet","u1","ds1"))
	_run(svc.train_model("t2","M2","arima","u2","ds2"))
	assert len(_run(svc.list_models("t1"))) == 1

def test_generate_forecast():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M","prophet","u1","ds1"))
	_run(svc.deploy_model("t1",m["id"]))
	f = _run(svc.generate_forecast("t1",m["id"],"7d","u1"))
	assert f["horizon"] == "7d" and len(f["forecast_data"]) > 0

def test_simulate_scenario():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M","prophet","u1","ds1"))
	sc = _run(svc.simulate_scenario("t1",m["id"],"Optimistic Q4","optimistic",{"growth": 0.15},"u1"))
	assert sc["scenario_type"] == "optimistic"

def test_delete_scenario():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M","prophet","u1","ds1"))
	sc = _run(svc.simulate_scenario("t1",m["id"],"S","base",{},"u1"))
	ok = _run(svc.delete_scenario("t1",sc["id"]))
	assert ok

def test_register_feature():
	svc = PredictiveAnalyticsService()
	feat = _run(svc.register_feature("t1","revenue","numerical","revenue_col","ds1","u1"))
	assert feat["feature_type"] == "numerical"

def test_serve_prediction():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M","random_forest","u1","ds1"))
	_run(svc.deploy_model("t1",m["id"]))
	pred = _run(svc.serve_prediction("t1",m["id"],{"feature_a": 1.0}))
	assert "output" in pred and pred["confidence"] is not None

def test_list_forecasts_by_model():
	svc = PredictiveAnalyticsService()
	m = _run(svc.train_model("t1","M","prophet","u1","ds1"))
	_run(svc.deploy_model("t1",m["id"]))
	_run(svc.generate_forecast("t1",m["id"],"7d","u1"))
	_run(svc.generate_forecast("t1",m["id"],"30d","u1"))
	assert len(_run(svc.list_forecasts("t1",m["id"]))) == 2

def test_stats():
	svc = PredictiveAnalyticsService()
	_run(svc.train_model("t1","M","prophet","u1","ds1"))
	stats = _run(svc.get_stats("t1"))
	assert stats["model_count"] == 1

def test_audit_events():
	svc = PredictiveAnalyticsService()
	_run(svc.train_model("t1","M","prophet","u1","ds1"))
	events = _run(svc.get_audit_events("t1"))
	assert len(events) >= 1
