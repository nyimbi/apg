"""Service tests for bia_psa Prescriptive Analytics."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import PrescriptiveAnalyticsService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_create_optimisation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","Staffing","linear_programming","minimise","Minimise cost","u1"))
	assert o["optimisation_type"] == "linear_programming" and o["state"] == "draft"

def test_run_optimisation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","O","linear_programming","minimise","desc","u1"))
	r = _run(svc.run_optimisation("t1",o["id"]))
	assert r["state"] == "completed" and r["result"] is not None

def test_archive_optimisation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","O","linear_programming","minimise","desc","u1"))
	a = _run(svc.archive_optimisation("t1",o["id"]))
	assert a["state"] == "archived"

def test_list_optimisations_scoped():
	svc = PrescriptiveAnalyticsService()
	_run(svc.create_optimisation("t1","O1","linear_programming","minimise","d","u1"))
	_run(svc.create_optimisation("t2","O2","genetic_algorithm","maximise","d","u2"))
	assert len(_run(svc.list_optimisations("t1"))) == 1

def test_generate_recommendation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","O","linear_programming","minimise","d","u1"))
	rec = _run(svc.generate_recommendation("t1",o["id"],"Reduce headcount","allocation","Reduce by 5%","u1"))
	assert rec["approval_state"] == "pending"

def test_approve_recommendation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","O","linear_programming","minimise","d","u1"))
	rec = _run(svc.generate_recommendation("t1",o["id"],"R","action","desc","u1"))
	approved = _run(svc.approve_recommendation("t1",rec["id"],"approver1"))
	assert approved["approval_state"] == "approved"

def test_act_on_approved_recommendation():
	svc = PrescriptiveAnalyticsService()
	o = _run(svc.create_optimisation("t1","O","linear_programming","minimise","d","u1"))
	rec = _run(svc.generate_recommendation("t1",o["id"],"R","action","desc","u1"))
	_run(svc.approve_recommendation("t1",rec["id"],"approver1"))
	acted = _run(svc.act_on_recommendation("t1",rec["id"],"actor1"))
	assert acted["acted_at"] is not None

def test_create_whatif():
	svc = PrescriptiveAnalyticsService()
	w = _run(svc.create_whatif("t1","Q4 Scenario","model-123",[{"name":"growth","value":0.1}],"u1"))
	assert w["state"] == "draft"

def test_run_whatif():
	svc = PrescriptiveAnalyticsService()
	w = _run(svc.create_whatif("t1","W","model-123",[{"name":"x","value":1.0}],"u1"))
	r = _run(svc.run_whatif("t1",w["id"]))
	assert r["state"] == "completed" and r["results"]["delta_pct"] == 8.3

def test_record_decision():
	svc = PrescriptiveAnalyticsService()
	d = _run(svc.record_decision("t1","binary","Cost savings analysis complete","u1"))
	assert d["decision_type"] == "binary"

def test_stats():
	svc = PrescriptiveAnalyticsService()
	_run(svc.create_optimisation("t1","O","linear_programming","minimise","d","u1"))
	stats = _run(svc.get_stats("t1"))
	assert stats["optimisation_count"] == 1

def test_audit_events():
	svc = PrescriptiveAnalyticsService()
	_run(svc.create_optimisation("t1","O","linear_programming","minimise","d","u1"))
	events = _run(svc.get_audit_events("t1"))
	assert len(events) >= 1
