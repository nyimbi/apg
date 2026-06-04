"""Service tests for bia_sbi Self-Service BI."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import SelfServiceBIService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_create_workspace():
	svc = SelfServiceBIService()
	w = _run(svc.create_workspace("t1","My Analysis","u1"))
	assert w["name"] == "My Analysis" and w["owner_id"] == "u1"

def test_list_workspaces_scoped():
	svc = SelfServiceBIService()
	_run(svc.create_workspace("t1","W1","u1"))
	_run(svc.create_workspace("t2","W2","u2"))
	assert len(_run(svc.list_workspaces("t1"))) == 1

def test_create_chart():
	svc = SelfServiceBIService()
	w = _run(svc.create_workspace("t1","W","u1"))
	c = _run(svc.create_chart("t1",w["id"],"Revenue","bar","ds1","u1"))
	assert c["chart_type"] == "bar"

def test_chart_added_to_workspace():
	svc = SelfServiceBIService()
	w = _run(svc.create_workspace("t1","W","u1"))
	c = _run(svc.create_chart("t1",w["id"],"C","line","ds1","u1"))
	w2 = _run(svc.get_workspace("t1",w["id"]))
	assert c["id"] in w2["charts"]

def test_delete_chart():
	svc = SelfServiceBIService()
	w = _run(svc.create_workspace("t1","W","u1"))
	c = _run(svc.create_chart("t1",w["id"],"C","pie","ds1","u1"))
	ok = _run(svc.delete_chart("t1",c["id"]))
	assert ok and _run(svc.get_chart("t1",c["id"])) is None

def test_create_catalogue_entry():
	svc = SelfServiceBIService()
	e = _run(svc.create_catalogue_entry("t1","Sales Data","ds1","u1","Main sales table"))
	assert e["state"] == "draft"

def test_approve_catalogue_entry():
	svc = SelfServiceBIService()
	e = _run(svc.create_catalogue_entry("t1","Sales Data","ds1","u1","desc"))
	ap = _run(svc.approve_catalogue_entry("t1",e["id"],"approver1"))
	assert ap["state"] == "published"

def test_create_sandbox():
	svc = SelfServiceBIService()
	sb = _run(svc.create_sandbox("t1","Exploration","u1",["ds1"]))
	assert sb["state"] == "active" and "ds1" in sb["datasource_ids"]

def test_sandbox_limit():
	svc = SelfServiceBIService()
	for i in range(5):
		_run(svc.create_sandbox("t1",f"SB{i}","u1"))
	try:
		_run(svc.create_sandbox("t1","SB6","u1"))
		assert False, "Should raise"
	except ValueError:
		pass

def test_expire_sandbox():
	svc = SelfServiceBIService()
	sb = _run(svc.create_sandbox("t1","SB","u1"))
	exp = _run(svc.expire_sandbox("t1",sb["id"]))
	assert exp["state"] == "expired"

def test_submit_nlq():
	svc = SelfServiceBIService()
	r = _run(svc.submit_nlq("t1","Show total revenue by month","u1"))
	assert "generated_sql" in r and r["confidence"] > 0

def test_nlq_history():
	svc = SelfServiceBIService()
	_run(svc.submit_nlq("t1","Q1","u1"))
	_run(svc.submit_nlq("t1","Q2","u1"))
	history = _run(svc.list_nlq_history("t1"))
	assert len(history) == 2

def test_stats():
	svc = SelfServiceBIService()
	_run(svc.create_workspace("t1","W","u1"))
	stats = _run(svc.get_stats("t1"))
	assert stats["workspace_count"] == 1

def test_audit_events():
	svc = SelfServiceBIService()
	_run(svc.create_workspace("t1","W","u1"))
	assert len(_run(svc.get_audit_events("t1"))) >= 1
