"""Service tests for bia_rpt Report Builder."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import ReportBuilderService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_create_report():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","Sales Report","tabular","u1","ds1"))
	assert r["state"] == "draft" and r["report_type"] == "tabular"

def test_publish_report():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	p = _run(svc.publish_report("t1",r["id"]))
	assert p["state"] == "published"

def test_run_published_report():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	_run(svc.publish_report("t1",r["id"]))
	run = _run(svc.run_report("t1",r["id"],"pdf"))
	assert run["status"] == "completed" and "output_ref" in run

def test_list_reports_scoped():
	svc = ReportBuilderService()
	_run(svc.create_report("t1","R1","tabular","u1","ds1"))
	_run(svc.create_report("t2","R2","summary","u2","ds2"))
	assert len(_run(svc.list_reports("t1"))) == 1

def test_archive_report():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	a = _run(svc.archive_report("t1",r["id"]))
	assert a["state"] == "archived"

def test_create_schedule():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	_run(svc.publish_report("t1",r["id"]))
	s = _run(svc.create_schedule("t1",r["id"],"daily","u1"))
	assert s["frequency"] == "daily"

def test_create_internal_distribution():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	d = _run(svc.create_distribution("t1",r["id"],"email","user@example.com","u1",is_external=False))
	assert d["approved"] is True

def test_external_distribution_pending():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	try:
		_run(svc.create_distribution("t1",r["id"],"sftp","partner@example.com","u1",is_external=True))
		assert False, "Should have raised"
	except ValueError:
		pass

def test_approve_distribution():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	d = _run(svc.create_distribution("t1",r["id"],"in_app","user1","u1",is_external=False))
	ap = _run(svc.approve_distribution("t1",d["id"],"approver1"))
	assert ap["approved"] is True

def test_run_history():
	svc = ReportBuilderService()
	r = _run(svc.create_report("t1","R","tabular","u1","ds1"))
	_run(svc.publish_report("t1",r["id"]))
	_run(svc.run_report("t1",r["id"],"pdf"))
	_run(svc.run_report("t1",r["id"],"xlsx"))
	runs = _run(svc.list_runs("t1",r["id"]))
	assert len(runs) == 2

def test_stats():
	svc = ReportBuilderService()
	_run(svc.create_report("t1","R","tabular","u1","ds1"))
	assert _run(svc.get_stats("t1"))["report_count"] == 1

def test_audit_events():
	svc = ReportBuilderService()
	_run(svc.create_report("t1","R","tabular","u1","ds1"))
	assert len(_run(svc.get_audit_events("t1"))) >= 1
