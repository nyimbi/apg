"""Service tests for bia_dwh Data Warehouse."""
from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import DataWarehouseService

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

def test_create_schema():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","Sales DW","star","transaction","u1"))
	assert s["schema_type"] == "star" and s["grain"] == "transaction"

def test_list_schemas_scoped():
	svc = DataWarehouseService()
	_run(svc.create_schema("t1","S1","star","t","u1"))
	_run(svc.create_schema("t2","S2","star","t","u2"))
	assert len(_run(svc.list_schemas("t1"))) == 1

def test_register_table():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	t = _run(svc.register_table("t1",s["id"],"fact_sales","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://sales"))
	assert t["table_type"] == "fact"

def test_table_count_incremented():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	_run(svc.register_table("t1",s["id"],"t1","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://x"))
	s2 = _run(svc.get_schema("t1",s["id"]))
	assert s2["table_count"] == 1

def test_create_etl_job():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	tbl = _run(svc.register_table("t1",s["id"],"t1","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://x"))
	j = _run(svc.create_etl_job("t1","Load Sales","src://raw_sales",tbl["id"],"incremental","u1"))
	assert j["state"] == "pending"

def test_run_etl_job():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	tbl = _run(svc.register_table("t1",s["id"],"t1","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://x"))
	j = _run(svc.create_etl_job("t1","J","src://x",tbl["id"],"full_refresh","u1"))
	j2 = _run(svc.run_etl_job("t1",j["id"]))
	assert j2["state"] == "completed"

def test_add_quality_rule():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	tbl = _run(svc.register_table("t1",s["id"],"t1","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://x"))
	r = _run(svc.add_quality_rule("t1",tbl["id"],"ID not null","not_null","u1","id"))
	assert r["rule_type"] == "not_null"

def test_list_quality_rules_by_table():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	tbl = _run(svc.register_table("t1",s["id"],"t1","fact",[{"name":"id","type":"bigint"}],"u1",lineage_ref="src://x"))
	_run(svc.add_quality_rule("t1",tbl["id"],"R1","not_null","u1","id"))
	_run(svc.add_quality_rule("t1",tbl["id"],"R2","unique","u1","id"))
	rules = _run(svc.list_quality_rules("t1",tbl["id"]))
	assert len(rules) == 2

def test_record_lineage():
	svc = DataWarehouseService()
	rec = _run(svc.record_lineage("t1","src_tbl","tgt_tbl","job1","ETL transform"))
	assert rec["source_table_id"] == "src_tbl"

def test_get_lineage_by_table():
	svc = DataWarehouseService()
	_run(svc.record_lineage("t1","src","tgt"))
	_run(svc.record_lineage("t1","other","tgt2"))
	rows = _run(svc.get_lineage("t1","src"))
	assert len(rows) == 1

def test_delete_schema():
	svc = DataWarehouseService()
	s = _run(svc.create_schema("t1","S","star","t","u1"))
	ok = _run(svc.delete_schema("t1",s["id"]))
	assert ok and _run(svc.get_schema("t1",s["id"])) is None

def test_stats():
	svc = DataWarehouseService()
	_run(svc.create_schema("t1","S","star","t","u1"))
	stats = _run(svc.get_stats("t1"))
	assert stats["schema_count"] == 1

def test_audit_events():
	svc = DataWarehouseService()
	_run(svc.create_schema("t1","S","star","t","u1"))
	events = _run(svc.get_audit_events("t1"))
	assert len(events) >= 1
