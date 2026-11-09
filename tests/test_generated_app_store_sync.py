"""D-10 prerequisite: RECORD_STORE and the SQLite tables must hold the same rows.

Three gates over the sync invariant that makes predicate pushdown safe:
JSON-file boot hydration, snapshot-restore reconciliation, and the runtime
dual-write pairs. Both sides are probed directly — RECORD_STORE out of the
executed namespace, SQLite through ``_sqlite_select_records``.
"""

from __future__ import annotations

import json

from compiler.compiler import compile_apg_string


SYNC_APP_SOURCE = """
module store_sync_probe version 1.0.0 {}

table Customer {
    name: str;
}
"""


_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)


def _boot_app(monkeypatch, *, data_file=None) -> dict:
	for key in _ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	if data_file is not None:
		monkeypatch.setenv("APG_DATA_FILE", str(data_file))
	result = compile_apg_string(SYNC_APP_SOURCE)
	assert result.success, result.errors
	namespace: dict = {"__file__": "generated_store_sync_app.py"}
	exec(compile(result.generated_files["app.py"], "generated_store_sync_app.py", "exec"), namespace)
	namespace["_flask_app"].config["TESTING"] = True
	return namespace


def _sqlite_rows(namespace, entity: str = "Customer") -> list[dict]:
	return namespace["_sqlite_select_records"](entity, include_deleted=True, tenant_scoped=False)


def _python_rows(namespace, entity: str = "Customer") -> list[dict]:
	return namespace["RECORD_STORE"][entity]


def _id_name_pairs(namespace, entity: str = "Customer") -> set[tuple[str, str]]:
	"""Both stores as (id, name) pairs — the invariant under test."""
	python_side = {(str(record.get("id")), str(record.get("name"))) for record in _python_rows(namespace, entity)}
	sqlite_side = {(str(row["id"]), str(row["name"])) for row in _sqlite_rows(namespace, entity)}
	assert python_side == sqlite_side
	return python_side


def test_json_boot_hydrates_sqlite(tmp_path, monkeypatch):
	"""Gate (a): a record that exists only in the data file lands in SQLite."""
	data_file = tmp_path / "seed.json"
	data_file.write_text(
		json.dumps({"records": {"Customer": [{"id": 1, "name": "seeded"}]}}),
		encoding="utf-8",
	)
	namespace = _boot_app(monkeypatch, data_file=data_file)

	rows = _sqlite_rows(namespace)
	assert [row["name"] for row in rows] == ["seeded"]
	assert {str(row["id"]) for row in rows} == {str(record.get("id")) for record in _python_rows(namespace)}
	assert _id_name_pairs(namespace) == {("1", "seeded")}
	assert namespace["_APG_SQLITE_SYNCED"] is True


def test_snapshot_restore_resyncs_sqlite(monkeypatch):
	"""Gate (b): restore rewrites SQLite to the snapshot and re-arms the flag."""
	namespace = _boot_app(monkeypatch)
	client = namespace["_flask_app"].test_client()
	created = client.post("/records/Customer", json={"name": "seeded"}).get_json()["record"]
	snapshot = namespace["_record_state_snapshot"]()

	# Diverge both stores away from the snapshot: a row too many in each side
	# and a changed value, with no SQLite rollback — restore alone must heal it.
	extra = client.post("/records/Customer", json={"name": "extra"}).get_json()["record"]
	changed = client.put(f"/records/Customer/{created['id']}", json={"name": "changed"})
	assert changed.status_code == 200, changed.get_json()

	namespace["_restore_record_state"](snapshot)

	assert _id_name_pairs(namespace) == {(str(created["id"]), "seeded")}
	sqlite_ids = {str(row["id"]) for row in _sqlite_rows(namespace)}
	assert str(extra["id"]) not in sqlite_ids
	assert namespace["_APG_SQLITE_SYNCED"] is True


def test_dual_write_keeps_both_sides_equal(monkeypatch):
	"""Gate (c): create, update, and soft-delete each leave the stores equal."""
	namespace = _boot_app(monkeypatch)
	client = namespace["_flask_app"].test_client()

	created = client.post("/records/Customer", json={"name": "first"}).get_json()["record"]
	assert _id_name_pairs(namespace) == {(str(created["id"]), "first")}
	assert namespace["_APG_SQLITE_SYNCED"] is True

	updated = client.put(f"/records/Customer/{created['id']}", json={"name": "renamed"})
	assert updated.status_code == 200, updated.get_json()
	assert updated.get_json()["record"]["name"] == "renamed"
	assert _id_name_pairs(namespace) == {(str(created["id"]), "renamed")}

	deleted = client.delete(f"/records/Customer/{created['id']}")
	assert deleted.status_code == 200, deleted.get_json()
	python_deleted = {str(r.get("id")): r.get("deleted_at") for r in _python_rows(namespace)}
	sqlite_deleted = {str(row["id"]): row.get("deleted_at") for row in _sqlite_rows(namespace)}
	assert set(python_deleted) == set(sqlite_deleted) == {str(created["id"])}
	assert python_deleted[str(created["id"])] not in (None, "")
	assert sqlite_deleted[str(created["id"])] not in (None, "")
