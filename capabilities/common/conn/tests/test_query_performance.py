"""CONN query performance optimization should execute real query callables."""

from __future__ import annotations

import pytest

from capabilities.common.conn import performance
from capabilities.common.conn.performance import CacheConfig, PerformanceConfig, PerformanceOptimizer


@pytest.fixture()
def fresh_optimizer(monkeypatch: pytest.MonkeyPatch):
	optimizer = PerformanceOptimizer(
		cache_config=CacheConfig(max_size=32, ttl_seconds=60),
		perf_config=PerformanceConfig(max_worker_processes=0),
	)
	monkeypatch.setattr(performance, "global_performance_optimizer", optimizer)

	yield optimizer

	optimizer.task_manager.thread_pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_inline_query_executor_runs_once_then_cache_serves_result(fresh_optimizer: PerformanceOptimizer) -> None:
	calls: list[tuple[str, dict[str, object]]] = []

	async def executor(query: str, params: dict[str, object]):
		calls.append((query, params))
		return [{"customer_id": params["customer_id"], "status": "active"}]

	first = await performance.optimize_query_performance(
		"select * from customers where customer_id = :customer_id",
		{"customer_id": "C-001"},
		executor=executor,
	)

	async def failing_executor(query: str, params: dict[str, object]):
		raise AssertionError("cached query should not call the executor again")

	second = await performance.optimize_query_performance(
		"select * from customers where customer_id = :customer_id",
		{"customer_id": "C-001"},
		executor=failing_executor,
	)

	assert first["executed"] is True
	assert first["cached"] is False
	assert first["execution_strategy"] == "executor:inline"
	assert first["row_count"] == 1
	assert first["result"] == [{"customer_id": "C-001", "status": "active"}]
	assert second["cached"] is True
	assert second["executed"] is True
	assert second["result"] == first["result"]
	assert len(calls) == 1


@pytest.mark.asyncio
async def test_registered_query_executor_uses_stable_param_cache_key(fresh_optimizer: PerformanceOptimizer) -> None:
	calls: list[dict[str, object]] = []

	def executor(query: str, params: dict[str, object]):
		calls.append(params)
		return {"rows": [{"total": params["a"] + params["b"]}], "row_count": 1}

	fresh_optimizer.register_query_executor("erp-reporting", executor)

	first = await performance.optimize_query_performance(
		"select :a + :b as total",
		{"b": 3, "a": 2},
		executor_name="erp-reporting",
	)
	second = await performance.optimize_query_performance(
		"select :a + :b as total",
		{"a": 2, "b": 3},
		executor_name="erp-reporting",
	)

	assert first["executed"] is True
	assert first["cached"] is False
	assert first["row_count"] == 1
	assert first["result"]["rows"] == [{"total": 5}]
	assert second["cached"] is True
	assert second["result"] == first["result"]
	assert calls == [{"b": 3, "a": 2}]


@pytest.mark.asyncio
async def test_query_without_executor_reports_not_executed(fresh_optimizer: PerformanceOptimizer) -> None:
	result = await performance.optimize_query_performance("select 1")

	assert result["executed"] is False
	assert result["cached"] is False
	assert result["status"] == "not_executed"
	assert result["execution_strategy"] == "unavailable"
	assert result["reason"] == "No query executor registered"
	assert fresh_optimizer.cache_manager.get_stats()["memory_cache"]["cache_size"] == 0


@pytest.mark.skipif(not performance.SQLALCHEMY_AVAILABLE, reason="SQLAlchemy is optional")
@pytest.mark.asyncio
async def test_existing_sqlalchemy_pool_can_execute_query(
	fresh_optimizer: PerformanceOptimizer,
	tmp_path,
) -> None:
	db_path = tmp_path / "query-performance.db"
	session_factory = fresh_optimizer.connection_pool_manager.get_pool(
		f"sqlite:///{db_path}",
		pool_name="reporting",
	)
	session = session_factory()
	try:
		session.execute(performance.text("create table customers (id integer primary key, name text)"))
		session.execute(performance.text("insert into customers (id, name) values (1, 'Amina')"))
		session.commit()
	finally:
		session.close()

	result = await performance.optimize_query_performance(
		"select name from customers where id = :customer_id",
		{"customer_id": 1},
		pool_name="reporting",
	)

	assert result["executed"] is True
	assert result["cached"] is False
	assert result["execution_strategy"] == "pool:reporting"
	assert result["row_count"] == 1
	assert result["result"]["rows"] == [{"name": "Amina"}]
