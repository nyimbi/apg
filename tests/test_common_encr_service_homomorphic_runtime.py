import json
from datetime import datetime, timedelta

import pytest

from capabilities.common.encr.models import HomomorphicCiphertext
from capabilities.common.encr.service import HomomorphicComputationEngine


def _ciphertext(value: float | int, *, tenant_id: str = "tenanthom123") -> HomomorphicCiphertext:
	payload = json.dumps({"value": value}, sort_keys=True).encode("utf-8")
	return HomomorphicCiphertext(
		tenant_id=tenant_id,
		session_id="session-homomorphic",
		ciphertext_data=payload,
		parameters={"encoding": "apg-test-json-value"},
		computation_context="analytics",
		data_type="float",
		data_size=len(payload),
		noise_level=0.02,
		operations_performed=[],
		operation_count=0,
		expires_at=datetime.utcnow() + timedelta(hours=1),
	)


@pytest.mark.asyncio
async def test_homomorphic_compute_adds_numeric_ciphertext_payloads():
	engine = HomomorphicComputationEngine()
	await engine.initialize()

	result = await engine.compute([_ciphertext(10), _ciphertext(32.5)], "add", "ledger")
	payload = json.loads(result.ciphertext_data.decode("utf-8"))

	assert payload == {"input_count": 2, "operation": "add", "result": 42.5}
	assert result.parameters["result_encoding"] == "apg-homomorphic-json-v1"
	assert result.operations_performed == ["add"]
	assert result.operation_count == 1


@pytest.mark.asyncio
async def test_homomorphic_compute_statistics_are_deterministic():
	engine = HomomorphicComputationEngine()
	values = [_ciphertext(2), _ciphertext(4), _ciphertext(10)]

	first = await engine.compute(values, "statistics", "metrics")
	second = await engine.compute(values, "statistics", "metrics")

	assert first.ciphertext_data == second.ciphertext_data
	assert json.loads(first.ciphertext_data.decode("utf-8"))["result"] == {
		"count": 3,
		"max": 10.0,
		"mean": 16.0 / 3.0,
		"min": 2.0,
		"sum": 16.0,
	}


@pytest.mark.asyncio
async def test_homomorphic_compute_rejects_cross_tenant_inputs():
	engine = HomomorphicComputationEngine()

	with pytest.raises(ValueError, match="tenant-isolated"):
		await engine.compute(
			[_ciphertext(1, tenant_id="tenanthom123"), _ciphertext(2, tenant_id="tenanthom456")],
			"add",
			"ledger",
		)
