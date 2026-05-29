"""Deterministic runtime helpers for quantum computing operations."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


BACKEND_TYPES = ("simulator", "qpu", "hybrid")
PROVIDERS = ("local", "ibm", "azure", "aws", "ionq", "rigetti", "other")
RETRY_POLICIES = ("none", "safe_retry", "provider_retry")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_backend_type(backend_type: str | None) -> str:
	value = (backend_type or "simulator").strip().lower()
	if value not in BACKEND_TYPES:
		raise ValueError(f"unsupported_quantum_backend_type:{value}")
	return value


def normalize_provider(provider: str | None) -> str:
	value = (provider or "local").strip().lower()
	if value in {"local-simulator", "simulator"}:
		value = "local"
	if value not in PROVIDERS:
		raise ValueError(f"unsupported_quantum_provider:{value}")
	return value


def normalize_retry_policy(policy: str | None) -> str:
	value = (policy or "safe_retry").strip().lower()
	if value not in RETRY_POLICIES:
		raise ValueError(f"unsupported_quantum_retry_policy:{value}")
	return value


def normalize_gates(gates: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
	return tuple(str(gate).strip().lower() for gate in gates or () if str(gate).strip())


def estimate_job_cost(backend_type: str, shot_count: int) -> float:
	rate = {"simulator": 0.0001, "hybrid": 0.0025, "qpu": 0.01}[normalize_backend_type(backend_type)]
	return round(max(0, int(shot_count)) * rate, 4)


def deterministic_measurements(job_id: str, shot_count: int, qubits: int) -> dict[str, int]:
	basis_states = max(1, min(4, 2 ** max(1, min(2, int(qubits)))))
	remaining = max(0, int(shot_count))
	counts: dict[str, int] = {}
	for index in range(basis_states):
		state = format(index, f"0{max(1, min(2, int(qubits)))}b")
		if index == basis_states - 1:
			count = remaining
		else:
			seed = f"{job_id}:{index}".encode("utf-8")
			portion = int(sha256(seed).hexdigest()[:4], 16) % (remaining + 1)
			count = min(remaining, portion)
		counts[state] = count
		remaining -= count
	return counts


def result_confidence(measurement_counts: dict[str, int]) -> float:
	total = sum(measurement_counts.values())
	if total <= 0:
		return 0.0
	return round(max(measurement_counts.values()) / total, 4)


def result_summary(measurement_counts: dict[str, int]) -> str:
	if not measurement_counts:
		return "no measurements recorded"
	winning_state = max(measurement_counts, key=measurement_counts.get)
	return f"dominant state {winning_state} with {measurement_counts[winning_state]} shots"


def validate_qubit_capacity(circuit_qubits: int, backend_qubits: int) -> bool:
	return int(circuit_qubits) <= int(backend_qubits)
