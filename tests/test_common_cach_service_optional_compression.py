"""CACH service should run without optional compression backends."""

from __future__ import annotations

import asyncio

from capabilities.common.cach.models import CompressionAlgorithm
from capabilities.common.cach.service import CacheService, CacheServiceConfig, lz4_frame, zstandard


def make_service() -> CacheService:
	return CacheService(
		CacheServiceConfig(
			ai_optimization_enabled=False,
			predictive_prefetching=False,
			audit_enabled=False,
			health_checks_enabled=False,
			metrics_enabled=False,
		)
	)


def test_cache_service_imports_and_selects_available_default_compression() -> None:
	service = make_service()

	default_algorithm = service._default_compression_algorithm()

	if lz4_frame is not None:
		assert default_algorithm == CompressionAlgorithm.LZ4
	else:
		assert default_algorithm == CompressionAlgorithm.GZIP


def test_cache_service_round_trips_when_requested_lz4_backend_is_missing() -> None:
	async def run() -> None:
		service = make_service()
		payload = {"payload": "x" * 500}

		assert await service.set("large", payload, compression=CompressionAlgorithm.LZ4)
		entry = service._cache_store["default:default:large"]
		value = await service.get("large")

		assert value == payload
		if lz4_frame is None:
			assert entry.compression_type == CompressionAlgorithm.NONE
			assert entry.compression_ratio == 1.0
		else:
			assert entry.compression_type == CompressionAlgorithm.LZ4

	asyncio.run(run())


def test_cache_service_explicit_zstd_backend_availability_is_honest() -> None:
	async def run() -> None:
		service = make_service()
		data = b"x" * 500

		compressed, algorithm, ratio = await service._apply_compression(data, CompressionAlgorithm.ZSTD)

		if zstandard is None:
			assert compressed == data
			assert algorithm == CompressionAlgorithm.NONE
			assert ratio == 1.0
		else:
			assert algorithm == CompressionAlgorithm.ZSTD
			assert ratio < 1.0

	asyncio.run(run())
