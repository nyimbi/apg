"""Regression coverage for v2 migration generated capability templates."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "scripts" / "migrations" / "migration_to_v2.py"
FORBIDDEN_MARKERS = (
	"TODO: Implement specific models",
	"TODO: Implement initialization logic",
	"Model implementation placeholder",
)


def _load_migration_module():
	spec = importlib.util.spec_from_file_location("apg_migration_to_v2", MIGRATION_PATH)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules["apg_migration_to_v2"] = module
	spec.loader.exec_module(module)
	return module


def test_migration_generated_capability_template_is_executable(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	module = _load_migration_module()
	capability_root = tmp_path / "capabilities"
	capability_path = capability_root / "sample_capability"
	migration = module.APGMigrationV2(str(capability_root))

	migration._create_capability_template(
		capability_path,
		{
			"template": "sample_capability",
			"description": "Sample capability",
		},
	)

	for generated_file in ["__init__.py", "models.py", "service.py"]:
		content = (capability_path / generated_file).read_text()
		assert not any(marker in content for marker in FORBIDDEN_MARKERS)
		compile(content, str(capability_path / generated_file), "exec")

	monkeypatch.syspath_prepend(str(capability_root))
	for module_name in ["sample_capability", "sample_capability.models", "sample_capability.service"]:
		sys.modules.pop(module_name, None)

	service_module = importlib.import_module("sample_capability.service")
	service = service_module.SampleCapabilityService()

	async def exercise_service():
		record = await service.create_record(name="Created record")
		return record, await service.get_record(record.id), await service.list_records(), await service.get_info()

	record, fetched, records, info = asyncio.run(exercise_service())

	assert record.name == "Created record"
	assert fetched == record
	assert records == [record]
	assert info["initialized"] is True
	assert info["records"] == 1
