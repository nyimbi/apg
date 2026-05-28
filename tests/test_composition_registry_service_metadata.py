from pathlib import Path

import pytest

from capabilities.composition.registry.service import CapabilityRegistryService


def _metadata_service() -> CapabilityRegistryService:
	service = object.__new__(CapabilityRegistryService)
	service.tenant_id = "metadata-test"
	return service


def test_capability_metadata_literal_extraction_preserves_string_values():
	service = _metadata_service()

	assert (
		service._extract_string_value('__capability_name__ = "Rules, Workflow"')
		== "Rules, Workflow"
	)
	assert service._extract_list_value(
		'__composition_keywords__ = ["risk, audit", "workflow"]',
		"",
	) == ["risk, audit", "workflow"]


@pytest.mark.asyncio
async def test_capability_metadata_extraction_accepts_multiline_literal_lists(
	tmp_path,
	monkeypatch,
):
	monkeypatch.chdir(tmp_path)
	init_file = Path("capabilities/common/rules/__init__.py")
	init_file.parent.mkdir(parents=True)
	init_file.write_text(
		'''
__capability_code__ = "COMMON_RULES"
__capability_name__ = "Common Rules"
__version__ = "1.2.3"
__description__ = "Executable rule definitions"
__composition_keywords__ = [
	"rules",
	"workflow, approval",
	"erp",
]
''',
		encoding="utf-8",
	)

	metadata = await _metadata_service()._extract_capability_metadata(init_file)

	assert metadata["capability_code"] == "COMMON_RULES"
	assert metadata["capability_name"] == "Common Rules"
	assert metadata["version"] == "1.2.3"
	assert metadata["composition_keywords"] == [
		"rules",
		"workflow, approval",
		"erp",
	]
	assert metadata["module_path"] == "capabilities.common.rules"
	assert metadata["category"] == "common"
	assert metadata["subcategory"] == "rules"
