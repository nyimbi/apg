"""Executable persistence regressions for CKM WFA visual designer."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_DESIGNER_PATH = REPO_ROOT / "capabilities" / "ckm" / "wfa" / "visual_designer.py"


def test_visual_designer_process_save_uses_persistence_boundary():
	source = VISUAL_DESIGNER_PATH.read_text(encoding="utf-8")

	assert "For now, simulate successful save" not in source
	assert "In production, save to database via process service" not in source
	assert "def __init__(self, process_service: Any | None = None):" in source
	assert "self.process_service = process_service" in source
	assert "self.process_definitions: Dict[Tuple[str, str], WBPMProcessDefinition] = {}" in source
	assert "save_result = await self._store_process_definition(process_definition, process_data, context)" in source
	assert "process_id = save_result.data.get(\"process_id\", process_definition.id)" in source
	assert "self.process_diagrams[self._process_storage_key(context.tenant_id, process_id)] = copy.deepcopy(session.process_diagram)" in source


def test_visual_designer_process_load_uses_saved_or_service_definition():
	source = VISUAL_DESIGNER_PATH.read_text(encoding="utf-8")

	assert "For now, create a sample diagram" not in source
	assert "process_definition = await self._load_process_definition(process_id, context)" in source
	assert "diagram_name=process_definition.process_name" in source
	assert "\"bpmn_xml\": process_definition.bpmn_xml" in source
	assert "async def _store_process_definition(" in source
	assert "async def _load_process_definition(" in source
	assert "create_process = getattr(self.process_service, \"create_process\", None)" in source
	assert "get_process = getattr(self.process_service, \"get_process\", None)" in source
