"""Import contract tests for standalone legacy CRM packages."""

from __future__ import annotations

import importlib


LEGACY_CRM_IMPORTS = [
	"capabilities.crm._legacy_models",
	"capabilities.crm.for",
	"capabilities.crm.for.models",
	"capabilities.crm.for.views",
	"capabilities.crm.for.blueprint",
	"capabilities.crm.ord",
	"capabilities.crm.ord.models",
	"capabilities.crm.ord.service",
	"capabilities.crm.ord.views",
	"capabilities.crm.ord.blueprint",
	"capabilities.crm.pri",
	"capabilities.crm.pri.models",
	"capabilities.crm.pri.service",
	"capabilities.crm.pri.views",
	"capabilities.crm.pri.blueprint",
	"capabilities.crm.pro",
	"capabilities.crm.pro.models",
	"capabilities.crm.pro.service",
	"capabilities.crm.pro.views",
	"capabilities.crm.pro.blueprint",
	"capabilities.crm.quo",
	"capabilities.crm.quo.models",
	"capabilities.crm.quo.service",
	"capabilities.crm.quo.views",
	"capabilities.crm.quo.blueprint",
]


def test_legacy_crm_subpackages_import_standalone() -> None:
	for module_name in LEGACY_CRM_IMPORTS:
		importlib.import_module(module_name)
