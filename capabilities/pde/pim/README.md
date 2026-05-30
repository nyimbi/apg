# Product Information Management

Product Information Management is the APG capability packet for catalogs, product records, attributes, variants, content, media assets, compliance, channel listings, publication, data quality, change control, and PIM-focused AI agents.

The packet is dependency-light at import time. Media systems, commerce channels, ERP item masters, translation memory, taxonomy services, workflow engines, and notification providers attach through APG composition adapters.

## What It Provides

- `product_catalog_lifecycle` and `product_record_lifecycle`.
- `product_attribute_lifecycle` and `product_variant_lifecycle`.
- `product_content_lifecycle` and `product_asset_lifecycle`.
- `product_compliance_lifecycle`.
- `product_channel_listing_lifecycle` and `product_publish_workflow`.
- `product_data_quality_workflow` and `product_change_workflow`.
- `pim_agents` for Codex, Claude Code, OpenCode, and Pi review agents.

## Example

```python
from capabilities.pde.pim.service import ProductInformationLifecycleService

service = ProductInformationLifecycleService()
catalog = service.create_catalog("cat-1", "tenant-a", "MAIN", "Main Catalog", "owner-1")
product = service.create_product("prod-1", "tenant-a", catalog["id"], "SKU-1", "Solar Charger", "physical", "owner-1")
attribute = service.define_attribute("attr-1", "tenant-a", "description", "Description", "rich_text", "owner-1")
service.set_attribute_value("val-1", "tenant-a", product["id"], attribute["id"], "Portable charger", "en")
content = service.enrich_content("content-1", "tenant-a", product["id"], "en", "Solar Charger", "Portable charger", True, "reviewer-1")
channel = service.create_channel_listing("listing-1", "tenant-a", product["id"], "web", "web-sku-1", "approver-1")
service.publish_product("pub-1", "tenant-a", product["id"], content["id"], channel["id"], "approver-2")
```

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/pde/pim/__init__.py capabilities/pde/pim/capability_contract.py capabilities/pde/pim/service.py capabilities/pde/pim/api.py capabilities/pde/pim/views.py capabilities/pde/pim/app.py capabilities/pde/pim/tests/conftest.py capabilities/pde/pim/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/pde/pim/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/pde/pim --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/pde/pim --json
```
