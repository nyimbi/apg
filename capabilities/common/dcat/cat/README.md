# Data Catalog (dcat_cat)

Dataset registry, data lineage graph, metadata tagging, glossary, Apache Atlas-compatible API, and ownership tracking.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/dcat/cat/health | Service health |
| GET | /api/dcat/cat/datasets | List datasets |
| POST | /api/dcat/cat/datasets | Register dataset |
| GET | /api/dcat/cat/datasets/{id} | Get dataset |
| PUT | /api/dcat/cat/datasets/{id} | Update dataset |
| DELETE | /api/dcat/cat/datasets/{id} | Delete dataset |
| GET | /api/dcat/cat/datasets/search?q= | Search datasets |
| POST | /api/dcat/cat/lineage | Add lineage edge |
| GET | /api/dcat/cat/lineage | List lineage edges |
| GET | /api/dcat/cat/lineage/{id}/upstream | Upstream lineage |
| GET | /api/dcat/cat/lineage/{id}/downstream | Downstream lineage |
| POST | /api/dcat/cat/glossary | Create term |
| GET | /api/dcat/cat/glossary | List terms |
| GET | /api/dcat/cat/glossary/{id} | Get term |
| DELETE | /api/dcat/cat/glossary/{id} | Delete term |
| POST | /api/dcat/cat/tags | Create tag |
| GET | /api/dcat/cat/tags | List tags |
| GET | /api/dcat/cat/statistics | Catalog statistics |
| GET | /api/dcat/cat/audit | Audit trail |
| GET | /api/dcat/cat/atlas/entity/{id} | Atlas-compatible entity |
| GET | /api/dcat/cat/impact/{id} | Impact analysis |
