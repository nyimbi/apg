# Sources Consulted

**Date:** 2026-07-10

This file records the evidence consulted for `README.md` and `rationale.md`. Local sources are authoritative for APG current state. External sources were used only to calibrate competitor expectations.

## Required Local Inputs

- `git log --oneline -25` - recent wave history and commit subjects.
- `docs/research/world-class-uplift-2026-07/README.md` - July 2026 uplift mission, wave plan, initial state-of-play notes.
- `docs/research/app-generator-competitive-landscape-2026/README.md` - competitor landscape, differentiation axes, market gaps, and world-class checklist.
- `ls capabilities/` - capability domain inventory requested by the task.

## Additional Local Evidence

- `README.md` - current APG positioning, architecture, generated artifacts, capability domains, generated UI workspaces, CLI list, and repository-level test evidence.
- `docs/research/app-generator-competitive-landscape-2026/sources.md` - prior external-source inventory for the competitive landscape.
- `docs/research/world-class-uplift-2026-07/findings-state-of-play.md` - earlier generated-app audit and test coverage map.
- `docs/research/world-class-uplift-2026-07/rationale.md` - security and wave design rationale.
- `spec/apg.g4` - grammar keywords, entity model, i18n/screen relationship grammar surface, enum/config fields.
- `compiler/code_generator.py` - generated Flask runtime, OpenAPI, i18n, tenancy filter, webhook, CSV, DB, ops, HTTP, and debug UI implementation evidence.
- `compiler/ast_builder.py` - source AST fallback parsing and current mapping of entities, capabilities, workflows, i18n metadata, and enum/entity types.
- `compiler/compiler.py` - compile pipeline, semantic analysis, generated test scaffolding, generated Python parse checks.
- `language_server/server.py` - pygls language server feature registration.
- `language_server/semantic_service.py` - dependency-light language service, fixture audit, completions, diagnostics, document symbols, rename, code actions.
- `tests/fixtures/language_server/catalog.json` - language-server fixture coverage tags and expectations.
- `vscode-extension/package.json` - VS Code extension contributions, commands, keybindings, task definitions, dependencies.
- `vscode-extension/src/extension.ts` - VS Code command implementation and language-server client startup.
- `tests/test_generated_app_hardening.py` - Wave A hardening coverage.
- `tests/test_generated_app_ops.py` - Wave B operations coverage.
- `tests/test_generated_app_http.py` - HTTP efficiency coverage.
- `tests/test_generated_app_a11y.py` - generated app accessibility coverage.
- `tests/test_generated_app_api.py` - OpenAPI, pagination/filtering/sorting, and CSV export coverage.
- `tests/test_generated_app_hardening2.py` - Wave F hardening coverage.
- `tests/test_generated_app_db.py` - Wave I database lifecycle coverage.
- `tests/test_generated_app_webhooks.py` - Wave J webhook coverage.
- `tests/test_generated_app_search.py` - Wave K FTS5 search coverage.
- `tests/test_generated_tests.py` - Wave K generated pytest scaffold coverage.
- `tests/test_generated_app_rbac.py` - Wave L field ACL and row ownership coverage.
- `tests/test_compiler_errors.py` - line/column, Did-you-mean, duplicate entity/field, and warning diagnostics.
- `tests/test_compiled_program_tables.py` - current behavior around `table` vs `entity` semantic model population.
- `tests/test_tooling_audit.py` - aggregate tooling surface audit.
- `tests/test_tooling_ergonomics.py` - schema/refactor/NL plan/studio ergonomic coverage.

## External Competitor References

Official product documentation was preferred for competitor feature calibration. The checks focused on whether each benchmark exposes the relevant primitive directly, not whether its implementation is better than APG's generated-code model.

### Retool

- https://docs.retool.com/apps/guides/forms-inputs/file-inputs - file input guide.
- https://docs.retool.com/apps/reference/components/file-input - file input component reference.
- https://docs.retool.com/queries/guides/files - file upload/storage action guide.
- https://docs.retool.com/workflows - workflows product documentation.
- https://docs.retool.com/workflows/tutorial - scheduled/automation workflow tutorial.
- https://docs.retool.com/org-users/concepts/internationalization - internationalization concept docs.
- https://docs.retool.com/apps/guides/app-management/localization - app localization guide.

Evidence used: file input docs cover file-type, file-count, parsing, validation, file size, and styling controls; Workflows docs cover building, scheduling, and monitoring jobs/alerts/ETL tasks; i18n/localization docs cover org-level translations and app localization.

### Budibase

- https://docs.budibase.com/docs/relationships - relationship data type docs.
- https://docs.budibase.com/docs/attachments - attachment field docs.
- https://docs.budibase.com/docs/attachment - attachment component docs.
- https://docs.budibase.com/docs/s3-file-upload - S3 upload component docs.
- https://docs.budibase.com/docs/automation-steps - automation step introduction.
- https://docs.budibase.com/docs/automation-building-101 - automation building guide.
- https://docs.budibase.com/docs/translations - translations settings docs.

Evidence used: relationships docs cover bidirectional row relationships; attachment docs cover attachment and attachment-list fields; automation docs cover backend logic such as sending email when data changes; translations docs cover user-facing text configuration.

### Wasp

- https://wasp.sh/docs/data-model/entities - entity and data-model docs.
- https://wasp.sh/docs/guides/integrations/file-upload - file upload integration guide.
- https://wasp.sh/docs/advanced/email - email sending docs.
- https://wasp.sh/docs/advanced/jobs - recurring/background jobs docs.
- https://wasp.sh/docs/api/%40wasp.sh/spec/interfaces/Job - job API reference.
- https://wasp.sh/docs/editor-setup - editor setup docs.
- https://github.com/wasp-lang/vscode-wasp - Wasp VS Code extension repository.

Evidence used: entity docs ground Wasp's Prisma-backed data model and relationships; file upload docs cover Multer-based uploads; email docs cover app email sending; jobs docs cover background/recurring work; editor setup docs and the extension repository calibrate APG's IDE gap.

### Appsmith

- https://docs.appsmith.com/reference/widgets/filepicker - Filepicker widget reference.
- https://docs.appsmith.com/connect-data/reference/using-smtp - SMTP datasource reference.
- https://docs.appsmith.com/connect-data/how-to-guides/send-emails-using-the-SMTP-plugin - sending email with SMTP plugin.
- https://docs.appsmith.com/workflows - Appsmith Workflows overview.
- https://docs.appsmith.com/workflows/reference/workflow-triggers - workflow trigger docs, including scheduled trigger.
- https://docs.appsmith.com/workflows/tutorials/create-workflow - basic workflow tutorial.

Evidence used: Filepicker docs establish a first-class upload widget; SMTP docs establish email-sending integration; workflow docs and workflow-trigger docs establish automation, webhook, datasource, and scheduled-trigger expectations.

### Directus

- https://directus.com/docs/guides/data-model/relationships - relationship model guide.
- https://directus.com/docs/guides/connect/relations - relational data guide.
- https://directus.com/docs/guides/files/access - file access guide.
- https://directus.com/docs/guides/files/manage - file management guide.
- https://directus.com/docs/guides/flows - Directus Flows automation guide.
- https://directus.com/docs/guides/content/translations - content translations guide.
- https://directus.com/docs/guides/data-model/fields - field/data-model guide.

Evidence used: relationship docs cover Many-to-One, One-to-Many, Many-to-Many, Many-to-Any, and translations; file docs cover upload/manage/access flows; Flows docs cover event-driven automation; translations docs cover localized content.

### Tooling References

- https://code.visualstudio.com/api/language-extensions/language-server-extension-guide - VS Code language server extension guide.
- https://code.visualstudio.com/api - VS Code extension API documentation.

## External Research Notes

- Official docs were checked directly for the competitor-specific P1/P2 gap calibration in this document.
- Official docs were preferred over vendor comparison blogs for the post Waves A-L gap list.
- Prior `docs/research/app-generator-competitive-landscape-2026/sources.md` remains the broader source inventory for market-level claims outside this narrower gap analysis.
