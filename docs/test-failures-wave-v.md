# Test Failures — Wave V Audit

Logged: 2026-07-31. All failures listed here **predate Wave V** (commit `92de4f9b`).
Wave V introduced zero new test failures; `tests/test_ci_config.py` 4/4 pass.

---

## test_compiler_baseline.py — 24 failures

**Root cause:** `self_test["passed"]` returns `False` inside generated `app.py`.
The generated application's self-test health-check reports a validation failure at runtime,
which is a code-generator regression introduced before Wave V.

**Representative file:line:** `tests/test_compiler_baseline.py:409`

```
assert self_test["passed"] is True
AssertionError: assert False is True
```

Additional failures in this file cascade from the same generated-code issue
(HTTP endpoints, record mutations, CLI `compile`/`release`/`package`/`evidence` commands
all depend on a healthy generated `app.py`).

**Do not fix here** — requires compiler/code_generator.py change.

---

## test_compiled_program_tables.py — 5 failures

**Root cause:** Same generated `app.py` regression. Smoke-test compilation of numbered
examples fails when `self_test` returns `passed: False`.

**File:line:** `tests/test_compiled_program_tables.py` (various)

---

## test_compiler_database_ast.py — 3 failures

**Root cause:** Generated validation logic rejects/accepts database references differently
than the test expects — a compiler regression predating Wave V.

---

## test_examples_parseable.py — 3 failures

**Root cause:** `test_numbered_apg_example_outputs_match_current_compiler` — checked-in
`examples/*/output/app.py` files don't match current compiler output (stale snapshots
from a prior wave). `test_numbered_apg_examples_release_evidence_passes` cascades.

**File:line:** `tests/test_examples_parseable.py`

---

## test_generated_app_api.py — 1 failure

**Root cause:** CSV export route returns unexpected content-type. Pre-existing codegen gap.

**File:line:** `tests/test_generated_app_api.py::test_records_csv_export_returns_text_csv_with_header`

---

## test_generated_app_hardening.py — 1 failure

**Root cause:** Production-mode API key enforcement not applied to unconfigured mutations.
Pre-existing codegen gap.

---

## test_generated_ui_assets.py — 1 failure

**Root cause:** External URLs found in generated output (`test_no_external_urls_in_generated_output`).
Pre-existing codegen gap.

---

## test_generated_ui_dashboard.py — 4 failures

**Root cause:** Generated dashboard templates missing expected analytics, detail, form,
and workflow wizard elements. Pre-existing template gaps.

---

## test_generated_ui_i18n.py — 1 failure

**Root cause:** Example 10 language-switcher/locale-cookie not generated. Pre-existing gap.

---

## test_package.py — 1 failure

**Root cause:** `apg` CLI entry-point binary not on `$PATH` — package installed with
`pip install -e .` in development mode but `~/.local/bin` or `$VIRTUAL_ENV/bin` not on
PATH during test subprocess. **Fix:** ensure `pip install -e .[dev]` is run in an active
venv and the venv's `bin/` is on PATH before running this test, or mark it `skipif` when
`shutil.which("apg") is None`.

**File:line:** `tests/test_package.py:25`

---

## test_repository_hygiene.py — 5 failures

**Root cause:** Hardcoded allowlists in the hygiene tests don't include files added by
Wave U (`CHANGELOG.md`, `VERSION`, `Makefile`) and Wave W (`docs/` with "kafka" mentions).

- `test_root_tracked_files_stay_intentional_and_minimal` — allowlist missing CHANGELOG.md etc.
- `test_root_tests_and_docs_stay_in_expected_directories` — CHANGELOG.md flagged as misplaced
- `test_apg_streaming_runtime_stays_bytewax_native` — docs/research README contains "kafka"

**Fix:** Update allowlists in `tests/test_repository_hygiene.py` to reflect Wave U/W additions.

---

## test_tooling_audit.py — 1 failure

**Root cause:** Tooling audit fixture catalog out of sync with current CLI surfaces.
Pre-existing gap.
