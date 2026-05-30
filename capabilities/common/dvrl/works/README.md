# DVRL Works

This directory contains historical and operational work artifacts for the APG
Data Virtualization capability. It is retained so contributors can understand
the analysis, remediation notes, validation reports, launch notes, and Singer
integration evidence that shaped the root DVRL capability.

The executable capability is `capabilities/common/dvrl`. Use the parent
directory for runtime imports, APG composition, package inspection, publish
plans, implementation audits, and tests.

## Contents

- `CODE_ANALYSIS_REPORT.md`: earlier analysis of implementation risks and
  incomplete paths.
- `REMEDIATION_PLAN.md`: remediation sequence used while hardening DVRL.
- `DEPLOYMENT_VERIFICATION_CHECKLIST.md`: deployment-readiness checklist.
- `reports/`: delivery, validation, launch, and integration reports.
- `cap_spec.md`: source-of-truth pointer to the parent DVRL package.

## How To Use This Folder

Use these files as supporting evidence when changing DVRL, especially when a
change touches connectors, Singer integration, deployment readiness, or
operational reporting.

Do not add runtime imports, generated app entrypoints, semantic metadata,
package manifests, or test entrypoints here. Those belong in
`capabilities/common/dvrl`.

## Maintenance Rules

- Keep this folder dependency-free.
- Keep new notes dated and scoped to DVRL evidence.
- Link implementation decisions back to the root DVRL `SPECIFICATION.md` and
  `PLAN.md`.
- When a report becomes authoritative runtime behavior, move that behavior into
  the root contract, service, API, view models, app metadata, and tests.
