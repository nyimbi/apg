# DVRL Works Capability Note

`capabilities/common/dvrl/works` is a DVRL working-artifact packet, not an
independent runtime capability.

The authoritative executable DVRL capability is one directory up:

- `../README.md`
- `../SPECIFICATION.md`
- `../PLAN.md`
- `../capability_contract.py`
- `../service.py`
- `../app.py`
- `../semantic_model.json`
- `../package_manifest.json`
- `../release_report.json`

This folder keeps analysis, remediation, deployment, validation, and delivery
reports that informed the root DVRL lifecycle packet. New generated
applications, APG compiler targets, tests, and composition metadata must use the
root DVRL package rather than treating this folder as a separate capability.
