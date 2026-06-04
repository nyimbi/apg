# Changelog — apg-ppm-tex

All notable changes to this package are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-06-02

### Added
- Initial production release of the **Time & Expense Management** capability.
- Capability contract with 10 provided services: timesheet_entry_and_management, expense_claim_workflow, approval_workflow_engine, billable_hour_tracking, reimbursement_processing....
- Deterministic rule engine with governance rules for all operations.
- Bytewax streaming support for real-time event processing.
- Flask Blueprint API with full CRUD routes and structured error responses.
- Pydantic v2 data models with tenant isolation and UUID7 primary keys.
- Comprehensive test suite covering happy path and edge cases.
- Standalone PyPI packaging (`pip install apg-ppm-tex`).

### Dependencies
- pydantic >= 2.0
- uuid6 >= 0.4
- sqlalchemy >= 2.0
- flask >= 3.0
