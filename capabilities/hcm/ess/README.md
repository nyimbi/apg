# Employee Self-Service (hcm_ess)

Capability for employee self-service operations: leave requests, payslip access, personal data management, expense claims, benefits enrolment, training registration, TOIL tracking, statutory reporting, and notification preferences.

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/health | Health check |
| GET | /api/hcm/ess/describe | Capability contract |
| GET | /api/hcm/ess/dashboard | Tenant ESS dashboard |
| GET | /api/hcm/ess/audit-events | Audit trail |

### Leave Requests

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/leave-requests | List leave requests |
| POST | /api/hcm/ess/leave-requests | Create leave request |
| GET | /api/hcm/ess/leave-requests/{id} | Get leave request |
| PATCH | /api/hcm/ess/leave-requests/{id} | Update pending leave request |
| PUT | /api/hcm/ess/leave-requests/{id}/approve | Approve leave |
| PUT | /api/hcm/ess/leave-requests/{id}/reject | Reject leave |
| PUT | /api/hcm/ess/leave-requests/{id}/cancel | Cancel leave |
| DELETE | /api/hcm/ess/leave-requests/{id} | Delete pending leave request |
| GET | /api/hcm/ess/leave-balance/{employee_id} | Get leave balances |
| GET | /api/hcm/ess/leave-conflicts | Check date-range conflicts |

### Leave Accrual

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/hcm/ess/leave-accrual | Accrue leave for one employee |
| POST | /api/hcm/ess/leave-accrual/batch | Batch accrual for many employees |

### Payslips

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/payslips | List payslips |
| GET | /api/hcm/ess/payslips/{id} | Get payslip |
| POST | /api/hcm/ess/payslips | Generate payslip |
| GET | /api/hcm/ess/payslips/ytd | Year-to-date aggregation |

### Expense Claims

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/expense-claims | List expense claims |
| POST | /api/hcm/ess/expense-claims | Create expense claim |
| GET | /api/hcm/ess/expense-claims/{id} | Get expense claim |
| PATCH | /api/hcm/ess/expense-claims/{id} | Update draft claim |
| PUT | /api/hcm/ess/expense-claims/{id}/submit | Submit claim |
| PUT | /api/hcm/ess/expense-claims/{id}/approve | Approve claim |
| DELETE | /api/hcm/ess/expense-claims/{id} | Delete draft claim |
| POST | /api/hcm/ess/expense-claims/bulk-submit | Submit multiple drafts |
| GET | /api/hcm/ess/expense-policy | Get expense policy |
| PUT | /api/hcm/ess/expense-policy | Upsert expense policy |
| POST | /api/hcm/ess/expense-policy/validate | Validate amount against policy |

### Benefits Enrolment

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/benefit-enrolments | List benefit enrolments |
| POST | /api/hcm/ess/benefit-enrolments | Enrol in benefit |
| GET | /api/hcm/ess/benefit-enrolments/{id} | Get enrolment |
| PATCH | /api/hcm/ess/benefit-enrolments/{id} | Update enrolment |
| PUT | /api/hcm/ess/benefit-enrolments/{id}/terminate | Terminate enrolment |

### Training Registration

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/training-registrations | List training |
| POST | /api/hcm/ess/training-registrations | Register for training |
| GET | /api/hcm/ess/training-registrations/{id} | Get registration |
| PATCH | /api/hcm/ess/training-registrations/{id} | Update registration |
| PUT | /api/hcm/ess/training-registrations/{id}/complete | Mark completed |
| DELETE | /api/hcm/ess/training-registrations/{id} | Delete pending |

### Certificates

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/hcm/ess/certificates | Record a certificate |
| GET | /api/hcm/ess/certificates/{employee_id} | List employee certificates |

### TOIL

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/hcm/ess/toil | Record overtime / TOIL |
| GET | /api/hcm/ess/toil/{employee_id} | Get TOIL balance and history |

### Statutory Reporting

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/statutory-report | Generate period statutory report |

### Notifications

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/notification-prefs/{employee_id} | Get notification prefs |
| PUT | /api/hcm/ess/notification-prefs/{employee_id} | Upsert notification prefs |
| POST | /api/hcm/ess/notifications/send | Send ESS notification |

### Personal Data

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/personal-data/{employee_id} | Get personal data |
| PUT | /api/hcm/ess/personal-data/{employee_id} | Upsert personal data |
| PATCH | /api/hcm/ess/personal-data/{employee_id} | Update specific fields |

## Service Methods

### New in v1.1.0

| Method | Description |
|--------|-------------|
| `check_leave_conflicts` | Detect overlapping approved/pending leave for date-range clash prevention |
| `accrue_leave` | Credit earned leave days for one employee for one pay period |
| `run_leave_accrual_batch` | Concurrent batch accrual for all tenant employees |
| `get_payslip_ytd` | Year-to-date gross, net, deductions aggregate |
| `upsert_expense_policy` | Set per-category spend limits for the tenant |
| `validate_expense_against_policy` | Non-raising policy check returning a structured result |
| `bulk_submit_expense_claims` | Submit multiple draft claims with policy enforcement |
| `record_toil` | Log overtime hours and credit TOIL days to leave balance |
| `get_toil_balance` | TOIL running balance and full history |
| `record_certificate` | Persist a professional certificate with expiry and CPD credits |
| `list_employee_certificates` | All certificates held by an employee, filterable by expiry |
| `generate_statutory_report` | PAYE/NSSF/NHIF per-employee rows + grand totals for a pay period |
| `upsert_notification_preferences` | Per-employee channel and event-type subscription settings |
| `send_ess_notification` | Emit a structured notification event for downstream delivery |

---

## World-Class Enhancements (v2.0)

- **I1.** ESS World-Class Improvements
- **I2.** Leave Clash Detection
- **I3.** Leave Accrual Engine
- **I4.** Delegation / Acting-For Leave Handover
- **I5.** Multi-Level Leave Approval Workflow
- **I6.** Payslip Year-to-Date Aggregation
- **I7.** Payslip PDF Generation
- **I8.** Expense Claim Bulk Submit
- **I9.** Expense Policy Engine
- **I10.** Benefit Open-Enrollment Window
- **I11.** Training Completion Certificate Store
- **I12.** Bulk Leave Accrual Run
- **I13.** Document Attachment Tracking
- **I14.** Employee Self-Service Notifications
- **I15.** Compliance / Statutory Reporting

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
