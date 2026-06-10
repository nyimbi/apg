# Employee Self-Service (hcm_ess)

Capability for employee self-service operations: leave requests, payslip access, personal data management, expense claims, benefits enrolment, and training registration.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/ess/health | Health check |
| GET | /api/hcm/ess/describe | Capability contract |
| GET | /api/hcm/ess/leave-requests | List leave requests |
| GET | /api/hcm/ess/leave-requests/{id} | Get leave request |
| POST | /api/hcm/ess/leave-requests | Create leave request |
| PUT | /api/hcm/ess/leave-requests/{id}/approve | Approve leave |
| PUT | /api/hcm/ess/leave-requests/{id}/reject | Reject leave |
| PUT | /api/hcm/ess/leave-requests/{id}/cancel | Cancel leave |
| DELETE | /api/hcm/ess/leave-requests/{id} | Delete leave request |
| GET | /api/hcm/ess/payslips | List payslips |
| GET | /api/hcm/ess/payslips/{id} | Get payslip |
| POST | /api/hcm/ess/payslips | Generate payslip |
| GET | /api/hcm/ess/expense-claims | List expense claims |
| POST | /api/hcm/ess/expense-claims | Create expense claim |
| PUT | /api/hcm/ess/expense-claims/{id} | Update expense claim |
| DELETE | /api/hcm/ess/expense-claims/{id} | Delete expense claim |
| GET | /api/hcm/ess/benefit-enrolments | List benefit enrolments |
| POST | /api/hcm/ess/benefit-enrolments | Enrol in benefit |
| GET | /api/hcm/ess/training-registrations | List training |
| POST | /api/hcm/ess/training-registrations | Register for training |
| GET | /api/hcm/ess/dashboard | Tenant ESS dashboard |
| GET | /api/hcm/ess/audit-events | Audit trail |
