# SACCO Member Registry (fintech_sacco_mem)

Member onboarding, KYC, share capital management, guarantor relationships, and exit processing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/mem/health | Service health |
| GET | /api/fintech/sacco/mem/members | List members |
| POST | /api/fintech/sacco/mem/members | Onboard new member |
| GET | /api/fintech/sacco/mem/members/{id} | Get member |
| PUT | /api/fintech/sacco/mem/members/{id} | Update member |
| DELETE | /api/fintech/sacco/mem/members/{id} | Exit member |
| GET | /api/fintech/sacco/mem/members/search?q= | Search members |
| GET | /api/fintech/sacco/mem/kyc | List KYC records |
| POST | /api/fintech/sacco/mem/kyc | Submit KYC |
| POST | /api/fintech/sacco/mem/kyc/{id}/approve | Approve KYC |
| POST | /api/fintech/sacco/mem/kyc/{id}/reject | Reject KYC |
| POST | /api/fintech/sacco/mem/shares/purchase | Purchase shares |
| POST | /api/fintech/sacco/mem/shares/transfer | Transfer shares |
| POST | /api/fintech/sacco/mem/exits | Initiate exit |
| POST | /api/fintech/sacco/mem/exits/{id}/complete | Complete exit |
| GET | /api/fintech/sacco/mem/summary | Membership summary |
| GET | /api/fintech/sacco/mem/audit | Audit events |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Risk-Tiered Member Scoring** [Enhancement]
- **I2. Guarantor Exposure Limit Enforcement** [Enhancement]
- **I3. Member Dividend Calculation Engine** [Enhancement]
- **I4. KYC Document Expiry Tracking and Re-verification Alerts** [Enhancement]
- **I5. Beneficiary / Next-of-Kin Share Inheritance Workflow** [Enhancement]
- **I6. Bulk KYC Import via CSV/JSON Batch** [Enhancement]
- **I7. Share Withdrawal / Partial Redemption** [Enhancement]
- **I8. Member Financial Health Dashboard Aggregation** [Enhancement]
- **I9. Member Merge (Duplicate Resolution)** [Enhancement]
- **I10. Configurable Member Tier / Segment Classification** [Enhancement]
- **I11. Dormancy Detection and Reactivation Workflow** [Enhancement]
- **I12. Employer-Linked Payroll Deduction Tracking** [Enhancement]
- **I13. Real-Time Duplicate National ID Detection Across Tenants** [Enhancement]
- **I14. Member Communication Event Log** [Enhancement]
- **I15. Share Capital Minimum Adequacy Check Before Guarantor Assignment** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
