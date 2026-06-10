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
