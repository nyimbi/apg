# SACCO Dividend & Distribution (fintech_sacco_div)

Annual surplus calculation, dividend declaration, rebate computation, member distributions, and tax withholding.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/div/health | Service health |
| GET | /api/fintech/sacco/div/years | List financial years |
| POST | /api/fintech/sacco/div/years | Create financial year |
| GET | /api/fintech/sacco/div/years/{id} | Get year |
| PUT | /api/fintech/sacco/div/years/{id} | Update year |
| DELETE | /api/fintech/sacco/div/years/{id} | Cancel year |
| POST | /api/fintech/sacco/div/years/{id}/close | Close year |
| POST | /api/fintech/sacco/div/years/{id}/allocate | Allocate surplus |
| GET | /api/fintech/sacco/div/years/{id}/allocations | List allocations |
| GET | /api/fintech/sacco/div/years/{id}/report | Annual report |
| GET | /api/fintech/sacco/div/declarations | List declarations |
| POST | /api/fintech/sacco/div/declarations | Declare dividend |
| GET | /api/fintech/sacco/div/declarations/{id} | Get declaration |
| PUT | /api/fintech/sacco/div/declarations/{id} | Update declaration |
| POST | /api/fintech/sacco/div/declarations/{id}/reverse | Reverse declaration |
| GET | /api/fintech/sacco/div/declarations/{id}/summary | Declaration summary |
| POST | /api/fintech/sacco/div/declarations/{id}/pay-all | Run payment batch |
| GET | /api/fintech/sacco/div/distributions | List distributions |
| POST | /api/fintech/sacco/div/distributions/compute | Compute for member |
| POST | /api/fintech/sacco/div/distributions/bulk-compute | Bulk compute |
| GET | /api/fintech/sacco/div/distributions/{id} | Get distribution |
| POST | /api/fintech/sacco/div/distributions/{id}/pay | Pay distribution |
| POST | /api/fintech/sacco/div/distributions/{id}/reverse | Reverse distribution |
| POST | /api/fintech/sacco/div/wht | File WHT return |
| GET | /api/fintech/sacco/div/wht | List WHT records |
| GET | /api/fintech/sacco/div/members/{id}/history | Member dividend history |
| GET | /api/fintech/sacco/div/audit | Audit events |
