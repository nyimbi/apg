# Guest Loyalty Programme (hos_loy)

Points accrual, tier management, redemption, partner rewards, and recognition preferences.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/loy/health | Health check |
| GET | /api/hospitality/loy/members | List members |
| POST | /api/hospitality/loy/members/enroll | Enroll member |
| GET | /api/hospitality/loy/members/{id} | Get member |
| PUT | /api/hospitality/loy/members/{id} | Update member |
| DELETE | /api/hospitality/loy/members/{id} | Deactivate member |
| GET | /api/hospitality/loy/members/{id}/transactions | List transactions |
| POST | /api/hospitality/loy/members/{id}/earn | Earn points |
| POST | /api/hospitality/loy/members/{id}/redeem | Redeem points |
| POST | /api/hospitality/loy/members/{id}/adjust | Adjust points |
| POST | /api/hospitality/loy/members/{id}/tier-upgrade | Force tier upgrade |
| GET | /api/hospitality/loy/members/{id}/preferences | Get preferences |
| PUT | /api/hospitality/loy/members/{id}/preferences | Set preferences |
| GET | /api/hospitality/loy/partners | List partners |
| POST | /api/hospitality/loy/partners | Create partner |
| POST | /api/hospitality/loy/members/{id}/partner-earn | Partner points |
| POST | /api/hospitality/loy/bonus-campaigns | Create campaign |
| GET | /api/hospitality/loy/bonus-campaigns | List campaigns |
| GET | /api/hospitality/loy/tier-distribution | Tier report |
| GET | /api/hospitality/loy/dashboard | Dashboard |
