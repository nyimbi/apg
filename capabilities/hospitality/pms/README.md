# Property Management System (hos_pms)

Room inventory, check-in/out, housekeeping, folio management, night audit, and group bookings for hospitality properties.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/pms/health | Service health check |
| GET | /api/hospitality/pms/rooms | List rooms |
| POST | /api/hospitality/pms/rooms | Create room |
| GET | /api/hospitality/pms/rooms/{id} | Get room |
| PUT | /api/hospitality/pms/rooms/{id} | Update room |
| DELETE | /api/hospitality/pms/rooms/{id} | Delete room |
| GET | /api/hospitality/pms/rooms/availability | Check availability |
| GET | /api/hospitality/pms/guests | List guests |
| POST | /api/hospitality/pms/guests | Create guest |
| GET | /api/hospitality/pms/guests/{id} | Get guest |
| PUT | /api/hospitality/pms/guests/{id} | Update guest |
| GET | /api/hospitality/pms/reservations | List reservations |
| POST | /api/hospitality/pms/reservations | Create reservation |
| GET | /api/hospitality/pms/reservations/{id} | Get reservation |
| PUT | /api/hospitality/pms/reservations/{id} | Update reservation |
| DELETE | /api/hospitality/pms/reservations/{id} | Cancel reservation |
| POST | /api/hospitality/pms/reservations/{id}/check-in | Check in guest |
| POST | /api/hospitality/pms/reservations/{id}/check-out | Check out guest |
| GET | /api/hospitality/pms/reservations/{id}/folio | Get folio summary |
| POST | /api/hospitality/pms/reservations/{id}/folio/charges | Add folio charge |
| GET | /api/hospitality/pms/housekeeping | List HK tasks |
| POST | /api/hospitality/pms/housekeeping | Create HK task |
| POST | /api/hospitality/pms/housekeeping/{id}/complete | Complete HK task |
| POST | /api/hospitality/pms/night-audit | Run night audit |
| GET | /api/hospitality/pms/dashboard | Dashboard summary |
| GET | /api/hospitality/pms/audit-events | Audit log |
