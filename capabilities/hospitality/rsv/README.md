# Reservations & Channel Manager (hos_rsv)

CRS, OTA channel distribution, GDS connectivity, availability sync, and booking engine.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/rsv/health | Health check |
| GET | /api/hospitality/rsv/channels | List channels |
| POST | /api/hospitality/rsv/channels | Create channel |
| GET | /api/hospitality/rsv/channels/{id} | Get channel |
| PUT | /api/hospitality/rsv/channels/{id} | Update channel |
| DELETE | /api/hospitality/rsv/channels/{id} | Deactivate channel |
| GET | /api/hospitality/rsv/bookings | List bookings |
| POST | /api/hospitality/rsv/bookings | Create booking |
| GET | /api/hospitality/rsv/bookings/{id} | Get booking |
| PUT | /api/hospitality/rsv/bookings/{id} | Update booking |
| DELETE | /api/hospitality/rsv/bookings/{id} | Cancel booking |
| GET | /api/hospitality/rsv/availability | Get availability |
| PUT | /api/hospitality/rsv/availability | Set availability |
| PUT | /api/hospitality/rsv/availability/bulk | Bulk set availability |
| POST | /api/hospitality/rsv/gds-connections | Create GDS connection |
| GET | /api/hospitality/rsv/gds-connections | List GDS connections |
| POST | /api/hospitality/rsv/gds-connections/{id}/sync | Sync GDS |
| POST | /api/hospitality/rsv/waitlist | Add to waitlist |
| GET | /api/hospitality/rsv/waitlist | List waitlist |
| GET | /api/hospitality/rsv/channel-performance | Channel analytics |
| GET | /api/hospitality/rsv/dashboard | Dashboard |
