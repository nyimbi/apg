# Events & Venue Management (hos_evn)

Event booking, venue configuration, catering BEO, AV requirements, billing, and contract management.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/evn/health | Health check |
| GET | /api/hospitality/evn/venues | List venues |
| POST | /api/hospitality/evn/venues | Create venue |
| GET | /api/hospitality/evn/venues/{id} | Get venue |
| PUT | /api/hospitality/evn/venues/{id} | Update venue |
| DELETE | /api/hospitality/evn/venues/{id} | Deactivate venue |
| GET | /api/hospitality/evn/event-bookings | List event bookings |
| POST | /api/hospitality/evn/event-bookings | Create booking |
| GET | /api/hospitality/evn/event-bookings/{id} | Get booking |
| PUT | /api/hospitality/evn/event-bookings/{id} | Update booking |
| POST | /api/hospitality/evn/event-bookings/{id}/confirm | Confirm booking |
| DELETE | /api/hospitality/evn/event-bookings/{id} | Cancel booking |
| POST | /api/hospitality/evn/beos | Generate BEO |
| GET | /api/hospitality/evn/beos | List BEOs |
| POST | /api/hospitality/evn/beos/{id}/finalise | Finalise BEO |
| POST | /api/hospitality/evn/contracts | Issue contract |
| POST | /api/hospitality/evn/contracts/{id}/sign | Sign contract |
| POST | /api/hospitality/evn/event-bookings/{id}/payments | Record payment |
| GET | /api/hospitality/evn/utilisation-report | Venue utilisation |
| GET | /api/hospitality/evn/dashboard | Dashboard |
