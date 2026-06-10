# Spa & Activities Management (hos_spa)

Treatment booking, therapist scheduling, inventory, retail, and membership management for hotel spa operations.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/spa/health | Health check |
| GET | /api/hospitality/spa/treatments | List treatments |
| POST | /api/hospitality/spa/treatments | Create treatment |
| GET | /api/hospitality/spa/treatments/{id} | Get treatment |
| PUT | /api/hospitality/spa/treatments/{id} | Update treatment |
| GET | /api/hospitality/spa/therapists | List therapists |
| POST | /api/hospitality/spa/therapists | Create therapist |
| GET | /api/hospitality/spa/therapists/{id}/schedule | Get schedule |
| GET | /api/hospitality/spa/appointments | List appointments |
| POST | /api/hospitality/spa/appointments | Book appointment |
| GET | /api/hospitality/spa/appointments/{id} | Get appointment |
| PUT | /api/hospitality/spa/appointments/{id} | Update appointment |
| DELETE | /api/hospitality/spa/appointments/{id} | Cancel appointment |
| POST | /api/hospitality/spa/appointments/{id}/complete | Complete & pay |
| GET | /api/hospitality/spa/memberships | List memberships |
| POST | /api/hospitality/spa/memberships | Create membership |
| POST | /api/hospitality/spa/memberships/{id}/renew | Renew membership |
| GET | /api/hospitality/spa/retail | List retail items |
| POST | /api/hospitality/spa/retail | Create retail item |
| POST | /api/hospitality/spa/retail/{id}/sell | Sell retail item |
| GET | /api/hospitality/spa/revenue-report | Daily revenue |
| GET | /api/hospitality/spa/therapist-utilisation | Utilisation |
| GET | /api/hospitality/spa/dashboard | Dashboard |
