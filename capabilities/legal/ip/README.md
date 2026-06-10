# leg_ip — Intellectual Property Registry

Patent, trademark, copyright portfolio management, renewal deadlines, licensing, royalties.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ip/health | Health check |
| GET | /api/legal/ip/assets | List IP assets |
| GET | /api/legal/ip/assets/{id} | Get asset |
| POST | /api/legal/ip/assets | Register asset |
| PUT | /api/legal/ip/assets/{id} | Update asset |
| DELETE | /api/legal/ip/assets/{id} | Abandon asset |
| POST | /api/legal/ip/assets/{id}/register | Record registration |
| GET | /api/legal/ip/renewals | List renewals |
| POST | /api/legal/ip/renewals | File renewal |
| POST | /api/legal/ip/renewals/{id}/confirm | Confirm renewal |
| GET | /api/legal/ip/licenses | List licenses |
| POST | /api/legal/ip/licenses | Grant license |
| DELETE | /api/legal/ip/licenses/{id} | Terminate license |
| GET | /api/legal/ip/royalties | List royalties |
| POST | /api/legal/ip/royalties | Record royalty |
| POST | /api/legal/ip/royalties/{id}/pay | Pay royalty |
| GET | /api/legal/ip/expiring | Expiring assets |
| GET | /api/legal/ip/portfolio | Portfolio summary |
| GET | /api/legal/ip/audit | Audit events |

## Service Class

`IntellectualPropertyService` — full IP lifecycle from application to expiry, with exclusive license conflict detection, royalty calculation, and auto renewal-due-date computation.
