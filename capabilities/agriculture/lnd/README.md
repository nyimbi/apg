# Land Management (agr_lnd)

Parcel cadastre, tenure registry, GPS boundary capture, title issuance, land transfer.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/lnd/health | Health check |
| GET | /api/agriculture/lnd/parcels | List parcels |
| POST | /api/agriculture/lnd/parcels | Register parcel |
| GET | /api/agriculture/lnd/parcels/{id} | Get parcel |
| PUT | /api/agriculture/lnd/parcels/{id} | Update parcel |
| DELETE | /api/agriculture/lnd/parcels/{id} | Delete parcel |
| GET | /api/agriculture/lnd/owners/{id}/holdings | Owner holdings |
| GET | /api/agriculture/lnd/boundaries | List boundaries |
| POST | /api/agriculture/lnd/boundaries | Capture GPS boundary |
| DELETE | /api/agriculture/lnd/boundaries/{id} | Delete boundary |
| GET | /api/agriculture/lnd/titles | List titles |
| POST | /api/agriculture/lnd/titles | Issue title |
| POST | /api/agriculture/lnd/titles/{id}/invalidate | Invalidate title |
| DELETE | /api/agriculture/lnd/titles/{id} | Delete title |
| GET | /api/agriculture/lnd/transfers | List transfers |
| POST | /api/agriculture/lnd/transfers | Initiate transfer |
| PUT | /api/agriculture/lnd/transfers/{id} | Update transfer |
| DELETE | /api/agriculture/lnd/transfers/{id} | Delete transfer |
| GET | /api/agriculture/lnd/registry-summary | Registry stats |
| GET | /api/agriculture/lnd/audit | Audit log |
