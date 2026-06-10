# Agri-Marketplace (agr_mkt)

Farmer produce listing, buyer matching, price discovery, escrow, auction management.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/mkt/health | Health check |
| GET | /api/agriculture/mkt/listings | Browse listings |
| POST | /api/agriculture/mkt/listings | Create listing |
| GET | /api/agriculture/mkt/listings/{id} | Get listing |
| PUT | /api/agriculture/mkt/listings/{id} | Update listing |
| DELETE | /api/agriculture/mkt/listings/{id} | Delete listing |
| POST | /api/agriculture/mkt/listings/{id}/publish | Publish listing |
| GET | /api/agriculture/mkt/listings/{id}/matches | Buyer matches |
| GET | /api/agriculture/mkt/bids | List bids |
| POST | /api/agriculture/mkt/bids | Place bid |
| POST | /api/agriculture/mkt/bids/{id}/respond | Accept/reject/counter |
| GET | /api/agriculture/mkt/prices | Price discovery |
| GET | /api/agriculture/mkt/escrows | List escrows |
| POST | /api/agriculture/mkt/escrows | Create escrow |
| POST | /api/agriculture/mkt/escrows/{id}/release | Release funds |
| GET | /api/agriculture/mkt/auctions | List auctions |
| POST | /api/agriculture/mkt/auctions | Create auction |
| POST | /api/agriculture/mkt/auctions/{id}/bid | Bid in auction |
| POST | /api/agriculture/mkt/auctions/{id}/close | Close auction |
| GET | /api/agriculture/mkt/summary | Market summary |
| GET | /api/agriculture/mkt/audit | Audit log |
