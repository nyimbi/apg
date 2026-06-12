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

## World-Class Enhancements (v2.0)

**I1. Dynamic Price Intelligence** — EWMA + 7-day seasonal index; `get_price_forecast(product_type, region, horizon_days)` returns confidence intervals [AI/ML]

**I2. Reputation & Trust Score Engine** — Two-sided 0–100 score with Bronze/Silver/Gold/Platinum tiers recomputed on every escrow settlement [Feature]

**I3. Geospatial Radius Search & Logistics Cost** — Haversine-based `search_nearby_listings` + per-km rate table cost estimation [Feature]

**I4. Bulk Spot-Contract Templates with Auto-Expiry** — Multi-delivery schedule contracts; `expire_stale_contracts()` emits `contract.expired` events [Feature]

**I5. Photo-Based Quality Grading** — Pluggable `QualityGraderAdapter` (rules-based default, swappable for vision-LLM) with auditable grade provenance [AI/ML]

**I6. Mobile Money Escrow with Partial Release** — M-Pesa STK-push funding; `release_escrow_partial(escrow_id, fraction, reason)` with tranche tracking [Integration]

**I7. Seasonal Supply Forecasting** — Farmer planting-intent aggregation → projected supply timeline with uncertainty bands per region/crop [AI/ML]

**I8. Automated Dispute Resolution** — 3-stage workflow (self-service → mediator → arbitration) with evidence chain and atomic escrow enforcement [Feature]

**I9. Auction Bid Sniping Protection** — Auto-extends `end_at` by N minutes on late bids; capped at `max_extensions` to prevent abuse [Feature]

**I10. Notification & Price Alert Engine** — Threshold-based `create_price_alert`; `evaluate_price_alerts` emits events for SMS/push delivery [UX]

**I11. Lot Aggregation for Smallholder Pooling** — Cooperative pooled listings with pro-rata proceeds split on escrow release [Feature]

**I12. Carbon Credit & Sustainability Certification** — `attach_sustainability_cert` with validity tracking; `sustainability_tier` label used in buyer matching weight [Compliance]

**I13. Demand Signal Broadcasting (Reverse RFQ)** — Buyers post demand specs; `match_demand_signal` scores farmer responses by price proximity and quality fit [Feature]

**I14. Transaction Analytics Dashboard API** — Pre-aggregated KPIs: GMV, conversion rates, escrow cycle time; supports ISO period + daily/weekly granularity [Feature]

**I15. Fraud Detection via Behavioral Anomaly Scoring** — `score_listing_fraud_risk` and `score_bid_fraud_risk` return 0–100 risk score + flag list [Security]

## New Methods

Three high-impact async methods from the v2.0 service:

### `get_price_forecast` — Forward pricing intelligence

```python
svc = AgrMarketplaceService(tenant_id="ke-rift")
forecast = await svc.get_price_forecast(
    product_type="maize",
    region="rift_valley",
    horizon_days=30,
)
# {"product_type": "maize", "region": "rift_valley",
#  "forecast": [{"date": "2026-06-15", "price_per_kg": 44.2,
#                "lower_ci": 41.8, "upper_ci": 46.6}, ...]}
```

Closes the information asymmetry that depresses farm-gate prices 20–40% vs. trader benchmarks.

### `place_auction_bid` — Anti-sniping auction bidding

```python
result = await svc.place_auction_bid(
    auction_id="auc_01j...",
    bidder_id="buyer_42",
    amount=52_000.0,
)
# If bid lands within extension_window_seconds of end_at,
# auction.end_at is extended and auction.extended event emitted.
# {"id": "auc_01j...", "current_bid": 52000.0,
#  "bid_count": 7, "end_at": "2026-06-12T15:35:00Z", ...}
```

Increments must exceed `current_bid + increment`; raises `ValueError("bid_too_low:min=...")` otherwise.

### `release_escrow_partial` — Tranche-based escrow release

```python
# 50% on delivery confirmation
await svc.release_escrow_partial(
    escrow_id="esc_01j...",
    fraction=0.5,
    reason="delivery_confirmed",
)
# Full release auto-triggers when cumulative fraction reaches 1.0.
# Emits escrow.partial_released; final tranche emits escrow.released.
```

Maps to how processors pay smallholders in practice; eliminates need for trust intermediaries.
