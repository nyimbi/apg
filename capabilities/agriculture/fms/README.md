# Farm Management System (agr_fms)

Parcel registry, input recording, labour scheduling, cost tracking, farm diary.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/fms/health | Health check |
| GET | /api/agriculture/fms/parcels | List parcels |
| POST | /api/agriculture/fms/parcels | Create parcel |
| GET | /api/agriculture/fms/parcels/{id} | Get parcel |
| PUT | /api/agriculture/fms/parcels/{id} | Update parcel |
| DELETE | /api/agriculture/fms/parcels/{id} | Delete parcel |
| GET | /api/agriculture/fms/parcels/{id}/summary | Cost summary |
| GET | /api/agriculture/fms/inputs | List input records |
| POST | /api/agriculture/fms/inputs | Record input usage |
| DELETE | /api/agriculture/fms/inputs/{id} | Delete input |
| GET | /api/agriculture/fms/labour | List labour schedules |
| POST | /api/agriculture/fms/labour | Create schedule |
| PUT | /api/agriculture/fms/labour/{id} | Update schedule |
| DELETE | /api/agriculture/fms/labour/{id} | Delete schedule |
| GET | /api/agriculture/fms/diary | List diary entries |
| POST | /api/agriculture/fms/diary | Create entry |
| PUT | /api/agriculture/fms/diary/{id} | Update entry |
| DELETE | /api/agriculture/fms/diary/{id} | Delete entry |
| GET | /api/agriculture/fms/costs | Cost summary |
| GET | /api/agriculture/fms/audit | Audit log |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Yield Recording and Cost-per-Kg Analysis** [Feature]
- **I2. Crop Season / Campaign Tracking** [Feature]
- **I3. Agrochemical Compliance Ledger (Pre-Harvest Interval Enforcement)** [Compliance]
- **I4. Soil Nutrient Balance Tracking** [AI/ML]
- **I5. Weather Event Correlation in Diary** [Integration]
- **I6. Labour Contractor / Worker Registry** [Feature]
- **I7. Irrigation Water Usage Tracking** [Feature]
- **I8. Parcel GeoJSON Boundary Storage and Area Validation** [Feature]
- **I9. Input Batch / Lot Traceability** [Compliance]
- **I10. Budget vs Actual Variance Reporting** [Feature]
- **I11. Automated Reorder Alerts for Input Inventory** [Feature]
- **I12. Crop Rotation History and Rotation Compliance** [Feature]
- **I13. Task Due-Date Escalation and Overdue Detection** [UX]
- **I14. Multi-Parcel Bulk Operation Broadcasting** [UX]
- **I15. Export-Ready Compliance Report Generation** [Compliance]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
