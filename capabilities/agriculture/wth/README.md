# Weather & Climate Analytics (agr_wth)

Forecast integration, alert thresholds, historical patterns, climate risk assessment.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/wth/health | Health check |
| GET | /api/agriculture/wth/forecasts | List forecasts |
| POST | /api/agriculture/wth/forecasts | Ingest forecast |
| GET | /api/agriculture/wth/forecasts/latest | Latest forecast |
| DELETE | /api/agriculture/wth/forecasts/{id} | Delete forecast |
| GET | /api/agriculture/wth/thresholds | List thresholds |
| POST | /api/agriculture/wth/thresholds | Create threshold |
| PUT | /api/agriculture/wth/thresholds/{id} | Update threshold |
| DELETE | /api/agriculture/wth/thresholds/{id} | Delete threshold |
| GET | /api/agriculture/wth/alerts | List alerts |
| POST | /api/agriculture/wth/alerts/{id}/acknowledge | Acknowledge alert |
| GET | /api/agriculture/wth/history | Historical patterns |
| POST | /api/agriculture/wth/history | Add historical record |
| GET | /api/agriculture/wth/history/normals | Monthly normals |
| POST | /api/agriculture/wth/risk-assessments | Compute risk |
| GET | /api/agriculture/wth/risk-assessments | List assessments |
| GET | /api/agriculture/wth/audit | Audit log |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Evapotranspiration (ET₀) Computation** [Feature]
- **I2. Growing Degree Days (GDD) Accumulation Tracking** [Feature]
- **I3. Probabilistic Forecast Ensemble Support** [Feature]
- **I4. Satellite-Derived Vegetation Stress Index (NDVI / VCI)** [Integration]
- **I5. Anomaly Detection on Historical Baselines (Z-score)** [AI/ML]
- **I6. Crop-Specific Heat-Unit Windows (Phenological Calendar)** [Feature]
- **I7. Multi-Source Forecast Consensus Scoring** [AI/ML]
- **I8. Seasonal Outlook Integration (3-6 Month Probabilistic)** [Integration]
- **I9. Weather-Triggered Advisory Generation** [UX]
- **I10. Microclimate Zone Interpolation** [Feature]
- **I11. Carbon Credit Weather Verification** [Compliance]
- **I12. Weather Index Insurance Parametric Trigger Export** [Compliance]
- **I13. Forecast Accuracy Backtesting** [AI/ML]
- **I14. Real-Time Push Notification Dispatch** [UX]
- **I15. Water Stress Index (WSI) Time-Series** [Feature]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
