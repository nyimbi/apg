# Hospitality Analytics (hos_ana)

RevPAR, ADR, occupancy, GOP PAR, segment analysis, pace reporting, and guest satisfaction intelligence.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/ana/health | Health check |
| GET | /api/hospitality/ana/kpi-snapshots | List KPI snapshots |
| POST | /api/hospitality/ana/kpi-snapshots | Record KPI snapshot |
| GET | /api/hospitality/ana/kpi-snapshots/{id} | Get snapshot |
| PUT | /api/hospitality/ana/kpi-snapshots/{id} | Update snapshot |
| DELETE | /api/hospitality/ana/kpi-snapshots/{id} | Delete snapshot |
| GET | /api/hospitality/ana/kpi-summary | Period KPI summary |
| GET | /api/hospitality/ana/segment-reports | List segment reports |
| POST | /api/hospitality/ana/segment-reports | Record segment data |
| GET | /api/hospitality/ana/segment-mix | Segment mix report |
| GET | /api/hospitality/ana/pace-reports | List pace reports |
| POST | /api/hospitality/ana/pace-reports | Record pace data |
| GET | /api/hospitality/ana/pace-comparison | Pace comparison |
| GET | /api/hospitality/ana/satisfaction-surveys | List surveys |
| POST | /api/hospitality/ana/satisfaction-surveys | Record survey |
| GET | /api/hospitality/ana/satisfaction-summary | NPS & satisfaction KPIs |
| POST | /api/hospitality/ana/benchmarks | Record benchmark |
| GET | /api/hospitality/ana/benchmarks | List benchmarks |
| POST | /api/hospitality/ana/competitive-sets | Create comp set |
| POST | /api/hospitality/ana/channel-revenue | Record channel revenue |
| GET | /api/hospitality/ana/channel-mix | Channel mix report |
| GET | /api/hospitality/ana/executive-dashboard | Executive dashboard |
| GET | /api/hospitality/ana/dashboard | Summary dashboard |
