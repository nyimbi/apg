# Portfolio Analytics

**Capability ID**: `ppm_pan` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Portfolio Analytics (pan) delivers executive-grade visibility across the project portfolio: strategic alignment scoring, risk-return matrices, capacity heat maps, performance scorecards, benchmark comparisons, and scenario analysis. All analytics are tenant-scoped, approval-gated for writes, and emitted as events for downstream consumption.

## Installation

```bash
pip install apg-ppm-pan
```

## Provides

- `portfolio_performance_dashboard`
- `strategic_alignment_scoring`
- `risk_return_analysis`
- `capacity_heat_map`
- `portfolio_investment_analysis`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-pan/dashboard` | `ppm_pan:view` | Overview |
| `/ppm-pan/portfolios` | `ppm_pan:portfolios` | Portfolios |
| `/ppm-pan/portfolios/<id>` | `ppm_pan:portfolios` | Portfolios |
| `/ppm-pan/alignment` | `ppm_pan:alignment` | Strategy |
| `/ppm-pan/risk-return` | `ppm_pan:risk` | Risk & Return |
| `/ppm-pan/capacity` | `ppm_pan:capacity` | Capacity |
| `/ppm-pan/performance` | `ppm_pan:performance` | Performance |
| `/ppm-pan/pipeline` | `ppm_pan:pipeline` | Pipeline |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_portfolio()`
- `get_portfolio()`
- `list_portfolios()`
- `portfolio_overview()`
- `strategic_alignment_score()`
- `score_alignment()`
- `list_alignment_scores()`
- `risk_return_analysis()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_pan` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_pan;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PAN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
