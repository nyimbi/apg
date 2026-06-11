# Multi-Currency Management

## Overview

Multi-Currency Management (MCY) provides full lifecycle management of currencies, exchange rates, FX revaluation, currency translation, and FX gain/loss reporting for organisations operating across multiple currencies. It enforces positive exchange rates, arms-length approval for manual rates, approval-gated revaluation posting, and tenant-scoped isolation of all currency data.

## Capability ID

`loc_mcy`

## Provides

| Service | Description |
|---------|-------------|
| `currency_configuration` | Configure currencies with ISO codes, decimal precision, and rounding modes |
| `exchange_rate_management` | Record, version, and query exchange rates by currency pair, rate type, and date |
| `fx_revaluation_workflow` | Create, approve, post, and reverse FX revaluation runs per entity and period |
| `currency_translation_workflow` | Create, approve, and post currency translation runs using IFRS/GAAP methods |
| `fx_gain_loss_reporting` | Generate FX gain/loss reports across posted revaluation runs |
| `multi_currency_rounding` | Apply configurable rounding modes per currency |
| `rate_feed_integration` | Ingest rates from central banks, Bloomberg, Reuters, XE, and custom API feeds |
| `currency_exposure_dashboard` | Aggregate currency counts, pending runs, and FX impact summaries |
| `fx_account_registry` | Maintain designated FX gain/loss and translation reserve accounts |

## Requires

| Capability | Reason |
|-----------|--------|
| `auth` | Permission enforcement |
| `audl` | Immutable audit trail |
| `mten` | Tenant context isolation |
| `conf` | Configuration management |
| `ntfy` | Alerts for rate expiry and pending approvals |
| `wflo` | Revaluation and translation approval workflows |
| `moni` | SLA monitoring for rate feeds and pending runs |
| `schd` | Scheduled rate feed imports and periodic revaluation triggers |
| `mqeb` | bytewax event streaming for rate and revaluation lifecycle events |

## Configuration

| Key | Type | Description |
|-----|------|-------------|
| `tenant_id` | string | Tenant identifier |
| `exchange_rates.approval_required_for_manual` | bool | Require approval for manually entered rates |
| `revaluation.approval_required` | bool | Require approval before posting revaluation |
| `translation.approval_required` | bool | Require approval before posting translation |
| `rounding.default_mode` | string | Default rounding mode (`round_half_even`) |
| `currencies.supported_currencies` | list | ISO 4217 codes available for configuration |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/loc-mcy/api/v1/currencies` | GET | List currencies | `loc_mcy:currencies` |
| `/loc-mcy/api/v1/currencies` | POST | Configure currency | `loc_mcy:currencies_write` |
| `/loc-mcy/api/v1/currencies/<id>` | GET | Get currency | `loc_mcy:currencies` |
| `/loc-mcy/api/v1/currencies/<id>` | PUT | Update currency | `loc_mcy:currencies_write` |
| `/loc-mcy/api/v1/exchange-rates` | GET | List exchange rates | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/exchange-rates` | POST | Record exchange rate | `loc_mcy:exchange_rates_write` |
| `/loc-mcy/api/v1/exchange-rates/bulk` | POST | Bulk record exchange rates | `loc_mcy:exchange_rates_write` |
| `/loc-mcy/api/v1/exchange-rates/<id>` | GET | Get exchange rate | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/exchange-rates/history` | GET | Rate history for a pair | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/exchange-rates/stale` | GET | Detect stale rates | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/exchange-rates/matrix` | GET | Rate matrix for a currency set | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/exchange-rates/spread-analysis` | GET | Spread & volatility analysis | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/convert` | GET | Convert currency amount | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/convert/batch` | POST | Batch currency conversion | `loc_mcy:exchange_rates` |
| `/loc-mcy/api/v1/fx-accounts` | GET | List FX accounts | `loc_mcy:fx_accounts` |
| `/loc-mcy/api/v1/fx-accounts` | POST | Register FX account | `loc_mcy:fx_accounts` |
| `/loc-mcy/api/v1/fx-accounts/<id>` | GET | Get FX account | `loc_mcy:fx_accounts` |
| `/loc-mcy/api/v1/revaluation` | GET | List revaluations | `loc_mcy:revaluation` |
| `/loc-mcy/api/v1/revaluation` | POST | Create revaluation | `loc_mcy:revaluation_write` |
| `/loc-mcy/api/v1/revaluation/<id>` | GET | Get revaluation | `loc_mcy:revaluation` |
| `/loc-mcy/api/v1/revaluation/<id>/post` | POST | Post revaluation | `loc_mcy:revaluation_write` |
| `/loc-mcy/api/v1/revaluation/<id>/reverse` | POST | Reverse revaluation | `loc_mcy:revaluation_write` |
| `/loc-mcy/api/v1/translation` | GET | List translations | `loc_mcy:translation` |
| `/loc-mcy/api/v1/translation` | POST | Create translation | `loc_mcy:translation_write` |
| `/loc-mcy/api/v1/translation/<id>` | GET | Get translation | `loc_mcy:translation` |
| `/loc-mcy/api/v1/translation/<id>/post` | POST | Post translation | `loc_mcy:translation_write` |
| `/loc-mcy/api/v1/fx-reporting` | GET | FX gain/loss report | `loc_mcy:fx_reporting` |
| `/loc-mcy/api/v1/fx-reporting/projection` | POST | FX impact projection | `loc_mcy:fx_reporting` |
| `/loc-mcy/api/v1/exposure/consolidated` | GET | Consolidated multi-entity exposure | `loc_mcy:fx_reporting` |
| `/loc-mcy/api/v1/period-close/checklist` | GET | Period-close readiness check | `loc_mcy:revaluation_write` |
| `/loc-mcy/api/v1/agents` | GET | List agents | `loc_mcy:admin` |
| `/loc-mcy/api/v1/agents` | POST | Register agent | `loc_mcy:admin` |
| `/loc-mcy/api/v1/dashboard` | GET | Dashboard summary | `loc_mcy:view` |
| `/loc-mcy/api/v1/audit-events` | GET | Audit log | `loc_mcy:admin` |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `tenant_context_required` | no tenant context | deny |
| `write_requires_policy` | write + no policy | deny |
| `currency_code_supported` | unsupported ISO code | deny |
| `currency_precision_valid` | decimal_places outside 0–6 | deny |
| `rate_value_positive` | rate ≤ 0 | deny |
| `manual_rate_approval_required` | manual source + no approval | deny |
| `rate_backdating_restricted` | backdated + no override | deny |
| `unapproved_revaluation_posting_denied` | post without approval | deny |
| `revaluation_reversal_requires_posted_status` | reverse non-posted | deny |
| `unapproved_translation_posting_denied` | post translation without approval | deny |
| `fx_gain_loss_account_bypass_denied` | bypass FX account | deny |
| `privileged_agent_action_requires_human_approval` | privileged + no approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| `CurrencyConfigResponse` | id, tenant_id, code, name, symbol, decimal_places, rounding_mode, status, is_functional |
| `ExchangeRateResponse` | id, tenant_id, from_currency, to_currency, rate, rate_type, rate_source, effective_date, expiry_date |
| `FxAccountResponse` | id, tenant_id, account_type, account_code, account_name, currency |
| `RevaluationResponse` | id, tenant_id, entity_id, period_start, period_end, revaluation_method, status, fx_gain_amount, fx_loss_amount |
| `CurrencyTranslationResponse` | id, tenant_id, entity_id, source_currency, target_currency, translation_method, status |
| `FxGainLossReport` | tenant_id, period_start, period_end, total_realised_gain, total_realised_loss, net_fx_impact |
| `McyAgentResponse` | id, tenant_id, name, runtime, role, scope |

## Streaming Events

| Event | Trigger |
|-------|---------|
| `currency_configured` | New currency configured |
| `exchange_rate_recorded` | Rate recorded |
| `bulk_rates_uploaded` | Bulk rate upload batch completed |
| `revaluation_created` | Revaluation run created |
| `revaluation_approved` | Revaluation approved |
| `revaluation_posted` | Revaluation posted to ledger |
| `revaluation_reversed` | Revaluation reversed |
| `translation_created` | Translation run created |
| `translation_posted` | Translation posted |
| `fx_gain_loss_calculated` | FX report generated |
| `period_close_checked` | Period-close checklist executed |
| `agent_registered` | Agent registered |

## Edge Cases Handled

- Same-currency conversion returns amount unchanged with rate=1.0 without requiring a rate record
- Inverse rate lookup: if KES→USD rate exists but USD→KES is queried, the inverse is calculated automatically
- Manual exchange rates require an approval reference regardless of backdating status
- Backdated rates require an explicit `backdating_override` field to prevent accidental historical corrections
- Revaluation reversal is only permitted on `posted` runs — reversing a `draft` or `approved` run is blocked
- FX gain/loss accounts must exist in the tenant's FX account registry before a revaluation can be created
- Currency code is normalised to uppercase ISO 4217 regardless of input casing

## Composability Notes

- `mco` (Multi-Country Operations) feeds functional currency assignments per entity into MCY
- `fin` general ledger consumes MCY-posted revaluation and translation journal entries
- `grc` uses MCY's FX exposure data for treasury risk reporting
- MCY emits all lifecycle events to `apg.loc.mcy.lifecycle` for downstream bytewax consumers
