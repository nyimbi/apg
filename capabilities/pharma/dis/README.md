# Pharmaceutical Distribution

## Overview
Manages pharmaceutical distribution operations including cold chain monitoring, product serialisation and verification, wholesale distribution authorisations, product recalls, GDP compliance, and import/export shipment tracking. Enforces WDA requirements, temperature monitoring, serialisation verification, and recall timeline obligations at every distribution boundary.

## Capability ID
`pharma_dis`

## Provides
- wholesale_distribution_workflow: WDA-gated wholesale dispatch with scope enforcement
- cold_chain_management_workflow: Temperature monitoring with excursion detection and escalation
- serialisation_verification_workflow: GS1/DSCSA/FMD serial number verification and aggregation
- recall_management_workflow: Class I/II/III recall initiation with regulatory notification and effectiveness check
- gdp_compliance_workflow: GDP self-inspection, deviation management, and corrective action
- wda_management_workflow: WDA registration, grant, renewal, and expiry alert management
- shipment_tracking_workflow: Multi-modal shipment lifecycle from planned to delivered
- temperature_excursion_workflow: Excursion severity assessment, impact assessment, and disposition
- import_export_workflow: Import permit verification and cross-border documentation
- distribution_audit_workflow: GDP audit trail and regulatory compliance evidence

## Requires
| Capability | Reason |
|------------|--------|
| auth | Access control for warehouse and distribution roles |
| audl | GDP-compliant audit trail |
| mten | Distributor-level data isolation |
| conf | Cold chain threshold and recall timeline configuration |
| ntfy | WDA expiry, excursion, and recall notifications |
| wflo | Recall approval and WDA renewal workflow |
| comp | GDP and FMD compliance enforcement |
| moni | Real-time temperature and shipment monitoring |
| mqeb | Event streaming for cold chain and recall events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| recalls.timeline_hours.class_i | Class I recall notification deadline | 24 |
| recalls.timeline_hours.class_ii | Class II recall notification deadline | 72 |
| cold_chain.temperature_monitoring_required | Mandatory for cold-chain products | true |
| wda.renewal_alert_days | Days before WDA expiry for renewal alert | 90 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-dis/api/v1/shipments | POST | Create shipment | pharma_dis:shipments |
| /pharma-dis/api/v1/shipments/<id>/dispatch | POST | Dispatch shipment | pharma_dis:shipments |
| /pharma-dis/api/v1/shipments/<id>/integrity | GET | Supply chain integrity check | pharma_dis:shipments |
| /pharma-dis/api/v1/cold-chain/excursions | POST | Report temperature excursion | pharma_dis:cold_chain |
| /pharma-dis/api/v1/cold-chain/telemetry | POST | Ingest IoT temperature telemetry | pharma_dis:cold_chain |
| /pharma-dis/api/v1/cold-chain/mkt | POST | Calculate MKT for a shipment | pharma_dis:cold_chain |
| /pharma-dis/api/v1/serialisation/verify | POST | Verify serial number | pharma_dis:serialisation |
| /pharma-dis/api/v1/serialisation/bulk | POST | Bulk-serialise product units | pharma_dis:serialisation |
| /pharma-dis/api/v1/serialisation/hierarchy/<sscc> | GET | Validate GS1 aggregation hierarchy | pharma_dis:serialisation |
| /pharma-dis/api/v1/recalls | POST | Initiate recall | pharma_dis:recalls |
| /pharma-dis/api/v1/recalls/<id>/propagate | POST | Propagate recall through network | pharma_dis:recalls |
| /pharma-dis/api/v1/wda | POST | Register WDA | pharma_dis:wda |
| /pharma-dis/api/v1/wda/expiry-alerts | GET | WDA expiry alerts | pharma_dis:wda |
| /pharma-dis/api/v1/wda/<id>/renew | POST | Initiate WDA renewal workflow | pharma_dis:wda |
| /pharma-dis/api/v1/distributors/<id>/gdp-risk | GET | GDP risk score for distributor | pharma_dis:gdp |
| /pharma-dis/api/v1/reports/regulatory | POST | Regulatory distribution report | pharma_dis:reports |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| wda_required_for_wholesale | Wholesale dispatch without active WDA | Deny — obtain WDA |
| cold_chain_monitoring_required | Cold-chain shipment dispatched without monitoring | Deny — activate monitoring |
| serialisation_verification_required | Shipment received without serial verification | Deny — verify serialisation |
| recall_class_i_24h | Class I recall not notified within 24h | Deny — expedite notification |
| recall_effectiveness_check_required | Recall closed without effectiveness check | Deny — complete check |
| gdp_supplier_qualification_required | New supplier added without qualification | Deny — qualify supplier |

## Data Models
- Shipment: shipment_number, distribution_channel, transport_mode, transport_condition, status
- ColdChainRecord: cold_chain_classification, min_temp_celsius, max_temp_celsius, logger_device_id
- TemperatureExcursion: severity, impact_assessment, disposition, regulatory_reported
- SerialisationRecord: serial_number, batch_number, standard, aggregation_level, verified
- RecallRecord: recall_number, recall_class, batch_numbers, status, effectiveness_check_date
- WholesaleDistributionAuthorisation: wda_number, market, scope, status, expiry_date
- GdpDeviationRecord: deviation_type, gdp_status, root_cause, capa_reference

## Streaming Events
- shipment_dispatched, shipment_delivered, shipment_exception
- cold_chain_excursion_detected, temperature_breach_escalated
- serialisation_verified, serialisation_violation_detected
- recall_initiated, recall_completed, gdp_deviation_recorded
- wda_expiring, wda_revoked

## Edge Cases Handled
- Wholesale dispatch requires an active WDA for the specific market; expired WDAs are rejected even if renewal is pending
- Cold-chain excursions at critical severity auto-trigger escalation audit event
- Serialisation violations on receipt create both a verification failure record and an audit event
- Class I recall clock starts at initiation regardless of whether batches have been fully identified
- WDA renewal alerts fire 90 days before expiry; a second alert fires at 30 days if renewal not submitted

## Async Service Methods (v1.1+)
| Method | Description |
|--------|-------------|
| `async_create_shipment(payload)` | Non-blocking shipment creation |
| `async_dispatch_shipment(...)` | Non-blocking dispatch with WDA/CoA checks |
| `async_deliver_shipment(...)` | Non-blocking delivery confirmation |
| `calculate_mkt(temperature_log, ...)` | ICH Q1A(R2) Mean Kinetic Temperature via Haynes equation |
| `ingest_cold_chain_telemetry(shipment_id, readings, ...)` | IoT logger batch ingest with Z-score anomaly detection |
| `propagate_recall_notification(recall_id, network, ...)` | Tiered recall notification across distribution network |
| `validate_aggregation_hierarchy(tenant_id, sscc)` | GS1 SSCC→case→unit hierarchy validation with GTIN check digit |
| `initiate_wda_renewal(wda_id, tenant_id, ...)` | WDA renewal with GDP Annex 17 document checklist |
| `gdp_risk_score(distributor_id, tenant_id, ...)` | Weighted GDP Risk Score (0–100) with band classification |
| `supply_chain_integrity_check(shipment_id, tenant_id)` | 5-point integrity gate: serials, recalls, cold chain, WDA, GDP |
| `async_regulatory_report(period, jurisdiction, ...)` | Extended FMD/DSCSA report with serialisation breakdown |
| `bulk_serialise_products(tenant_id, specs, ...)` | Batch serialisation with per-spec error isolation |

## Composability Notes
Receives released batches from `pharma_mfg` for dispatch. Recall data feeds `pharma_rec` post-market surveillance. Serialisation events integrate with national verification systems (EMVS, DSCSA Hub). GDP deviations link to `pharma_qms` CAPA workflow. IoT telemetry ingestion integrates with cold-chain logger vendors (Elpro, Sensitech, DeltaTrak) via MQTT/REST webhooks.

---

## World-Class Enhancements (v2.0)

- **I1.** Pharmaceutical Distribution — World Class Improvements
- **I2.** Async-First Service Architecture
- **I3.** Repository Pattern with Async PostgreSQL Backend
- **I4.** Cryptographic Serialisation Verification (DSCSA 2023 / EU FMD)
- **I5.** Real-Time IoT Cold Chain Telemetry Ingestion
- **I6.** Mean Kinetic Temperature (MKT) Calculation Engine
- **I7.** Multi-Tier Recall Propagation Engine
- **I8.** Returns Disposition Automation with Regulatory Quarantine Workflow
- **I9.** Blockchain-Anchored Track-and-Trace Ledger
- **I10.** GDP Risk Scoring and Predictive Compliance Dashboard
- **I11.** Cross-Border Import/Export Permit Automation
- **I12.** Demand-Driven Distribution Planning
- **I13.** Serialisation Aggregation Hierarchy Validation
- **I14.** Automated WDA Renewal Workflow
- **I15.** Event-Driven Architecture with Domain Events

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
