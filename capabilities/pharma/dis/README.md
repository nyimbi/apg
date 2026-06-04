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
| /pharma-dis/api/v1/cold-chain/excursions | POST | Report temperature excursion | pharma_dis:cold_chain |
| /pharma-dis/api/v1/serialisation/verify | POST | Verify serial number | pharma_dis:serialisation |
| /pharma-dis/api/v1/recalls | POST | Initiate recall | pharma_dis:recalls |
| /pharma-dis/api/v1/wda | POST | Register WDA | pharma_dis:wda |
| /pharma-dis/api/v1/wda/expiry-alerts | GET | WDA expiry alerts | pharma_dis:wda |

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

## Composability Notes
Receives released batches from `pharma_mfg` for dispatch. Recall data feeds `pharma_rec` post-market surveillance. Serialisation events integrate with national verification systems. GDP deviations link to `pharma_qms` CAPA workflow.
