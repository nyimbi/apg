# Exploration Data Management

## Overview
Manages the full lifecycle of mineral exploration data from drill-hole collar logging through downhole surveys, geological interval logging, geochemical assay management, QAQC monitoring, resource estimation workflows, and JORC/NI 43-101/SAMREC compliance reporting. Enforces data integrity rules including interval non-overlap, competent person requirements, and QAQC insertion obligations before any resource can be published.

## Capability ID
`mining_exp`

## Provides
| Service | Description |
|---|---|
| drillhole_collar_management | Unique collar registration with coordinate system enforcement |
| downhole_survey_management | Dip/azimuth survey data linked to collars |
| lithology_logging | Geological interval logging with lithology, oxidation state, and mineralisation style |
| assay_data_management | Bulk assay import with interval overlap validation and lab cert tracking |
| qaqc_monitoring | Blank/standard/duplicate insertion and flag management |
| resource_estimation_workflow | Competent-person-gated resource estimate creation, approval, and publication |
| jorc_reporting_workflow | JORC 2012 compliant reporting with sign-off gate |
| ni_43_101_reporting_workflow | NI 43-101 compliant technical reporting |
| geological_map_management | Spatial drillhole data for GeoJSON map rendering |
| exploration_target_delineation | Non-JORC exploration target scoping records |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication and permission checks |
| audl | Full audit trail for all data changes |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration |
| ntfy | Notifications for resource approval and QAQC flags |
| wflo | Resource estimate and report approval workflows |
| nlpc | Natural language search across assay and geology data |
| geos | Spatial collar indexing and map queries |
| srch | Full-text search across hole names, prospects, reports |
| mqeb | Event streaming for downstream consumers |

## Configuration
| Key | Default | Description |
|---|---|---|
| drill_holes.collar_survey_required | true | Requires surveyed collar coordinates |
| drill_holes.down_hole_survey_required | true | Requires downhole dip/azimuth surveys |
| sampling.qaqc_insertion_required | true | Enforces QAQC sample insertion per batch |
| assays.lab_cert_required | true | Lab certificate ref mandatory on all assays |
| resources.competent_person_required | true | CP assignment mandatory for resource estimates |
| reporting.public_disclosure_review_required | true | Requires approval before external publication |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-exp/drillholes | GET | List collars | mining_exp:view |
| /api/mining-exp/drillholes | POST | Create collar | mining_exp:write |
| /api/mining-exp/drillholes/:id | GET | Get collar | mining_exp:view |
| /api/mining-exp/drillholes/:id/depth | PATCH | Update actual depth | mining_exp:write |
| /api/mining-exp/drillholes/:id/desurvey | GET | 3D desurveyed path | mining_exp:view |
| /api/mining-exp/drillholes/:id/surveys | POST | Record deviation survey | mining_exp:write |
| /api/mining-exp/drillholes/:id/gaps | GET | Detect sampling gaps | mining_exp:view |
| /api/mining-exp/drillholes/:id/composites | GET | Fixed-length grade composites | mining_exp:view |
| /api/mining-exp/assays | GET | List assay results | mining_exp:view |
| /api/mining-exp/assays/import | POST | Bulk import assays | mining_exp:write |
| /api/mining-exp/assays/hole/:hole_id | GET | Assays for a hole | mining_exp:view |
| /api/mining-exp/assays/:id/qaqc-flag | POST | Flag QAQC result | mining_exp:write |
| /api/mining-exp/geology | POST | Log geology interval | mining_exp:write |
| /api/mining-exp/geology/hole/:hole_id | GET | Geology for a hole | mining_exp:view |
| /api/mining-exp/resources | GET/POST | List/create resource estimates | mining_exp:resources |
| /api/mining-exp/resources/:id/approve | POST | Approve estimate | mining_exp:resources |
| /api/mining-exp/resources/:id/publish | POST | Publish estimate | mining_exp:resources |
| /api/mining-exp/resources/domains | GET/POST | List/create resource domains | mining_exp:resources |
| /api/mining-exp/resources/domains/assign | POST | Assign intervals to domains | mining_exp:resources |
| /api/mining-exp/reports | GET/POST | List/create compliance reports | mining_exp:reports |
| /api/mining-exp/reports/:id/sign-off | POST | CP sign-off | mining_exp:reports |
| /api/mining-exp/reports/:id/publish | POST | Publish report | mining_exp:reports |
| /api/mining-exp/summary | GET | KPI summary | mining_exp:view |
| /api/mining-exp/licences/:id/validate-programme | GET | Spatial boundary validation | mining_exp:spatial |
| /api/mining-exp/competent-persons | POST | Register CP | mining_exp:admin |
| /api/mining-exp/competent-persons/:id/validate | GET | Validate CP credential | mining_exp:view |
| /api/mining-exp/bulk-density | POST | Record bulk density measurement | mining_exp:write |
| /api/mining-exp/bulk-density/summary | GET | BD statistics by lithology | mining_exp:view |
| /api/mining-exp/expenditures | POST | Record expenditure line item | mining_exp:finance |
| /api/mining-exp/expenditures/cost-per-unit | GET | Cost per resource unit | mining_exp:finance |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| tenant_context_required | No tenant context | DENY |
| assay_requires_collar | Collar doesn't exist | DENY — create collar first |
| assay_from_to_required | No from/to | DENY |
| qaqc_insertion_required | No QAQC in batch | DENY |
| qaqc_bypass_denied | Bypass without authority | DENY |
| interval_overlap_check | Overlapping assay intervals | DENY |
| resource_competent_person_required | No CP assigned | DENY |
| resource_approval_required_for_publication | Not approved | DENY |
| report_competent_person_sign_off | CP not signed | DENY |
| delete_approved_resource_denied | Approved resource delete | DENY — supersede instead |
| cross_tenant_read_denied | Cross-tenant access | DENY |

## Data Models
| Model | Key Fields |
|---|---|
| DrillholeCollarCreate/Response | hole_id, hole_type, easting, northing, elevation_m, coordinate_system, planned/actual_depth_m |
| AssayResultCreate/Response | hole_id, sample_id, from_m, to_m, commodity, grade_value, assay_method, lab_certificate_ref, qaqc_flag |
| GeologyIntervalCreate/Response | hole_id, from_m, to_m, lithology_code, oxidation_state, mineralisation_style, RQD, TCR |
| ResourceEstimateCreate/Response | classification, reporting_standard, tonnes, grade_value, competent_person_id, review_status, published |
| ComplianceReportCreate/Response | reporting_standard, resource_estimate_ids, competent_person_id, competent_person_signed, published |

## Streaming Events
- `drillhole_collar_recorded` — new collar registered
- `assay_result_imported` — assay batch imported
- `qaqc_flag_raised` — QAQC failure detected
- `qaqc_flag_resolved` — QAQC issue cleared
- `resource_estimate_submitted` — estimate pending review
- `resource_estimate_approved` — estimate approved by CP
- `compliance_report_published` — public disclosure made
- `geological_map_updated` — spatial data updated

## Edge Cases Handled
- Duplicate `hole_id` within tenant rejected at creation time
- Assay intervals checked for overlap per hole and commodity; partial overlap rejected
- Resource estimates in APPROVED status cannot be directly edited; must create superseding version
- Compliance reports require competent person sign-off matching the assigned CP ID (not just any admin)
- Negative grade values are rejected at model validation layer
- Azimuth validated to [0, 360) range; dip validated to [-90, 0] (downward negative convention)
- Collar not found → assay import fails atomically for entire batch
- Cross-tenant data access rejected with AssertionError

## Composability Notes
- Feeds resource estimates into `mining_pro` grade control cutoff decisions
- Geological data consumed by `mining_ore` for feed characterisation
- JORC reports referenced in `mining_env` ESG and closure documentation
- Integrates with `geos` for spatial indexing and map tile serving
- Assay data consumed by `ragn` for geological RAG queries
- Expenditure data feeds `mining_fin` for exploration budget reporting
- Desurveyed 3D collar paths exported to `mining_3d` for block model alignment
- CP credential registry consumed by `wflo` for automated approval gating
