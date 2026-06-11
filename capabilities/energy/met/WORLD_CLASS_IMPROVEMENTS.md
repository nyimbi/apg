# Smart Metering (energy_met) — World-Class Improvements

## 1. Load Profile Disaggregation
**Category**: Analytics / AI  
**Justification**: Utility customers and grid operators benefit from knowing which appliances drive consumption peaks. Non-intrusive load monitoring (NILM) applied to 15-min interval data enables appliance-level insights without extra hardware.  
**Implementation**: Add `async disaggregate_load_profile(meter_id, period)` that runs a lightweight NILM model (edge-hosted Ollama inference) over interval data, returning per-appliance energy estimates and confidence scores.  
**Competitor**: Landis+Gyr Gridstream NILM module; Itron Analytics Engine.

## 2. Voltage Quality Monitoring
**Category**: Power Quality / Reliability  
**Justification**: Utilities face regulatory obligations (EN 50160, ANSI C84.1) to maintain voltage within tolerance. Detecting under/over-voltage events at the meter edge reduces field-crew truck rolls and preempts customer complaints.  
**Implementation**: Add `async submit_voltage_event(meter_id, phase, voltage_rms, duration_ms, event_type)` storing `VoltageQualityEvent` records; add `async voltage_quality_report(meter_id, period)` summarising sag/swell/interruption counts against the applicable standard.  
**Competitor**: Elster EnergyAxis EN50160 module; Itron Riva voltage quality analytics.

## 3. Net Metering & Bi-Directional Flow Tracking
**Category**: Distributed Energy Resources  
**Justification**: Solar and battery penetration is growing. Standard kWh registers miss export energy. ISO 15118 and ANSI C12.19 require separate import/export registers; regulators mandate net metering reconciliation for feed-in tariff settlement.  
**Implementation**: Add `async submit_bidirectional_reading(meter_id, import_kwh, export_kwh, net_kwh, timestamp)` and `async net_metering_reconciliation(meter_id, period)` computing gross consumption, gross generation, and net energy exchanged against a configurable feed-in tariff.  
**Competitor**: Itron OpenWay Riva DER; Landis+Gyr E660 bi-directional register.

## 4. Real-Time Fraud Scoring Pipeline
**Category**: Revenue Protection  
**Justification**: Non-technical losses (NTL) cost utilities 5–30 % of revenue in emerging markets. Rule-based tamper detection is insufficient; ML-driven anomaly scoring combining consumption patterns, peer comparison, and physical tamper signals reduces false positives while catching sophisticated bypass attacks.  
**Implementation**: Add `async score_meter_fraud_risk(meter_id)` that computes a 0–1 risk score from: peer-group z-score on daily kWh, tamper signal history, command history anomalies, and billing payment gaps. Persist score to `MeterFraudScore` model; emit alert when score > configurable threshold.  
**Competitor**: Oracle Utilities NTL Analytics; Elster Connexo fraud detection.

## 5. Predictive Meter Health Scoring
**Category**: Asset Management  
**Justification**: Meter failure causes unbilled energy and missed DR participation. Predictive health scoring using communication failure rate, battery voltage trends, and firmware age enables proactive replacement campaigns, cutting field costs versus reactive replacement.  
**Implementation**: Add `async compute_meter_health_score(meter_id)` returning a 0–100 score derived from: communication success rate (last 30 days), days since last successful read, firmware age, and open tamper events. Store in `MeterHealthScore`; expose fleet-level `async meter_health_fleet_report()`.  
**Competitor**: Itron Meter Asset Management; Landis+Gyr Service & Support analytics.

## 6. Dynamic Load Limiting (Prepayment Credit Management)
**Category**: Revenue Protection / Prepayment  
**Justification**: Token-based prepayment (STS/IEC 62055-41) is the dominant billing model in Africa and South Asia. Dynamic load limiting tied to credit balance reduces meter-level debt risk and enables graduated reconnection without field visits.  
**Implementation**: Add `async set_load_limit(meter_id, limit_kw, authorised_by)` issuing a `set_load_limit` remote command; add `async update_prepayment_credit(meter_id, credit_units, token_id)` recording top-up events in a `PrepaymentTransaction` model; add `async prepayment_credit_alert(meter_id, threshold_kwh)` triggering an SMS/push via `ntfy`.  
**Competitor**: Hexing Smart HXE310 STS prepayment; Landis+Gyr KEYSTONE.

## 7. OpenADR 2.0b Demand Response Protocol
**Category**: Grid Integration / Standards Compliance  
**Justification**: OpenADR 2.0b is the FERC-endorsed standard for automated DR exchange between utilities (VTN) and customer systems (VEN). Native support enables interoperability with ISO/RTO demand response markets and large C&I customers with BAS/BMS integration.  
**Implementation**: Add `async publish_open_adr_event(dr_id, ven_ids, signal_type, signal_level)` that serialises a DR event to an oadrDistributeEvent XML payload and POSTs to registered VEN endpoints; add `async receive_oadr_opt(ven_id, dr_id, opt_type)` handling optIn/optOut responses and updating `DemandResponseEvent.opt_out_meter_ids`.  
**Competitor**: EnerNOC DRBizNet OpenADR stack; AutoGrid Flex.

## 8. Edge Firmware OTA Orchestration
**Category**: Security / Operations  
**Justification**: Unpatched meter firmware is a critical cyber-attack surface (ICS-CERT advisories 2022–2025 cite multiple AMI vulnerabilities). Orchestrated OTA campaigns with rollback, canary deployment (5 % pilot before fleet roll-out), and per-meter cryptographic signature verification reduce patching lag from months to days.  
**Implementation**: Add `async initiate_firmware_campaign(tenant_id, firmware_version, target_meter_ids, canary_pct, signed_manifest_uri)` creating a `FirmwareCampaign` record; add `async track_firmware_campaign(campaign_id)` returning per-meter update status; add `async rollback_firmware_campaign(campaign_id, reason)` issuing rollback commands to all updated meters.  
**Competitor**: Itron Firmware Management; Landis+Gyr Remote Firmware Update.

## 9. MDMS Integration via IEC 61968-9 Adapter
**Category**: Interoperability / Standards  
**Justification**: Utilities already run Meter Data Management Systems (Oracle MDM, Itron MDM, OSIsoft PI). Exporting interval data in IEC 61968-9 MeterReading XML enables zero-friction integration without custom ETL pipelines, satisfying regulatory interoperability mandates.  
**Implementation**: Add `async export_mdm_reading_xml(meter_id, period)` serialising interval readings to IEC 61968-9 MeterReading XML; add `async push_readings_to_mdms(mdms_endpoint, meter_ids, period)` doing a bulk HTTP POST with retry and delivery acknowledgement.  
**Competitor**: Oracle Utilities MDM IEC 61968 adapter; Siemens eMeter MDM.

## 10. Loss Calculations & Technical Loss Attribution
**Category**: Grid Operations / Engineering  
**Justification**: Distribution system losses (technical + non-technical) directly impact profitability and regulatory reporting. Automated loss attribution—comparing feeder-level substation generation against the sum of customer meter readings—pinpoints loss hotspots without expensive DMS integration.  
**Implementation**: Add `async compute_feeder_losses(feeder_id, period)` that aggregates meter readings for all meters on a feeder, compares to substation injections (provided as input), and returns technical loss estimate, NTL estimate, and per-section loss breakdown. Store results in `FeederLossRecord`.  
**Competitor**: ABB Network Manager Loss Analysis; Schneider Electric Advanced Distribution Management.

## 11. Meter Data Streaming via MQTT/Kafka Bridge
**Category**: Real-Time Data / Integration  
**Justification**: Batch AMI polling (15–60 min cycle) is incompatible with real-time grid edge control (sub-minute balancing, V/VAR optimisation). Publishing readings to an MQTT broker or Kafka topic as they arrive enables real-time analytics pipelines, DR systems, and SCADA integration at scale.  
**Implementation**: Add `async publish_reading_to_stream(reading_id, broker_url, topic)` serialising an `IntervalReading` to a JSON CloudEvent and publishing via aiomqtt or aiokafka. Add `async configure_streaming_bridge(broker_type, broker_url, topic_prefix, enabled)` stored in tenant configuration.  
**Competitor**: Itron Riva Edge streaming; Landis+Gyr Gridstream cloud broker.

## 12. Outage Detection & FLISR Event Correlation
**Category**: Reliability / Outage Management  
**Justification**: Communication failure from a cluster of meters is a reliable outage indicator. Correlating meter communication loss with GIS topology enables outage boundary inference in < 2 minutes—faster than customer call-in and without smart switches. Feeds `energy_dis` for SAIDI/SAIFI tracking.  
**Implementation**: Add `async detect_outage_cluster(feeder_id, threshold_pct, window_minutes)` grouping meters with communication loss exceeding `threshold_pct` over the last `window_minutes`, inferring outage boundary from GIS adjacency, and emitting an `OutageEvent` to `energy_dis`. Add `async correlate_restoration(outage_event_id)` detecting when the cluster returns to communication and computing interruption duration.  
**Competitor**: Oracle Utilities NMS outage detection; Milsoft WindMil MDM correlation.

## 13. Time-of-Use & Critical Peak Pricing Tariff Engine
**Category**: Billing Integration / DR  
**Justification**: TOU and CPP tariffs require accurate time-stamped interval data aggregated by tariff period (on-peak, off-peak, super-off-peak, CPP). A built-in tariff engine that maps interval readings to tariff buckets enables direct feeding of `energy_bil` without a separate MDM, reducing settlement latency from days to hours.  
**Implementation**: Add `async apply_tou_tariff(meter_id, tariff_schedule, period)` that classifies each interval reading into a tariff bucket using the schedule's time-window definitions and returns per-bucket kWh totals and a pre-bill summary. Store schedule in `TouTariffSchedule` model.  
**Competitor**: Itron Analytics TOU bucketing; Oracle Utilities Rate Engine.

## 14. Cybersecurity Event Log (NERC CIP / IEC 62351 Compliance)
**Category**: Security / Compliance  
**Justification**: NERC CIP-007 (North America) and IEC 62351 (international) mandate detailed security event logging for AMI components including authentication failures, firmware changes, and remote command issuance. Dedicated cybersecurity logging with tamper-evident hash chaining satisfies audit requirements and feeds SIEM integration.  
**Implementation**: Add `async log_security_event(meter_id, event_type, source_ip, severity, description)` storing a `SecurityEvent` record with SHA-256 hash chained to the previous event; add `async export_security_log(period, format)` generating a NERC CIP-compliant CSV/JSON report. Integrate with `intel` capability threat detection pipeline.  
**Competitor**: Itron Secure Operations Center; Landis+Gyr Advanced Security Module.

## 15. Carbon & Emissions Tracking per Meter
**Category**: ESG / Sustainability Reporting  
**Justification**: Corporate ESG mandates (SEC Climate Disclosure Rule, IFRS S2, GHG Protocol Scope 2) require utility customers to report location-based and market-based Scope 2 emissions. Attaching grid emissions intensity (kg CO₂e/kWh) to interval readings enables per-meter carbon footprint reports without third-party data brokers.  
**Implementation**: Add `async submit_grid_emission_factor(region_id, timestamp, kg_co2e_per_kwh, source)` storing `GridEmissionFactor` records; add `async compute_meter_carbon_footprint(meter_id, period)` joining interval readings with the closest-in-time emission factor and returning total kg CO₂e, average intensity, and a time-series of hourly emissions. Expose via `/energy-met/api/v1/meters/<id>/carbon`.  
**Competitor**: Arcadia Power Grid Emissions API; WattTime AER data service.
