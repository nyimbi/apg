# APG Platform Security Incident Response Runbook

**Classification**: Internal — Restricted  
**Owner**: Datacraft Security Team  
**Review cycle**: Annual (SOC 2 Type II requirement)

---

## 1. Purpose

This runbook documents the procedures for identifying, classifying, containing,
eradicating, recovering from, and learning from security incidents on the APG Platform.
Compliance requirement: SOC 2 Type II — Security, Availability, and Confidentiality.

---

## 2. Incident Classification

| Severity | Examples | Response Time | Escalation |
|----------|----------|---------------|------------|
| **P1 Critical** | Data breach, ransomware, active intrusion | 15 minutes | All hands + external IR firm |
| **P2 High** | Unauthorized PAN access, suspected PHI breach | 1 hour | Security team + management |
| **P3 Medium** | Failed authentication spike, AML alert threshold | 4 hours | Security team |
| **P4 Low** | Single failed login, routine compliance flag | Next business day | Security team |

---

## 3. Contact Directory

| Role | Contact | Phone | Hours |
|------|---------|-------|-------|
| Security Lead | nyimbi@gmail.com | [PHONE] | 24/7 |
| Engineering On-call | [PAGERDUTY] | | 24/7 |
| Legal / Privacy | [EMAIL] | | Business hours |
| HIPAA Privacy Officer | [EMAIL] | | Business hours |
| PCI DSS QSA | [EMAIL] | | Business hours |

---

## 4. Detection and Triage

### 4.1 Detection Sources

APG Platform generates security signals via:

1. **NATS audit events**: `apg.events.audl.audit_event` — all platform operations
2. **OPA access decisions**: `apg.events.auth.access_decision` — authorization failures
3. **NATS detokenization events**: `apg.events.vault.pan_detokenized` — PCI scope access
4. **NATS signature events**: `apg.events.esig.signature_created` — GxP-regulated actions
5. **AML alerts**: `apg.events.fintech_aml.*` — financial crime signals
6. **Fraud signals**: `apg.events.fintech_fraud.*` — transaction fraud flags

### 4.2 Initial Triage (15 minutes)

```bash
# 1. Check NATS for anomalous event volume
nats sub "apg.events.>" --count=1000 --timeout=60s | jq '.event_type' | sort | uniq -c

# 2. Check OPA for access_denied spikes
nats sub "apg.events.auth.access_decision" --count=100 | jq 'select(.details.decision=="deny")'

# 3. Verify audit chain integrity
psql $APG_DATABASE_URL -c "
  SELECT id, checksum, chain_hash, prev_hash,
    sha256(concat(prev_hash, checksum)::bytea) = decode(chain_hash, 'hex') as chain_valid
  FROM apg_audit_events
  ORDER BY timestamp DESC LIMIT 100
"

# 4. Check for unauthorized PAN detokenization
nats sub "apg.events.vault.pan_detokenized" --count=50 | jq '.details.requester_id'
```

---

## 5. Response Procedures by Incident Type

### 5.1 Potential Data Breach / Unauthorized PHI Access

**HIPAA Breach Notification timeline: 30 days from discovery**

1. **Contain** (immediate):
   ```bash
   # Revoke OPA role for suspected user
   curl -X POST http://apg-opa:8181/v1/data/apg/roles \
     -d '{"revoke": {"user_id": "[SUSPECT]"}}'
   
   # Review PHI access audit trail
   psql $APG_DATABASE_URL -c "
     SELECT actor_id, action, timestamp, resource_id
     FROM apg_audit_events
     WHERE event_type = 'data_read'
     AND payload->>'capability_id' LIKE 'healthcare_%'
     AND timestamp > now() - interval '24 hours'
     ORDER BY timestamp DESC
   "
   ```

2. **Assess**: Review `contains_pii` flagged events and determine scope of exposure.

3. **Notify**: If PHI was accessed without authorization:
   - Notify HIPAA Privacy Officer within 24 hours
   - Prepare BAA breach notification within 30 days
   - Template: `docs/compliance/hipaa/baa_template.md`

4. **Document**: Record in incident log with ISO 27001 format.

### 5.2 PCI DSS Scope Incident (Unauthorized Card Data Access)

**PCI DSS: Notify QSA within 72 hours**

1. **Immediately isolate** the PCI namespace:
   ```bash
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: pci-emergency-lockdown
     namespace: apg-pci
   spec:
     podSelector: {}
     policyTypes: [Ingress, Egress]
     ingress: []
     egress: []
   EOF
   ```

2. **Check tokenization audit**:
   ```bash
   nats sub "apg.events.vault.pan_detokenized" --count=1000 | \
     jq 'select(.details.requester_role != "pci_authorized")'
   ```

3. **Contact PCI QSA and legal counsel** immediately.

4. **Preserve evidence**: Export NATS JetStream for forensics.

### 5.3 AML Suspicious Activity (SAR Requirement)

1. **Within 24 hours**: Freeze flagged customer transactions via `fintech_aml` capability.
2. **Within 30 days**: File Suspicious Activity Report (SAR) with FIU.
3. **Document** all actions in immutable audit log — tipping off customer is prohibited.

### 5.4 Ransomware / System Compromise

1. **Isolate** affected nodes immediately (remove from k8s cluster).
2. **Preserve** NATS JetStream stream as forensic evidence.
3. **Restore** from last verified backup — PostgreSQL + NATS streams.
4. **Verify audit chain integrity** before resuming operations.
5. **Engage external IR firm** within 2 hours.

---

## 6. Post-Incident Review

Within **5 business days** of resolution:

- [ ] Root cause analysis documented
- [ ] Timeline reconstructed from NATS audit stream
- [ ] OPA policies updated to prevent recurrence
- [ ] Control gaps identified and remediated
- [ ] Lessons learned distributed to engineering and security teams
- [ ] Customer notification sent if required (HIPAA, GDPR, PCI DSS)
- [ ] Incident report filed in `docs/security/incidents/[DATE]-[INCIDENT-ID].md`

---

## 7. Key APG Security Invariants

These must hold at all times. Verify during any security incident:

1. **Audit chain**: Every row in `apg_audit_events` must satisfy
   `chain_hash = SHA256(prev_hash || checksum)` — breaks indicate tampering.

2. **Append-only audit**: No `UPDATE` or `DELETE` on `apg_audit_events` —
   verify with `SELECT * FROM pg_rules WHERE tablename = 'apg_audit_events'`.

3. **PAN tokenization**: No plaintext card numbers in any log, database column,
   or NATS event payload — only tokens (same format, starts with BIN, Luhn-valid).

4. **Electronic signatures**: All GxP-regulated record approvals have signature
   records in `apg_electronic_signatures` with non-empty `meaning`, `signer_id`,
   `timestamp` — verify with `SELECT * WHERE is_valid = false`.

5. **PHI minimum necessary**: All `healthcare_*` data access has a `purpose`
   field in the OPA context — denies without documented purpose.

---

*Last reviewed: [DATE] | Next review: [DATE+1 year]*  
*Approved by: [Security Lead signature]*
