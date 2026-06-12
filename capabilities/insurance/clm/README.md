# Claims Management (ins_clm)

End-to-end claims lifecycle: FNOL, loss assessment, reserve management, payment processing,
fraud detection, subrogation, litigation tracking, STP, SLA compliance, and regulatory reporting.

© 2025 Datacraft | Author: Nyimbi Odero

## Features

- **FNOL Registration** — multi-channel intake with velocity pre-check
- **Complexity Triage** — deterministic + score-based tier assignment at intake (simple/standard/complex/catastrophic)
- **Straight-Through Processing (STP)** — auto-approve low-complexity, low-fraud claims in one atomic step
- **Reserve Management** — OCR, IBNR, ALAE reserve types with adequacy monitoring and auto-warning
- **Payment Processing** — partial, full, advance, ex-gratia, recoverable-advance disbursements
- **Excess / Deductible Engine** — stacked multi-rule excess computation with audit trail
- **Fraud Detection** — score-based and manual flagging; velocity burst alerts
- **Claim Velocity Check** — rolling-window anomaly detection per policy and claimant
- **Litigation Management** — matter lifecycle, event log, legal cost tracking
- **Subrogation** — initiation and incremental recovery recording
- **Multi-Currency FX** — immutable rate conversion with full provenance
- **Regulatory Reporting** — large-loss notifications (IRA Kenya C-4 format)
- **SLA Compliance Dashboard** — portfolio heat-map with breach event emission
- **Loss Ratio Report** — earned-premium vs. incurred losses analytics
- **Full Audit Trail** — immutable event log per tenant

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/clm/health | Service health check |
| GET | /api/insurance/clm/describe | Capability description |
| GET | /api/insurance/clm/claims | List claims |
| POST | /api/insurance/clm/claims | Register FNOL |
| GET | /api/insurance/clm/claims/{id} | Get claim detail |
| PUT | /api/insurance/clm/claims/{id} | Update claim |
| DELETE | /api/insurance/clm/claims/{id} | Withdraw claim |
| POST | /api/insurance/clm/claims/{id}/reserve | Set reserve |
| POST | /api/insurance/clm/claims/{id}/payment | Process payment |
| POST | /api/insurance/clm/claims/{id}/fraud | Fraud assessment |
| POST | /api/insurance/clm/claims/{id}/approve | Approve claim |
| POST | /api/insurance/clm/claims/{id}/subrogation | Initiate subrogation |
| POST | /api/insurance/clm/claims/{id}/stp | Evaluate STP eligibility |
| POST | /api/insurance/clm/claims/{id}/complexity | Score claim complexity |
| GET | /api/insurance/clm/claims/{id}/reserve/adequacy | Reserve adequacy check |
| POST | /api/insurance/clm/claims/{id}/excess | Compute applicable excess |
| POST | /api/insurance/clm/claims/{id}/fx | FX currency conversion |
| POST | /api/insurance/clm/claims/{id}/litigation | Open litigation matter |
| POST | /api/insurance/clm/litigation/{id}/events | Log litigation event |
| GET | /api/insurance/clm/velocity | Claim velocity check |
| GET | /api/insurance/clm/regulatory/large-loss | Large-loss notifications |
| GET | /api/insurance/clm/sla | SLA compliance dashboard |
| GET | /api/insurance/clm/summary | Claims portfolio summary |
| GET | /api/insurance/clm/loss-ratio | Loss ratio report |
| GET | /api/insurance/clm/audit | Audit trail |

## Quick Start

```python
from capabilities.insurance.clm.service import ClaimsManagementService
from decimal import Decimal

svc = ClaimsManagementService(tenant_id="acme_insurance")

# 1. Register FNOL
claim = await svc.register_fnol(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2026-001",
    claimant_name="Jane Smith",
    claimant_id="ID-99999",
    incident_date="2026-05-20",
    incident_description="Vehicle rear-ended at intersection",
    estimated_loss=Decimal("35000"),
    reported_by="agent_002",
)

# 2. Score complexity
complexity = await svc.score_claim_complexity(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    injury_involved=False,
)

# 3. STP fast-track for simple claims
stp = await svc.evaluate_stp_eligibility(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    stp_loss_ceiling=Decimal("50000"),
)
if stp["eligible"]:
    print("Auto-approved:", stp["auto_approved_amount"])
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. AI-Powered Predictive Reserve Adequacy Scoring** [AI/ML]
- **I2. Real-Time Multi-Factor Fraud Network Graph** [AI/ML]
- **I3. Automated STP (Straight-Through Processing) for Low-Complexity Claims** [Feature]
- **I4. Litigation Management with Matter Lifecycle Tracking** [Feature]
- **I5. Automated Regulatory Compliance & Statutory Reporting Engine** [Compliance]
- **I6. Dynamic Excess & Deductible Management** [Feature]
- **I7. Document Intelligence — OCR & Evidence Classification** [AI/ML]
- **I8. Claims Velocity & Frequency Anomaly Detection** [Security]
- **I9. Multi-Channel FNOL — WhatsApp, USSD, Email, API** [Integration]
- **I10. Intelligent Reserve Adequacy Warnings & Escalation** [Feature]
- **I11. Claimant Self-Service Portal & Status Push Notifications** [UX]
- **I12. Subrogation Recovery Optimisation with Third-Party Liability Scoring** [Feature]
- **I13. Multi-Currency Claims with Real-Time FX Settlement** [Feature]
- **I14. Claims Triage & Complexity Scoring at FNOL** [AI/ML]
- **I15. Actuarial IBNR Estimation with Development Triangle Export** [Compliance]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
