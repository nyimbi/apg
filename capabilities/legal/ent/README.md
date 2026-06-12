# leg_ent — Entity & Corporate Secretary

Company registry, board management, statutory filings, annual returns, share register.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ent/health | Health check |
| GET | /api/legal/ent/entities | List entities |
| GET | /api/legal/ent/entities/{id} | Get entity |
| POST | /api/legal/ent/entities | Register entity |
| PUT | /api/legal/ent/entities/{id} | Update entity |
| DELETE | /api/legal/ent/entities/{id} | Deactivate entity |
| GET | /api/legal/ent/entities/{id}/directors | List directors |
| POST | /api/legal/ent/directors | Appoint director |
| PUT | /api/legal/ent/directors/{id} | Update director |
| DELETE | /api/legal/ent/directors/{id} | Remove director |
| GET | /api/legal/ent/entities/{id}/shareholders | List shareholders |
| POST | /api/legal/ent/shareholders | Register shareholder |
| POST | /api/legal/ent/shareholders/transfer | Transfer shares |
| GET | /api/legal/ent/filings | List filings |
| POST | /api/legal/ent/filings | Schedule filing |
| POST | /api/legal/ent/filings/{id}/complete | Complete filing |
| DELETE | /api/legal/ent/filings/{id} | Cancel filing |
| POST | /api/legal/ent/resolutions | Create board resolution |
| GET | /api/legal/ent/entities/{id}/resolutions | List resolutions |
| GET | /api/legal/ent/dashboard | Corporate dashboard |
| GET | /api/legal/ent/audit | Audit events |

## Service Class

`EntityCorporateSecretaryService` — entity registration, director appointment/removal, share register management, share transfers, statutory filing tracking, board resolutions.

### Core async methods

```python
svc = EntityCorporateSecretaryService(tenant_id="acme")

# Register a new entity
entity = await svc.create_entity(
    tenant_id="acme",
    legal_name="Acme Holdings Ltd",
    entity_type="private_company",
    jurisdiction="KE",
    registration_number="CPR/2024/001234",
)

# Appoint a director
director = await svc.appoint_director(
    tenant_id="acme",
    entity_id=entity["id"],
    full_name="Jane Doe",
    id_number="12345678",
    appointment_date="2024-01-15",
    nationality="KE",
)

# Transfer shares
transfer = await svc.transfer_shares(
    tenant_id="acme",
    entity_id=entity["id"],
    from_shareholder_id="sh_abc",
    to_shareholder_id="sh_xyz",
    number_of_shares=10000,
    consideration=500000.00,
    transfer_date="2024-06-01",
)
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements moving `leg_ent` from a capable registry tool to a best-in-class corporate governance platform.

**I1. Ownership Graph & UBO Disclosure** — recursive async traversal resolves beneficial owner chains across subsidiary trees, collapsing indirect ownership to direct-equivalent percentages [Compliance]

**I2. Compliance Calendar with SLA Breach Prediction** — `predict_filing_risk` scores entities by filing latency history and flags breach risk 30/14/7 days out via `get_upcoming_deadlines` [AI/ML]

**I3. Registered Capital & Share Ledger in Decimal** — replaces `float` monetary fields with `Decimal` throughout the share register; `compute_share_capital_summary` returns authorised/issued/paid-up per share class [Feature]

**I4. Board Committee Management** — `create_committee` links directors, charter text, and quorum rules to an entity; `record_committee_meeting` closes the audit cycle [Feature]

**I5. Document Vault with Version Control** — `attach_document` stores metadata + SHA-256 hash per version; `list_entity_documents` filters by doc_type; no overwrites — each version is a new record [Feature]

**I6. Multi-Jurisdiction Compliance Rules Engine** — `JURISDICTION_RULES` dict drives `validate_entity_compliance`, returning a gap list against jurisdiction-specific director minimums, filing cadences, and UBO thresholds [Compliance]

**I7. Power of Attorney & Signatory Register** — `grant_power_of_attorney` stores scope/expiry; `list_active_signatories` feeds downstream `leg_con` signature-verification flows [Feature]

**I8. Charges / Security Interest Register (PPSA-Aware)** — `register_charge` tracks chargee, amount (Decimal), and registration deadline; overdue registration raises a compliance alert automatically [Compliance]

**I9. Annual Return Auto-Generation** — `generate_annual_return_pack` snapshots directors, share register, and registered address into a structured dict mapping to jurisdiction-specific form fields (CR12/CS14) [Feature]

**I10. Beneficial Ownership Threshold Alerts** — `_check_ownership_thresholds` fires after every allotment/transfer; crossing 10%/25%/51% appends a compliance alert and emits an audit event [Compliance / AI/ML]

**I11. Director Conflict-of-Interest Register** — `declare_director_interest` records counterparty, nature, and resolution; cross-references `leg_con` contract IDs for linked-capability audit trails [Compliance]

**I12. Corporate Hierarchy Visualisation Data** — `get_corporate_hierarchy` returns a DFS tree over `parent_entity_id` links with ownership percentages at each edge, ready for D3/Lucidchart rendering [UX]

**I13. Statutory Deadline Notification Hooks** — `register_deadline_hook` stores channel/endpoint/days-before config; `fire_due_notifications` dispatches webhook payloads daily without transport coupling [Integration]

**I14. Entity Health Score** — `compute_entity_health_score` aggregates five weighted sub-scores (filing 30%, directors 20%, share register 20%, charges 15%, UBO 15%) into a 0–100 composite with a deduction breakdown [AI/ML]

**I15. Cross-Capability Composability Hooks** — `resolve_entity_ref` is the canonical single entry point returning `{id, legal_name, registration_number, jurisdiction, status}`; eliminates duplicate entity lookups across `leg_con`, `hr_emp`, `fin_tax`, `fin_bank` [Integration]

---

## New Methods (v2.0 examples)

### `resolve_entity_ref` — cross-capability canonical lookup

```python
# Called by leg_con, hr_emp, fin_tax, fin_bank — never implement entity lookup twice
ref = await svc.resolve_entity_ref(tenant_id="acme", entity_id="ent_abc123")
# -> {"id": "ent_abc123", "legal_name": "Acme Holdings Ltd",
#     "registration_number": "CPR/2024/001234", "jurisdiction": "KE", "status": "active"}
```

### `compute_entity_health_score` — governance risk triage

```python
# Triage 200 entities in seconds; drill into breakdown for remediation priorities
score = await svc.compute_entity_health_score(tenant_id="acme", entity_id="ent_abc123")
# -> {
#     "score": 74,
#     "breakdown": {
#         "filing_compliance": {"score": 24, "max": 30, "deductions": ["annual_return_overdue_14d"]},
#         "director_adequacy": {"score": 20, "max": 20},
#         "share_register":    {"score": 14, "max": 20, "deductions": ["ubo_unresolved_chain"]},
#         "charges_disclosure": {"score": 15, "max": 15},
#         "ubo_completeness":  {"score": 1,  "max": 15, "deductions": ["indirect_chain_depth_3_unresolved"]},
#     }
# }
```

### `generate_annual_return_pack` — CR12/CS14 data assembly

```python
# Eliminates 2-4 hours of manual data gathering per entity per year
pack = await svc.generate_annual_return_pack(
    tenant_id="acme",
    entity_id="ent_abc123",
    financial_year_end="2024-12-31",
)
# -> AnnualReturnPack with directors[], shareholders[], registered_address,
#    share_capital_summary, filing_period, completeness_flags
#    Raises ValueError if any required field is missing before returning.
```

---

## Composability

`leg_ent` is the authoritative entity source for the APG platform. Other capabilities reference entities exclusively via `resolve_entity_ref` — never by querying the entity store directly.

| Downstream capability | Integration point |
|-----------------------|-------------------|
| `leg_con` (Contracts) | Entity party lookup, signatory register |
| `hr_emp` (Employment) | Employer entity ref, director cross-check |
| `fin_tax` (Tax) | Registered capital, financial year dates |
| `fin_bank` (Banking) | Authorised signatories, charges register |

---

© 2025 Datacraft · www.datacraft.co.ke
