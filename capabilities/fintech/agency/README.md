# Agency Banking

## Overview
Agency Banking extends financial services reach through a network of accredited third-party outlets — retail shops, pharmacies, petrol stations, mobile agents, cooperatives, and community banks — operating under a governed program structure. Each outlet holds a float account, serves KYC/AML-verified customers, and processes transactions across services including cash-in/out, bill payment, airtime, loan disbursement, card services, and government payments.

The capability manages the full lifecycle: program registration and outlet onboarding, agent accreditation and float management, transaction recording, commission settlement, dispute handling, field supervision visits, and AI-assisted agent workflows. All activity streams to `apg.fintech.agency.lifecycle` via Bytewax.

## Capability ID
`fintech_agency`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| agency_program_governance | Register and govern agency programs with owner, country, currency, and settlement-model constraints |
| agency_outlet_lifecycle | Onboard outlets with license, location, and security-plan evidence; manage channel assignments |
| agency_agent_accreditation | Accredit human agents with identity, training, and background-check evidence |
| agency_float_management | Open and reconcile float accounts linked to outlets with ledger references |
| agency_customer_workflow | Onboard customers with KYC, AML, fraud, and consent evidence under tier controls |
| agency_transaction_workflow | Record channel transactions with float sufficiency checks and high-value approval gates |
| agency_cash_movement_workflow | Record float top-ups, cash pickups, vault rebalances with custodian assignment |
| agency_commission_settlement_workflow | Settle outlet commissions with reconciliation and payment-reference evidence |
| agency_dispute_workflow | Open and resolve agent transaction disputes with evidence and reviewer assignment |
| agency_supervision_workflow | Record field supervision visits with outcome tracking and remediation plans |
| agency_ai_agent_workflow | Register and govern AI agents acting in agency banking roles |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication and session management |
| audl | Immutable audit trail |
| ntfy | Customer and agent notifications |
| nlpc | NLP for narrative and evidence analysis |
| keym | Key management |
| fintech_payments | Payment execution for transactions |
| fintech_wallets | Wallet backing for float accounts |
| fintech_cards | Card service operations at outlets |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening for customers and transactions |
| fintech_fraud | Fraud signal scoring |
| fintech_remittance | Cross-border remittance via agent networks |
| fintech_neobanking | Account opening at outlets |
| fintech_lending | Loan disbursement and collection |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| transactions.daily_limit | number | 200000 | Per-outlet daily transaction ceiling |
| transactions.high_value_threshold | number | 100000 | Threshold requiring human approval |
| cash_movements.high_value_threshold | number | 100000 | Cash movement approval threshold |
| outlets.minimum_initial_float | number | 500 | Minimum float to onboard outlet |
| customers.supported_tiers | list | tier_1, tier_2, tier_3 | Allowed KYC tiers |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-agency/dashboard | GET | fintech_agency:view | Overview |
| programs | /fintech-agency/programs | GET/POST | fintech_agency:manage_programs | Programs |
| outlets | /fintech-agency/outlets | GET/POST | fintech_agency:manage_outlets | Network |
| agents | /fintech-agency/agents | GET/POST | fintech_agency:manage_agents | Network |
| float_accounts | /fintech-agency/float-accounts | GET/POST | fintech_agency:float | Liquidity |
| customers | /fintech-agency/customers | GET/POST | fintech_agency:customers | Customers |
| transactions | /fintech-agency/transactions | GET/POST | fintech_agency:transactions | Transactions |
| cash_movements | /fintech-agency/cash-movements | GET/POST | fintech_agency:liquidity | Liquidity |
| commissions | /fintech-agency/commissions | GET/POST | fintech_agency:commissions | Settlement |
| disputes | /fintech-agency/disputes | GET/POST | fintech_agency:disputes | Servicing |
| supervision | /fintech-agency/supervision | GET/POST | fintech_agency:supervision | Field Control |
| ai_agents | /fintech-agency/ai-agents | GET/POST | fintech_agency:admin | Automation |
| settings | /fintech-agency/settings | GET/POST | fintech_agency:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| agency_write_requires_policy | Write without policy evidence | deny |
| outlet_initial_float_valid | Below minimum initial float | deny |
| transaction_float_sufficient | Cash-out with insufficient float | deny |
| transaction_limit_valid | Amount exceeds daily limit | deny |
| transaction_currency_matches_float | Currency differs from float account | deny |
| high_value_transaction_requires_approval | Amount > 100,000 without human approval | require_review |
| high_value_cash_movement_requires_approval | Cash movement > 100,000 without approval | require_review |
| supervision_findings_require_remediation | Findings present without remediation plan | require_review |
| agency_batch_requires_bytewax | Batch without Bytewax stream | deny |
| privileged_agency_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| AgencyProgram | id, tenant_id, name, owner_id, country, currency, settlement_model, services, status |
| AgencyOutlet | id, program_id, outlet_type, country, license_reference, location_reference, initial_float, status |
| AccreditedAgent | id, outlet_id, name, identity_reference, training_reference, background_check_reference, status |
| FloatAccount | id, outlet_id, currency, ledger_reference, balance, status |
| AgencyCustomer | id, outlet_id, customer_reference, kyc_profile_id, tier, consent_reference, aml_reference, fraud_reference |
| AgencyTransaction | id, outlet_id, agent_id, customer_id, float_account_id, service, channel, currency, amount, status |
| CashMovement | id, outlet_id, movement_type, currency, amount, custodian_id, status |
| CommissionSettlement | id, outlet_id, currency, amount, reconciliation_reference, payment_reference |
| AgencyDispute | id, transaction_id, reason, evidence_references, reviewer_id, status |
| SupervisionVisit | id, outlet_id, supervisor_id, outcome, findings, remediation_plan, evidence_references |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| agency_program_registered | New program created |
| agency_outlet_onboarded | Outlet passes onboarding checks |
| agency_agent_accredited | Agent accreditation approved |
| float_account_opened | Float account opened for outlet |
| agency_customer_onboarded | Customer enrolled at outlet |
| agency_transaction_recorded | Transaction posted |
| cash_movement_recorded | Float cash movement recorded |
| commission_settlement_recorded | Commission paid to outlet |
| agency_dispute_opened | Dispute filed against transaction |
| supervision_visit_recorded | Field supervision visit recorded |
| agency_ai_agent_registered | AI agent registered for agency role |

## Edge Cases Handled
- Cash-out transactions blocked when available float is insufficient — the check fires before posting; callers must compute `float_sufficient` in context before invoking the rule engine
- Transaction currency must match float account currency; mismatched currencies are denied even if both appear in `SUPPORTED_CURRENCIES`
- High-value transactions (>100,000) require a human approval record — not just a flag — before the transaction is posted
- Supervision visits with a non-empty findings list require an attached remediation plan before the visit can be closed
- Agent AI runtimes and roles are validated against allow-lists at registration time; unknown values are denied, not silently ignored
- `same_country` is blocked for cross-border corridors; agency transactions remain domestic by design

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide the identity and AML evidence required before customers can be onboarded; `fintech_fraud` provides fraud signals for each transaction
- **Downstream**: `fintech_payments` and `fintech_wallets` execute underlying money movement; `fintech_lending` uses agent outlets as disbursement and collection points; `fintech_cards` enables card services at physical outlets
- **Peer**: Commonly deployed alongside `fintech_mobile` (USSD/mobile channels feeding into agency transactions) and `fintech_remittance` (cross-border payouts via agent networks)

## Development Notes
- All write operations require both tenant context and attached policy evidence — two separate checks that must both pass
- The `agency_batch_requires_bytewax` guardrail enforces Bytewax as the only valid stream processor; direct DB writes bypass event ordering and are rejected
- Float sufficiency is a caller-computed context flag (`float_sufficient`); the rule engine does not query balances directly
- Condition keys ending in `_ne` invert the match: `event_stream_ne: "bytewax"` fires when the stream is NOT bytewax
- Supported countries and currencies are fixed compile-time lists; adding a new corridor requires updating the constant arrays and redeploying the contract
