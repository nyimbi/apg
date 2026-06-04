# Decentralized Finance

## Overview
Decentralized Finance provides governed operations over DeFi protocols: protocol registry, position management (supply, borrow, liquidity, stake, vault share), action execution workflow (deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance), yield strategy management, reward accruals, governance voting, risk tier assessments, and reviews. Every action against a DeFi protocol requires an approval reference before it is recorded, enforcing human oversight over autonomous on-chain interactions.

Position-protocol consistency is enforced — actions on a position must reference the same protocol the position belongs to. Health factors must be positive on position opening. All DeFi lifecycle events stream to `apg.fintech.defi.lifecycle` via Bytewax.

## Capability ID
`fintech_defi`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| defi_protocol_workflow | Register DeFi protocols with type, network reference, risk tier, owner, and evidence |
| defi_position_workflow | Open positions with protocol, account, asset pair, type, amount, collateral, and health factor |
| defi_action_workflow | Record protocol actions with type, amount, requester, approval, and status |
| defi_yield_strategy_workflow | Register yield strategies with target APY, max risk tier, owner, and evidence |
| defi_reward_workflow | Record protocol rewards with type, asset, amount, and evidence |
| defi_governance_workflow | Record governance votes with protocol, proposal, vote choice, voter, and evidence |
| defi_risk_workflow | Record risk assessments with tier, reviewer, and evidence |
| defi_review_workflow | Governance reviews for protocols, positions, and strategy decisions |
| defi_agent_workflow | Register AI agents for protocol monitoring, position reconciliation, and liquidity risk |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_blockchain | Network and protocol infrastructure |
| fintech_crypto | Asset and custody backing |
| fintech_wallets | Wallet references for positions |
| fintech_risk | Risk tier and appetite context |
| fintech_compliance | Compliance obligation evidence |
| fintech_regtech | Regulatory framework context |
| fintech_aml | AML screening for DeFi activity |
| fintech_kyc | Identity verification for account holders |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| protocols.supported_protocol_types | list | lending_pool, liquidity_pool, staking, yield_vault, dex, bridge, derivatives, insurance_pool | Protocol categories |
| protocols.supported_risk_tiers | list | low, medium, high, critical | Risk tier classification |
| positions.supported_position_types | list | supply, borrow, liquidity, stake, vault_share, short, long, cover | Position types |
| actions.supported_action_types | list | deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance | Action types |
| governance.supported_vote_choices | list | for, against, abstain | Valid vote choices |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-defi/dashboard | GET | fintech_defi:view | Overview |
| protocols | /fintech-defi/protocols | GET/POST | fintech_defi:protocols | Protocols |
| positions | /fintech-defi/positions | GET/POST | fintech_defi:positions | Portfolio |
| actions | /fintech-defi/actions | GET/POST | fintech_defi:actions | Operations |
| yield_strategies | /fintech-defi/yield-strategies | GET/POST | fintech_defi:yield | Strategies |
| rewards | /fintech-defi/rewards | GET/POST | fintech_defi:rewards | Portfolio |
| governance | /fintech-defi/governance | GET/POST | fintech_defi:governance | Governance |
| risk | /fintech-defi/risk | GET/POST | fintech_defi:risk | Risk |
| reviews | /fintech-defi/reviews | GET/POST | fintech_defi:reviews | Governance |
| agents | /fintech-defi/agents | GET/POST | fintech_defi:admin | Automation |
| settings | /fintech-defi/settings | GET/POST | fintech_defi:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| protocol_risk_supported | Unsupported risk tier on protocol | deny |
| position_health_factor_valid | Negative or zero health factor | deny |
| position_collateral_valid | Negative collateral amount | deny |
| action_position_protocol_match | Action position belongs to different protocol | deny |
| action_approval_required | Action without approval reference | deny |
| strategy_target_apy_valid | Negative target APY | deny |
| governance_vote_supported | Unsupported vote choice | deny |
| defi_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_defi_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| DeFiProtocol | id, protocol_type, network_reference, protocol_reference, risk_tier, owner_id, evidence_reference |
| DeFiPosition | id, protocol_id, account_reference, asset_pair_reference, position_type, amount, collateral, health_factor, evidence_reference |
| DeFiAction | id, protocol_id, position_id, action_type, amount, requester_id, approval_reference, status, evidence_reference |
| YieldStrategy | id, protocol_id, strategy_reference, target_apy, max_risk_tier, owner_id, evidence_reference |
| DeFiReward | id, position_id, reward_type, asset_reference, amount, evidence_reference |
| GovernanceVote | id, protocol_id, proposal_reference, vote_choice, voter_id, evidence_reference |
| DeFiRiskAssessment | id, reference, risk_tier, reviewer_id, evidence_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| defi_protocol_registered | Protocol registered |
| defi_position_opened | Position opened |
| defi_action_recorded | Protocol action recorded |
| defi_yield_strategy_registered | Yield strategy registered |
| defi_reward_recorded | Reward accrual recorded |
| defi_governance_vote_recorded | Governance vote cast |
| defi_risk_assessment_recorded | Risk assessment recorded |
| defi_review_recorded | Governance review completed |
| defi_agent_registered | AI agent registered |

## Edge Cases Handled
- Protocol-position consistency: an action on a position (e.g., withdraw) is rejected if the `position_protocol_match` flag is false — the action must target the same protocol the position belongs to
- Health factor must be strictly positive at position opening; a health factor of zero implies immediate liquidation eligibility and is rejected
- Collateral is non-negative (zero collateral is valid for unsecured positions); negative collateral is rejected
- Target APY on yield strategies must be non-negative; a strategy targeting negative returns would be financially absurd and is rejected
- Every DeFi action (even a claim of rewards) requires an approval reference — autonomous on-chain execution without human oversight is blocked

## Composability
- **Upstream**: `fintech_blockchain` provides the network infrastructure; `fintech_crypto` provides custody accounts and asset references backing DeFi positions
- **Downstream**: `fintech_portfolio` consumes position and balance data for portfolio-level tracking; `fintech_risk` receives DeFi risk assessments as exposure records
- **Peer**: Deployed alongside `fintech_crypto` (asset operations) and `fintech_blockchain` (network layer) in a full digital asset stack

## Development Notes
- `SUPPORTED_PROTOCOL_TYPES` maps to DeFi primitive categories; `insurance_pool` is included for on-chain coverage products, not traditional insurance
- The `action_position_protocol_match` rule fires when `position_protocol_match: False` is set in context — callers must compute this before invoking the rule engine
- Governance votes use `for/against/abstain` which maps to standard on-chain governance; `abstain` is a valid choice and counts toward quorum in most protocols
- Health factor convention follows Aave/Compound pattern: values > 1.0 are solvent, < 1.0 subject to liquidation; the rule only checks positivity, not the 1.0 threshold
