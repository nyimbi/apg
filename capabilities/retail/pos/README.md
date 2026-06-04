# Point of Sale

## Overview
Provides complete POS transaction processing for physical retail: terminal registration and heartbeat monitoring, session lifecycle management with opening float and reconciliation enforcement, sale/refund/void/exchange transaction posting with automated total calculation and digital signing, cash event recording with safe-drop thresholds, till reconciliation with variance reporting and approval, multi-format receipt issuance, and full offline resilience with configurable floor limits and store-and-forward queuing.

## Capability ID
`retail_pos`

## Provides
| Service | Description |
|---|---|
| pos_transaction_processing | Sale, refund, exchange, and no-sale posting |
| pos_session_management | Open/suspend/resume/close session lifecycle |
| pos_cash_management | Float, petty cash, safe drop, and pickup events |
| pos_till_reconciliation | System vs counted cash variance with approval |
| pos_receipt_management | Print, email, SMS, and digital wallet receipts |
| pos_discount_management | Percentage, fixed, BOGO, and manager override discounts |
| pos_offline_resilience | Floor limits, store-and-forward, and offline loyalty |
| pos_payment_processing | Card, cash, mobile money, gift card, loyalty points |
| pos_void_management | Same-terminal void within configurable window |
| pos_audit_trail | Signed transaction ledger with session context |

## Requires
| Capability | Reason |
|---|---|
| auth | Cashier authentication and manager override validation |
| audl | Signed transaction and cash event audit trail |
| mten | Tenant context isolation per store/terminal |
| conf | Terminal, session, and payment configuration |
| ntfy | Session variance and safe-drop threshold alerts |
| mqeb | Bytewax stream for offline transaction sync |
| moni | Terminal heartbeat and session health monitoring |
| comp | PCI DSS and fiscal compliance |

## Configuration
| Key | Default | Description |
|---|---|---|
| transactions.void_window_minutes | 30 | Void eligibility window |
| transactions.manager_override_required_above | 5,000 | Override threshold |
| transactions.max_value_per_transaction | 1,000,000 | Hard transaction cap |
| cash.safe_drop_threshold | 50,000 | Auto-prompt safe drop |
| cash.starting_float_required | true | Float required on session open |
| sessions.reconciliation_required_on_close | true | Reconcile before close |
| offline.floor_limit | 5,000 | Max offline transaction value |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-pos/api/v1/terminals | GET/POST | List/register terminals | retail_pos:admin |
| /retail-pos/api/v1/terminals/<id>/heartbeat | POST | Terminal heartbeat | retail_pos:transact |
| /retail-pos/api/v1/sessions | GET/POST | List/open sessions | retail_pos:view/transact |
| /retail-pos/api/v1/sessions/<id> | GET/PUT | Session summary/status update | retail_pos:view/transact |
| /retail-pos/api/v1/transactions | GET/POST | List/post transactions | retail_pos:view/transact |
| /retail-pos/api/v1/transactions/<id> | GET/DELETE | Get/void transaction | retail_pos:view/void |
| /retail-pos/api/v1/voids | POST | Post void | retail_pos:void |
| /retail-pos/api/v1/cash | GET/POST | List/record cash events | retail_pos:view/transact |
| /retail-pos/api/v1/reconcile | POST | Create reconciliation | retail_pos:reconcile |
| /retail-pos/api/v1/reconcile/<id>/approve | PUT | Approve reconciliation | retail_pos:admin |
| /retail-pos/api/v1/receipts | GET/POST | List/issue receipts | retail_pos:view/transact |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| transaction_requires_open_session | session status=closed | deny |
| unsigned_transaction_denied | transaction_signed=False | deny |
| void_requires_reason | void without reason | deny |
| cross_terminal_void_denied | void on different terminal | deny |
| void_window_expired | outside void window | deny |
| discount_exceeds_max_denied | discount > max_pct | deny |
| large_discount_requires_manager | above threshold, no override | deny |
| session_reconciliation_required | close without reconcile | deny |
| unreconciled_carry_over_denied | open with prior unreconciled | deny |
| offline_floor_limit_enforced | offline + exceeds limit | deny |
| safe_drop_required | cash > safe_drop_threshold | deny |

## Data Models
| Model | Key Fields |
|---|---|
| PosTerminalResponse | id, terminal_code, terminal_type, status |
| PosSessionResponse | id, session_number, status, opening_float, total_sales |
| PosTransactionResponse | id, transaction_number, transaction_type, grand_total, transaction_signed |
| PosTransactionLineItem | sku, quantity, unit_price, line_total, tax_amount |
| PosCashEventResponse | id, cash_event_type, amount, balance_after |
| PosReconciliationResponse | id, variance, status, approved_by |
| PosReceiptResponse | id, receipt_number, receipt_type |
| PosVoidResponse | id, void_reason, status |

## Streaming Events
- `session_opened`, `session_closed`
- `transaction_posted`, `refund_posted`, `void_posted`
- `cash_event_recorded`, `reconciliation_completed`
- `terminal_offline`, `terminal_online`
- `discount_applied`, `manager_override_recorded`

## Edge Cases Handled
- Duplicate session open on same terminal: blocked by open session check
- Void outside time window: redirected to refund
- Cross-terminal void: denied to prevent fraud
- Cash event yielding negative till balance: assertion blocks
- Unreconciled session carry-over: new open blocked
- Offline floor limit: requires manual authorisation code above limit
- Safe drop threshold exceeded: transaction blocked until drop performed

## Composability Notes
- **retail_loy** earn/redeem triggered at transaction posting
- **retail_prm** discounts applied to transaction line items before posting
- **retail_omc** inventory reservation released or confirmed on sale
- **retail_sin** conversion events derived from POS session close
