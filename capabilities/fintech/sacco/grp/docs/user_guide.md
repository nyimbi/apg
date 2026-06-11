# SACCO Group Lending — User Guide

## Overview

Group lending pools savings and distributes credit to the group collectively.
Members bear **joint liability**: if one member defaults, the rest are expected
to cover the shortfall. This social collateral mechanism is the primary credit
control in the absence of formal collateral.

Supported group structures:

| Structure | Lending model |
|-----------|---------------|
| **Chama** | Group applies for a single loan; each member receives a share; group repays jointly |
| **Welfare** | Emergency or welfare loans disbursed from pooled contributions; no external borrowing required |
| **Merry-Go-Round (MGR)** | No loan — contributions are collected and handed to one member per cycle on a rotating basis |
| **Investment Club** | Members invest pooled savings; credit may be offered against the portfolio |

---

## Workflow: Chama / Welfare Group Loan

### 1. Register the Group

```http
POST /api/fintech/sacco/grp/groups
X-Tenant-ID: my-sacco

{
  "name": "Umoja Chama",
  "group_type": "CHAMA",
  "registration_number": "CHM-2024-001",
  "meeting_day": "Monday",
  "meeting_frequency": "MONTHLY"
}
```

### 2. Add Members

```http
POST /api/fintech/sacco/grp/groups/{group_id}/members
{
  "member_id": "mbr-001",
  "role": "CHAIRPERSON",
  "joining_date": "2024-01-15",
  "initial_contribution": "500.00"
}
```

Roles: `CHAIRPERSON`, `SECRETARY`, `TREASURER`, `MEMBER`.

### 3. Record Monthly Contributions

```http
POST /api/fintech/sacco/grp/groups/{group_id}/contributions
{
  "contribution_type": "MONTHLY",
  "meeting_date": "2024-02-05",
  "contributions": [
    {"member_id": "mbr-001", "amount": "1000.00"},
    {"member_id": "mbr-002", "amount": "1000.00"},
    {"member_id": "mbr-003", "amount": "1000.00"}
  ]
}
```

### 4. Apply for a Group Loan

All **active** group members automatically become joint borrowers.

```http
POST /api/fintech/sacco/grp/loans
{
  "group_id": "{group_id}",
  "requested_amount": "90000.00",
  "purpose": "Stock purchase for members",
  "tenure_months": 12,
  "applied_by": "mbr-001"
}
```

A group may hold only **one active loan** at a time.

### 5. Approve the Loan

```http
POST /api/fintech/sacco/grp/loans/{loan_id}/approve
{
  "approved_amount": "90000.00",
  "approved_by": "officer-jane",
  "conditions": "Must maintain monthly contributions"
}
```

### 6. Disburse to Individual Members

Each member receives their share directly into their nominated account.
The sum of disbursement amounts must equal `approved_amount`.

```http
POST /api/fintech/sacco/grp/loans/{loan_id}/disburse
{
  "disbursement_instructions": [
    {"member_id": "mbr-001", "amount": "30000.00", "account_id": "acc-001"},
    {"member_id": "mbr-002", "amount": "30000.00", "account_id": "acc-002"},
    {"member_id": "mbr-003", "amount": "30000.00", "account_id": "acc-003"}
  ]
}
```

### 7. Record Repayments

Track which member contributed to each installment.

```http
POST /api/fintech/sacco/grp/loans/{loan_id}/repayments
{
  "total_amount": "9000.00",
  "payment_date": "2024-03-05",
  "payment_ref": "MPE-20240305",
  "member_contributions": [
    {"member_id": "mbr-001", "amount": "3000.00"},
    {"member_id": "mbr-002", "amount": "3000.00"},
    {"member_id": "mbr-003", "amount": "3000.00"}
  ]
}
```

### 8. Monitor Arrears and Default

Check the group arrears position:

```http
GET /api/fintech/sacco/grp/loans/{loan_id}/arrears?as_of_date=2024-06-01
```

Get members who have not contributed to any repayment:

```http
GET /api/fintech/sacco/grp/loans/{loan_id}/defaulting-members
```

Invoke joint liability — notifies other members that they must cover the defaulter's share:

```http
POST /api/fintech/sacco/grp/loans/{loan_id}/joint-liability
{
  "defaulting_member_id": "mbr-002"
}
```

---

## Workflow: Merry-Go-Round (MGR)

### 1. Register MGR Group

```http
POST /api/fintech/sacco/grp/groups
{
  "name": "Tumaini MGR",
  "group_type": "MERRY_GO_ROUND",
  "meeting_frequency": "MONTHLY"
}
```

### 2. Add Members and Set Rotation Order

```http
PUT /api/fintech/sacco/grp/groups/{group_id}/mgr/order
{
  "member_order": ["mbr-001", "mbr-002", "mbr-003", "mbr-004"]
}
```

### 3. Record Baseline Contribution Amounts

Use a standard `MERRY_GO_ROUND` contribution record so the system knows each
member's contribution amount per cycle.

```http
POST /api/fintech/sacco/grp/groups/{group_id}/contributions
{
  "contribution_type": "MERRY_GO_ROUND",
  "meeting_date": "2024-01-08",
  "contributions": [
    {"member_id": "mbr-001", "amount": "2000.00"},
    {"member_id": "mbr-002", "amount": "2000.00"},
    {"member_id": "mbr-003", "amount": "2000.00"},
    {"member_id": "mbr-004", "amount": "2000.00"}
  ]
}
```

### 4. Process Each Round

The beneficiary does not contribute in their own round. The total collected
is disbursed to them.

```http
POST /api/fintech/sacco/grp/groups/{group_id}/mgr/process
{
  "round_date": "2024-02-05",
  "beneficiary_member_id": "mbr-001"
}
```

Response includes `total_collected` and `next_beneficiary_member_id`.

### 5. View Schedule

```http
GET /api/fintech/sacco/grp/groups/{group_id}/mgr/schedule
```

---

## Performance Score

Scores from 0 to 100. Weighted equally between:

- **Repayment rate** (total repaid / total disbursed)
- **Contribution compliance** (sessions attended / total sessions)

| Score | Grade | Meaning |
|-------|-------|---------|
| 90–100 | A | Excellent |
| 75–89 | B | Good |
| 55–74 | C | Fair |
| 35–54 | D | Poor |
| 0–34 | E | Critical |

```http
GET /api/fintech/sacco/grp/groups/{group_id}/performance
```

---

## Group Statement

Full chronological ledger — contributions, disbursements, repayments — with
running balance.

```http
GET /api/fintech/sacco/grp/groups/{group_id}/statement?from_date=2024-01-01&to_date=2024-12-31
```

---

## Member Exit

A member cannot exit while a group loan is active.

```http
DELETE /api/fintech/sacco/grp/groups/{group_id}/members/{member_id}
{
  "exit_date": "2024-06-30",
  "reason": "Relocation",
  "payout_amount": "15000.00"
}
```

---

## Composition Keywords

- `group_loan` — apply / approve / disburse / repay cycle
- `joint_liability` — trigger coverage call on group members
- `merry_go_round` — rotating kitty disbursement
- `contribution_compliance` — member payment discipline reporting
- `group_performance` — aggregated score for credit decisioning
