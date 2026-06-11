# FOSA User Guide

## What Is FOSA?

The Front Office Service Activity (FOSA) is the transactional banking division of your SACCO. While BOSA manages long-term savings and loan portfolios, FOSA gives members a current-account experience: instant deposits, withdrawals, M-PESA integration, ATM cards, and automated standing orders — all under the SACCO umbrella.

---

## Account Types

| Type | Use Case | Typical Features |
|---|---|---|
| **CURRENT** | Day-to-day transactions | Unlimited deposits/withdrawals, ATM card, M-PESA |
| **SALARY** | Salary processing | Employer payroll credit, automatic loan deductions |
| **FIXED_DEPOSIT** | Term savings | Higher interest rate, maturity date, limited withdrawals |

---

## Opening a FOSA Account

**Prerequisites:**
- Active SACCO membership
- KYC documents submitted and verified
- Joining fee paid

**Steps:**
1. Visit FOSA teller or use the member portal
2. Select account type
3. Deposit opening balance (KES 0 minimum for CURRENT)
4. Receive account number in format `FOSA-CUR-XXXX-00000001`

---

## Deposits

Funds can be deposited through:

| Channel | How | Processing Time |
|---|---|---|
| **Teller** | Cash at counter | Instant |
| **M-PESA** | Pay bill / send money | Instant on confirmation |
| **Bank Transfer** | EFT / RTGS | Same day to next business day |

---

## Withdrawals

**Daily Limits** (defaults, adjustable by SACCO management):
- Withdrawal: KES 100,000/day
- Transfer: KES 200,000/day

Withdrawal channels: Teller counter, ATM, M-PESA B2C.

Accounts that are **frozen** cannot process withdrawals until unfrozen by management.

---

## M-PESA Integration

### Depositing via M-PESA (Cash-In)
Send money to the SACCO's M-PESA paybill/till using your account number as the reference. The system confirms via Safaricom Daraja C2B callback and credits your account instantly. Duplicate M-PESA references are rejected automatically (idempotent).

### Withdrawing via M-PESA (Cash-Out)
Request a B2C disbursement to your registered phone number. Funds are deducted immediately and sent via Safaricom B2C API.

---

## ATM Cards

Supported card schemes: VISA, Mastercard, Prepaid.

- One active card per scheme per account
- Cards valid for 3 years from issue date
- Card name embossed from account name (max 26 characters)

**Blocking a card:** Report lost/stolen to teller or call member services. Instant block applied.  
**Unblocking:** Requires authorization from branch manager or above.

---

## BOSA Transfers

Move funds between your FOSA and BOSA accounts:

| Direction | Approval Required |
|---|---|
| FOSA → BOSA | Never (member-initiated) |
| BOSA → FOSA | Only if amount > KES 50,000 |

---

## Standing Orders

Automate recurring payments (e.g., monthly savings top-up, rent, loan contributions):

| Frequency | Description |
|---|---|
| daily | Every calendar day |
| weekly | Every 7 days |
| biweekly | Every 14 days |
| monthly | Same day each month |
| quarterly | Every ~91 days |
| annually | Every 365 days |

Standing orders are processed by the nightly batch job. Failed executions (insufficient funds) increment the failure counter but do not cancel the order.

---

## Overdraft Facility

1. Member submits overdraft request with purpose and desired amount
2. Credit committee reviews and approves/declines
3. On approval, the overdraft limit is set on the account
4. Available balance = book_balance + overdraft_limit - overdraft_used
5. Overdraft expires on the set date; renewal requires fresh application

---

## Account Dormancy

An account inactive for **6 consecutive months** is flagged dormant. Dormant accounts:
- Cannot process normal transactions
- Appear in the dormancy report
- Require a **reactivation deposit** (default KES 500) to restore

---

## Mini Statement

The last 10 transactions (or N as requested) sorted newest-first. Available at teller counter, member portal, or ATM.

---

## Full Statement

Date-range statement showing:
- Opening balance
- All debits and credits
- Closing balance
- Total credits / total debits

Available in portal, or request at teller for printed copy.

---

## Teller End-of-Day

Each teller's cash position is tracked:
- Opening float (configured per teller)
- Total deposits received
- Total withdrawals paid out
- Closing float = opening + deposits − withdrawals
- Variance = closing float − physical cash count

---

## Contact & Support

- **SACCO helpline**: per your SACCO's configured contact
- **M-PESA disputes**: quote the `mpesa_reference` from your confirmation SMS
- **Card disputes**: quote the masked card number and transaction date
