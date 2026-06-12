# SACCO Dividend & Distribution (fintech_sacco_div)

Annual surplus calculation, dividend declaration, rebate computation, member distributions, and tax withholding.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/div/health | Service health |
| GET | /api/fintech/sacco/div/years | List financial years |
| POST | /api/fintech/sacco/div/years | Create financial year |
| GET | /api/fintech/sacco/div/years/{id} | Get year |
| PUT | /api/fintech/sacco/div/years/{id} | Update year |
| DELETE | /api/fintech/sacco/div/years/{id} | Cancel year |
| POST | /api/fintech/sacco/div/years/{id}/close | Close year |
| POST | /api/fintech/sacco/div/years/{id}/allocate | Allocate surplus |
| GET | /api/fintech/sacco/div/years/{id}/allocations | List allocations |
| GET | /api/fintech/sacco/div/years/{id}/report | Annual report |
| GET | /api/fintech/sacco/div/declarations | List declarations |
| POST | /api/fintech/sacco/div/declarations | Declare dividend |
| GET | /api/fintech/sacco/div/declarations/{id} | Get declaration |
| PUT | /api/fintech/sacco/div/declarations/{id} | Update declaration |
| POST | /api/fintech/sacco/div/declarations/{id}/reverse | Reverse declaration |
| GET | /api/fintech/sacco/div/declarations/{id}/summary | Declaration summary |
| POST | /api/fintech/sacco/div/declarations/{id}/pay-all | Run payment batch |
| GET | /api/fintech/sacco/div/distributions | List distributions |
| POST | /api/fintech/sacco/div/distributions/compute | Compute for member |
| POST | /api/fintech/sacco/div/distributions/bulk-compute | Bulk compute |
| GET | /api/fintech/sacco/div/distributions/{id} | Get distribution |
| POST | /api/fintech/sacco/div/distributions/{id}/pay | Pay distribution |
| POST | /api/fintech/sacco/div/distributions/{id}/reverse | Reverse distribution |
| POST | /api/fintech/sacco/div/wht | File WHT return |
| GET | /api/fintech/sacco/div/wht | List WHT records |
| GET | /api/fintech/sacco/div/members/{id}/history | Member dividend history |
| GET | /api/fintech/sacco/div/audit | Audit events |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Tiered Dividend Rate Engine | Business Logic | Flat rates disadvantage long-tenure members and depress loyalty; tiered rates by share bracket increase retention measurably. Equity Bank's SACCO arms use tiered 10/12/15% structures. | Implement `compute_tiered_dividend()` that accepts a rate schedule (list of `{min_shares, max_shares, rate_pct}` brackets) and applies the correct tier per member, storing the applied tier in the distribution record. | Equity Bank SACCO, Harambee SACCO Kenya** [Enhancement]
- **I2. Prorated Distributions for Mid-Year Members | Accuracy & Compliance | Members who join mid-year receive full-year dividends under current logic — a material mis-statement. SACCO Societies Regulations 2010 require pro-rata treatment. | Add `compute_prorated_distribution()` that accepts `membership_start_date` and `year_start_date` / `year_end_date`, computes a `proration_factor` (days_active / year_days), and applies it to both dividend and rebate before WHT. | Stima SACCO, Kenya Police SACCO** [Enhancement]
- **I3. Multi-Currency Distribution Support | Internationalisation | Diaspora SACCOs hold USD/EUR deposits alongside KES share capital; paying in a single currency creates FX exposure for the SACCO. | Add `declare_multicurrency_dividend()` with a `currency_rates` dict, store `currency` and `fx_rate_applied` on each distribution record, and compute net-payable in settlement currency. | Diaspora SACCO Uganda, East Africa SACCO platforms** [Enhancement]
- **I4. WHT Exemption Certificates | Tax Compliance | KRA issues WHT exemption certs (Form P9) to qualifying members; deducting 5% from exempt members violates the Income Tax Act. | Add `register_wht_exemption()` to record cert number, validity dates, and issuing authority per member; modify distribution computation to check exemptions and set `wht_rate_pct=0` when valid. | KRA iTax integration, Old Mutual Kenya SACCO** [Enhancement]
- **I5. Dividend Reinvestment Plan (DRIP) | Member Value | Members who elect DRIP compound wealth faster; SACCOs retain liquidity. Industry SACCOs with DRIP show 18% higher member equity growth. | Add `enroll_member_drip()` and `process_drip_elections()` that convert net payable into equivalent shares at par value, credit shares, and emit a `drip_reinvested` event instead of a cash payment. | Mwalimu SACCO, Co-op Bank SACCO** [Enhancement]
- **I6. Dispute & Correction Workflow | Operational Integrity | Distributed amounts are sometimes challenged; current reversal-only flow is blunt and creates negative audit trail. | Add `raise_distribution_dispute()`, `resolve_distribution_dispute()` with resolution types `{approved, rejected, partial_adjustment}`, maintaining full state machine: `pending → disputed → resolved → paid/reversed`. | South African SACCO Alliance, WOCCU best practices** [Enhancement]
- **I7. Regulatory Limit Enforcement | Compliance | SASRA caps dividend rates (currently 8% floor for statutory reserves, max declared dividend tied to surplus). Declaring beyond limits is a licensing risk. | Add `validate_declaration_against_regulatory_limits()` that checks: statutory_reserve >= 20% of surplus, dividend_pool <= 70% of surplus, rate_pct within SASRA-advised ceiling, returning a structured compliance report. | SASRA Kenya, USSD cooperative regulations** [Enhancement]
- **I8. Member Dividend Forecast | Member Experience | Members want forward-looking projections before the AGM to decide on share top-ups; this is standard in Scandinavian cooperative banks. | Add `forecast_member_dividend()` accepting projected surplus, pool percentages, and member data to produce an illustrative distribution estimate with sensitivity bands at ±10% surplus variance. | Nordea cooperative, DanBred SACCO model** [Enhancement]
- **I9. Payment Method Reconciliation | Financial Controls | Payment references from M-Pesa, bank transfers, and cheques must reconcile against third-party confirmation files to close the payment cycle. | Add `reconcile_payment_run()` that ingests a list of `{payment_reference, confirmed_amount, confirmed_at}` records, matches against distributions, marks `reconciled` or `reconciliation_mismatch`, and returns a reconciliation report. | Equity Bank reconciliation module, Safaricom bulk disbursement API** [Enhancement]
- **I10. Unclaimed Dividends Escrow | Dormancy Management | Unclaimed dividends become dormant liabilities; Kenya's Unclaimed Financial Assets Act requires escalation to UFAA after 5 years. | Add `flag_unclaimed_distributions()` that marks distributions unpaid beyond a configurable threshold as `unclaimed`, computes total escrow liability, and generates a UFAA submission report. | UFAA Kenya Act 2011, CBA SACCO compliance** [Enhancement]
- **I11. Batch Progress Tracking & Resumability | Operational Resilience | Large payment batches (5,000+ members) fail midway with no resume capability, forcing full re-runs and double-payments. | Add `create_payment_run_checkpoint()` that serialises run state (cursor position, totals) and `resume_payment_run()` that picks up from the last checkpoint, skipping already-paid distributions using idempotency keys. | KWFT SACCO operations, SASRA operational guidelines** [Enhancement]
- **I12. Comparative Year-on-Year Analytics | Governance | Boards need YoY trend analysis for AGM presentations; currently each year is siloed. | Add `generate_yoy_comparison()` that accepts a list of year_ids and returns a structured table of surplus, pools, rates, members paid, and total disbursed per year with percentage-change columns. | Mwalimu National SACCO annual reports, WOCCU PEARLS model** [Enhancement]
- **I13. Member Statement Generation | Regulatory Communication | SASRA requires member dividend statements be issued within 30 days of payment; current export is declaration-level, not member-level. | Add `generate_member_statement()` that produces a structured per-member statement including year, share capital, savings balance, gross dividend, gross rebate, WHT cert details, net received, and payment reference — suitable for PDF rendering. | Kenya Revenue Authority P9 statement format, Co-op Bank member portal** [Enhancement]
- **I14. Board Resolution Workflow | Governance Audit | The current `board_resolution_ref` is a free-text field with no validation chain; external auditors require a traceable approval workflow. | Add `create_board_resolution()` with fields for resolution_number, meeting_date, quorum_count, votes_for, votes_against, and `approve_board_resolution()` requiring a minimum quorum; link declarations to validated resolution IDs only. | ICPAK corporate governance guidelines, CMA Kenya SACCO listing rules** [Enhancement]
- **I15. Interest Income Allocation Tracing | Audit Trail | Loan interest income (typically 60-80% of SACCO income) should be traced to the dividend pool to satisfy SASRA Form 2 requirements; current surplus is a single undifferentiated figure. | Add `record_income_component()` to break total income into typed streams (`{loan_interest, investment_income, fee_income, other_income}`) and update `annual_report()` to include income composition and percentage contribution to surplus. | SASRA Form 2 reporting, Stima SACCO audited accounts structure** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
