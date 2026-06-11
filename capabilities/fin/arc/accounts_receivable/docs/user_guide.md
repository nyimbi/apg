# APG Accounts Receivable - User Guide

**AI-Powered Accounts Receivable Management for the APG Platform**

Version 1.0 | © 2025 Datacraft | Author: Nyimbi Odero

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Customer Management](#customer-management)
4. [Invoice Management](#invoice-management)
5. [Payment Processing](#payment-processing)
6. [Collections Management](#collections-management)
7. [AI-Powered Features](#ai-powered-features)
8. [Analytics & Reporting](#analytics--reporting)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The APG Accounts Receivable capability provides comprehensive AR automation with AI-powered insights, designed for enterprise-scale financial operations. Built on the APG platform, it integrates seamlessly with existing business processes while providing advanced automation and intelligence.

### Key Features

🤖 **AI-Powered Operations**
- Automated credit scoring using federated learning
- Intelligent collections optimization
- Predictive cash flow forecasting
- Risk assessment and monitoring

💼 **Complete AR Workflow**
- Customer lifecycle management
- Invoice creation and tracking
- Payment processing and application
- Collections automation
- Dispute resolution

📊 **Advanced Analytics**
- Real-time AR dashboards
- Aging analysis and reports
- Performance metrics and KPIs
- Predictive insights

🔧 **Enterprise Integration**
- Multi-tenant architecture
- Role-based access control
- Audit trails and compliance
- API-first design

---

## Getting Started

### Prerequisites

- Access to APG platform with AR capability enabled
- Appropriate user permissions (ar:read, ar:write, ar:ai)
- Web browser (Chrome, Firefox, Safari, Edge)

### Initial Setup

1. **Login to APG Platform**
   ```
   Navigate to your APG instance and authenticate
   ```

2. **Access AR Module**
   - From the main menu, select "Financial Management"
   - Click on "Accounts Receivable"

3. **Dashboard Overview**
   - Review the AR dashboard for current metrics
   - Familiarize yourself with the navigation menu
   - Check permissions and available features

### Navigation Menu

- **Dashboard**: Overview metrics and KPIs
- **Customers**: Customer management and profiles
- **Invoices**: Invoice creation and tracking
- **Payments**: Payment processing and history
- **Collections**: Collections activities and optimization
- **Analytics**: Detailed reports and insights
- **AI Tools**: Credit scoring, forecasting, optimization

---

## Customer Management

### Creating Customers

1. **Navigate to Customers**
   - Click "Customers" in the main menu
   - Select "Add New Customer"

2. **Customer Information**
   ```
   Customer Code: ACME001 (unique identifier)
   Legal Name: ACME Corporation
   Display Name: ACME Corp (optional)
   Customer Type: Corporation/Individual/Partnership/Government
   Status: Active/Inactive/Suspended/Closed
   ```

3. **Credit Information**
   ```
   Credit Limit: $50,000.00
   Payment Terms: 30 days
   Currency: USD
   ```

4. **Contact Information**
   ```
   Contact Email: billing@acme.com
   Contact Phone: +1-555-123-4567
   Billing Address: 123 Business St, City, ST 12345
   ```

5. **Save and Review**
   - Click "Save Customer"
   - Review the customer profile
   - Note the automatically assigned customer ID

### Customer Profiles

Each customer profile displays:

**Basic Information**
- Customer details and contact information
- Current credit limit and utilization
- Payment terms and status

**Financial Summary**
- Total outstanding balance
- Overdue amounts
- Payment history summary
- Credit score (if assessed)

**Recent Activity**
- Recent invoices and payments
- Collection activities
- Communication history

**AI Insights** (if available)
- Credit risk assessment
- Payment prediction
- Recommended actions

### Managing Customer Records

**Editing Customer Information**
1. Navigate to customer profile
2. Click "Edit Customer"
3. Update required fields
4. Save changes

**Credit Limit Management**
1. Go to customer profile
2. Click "Manage Credit"
3. Review current utilization
4. Update credit limit (requires approval)
5. Document reason for change

**Customer Status Changes**
- **Active**: Normal business operations
- **Inactive**: Temporarily disabled
- **Suspended**: Credit hold applied
- **Closed**: Account permanently closed

---

## Invoice Management

### Creating Invoices

1. **New Invoice**
   - Navigate to "Invoices" → "Create Invoice"
   - Select customer from dropdown
   - System validates credit availability

2. **Invoice Details**
   ```
   Invoice Number: INV-2025-001 (auto-generated)
   Invoice Date: Today's date
   Due Date: Based on customer payment terms
   Description: Goods/services description
   ```

3. **Amount Information**
   ```
   Total Amount: $10,000.00
   Currency: USD (inherited from customer)
   Tax Information: (if applicable)
   ```

4. **Review and Submit**
   - Verify all information
   - Check credit limit compliance
   - Submit for processing

### Invoice Lifecycle

**Draft Status**
- Invoice created but not sent
- Can be edited or deleted
- Not included in AR balances

**Sent Status**
- Invoice delivered to customer
- Included in AR aging
- Payment tracking begins

**Paid Status**
- Full payment received
- Invoice closed
- Archived for reporting

**Overdue Status**
- Past due date with outstanding balance
- Eligible for collections
- Risk indicators activated

**Cancelled Status**
- Invoice voided
- Removed from AR balances
- Audit trail maintained

### Bulk Invoice Operations

**Batch Creation**
1. Navigate to "Invoices" → "Bulk Import"
2. Download template spreadsheet
3. Fill in invoice data
4. Upload and validate
5. Review and approve batch

**Batch Actions**
- Send multiple invoices
- Apply payments to multiple invoices
- Update invoice statuses
- Generate batch reports

---

## Payment Processing

### Recording Payments

1. **Manual Payment Entry**
   - Navigate to "Payments" → "Record Payment"
   - Select customer
   - Enter payment details

2. **Payment Information**
   ```
   Payment Reference: PAY-2025-001
   Payment Date: Date received
   Payment Amount: $8,000.00
   Payment Method: Check/Wire/Credit Card/ACH
   Bank Reference: (if applicable)
   ```

3. **Payment Application**
   - System suggests invoice application
   - Review and adjust allocations
   - Apply to specific invoices
   - Handle overpayments/underpayments

### Automated Payment Processing

**Bank File Integration**
- Configure bank file formats
- Set up automated import schedules
- Review and approve matches
- Handle exceptions and unmatched items

**Payment Matching**
- Automatic matching by reference numbers
- Fuzzy matching for partial references
- Manual matching for exceptions
- Dispute resolution workflow

### Payment Analytics

**Payment Performance**
- Average days to pay
- Payment method preferences
- Seasonal payment patterns
- Customer payment reliability

**Cash Application Metrics**
- Matching accuracy rates
- Processing times
- Exception rates
- Automation effectiveness

---

## Collections Management

### Collections Workflow

1. **Automatic Identification**
   - System identifies overdue invoices
   - Applies aging categories (1-30, 31-60, 61-90, 90+ days)
   - Prioritizes by amount and customer risk

2. **Collections Activities**
   - Email reminders (automated)
   - Phone call scheduling
   - Letter generation
   - Legal action tracking

3. **Activity Documentation**
   ```
   Activity Type: Phone Call/Email/Letter/Meeting
   Contact Method: Phone/Email/Mail/In-Person
   Outcome: Promised Payment/Dispute/No Response
   Follow-up Date: Next scheduled contact
   Notes: Detailed conversation notes
   ```

### Collections Optimization

**AI-Powered Strategy**
1. Navigate to "Collections" → "AI Optimization"
2. Select customers or use batch mode
3. Review AI recommendations
4. Execute optimized strategies

**Strategy Types**
- **Email Sequence**: Automated email campaigns
- **Phone Campaigns**: Structured calling programs
- **Payment Plans**: Negotiated payment arrangements
- **Legal Action**: Escalation to legal proceedings

**Success Tracking**
- Collection success rates by strategy
- Time to resolution
- Cost per collection
- Customer satisfaction impact

---

## AI-Powered Features

### Credit Scoring

**Individual Assessment**
1. Navigate to customer profile
2. Click "AI Credit Assessment"
3. Configure assessment parameters
4. Review results and recommendations

**Assessment Results**
```
Credit Score: 720 (300-850 scale)
Risk Level: Medium
Confidence: 85%
Recommended Credit Limit: $75,000
Payment Prediction: 32 days average
```

**Batch Assessment**
1. Navigate to "AI Tools" → "Credit Assessment"
2. Select multiple customers
3. Configure batch parameters
4. Review batch results
5. Apply recommendations

### Collections Optimization

**Strategy Optimization**
1. Access "Collections" → "AI Optimization"
2. Select optimization scope (single/batch/campaign)
3. Choose scenario type (realistic/optimistic/pessimistic)
4. Review strategy recommendations
5. Execute approved strategies

**Optimization Results**
```
Recommended Strategy: Email Sequence
Success Probability: 72%
Estimated Resolution: 12 days
Priority Level: Medium
Cost Estimate: $15
```

### Cash Flow Forecasting

**Forecast Generation**
1. Navigate to "Analytics" → "Cash Flow Forecast"
2. Set forecast parameters
   ```
   Start Date: Today
   End Date: 90 days forward
   Period: Daily/Weekly/Monthly
   Scenario: Realistic/Optimistic/Pessimistic
   ```
3. Include seasonal trends and external factors
4. Generate forecast
5. Export results

**Forecast Analysis**
- Expected collections by period
- Confidence intervals
- Risk factors and opportunities
- Scenario comparisons
- Liquidity recommendations

---

## Analytics & Reporting

### AR Dashboard

**Key Metrics Display**
- Total AR Balance: $450,000
- Overdue Amount: $125,000
- Current Month Collections: $92,000
- Average Days to Pay: 31.5 days
- Collection Effectiveness: 85%

**Visual Analytics**
- AR aging charts
- Collection trend graphs
- Customer performance metrics
- Payment pattern analysis

### Standard Reports

**Aging Report**
- Current, 1-30, 31-60, 61-90, 90+ day buckets
- Customer and invoice details
- Percentage distributions
- Trend analysis

**Collection Performance Report**
- Activity summaries by collector
- Success rates by strategy
- Resolution times
- Cost analysis

**Cash Flow Report**
- Historical cash flow patterns
- Forecast vs. actual analysis
- Seasonal adjustments
- Variance explanations

### Custom Reports

**Report Builder**
1. Navigate to "Analytics" → "Custom Reports"
2. Select data sources and fields
3. Apply filters and groupings
4. Configure visualization options
5. Save and schedule reports

**Export Options**
- PDF for presentation
- Excel for analysis
- CSV for data integration
- Dashboard integration

---

## Advanced Features

### Multi-Tenant Operations

**Tenant Management**
- Separate AR data by tenant/entity
- Cross-tenant reporting (if permitted)
- Tenant-specific configurations
- Data isolation and security

**User Permissions**
- Role-based access control
- Field-level security
- Action permissions (read/write/delete)
- AI feature access control

### API Integration

**REST API Access**
- Full CRUD operations
- Bulk data operations
- Real-time integrations
- Webhook notifications

**Common Integration Patterns**
```python
# Customer creation via API
import requests

customer_data = {
    "customer_code": "API001",
    "legal_name": "API Test Customer",
    "credit_limit": 25000.00
}

response = requests.post(
    "https://api.apg.platform/ar/customers",
    json=customer_data,
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### Workflow Automation

**Business Rules Engine**
- Automated credit limit adjustments
- Collections escalation rules
- Payment application logic
- Exception handling workflows

**Notification System**
- Email alerts for overdue accounts
- Dashboard notifications
- Mobile push notifications
- Slack/Teams integration

### Compliance and Auditing

**Audit Trails**
- Complete activity logging
- User action tracking
- Data change history
- Compliance reporting

**Regulatory Compliance**
- GDPR data protection
- SOX financial controls
- Industry-specific requirements
- Data retention policies

---

## Troubleshooting

### Common Issues

**Login and Access Problems**
- **Issue**: Cannot access AR module
- **Solution**: Verify user permissions with administrator
- **Check**: User has ar:read permission minimum

**Customer Creation Errors**
- **Issue**: Customer code already exists
- **Solution**: Use unique customer codes
- **Check**: Customer code validation rules

**Invoice Processing Issues**
- **Issue**: Invoice creation fails
- **Solution**: Verify customer credit limit availability
- **Check**: Customer status is Active

**Payment Application Problems**
- **Issue**: Payment cannot be applied to invoice
- **Solution**: Check invoice status and currency match
- **Check**: Invoice not already fully paid

### Performance Issues

**Slow Loading Times**
- Clear browser cache
- Check network connectivity
- Contact system administrator
- Review system status page

**Report Generation Delays**
- Reduce date ranges for large reports
- Use filters to limit data scope
- Schedule reports for off-peak hours
- Consider using cached data options

### Data Issues

**Missing or Incorrect Data**
- Verify data entry permissions
- Check required field validation
- Review audit logs for changes
- Contact support for data recovery

**Synchronization Problems**
- Check integration status
- Verify API connectivity
- Review error logs
- Restart failed integrations

### AI Feature Issues

**Credit Scoring Not Available**
- Verify ai:read permission
- Check customer data completeness
- Ensure AI service connectivity
- Review minimum data requirements

**Forecast Generation Errors**
- Verify sufficient historical data
- Check date range validity
- Ensure time series service availability
- Review data quality metrics

### Getting Help

**Self-Service Resources**
- Online documentation portal
- Video tutorials and training
- FAQ database
- Community forums

**Support Channels**
- Email: support@datacraft.co.ke
- Phone: +254-xxx-xxx-xxxx
- Live chat during business hours
- Priority support for enterprise customers

**Escalation Process**
1. Check documentation and FAQ
2. Contact first-level support
3. Escalate to technical specialists
4. Engage development team if needed

---

## Best Practices

### Data Management
- Maintain accurate customer information
- Regular data validation and cleanup
- Implement consistent naming conventions
- Monitor data quality metrics

### Security
- Use strong authentication methods
- Regular permission reviews
- Monitor audit logs
- Follow data privacy guidelines

### Performance Optimization
- Use filters and pagination for large datasets
- Schedule resource-intensive operations
- Monitor system performance metrics
- Optimize database queries

### User Training
- Provide comprehensive user training
- Create role-specific documentation
- Establish change management processes
- Regular refresher training sessions

---

## New Features (v2.2)

### Recurring Billing Engine

Automate monthly, quarterly, or annual invoice generation without manual intervention.

**Setup a recurring schedule:**

```python
svc = AccountsReceivableService(tenant_id="acme")
schedule = await svc.create_recurring_schedule(
    customer_id="cust-001",
    template_lines=[{
        "description": "Monthly SaaS subscription",
        "quantity": 1,
        "unit_price": 1200.00,
        "tax_rate": 0.16,
        "revenue_account": "4100",
    }],
    frequency="monthly",
    start_date="2026-07-01",
    end_date="2027-06-30",
    payment_terms="NET30",
)

# Run on the first of each month (scheduled job)
result = await svc.run_recurring_invoicing(as_of_date="2026-07-01")
# result["invoices_created"] == 1
```

---

### Dynamic Early-Payment Discounts

Offer customers sliding-scale discounts to accelerate cash collection.

**Generate and accept a discount offer:**

```python
offer = await svc.calculate_dynamic_discount(
    invoice_id="inv-001",
    cost_of_capital_pct=12.0,   # your cost of capital
)
# offer["tiers"] — list of {pay_by, discount_pct, amount_payable}

# Customer accepts the 5-day tier (index 1)
result = await svc.accept_early_payment_discount(offer_id=offer["id"], tier_index=1)
```

The system computes the break-even discount rate from your cost of capital, so every offer is financially sound.

---

### Instalment Plans

Allow customers to pay large invoices in structured instalments.

```python
plan_result = await svc.create_instalment_plan(
    invoice_id="inv-002",
    num_instalments=3,
    frequency="monthly",
    first_due_date="2026-07-15",
)
# plan_result["instalments"] — 3 records, each with its own due date and amount

# When a payment arrives for instalment 1:
await svc.process_instalment_payment(
    instalment_id=plan_result["instalments"][0]["id"],
    payment_id="pay-001",
)
```

---

### Period-Close Checklist

Replace manual spreadsheet checklists with a structured, auditable close process.

```python
close_result = await svc.run_period_close_checklist(period="2026-06")
# close_result["overall_status"] — "pass" or "fail"
# close_result["steps"] — per-step pass/fail/skipped detail
```

Steps executed automatically:
1. Unposted approved invoice check
2. Foreign-currency revaluation
3. ECL bad-debt provision calculation
4. AR sub-ledger to GL reconciliation
5. Aging report snapshot
6. Dunning letter archive count

---

### Customer Churn Risk Scoring

Identify at-risk customers before they stop paying, using AR behavioural signals.

```python
score = await svc.calculate_churn_risk_score(customer_id="cust-001")
# score["churn_risk_score"]  — 0.0 (low) to 1.0 (high)
# score["risk_band"]         — "low" | "medium" | "high"
# score["signals"]           — payment delay slope, dispute freq, etc.

# Score all active customers in one call:
batch = await svc.run_churn_scoring()
# batch["elevated_risk_customers"] — IDs of customers with score >= 0.7
```

---

### IFRS 9 Scenario-Based ECL

Provide auditors and the board with multi-scenario ECL disclosures.

```python
ecl = await svc.calculate_ecl_scenarios()
# Returns base / adverse / severe scenario results plus probability-weighted ECL
print(ecl["probability_weighted_ecl"])

# Supply custom scenarios aligned to central bank stress tests:
ecl = await svc.calculate_ecl_scenarios(scenarios={
    "base":    {"current": 0.005, "1_30": 0.01, "31_60": 0.05, "61_90": 0.15, "91_120": 0.40, "120_plus": 0.85},
    "adverse": {"current": 0.015, "1_30": 0.03, "31_60": 0.10, "61_90": 0.30, "91_120": 0.65, "120_plus": 1.00},
})
```

---

### Customer Self-Service Portal

Give customers secure, read-only access to their open invoices without calling AR staff.

```python
# Generate a 72-hour portal token and send it to the customer
token_info = await svc.generate_customer_portal_token(customer_id="cust-001", expires_in_hours=72)
# Send token_info["token"] to the customer via email

# Customer-facing endpoint (validate and return invoices):
open_invoices = await svc.portal_get_open_invoices(
    token=token_info["token"],
    customer_id="cust-001",
)
```

Internal fields (tenant_id, workflow IDs) are stripped before data is returned to the customer.

---

### Bank Statement Auto-Reconciliation

Process ISO 20022 / MT940 bank statement lines and auto-match them to open invoices.

```python
statement_result = await svc.ingest_bank_statement(
    statement_lines=[
        {
            "amount": 5000.00,
            "currency": "USD",
            "reference": "Acme Corp payment",
            "customer_id": "cust-001",   # optional — looked up by name if absent
            "value_date": "2026-06-10",
        },
    ],
    statement_date="2026-06-10",
    bank_account="ACC-001-USD",
)
print(f"Matched: {statement_result['matched_count']}, Unmatched: {statement_result['unmatched_count']}")
```

Unmatched lines are flagged for manual review in the `unmatched` array.

---

*For additional support and training resources, visit the APG Platform documentation portal or contact our support team.*