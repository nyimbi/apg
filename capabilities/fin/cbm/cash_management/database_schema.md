# APG Cash Management - Database Schema Documentation

**Version**: 1.0  
**Date**: January 2025  
**© 2025 Datacraft. All rights reserved.**

---

## Overview

The APG Cash Management database schema is designed for enterprise-scale treasury operations with multi-tenant architecture, real-time performance, and comprehensive audit compliance. The schema supports the complete cash management lifecycle from bank connectivity to AI-powered optimization.

### Key Design Principles

1. **APG Multi-Tenant Architecture**: Hash-based partitioning by `tenant_id` for horizontal scaling
2. **Time-Series Optimization**: Range partitioning for historical data with automatic partition management  
3. **Performance First**: Comprehensive indexing strategy for sub-second query response times
4. **Audit Compliance**: Complete audit trails with regulatory compliance support
5. **Extensibility**: JSONB fields for flexible metadata and future enhancements

---

## Schema Structure

### Database Organization

```
cm_cash_management (schema)
├── Core Entity Tables
│   ├── cm_banks                    # Bank master data
│   ├── cm_cash_accounts            # Cash account details
│   └── cm_optimization_rules       # Cash optimization policies
├── Time-Series Tables
│   ├── cm_cash_positions           # Daily cash positions
│   ├── cm_cash_flows              # Transaction-level flows
│   └── cm_cash_alerts             # Real-time alerts
├── Forecasting Tables
│   ├── cm_cash_forecasts          # AI-powered forecasts
│   └── cm_forecast_assumptions    # Scenario modeling data
├── Investment Tables
│   ├── cm_investments             # Investment portfolio
│   └── cm_investment_opportunities # AI-curated opportunities
└── Compliance Tables
    ├── cm_audit_trail             # Complete activity log
    └── cm_performance_metrics     # KPI tracking
```

---

## Core Entity Tables

### cm_banks

Master data for banking relationships and connectivity.

**Key Features:**
- Multi-tenant partitioning (16 hash partitions)
- Unique constraints on `bank_code` and `swift_code` per tenant
- JSONB fields for flexible contact information and fee structures
- API integration metadata for real-time bank connectivity
- Soft delete with audit trail support

**Business Logic:**
- Enforces SWIFT code format validation (8 or 11 characters)
- Tracks bank relationship status and credit ratings
- Stores encrypted API credentials for secure bank integrations
- Performance monitoring via `last_api_sync` timestamps

**Partitioning Strategy:**
```sql
PARTITION BY HASH (tenant_id)
-- 16 partitions: cm_banks_0 through cm_banks_15
```

### cm_cash_accounts

Detailed cash account information with real-time balance tracking.

**Key Features:**
- Multi-tenant partitioning for horizontal scaling
- Real-time balance fields with pending transaction support
- Automated cash sweeping configuration
- Reconciliation status tracking
- Interest rate and fee schedule management

**Balance Management:**
- `current_balance`: Book balance from bank
- `available_balance`: Available for use (excludes holds)
- `pending_credits`: Incoming transactions not yet cleared
- `pending_debits`: Outgoing transactions not yet processed

**Automation Features:**
- Cash sweep configuration with target accounts and thresholds
- Interest rate tracking for yield optimization
- Fee schedule management for cost analysis

### cm_optimization_rules

AI-powered optimization policies and decision rules.

**Key Features:**
- Configurable optimization goals (yield, risk, liquidity)
- Multi-dimensional rule scope (entities, currencies, account types)
- Risk controls and concentration limits
- Automated execution with approval thresholds
- Performance tracking and machine learning enhancement

**Rule Categories:**
- Investment optimization (yield maximization, risk minimization)
- Cash sweeping (liquidity management, concentration reduction)
- FX hedging (exposure management, timing optimization)
- Risk management (stress testing, compliance monitoring)

---

## Time-Series Tables

### cm_cash_positions

Daily aggregated cash positions across all accounts and entities.

**Key Features:**
- Monthly range partitioning for performance optimization
- Multi-currency position aggregation
- Risk metrics calculation (concentration, liquidity ratios)
- Projected cash flows for forward-looking analysis
- Executive dashboard optimization

**Partitioning Strategy:**
```sql
PARTITION BY RANGE (position_date)
-- Monthly partitions: cm_cash_positions_YYYY_MM
-- Automatic partition creation for current + 2 years
```

**Key Metrics:**
- Total cash, available cash, restricted cash, invested cash
- Account type breakdown (checking, savings, money market, investment)
- Risk indicators (days cash on hand, stress test coverage)
- Projected flows (inflows, outflows, net flow)

### cm_cash_flows

Transaction-level cash flow tracking for detailed analysis.

**Key Features:**
- Monthly range partitioning for high-volume transaction data
- Source system integration (AP, AR, GL modules)
- Forecasting attributes with confidence scoring
- Full-text search capabilities for descriptions and counterparties
- Categorization for business intelligence

**Data Integration:**
- Links to source transactions via `transaction_id` and `source_module`
- Counterparty tracking for relationship analysis
- Cost center and department allocation for management reporting
- Recurring pattern identification for forecasting improvement

### cm_cash_alerts

Real-time alerting system for proactive cash management.

**Key Features:**
- Time-based partitioning for alert lifecycle management
- Escalation workflow with configurable levels
- Multi-channel notification tracking
- Automated resolution capabilities
- Performance analytics for alert effectiveness

**Alert Types:**
- Balance thresholds (low/high cash positions)
- Forecast shortfalls (predicted cash gaps)
- Investment maturities (reinvestment opportunities)
- Risk violations (concentration, compliance breaches)
- System issues (bank connectivity, data quality)

---

## Forecasting Tables

### cm_cash_forecasts

AI-powered cash forecasting with machine learning models.

**Key Features:**
- Multi-scenario forecasting (base case, optimistic, pessimistic, stress test)
- Confidence intervals and statistical measures
- Model performance tracking for continuous improvement
- Forecast accuracy backtesting
- Risk assessment integration

**Machine Learning Integration:**
- Model metadata tracking (`model_used`, `model_version`)
- Feature importance analysis via JSONB storage
- Training data period documentation
- Automated accuracy measurement and error analysis

**Risk Analytics:**
- Value at Risk (VaR) calculations
- Shortfall probability estimation
- Stress test scenario results
- Confidence level tracking

### cm_forecast_assumptions

Transparent assumption management for scenario modeling.

**Key Features:**
- Assumption categorization and documentation
- Statistical distribution modeling
- Sensitivity analysis with correlation factors
- Assumption review and validation workflow
- Scenario impact calculation

**Statistical Modeling:**
- Multiple distribution types (normal, uniform, triangular)
- Mean, standard deviation, min/max value tracking
- Correlation factor analysis for assumption interdependencies
- Sensitivity coefficient calculation for forecast impact

---

## Investment Tables

### cm_investments

Portfolio management for short-term investments.

**Key Features:**
- Complete investment lifecycle tracking
- Performance measurement and benchmarking
- Risk management with rating integration
- Maturity monitoring and reinvestment planning
- AI optimization scoring

**Investment Types:**
- Money market funds, treasury bills, commercial paper
- Certificates of deposit, term deposits
- Government and corporate bonds
- Repurchase agreements

**Performance Tracking:**
- Expected vs. actual return analysis
- Accrued interest calculations
- Mark-to-market valuation
- Benchmark comparison capabilities

### cm_investment_opportunities

AI-curated investment opportunities with scoring.

**Key Features:**
- Real-time opportunity identification
- Multi-dimensional AI scoring (yield, risk, liquidity, fit)
- Availability window management
- Automated recommendation engine
- Counterparty risk assessment

**AI Scoring Methodology:**
- Yield score: Return attractiveness relative to market
- Risk score: Credit and operational risk assessment  
- Liquidity score: Ease of early redemption
- Fit score: Portfolio diversification and strategy alignment

---

## Compliance and Monitoring

### cm_audit_trail

Comprehensive audit logging for regulatory compliance.

**Key Features:**
- Complete user activity tracking
- Regulatory impact flagging
- Data retention policy management
- Before/after value tracking
- Multi-dimensional search capabilities

**Audit Coverage:**
- All CRUD operations on financial data
- User authentication and authorization events
- System configuration changes
- Automated decision execution
- Regulatory report generation

### cm_performance_metrics

KPI tracking and performance monitoring.

**Key Features:**
- Real-time metrics calculation
- Historical trend analysis
- Target vs. actual performance tracking
- Multi-dimensional analytics
- Automated alerting for performance degradation

**Metric Categories:**
- Operational metrics (processing time, accuracy, throughput)
- Business metrics (yield optimization, cost reduction, risk exposure)
- User adoption metrics (feature utilization, user satisfaction)
- System performance metrics (response time, availability, error rates)

---

## Partitioning Strategy

### Hash Partitioning (Multi-Tenant)

Used for master data tables that grow with tenant count:

**Tables:** `cm_banks`, `cm_cash_accounts`, `cm_cash_forecasts`, `cm_forecast_assumptions`, `cm_investments`, `cm_investment_opportunities`, `cm_optimization_rules`

**Configuration:**
- 16 hash partitions per table
- Partition by `tenant_id` for even distribution
- Supports horizontal scaling as tenant count grows

**Benefits:**
- Even data distribution across partitions
- Parallel query execution for performance
- Tenant isolation for security and compliance
- Simplified backup and maintenance operations

### Range Partitioning (Time-Series)

Used for high-volume time-series data:

**Tables:** `cm_cash_positions`, `cm_cash_flows`, `cm_cash_alerts`, `cm_audit_trail`, `cm_performance_metrics`

**Configuration:**
- Monthly partitions for current year + 2 years
- Automatic partition creation and pruning
- Partition elimination for query optimization

**Benefits:**
- Efficient queries on date ranges
- Automated data lifecycle management
- Parallel maintenance operations
- Storage optimization for historical data

---

## Indexing Strategy

### Primary Indexes

**Unique Business Constraints:**
- Bank codes and SWIFT codes per tenant
- Account numbers per bank and tenant
- Investment numbers per tenant
- Rule codes per tenant

**Performance Indexes:**
- Multi-column indexes for common query patterns
- Partial indexes for active records only
- Functional indexes for calculated fields
- GIN indexes for JSONB and full-text search

### Index Categories

1. **Entity Lookup Indexes**: Fast primary key and unique constraint lookups
2. **Foreign Key Indexes**: Efficient join operations between related tables
3. **Filter Indexes**: Optimized filtering on status, type, and date fields
4. **Aggregate Indexes**: Support for sum, count, and grouping operations
5. **Search Indexes**: Full-text search and pattern matching capabilities

### Index Monitoring

**Usage Statistics:**
```sql
SELECT * FROM cm_index_usage_stats;
```

**Size Analysis:**
```sql
SELECT * FROM cm_table_sizes;
```

---

## Performance Characteristics

### Query Performance Targets

- **Dashboard Queries**: < 500ms response time
- **Lookup Operations**: < 100ms response time
- **Bulk Analytics**: < 5 seconds for monthly data
- **Real-time Alerts**: < 200ms processing time
- **Report Generation**: < 30 seconds for complex reports

### Scalability Metrics

- **Data Volume**: 100M+ transactions per year per large tenant
- **Concurrent Users**: 1,000+ users across all tenants
- **API Throughput**: 10,000+ requests per minute
- **Storage Growth**: 50GB+ per year per large tenant
- **Backup Window**: < 4 hours for full database backup

### Maintenance Windows

- **Index Maintenance**: Weekly during off-peak hours
- **Partition Management**: Monthly automated process
- **Statistics Updates**: Daily automated refresh
- **Constraint Validation**: Quarterly consistency checks

---

## Data Retention and Archival

### Retention Policies

**Operational Data:**
- Cash positions: 7 years (regulatory requirement)
- Cash flows: 7 years (tax and audit compliance)
- Investment records: 7 years after maturity
- Bank account data: 7 years after closure

**Audit and Compliance:**
- Audit trail: 10 years (SOX compliance)
- Performance metrics: 5 years (operational analysis)
- User activity logs: 2 years (security monitoring)
- System logs: 1 year (troubleshooting support)

### Archival Strategy

**Automated Archival:**
- Monthly partition archival for data older than retention period
- Compressed storage for archived partitions
- Read-only access to archived data for compliance queries
- Secure deletion after retention period expiration

**Archive Storage:**
- Cold storage for infrequently accessed data
- Encryption at rest for sensitive financial data
- Geographic distribution for disaster recovery
- Compliance validation before data deletion

---

## Security and Compliance

### Data Security

**Encryption:**
- Data at rest: AES-256 encryption for all financial data
- Data in transit: TLS 1.3 for all database connections
- Backup encryption: Separate encryption keys for backup files
- Key management: Hardware security modules (HSM) for key storage

**Access Control:**
- Row-level security (RLS) for multi-tenant data isolation
- Column-level permissions for sensitive fields
- Audit logging for all data access operations
- Regular access reviews and privilege validation

### Regulatory Compliance

**SOX Compliance:**
- Complete audit trails for all financial transactions
- Segregation of duties for critical operations
- Regular internal controls testing
- Quarterly compliance reporting

**GDPR Compliance:**
- Data subject identification and tracking
- Right to erasure implementation
- Data portability support
- Privacy impact assessments

**Industry Standards:**
- PCI DSS for payment data handling
- ISO 27001 for information security management
- SOC 2 Type II for service organization controls
- FFIEC guidelines for financial institutions

---

## Disaster Recovery

### Backup Strategy

**Full Backups:**
- Daily full database backups during maintenance window
- Geographic replication to secondary data center
- Backup validation and restoration testing
- 30-day backup retention for operational recovery

**Incremental Backups:**
- Hourly transaction log backups for minimal data loss
- Real-time replication for critical trading data
- Continuous data protection (CDP) for audit trails
- Point-in-time recovery capabilities

### Recovery Procedures

**Recovery Time Objectives (RTO):**
- Critical systems: 15 minutes
- Standard operations: 4 hours
- Reporting systems: 24 hours
- Historical analysis: 72 hours

**Recovery Point Objectives (RPO):**
- Trading operations: 5 minutes
- Cash management: 15 minutes
- Reporting data: 1 hour
- Analytics: 4 hours

---

## Migration and Deployment

### Schema Versioning

**Migration Scripts:**
- `001_initial_schema.sql`: Base schema creation
- `002_indexes_optimization.sql`: Performance optimization
- Future migrations numbered sequentially

**Version Control:**
- All schema changes tracked in version control
- Automated migration testing in staging environment
- Rollback procedures for failed migrations
- Database schema documentation updates

### Deployment Process

**Environment Progression:**
1. Development: Initial development and unit testing
2. Integration: Cross-module integration testing
3. Staging: Production-like performance testing
4. Production: Controlled deployment with monitoring

**Deployment Validation:**
- Schema comparison between environments
- Data integrity validation post-migration
- Performance benchmark verification
- Functional testing of critical operations

---

## Monitoring and Maintenance

### Health Monitoring

**Database Health:**
- Connection pool monitoring and alerting
- Query performance analysis and optimization
- Storage utilization tracking and forecasting
- Index effectiveness measurement

**Data Quality:**
- Constraint violation monitoring
- Data freshness validation
- Referential integrity checking
- Business rule compliance verification

### Maintenance Procedures

**Daily Maintenance:**
- Statistics updates for query optimization
- Log file rotation and archival
- Backup validation and verification
- Performance metric collection

**Weekly Maintenance:**
- Index maintenance and rebuilding
- Database fragmentation analysis
- Partition pruning and creation
- Security access review

**Monthly Maintenance:**
- Full database integrity check
- Storage optimization and cleanup
- Performance benchmark testing
- Disaster recovery testing

---

**© 2025 Datacraft. All rights reserved.**  
**Author**: Nyimbi Odero | APG Platform Architect  
**Last Updated**: January 2025  
**Schema Version**: 1.0