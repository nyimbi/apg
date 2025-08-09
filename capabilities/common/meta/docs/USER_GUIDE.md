# APG Metadata Management User Guide

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Web Interface](#web-interface)
- [Data Discovery](#data-discovery)
- [Asset Management](#asset-management)
- [Search & Navigation](#search--navigation)
- [Data Lineage](#data-lineage)
- [AI-Powered Classification](#ai-powered-classification)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Introduction

APG Metadata Management is a revolutionary enterprise metadata platform that provides comprehensive data cataloging, lineage tracking, and AI-powered classification capabilities. It surpasses industry leaders like Informatica EDC and Apache Atlas through advanced features and intuitive user experience.

### Key Capabilities

🔍 **Intelligent Discovery** - Automated metadata discovery from 15+ data sources  
🧠 **AI Classification** - Automatic PII/PHI detection and data classification  
📊 **Visual Lineage** - Interactive data lineage visualization and impact analysis  
🔍 **Natural Language Search** - Search your data using plain English  
⚡ **Real-time Updates** - Live metadata synchronization and change detection  
🔒 **Enterprise Security** - Multi-tenant architecture with role-based access  

---

## Getting Started

### System Requirements

- **Python:** 3.9+ with async support
- **Database:** PostgreSQL 12+ (primary), Neo4j 4+ (lineage), Redis 6+ (cache)
- **Memory:** 8GB+ RAM for production deployments
- **Storage:** 100GB+ for metadata and logs
- **Network:** HTTP/HTTPS access to data sources

### Quick Installation

1. **Install the capability:**
   ```bash
   pip install -e /path/to/apg/capabilities/common/meta
   ```

2. **Initialize the service:**
   ```python
   from capabilities.common.meta import initialize_capability
   
   service = await initialize_capability()
   ```

3. **Access the web interface:**
   Navigate to `http://localhost:5000/metadata/dashboard`

### First Steps

1. **Configure data sources** in the Discovery section
2. **Run your first discovery job** to populate the catalog
3. **Explore the dashboard** to understand your data landscape
4. **Set up classification rules** for automated data governance

---

## Core Concepts

### Assets
**Assets** are the fundamental entities in your metadata catalog:

- **Tables/Views** - Database tables and views
- **Files** - CSV, JSON, Parquet, Avro files
- **APIs** - REST endpoints, GraphQL schemas
- **ML Models** - Machine learning models and experiments
- **Pipelines** - ETL/ELT workflows and transformations
- **Reports** - Business intelligence reports and dashboards

### Metadata Properties
Each asset includes comprehensive metadata:

```
Asset Metadata
├── Basic Info (name, type, description, owner)
├── Schema (columns, data types, constraints)
├── Quality Metrics (completeness, accuracy, freshness)
├── Classification (PII, PHI, confidential, public)
├── Lineage (upstream/downstream relationships)
├── Usage Statistics (query frequency, users)
└── Custom Attributes (business context)
```

### Data Lineage
**Lineage** tracks how data flows through your organization:

- **Column-Level Lineage** - Track individual field transformations
- **Table-Level Lineage** - Understand dataset relationships  
- **Cross-System Lineage** - Follow data across multiple platforms
- **Impact Analysis** - Assess downstream effects of changes

### Classification System
AI-powered classification organizes data by sensitivity:

| Level | Description | Examples |
|-------|-------------|----------|
| `PUBLIC` | Publicly available data | Product catalogs, marketing content |
| `INTERNAL` | Internal business data | Sales reports, operational metrics |
| `CONFIDENTIAL` | Sensitive business data | Financial records, strategic plans |
| `RESTRICTED` | Highly sensitive data | HR records, security logs |
| `PII` | Personally identifiable information | Names, emails, addresses |
| `SENSITIVE_PII` | Sensitive personal data | SSNs, medical records, financial info |

---

## Web Interface

### Dashboard Overview

The main dashboard provides a comprehensive view of your data landscape:

**Key Metrics Panel:**
- Total assets discovered
- Data quality trends  
- Classification distribution
- Discovery job status
- User activity summary

**Visual Analytics:**
- Asset type distribution (pie chart)
- Data quality trends over time (line chart)
- Source system coverage (bar chart)
- Classification heat map

**Recent Activity Feed:**
- Newly discovered assets
- Classification updates
- Quality alerts
- User actions

### Navigation Menu

**📊 Dashboard** - Overview and key metrics  
**🔍 Discovery** - Manage data source connections and discovery jobs  
**📁 Assets** - Browse and manage metadata assets  
**🔍 Search** - Intelligent search across all assets  
**📈 Lineage** - Interactive lineage visualization  
**🧠 Classification** - AI classification rules and results  
**⚙️ Settings** - System configuration and user management  

---

## Data Discovery

### Setting Up Data Sources

Discovery is the process of automatically cataloging metadata from your data sources.

#### Step 1: Create a Connection

Navigate to **Discovery → Data Sources → Add New Source**

**Database Connection Example:**
```
Connection Name: Production PostgreSQL
Type: PostgreSQL  
Host: prod-db.company.com
Port: 5432
Database: ecommerce
Username: metadata_reader
Password: [secure_password]

Advanced Options:
✓ Enable SSL
✓ Include system tables
□ Skip empty tables
```

**File System Connection Example:**
```
Connection Name: Data Lake S3
Type: Amazon S3
Bucket: company-data-lake
Region: us-east-1
Access Key: [aws_access_key]
Secret Key: [aws_secret_key]

Include Patterns:
- /warehouse/*/*
- /analytics/reports/*

Exclude Patterns:  
- /temp/*
- /staging/test_*
```

#### Step 2: Configure Discovery Schedule

**One-time Discovery:**
- Immediate scan of all available assets
- Best for initial setup or ad-hoc exploration

**Recurring Discovery:**
- Scheduled scans (daily, weekly, monthly)
- Automated metadata updates
- Change detection and alerting

**Real-time Discovery:**  
- Event-driven updates
- Integration with data pipeline systems
- Immediate reflection of schema changes

#### Step 3: Run Discovery Job

1. Click **"Run Discovery"** for immediate execution
2. Monitor progress in the **"Discovery Jobs"** section  
3. Review discovered assets in the **Assets** tab
4. Check for any errors or warnings

### Discovery Results

After discovery completes, you'll see:

**Assets Discovered:**
- Number of new assets found
- Updated existing assets
- Asset type breakdown

**Schema Analysis:**
- Column-level metadata extracted
- Data type inference
- Key/constraint detection
- Sample data profiling

**Quality Assessment:**
- Completeness scores
- Data validity checks
- Pattern recognition
- Anomaly detection

**Classification Results:**
- Automatic PII/PHI detection  
- Sensitivity classification
- Compliance tag assignment
- Risk assessment

---

## Asset Management

### Browsing Assets

The **Assets** section provides multiple ways to explore your metadata catalog:

#### List View
- Tabular display with key metadata
- Sortable columns (name, type, quality, owner)
- Quick filters and search
- Bulk operations support

#### Card View  
- Visual asset cards with key metrics
- Quality score indicators
- Classification badges
- Owner avatars

#### Tree View
- Hierarchical organization by source system
- Expandable database/schema structure  
- Context-aware navigation
- Breadcrumb navigation

### Asset Details Page

Each asset has a comprehensive details page with multiple tabs:

#### 📋 Overview
- Basic metadata and description
- Owner and steward information
- Tags and custom attributes
- Quality score breakdown
- Classification and sensitivity level

#### 📊 Schema
- Column definitions and data types
- Primary/foreign key relationships  
- Constraints and indexes
- Column-level classifications
- Data profiling statistics

#### 📈 Lineage  
- Interactive lineage diagram
- Upstream and downstream relationships
- Transformation logic display
- Impact analysis tools

#### 📊 Quality
- Data quality metrics over time
- Completeness and validity trends
- Anomaly detection results
- Quality rule execution history

#### 👥 Usage
- Query frequency and patterns
- User access statistics  
- Popular column usage
- Performance metrics

#### 📝 Documentation
- Business glossary integration
- User comments and annotations
- Change log and version history
- Related documentation links

### Editing Asset Metadata

Users can enhance discovered metadata:

**Basic Information:**
- Update descriptions and display names
- Assign owners and stewards  
- Add business tags
- Set custom attributes

**Business Context:**
- Link to business glossary terms
- Add business rules and constraints
- Define data retention policies
- Associate with business processes

**Quality Rules:**
- Define custom quality checks
- Set quality thresholds
- Configure alerting rules  
- Schedule quality monitoring

---

## Search & Navigation

### Intelligent Search

APG's search capabilities go far beyond simple text matching:

#### Natural Language Search
Ask questions in plain English:

- *"Show me all customer data with email addresses"*
- *"Find high quality tables from the sales system"*  
- *"What contains PII information?"*
- *"Tables updated last week with quality issues"*

#### Semantic Search
The system understands meaning and context:

- Search for "revenue" finds assets with "sales", "income", "earnings"
- "Customer info" matches "client_data", "user_profiles", "account_details"
- Automatic expansion of business terms and synonyms

#### Advanced Filters

**By Asset Type:**
```
Tables ✓  Views ✓  Files ✓  APIs ✗  Models ✗
```

**By Data Classification:**
```
☐ Public  ☑ Internal  ☑ Confidential  ☐ PII
```

**By Quality Score:**
```
Quality Score: [>=] [0.8] (High quality assets only)
```

**By Source System:**  
```
PostgreSQL ✓  MySQL ✗  S3 ✓  Snowflake ✗
```

**By Date Range:**
```
Created: [2024-01-01] to [2024-12-31]
Updated: [Last 30 days ▼]
```

#### Search Results

Results are ranked by relevance and include:

- **Relevance Score** - AI-calculated matching score
- **Match Highlights** - Text highlighting in descriptions/names
- **Quick Actions** - View details, lineage, or export
- **Context Cards** - Key metadata preview

### Faceted Navigation

Use faceted navigation to drill down into your catalog:

**Source Systems → Databases → Schemas → Tables**
```
PostgreSQL (1,245 assets)
├── ecommerce_prod (892 assets)  
│   ├── public (445 assets)
│   ├── analytics (321 assets)
│   └── staging (126 assets)
└── hr_system (353 assets)
    ├── employees (234 assets)
    └── payroll (119 assets)
```

**Data Classifications:**
```
CONFIDENTIAL (2,341 assets)
├── Financial Data (892 assets)
├── Customer Data (654 assets)  
├── Employee Data (432 assets)
└── Strategic Data (363 assets)
```

---

## Data Lineage

### Understanding Lineage

Data lineage shows how information flows through your organization, helping you:

- **Track Data Origins** - Understand where data comes from
- **Impact Analysis** - Assess effects of changes
- **Compliance** - Meet regulatory requirements
- **Debugging** - Troubleshoot data quality issues
- **Optimization** - Identify bottlenecks and inefficiencies

### Lineage Visualization

#### Interactive Lineage Graph

The lineage viewer provides an interactive network diagram:

**Graph Elements:**
- **Nodes** - Assets (tables, files, reports)
- **Edges** - Data relationships and transformations
- **Colors** - Asset types and classifications
- **Thickness** - Data volume or frequency

**Navigation Controls:**
- **Zoom** - Mouse wheel or +/- buttons
- **Pan** - Click and drag to move around
- **Focus** - Click asset to center and highlight
- **Filter** - Hide/show specific asset types or systems

**Layout Options:**
- **Hierarchical** - Top-down or left-right flow
- **Force-Directed** - Automatic optimal positioning  
- **Circular** - Circular arrangement around central asset
- **Timeline** - Chronological data flow sequence

#### Lineage Levels

**Table-Level Lineage:**
```
raw_customers → customer_cleansing → customer_360 → customer_reports
```

**Column-Level Lineage:**  
```
customers.email → cleansed.email_clean → analytics.customer_email
customers.phone → cleansed.phone_clean → analytics.contact_phone
```

**Cross-System Lineage:**
```
PostgreSQL → ETL Pipeline → Data Warehouse → BI Reports
```

### Lineage Analysis Features

#### Impact Analysis
Understand the downstream effects of changes:

1. Select an asset in the lineage graph
2. Choose "Analyze Impact" from the context menu
3. Specify the type of change (schema, data, system)
4. Review the impact assessment report

**Sample Impact Report:**
```
Change: Remove 'legacy_customer_id' column from customers table

Direct Impact:
- customer_analytics view (CRITICAL) - References removed column
- daily_report ETL (HIGH) - Uses column in JOIN operation

Indirect Impact:  
- executive_dashboard (MEDIUM) - Via customer_analytics dependency
- customer_segmentation_model (LOW) - Uses aggregated data

Recommendations:
1. Update customer_analytics view definition
2. Modify ETL to use new_customer_id instead
3. Test downstream reports and models
4. Notify 12 affected users
```

#### Root Cause Analysis
Trace data quality issues to their source:

1. Identify problematic data in downstream system
2. Follow lineage backwards to source systems
3. Examine transformation logic at each step
4. Identify where data corruption occurred

#### Compliance Reporting
Generate lineage reports for auditing:

- **Data Flow Documentation** - Complete data journey maps
- **Processing Inventory** - All systems that touch specific data
- **Retention Tracking** - Data lifecycle and disposal points
- **Access Audit** - Who has accessed data along the pipeline

### Managing Lineage

#### Automatic Lineage Detection

The system automatically detects lineage through:

**SQL Analysis:**
- Parse SELECT, INSERT, UPDATE statements
- Extract table and column references  
- Identify JOIN relationships
- Detect aggregation and transformation logic

**ETL Tool Integration:**
- Airflow DAG analysis
- dbt model parsing
- Spark job monitoring
- Kafka topic tracing

**API Call Tracing:**
- REST API dependency mapping
- GraphQL schema relationships
- Message queue flow tracking

#### Manual Lineage Creation

For systems without automatic detection:

1. Navigate to **Lineage → Create Relationship**
2. Select source and target assets
3. Choose relationship type:
   - **Direct Copy** - Exact data replication
   - **Transformation** - Data processing/calculation
   - **Aggregation** - Summarization/grouping
   - **Join** - Combining multiple sources
   - **Filter** - Subset selection
4. Document transformation logic
5. Save the relationship

#### Lineage Validation

Ensure lineage accuracy through:

**Automated Testing:**
- Data flow verification
- Schema consistency checks
- Volume reconciliation
- Freshness validation

**User Feedback:**
- Community validation features
- Expert review workflows
- Accuracy rating system
- Crowdsourced corrections

---

## AI-Powered Classification

### Automatic Classification

APG's AI classification engine automatically identifies and categorizes your data:

#### Multi-Method Classification

The system uses multiple AI techniques:

**Pattern Matching:**
- Regular expressions for email, phone, SSN patterns
- Named entity recognition for names and addresses
- Format validation for standardized identifiers

**Statistical Analysis:**
- Cardinality analysis for key detection
- Distribution analysis for categorical data
- Null rate analysis for data quality
- Length pattern analysis for text fields

**Machine Learning:**
- Federated learning models trained on enterprise data
- Transfer learning from pre-trained models
- Ensemble methods combining multiple classifiers
- Confidence scoring and uncertainty quantification

**Natural Language Processing:**
- Column name semantic analysis
- Description and comment analysis  
- Business glossary term matching
- Context-aware classification

#### Classification Categories

**Personally Identifiable Information (PII):**
- Names, emails, phone numbers
- Addresses, ZIP codes  
- Social security numbers
- Driver's license numbers

**Protected Health Information (PHI):**
- Medical record numbers
- Health plan beneficiary numbers
- Biometric identifiers
- Full face photographs

**Financial Information:**  
- Credit card numbers
- Bank account numbers
- Financial account identifiers
- Payment processor tokens

**Business Sensitive:**
- Customer lists and segments
- Pricing and contract information
- Strategic business data
- Intellectual property

### Classification Management

#### Review and Validation

Classification results require human oversight:

**Classification Queue:**
1. Navigate to **Classification → Review Queue**
2. Filter by confidence level or asset type
3. Review AI suggestions and confidence scores
4. Accept, reject, or modify classifications
5. Provide feedback to improve future accuracy

**Batch Operations:**
- Approve multiple classifications at once
- Apply templates for similar assets
- Bulk reject low-confidence results
- Export classifications for external review

#### Custom Classification Rules

Create organization-specific rules:

**Rule Creation:**
```
Rule Name: Customer Contact Information
Classification: PII  
Confidence: 0.9

Conditions:
- Column name contains: email, phone, contact, address
- Data type: VARCHAR or TEXT
- Sample data matches: email/phone regex patterns
- Table context: customer, user, client related

Actions:
- Apply PII classification
- Add "contact_info" tag  
- Set retention period: 7 years
- Enable data masking
```

**Rule Types:**
- **Pattern Rules** - Based on data patterns
- **Context Rules** - Based on table/column context
- **Hybrid Rules** - Combination of pattern and context
- **ML Rules** - Custom machine learning models

#### Classification Analytics

Monitor classification performance:

**Accuracy Metrics:**
- Overall classification accuracy percentage
- Per-category precision and recall
- Confidence score distributions
- False positive/negative rates

**Coverage Analysis:**
- Percentage of assets classified
- Unclassified asset identification
- Classification completeness by system
- Progress tracking over time

**Compliance Dashboards:**
- PII/PHI data inventory
- Regulatory compliance status
- Risk assessment summaries
- Audit trail reports

---

## Integration Patterns

### Enterprise System Integration

APG integrates seamlessly with your existing data infrastructure:

#### Data Catalog Integration

**Apache Atlas Integration:**
```python
from apg_metadata.integrations import AtlasSync

sync = AtlasSync(
    atlas_url="http://atlas.company.com:21000",
    apg_service=metadata_service
)

# Bidirectional sync
await sync.sync_from_atlas()
await sync.sync_to_atlas()
```

**Informatica EDC Integration:**
```python
from apg_metadata.integrations import EDCSync

sync = EDCSync(
    edc_url="https://edc.company.com",
    credentials=edc_credentials
)

# Import existing catalog
await sync.import_edc_catalog()
```

#### BI Tool Integration

**Tableau Integration:**
- Publish data sources with metadata
- Sync worksheet and dashboard lineage
- Update column descriptions automatically
- Track usage analytics

**Power BI Integration:**  
- Dataset metadata synchronization
- Report lineage tracking
- Semantic model documentation
- User access analytics

#### Data Pipeline Integration

**Apache Airflow:**
```python
from apg_metadata.integrations import AirflowLineage

# Automatic lineage from DAG definitions
lineage = AirflowLineage(dag_folder="/opt/airflow/dags")
await lineage.extract_lineage()
```

**dbt Integration:**
```python
from apg_metadata.integrations import DbtSync

sync = DbtSync(
    project_dir="/path/to/dbt/project",
    profiles_dir="~/.dbt"
)

# Extract model lineage and documentation  
await sync.sync_models()
```

### Real-time Integration

#### Event-Driven Updates

Set up real-time metadata updates:

**Database Change Detection:**
```sql
-- PostgreSQL trigger example
CREATE OR REPLACE FUNCTION notify_schema_change()
RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('schema_change', json_build_object(
        'table', TG_TABLE_NAME,
        'operation', TG_OP,
        'timestamp', now()
    )::text);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables
CREATE TRIGGER schema_change_trigger
    AFTER DDL ON SCHEMA public
    EXECUTE FUNCTION notify_schema_change();
```

**Kafka Integration:**  
```python
from apg_metadata.integrations import KafkaListener

listener = KafkaListener(
    bootstrap_servers='kafka.company.com:9092',
    topics=['schema-changes', 'data-quality-alerts']
)

# Real-time metadata updates
await listener.start_consuming()
```

#### Webhook Notifications

Configure webhooks for external system updates:

```python
from apg_metadata.webhooks import WebhookManager

webhook_manager = WebhookManager()

# Register webhook for asset changes
await webhook_manager.register_webhook(
    url="https://your-system.com/metadata-webhook",
    events=["asset.created", "asset.updated", "classification.completed"],
    headers={"Authorization": "Bearer your-token"}
)
```

### API Integration

#### REST API Usage

Integrate with custom applications:

```javascript
// JavaScript example
const client = new MetadataClient({
    baseUrl: 'https://metadata.company.com',
    apiKey: 'your-api-key'
});

// Search for specific data
const assets = await client.searchAssets({
    query: 'customer email data',
    filters: { classification: 'PII' }
});

// Get lineage for impact analysis
const lineage = await client.getAssetLineage(assetId, {
    direction: 'downstream',
    maxDepth: 3
});
```

#### GraphQL Integration

Use GraphQL for flexible queries:

```graphql
# Get asset with related information
query GetAssetDetails($id: ID!) {
    asset(id: $id) {
        name
        description
        classification
        columns {
            name
            dataType
            classification
        }
        upstreamAssets {
            name
            assetType
        }
        qualityMetrics {
            overallScore
            completeness
        }
    }
}
```

### Compliance Integration

#### Data Privacy Tools

**OneTrust Integration:**
- Data mapping synchronization
- Privacy impact assessment data
- Consent management alignment
- Regulatory compliance reporting

**Privacera Integration:**
- Policy synchronization
- Access control updates
- Data masking rule application
- Audit log correlation

#### GRC Platform Integration

**ServiceNow GRC:**
- Risk assessment data
- Compliance status updates
- Policy violation alerts
- Remediation workflow triggers

**Archer GRC:**
- Control effectiveness data
- Risk register updates
- Audit finding correlation
- Executive reporting

---

## Best Practices

### Data Governance

#### Establish Clear Ownership

**Asset Ownership Model:**
```
Data Owner (Business):
- Defines business purpose and usage
- Sets access and retention policies
- Makes classification decisions
- Approves usage for new purposes

Data Steward (Technical):  
- Maintains metadata accuracy
- Monitors data quality
- Implements governance policies
- Coordinates with technical teams

Data Custodian (Operations):
- Implements access controls
- Manages backups and archival
- Ensures security compliance
- Handles day-to-day operations
```

#### Classification Standards

**Develop Consistent Taxonomy:**
```
Business Classification:
├── Public (Marketing, Product Info)
├── Internal (Operational, Metrics)
├── Confidential (Financial, Strategic)
└── Restricted (HR, Legal, Security)

Technical Classification:
├── PII (Names, Emails, Addresses)  
├── Sensitive PII (SSN, Medical, Financial)
├── Anonymized (De-identified data)
└── Synthetic (Test/Demo data)

Regulatory Classification:
├── GDPR (EU Personal Data)
├── CCPA (California Consumer Data)
├── HIPAA (Protected Health Information)
└── SOX (Financial Reporting Data)
```

#### Quality Management

**Data Quality Framework:**

1. **Completeness** - Are all required fields populated?
2. **Accuracy** - Does the data reflect real-world values?
3. **Consistency** - Is data uniform across systems?
4. **Validity** - Does data conform to business rules?
5. **Uniqueness** - Are there inappropriate duplicates?
6. **Timeliness** - Is data current and up-to-date?

**Quality Rules Implementation:**
```python
# Example quality rules
quality_rules = [
    {
        "name": "Email Format Validation",
        "description": "Ensure email fields contain valid email addresses",
        "rule_type": "format",
        "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "threshold": 0.95,
        "severity": "high"
    },
    {
        "name": "Customer ID Uniqueness", 
        "description": "Customer IDs must be unique within the table",
        "rule_type": "uniqueness",
        "column": "customer_id",
        "threshold": 1.0,
        "severity": "critical"
    }
]
```

### Metadata Management

#### Documentation Standards

**Asset Documentation Template:**
```markdown
# Asset: [Asset Name]

## Business Purpose
Brief description of what this asset represents and how it's used.

## Data Dictionary  
| Column | Type | Description | Business Rules |
|--------|------|-------------|----------------|
| id     | INT  | Unique identifier | Primary key, auto-increment |
| name   | TEXT | Customer name | Required, 2-100 characters |

## Quality Expectations
- Completeness: 95%+ for required fields
- Freshness: Updated within 4 hours
- Accuracy: Validated against source systems

## Access Information
- Owner: data.team@company.com
- Steward: jane.smith@company.com  
- Access Level: Internal
- Request Process: Submit ticket to IT

## Related Assets
- Upstream: raw_customer_data, customer_interactions
- Downstream: customer_analytics, customer_reports
```

#### Tagging Strategy

**Implement Consistent Tagging:**
```
Business Tags:
- domain:sales, domain:marketing, domain:finance
- process:etl, process:reporting, process:analytics
- criticality:high, criticality:medium, criticality:low

Technical Tags:
- format:csv, format:parquet, format:json
- frequency:daily, frequency:weekly, frequency:realtime
- size:large, size:medium, size:small

Governance Tags:
- pii:yes, pii:no
- retention:7years, retention:permanent
- encryption:required, encryption:optional
```

### Performance Optimization

#### Discovery Optimization

**Efficient Discovery Scheduling:**
- Schedule intensive scans during off-peak hours
- Use incremental discovery for frequent updates
- Implement change detection to minimize processing
- Partition large discovery jobs by schema or table type

**Connection Pool Management:**
```python
# Optimize database connections
connection_config = {
    "max_connections": 10,
    "connection_timeout": 30,
    "query_timeout": 300,
    "retry_attempts": 3,
    "pool_recycle": 3600
}
```

#### Search Performance

**Search Optimization Strategies:**
- Use specific filters to narrow results
- Leverage caching for frequent queries
- Index commonly searched fields
- Implement query result pagination

**Efficient Search Patterns:**
```python
# Good: Specific, filtered search
results = await client.search_assets(
    query="customer email",
    filters={"asset_type": "table", "source_system": "postgresql"},
    limit=20
)

# Avoid: Overly broad searches
results = await client.search_assets(query="*", limit=10000)
```

### Security Best Practices

#### Access Control

**Role-Based Access Control (RBAC):**
```
Metadata Admin:
- Full system configuration access
- User management capabilities  
- Global settings modification
- Discovery job management

Data Steward:
- Asset metadata editing
- Classification management
- Quality rule configuration  
- Usage monitoring

Business User:
- Asset browsing and search
- Lineage visualization
- Documentation reading
- Usage analytics viewing

Auditor:
- Read-only access to all metadata
- Audit log access
- Compliance reporting
- Export capabilities
```

#### Sensitive Data Handling

**PII/PHI Protection:**
- Never store actual PII in metadata
- Use data samples sparingly and with masking
- Implement field-level encryption for sensitive metadata
- Regular audit of classification accuracy

**Sample Data Masking:**
```python
# Mask sensitive data in samples
masked_samples = [
    "user_****@company.com",  # Email masking
    "555-***-1234",           # Phone masking  
    "****-****-****-1234"     # Credit card masking
]
```

---

## Troubleshooting

### Common Issues

#### Discovery Problems

**Issue: Discovery job fails with connection timeout**

*Symptoms:*
- Discovery jobs fail after a few minutes
- Error message: "Connection timeout after 30 seconds"
- No assets discovered from specific source

*Solutions:*
1. Check network connectivity to data source
2. Verify credentials are correct and not expired
3. Increase connection timeout in connector configuration
4. Check firewall rules and security groups
5. Validate SSL certificates if using encrypted connections

*Prevention:*
- Implement connection health checks
- Monitor network latency
- Set up alerts for credential expiration

**Issue: Partial discovery results**

*Symptoms:*
- Discovery completes but fewer assets than expected
- Missing schemas or tables
- Inconsistent results across runs

*Solutions:*
1. Review include/exclude patterns in connector configuration
2. Check database permissions for metadata queries
3. Verify account has access to all required schemas
4. Look for case sensitivity issues in pattern matching
5. Increase query timeout for large catalogs

#### Search Issues  

**Issue: Search returns no results for known assets**

*Symptoms:*
- Empty search results for assets that exist
- Specific asset names don't return matches
- Search works for some assets but not others

*Solutions:*
1. Verify assets were successfully discovered and indexed
2. Check search index status and refresh if needed
3. Review asset names for special characters or encoding issues
4. Ensure search service is running and healthy
5. Clear search cache and rebuild indexes

**Issue: Slow search performance**

*Symptoms:*
- Search queries take longer than 5 seconds
- Timeout errors on complex searches
- Poor user experience with search interface

*Solutions:*
1. Add more specific filters to narrow search scope
2. Increase search service memory allocation
3. Optimize database indexes on commonly searched fields
4. Implement search query caching
5. Consider scaling search infrastructure

#### Lineage Issues

**Issue: Missing lineage relationships**

*Symptoms:*
- Expected lineage connections don't appear
- Lineage graph shows disconnected assets
- Impact analysis is incomplete

*Solutions:*
1. Enable lineage extraction in discovery configuration
2. Check if source system supports automatic lineage detection
3. Manually create missing lineage relationships
4. Verify SQL parsing is working for ETL jobs
5. Review transformation logic extraction settings

#### Classification Problems

**Issue: Incorrect data classifications**

*Symptoms:*
- Public data classified as PII
- Sensitive data not flagged appropriately
- Low confidence scores on obvious classifications

*Solutions:*
1. Review and adjust classification rules
2. Provide more training data for ML models
3. Update pattern matching regular expressions
4. Add business context to improve accuracy
5. Implement human-in-the-loop validation process

### Diagnostic Tools

#### Health Check Commands

**Service Health Verification:**
```python
# Check overall service health
health = await service.get_health_status()
print(f"Status: {health['status']}")
print(f"Database: {health['database_connection']}")
print(f"Search: {health['search_service']}")
print(f"Cache: {health['cache_service']}")
```

**Discovery Job Diagnostics:**
```python
# Get detailed job information
job_details = await discovery_service.get_job_details(job_id)
print(f"Status: {job_details['status']}")
print(f"Progress: {job_details['progress_percentage']}%")
print(f"Errors: {len(job_details['errors'])}")
for error in job_details['errors']:
    print(f"  - {error['message']}")
```

#### Log Analysis

**Important Log Locations:**
- Application logs: `/var/log/apg-metadata/app.log`
- Discovery logs: `/var/log/apg-metadata/discovery.log`  
- Database logs: `/var/log/postgresql/`
- Search logs: `/var/log/apg-metadata/search.log`

**Common Log Patterns:**
```bash
# Search for connection errors
grep "connection.*error" /var/log/apg-metadata/app.log

# Find discovery failures
grep "discovery.*failed" /var/log/apg-metadata/discovery.log

# Check classification issues  
grep "classification.*error" /var/log/apg-metadata/app.log

# Monitor performance issues
grep "timeout\|slow" /var/log/apg-metadata/*.log
```

#### Performance Monitoring

**Key Metrics to Monitor:**
- Discovery job success rate and duration
- Search response time and error rate  
- Database connection pool utilization
- Memory and CPU usage patterns
- Cache hit rate and eviction frequency

**Monitoring Setup:**
```python
# Enable performance monitoring
monitoring_config = {
    "enable_metrics": True,
    "metrics_endpoint": "/metrics",
    "slow_query_threshold": 5.0,
    "error_alert_threshold": 0.05
}
```

### Getting Help

#### Support Channels

**Documentation Resources:**
- User Guide (this document)
- API Reference
- Administrator Guide
- Developer Documentation

**Community Support:**
- GitHub Issues: Report bugs and feature requests
- Discussion Forums: Community Q&A
- Knowledge Base: Common solutions and workarounds

**Enterprise Support:**
- Priority technical support
- Dedicated customer success manager
- Professional services for implementation
- Custom training and workshops

#### Creating Support Requests

**Information to Include:**
1. **Environment Details:**
   - APG version and configuration
   - Database versions and setup
   - Operating system and hardware specs

2. **Problem Description:**
   - Specific error messages
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Impact on users and business operations

3. **Diagnostic Information:**
   - Relevant log file excerpts
   - Health check results
   - Configuration files (redacted)
   - Screenshot or screen recordings

4. **Troubleshooting Attempted:**
   - Steps already taken to resolve
   - Configuration changes tried
   - Workarounds currently in use

---

*For additional assistance, please contact our support team or visit the knowledge base at docs.apg-metadata.com*