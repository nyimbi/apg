# APG Capabilities - ERP Module Structure

This document describes the reorganized APG capabilities structure following enterprise ERP module conventions with three-character abbreviations for quick reference.

## Directory Structure

The capabilities are organized into horizontal ERP modules with vertical industry-specific modules:

### 1. Financial Management (FIN)
Core financial operations and accounting capabilities
- **GLR** - General Ledger 
- **APY** - Accounts Payable
- **ARC** - Accounts Receivable
- **CBM** - Cash & Bank Management
- **BFC** - Budgeting & Forecasting
- **FAM** - Fixed Asset Management
- **EXM** - Expense Management
- **TXM** - Tax Management
- **TRM** - Treasury & Risk Management
- **FCO** - Financial Consolidation

*Current implementations: billing, fed (federated learning)*

### 2. Supply Chain Management (SCM)
Complete supply chain and logistics management
- **PRC** - Procurement / Purchasing
- **INV** - Inventory Management
- **WMS** - Warehouse Management
- **OMT** - Order Management
- **SRM** - Supplier Relationship Management
- **LOG** - Logistics & Transportation
- **DPL** - Demand Planning
- **SPL** - Supply Planning
- **CTM** - Contract Management
- **RRL** - Returns & Reverse Logistics

### 3. Manufacturing & Production (MFG)
Manufacturing execution and planning systems
- **BOM** - Bill of Materials
- **PPL** - Production Planning
- **SFC** - Shop Floor Control
- **QMS** - Quality Management
- **MES** - Manufacturing Execution System
- **PLM** - Product Lifecycle Management
- **MRO** - Maintenance, Repair, Operations
- **MRP** - Material Requirements Planning
- **CAP** - Capacity Planning
- **RFM** - Recipe/Formula Management

*Current implementations: All manufacturing modules merged*

### 4. Human Capital Management (HCM)
Workforce and human resources management
- **CHR** - Core HR
- **PAY** - Payroll
- **TAT** - Time & Attendance
- **REC** - Recruitment / Talent Acquisition
- **PRF** - Performance Management
- **LND** - Learning & Development
- **BEN** - Benefits Administration
- **SCP** - Succession Planning
- **ESS** - Employee Self-Service
- **ORG** - Organizational Management

### 5. Customer Relationship Management (CRM)
Customer lifecycle and relationship management
- **SFA** - Sales Force Automation
- **CSS** - Customer Service / Support
- **MKT** - Marketing Automation
- **CAN** - Customer Analytics
- **FSM** - Field Service Management
- **CSM** - Contract & Subscription Management
- **CDP** - Customer Data Platform

### 6. Project & Portfolio Management (PPM)
Project planning, execution, and portfolio optimization
- **PPS** - Project Planning & Scheduling
- **RES** - Resource Management
- **PAC** - Project Accounting
- **TEX** - Time & Expense Tracking
- **PBL** - Project Billing
- **PAN** - Portfolio Analysis

### 7. Business Intelligence & Analytics (BIA)
Advanced analytics, reporting, and business intelligence
- **RPT** - Operational Reporting
- **DSH** - Dashboards & KPIs
- **DWH** - Data Warehousing
- **PDA** - Predictive Analytics
- **SBI** - Self-Service BI
- **PSA** - Prescriptive Analytics

### 8. Governance, Risk & Compliance (GRC)
Enterprise compliance and risk management
- **AUD** - Audit Management
- **POL** - Policy Management
- **RCM** - Regulatory Compliance
- **RSA** - Risk Assessment
- **ICM** - Internal Controls
- **DOC** - Document Management

*Current implementations: doc (document_service)*

### 9. Enterprise Asset Management (EAM)
Physical asset lifecycle and maintenance management
- **AST** - Asset Tracking
- **MSC** - Maintenance Scheduling
- **LCC** - Lifecycle Costing
- **WOM** - Work Order Management

### 10. Product Data & Engineering (PDE)
Product information and engineering data management
- **PMD** - Product Master Data
- **SPD** - Specifications & Drawings
- **ECM** - Engineering Change Management
- **CFM** - Configuration Management

### 11. E-Commerce & Digital Sales (ECD)
Digital commerce and online sales capabilities
- **WST** - Web Storefront
- **OOR** - Order Orchestration
- **CPT** - Customer Portal
- **PGI** - Payment Gateway Integration
- **DTM** - Digital Twin Marketplace

*Current implementations: dtm (digital twin marketplace)*

### 12. Collaboration & Knowledge Management (CKM)
Enterprise collaboration and knowledge management systems
- **ECN** - Enterprise Content Management
- **KBS** - Knowledge Base
- **WFA** - Workflow Automation
- **TCT** - Team Collaboration Tools

*Current implementations: notification, real_time_collaboration*

### 13. Integration & Middleware (INT)
System integration and middleware services
- **API** - API Management
- **ESB** - Enterprise Service Bus
- **ETL** - ETL / Data Integration
- **DSY** - Data Synchronization
- **IOT** - IoT Management

*Current implementations: iot (IoT management)*

### 14. Mobile & Remote Access (MOB)
Mobile applications and remote workforce capabilities
- **MAP** - Mobile ERP Apps
- **RWF** - Remote Workforce Tools

### 15. Localization & Multi-Entity (LOC)
Multi-currency, multi-language, and multi-company support
- **MCY** - Multi-Currency
- **MLG** - Multi-Language
- **MCO** - Multi-Company / Intercompany

## Industry-Specific (Vertical) Modules

### Healthcare & Medical (HCR)
Healthcare industry-specific capabilities
- **PMT** - Patient Management
- **EMR** - Electronic Medical Records
- **CLI** - Clinical Workflows
- **REG** - Regulatory Compliance
- **DEV** - Medical Device Integration
- **ANA** - Healthcare Analytics
- **TEL** - Telemedicine Platform
- **PHA** - Pharmacy Management
- **LAB** - Laboratory Information System

### Pharmaceutical & Life Sciences (PHL)
Pharmaceutical industry-specific capabilities
- **CTR** - Clinical Trial Management
- **REG** - Regulatory Affairs
- **QMS** - Quality Management System
- **MFG** - Pharmaceutical Manufacturing
- **DIS** - Drug Discovery
- **PVI** - Pharmacovigilance
- **SUP** - Supply Chain Management
- **COM** - Commercial Operations

### Energy & Utilities (ENU)
Energy and utilities management capabilities

### Telecommunications (TEL)
Telecommunications industry capabilities

### Transportation & Logistics (TRL)
Transportation and logistics management

### Real Estate & Facilities (REF)
Real estate and facility management

### Government & Public Sector (GPS)
Government and public sector capabilities

### Mining & Resources (MNR)
Mining and natural resources management

### Education & Academic (EDU)
Educational institution management

Additional vertical modules follow the same three-character naming convention.

## Common Capabilities

The `common/` directory remains for cross-cutting capabilities that don't fit into specific ERP modules:
- Authentication & authorization (auth_rbac)
- Multi-factor authentication (mfa)
- Computer vision processing
- Natural language processing (nlp)
- Biometric services
- AI orchestration
- And other foundational services

## Migration Notes

1. **Document Service**: Moved from `common/document_service` to `grc/doc`
2. **Billing**: Moved from `common/billing` to `fin/billing`
3. **Notification**: Moved from `common/notification` to `ckm/notification`
4. **Real-time Collaboration**: Moved from `common/real_time_collaboration` to `ckm/real_time_collaboration`
5. **Manufacturing**: Merged `manufacturing/` directory into `mfg/` with 3-character sub-modules:
   - Bill of Materials → `mfg/bom`
   - Production Planning → `mfg/ppl`
   - Shop Floor Control → `mfg/sfc`
   - Quality Management → `mfg/qms`
   - Manufacturing Execution System → `mfg/mes`
   - Maintenance/Predictive Maintenance → `mfg/mro`
   - Material Requirements Planning → `mfg/mrp` (new)
   - Capacity Planning → `mfg/cap` (new)
   - Recipe/Formula Management → `mfg/rfm` (new)
6. **IoT Management**: Moved from `iot_management/` to `int/iot`
7. **Digital Twin Marketplace**: Moved from `digital_twin_marketplace/` to `ecd/dtm`
8. **Federated Learning**: Moved from `federated_learning/` to `fin/fed`
9. **Platform Foundation**: Moved to `common/platform_foundation`
10. **Security Operations**: Moved to `common/security_operations`
11. **Industry Vertical Solutions**: Reorganized into individual industry directories:
   - `industry_vertical_solutions/healthcare_medical` → `healthcare/`
   - `industry_vertical_solutions/pharmaceutical_life_sciences` → `pharma/`
   - `industry_vertical_solutions/energy_utilities` → `energy/`
   - `industry_vertical_solutions/telecommunications` → `telecom/`
   - `industry_vertical_solutions/transportation_logistics` → `transport/`
   - `industry_vertical_solutions/real_estate_facilities` → `realestate/`
   - `industry_vertical_solutions/government_public_sector` → `government/`
   - `industry_vertical_solutions/mining_resources` → `mining/`
   - `industry_vertical_solutions/education_academic` → `education/`

## Usage

Each capability can be imported using its full path:

```python
# Import document management capability
from apg.capabilities.grc.doc import APGDocumentService

# Import billing capability  
from apg.capabilities.fin.billing import BillingService

# Import notification capability
from apg.capabilities.ckm.notification import NotificationService
```

## Development Guidelines

1. **Naming**: Use three-character abbreviations for all modules
2. **Structure**: Each capability should have:
   - `__init__.py` - Package initialization
   - `service.py` - Main service implementation
   - `models.py` - Data models
   - `views.py` - API views/endpoints
   - `blueprint.py` - Flask-AppBuilder integration (if applicable)
   - `cap_spec.md` - Capability specification
   - `todo.md` - Development tracking

3. **Documentation**: Each module should include comprehensive documentation
4. **Testing**: All capabilities should include unit and integration tests
5. **APG Integration**: Follow APG patterns for composition, security, and audit

© 2025 Datacraft. All rights reserved.