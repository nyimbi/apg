# Enterprise Asset Management - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Asset Management](#asset-management)
4. [Work Order Management](#work-order-management)
5. [Maintenance Operations](#maintenance-operations)
6. [Inventory Management](#inventory-management)
7. [Performance Analytics](#performance-analytics)
8. [Mobile Application](#mobile-application)
9. [Reports and Compliance](#reports-and-compliance)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Initial Setup

#### 1. Access the System
- **URL**: https://eam.apg.datacraft.co.ke
- **Login**: Use your APG platform credentials
- **Multi-factor Authentication**: Complete 2FA setup if required

#### 2. First-Time Configuration
After logging in, you'll be guided through initial setup:

1. **Organization Profile**: Configure your company details
2. **Location Hierarchy**: Set up your facility structure
3. **Asset Categories**: Define asset types for your organization
4. **User Permissions**: Assign role-based access controls

#### 3. Navigation Overview
The EAM interface is organized into main sections:

```
Main Navigation:
├── Dashboard - KPI overview and alerts
├── Asset Management - Asset lifecycle and hierarchy
├── Work Orders - Maintenance and repair tasks
├── Inventory - Parts and materials management
├── Analytics - Performance insights and reports
└── Settings - Configuration and preferences
```

### User Roles and Permissions

| Role | Permissions | Typical Users |
|------|-------------|---------------|
| **EAM Administrator** | Full system access, configuration | IT Manager, EAM System Admin |
| **Maintenance Manager** | Manage work orders, view analytics | Maintenance Supervisor |
| **Maintenance Technician** | Execute work orders, update status | Field Technicians |
| **Asset Manager** | Manage asset data, lifecycle | Asset Coordinators |
| **Inventory Manager** | Manage parts, stock levels | Warehouse Staff |
| **Viewer** | Read-only access to reports | Management, Auditors |

## Dashboard Overview

### Main Dashboard

The dashboard provides real-time insights into your asset management operations:

#### Key Performance Indicators (KPIs)

1. **Asset Health Overview**
   - Average health score across all assets
   - Number of assets requiring attention
   - Critical health alerts

2. **Work Order Status**
   - Open work orders by priority
   - Overdue maintenance tasks
   - Completion rates

3. **Inventory Status**
   - Low stock alerts
   - Items requiring reorder
   - Total inventory value

4. **Performance Metrics**
   - Overall Equipment Effectiveness (OEE)
   - Mean Time Between Failures (MTBF)
   - Maintenance cost trends

#### Dashboard Widgets

**Asset Health Distribution**
- Pie chart showing assets by condition status
- Click to drill down into specific categories

**Maintenance Calendar**
- Monthly view of scheduled maintenance
- Color-coded by priority and status
- Drag-and-drop rescheduling

**Critical Alerts**
- Real-time notifications for urgent issues
- Direct links to related assets or work orders

**Cost Analytics**
- Monthly maintenance spending trends
- Budget vs. actual comparisons

### Customizing Your Dashboard

1. **Add/Remove Widgets**: Click the gear icon to customize
2. **Resize Widgets**: Drag corners to adjust size
3. **Rearrange Layout**: Drag widgets to reposition
4. **Filter Data**: Set date ranges and asset filters

## Asset Management

### Creating Assets

#### Step 1: Asset Basic Information
```
Navigation: Asset Management > Assets > New Asset
```

**Required Fields:**
- Asset Name: Descriptive name (e.g., "CNC Machine #1")
- Asset Number: Unique identifier (auto-generated or manual)
- Asset Type: Equipment, Vehicle, Facility, Infrastructure
- Location: Select from hierarchy

**Optional Fields:**
- Manufacturer, Model, Serial Number
- Purchase Cost, Installation Date
- Criticality Level (Low, Medium, High, Critical)

#### Step 2: Technical Details
- Specifications and capabilities
- Operating parameters
- Warranty information
- Digital twin configuration

#### Step 3: Maintenance Configuration
- Maintenance strategy (Reactive, Preventive, Predictive)
- Frequency and scheduling
- Required skills and resources
- Safety considerations

### Asset Hierarchy

#### Creating Location Structure
```
Example Hierarchy:
Site: Manufacturing Plant
├── Building: Production Building A
│   ├── Floor: Ground Floor
│   │   ├── Area: Production Line 1
│   │   └── Area: Quality Control
│   └── Floor: Mezzanine
└── Building: Warehouse
```

#### Managing Parent-Child Relationships
- **Parent Assets**: Major equipment or systems
- **Child Assets**: Components or sub-assemblies
- **Benefits**: Cascading maintenance, cost rollup, failure analysis

### Asset Lifecycle Management

#### 1. Planning Phase
- Asset requisition and approval
- Vendor selection and procurement
- Installation planning

#### 2. Acquisition Phase
- Purchase order processing
- Delivery and inspection
- Asset registration

#### 3. Installation Phase
- Site preparation and installation
- Commissioning and testing
- Training and documentation

#### 4. Operation Phase
- Regular monitoring and maintenance
- Performance tracking
- Condition assessment

#### 5. Retirement Phase
- End-of-life planning
- Disposal or recycling
- Asset decommissioning

### Asset Health Monitoring

#### Health Score Calculation
The system calculates asset health based on:
- Condition assessments
- Performance metrics
- Maintenance history
- Age and usage patterns

#### Health Score Ranges
- **90-100%**: Excellent - Operating optimally
- **80-89%**: Good - Normal operation with minor issues
- **70-79%**: Fair - Requires attention soon
- **60-69%**: Poor - Immediate action needed
- **Below 60%**: Critical - Risk of failure

#### Setting Health Thresholds
```
Navigation: Assets > [Select Asset] > Health Configuration
```
- Define custom thresholds for your assets
- Set automatic alert triggers
- Configure escalation procedures

## Work Order Management

### Creating Work Orders

#### 1. Basic Work Order Information
```
Navigation: Work Orders > New Work Order
```

**Required Fields:**
- Title: Brief description of work
- Work Type: Maintenance, Repair, Inspection, Installation
- Priority: Low, Medium, High, Emergency
- Asset: Select from asset hierarchy

**Optional Fields:**
- Description: Detailed work instructions
- Estimated Duration and Cost
- Required Skills and Crew Size
- Safety Requirements

#### 2. Scheduling
- Requested Date: When work is requested
- Scheduled Start/End: Planned execution window
- Assignment: Technician or team
- Resource Requirements: Tools, parts, equipment

#### 3. Planning Details
- Work Instructions: Step-by-step procedures
- Safety Precautions: Required PPE and protocols
- Parts List: Required materials and quantities
- Documentation: Drawings, manuals, procedures

### Work Order Status Workflow

```
Draft → Submitted → Approved → Scheduled → In Progress → Completed → Closed
```

#### Status Descriptions
- **Draft**: Work order being created
- **Submitted**: Awaiting supervisor approval
- **Approved**: Authorized for scheduling
- **Scheduled**: Assigned and planned
- **In Progress**: Actively being executed
- **Completed**: Work finished, pending review
- **Closed**: Final review complete

### Executing Work Orders

#### For Maintenance Technicians

1. **Access Assigned Work Orders**
   ```
   Navigation: Work Orders > My Assignments
   ```

2. **Start Work Order**
   - Review work instructions and safety requirements
   - Gather required tools and parts
   - Update status to "In Progress"

3. **Document Work Performed**
   - Record actual time spent
   - Note any issues encountered
   - Photograph work completed
   - Update asset condition

4. **Complete Work Order**
   - Verify all work completed
   - Record parts consumed
   - Rate work quality
   - Submit for closure

#### Mobile Work Order Execution
- Offline capability for field work
- Photo and signature capture
- Barcode scanning for assets/parts
- Voice-to-text work notes

### Work Order Types

#### Preventive Maintenance
- Regularly scheduled maintenance tasks
- Based on time, usage, or condition
- Helps prevent unexpected failures

#### Corrective Maintenance
- Repairs in response to failures
- Emergency and non-emergency repairs
- Root cause analysis documentation

#### Predictive Maintenance
- Condition-based maintenance
- Triggered by sensor data or inspections
- Optimizes timing for maximum efficiency

#### Project Work
- Major installations or upgrades
- Multi-phase work requiring planning
- Resource coordination and tracking

## Maintenance Operations

### Preventive Maintenance Program

#### Setting Up PM Schedules

1. **Define Maintenance Tasks**
   ```
   Navigation: Assets > [Select Asset] > Maintenance Plans
   ```
   - Task descriptions and procedures
   - Required skills and resources
   - Safety requirements
   - Quality standards

2. **Create Scheduling Rules**
   - **Time-based**: Every 30 days, quarterly, annually
   - **Usage-based**: Every 1000 hours, 10,000 cycles
   - **Condition-based**: When parameter exceeds threshold

3. **Generate Work Orders**
   - Automatic work order creation
   - Lead time for planning
   - Resource availability checking

#### Maintenance Task Types

**Inspection Tasks**
- Visual inspections
- Measurement and readings
- Condition assessments
- Compliance checks

**Service Tasks**
- Lubrication and adjustments
- Cleaning and calibration
- Component replacement
- System testing

**Overhaul Tasks**
- Major component replacement
- System rebuilds
- Comprehensive testing
- Documentation updates

### Predictive Maintenance

#### Condition Monitoring

**Vibration Analysis**
- Equipment condition assessment
- Early fault detection
- Trend analysis and alarming

**Thermal Imaging**
- Electrical connection monitoring
- Bearing condition assessment
- Heat distribution analysis

**Oil Analysis**
- Contamination detection
- Wear particle analysis
- Additive depletion monitoring

**Performance Monitoring**
- Energy consumption tracking
- Output quality monitoring
- Efficiency trending

#### Integration with Digital Twins
- Real-time sensor data analysis
- Predictive failure modeling
- Optimal maintenance timing
- Virtual testing and simulation

### Maintenance Planning and Scheduling

#### Weekly Planning Process

1. **Review Upcoming Work**
   - Scheduled preventive maintenance
   - Outstanding corrective work
   - Emergency priorities

2. **Resource Allocation**
   - Technician assignments
   - Tool and equipment scheduling
   - Parts availability verification

3. **Coordination Meetings**
   - Production schedule alignment
   - Safety planning and permits
   - Cross-functional coordination

#### Shutdown Planning
- Major maintenance events
- Multi-department coordination
- Critical path scheduling
- Resource mobilization

## Inventory Management

### Parts and Materials Management

#### Creating Inventory Items

```
Navigation: Inventory > Items > New Item
```

**Basic Information:**
- Part Number: Unique identifier
- Description: Clear part description
- Category: Organizational grouping
- Type: Spare Part, Consumable, Tool, Material

**Stock Management:**
- Current Stock Level
- Minimum/Maximum Stock Levels
- Reorder Point and Quantity
- Storage Location

**Vendor Information:**
- Primary Supplier
- Lead Time
- Unit Cost and Pricing
- Alternative Suppliers

#### Stock Level Management

**Setting Stock Levels:**
- **Minimum Stock**: Safety stock level
- **Maximum Stock**: Prevent overstocking
- **Reorder Point**: Trigger for replenishment
- **Reorder Quantity**: Economic order quantity

**Stock Level Calculation:**
```
Reorder Point = (Lead Time × Usage Rate) + Safety Stock
```

### Inventory Transactions

#### Stock Movements

**Issue Transactions**
- Parts used for work orders
- Regular consumption tracking
- Return of unused materials

**Receipt Transactions**
- Purchase order deliveries
- Stock adjustments
- Physical count corrections

**Transfer Transactions**
- Between storage locations
- Between facilities
- Department allocations

#### Physical Inventory

**Cycle Counting**
- Regular counting of high-value items
- ABC analysis prioritization
- Variance investigation and correction

**Annual Physical Count**
- Complete inventory verification
- System reconciliation
- Process improvement identification

### Procurement Integration

#### Purchase Requisitions
- Automatic generation for low stock
- Approval workflows
- Vendor selection and pricing

#### Purchase Orders
- Order creation and approval
- Delivery tracking
- Receipt verification and matching

#### Vendor Management
- Supplier performance tracking
- Cost analysis and negotiation
- Quality and delivery metrics

### Cost Management

#### Inventory Valuation
- First-In-First-Out (FIFO) costing
- Weighted average cost calculation
- Standard cost variance analysis

#### Cost Control
- Inventory aging reports
- Slow-moving stock identification
- Obsolete inventory management

## Performance Analytics

### Asset Performance Metrics

#### Overall Equipment Effectiveness (OEE)

**OEE Calculation:**
```
OEE = Availability × Performance × Quality

Where:
- Availability = Operating Time / Planned Production Time
- Performance = (Total Count / Operating Time) / Ideal Run Rate
- Quality = Good Count / Total Count
```

**Interpreting OEE:**
- **85%+**: World-class performance
- **60-85%**: Acceptable performance
- **40-60%**: Needs improvement
- **<40%**: Unacceptable performance

#### Key Performance Indicators

**Reliability Metrics:**
- Mean Time Between Failures (MTBF)
- Mean Time To Repair (MTTR)
- Equipment availability percentage

**Maintenance Metrics:**
- Preventive vs. corrective maintenance ratio
- Maintenance cost per asset
- Schedule compliance percentage

**Cost Metrics:**
- Total cost of ownership
- Maintenance cost trends
- Return on maintenance investment

### Reporting and Dashboards

#### Standard Reports

**Asset Reports:**
- Asset register and hierarchy
- Asset health and condition summary
- Depreciation and valuation reports

**Maintenance Reports:**
- Work order completion analysis
- Maintenance cost analysis
- Downtime and availability reports

**Inventory Reports:**
- Stock status and valuation
- Purchase analysis and vendor performance
- Usage and consumption patterns

#### Custom Analytics

**Report Builder:**
- Drag-and-drop report creation
- Custom calculations and formulas
- Flexible filtering and grouping

**Dashboard Designer:**
- Personalized KPI dashboards
- Real-time data visualization
- Alert and notification setup

### Predictive Analytics

#### Failure Prediction
- Machine learning models for failure prediction
- Risk assessment and prioritization
- Recommended maintenance actions

#### Optimization Recommendations
- Maintenance schedule optimization
- Inventory level optimization
- Resource allocation suggestions

## Mobile Application

### Mobile Features

#### Work Order Management
- View assigned work orders
- Update work order status
- Capture photos and signatures
- Record time and materials

#### Asset Information
- Search and browse assets
- View asset details and history
- Update asset condition
- Create inspection reports

#### Inventory Operations
- Check stock levels
- Record stock movements
- Request parts and materials
- Scan barcodes for accuracy

### Offline Capability

#### Offline Work
- Download work orders for offline access
- Complete work without network connectivity
- Sync data when connection restored
- Conflict resolution for simultaneous updates

#### Data Synchronization
- Automatic sync when online
- Manual sync option available
- Progress indicators and status
- Error handling and retry logic

### Mobile App Installation

#### iOS Installation
1. Download from App Store: "APG EAM Mobile"
2. Enter server URL and credentials
3. Complete setup wizard
4. Enable push notifications

#### Android Installation
1. Download from Google Play: "APG EAM Mobile"
2. Configure server connection
3. Set up user authentication
4. Enable required permissions

## Reports and Compliance

### Regulatory Compliance

#### Audit Trail
- Complete history of all changes
- User identification and timestamps
- Before/after values for modifications
- Regulatory compliance documentation

#### Standard Compliance
- ISO 55000 Asset Management
- OSHA Safety Requirements
- Environmental Regulations
- Industry-Specific Standards

### Report Generation

#### Scheduled Reports
- Automatic report generation
- Email delivery to stakeholders
- Multiple format options (PDF, Excel, CSV)
- Custom distribution lists

#### Ad-Hoc Reporting
- On-demand report creation
- Interactive filters and parameters
- Drill-down capabilities
- Export and sharing options

### Data Export

#### Export Formats
- Microsoft Excel (.xlsx)
- Comma Separated Values (.csv)
- Portable Document Format (.pdf)
- JavaScript Object Notation (.json)

#### Bulk Data Export
- Complete asset database
- Historical maintenance records
- Inventory transactions
- Performance metrics

## Troubleshooting

### Common Issues and Solutions

#### Login and Access Issues

**Problem**: Cannot log in to the system
**Solutions:**
1. Verify username and password
2. Check caps lock and keyboard language
3. Clear browser cache and cookies
4. Contact system administrator for account status

**Problem**: Access denied to specific features
**Solutions:**
1. Verify user role and permissions
2. Contact supervisor for access request
3. Check multi-tenant access rights
4. Ensure proper license assignment

#### Performance Issues

**Problem**: Slow system response
**Solutions:**
1. Check internet connection speed
2. Clear browser cache and temporary files
3. Close unnecessary browser tabs
4. Contact IT support for server status

**Problem**: Reports taking too long to load
**Solutions:**
1. Reduce date range for reports
2. Apply filters to limit data scope
3. Schedule large reports for off-peak hours
4. Contact system administrator

#### Data Issues

**Problem**: Missing or incorrect data
**Solutions:**
1. Verify data entry permissions
2. Check data synchronization status
3. Review recent system updates
4. Submit support ticket with details

**Problem**: Unable to save changes
**Solutions:**
1. Check required field completion
2. Verify data format and validation rules
3. Ensure adequate system permissions
4. Try refreshing the page and retry

### Getting Help

#### Self-Service Resources
- **User Guide**: This comprehensive guide
- **Video Tutorials**: Step-by-step demonstrations
- **FAQ Section**: Common questions and answers
- **Community Forum**: User discussions and tips

#### Support Channels
- **Help Desk**: Submit support tickets online
- **Phone Support**: Call during business hours
- **Email Support**: Technical assistance via email
- **Live Chat**: Real-time support during peak hours

#### Training Resources
- **New User Orientation**: Introduction to system basics
- **Role-Based Training**: Specific training for user roles
- **Advanced Features**: Training on complex functionality
- **Administrator Training**: System configuration and management

### Best Practices

#### Data Quality
- Enter complete and accurate information
- Use standardized naming conventions
- Regularly review and update data
- Maintain consistent data formats

#### Security
- Use strong passwords and change regularly
- Log out when leaving workstation
- Report suspicious activity immediately
- Follow data handling procedures

#### Efficiency
- Use keyboard shortcuts for common tasks
- Leverage bulk operations when possible
- Set up personalized dashboards
- Automate routine processes where available

---

*User Guide Version 1.0 - Last Updated: 2024-01-01*

For additional support, contact:
- **Email**: support@datacraft.co.ke
- **Phone**: +254-XXX-XXXXXX
- **Website**: https://www.datacraft.co.ke/support