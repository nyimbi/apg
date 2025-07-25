# APG Hierarchical Capability Architecture

## Overview
Restructure APG capabilities into a hierarchical, composable system where:
- **Capabilities** are main functional areas (Core Financials, Manufacturing, etc.)
- **Sub-capabilities** are specific modules within capabilities that can be independently composed
- Each sub-capability is a full Python package with its own models, views, services, and APIs
- APG programmers can compose custom solutions by selecting specific sub-capabilities

## Directory Structure

```
apg/
├── capabilities/
│   ├── core_financials/
│   │   ├── __init__.py
│   │   ├── models.py                    # Shared models for the capability
│   │   ├── views.py                     # Capability dashboard and overview
│   │   ├── service.py                   # Capability-level orchestration
│   │   ├── general_ledger/              # Sub-capability
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── views.py
│   │   │   ├── service.py
│   │   │   ├── api.py
│   │   │   └── blueprint.py
│   │   ├── accounts_payable/
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── views.py
│   │   │   ├── service.py
│   │   │   ├── api.py
│   │   │   └── blueprint.py
│   │   ├── accounts_receivable/
│   │   ├── cash_management/
│   │   ├── fixed_asset_management/
│   │   ├── budgeting_forecasting/
│   │   ├── financial_reporting/
│   │   └── cost_accounting/
│   ├── human_resources/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── service.py
│   │   ├── payroll/
│   │   ├── time_attendance/
│   │   ├── employee_data_management/
│   │   ├── recruitment_onboarding/
│   │   ├── performance_management/
│   │   ├── benefits_administration/
│   │   └── learning_development/
│   ├── manufacturing/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── service.py
│   │   ├── production_planning/
│   │   ├── material_requirements_planning/
│   │   ├── shop_floor_control/
│   │   ├── bill_of_materials/
│   │   ├── capacity_planning/
│   │   ├── quality_management/
│   │   ├── recipe_formula_management/
│   │   └── manufacturing_execution_system/
│   └── [other capabilities...]
├── composition/
│   ├── __init__.py
│   ├── composer.py                      # Sub-capability composition engine
│   ├── registry.py                      # Sub-capability registry
│   └── blueprints.py                    # Blueprint management
└── [other APG directories...]
```

## Sub-Capability Structure

Each sub-capability follows a consistent structure:

```python
# Example: core_financials/general_ledger/
__init__.py          # Package initialization and metadata
models.py           # Database models specific to GL
views.py            # Flask-AppBuilder views for GL
service.py          # Business logic and API services
api.py              # REST API endpoints
blueprint.py        # Flask blueprint registration
config.py           # Sub-capability specific configuration
requirements.txt    # Sub-capability dependencies
README.md           # Documentation
```

## Composition System

### Sub-Capability Metadata
Each sub-capability defines metadata for composition:

```python
# core_financials/general_ledger/__init__.py
SUBCAPABILITY_META = {
    'name': 'General Ledger',
    'code': 'GL',
    'version': '1.0.0',
    'capability': 'core_financials',
    'description': 'Central repository for all financial transactions',
    'industry_focus': 'All',
    'dependencies': [],
    'optional_dependencies': ['accounts_payable', 'accounts_receivable'],
    'database_tables': ['gl_account', 'gl_journal_entry', 'gl_transaction'],
    'api_endpoints': ['/api/gl/accounts', '/api/gl/journals', '/api/gl/reports'],
    'views': ['GLAccountModelView', 'GLJournalModelView', 'GLDashboardView'],
    'permissions': ['gl.read', 'gl.write', 'gl.admin'],
    'menu_items': [
        {'name': 'Chart of Accounts', 'endpoint': 'gl.accounts'},
        {'name': 'Journal Entries', 'endpoint': 'gl.journals'},
        {'name': 'GL Reports', 'endpoint': 'gl.reports'}
    ]
}
```

### Composition Engine
APG programmers can compose applications by selecting sub-capabilities:

```python
# Example composition
from apg.composition import APGComposer

composer = APGComposer()

# Add sub-capabilities
composer.add_subcapability('core_financials.general_ledger')
composer.add_subcapability('core_financials.accounts_payable')
composer.add_subcapability('core_financials.accounts_receivable')
composer.add_subcapability('manufacturing.production_planning')
composer.add_subcapability('manufacturing.material_requirements_planning')

# Validate dependencies
composer.validate_composition()

# Generate application
app = composer.create_application(
    name='Manufacturing ERP',
    database_url='postgresql://...',
    additional_config={...}
)
```

## Industry-Specific Templates

Pre-defined compositions for different industries:

```python
# Manufacturing template
MANUFACTURING_TEMPLATE = [
    'core_financials.general_ledger',
    'core_financials.accounts_payable', 
    'core_financials.accounts_receivable',
    'core_financials.cost_accounting',
    'manufacturing.production_planning',
    'manufacturing.material_requirements_planning',
    'manufacturing.shop_floor_control',
    'manufacturing.bill_of_materials',
    'manufacturing.quality_management',
    'inventory_management.stock_tracking_control',
    'inventory_management.batch_lot_tracking',
    'procurement_purchasing.purchase_order_management',
    'procurement_purchasing.vendor_management'
]

# Pharmaceutical template  
PHARMACEUTICAL_TEMPLATE = [
    'core_financials.general_ledger',
    'core_financials.accounts_payable',
    'manufacturing.recipe_formula_management',
    'manufacturing.quality_management', 
    'pharmaceutical_specific.regulatory_compliance',
    'pharmaceutical_specific.product_serialization_tracking',
    'pharmaceutical_specific.batch_release_management',
    'inventory_management.batch_lot_tracking',
    'inventory_management.expiry_date_management'
]
```

## Benefits

1. **Modularity**: Only include needed functionality
2. **Industry Focus**: Easy to create industry-specific solutions
3. **Maintenance**: Smaller, focused codebases per sub-capability
4. **Testing**: Independent testing of sub-capabilities
5. **Deployment**: Selective deployment of features
6. **Customization**: Easy to customize specific modules without affecting others
7. **Scalability**: Teams can work on different sub-capabilities independently

## Migration Path

1. **Phase 1**: Create new directory structure
2. **Phase 2**: Migrate existing capabilities to hierarchical structure
3. **Phase 3**: Implement composition engine
4. **Phase 4**: Create industry templates
5. **Phase 5**: Add advanced composition features (conditional loading, feature flags, etc.)

## Implementation Priority

Based on the CSV, implement in this order:
1. **Core Financials** (foundational for all ERP)
2. **Human Resources** (universal need)
3. **Manufacturing** (complex but well-defined)
4. **Inventory Management** (supports multiple industries)
5. **Sales & Order Management** (customer-facing)
6. **Supply Chain Management** (operational efficiency)
7. Industry-specific capabilities (Pharmaceutical, Mining, etc.)