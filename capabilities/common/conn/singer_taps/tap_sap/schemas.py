"""
SAP Schema Definitions
Predefined schemas for major SAP entities
"""

# Common SAP field types
SAP_DATE_TYPE = {"type": "string", "format": "date"}
SAP_DATETIME_TYPE = {"type": "string", "format": "date-time"}
SAP_CURRENCY_TYPE = {"type": "number", "multipleOf": 0.01}
SAP_QUANTITY_TYPE = {"type": "number"}
SAP_BOOLEAN_TYPE = {"type": "boolean"}

# Financial Accounting Schemas
GL_ACCOUNT_SCHEMA = {
    "type": "object",
    "properties": {
        "chart_of_accounts": {"type": "string"},
        "account_number": {"type": "string"},
        "account_group": {"type": "string"},
        "short_text": {"type": "string"},
        "long_text": {"type": "string"},
        "account_type": {"type": "string"},
        "balance_sheet_account": SAP_BOOLEAN_TYPE,
        "pl_account": SAP_BOOLEAN_TYPE,
        "blocked": SAP_BOOLEAN_TYPE,
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"}
    }
}

COST_CENTER_SCHEMA = {
    "type": "object",
    "properties": {
        "controlling_area": {"type": "string"},
        "cost_center": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "responsible_person": {"type": "string"},
        "cost_center_category": {"type": "string"},
        "hierarchy_area": {"type": "string"},
        "valid_from": SAP_DATE_TYPE,
        "valid_to": SAP_DATE_TYPE,
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"}
    }
}

ACCOUNTING_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "company_code": {"type": "string"},
        "document_number": {"type": "string"},
        "fiscal_year": {"type": "string"},
        "document_type": {"type": "string"},
        "document_date": SAP_DATE_TYPE,
        "posting_date": SAP_DATE_TYPE,
        "reference": {"type": "string"},
        "header_text": {"type": "string"},
        "currency": {"type": "string"},
        "exchange_rate": {"type": "number"},
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "created_at": {"type": "string"},
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"}
    }
}

# Material Management Schemas
MATERIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "material_type": {"type": "string"},
        "industry_sector": {"type": "string"},
        "material_group": {"type": "string"},
        "base_unit": {"type": "string"},
        "weight_unit": {"type": "string"},
        "gross_weight": SAP_QUANTITY_TYPE,
        "net_weight": SAP_QUANTITY_TYPE,
        "volume": SAP_QUANTITY_TYPE,
        "volume_unit": {"type": "string"},
        "material_description": {"type": "string"},
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"},
        "deletion_flag": SAP_BOOLEAN_TYPE
    }
}

VENDOR_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "vendor_name": {"type": "string"},
        "vendor_name_2": {"type": "string"},
        "search_term": {"type": "string"},
        "street": {"type": "string"},
        "city": {"type": "string"},
        "postal_code": {"type": "string"},
        "country": {"type": "string"},
        "region": {"type": "string"},
        "language": {"type": "string"},
        "telephone": {"type": "string"},
        "fax": {"type": "string"},
        "email": {"type": "string"},
        "tax_number": {"type": "string"},
        "industry": {"type": "string"},
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"},
        "blocked": SAP_BOOLEAN_TYPE,
        "deletion_flag": SAP_BOOLEAN_TYPE
    }
}

CUSTOMER_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "customer_name": {"type": "string"},
        "customer_name_2": {"type": "string"},
        "search_term": {"type": "string"},
        "street": {"type": "string"},
        "city": {"type": "string"},
        "postal_code": {"type": "string"},
        "country": {"type": "string"},
        "region": {"type": "string"},
        "language": {"type": "string"},
        "telephone": {"type": "string"},
        "fax": {"type": "string"},
        "email": {"type": "string"},
        "tax_number": {"type": "string"},
        "industry": {"type": "string"},
        "customer_group": {"type": "string"},
        "sales_organization": {"type": "string"},
        "distribution_channel": {"type": "string"},
        "division": {"type": "string"},
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"},
        "blocked": SAP_BOOLEAN_TYPE,
        "deletion_flag": SAP_BOOLEAN_TYPE
    }
}

# Purchase Order Schemas
PURCHASE_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "purchase_order": {"type": "string"},
        "company_code": {"type": "string"},
        "document_type": {"type": "string"},
        "vendor": {"type": "string"},
        "purchase_organization": {"type": "string"},
        "purchase_group": {"type": "string"},
        "document_date": SAP_DATE_TYPE,
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "currency": {"type": "string"},
        "exchange_rate": {"type": "number"},
        "payment_terms": {"type": "string"},
        "incoterms": {"type": "string"},
        "total_value": SAP_CURRENCY_TYPE,
        "release_indicator": {"type": "string"},
        "blocked": SAP_BOOLEAN_TYPE,
        "deletion_indicator": SAP_BOOLEAN_TYPE
    }
}

# Sales Order Schemas
SALES_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "sales_document": {"type": "string"},
        "document_type": {"type": "string"},
        "sales_organization": {"type": "string"},
        "distribution_channel": {"type": "string"},
        "division": {"type": "string"},
        "sold_to_party": {"type": "string"},
        "ship_to_party": {"type": "string"},
        "bill_to_party": {"type": "string"},
        "payer": {"type": "string"},
        "document_date": SAP_DATE_TYPE,
        "created_on": SAP_DATE_TYPE,
        "created_by": {"type": "string"},
        "currency": {"type": "string"},
        "exchange_rate": {"type": "number"},
        "payment_terms": {"type": "string"},
        "incoterms": {"type": "string"},
        "net_value": SAP_CURRENCY_TYPE,
        "tax_amount": SAP_CURRENCY_TYPE,
        "blocked": SAP_BOOLEAN_TYPE,
        "deletion_indicator": SAP_BOOLEAN_TYPE,
        "overall_status": {"type": "string"},
        "delivery_status": {"type": "string"},
        "billing_status": {"type": "string"}
    }
}

# Employee Schema (HR)
EMPLOYEE_SCHEMA = {
    "type": "object",
    "properties": {
        "personnel_number": {"type": "string"},
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "middle_name": {"type": "string"},
        "title": {"type": "string"},
        "gender": {"type": "string"},
        "birth_date": SAP_DATE_TYPE,
        "nationality": {"type": "string"},
        "employee_group": {"type": "string"},
        "employee_subgroup": {"type": "string"},
        "organizational_unit": {"type": "string"},
        "position": {"type": "string"},
        "job": {"type": "string"},
        "start_date": SAP_DATE_TYPE,
        "end_date": SAP_DATE_TYPE,
        "company_code": {"type": "string"},
        "personnel_area": {"type": "string"},
        "cost_center": {"type": "string"},
        "work_schedule": {"type": "string"},
        "created_on": SAP_DATE_TYPE,
        "changed_on": SAP_DATE_TYPE,
        "changed_by": {"type": "string"}
    }
}

# Business One Schemas
BUSINESS_ONE_BP_SCHEMA = {
    "type": "object",
    "properties": {
        "CardCode": {"type": "string"},
        "CardName": {"type": "string"},
        "CardType": {"type": "string"},
        "GroupCode": {"type": "integer"},
        "Address": {"type": "string"},
        "ZipCode": {"type": "string"},
        "MailAddress": {"type": "string"},
        "MailZipCode": {"type": "string"},
        "Phone1": {"type": "string"},
        "Phone2": {"type": "string"},
        "Fax": {"type": "string"},
        "ContactPerson": {"type": "string"},
        "Notes": {"type": "string"},
        "PayTermsGrpCode": {"type": "integer"},
        "CreditLimit": SAP_CURRENCY_TYPE,
        "MaxCommitment": SAP_CURRENCY_TYPE,
        "DiscountPercent": {"type": "number"},
        "VatStatus": {"type": "string"},
        "VatLiable": {"type": "string"},
        "ECVatGroup": {"type": "string"},
        "Currency": {"type": "string"},
        "RateDiffAccount": {"type": "string"},
        "CreateDate": SAP_DATE_TYPE,
        "UpdateDate": SAP_DATE_TYPE,
        "CreateTime": {"type": "integer"},
        "UpdateTime": {"type": "integer"},
        "Frozen": {"type": "string"},
        "FrozenFrom": SAP_DATE_TYPE,
        "FrozenTo": SAP_DATE_TYPE,
        "Valid": {"type": "string"},
        "ValidFrom": SAP_DATE_TYPE,
        "ValidTo": SAP_DATE_TYPE
    }
}

# SuccessFactors Schemas
SUCCESSFACTORS_EMPLOYEE_SCHEMA = {
    "type": "object",
    "properties": {
        "userId": {"type": "string"},
        "personIdExternal": {"type": "string"},
        "startDate": SAP_DATE_TYPE,
        "endDate": SAP_DATE_TYPE,
        "assignmentIdExternal": {"type": "string"},
        "managerId": {"type": "string"},
        "position": {"type": "string"},
        "businessUnit": {"type": "string"},
        "division": {"type": "string"},
        "department": {"type": "string"},
        "costCenter": {"type": "string"},
        "location": {"type": "string"},
        "jobCode": {"type": "string"},
        "jobTitle": {"type": "string"},
        "payGrade": {"type": "string"},
        "employmentType": {"type": "string"},
        "employeeClass": {"type": "string"},
        "customManager": {"type": "string"},
        "timezone": {"type": "string"},
        "isContingentWorker": SAP_BOOLEAN_TYPE,
        "lastModifiedDateTime": SAP_DATETIME_TYPE,
        "createdDateTime": SAP_DATETIME_TYPE
    }
}

# Master schema mapping
SAP_SCHEMAS = {
    # Financial Accounting
    "general_ledger_accounts": GL_ACCOUNT_SCHEMA,
    "cost_centers": COST_CENTER_SCHEMA,
    "accounting_documents": ACCOUNTING_DOCUMENT_SCHEMA,

    # Materials Management
    "materials": MATERIAL_SCHEMA,
    "vendors": VENDOR_SCHEMA,
    "customers": CUSTOMER_SCHEMA,
    "purchase_orders": PURCHASE_ORDER_SCHEMA,

    # Sales & Distribution
    "sales_orders": SALES_ORDER_SCHEMA,

    # Human Resources
    "employees": EMPLOYEE_SCHEMA,

    # Business One
    "business_partners": BUSINESS_ONE_BP_SCHEMA,

    # SuccessFactors
    "successfactors_employees": SUCCESSFACTORS_EMPLOYEE_SCHEMA
}


def get_schema(stream_name: str) -> dict:
    """Get predefined schema for a stream"""
    return SAP_SCHEMAS.get(stream_name, {})


def get_all_schemas() -> dict:
    """Get all predefined schemas"""
    return SAP_SCHEMAS