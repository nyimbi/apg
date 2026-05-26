"""
SAP Stream Definitions
Defines available streams for different SAP systems
"""

# SAP ERP/S4HANA Streams
ERP_STREAMS = {
    # Financial Accounting
    "general_ledger_accounts": {
        "table_name": "SKA1",
        "endpoint": "GL_ACCOUNT_MASTER",
        "key_properties": ["chart_of_accounts", "account_number"],
        "replication_method": "FULL_TABLE"
    },
    "cost_centers": {
        "table_name": "CSKS",
        "endpoint": "COST_CENTER",
        "key_properties": ["controlling_area", "cost_center"],
        "replication_method": "FULL_TABLE"
    },
    "profit_centers": {
        "table_name": "CEPC",
        "endpoint": "PROFIT_CENTER",
        "key_properties": ["controlling_area", "profit_center"],
        "replication_method": "FULL_TABLE"
    },
    "accounting_documents": {
        "table_name": "BKPF",
        "endpoint": "ACCOUNTING_DOCUMENT",
        "key_properties": ["company_code", "document_number", "fiscal_year"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "CPUDT"
    },
    "line_items": {
        "table_name": "BSEG",
        "endpoint": "ACCOUNTING_LINE_ITEM",
        "key_properties": ["company_code", "document_number", "fiscal_year", "line_item"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "CPUDT"
    },

    # Materials Management
    "materials": {
        "table_name": "MARA",
        "endpoint": "MATERIAL_MASTER",
        "key_properties": ["material"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "LAEDA"
    },
    "material_plants": {
        "table_name": "MARC",
        "endpoint": "MATERIAL_PLANT_DATA",
        "key_properties": ["material", "plant"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "LAEDA"
    },
    "vendors": {
        "table_name": "LFA1",
        "endpoint": "VENDOR_MASTER",
        "key_properties": ["vendor"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "LAEDA"
    },
    "customers": {
        "table_name": "KNA1",
        "endpoint": "CUSTOMER_MASTER",
        "key_properties": ["customer"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "LAEDA"
    },
    "purchase_orders": {
        "table_name": "EKKO",
        "endpoint": "PURCHASE_ORDER_HEADER",
        "key_properties": ["purchase_order"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "AEDAT"
    },
    "purchase_order_items": {
        "table_name": "EKPO",
        "endpoint": "PURCHASE_ORDER_ITEM",
        "key_properties": ["purchase_order", "item"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "AEDAT"
    },

    # Sales & Distribution
    "sales_orders": {
        "table_name": "VBAK",
        "endpoint": "SALES_ORDER_HEADER",
        "key_properties": ["sales_document"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "ERDAT"
    },
    "sales_order_items": {
        "table_name": "VBAP",
        "endpoint": "SALES_ORDER_ITEM",
        "key_properties": ["sales_document", "item"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "ERDAT"
    },
    "deliveries": {
        "table_name": "LIKP",
        "endpoint": "DELIVERY_HEADER",
        "key_properties": ["delivery"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "ERDAT"
    },
    "invoices": {
        "table_name": "VBRK",
        "endpoint": "BILLING_DOCUMENT",
        "key_properties": ["billing_document"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "ERDAT"
    },

    # Human Resources
    "employees": {
        "table_name": "PA0001",
        "endpoint": "EMPLOYEE_MASTER",
        "key_properties": ["personnel_number"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "AEDTM"
    },
    "organizational_units": {
        "table_name": "PA0001",
        "endpoint": "ORG_UNIT",
        "key_properties": ["org_unit"],
        "replication_method": "FULL_TABLE"
    },

    # Plant Maintenance
    "equipment": {
        "table_name": "EQUI",
        "endpoint": "EQUIPMENT_MASTER",
        "key_properties": ["equipment"],
        "replication_method": "INCREMENTAL",
        "replication_key": "changed_on",
        "date_field": "AEDAT"
    },
    "work_orders": {
        "table_name": "AUFK",
        "endpoint": "MAINTENANCE_ORDER",
        "key_properties": ["order_number"],
        "replication_method": "INCREMENTAL",
        "replication_key": "created_on",
        "date_field": "ERDAT"
    }
}

# SAP Business One Streams
BUSINESS_ONE_STREAMS = {
    "business_partners": {
        "endpoint": "BusinessPartners",
        "key_properties": ["CardCode"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "items": {
        "endpoint": "Items",
        "key_properties": ["ItemCode"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "sales_orders": {
        "endpoint": "Orders",
        "key_properties": ["DocEntry"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "purchase_orders": {
        "endpoint": "PurchaseOrders",
        "key_properties": ["DocEntry"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "invoices": {
        "endpoint": "Invoices",
        "key_properties": ["DocEntry"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "payments": {
        "endpoint": "IncomingPayments",
        "key_properties": ["DocEntry"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    },
    "general_ledger": {
        "endpoint": "ChartOfAccounts",
        "key_properties": ["Code"],
        "replication_method": "FULL_TABLE"
    },
    "inventory": {
        "endpoint": "StockTakings",
        "key_properties": ["DocEntry"],
        "replication_method": "INCREMENTAL",
        "replication_key": "UpdateDate"
    }
}

# SAP SuccessFactors Streams
SUCCESSFACTORS_STREAMS = {
    "employees": {
        "endpoint": "EmpEmployment",
        "key_properties": ["userId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "employee_personal_info": {
        "endpoint": "PerPersonal",
        "key_properties": ["personIdExternal"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "positions": {
        "endpoint": "Position",
        "key_properties": ["code"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "job_info": {
        "endpoint": "EmpJob",
        "key_properties": ["userId", "startDate"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "compensation": {
        "endpoint": "EmpCompensation",
        "key_properties": ["userId", "startDate"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "performance_reviews": {
        "endpoint": "FormContent",
        "key_properties": ["formContentId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    },
    "goals": {
        "endpoint": "SimpleGoal",
        "key_properties": ["id"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedBy"
    },
    "time_off": {
        "endpoint": "EmpTimeOff",
        "key_properties": ["userId", "startDate"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModifiedDateTime"
    }
}

# SAP Concur Streams
CONCUR_STREAMS = {
    "expense_reports": {
        "endpoint": "expense/expensereports",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    },
    "expense_entries": {
        "endpoint": "expense/entries",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    },
    "travel_requests": {
        "endpoint": "travelrequest/requests",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    },
    "vendors": {
        "endpoint": "invoice/vendors",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    },
    "users": {
        "endpoint": "user/v1.0/user",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    },
    "allocations": {
        "endpoint": "expense/allocations",
        "key_properties": ["ID"],
        "replication_method": "INCREMENTAL",
        "replication_key": "LastModified"
    }
}

# SAP Ariba Streams
ARIBA_STREAMS = {
    "procurement_documents": {
        "endpoint": "procurement/documents",
        "key_properties": ["documentId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "suppliers": {
        "endpoint": "supplier/suppliers",
        "key_properties": ["supplierId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "contracts": {
        "endpoint": "contract/workspaces",
        "key_properties": ["workspaceId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "purchase_orders": {
        "endpoint": "procurement/purchaseorders",
        "key_properties": ["orderId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "invoices": {
        "endpoint": "invoice/invoices",
        "key_properties": ["invoiceId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "sourcing_projects": {
        "endpoint": "sourcing/projects",
        "key_properties": ["projectId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    }
}

# SAP Fieldglass Streams
FIELDGLASS_STREAMS = {
    "workers": {
        "endpoint": "workers",
        "key_properties": ["workerId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "suppliers": {
        "endpoint": "suppliers",
        "key_properties": ["supplierId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "work_orders": {
        "endpoint": "workorders",
        "key_properties": ["workOrderId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "time_sheets": {
        "endpoint": "timesheets",
        "key_properties": ["timesheetId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "expenses": {
        "endpoint": "expenses",
        "key_properties": ["expenseId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    },
    "invoices": {
        "endpoint": "invoices",
        "key_properties": ["invoiceId"],
        "replication_method": "INCREMENTAL",
        "replication_key": "lastModified"
    }
}

# Master stream mapping
STREAM_MAPS = {
    "erp": ERP_STREAMS,
    "s4hana": ERP_STREAMS,  # S/4HANA uses same tables as ERP
    "business_one": BUSINESS_ONE_STREAMS,
    "successfactors": SUCCESSFACTORS_STREAMS,
    "concur": CONCUR_STREAMS,
    "ariba": ARIBA_STREAMS,
    "fieldglass": FIELDGLASS_STREAMS
}


def get_streams_for_system(system_type: str) -> dict:
    """Get available streams for a specific SAP system type"""
    return STREAM_MAPS.get(system_type, {})


def get_all_supported_systems() -> list:
    """Get list of all supported SAP system types"""
    return list(STREAM_MAPS.keys())


def get_stream_config(system_type: str, stream_name: str) -> dict:
    """Get configuration for a specific stream"""
    system_streams = STREAM_MAPS.get(system_type, {})
    return system_streams.get(stream_name, {})