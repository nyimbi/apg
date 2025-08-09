"""
APG Customer Relationship Management - Test Suite

Comprehensive test suite for the CRM capability providing unit tests,
integration tests, and end-to-end testing with complete coverage of
all CRM functionality and APG integrations.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

__version__ = "1.0.0"

# Test configuration
TEST_DATABASE_CONFIG = {
	"host": "localhost",
	"port": 5432,
	"database": "crm_test_db",
	"user": "crm_test_user", 
	"password": "crm_test_password"
}

# Test data constants
TEST_TENANT_ID = "test_tenant_123"
TEST_USER_ID = "test_user_456"
TEST_CONTACT_ID = "test_contact_789"
TEST_ACCOUNT_ID = "test_account_101"
TEST_LEAD_ID = "test_lead_102"
TEST_OPPORTUNITY_ID = "test_opportunity_103"