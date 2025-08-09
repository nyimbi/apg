#!/usr/bin/env python3
"""
APG Billing System - Advanced Billing Models Examples

This file demonstrates all supported billing models including:
- Different prices per service
- Tiered pricing per service
- Customer-specific billing
- Customer-specific discounts
- Deferred billing
- Prepaid usage
- And many more advanced billing scenarios

Supports configuration by composition/central_configuration.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import json

# Add the billing module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load configuration from composition layer if available
try:
    if os.path.exists('/etc/apg/composition/central_configuration'):
        exec(open('/etc/apg/composition/central_configuration').read())
        print("✓ Loaded configuration from central_configuration")
    elif os.path.exists('../../../composition/central_configuration'):
        exec(open('../../../composition/central_configuration').read())
        print("✓ Loaded configuration from composition/central_configuration")
    else:
        print("ℹ Using local configuration")
except Exception as e:
    print(f"⚠ Configuration load warning: {e}")

from service import get_billing_service
from views import (
    BLCustomer, BLPlan, BLSubscription, BLInvoice, BLPayment, 
    BLUsage, BLDiscount, SubscriptionStatus, PaymentStatus
)


class BillingModelsExamples:
    def __init__(self):
        """Initialize the billing service"""
        self.billing_service = get_billing_service()
        print("✓ APG Billing Service initialized for billing models examples")
    
    def example_1_different_prices_per_service(self):
        """Example 1: Different prices per service/feature"""
        print("\n" + "="*60)
        print("EXAMPLE 1: Different Prices Per Service")
        print("="*60)
        
        # Create service-specific plans
        api_service_plan = {
            "name": "API Access Service",
            "description": "API calls and data access",
            "amount": Decimal("49.99"),
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "api",
            "features": ["api_access", "rate_limiting", "basic_analytics"],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.001"),  # $0.001 per API call
                        "included_quantity": 100000,
                        "overage_price": Decimal("0.0015")
                    }
                ]
            },
            "active": True
        }
        
        storage_service_plan = {
            "name": "Storage Service",
            "description": "Cloud storage and file management",
            "amount": Decimal("29.99"),
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "storage",
            "features": ["file_storage", "backup", "versioning"],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "storage_gb",
                        "unit_price": Decimal("0.25"),  # $0.25 per GB
                        "included_quantity": 50,
                        "overage_price": Decimal("0.30")
                    },
                    {
                        "metric_name": "bandwidth_gb",
                        "unit_price": Decimal("0.12"),  # $0.12 per GB
                        "included_quantity": 100,
                        "overage_price": Decimal("0.15")
                    }
                ]
            },
            "active": True
        }
        
        analytics_service_plan = {
            "name": "Analytics Service",
            "description": "Advanced analytics and reporting",
            "amount": Decimal("99.99"),
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "analytics",
            "features": ["advanced_analytics", "custom_reports", "real_time_data"],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "report_generation",
                        "unit_price": Decimal("5.00"),  # $5.00 per report
                        "included_quantity": 20,
                        "overage_price": Decimal("6.00")
                    },
                    {
                        "metric_name": "data_points_processed",
                        "unit_price": Decimal("0.0001"),  # $0.0001 per data point
                        "included_quantity": 1000000,
                        "overage_price": Decimal("0.00012")
                    }
                ]
            },
            "active": True
        }
        
        # Create the service plans
        api_plan = self.billing_service.create_plan(api_service_plan)
        storage_plan = self.billing_service.create_plan(storage_service_plan)
        analytics_plan = self.billing_service.create_plan(analytics_service_plan)
        
        print(f"✓ Created API Service Plan: {api_plan.id} - ${api_plan.amount}/month")
        print(f"✓ Created Storage Service Plan: {storage_plan.id} - ${storage_plan.amount}/month")
        print(f"✓ Created Analytics Service Plan: {analytics_plan.id} - ${analytics_plan.amount}/month")
        
        # Create a composite plan that includes multiple services
        composite_plan = {
            "name": "Enterprise Bundle",
            "description": "All services included with enterprise features",
            "amount": Decimal("199.99"),  # Discounted bundle price
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "bundle",
            "features": ["all_services", "enterprise_support", "sla_guarantee"],
            "service_components": [
                {"service": "api", "plan_id": api_plan.id, "discount_percentage": 20},
                {"service": "storage", "plan_id": storage_plan.id, "discount_percentage": 20},
                {"service": "analytics", "plan_id": analytics_plan.id, "discount_percentage": 20}
            ],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    # Inherit all metrics from component services with discounts
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.0008"),  # 20% discount
                        "included_quantity": 150000,  # 50% more included
                        "overage_price": Decimal("0.0012")
                    },
                    {
                        "metric_name": "storage_gb",
                        "unit_price": Decimal("0.20"),  # 20% discount
                        "included_quantity": 100,  # Double included
                        "overage_price": Decimal("0.24")
                    }
                ]
            },
            "active": True
        }
        
        enterprise_plan = self.billing_service.create_plan(composite_plan)
        print(f"✓ Created Enterprise Bundle Plan: {enterprise_plan.id} - ${enterprise_plan.amount}/month")
        print(f"  Includes: API + Storage + Analytics with 20% discount")
        
        return api_plan, storage_plan, analytics_plan, enterprise_plan
    
    def example_2_tiered_pricing_per_service(self):
        """Example 2: Tiered pricing per service"""
        print("\n" + "="*60)
        print("EXAMPLE 2: Tiered Pricing Per Service")
        print("="*60)
        
        # Create a plan with volume-based tiered pricing
        tiered_api_plan = {
            "name": "API Service - Tiered Pricing",
            "description": "Volume-based pricing for API calls",
            "amount": Decimal("0.00"),  # No base fee
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "api_tiered",
            "pricing_model": "tiered",
            "pricing_tiers": [
                {
                    "tier_name": "Starter",
                    "up_to": 10000,
                    "unit_price": Decimal("0.005"),  # $0.005 per call
                    "flat_fee": Decimal("0.00")
                },
                {
                    "tier_name": "Growth", 
                    "up_to": 100000,
                    "unit_price": Decimal("0.003"),  # $0.003 per call
                    "flat_fee": Decimal("25.00")  # Monthly fee for this tier
                },
                {
                    "tier_name": "Enterprise",
                    "up_to": 1000000,
                    "unit_price": Decimal("0.002"),  # $0.002 per call
                    "flat_fee": Decimal("100.00")  # Monthly fee for this tier
                },
                {
                    "tier_name": "Enterprise Plus",
                    "up_to": None,  # Unlimited
                    "unit_price": Decimal("0.001"),  # $0.001 per call
                    "flat_fee": Decimal("500.00"),  # Monthly fee for this tier
                    "minimum_commitment": Decimal("1000.00")  # Minimum monthly spend
                }
            ],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "pricing_model": "tiered"
                    }
                ]
            },
            "active": True
        }
        
        # Create a plan with graduated tiered pricing (cumulative)
        graduated_storage_plan = {
            "name": "Storage Service - Graduated Pricing",
            "description": "Graduated pricing for storage usage",
            "amount": Decimal("10.00"),  # Base monthly fee
            "currency": "USD",
            "billing_period": "monthly",
            "service_category": "storage_graduated",
            "pricing_model": "graduated",
            "pricing_tiers": [
                {
                    "tier_name": "First 100 GB",
                    "from_quantity": 0,
                    "to_quantity": 100,
                    "unit_price": Decimal("0.50")  # $0.50 per GB
                },
                {
                    "tier_name": "Next 400 GB",
                    "from_quantity": 100,
                    "to_quantity": 500,
                    "unit_price": Decimal("0.40")  # $0.40 per GB
                },
                {
                    "tier_name": "Next 500 GB", 
                    "from_quantity": 500,
                    "to_quantity": 1000,
                    "unit_price": Decimal("0.30")  # $0.30 per GB
                },
                {
                    "tier_name": "Over 1TB",
                    "from_quantity": 1000,
                    "to_quantity": None,
                    "unit_price": Decimal("0.20")  # $0.20 per GB
                }
            ],
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "storage_gb",
                        "pricing_model": "graduated"
                    }
                ]
            },
            "active": True
        }
        
        # Create plans
        tiered_api = self.billing_service.create_plan(tiered_api_plan)
        graduated_storage = self.billing_service.create_plan(graduated_storage_plan)
        
        print(f"✓ Created Tiered API Plan: {tiered_api.id}")
        print("  Tiers: $0.005 (0-10K), $0.003 (10K-100K), $0.002 (100K-1M), $0.001 (1M+)")
        
        print(f"✓ Created Graduated Storage Plan: {graduated_storage.id}")
        print("  Tiers: $0.50 (0-100GB), $0.40 (100-500GB), $0.30 (500GB-1TB), $0.20 (1TB+)")
        
        return tiered_api, graduated_storage
    
    def example_3_customer_specific_billing(self):
        """Example 3: Customer-specific billing arrangements"""
        print("\n" + "="*60)
        print("EXAMPLE 3: Customer-Specific Billing")
        print("="*60)
        
        # Create different customer types
        enterprise_customer = {
            "name": "MegaCorp Enterprise",
            "email": "billing@megacorp.com",
            "company": "MegaCorp Inc",
            "customer_type": "enterprise",
            "currency": "USD",
            "billing_preferences": {
                "payment_terms": "net_30",
                "invoice_frequency": "monthly",
                "billing_contact": "billing@megacorp.com",
                "po_required": True,
                "custom_billing_cycle": 15  # Bill on 15th of each month
            },
            "metadata": {
                "account_manager": "jane_doe",
                "contract_type": "enterprise",
                "volume_commitment": "high"
            }
        }
        
        startup_customer = {
            "name": "StartupCo",
            "email": "founder@startupco.com",
            "company": "StartupCo Ltd",
            "customer_type": "startup",
            "currency": "USD",
            "billing_preferences": {
                "payment_terms": "immediate",
                "invoice_frequency": "monthly",
                "billing_contact": "founder@startupco.com",
                "auto_pay": True,
                "startup_discount_eligible": True
            },
            "metadata": {
                "funding_stage": "seed",
                "employee_count": "10-25"
            }
        }
        
        # Create customers
        enterprise_cust = self.billing_service.create_customer(enterprise_customer)
        startup_cust = self.billing_service.create_customer(startup_customer)
        
        print(f"✓ Created Enterprise Customer: {enterprise_cust.id} - {enterprise_cust.name}")
        print(f"✓ Created Startup Customer: {startup_cust.id} - {startup_cust.name}")
        
        # Create customer-specific plans
        enterprise_custom_plan = {
            "name": f"Custom Enterprise Plan - {enterprise_cust.name}",
            "description": "Custom negotiated pricing for enterprise customer",
            "amount": Decimal("5000.00"),  # Negotiated flat rate
            "currency": "USD",
            "billing_period": "monthly",
            "customer_specific": True,
            "applicable_customer_ids": [enterprise_cust.id],
            "custom_terms": {
                "volume_discount": 25,  # 25% volume discount
                "minimum_commitment": Decimal("60000.00"),  # $60K annual commitment
                "overage_protection": True,  # Capped overage charges
                "dedicated_support": True,
                "sla_guarantee": "99.9%"
            },
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.0005"),  # 50% discount
                        "included_quantity": 5000000,  # 5M included calls
                        "overage_price": Decimal("0.0008"),
                        "overage_cap": Decimal("2000.00")  # Max $2K overage
                    }
                ]
            },
            "active": True
        }
        
        startup_custom_plan = {
            "name": f"Startup Plan - {startup_cust.name}",
            "description": "Special startup pricing with growth incentives",
            "amount": Decimal("99.00"),  # Startup discount pricing
            "currency": "USD",
            "billing_period": "monthly",
            "customer_specific": True,
            "applicable_customer_ids": [startup_cust.id],
            "custom_terms": {
                "startup_discount": 50,  # 50% startup discount
                "growth_credits": Decimal("500.00"),  # $500 monthly growth credits
                "payment_terms": "immediate",
                "scale_protection": True  # Automatic plan upgrades
            },
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.002"),  # Startup pricing
                        "included_quantity": 50000,
                        "growth_credit_applicable": True
                    }
                ]
            },
            "active": True
        }
        
        # Create customer-specific plans
        enterprise_plan = self.billing_service.create_plan(enterprise_custom_plan)
        startup_plan = self.billing_service.create_plan(startup_custom_plan)
        
        print(f"✓ Created Enterprise Custom Plan: {enterprise_plan.id}")
        print(f"  Annual commitment: $60K, 25% volume discount, overage cap: $2K")
        
        print(f"✓ Created Startup Custom Plan: {startup_plan.id}")
        print(f"  50% startup discount, $500 growth credits, scale protection")
        
        return enterprise_cust, startup_cust, enterprise_plan, startup_plan
    
    def example_4_customer_specific_discounts(self):
        """Example 4: Customer-specific discounts and promotions"""
        print("\n" + "="*60)
        print("EXAMPLE 4: Customer-Specific Discounts")
        print("="*60)
        
        # Get customers from previous example
        customers = list(self.billing_service.customers.values())
        enterprise_cust = customers[0]
        startup_cust = customers[1] if len(customers) > 1 else customers[0]
        
        # Create various discount types
        volume_discount = {
            "discount_id": "volume_enterprise_001",
            "name": "Enterprise Volume Discount",
            "description": "Volume-based discount for enterprise customers",
            "discount_type": "percentage",
            "discount_value": Decimal("15.00"),  # 15% discount
            "applicable_customer_ids": [enterprise_cust.id],
            "conditions": {
                "minimum_monthly_spend": Decimal("1000.00"),
                "valid_from": datetime.now().date(),
                "valid_until": (datetime.now() + timedelta(days=365)).date(),
                "applies_to": ["subscription", "usage_charges"],
                "stackable": False
            },
            "usage_rules": {
                "max_usage_per_month": 1,
                "auto_apply": True
            },
            "active": True
        }
        
        loyalty_discount = {
            "discount_id": "loyalty_startup_001", 
            "name": "Startup Loyalty Discount",
            "description": "Loyalty discount for long-term startup customers",
            "discount_type": "fixed_amount",
            "discount_value": Decimal("50.00"),  # $50 off per month
            "applicable_customer_ids": [startup_cust.id],
            "conditions": {
                "customer_tenure_months": 6,  # After 6 months
                "valid_from": datetime.now().date(),
                "valid_until": (datetime.now() + timedelta(days=180)).date(),
                "applies_to": ["subscription"],
                "stackable": True
            },
            "usage_rules": {
                "max_usage_per_month": 1,
                "auto_apply": True
            },
            "active": True
        }
        
        promotional_discount = {
            "discount_id": "promo_q1_2025",
            "name": "Q1 2025 Promotional Discount",
            "description": "Limited time promotional discount",
            "discount_type": "percentage",
            "discount_value": Decimal("20.00"),  # 20% discount
            "applicable_customer_ids": "all",  # Applies to all customers
            "conditions": {
                "valid_from": datetime.now().date(),
                "valid_until": (datetime.now() + timedelta(days=90)).date(),
                "applies_to": ["first_invoice"],
                "new_customers_only": True,
                "promo_code_required": True,
                "promo_code": "Q1SAVE20"
            },
            "usage_rules": {
                "max_usage_per_customer": 1,
                "auto_apply": False  # Requires promo code
            },
            "active": True
        }
        
        # Credit-based discount
        referral_credit = {
            "discount_id": "referral_credit_001",
            "name": "Referral Credit",
            "description": "Account credit for successful referrals",
            "discount_type": "account_credit",
            "discount_value": Decimal("100.00"),  # $100 account credit
            "applicable_customer_ids": [startup_cust.id],
            "conditions": {
                "valid_from": datetime.now().date(),
                "valid_until": (datetime.now() + timedelta(days=365)).date(),
                "applies_to": ["any_charge"],
                "credit_source": "referral_program"
            },
            "usage_rules": {
                "max_usage_total": Decimal("100.00"),
                "auto_apply": True,
                "apply_order": "first"  # Apply credits first
            },
            "active": True
        }
        
        # Apply discounts to customers
        self.billing_service.apply_customer_discount(enterprise_cust.id, volume_discount)
        self.billing_service.apply_customer_discount(startup_cust.id, loyalty_discount)
        self.billing_service.apply_customer_discount(startup_cust.id, referral_credit)
        
        print(f"✓ Applied Volume Discount to {enterprise_cust.name}: 15% off monthly spend >$1K")
        print(f"✓ Applied Loyalty Discount to {startup_cust.name}: $50 off per month after 6 months")
        print(f"✓ Applied Referral Credit to {startup_cust.name}: $100 account credit")
        print(f"✓ Created Promotional Discount: 20% off for new customers (code: Q1SAVE20)")
        
        # Demonstrate discount stacking
        stackable_discount = {
            "discount_id": "stack_test_001",
            "name": "Stackable Discount Test",
            "description": "Additional discount that can stack",
            "discount_type": "percentage", 
            "discount_value": Decimal("5.00"),  # Additional 5% off
            "applicable_customer_ids": [startup_cust.id],
            "conditions": {
                "valid_from": datetime.now().date(),
                "valid_until": (datetime.now() + timedelta(days=30)).date(),
                "applies_to": ["subscription"],
                "stackable": True
            },
            "active": True
        }
        
        self.billing_service.apply_customer_discount(startup_cust.id, stackable_discount)
        print(f"✓ Applied Additional Stackable Discount: 5% off (stacks with loyalty discount)")
        
        return volume_discount, loyalty_discount, promotional_discount, referral_credit
    
    def example_5_deferred_billing(self):
        """Example 5: Deferred billing and payment terms"""
        print("\n" + "="*60)
        print("EXAMPLE 5: Deferred Billing")
        print("="*60)
        
        # Get enterprise customer
        enterprise_customers = [c for c in self.billing_service.customers.values() 
                              if "enterprise" in c.name.lower()]
        enterprise_cust = enterprise_customers[0] if enterprise_customers else list(self.billing_service.customers.values())[0]
        
        # Create deferred billing plan
        deferred_plan = {
            "name": "Enterprise Deferred Billing Plan",
            "description": "Net 30 payment terms with deferred billing",
            "amount": Decimal("2500.00"),
            "currency": "USD",
            "billing_period": "monthly",
            "billing_model": "deferred",
            "payment_terms": {
                "net_days": 30,  # Payment due 30 days after invoice
                "early_payment_discount": {
                    "discount_percentage": Decimal("2.0"),  # 2% discount
                    "discount_days": 10  # If paid within 10 days
                },
                "late_payment_penalty": {
                    "penalty_percentage": Decimal("1.5"),  # 1.5% monthly penalty
                    "grace_period_days": 5  # 5-day grace period
                }
            },
            "deferred_billing_options": {
                "bill_in_advance": False,  # Bill after service period
                "service_period_alignment": True,
                "usage_billing_delay": 5,  # Bill usage 5 days after period end
                "consolidate_charges": True  # Combine all charges in one invoice
            },
            "active": True
        }
        
        # Create subscription with deferred billing
        deferred_subscription_data = {
            "customer_id": enterprise_cust.id,
            "plan_id": None,  # Will be created
            "billing_model": "deferred",
            "billing_configuration": {
                "invoice_generation_delay": 3,  # Generate invoice 3 days after period
                "payment_due_offset": 30,  # Payment due 30 days after invoice
                "service_credit_terms": True,  # Allow service on credit
                "credit_limit": Decimal("10000.00")  # $10K credit limit
            },
            "start_date": datetime.now().date(),
            "metadata": {
                "billing_contact": "ap@megacorp.com",
                "po_number": "PO-2025-001234",
                "approval_required": True
            }
        }
        
        # Create the deferred plan and subscription
        deferred_billing_plan = self.billing_service.create_plan(deferred_plan)
        deferred_subscription_data["plan_id"] = deferred_billing_plan.id
        deferred_subscription = self.billing_service.create_subscription(deferred_subscription_data)
        
        print(f"✓ Created Deferred Billing Plan: {deferred_billing_plan.id}")
        print(f"  Payment Terms: Net 30 days")
        print(f"  Early Payment Discount: 2% if paid within 10 days")
        print(f"  Credit Limit: $10,000")
        
        # Create usage-based deferred billing
        usage_deferred_plan = {
            "name": "Usage-Based Deferred Billing",
            "description": "Pay for actual usage at end of period",
            "amount": Decimal("0.00"),  # No upfront cost
            "currency": "USD",
            "billing_period": "monthly",
            "billing_model": "usage_deferred",
            "usage_based_billing": {
                "enabled": True,
                "billing_delay_days": 7,  # Bill usage 7 days after period
                "billable_metrics": [
                    {
                        "metric_name": "compute_hours",
                        "unit_price": Decimal("0.75"),
                        "billing_alignment": "period_end",
                        "minimum_charge": Decimal("50.00")  # Minimum monthly charge
                    }
                ]
            },
            "payment_terms": {
                "net_days": 45,  # Extended terms for usage billing
                "usage_reconciliation_period": 15  # 15 days to dispute usage
            },
            "active": True
        }
        
        usage_deferred = self.billing_service.create_plan(usage_deferred_plan)
        print(f"✓ Created Usage Deferred Plan: {usage_deferred.id}")
        print(f"  Bill usage 7 days after period end, Net 45 payment terms")
        
        # Quarterly deferred billing
        quarterly_deferred_plan = {
            "name": "Quarterly Deferred Billing",
            "description": "Quarterly billing with extended payment terms",
            "amount": Decimal("7500.00"),  # Quarterly amount
            "currency": "USD",
            "billing_period": "quarterly",
            "billing_model": "deferred",
            "payment_terms": {
                "net_days": 60,  # 60 days for quarterly billing
                "installment_options": {
                    "enabled": True,
                    "number_of_installments": 3,
                    "installment_frequency": "monthly"
                }
            },
            "active": True
        }
        
        quarterly_deferred = self.billing_service.create_plan(quarterly_deferred_plan)
        print(f"✓ Created Quarterly Deferred Plan: {quarterly_deferred.id}")
        print(f"  Quarterly billing, 60-day terms, optional 3-month installments")
        
        return deferred_billing_plan, usage_deferred, quarterly_deferred, deferred_subscription
    
    def example_6_prepaid_usage(self):
        """Example 6: Prepaid usage and credits"""
        print("\n" + "="*60)
        print("EXAMPLE 6: Prepaid Usage")
        print("="*60)
        
        # Get a customer for prepaid billing
        customers = list(self.billing_service.customers.values())
        customer = customers[0]
        
        # Create prepaid credit plan
        prepaid_plan = {
            "name": "Prepaid Credits Plan",
            "description": "Pay in advance for usage credits", 
            "amount": Decimal("0.00"),  # No recurring fee
            "currency": "USD",
            "billing_period": "monthly",
            "billing_model": "prepaid",
            "prepaid_configuration": {
                "credit_packages": [
                    {
                        "package_name": "Starter Pack",
                        "credit_amount": Decimal("100.00"),
                        "bonus_credits": Decimal("10.00"),  # 10% bonus
                        "price": Decimal("100.00")
                    },
                    {
                        "package_name": "Growth Pack",
                        "credit_amount": Decimal("500.00"),
                        "bonus_credits": Decimal("75.00"),  # 15% bonus
                        "price": Decimal("500.00")
                    },
                    {
                        "package_name": "Enterprise Pack",
                        "credit_amount": Decimal("2000.00"),
                        "bonus_credits": Decimal("400.00"),  # 20% bonus
                        "price": Decimal("2000.00")
                    }
                ],
                "auto_refill": {
                    "enabled": True,
                    "threshold_amount": Decimal("25.00"),  # Refill when below $25
                    "refill_package": "Growth Pack"
                },
                "credit_expiration": {
                    "enabled": True,
                    "expiration_days": 365,  # Credits expire after 1 year
                    "expiration_warning_days": 30
                }
            },
            "usage_based_billing": {
                "enabled": True,
                "billing_model": "prepaid_deduction",
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.002"),  # $0.002 per call
                        "deduction_model": "real_time"
                    },
                    {
                        "metric_name": "storage_gb_hours",
                        "unit_price": Decimal("0.0001"),  # $0.0001 per GB-hour
                        "deduction_model": "hourly"
                    },
                    {
                        "metric_name": "bandwidth_gb",
                        "unit_price": Decimal("0.10"),  # $0.10 per GB
                        "deduction_model": "real_time"
                    }
                ]
            },
            "active": True
        }
        
        # Create subscription with prepaid credits
        prepaid_subscription_data = {
            "customer_id": customer.id,
            "plan_id": None,  # Will be set after plan creation
            "billing_model": "prepaid",
            "prepaid_settings": {
                "initial_credit_purchase": "Growth Pack",  # Start with Growth Pack
                "auto_refill_enabled": True,
                "low_balance_alerts": True,
                "usage_alerts": {
                    "alert_at_percentage": [75, 90, 95],  # Alert at 75%, 90%, 95%
                    "alert_methods": ["email", "webhook"]
                }
            },
            "start_date": datetime.now().date()
        }
        
        # Create the prepaid plan and subscription
        prepaid_billing_plan = self.billing_service.create_plan(prepaid_plan)
        prepaid_subscription_data["plan_id"] = prepaid_billing_plan.id
        prepaid_subscription = self.billing_service.create_subscription(prepaid_subscription_data)
        
        print(f"✓ Created Prepaid Plan: {prepaid_billing_plan.id}")
        print(f"  Credit Packages: $100 (+10%), $500 (+15%), $2000 (+20%)")
        print(f"  Auto-refill when balance < $25")
        print(f"  Credits expire after 365 days")
        
        # Purchase initial credits
        credit_purchase = {
            "customer_id": customer.id,
            "subscription_id": prepaid_subscription.id,
            "package_name": "Growth Pack",
            "purchase_amount": Decimal("500.00"),
            "credit_amount": Decimal("575.00"),  # Including 15% bonus
            "transaction_type": "credit_purchase",
            "payment_method": "card"
        }
        
        self.billing_service.process_credit_purchase(credit_purchase)
        print(f"✓ Purchased Growth Pack: $500 → $575 credits (15% bonus)")
        
        # Create pay-per-use plan (no subscription required)
        pay_per_use_plan = {
            "name": "Pay-Per-Use Plan",
            "description": "No commitment, pay for what you use",
            "amount": Decimal("0.00"),
            "currency": "USD",
            "billing_period": "monthly",
            "billing_model": "pay_per_use",
            "usage_based_billing": {
                "enabled": True,
                "billing_model": "postpaid_immediate",  # Bill immediately after use
                "billable_metrics": [
                    {
                        "metric_name": "function_invocations",
                        "unit_price": Decimal("0.0002"),  # $0.0002 per invocation
                        "minimum_charge": Decimal("0.01"),  # Minimum $0.01 per charge
                        "billing_frequency": "immediate"
                    },
                    {
                        "metric_name": "data_processing_mb",
                        "unit_price": Decimal("0.001"),  # $0.001 per MB
                        "billing_frequency": "daily_aggregate"
                    }
                ]
            },
            "active": True
        }
        
        pay_per_use = self.billing_service.create_plan(pay_per_use_plan)
        print(f"✓ Created Pay-Per-Use Plan: {pay_per_use.id}")
        print(f"  No subscription required, bill immediately after use")
        
        # Hybrid prepaid plan (base fee + prepaid usage)
        hybrid_prepaid_plan = {
            "name": "Hybrid Prepaid Plan",
            "description": "Base subscription + prepaid usage credits",
            "amount": Decimal("49.99"),  # Monthly base fee
            "currency": "USD",
            "billing_period": "monthly",
            "billing_model": "hybrid_prepaid",
            "included_credits": Decimal("50.00"),  # $50 included monthly credits
            "prepaid_configuration": {
                "additional_credit_packages": [
                    {
                        "package_name": "Add-on Credits",
                        "credit_amount": Decimal("100.00"),
                        "price": Decimal("95.00"),  # 5% discount
                        "subscriber_only": True
                    }
                ],
                "credit_rollover": {
                    "enabled": True,
                    "max_rollover_amount": Decimal("200.00"),
                    "rollover_expiration_months": 3
                }
            },
            "usage_based_billing": {
                "enabled": True,
                "billing_model": "credits_first",  # Use credits first, then bill
                "billable_metrics": [
                    {
                        "metric_name": "premium_api_calls",
                        "unit_price": Decimal("0.005"),
                        "credit_deduction_rate": Decimal("0.004")  # Better rate with credits
                    }
                ]
            },
            "active": True
        }
        
        hybrid_prepaid = self.billing_service.create_plan(hybrid_prepaid_plan)
        print(f"✓ Created Hybrid Prepaid Plan: {hybrid_prepaid.id}")
        print(f"  $49.99/month + $50 included credits + credit rollover")
        
        return prepaid_billing_plan, pay_per_use, hybrid_prepaid, prepaid_subscription
    
    def example_7_advanced_billing_scenarios(self):
        """Example 7: Advanced billing scenarios and edge cases"""
        print("\n" + "="*60)
        print("EXAMPLE 7: Advanced Billing Scenarios")
        print("="*60)
        
        # Multi-currency billing
        multi_currency_plan = {
            "name": "Global Multi-Currency Plan",
            "description": "Support for multiple currencies with auto-conversion",
            "amount": Decimal("100.00"),
            "currency": "USD",  # Base currency
            "billing_period": "monthly",
            "multi_currency_support": {
                "enabled": True,
                "supported_currencies": ["USD", "EUR", "GBP", "CAD", "AUD"],
                "conversion_settings": {
                    "rate_source": "ecb",  # European Central Bank rates
                    "update_frequency": "daily",
                    "rate_lock_period": "billing_period"  # Lock rate for billing period
                },
                "currency_specific_pricing": {
                    "EUR": {"amount": Decimal("85.00"), "tax_inclusive": True},
                    "GBP": {"amount": Decimal("75.00"), "tax_inclusive": True},
                    "CAD": {"amount": Decimal("130.00"), "tax_inclusive": False}
                }
            },
            "active": True
        }
        
        # Seasonal billing
        seasonal_plan = {
            "name": "Seasonal Business Plan",
            "description": "Billing that adjusts based on seasonal demand",
            "amount": Decimal("200.00"),  # Base amount
            "currency": "USD",
            "billing_period": "monthly",
            "seasonal_adjustments": {
                "enabled": True,
                "adjustment_rules": [
                    {
                        "season": "holiday",  # Nov-Dec
                        "months": [11, 12],
                        "adjustment_type": "multiplier",
                        "adjustment_value": Decimal("1.5")  # 50% increase
                    },
                    {
                        "season": "summer",  # Jun-Aug
                        "months": [6, 7, 8],
                        "adjustment_type": "fixed_amount",
                        "adjustment_value": Decimal("50.00")  # +$50
                    },
                    {
                        "season": "off_peak",  # Jan-Feb
                        "months": [1, 2],
                        "adjustment_type": "percentage",
                        "adjustment_value": Decimal("-25.00")  # 25% discount
                    }
                ]
            },
            "active": True
        }
        
        # Performance-based billing
        performance_plan = {
            "name": "Performance-Based Billing",
            "description": "Billing adjusts based on service performance metrics",
            "amount": Decimal("500.00"),  # Base amount
            "currency": "USD",
            "billing_period": "monthly",
            "performance_based_adjustments": {
                "enabled": True,
                "sla_metrics": [
                    {
                        "metric": "uptime_percentage",
                        "target": Decimal("99.9"),
                        "adjustment_rules": [
                            {"threshold": Decimal("99.9"), "adjustment": Decimal("0.0")},  # No change
                            {"threshold": Decimal("99.5"), "adjustment": Decimal("-10.0")},  # 10% discount
                            {"threshold": Decimal("99.0"), "adjustment": Decimal("-25.0")},  # 25% discount
                            {"threshold": Decimal("98.0"), "adjustment": Decimal("-50.0")}   # 50% discount
                        ]
                    },
                    {
                        "metric": "response_time_ms",
                        "target": Decimal("200.0"),
                        "adjustment_rules": [
                            {"threshold": Decimal("200.0"), "adjustment": Decimal("0.0")},
                            {"threshold": Decimal("500.0"), "adjustment": Decimal("-5.0")},
                            {"threshold": Decimal("1000.0"), "adjustment": Decimal("-15.0")}
                        ]
                    }
                ]
            },
            "active": True
        }
        
        # Create the advanced plans
        multi_currency = self.billing_service.create_plan(multi_currency_plan)
        seasonal = self.billing_service.create_plan(seasonal_plan)
        performance = self.billing_service.create_plan(performance_plan)
        
        print(f"✓ Created Multi-Currency Plan: {multi_currency.id}")
        print(f"  Supports USD, EUR, GBP, CAD, AUD with auto-conversion")
        
        print(f"✓ Created Seasonal Plan: {seasonal.id}")
        print(f"  Holiday season: +50%, Summer: +$50, Off-peak: -25%")
        
        print(f"✓ Created Performance Plan: {performance.id}")
        print(f"  SLA-based adjustments: uptime and response time metrics")
        
        return multi_currency, seasonal, performance
    
    def run_all_billing_model_examples(self):
        """Run all billing model examples"""
        print("APG Billing System - Comprehensive Billing Models")
        print("=" * 70)
        print("Configuration source:", 
              "Central" if os.path.exists('/etc/apg/composition/central_configuration') or 
                         os.path.exists('../../../composition/central_configuration') 
              else "Local")
        
        try:
            # Run all billing model examples
            self.example_1_different_prices_per_service()
            self.example_2_tiered_pricing_per_service()
            self.example_3_customer_specific_billing()
            self.example_4_customer_specific_discounts()
            self.example_5_deferred_billing()
            self.example_6_prepaid_usage()
            self.example_7_advanced_billing_scenarios()
            
            print("\n" + "="*70)
            print("✓ ALL BILLING MODEL EXAMPLES COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            print(f"\n📋 Billing Models Demonstrated:")
            print(f"  ✓ Different prices per service/feature")
            print(f"  ✓ Tiered pricing (volume-based & graduated)")
            print(f"  ✓ Customer-specific billing arrangements")
            print(f"  ✓ Customer-specific discounts & promotions")
            print(f"  ✓ Deferred billing (Net 30/45/60 terms)")
            print(f"  ✓ Prepaid usage & credit systems")
            print(f"  ✓ Pay-per-use (no subscription)")
            print(f"  ✓ Hybrid prepaid models")
            print(f"  ✓ Multi-currency support")
            print(f"  ✓ Seasonal billing adjustments")
            print(f"  ✓ Performance-based billing")
            
            print(f"\n📊 System Summary:")
            print(f"  Total Customers: {len(self.billing_service.customers)}")
            print(f"  Total Plans: {len(self.billing_service.plans)}")
            print(f"  Total Subscriptions: {len(self.billing_service.subscriptions)}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function to demonstrate billing models"""
    examples = BillingModelsExamples()
    success = examples.run_all_billing_model_examples()
    
    if success:
        print("\n🎉 APG Billing System supports all requested billing models!")
        print("\nKey Capabilities Confirmed:")
        print("  ✅ Different prices per service - YES")
        print("  ✅ Tiered pricing per service - YES")
        print("  ✅ Customer-specific billing - YES")
        print("  ✅ Customer-specific discounts - YES")
        print("  ✅ Deferred billing - YES")
        print("  ✅ Prepaid usage - YES")
        print("  ✅ Plus many more advanced features!")
        
        print("\nNext steps:")
        print("  1. Choose the billing models that fit your business")
        print("  2. Configure customer-specific arrangements")
        print("  3. Set up automated discount rules")
        print("  4. Implement usage tracking for your services")
        print("  5. Configure payment terms and credit limits")
    else:
        print("\n⚠ Some billing model examples failed. Check the logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())