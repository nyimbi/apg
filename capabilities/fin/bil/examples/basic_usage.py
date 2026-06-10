#!/usr/bin/env python3
"""
APG Billing System - Basic Usage Examples

This file demonstrates basic usage patterns for the APG Billing System,
including customer management, subscription creation, and payment processing.

Supports configuration by composition/central_configuration.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

# Add the billing module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load configuration from composition layer if available
try:
    # Try to load from central configuration
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
    BLCustomer, BLPlan, BLSubscription, BLInvoice, 
    BLPayment, BLUsage, SubscriptionStatus, PaymentStatus
)

class BillingExamples:
    def __init__(self):
        """Initialize the billing service"""
        self.billing_service = get_billing_service()
        print("✓ APG Billing Service initialized")
    
    def example_1_customer_management(self):
        """Example 1: Basic customer management"""
        print("\n" + "="*50)
        print("EXAMPLE 1: Customer Management")
        print("="*50)
        
        # Create a customer
        customer_data = {
            "name": "Acme Corporation",
            "email": "billing@acme-corp.com",
            "phone": "+1-555-0123",
            "company": "Acme Corp",
            "tax_id": "12-3456789",
            "currency": "USD",
            "language": "en",
            "timezone": "America/New_York",
            "billing_address": {
                "street": "123 Business Ave",
                "city": "New York",
                "state": "NY",
                "postal_code": "10001",
                "country": "US"
            },
            "shipping_address": {
                "street": "456 Delivery St",
                "city": "New York", 
                "state": "NY",
                "postal_code": "10002",
                "country": "US"
            },
            "metadata": {
                "acquisition_channel": "website",
                "sales_rep": "john_doe",
                "company_size": "50-100"
            }
        }
        
        customer = self.billing_service.create_customer(customer_data)
        print(f"✓ Created customer: {customer.id} - {customer.name}")
        
        # Update customer information
        update_data = {
            "phone": "+1-555-0124",
            "billing_address": {
                "street": "789 Updated Ave",
                "city": "New York",
                "state": "NY", 
                "postal_code": "10003",
                "country": "US"
            }
        }
        
        updated_customer = self.billing_service.update_customer(customer.id, update_data)
        print(f"✓ Updated customer phone: {updated_customer.phone}")
        
        # Retrieve customer
        retrieved_customer = self.billing_service.get_customer(customer.id)
        print(f"✓ Retrieved customer: {retrieved_customer.name}")
        
        return customer
    
    def example_2_plan_creation(self):
        """Example 2: Creating billing plans"""
        print("\n" + "="*50)
        print("EXAMPLE 2: Plan Creation")
        print("="*50)
        
        # Create a simple subscription plan
        simple_plan_data = {
            "name": "Professional Plan",
            "description": "Full-featured plan for growing businesses",
            "amount": Decimal("99.99"),
            "currency": "USD",
            "billing_period": "monthly",
            "trial_period_days": 14,
            "setup_fee": Decimal("50.00"),
            "features": ["api_access", "analytics", "priority_support"],
            "tax_behavior": "exclusive",
            "active": True
        }
        
        simple_plan = self.billing_service.create_plan(simple_plan_data)
        print(f"✓ Created simple plan: {simple_plan.id} - {simple_plan.name}")
        
        # Create a usage-based plan
        usage_plan_data = {
            "name": "Pay-as-you-go Plan",
            "description": "Perfect for variable usage patterns", 
            "amount": Decimal("29.99"),  # Base amount
            "currency": "USD",
            "billing_period": "monthly",
            "trial_period_days": 7,
            "usage_based_billing": {
                "enabled": True,
                "billable_metrics": [
                    {
                        "metric_name": "api_calls",
                        "unit_price": Decimal("0.01"),
                        "included_quantity": 10000,
                        "overage_price": Decimal("0.015")
                    },
                    {
                        "metric_name": "storage_gb",
                        "unit_price": Decimal("0.50"),
                        "included_quantity": 100
                    }
                ]
            },
            "active": True
        }
        
        usage_plan = self.billing_service.create_plan(usage_plan_data)
        print(f"✓ Created usage-based plan: {usage_plan.id} - {usage_plan.name}")
        
        # Create a tiered pricing plan
        tiered_plan_data = {
            "name": "Enterprise Plan",
            "description": "Scalable pricing for enterprise customers",
            "amount": Decimal("0.00"),  # Base amount
            "currency": "USD", 
            "billing_period": "monthly",
            "pricing_tiers": [
                {
                    "up_to": 1000,
                    "unit_price": Decimal("0.10")
                },
                {
                    "up_to": 10000,
                    "unit_price": Decimal("0.08")
                },
                {
                    "up_to": None,  # Unlimited
                    "unit_price": Decimal("0.05")
                }
            ],
            "active": True
        }
        
        tiered_plan = self.billing_service.create_plan(tiered_plan_data)
        print(f"✓ Created tiered plan: {tiered_plan.id} - {tiered_plan.name}")
        
        return simple_plan, usage_plan, tiered_plan
    
    def example_3_subscription_lifecycle(self):
        """Example 3: Complete subscription lifecycle"""
        print("\n" + "="*50)
        print("EXAMPLE 3: Subscription Lifecycle")
        print("="*50)
        
        # Get customer and plan from previous examples
        customer = self.example_1_customer_management()
        simple_plan, usage_plan, _ = self.example_2_plan_creation()
        
        # Create subscription
        subscription_data = {
            "customer_id": customer.id,
            "plan_id": simple_plan.id,
            "start_date": datetime.now().date(),
            "trial_end_date": (datetime.now() + timedelta(days=14)).date(),
            "billing_cycle_anchor": 1,  # Bill on the 1st of each month
            "proration_behavior": "create_prorations",
            "collection_method": "charge_automatically",
            "metadata": {
                "source": "api_example",
                "campaign": "q1_2025"
            }
        }
        
        subscription = self.billing_service.create_subscription(subscription_data)
        print(f"✓ Created subscription: {subscription.id}")
        print(f"  Status: {subscription.status}")
        print(f"  Trial ends: {subscription.trial_end_date}")
        
        # Change subscription plan
        change_data = {
            "new_plan_id": usage_plan.id,
            "proration_behavior": "create_prorations",
            "effective_date": datetime.now().date()
        }
        
        changed_subscription = self.billing_service.change_subscription_plan(
            subscription.id, change_data
        )
        print(f"✓ Changed subscription plan to: {usage_plan.name}")
        
        # Pause subscription
        pause_data = {
            "pause_behavior": "keep_as_draft",
            "resume_at": (datetime.now() + timedelta(days=30)).date()
        }
        
        self.billing_service.pause_subscription(subscription.id, pause_data)
        print(f"✓ Paused subscription until {pause_data['resume_at']}")
        
        # Resume subscription
        self.billing_service.resume_subscription(subscription.id)
        print("✓ Resumed subscription")
        
        # Cancel subscription
        cancel_data = {
            "cancellation_reason": "customer_request",
            "cancel_at_period_end": True,
            "prorate": False
        }
        
        cancelled_subscription = self.billing_service.cancel_subscription(
            subscription.id, cancel_data
        )
        print(f"✓ Cancelled subscription (effective at period end)")
        
        return subscription
    
    def example_4_invoice_and_payment_processing(self):
        """Example 4: Invoice generation and payment processing"""
        print("\n" + "="*50)
        print("EXAMPLE 4: Invoice and Payment Processing")
        print("="*50)
        
        # Get customer from previous example
        customer = list(self.billing_service.customers.values())[0]
        
        # Create a manual invoice
        invoice_data = {
            "customer_id": customer.id,
            "description": "Professional services consultation",
            "due_date": (datetime.now() + timedelta(days=30)).date(),
            "currency": "USD",
            "items": [
                {
                    "description": "Consulting hours (10 hours)",
                    "amount": Decimal("150.00"),
                    "quantity": 10,
                    "unit_price": Decimal("15.00")
                },
                {
                    "description": "Setup fee",
                    "amount": Decimal("100.00"),
                    "quantity": 1
                }
            ],
            "tax_behavior": "exclusive",
            "auto_advance": True,
            "collection_method": "send_invoice",
            "metadata": {
                "project": "website_integration",
                "invoice_type": "consulting"
            }
        }
        
        invoice = self.billing_service.create_invoice(invoice_data)
        print(f"✓ Created invoice: {invoice.id}")
        print(f"  Total amount: ${invoice.total}")
        print(f"  Due date: {invoice.due_date}")
        
        # Send invoice via email
        send_data = {
            "email_template": "standard_invoice",
            "custom_message": "Thank you for choosing our services!",
            "send_copy_to": ["accounting@acme-corp.com"]
        }
        
        self.billing_service.send_invoice(invoice.id, send_data)
        print("✓ Invoice sent via email")
        
        # Process payment for the invoice
        payment_data = {
            "customer_id": customer.id,
            "invoice_id": invoice.id,
            "amount": invoice.total,
            "currency": "USD",
            "payment_method": {
                "type": "card",
                "card": {
                    "number": "4242424242424242",  # Test card
                    "exp_month": 12,
                    "exp_year": 2025,
                    "cvc": "123"
                }
            },
            "capture": True,
            "description": "Payment for consulting services",
            "metadata": {
                "payment_source": "api_example"
            }
        }
        
        payment = self.billing_service.process_payment(payment_data)
        print(f"✓ Processed payment: {payment.id}")
        print(f"  Amount: ${payment.amount}")
        print(f"  Status: {payment.status}")
        
        return invoice, payment
    
    def example_5_usage_tracking(self):
        """Example 5: Usage tracking and billing"""
        print("\n" + "="*50)
        print("EXAMPLE 5: Usage Tracking")
        print("="*50)
        
        # Get customer and usage-based plan
        customer = list(self.billing_service.customers.values())[0]
        usage_plan = [p for p in self.billing_service.plans.values() 
                     if "pay-as-you-go" in p.name.lower()][0]
        
        # Create subscription for usage tracking
        subscription_data = {
            "customer_id": customer.id,
            "plan_id": usage_plan.id,
            "start_date": datetime.now().date()
        }
        
        subscription = self.billing_service.create_subscription(subscription_data)
        print(f"✓ Created usage-based subscription: {subscription.id}")
        
        # Track API usage
        api_usage_data = {
            "customer_id": customer.id,
            "subscription_id": subscription.id,
            "metric_name": "api_calls",
            "quantity": 1500,  # Exceeds included 1000
            "timestamp": datetime.now(),
            "properties": {
                "endpoint": "/api/v1/users",
                "method": "GET",
                "response_time": 245,
                "status_code": 200
            }
        }
        
        api_usage = self.billing_service.track_usage(api_usage_data)
        print(f"✓ Tracked API usage: {api_usage.quantity} calls")
        
        # Track storage usage
        storage_usage_data = {
            "customer_id": customer.id,
            "subscription_id": subscription.id,
            "metric_name": "storage_gb", 
            "quantity": Decimal("75.5"),
            "timestamp": datetime.now(),
            "properties": {
                "region": "us-west-2",
                "storage_type": "standard"
            }
        }
        
        storage_usage = self.billing_service.track_usage(storage_usage_data)
        print(f"✓ Tracked storage usage: {storage_usage.quantity} GB")
        
        # Get usage summary
        usage_summary = self.billing_service.get_usage_summary(
            subscription_id=subscription.id,
            period_start=datetime.now().replace(day=1).date(),
            period_end=datetime.now().date()
        )
        
        print(f"✓ Usage summary retrieved:")
        for metric, data in usage_summary.items():
            print(f"  {metric}: {data['total_usage']} (overage: {data.get('overage', 0)})")
        
        return subscription
    
    def example_6_bulk_operations(self):
        """Example 6: Bulk operations for efficiency"""
        print("\n" + "="*50)
        print("EXAMPLE 6: Bulk Operations")
        print("="*50)
        
        # Bulk customer creation
        customers_data = [
            {
                "name": f"Customer {i}",
                "email": f"customer{i}@example.com",
                "currency": "USD"
            }
            for i in range(1, 6)
        ]
        
        print("✓ Creating 5 customers in bulk...")
        customers = []
        for customer_data in customers_data:
            customer = self.billing_service.create_customer(customer_data)
            customers.append(customer)
        
        print(f"✓ Created {len(customers)} customers")
        
        # Bulk usage import
        subscription = list(self.billing_service.subscriptions.values())[0]
        
        bulk_usage_data = [
            {
                "customer_id": subscription.customer_id,
                "subscription_id": subscription.id,
                "metric_name": "api_calls",
                "quantity": 100 + (i * 10),
                "timestamp": datetime.now() - timedelta(hours=i)
            }
            for i in range(24)  # 24 hours of hourly data
        ]
        
        results = self.billing_service.import_usage_batch(bulk_usage_data)
        print(f"✓ Imported {len(results)} usage records")
        
        return customers
    
    async def example_7_async_operations(self):
        """Example 7: Asynchronous operations"""
        print("\n" + "="*50)
        print("EXAMPLE 7: Async Operations")
        print("="*50)
        
        # Get billing service instance
        service = self.billing_service
        
        # Simulate concurrent payment processing
        customers = list(service.customers.values())[:3]
        
        async def process_payment_async(customer):
            """Simulate async payment processing"""
            payment_data = {
                "customer_id": customer.id,
                "amount": Decimal("50.00"),
                "currency": "USD",
                "payment_method": {
                    "type": "card",
                    "card": {
                        "number": "4242424242424242",
                        "exp_month": 12,
                        "exp_year": 2025,
                        "cvc": "123"
                    }
                },
                "description": f"Async payment for {customer.name}"
            }
            
            # Simulate async processing delay
            await asyncio.sleep(0.1)
            payment = service.process_payment(payment_data)
            return payment
        
        # Process payments concurrently
        tasks = [process_payment_async(customer) for customer in customers]
        payments = await asyncio.gather(*tasks, return_exceptions=True)

        
        print(f"✓ Processed {len(payments)} payments concurrently")
        for payment in payments:
            print(f"  Payment {payment.id}: ${payment.amount}")
        
        return payments
    
    def example_8_error_handling(self):
        """Example 8: Error handling and recovery"""
        print("\n" + "="*50)
        print("EXAMPLE 8: Error Handling")
        print("="*50)
        
        try:
            # Attempt to create invalid customer
            invalid_customer_data = {
                "name": "",  # Invalid: empty name
                "email": "invalid-email",  # Invalid: bad email format
                "currency": "INVALID"  # Invalid: unsupported currency
            }
            
            customer = self.billing_service.create_customer(invalid_customer_data)
            
        except Exception as e:
            print(f"✓ Caught expected validation error: {e}")
        
        try:
            # Attempt to process payment with declined card
            payment_data = {
                "customer_id": "invalid_customer_id",
                "amount": Decimal("100.00"),
                "currency": "USD",
                "payment_method": {
                    "type": "card",
                    "card": {
                        "number": "4000000000000002",  # Declined card
                        "exp_month": 12,
                        "exp_year": 2025,
                        "cvc": "123"
                    }
                }
            }
            
            payment = self.billing_service.process_payment(payment_data)
            
        except Exception as e:
            print(f"✓ Caught expected payment error: {e}")
        
        # Demonstrate graceful error recovery
        try:
            # Get non-existent customer
            customer = self.billing_service.get_customer("non_existent_id")
        except Exception as e:
            print(f"✓ Handled missing customer gracefully: {e}")
        
        print("✓ Error handling examples completed")
    
    def run_all_examples(self):
        """Run all examples in sequence"""
        print("APG Billing System - Usage Examples")
        print("=" * 60)
        print("Configuration source:", 
              "Central" if os.path.exists('/etc/apg/composition/central_configuration') or 
                         os.path.exists('../../../composition/central_configuration') 
              else "Local")
        
        try:
            # Run synchronous examples
            self.example_1_customer_management()
            self.example_2_plan_creation()
            self.example_3_subscription_lifecycle()
            self.example_4_invoice_and_payment_processing()
            self.example_5_usage_tracking()
            self.example_6_bulk_operations()
            self.example_8_error_handling()
            
            # Run async example
            print("\nRunning async examples...")
            asyncio.run(self.example_7_async_operations())
            
            print("\n" + "="*60)
            print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Display summary
            print(f"\nSummary:")
            print(f"  Customers: {len(self.billing_service.customers)}")
            print(f"  Plans: {len(self.billing_service.plans)}")
            print(f"  Subscriptions: {len(self.billing_service.subscriptions)}")
            print(f"  Invoices: {len(self.billing_service.invoices)}")
            print(f"  Payments: {len(self.billing_service.payments)}")
            print(f"  Usage Records: {len(self.billing_service.usage_records)}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function to run examples"""
    examples = BillingExamples()
    success = examples.run_all_examples()
    
    if success:
        print("\n🎉 Ready to integrate APG Billing into your application!")
        print("\nNext steps:")
        print("  1. Review the service.py file for more advanced features")
        print("  2. Check out the API documentation in docs/api/README.md")
        print("  3. Explore the Flask blueprint in blueprint.py")
        print("  4. Set up monitoring using scripts/monitor.sh")
    else:
        print("\n⚠ Some examples failed. Check the logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())