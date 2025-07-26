"""
APG Capability Registry - Load Testing with Locust

Production-grade load testing scenarios to validate system performance
under realistic user loads and traffic patterns.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
import random
import time
from datetime import datetime
from typing import List, Dict, Any

from locust import HttpUser, task, between, events
from uuid_extensions import uuid7str

# =============================================================================
# Test Data Generation
# =============================================================================

class TestDataGenerator:
    """Generate realistic test data for load testing."""
    
    CATEGORIES = [
        "foundation_infrastructure",
        "business_operations", 
        "analytics_intelligence",
        "manufacturing_operations",
        "financial_services",
        "communication",
        "security_compliance"
    ]
    
    CAPABILITY_NAMES = [
        "User Management System",
        "Payment Gateway",
        "Email Service",
        "Analytics Engine",
        "File Storage",
        "Notification Service",
        "Authentication Provider",
        "Data Pipeline",
        "Workflow Engine",
        "Reporting Service"
    ]
    
    BUSINESS_REQUIREMENTS = [
        "user_authentication",
        "payment_processing",
        "data_analytics",
        "file_management",
        "communication",
        "workflow_automation",
        "reporting",
        "security",
        "compliance",
        "integration"
    ]
    
    @classmethod
    def generate_capability(cls, prefix: str = "LOAD_TEST") -> Dict[str, Any]:
        """Generate a realistic capability for testing."""
        name = random.choice(cls.CAPABILITY_NAMES)
        category = random.choice(cls.CATEGORIES)
        
        return {
            "capability_code": f"{prefix}_{uuid7str()[:8]}",
            "capability_name": f"{name} {random.randint(1, 1000)}",
            "description": f"Load testing {name.lower()} for performance validation",
            "long_description": f"Comprehensive {name.lower()} implementation with advanced features and enterprise-grade security.",
            "category": category,
            "subcategory": "performance_testing",
            "version": f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "target_users": random.sample(["developers", "administrators", "end_users", "analysts"], k=random.randint(1, 3)),
            "business_value": f"Provides essential {name.lower()} functionality for enterprise applications",
            "use_cases": random.sample(["automation", "integration", "analytics", "security", "compliance"], k=random.randint(2, 4)),
            "industry_focus": random.sample(["technology", "finance", "healthcare", "retail", "manufacturing"], k=random.randint(1, 3)),
            "composition_keywords": [name.lower().replace(" ", "_"), "enterprise", "scalable", "secure"],
            "provides_services": [f"{name.lower().replace(' ', '_')}_service", "health_check", "metrics"],
            "data_models": [f"{name.replace(' ', '')}", f"{name.replace(' ', '')}Config"],
            "api_endpoints": [f"/api/{name.lower().replace(' ', '-')}", f"/api/{name.lower().replace(' ', '-')}/health"],
            "multi_tenant": random.choice([True, False]),
            "audit_enabled": random.choice([True, False]),
            "security_integration": random.choice([True, False]),
            "performance_optimized": random.choice([True, False]),
            "ai_enhanced": random.choice([True, False]),
            "complexity_score": round(random.uniform(1.0, 5.0), 1),
            "metadata": {
                "load_test": True,
                "test_id": uuid7str(),
                "created_by": "load_tester"
            }
        }
    
    @classmethod
    def generate_composition(cls, capability_ids: List[str], prefix: str = "LOAD_COMP") -> Dict[str, Any]:
        """Generate a realistic composition for testing."""
        comp_types = ["enterprise_portal", "data_platform", "ecommerce_suite", "analytics_dashboard", "workflow_system"]
        
        return {
            "name": f"Load Test Composition {random.randint(1, 10000)}",
            "description": f"Performance testing composition with {len(capability_ids)} capabilities",
            "composition_type": random.choice(comp_types),
            "capability_ids": capability_ids,
            "business_requirements": random.sample(cls.BUSINESS_REQUIREMENTS, k=random.randint(2, 5)),
            "compliance_requirements": random.sample(["gdpr", "hipaa", "sox", "pci_dss"], k=random.randint(0, 2)),
            "target_users": random.sample(["developers", "administrators", "end_users", "analysts"], k=random.randint(1, 3)),
            "deployment_strategy": random.choice(["standard", "high_availability", "multi_region"]),
            "is_template": random.choice([True, False]),
            "is_public": random.choice([True, False]),
            "configuration": {
                "load_balancing": random.choice(["round_robin", "weighted", "least_connections"]),
                "scaling_policy": random.choice(["manual", "auto", "scheduled"]),
                "monitoring_enabled": True,
                "backup_enabled": random.choice([True, False])
            },
            "environment_settings": {
                "environment": "load_testing",
                "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                "resource_tier": random.choice(["basic", "standard", "premium"])
            }
        }

# =============================================================================
# Base User Classes
# =============================================================================

class BaseRegistryUser(HttpUser):
    """Base user class for registry operations."""
    
    def on_start(self):
        """Called when user starts."""
        self.capability_ids = []
        self.composition_ids = []
        self.auth_headers = self._get_auth_headers()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        # In a real scenario, you'd authenticate and get a token
        return {
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json",
            "X-Tenant-ID": f"load_test_tenant_{random.randint(1, 10)}"
        }
    
    def _handle_response(self, response, operation: str):
        """Handle response and extract data."""
        if response.status_code >= 400:
            print(f"Error in {operation}: {response.status_code} - {response.text}")
            return None
        
        try:
            return response.json()
        except json.JSONDecodeError:
            print(f"Invalid JSON response in {operation}")
            return None

# =============================================================================
# User Behavior Classes
# =============================================================================

class CapabilityManagerUser(BaseRegistryUser):
    """User that manages capabilities - creates, reads, updates."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    weight = 3  # Higher weight = more users of this type
    
    @task(10)
    def list_capabilities(self):
        """List capabilities with pagination."""
        params = {
            "page": random.randint(1, 5),
            "per_page": random.randint(10, 50)
        }
        
        with self.client.get("/api/capabilities", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list capabilities: {response.status_code}")
    
    @task(8)
    def search_capabilities(self):
        """Search capabilities by various criteria."""
        search_terms = ["user", "payment", "email", "analytics", "file", "workflow", "security"]
        categories = TestDataGenerator.CATEGORIES
        
        params = {
            "search": random.choice(search_terms),
            "category": random.choice(categories) if random.random() < 0.5 else None,
            "min_quality_score": random.uniform(0.5, 0.9) if random.random() < 0.3 else None,
            "page": 1,
            "per_page": 20
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        with self.client.get("/api/capabilities", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                data = self._handle_response(response, "search_capabilities")
                if data and "items" in data:
                    # Store some capability IDs for other operations
                    for item in data["items"][:3]:
                        if "capability_id" in item:
                            self.capability_ids.append(item["capability_id"])
            else:
                response.failure(f"Failed to search capabilities: {response.status_code}")
    
    @task(5)
    def create_capability(self):
        """Create a new capability."""
        capability_data = TestDataGenerator.generate_capability("LOCUST_CAP")
        
        with self.client.post("/api/capabilities", json=capability_data, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 201:
                response.success()
                data = self._handle_response(response, "create_capability")
                if data and "data" in data and "capability_id" in data["data"]:
                    self.capability_ids.append(data["data"]["capability_id"])
            else:
                response.failure(f"Failed to create capability: {response.status_code}")
    
    @task(3)
    def get_capability_details(self):
        """Get detailed information about a specific capability."""
        if not self.capability_ids:
            return
        
        capability_id = random.choice(self.capability_ids)
        
        with self.client.get(f"/api/capabilities/{capability_id}", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Remove invalid ID
                if capability_id in self.capability_ids:
                    self.capability_ids.remove(capability_id)
                response.success()  # 404 is expected sometimes
            else:
                response.failure(f"Failed to get capability details: {response.status_code}")
    
    @task(2)
    def update_capability(self):
        """Update an existing capability."""
        if not self.capability_ids:
            return
        
        capability_id = random.choice(self.capability_ids)
        updates = {
            "description": f"Updated description at {datetime.utcnow().isoformat()}",
            "quality_score": round(random.uniform(0.7, 1.0), 2),
            "metadata": {
                "updated_by_load_test": True,
                "update_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        with self.client.put(f"/api/capabilities/{capability_id}", json=updates, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Remove invalid ID
                if capability_id in self.capability_ids:
                    self.capability_ids.remove(capability_id)
                response.success()  # 404 is expected sometimes
            else:
                response.failure(f"Failed to update capability: {response.status_code}")

class CompositionDesignerUser(BaseRegistryUser):
    """User that works with compositions - creates, validates, manages."""
    
    wait_time = between(2, 5)  # Longer wait time for complex operations
    weight = 2
    
    @task(8)
    def list_compositions(self):
        """List existing compositions."""
        params = {
            "page": random.randint(1, 3),
            "per_page": random.randint(10, 25)
        }
        
        with self.client.get("/api/compositions", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list compositions: {response.status_code}")
    
    @task(6)
    def validate_composition(self):
        """Validate a composition of capabilities."""
        if len(self.capability_ids) < 2:
            # Get some capability IDs first
            self._populate_capability_ids()
        
        if len(self.capability_ids) >= 2:
            # Select 2-5 capabilities for validation
            selected_caps = random.sample(self.capability_ids, min(random.randint(2, 5), len(self.capability_ids)))
            
            with self.client.post("/api/compositions/validate", json=selected_caps, headers=self.auth_headers, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Failed to validate composition: {response.status_code}")
    
    @task(4)
    def create_composition(self):
        """Create a new composition."""
        if len(self.capability_ids) < 2:
            self._populate_capability_ids()
        
        if len(self.capability_ids) >= 2:
            selected_caps = random.sample(self.capability_ids, min(random.randint(2, 4), len(self.capability_ids)))
            composition_data = TestDataGenerator.generate_composition(selected_caps, "LOCUST_COMP")
            
            with self.client.post("/api/compositions", json=composition_data, headers=self.auth_headers, catch_response=True) as response:
                if response.status_code == 201:
                    response.success()
                    data = self._handle_response(response, "create_composition")
                    if data and "data" in data and "composition_id" in data["data"]:
                        self.composition_ids.append(data["data"]["composition_id"])
                else:
                    response.failure(f"Failed to create composition: {response.status_code}")
    
    def _populate_capability_ids(self):
        """Get some capability IDs for composition operations."""
        response = self.client.get("/api/capabilities", params={"page": 1, "per_page": 20}, headers=self.auth_headers)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                self.capability_ids.extend([item.get("capability_id") for item in data["items"] if item.get("capability_id")])

class AnalyticsUser(BaseRegistryUser):
    """User that primarily consumes analytics and dashboard data."""
    
    wait_time = between(3, 8)  # Analytics users check periodically
    weight = 1
    
    @task(15)
    def view_dashboard(self):
        """View the main registry dashboard."""
        with self.client.get("/api/registry/dashboard", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get dashboard data: {response.status_code}")
    
    @task(10)
    def view_usage_analytics(self):
        """View usage analytics."""
        params = {}
        if random.random() < 0.3:
            # Sometimes filter by date range
            params["start_date"] = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - 
                                  timedelta(days=random.randint(1, 30))).isoformat()
        
        with self.client.get("/api/analytics/usage", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get usage analytics: {response.status_code}")
    
    @task(8)
    def view_performance_analytics(self):
        """View performance analytics."""
        time_ranges = ["1d", "7d", "30d", "90d"]
        params = {
            "time_range": random.choice(time_ranges)
        }
        
        with self.client.get("/api/analytics/performance", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get performance analytics: {response.status_code}")
    
    @task(5)
    def check_registry_health(self):
        """Check registry health status."""
        with self.client.get("/api/registry/health", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get registry health: {response.status_code}")

class MobileUser(BaseRegistryUser):
    """Mobile user with different usage patterns."""
    
    wait_time = between(1, 4)  # Mobile users are more active
    weight = 1
    
    @task(12)
    def get_mobile_capabilities(self):
        """Get mobile-optimized capabilities."""
        params = {
            "category": random.choice(TestDataGenerator.CATEGORIES) if random.random() < 0.4 else None,
            "limit": random.randint(10, 50),
            "offset": random.randint(0, 20)
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        with self.client.get("/api/mobile/capabilities", params=params, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get mobile capabilities: {response.status_code}")
    
    @task(3)
    def sync_mobile_data(self):
        """Sync mobile data."""
        sync_data = {
            "force_full_sync": random.choice([True, False])
        }
        
        with self.client.post("/api/mobile/sync", json=sync_data, headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to sync mobile data: {response.status_code}")

# =============================================================================
# Event Handlers for Reporting
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("🚀 APG Capability Registry Load Test Starting")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.user_count if hasattr(environment.runner, 'user_count') else 'Unknown'}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("🏁 APG Capability Registry Load Test Completed")
    
    # Calculate summary statistics
    stats = environment.stats.total
    print(f"Total requests: {stats.num_requests}")
    print(f"Failures: {stats.num_failures}")
    print(f"Median response time: {stats.median_response_time}ms")
    print(f"95th percentile: {stats.get_response_time_percentile(0.95)}ms")
    print(f"Average RPS: {stats.total_rps}")
    
    if stats.num_failures > 0:
        failure_rate = stats.num_failures / stats.num_requests
        print(f"Failure rate: {failure_rate:.2%}")

# =============================================================================
# Custom Load Test Shapes
# =============================================================================

from locust import LoadTestShape

class SteppedLoadShape(LoadTestShape):
    """
    A load test shape that increases users in steps.
    """
    
    step_time = 60  # seconds
    step_load = 10  # users
    spawn_rate = 2  # users per second
    time_limit = 600  # total test time in seconds
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = int(run_time / self.step_time) + 1
        user_count = min(current_step * self.step_load, 100)  # Max 100 users
        
        return (user_count, self.spawn_rate)

class SpikeLoadShape(LoadTestShape):
    """
    A load test shape that simulates traffic spikes.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 2)
        elif run_time < 120:
            return (50, 5)  # Spike
        elif run_time < 180:
            return (20, 2)  # Cool down
        elif run_time < 240:
            return (80, 8)  # Bigger spike
        elif run_time < 300:
            return (30, 3)  # Cool down
        else:
            return None  # End test