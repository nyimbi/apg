#!/usr/bin/env python3
"""
APG Employee Data Management - Load Testing Suite

Comprehensive load testing for the APG Employee Management API using Locust.
Tests various endpoints under different load conditions to ensure scalability.
"""

import json
import random
import time
from typing import Dict, List, Any

from locust import HttpUser, task, between, events
from locust.env import Environment


class EmployeeManagementUser(HttpUser):
	"""
	Load testing user for APG Employee Management API.
	Simulates realistic user behavior with various operations.
	"""
	
	wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
	
	def on_start(self):
		"""Initialize user session and authenticate."""
		self.auth_token = None
		self.employee_ids = []
		self.login()
	
	def login(self):
		"""Authenticate user and get JWT token."""
		login_data = {
			"username": f"test_user_{random.randint(1000, 9999)}",
			"password": "test_password_123"
		}
		
		response = self.client.post(
			"/api/v1/auth/login",
			json=login_data,
			catch_response=True
		)
		
		if response.status_code == 200:
			self.auth_token = response.json().get("access_token")
			self.client.headers.update({
				"Authorization": f"Bearer {self.auth_token}"
			})
		else:
			# Use mock token for load testing
			self.auth_token = "mock_token_for_load_testing"
			self.client.headers.update({
				"Authorization": f"Bearer {self.auth_token}"
			})
	
	@task(5)
	def get_employees_list(self):
		"""Test employee listing with pagination."""
		params = {
			"page": random.randint(1, 10),
			"per_page": random.choice([10, 25, 50, 100]),
			"search": random.choice(["", "john", "smith", "manager"])
		}
		
		with self.client.get(
			"/api/v1/employees",
			params=params,
			catch_response=True,
			name="GET /api/v1/employees"
		) as response:
			if response.status_code == 200:
				data = response.json()
				if "employees" in data:
					# Store some employee IDs for later use
					employees = data["employees"][:5]
					for emp in employees:
						if emp.get("id") not in self.employee_ids:
							self.employee_ids.append(emp["id"])
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(3)
	def get_employee_detail(self):
		"""Test individual employee retrieval."""
		if not self.employee_ids:
			# Use mock ID if no real IDs available
			employee_id = f"emp_{random.randint(1000, 9999)}"
		else:
			employee_id = random.choice(self.employee_ids)
		
		with self.client.get(
			f"/api/v1/employees/{employee_id}",
			catch_response=True,
			name="GET /api/v1/employees/{id}"
		) as response:
			if response.status_code in [200, 404]:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(2)
	def create_employee(self):
		"""Test employee creation."""
		employee_data = self._generate_employee_data()
		
		with self.client.post(
			"/api/v1/employees",
			json=employee_data,
			catch_response=True,
			name="POST /api/v1/employees"
		) as response:
			if response.status_code in [200, 201]:
				data = response.json()
				if "employee" in data and "id" in data["employee"]:
					self.employee_ids.append(data["employee"]["id"])
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(2)
	def update_employee(self):
		"""Test employee updates."""
		if not self.employee_ids:
			return
		
		employee_id = random.choice(self.employee_ids)
		update_data = {
			"first_name": f"Updated_{random.randint(100, 999)}",
			"last_name": f"Name_{random.randint(100, 999)}",
			"department": random.choice(["Engineering", "Sales", "Marketing", "HR"])
		}
		
		with self.client.put(
			f"/api/v1/employees/{employee_id}",
			json=update_data,
			catch_response=True,
			name="PUT /api/v1/employees/{id}"
		) as response:
			if response.status_code in [200, 404]:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(1)
	def delete_employee(self):
		"""Test employee deletion."""
		if len(self.employee_ids) < 5:  # Keep some employees
			return
		
		employee_id = self.employee_ids.pop()
		
		with self.client.delete(
			f"/api/v1/employees/{employee_id}",
			catch_response=True,
			name="DELETE /api/v1/employees/{id}"
		) as response:
			if response.status_code in [200, 204, 404]:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(2)
	def search_employees(self):
		"""Test employee search functionality."""
		search_queries = [
			"manager",
			"engineer",
			"john",
			"sales",
			"senior",
			"developer"
		]
		
		search_data = {
			"query": random.choice(search_queries),
			"filters": {
				"department": random.choice(["", "Engineering", "Sales"]),
				"location": random.choice(["", "New York", "San Francisco"])
			}
		}
		
		with self.client.post(
			"/api/v1/employees/search",
			json=search_data,
			catch_response=True,
			name="POST /api/v1/employees/search"
		) as response:
			if response.status_code == 200:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(1)
	def ai_analysis(self):
		"""Test AI analysis endpoints."""
		if not self.employee_ids:
			return
		
		employee_id = random.choice(self.employee_ids)
		
		with self.client.post(
			f"/api/v1/employees/{employee_id}/ai-analysis",
			json={"analysis_type": "performance_prediction"},
			catch_response=True,
			name="POST /api/v1/employees/{id}/ai-analysis"
		) as response:
			if response.status_code in [200, 202, 404]:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(1)
	def get_analytics_dashboard(self):
		"""Test analytics dashboard endpoint."""
		params = {
			"time_range": random.choice(["7d", "30d", "90d", "1y"]),
			"department": random.choice(["", "Engineering", "Sales", "Marketing"])
		}
		
		with self.client.get(
			"/api/v1/analytics/dashboard",
			params=params,
			catch_response=True,
			name="GET /api/v1/analytics/dashboard"
		) as response:
			if response.status_code == 200:
				response.success()
			else:
				response.failure(f"Failed with status {response.status_code}")
	
	@task(1)
	def health_check(self):
		"""Test health check endpoint."""
		with self.client.get(
			"/api/v1/health",
			catch_response=True,
			name="GET /api/v1/health"
		) as response:
			if response.status_code == 200:
				response.success()
			else:
				response.failure(f"Health check failed with status {response.status_code}")
	
	def _generate_employee_data(self) -> Dict[str, Any]:
		"""Generate realistic employee data for testing."""
		first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa"]
		last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
		departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
		positions = ["Manager", "Senior", "Junior", "Lead", "Director", "Analyst"]
		
		return {
			"employee_number": f"EMP{random.randint(10000, 99999)}",
			"first_name": random.choice(first_names),
			"last_name": random.choice(last_names),
			"email": f"test.user.{random.randint(1000, 9999)}@company.com",
			"department": random.choice(departments),
			"position": f"{random.choice(positions)} {random.choice(['Developer', 'Analyst', 'Specialist'])}",
			"hire_date": "2023-01-01",
			"salary": random.randint(50000, 150000),
			"location": random.choice(["New York", "San Francisco", "Chicago", "Austin"]),
			"manager_id": None,
			"skills": random.sample(["Python", "JavaScript", "SQL", "AWS", "Docker", "Kubernetes"], 3),
			"phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
		}


class HighVolumeUser(EmployeeManagementUser):
	"""
	High-volume user for stress testing.
	Performs more aggressive operations with shorter wait times.
	"""
	
	wait_time = between(0.1, 0.5)  # Very short wait times
	
	@task(10)
	def rapid_employee_list(self):
		"""Rapid-fire employee listing."""
		self.get_employees_list()
	
	@task(5)
	def bulk_operations(self):
		"""Perform bulk operations."""
		# Create multiple employees in quick succession
		for _ in range(3):
			self.create_employee()
			time.sleep(0.1)


class ReportingUser(HttpUser):
	"""
	User focused on reporting and analytics endpoints.
	Simulates business users generating reports.
	"""
	
	wait_time = between(2, 5)
	
	def on_start(self):
		self.login()
	
	def login(self):
		"""Mock authentication for reporting user."""
		self.client.headers.update({
			"Authorization": "Bearer mock_reporting_token"
		})
	
	@task(3)
	def generate_employee_report(self):
		"""Generate employee reports."""
		report_params = {
			"report_type": random.choice(["headcount", "turnover", "performance", "diversity"]),
			"date_range": random.choice(["last_month", "last_quarter", "last_year"]),
			"department": random.choice(["", "Engineering", "Sales", "Marketing"])
		}
		
		with self.client.post(
			"/api/v1/reports/generate",
			json=report_params,
			catch_response=True,
			name="POST /api/v1/reports/generate"
		) as response:
			if response.status_code in [200, 202]:
				response.success()
			else:
				response.failure(f"Report generation failed with status {response.status_code}")
	
	@task(2)
	def export_data(self):
		"""Test data export functionality."""
		export_params = {
			"format": random.choice(["csv", "xlsx", "pdf"]),
			"filters": {
				"department": random.choice(["Engineering", "Sales"]),
				"active": True
			}
		}
		
		with self.client.post(
			"/api/v1/employees/export",
			json=export_params,
			catch_response=True,
			name="POST /api/v1/employees/export"
		) as response:
			if response.status_code in [200, 202]:
				response.success()
			else:
				response.failure(f"Export failed with status {response.status_code}")


# Load test scenarios
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
	"""Called when the test starts."""
	print("🚀 Starting APG Employee Management Load Test")
	print(f"Target host: {environment.host}")
	print(f"Users: {getattr(environment, 'runner', None) and environment.runner.user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
	"""Called when the test stops."""
	print("📊 APG Employee Management Load Test Complete")
	if environment.runner:
		print(f"Total requests: {environment.runner.stats.total.num_requests}")
		print(f"Total failures: {environment.runner.stats.total.num_failures}")
		print(f"Average response time: {environment.runner.stats.total.avg_response_time:.2f}ms")


if __name__ == "__main__":
	# Run load test programmatically
	import os
	from locust.env import Environment
	from locust.stats import stats_printer, stats_history
	from locust.log import setup_logging
	
	setup_logging("INFO", None)
	
	# Setup environment
	env = Environment(user_classes=[EmployeeManagementUser])
	env.create_local_runner()
	
	# Start users
	user_count = int(os.getenv("LOAD_TEST_USERS", 10))
	spawn_rate = int(os.getenv("LOAD_TEST_SPAWN_RATE", 2))
	run_time = int(os.getenv("LOAD_TEST_DURATION", 60))
	
	print(f"Starting load test with {user_count} users, spawn rate {spawn_rate}, duration {run_time}s")
	
	env.runner.start(user_count, spawn_rate)
	
	# Run for specified duration
	import gevent
	gevent.spawn_later(run_time, lambda: env.runner.quit())
	
	# Start stats printing
	gevent.spawn(stats_printer(env.stats))
	gevent.spawn(stats_history, env.runner)
	
	# Wait for test to complete
	env.runner.greenlet.join()
	
	# Print final stats
	print("\n" + "="*50)
	print("FINAL LOAD TEST RESULTS")
	print("="*50)
	print(f"Total requests: {env.runner.stats.total.num_requests}")
	print(f"Total failures: {env.runner.stats.total.num_failures}")
	print(f"Failure rate: {env.runner.stats.total.fail_ratio:.2%}")
	print(f"Average response time: {env.runner.stats.total.avg_response_time:.2f}ms")
	print(f"95th percentile: {env.runner.stats.total.get_response_time_percentile(0.95):.2f}ms")
	print(f"Requests per second: {env.runner.stats.total.total_rps:.2f}")