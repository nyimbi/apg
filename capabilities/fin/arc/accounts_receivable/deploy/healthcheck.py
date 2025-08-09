#!/usr/bin/env python3
"""
APG Accounts Receivable - Health Check Script
Comprehensive health checking for containerized AR services

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any, List
import urllib.request
import socket
import subprocess


class HealthChecker:
	"""Comprehensive health checker for APG AR services."""
	
	def __init__(self):
		self.checks: List[Dict[str, Any]] = []
		self.start_time = time.time()
		
	async def run_health_checks(self) -> Dict[str, Any]:
		"""Run all health checks and return status."""
		print("🏥 Running APG AR Health Checks")
		print("=" * 40)
		
		# Core service checks
		await self.check_api_health()
		await self.check_database_connectivity()
		await self.check_redis_connectivity()
		await self.check_disk_space()
		await self.check_memory_usage()
		await self.check_configuration()
		
		# Optional AI service checks
		await self.check_ai_services()
		
		# Generate health report
		return self.generate_health_report()
	
	async def check_api_health(self):
		"""Check API service health."""
		check_name = "API Service"
		
		try:
			host = os.getenv('APG_AR_HOST', '0.0.0.0')
			port = int(os.getenv('APG_AR_PORT', '8000'))
			
			# Try to connect to the API port
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.settimeout(5)
			result = sock.connect_ex((host, port))
			sock.close()
			
			if result == 0:
				# Try to make an HTTP request to health endpoint
				try:
					url = f"http://{host}:{port}/health"
					with urllib.request.urlopen(url, timeout=5) as response:
						if response.status == 200:
							self.add_check(check_name, True, "API service is responding")
						else:
							self.add_check(check_name, False, f"API returned status {response.status}")
				except Exception as e:
					self.add_check(check_name, False, f"HTTP request failed: {e}")
			else:
				self.add_check(check_name, False, f"Cannot connect to port {port}")
				
		except Exception as e:
			self.add_check(check_name, False, f"API health check failed: {e}")
	
	async def check_database_connectivity(self):
		"""Check PostgreSQL database connectivity."""
		check_name = "Database Connectivity"
		
		try:
			database_url = os.getenv('DATABASE_URL')
			if not database_url:
				self.add_check(check_name, False, "DATABASE_URL not configured")
				return
			
			# Parse database URL to get host and port
			if '://' in database_url:
				parts = database_url.split('://', 1)[1]
				if '@' in parts:
					auth_and_host = parts.split('@', 1)[1]
					if '/' in auth_and_host:
						host_and_port = auth_and_host.split('/', 1)[0]
						if ':' in host_and_port:
							host, port = host_and_port.split(':', 1)
							port = int(port)
						else:
							host = host_and_port
							port = 5432
						
						# Test connection
						sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
						sock.settimeout(10)
						result = sock.connect_ex((host, port))
						sock.close()
						
						if result == 0:
							self.add_check(check_name, True, f"Database accessible at {host}:{port}")
						else:
							self.add_check(check_name, False, f"Cannot connect to database at {host}:{port}")
					else:
						self.add_check(check_name, False, "Invalid DATABASE_URL format")
				else:
					self.add_check(check_name, False, "Invalid DATABASE_URL format")
			else:
				self.add_check(check_name, False, "Invalid DATABASE_URL format")
				
		except Exception as e:
			self.add_check(check_name, False, f"Database connectivity check failed: {e}")
	
	async def check_redis_connectivity(self):
		"""Check Redis connectivity."""
		check_name = "Redis Connectivity"
		
		try:
			redis_url = os.getenv('REDIS_URL')
			if not redis_url:
				self.add_check(check_name, False, "REDIS_URL not configured")
				return
			
			# Parse Redis URL
			if redis_url.startswith('redis://'):
				url_part = redis_url[8:]  # Remove redis://
				if ':' in url_part:
					host, port_and_db = url_part.split(':', 1)
					if '/' in port_and_db:
						port = int(port_and_db.split('/', 1)[0])
					else:
						port = int(port_and_db)
				else:
					host = url_part
					port = 6379
				
				# Test connection
				sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				sock.settimeout(5)
				result = sock.connect_ex((host, port))
				sock.close()
				
				if result == 0:
					self.add_check(check_name, True, f"Redis accessible at {host}:{port}")
				else:
					self.add_check(check_name, False, f"Cannot connect to Redis at {host}:{port}")
			else:
				self.add_check(check_name, False, "Invalid REDIS_URL format")
				
		except Exception as e:
			self.add_check(check_name, False, f"Redis connectivity check failed: {e}")
	
	async def check_disk_space(self):
		"""Check available disk space."""
		check_name = "Disk Space"
		
		try:
			# Check main application directory
			result = subprocess.run(['df', '/opt/apg'], capture_output=True, text=True)
			if result.returncode == 0:
				lines = result.stdout.strip().split('\n')
				if len(lines) >= 2:
					fields = lines[1].split()
					if len(fields) >= 5:
						available = int(fields[3])  # Available space in KB
						usage_percent = int(fields[4].rstrip('%'))
						
						if usage_percent < 90:
							self.add_check(check_name, True, f"Disk usage: {usage_percent}%, {available/1024/1024:.1f}GB available")
						else:
							self.add_check(check_name, False, f"Disk usage critical: {usage_percent}%")
					else:
						self.add_check(check_name, False, "Cannot parse disk usage")
				else:
					self.add_check(check_name, False, "Cannot parse df output")
			else:
				self.add_check(check_name, False, f"df command failed: {result.stderr}")
				
		except Exception as e:
			self.add_check(check_name, False, f"Disk space check failed: {e}")
	
	async def check_memory_usage(self):
		"""Check memory usage."""
		check_name = "Memory Usage"
		
		try:
			# Read /proc/meminfo
			with open('/proc/meminfo', 'r') as f:
				meminfo = f.read()
			
			mem_total = None
			mem_available = None
			
			for line in meminfo.split('\n'):
				if line.startswith('MemTotal:'):
					mem_total = int(line.split()[1])  # in KB
				elif line.startswith('MemAvailable:'):
					mem_available = int(line.split()[1])  # in KB
			
			if mem_total and mem_available:
				mem_used = mem_total - mem_available
				usage_percent = (mem_used / mem_total) * 100
				
				if usage_percent < 90:
					self.add_check(check_name, True, f"Memory usage: {usage_percent:.1f}%, {mem_available/1024/1024:.1f}GB available")
				else:
					self.add_check(check_name, False, f"Memory usage critical: {usage_percent:.1f}%")
			else:
				self.add_check(check_name, False, "Cannot read memory information")
				
		except Exception as e:
			self.add_check(check_name, False, f"Memory usage check failed: {e}")
	
	async def check_configuration(self):
		"""Check critical configuration parameters."""
		check_name = "Configuration"
		
		try:
			required_env_vars = [
				'APG_ENVIRONMENT',
				'DATABASE_URL',
				'REDIS_URL',
				'SECRET_KEY'
			]
			
			missing_vars = []
			for var in required_env_vars:
				if not os.getenv(var):
					missing_vars.append(var)
			
			if not missing_vars:
				self.add_check(check_name, True, "All required environment variables are set")
			else:
				self.add_check(check_name, False, f"Missing environment variables: {', '.join(missing_vars)}")
				
		except Exception as e:
			self.add_check(check_name, False, f"Configuration check failed: {e}")
	
	async def check_ai_services(self):
		"""Check AI service connectivity (optional)."""
		check_name = "AI Services"
		
		try:
			# These would be URLs to APG AI services
			ai_services = {
				'Federated Learning': os.getenv('FEDERATED_LEARNING_URL'),
				'AI Orchestration': os.getenv('AI_ORCHESTRATION_URL'),
				'Time Series Analytics': os.getenv('TIME_SERIES_ANALYTICS_URL')
			}
			
			service_status = []
			for service_name, service_url in ai_services.items():
				if service_url:
					try:
						# Simple connectivity check
						with urllib.request.urlopen(f"{service_url}/health", timeout=5) as response:
							if response.status == 200:
								service_status.append(f"{service_name}: OK")
							else:
								service_status.append(f"{service_name}: HTTP {response.status}")
					except Exception:
						service_status.append(f"{service_name}: Unavailable")
				else:
					service_status.append(f"{service_name}: Not configured")
			
			# AI services are optional, so this is informational
			self.add_check(check_name, True, "; ".join(service_status))
			
		except Exception as e:
			self.add_check(check_name, True, f"AI services check failed: {e} (non-critical)")
	
	def add_check(self, name: str, passed: bool, message: str):
		"""Add a health check result."""
		status_icon = "✅" if passed else "❌"
		print(f"{status_icon} {name}: {message}")
		
		self.checks.append({
			'name': name,
			'passed': passed,
			'message': message,
			'timestamp': time.time()
		})
	
	def generate_health_report(self) -> Dict[str, Any]:
		"""Generate comprehensive health report."""
		passed_checks = [c for c in self.checks if c['passed']]
		failed_checks = [c for c in self.checks if not c['passed']]
		
		overall_health = len(failed_checks) == 0
		
		report = {
			'status': 'healthy' if overall_health else 'unhealthy',
			'timestamp': time.time(),
			'duration_ms': (time.time() - self.start_time) * 1000,
			'checks': {
				'total': len(self.checks),
				'passed': len(passed_checks),
				'failed': len(failed_checks)
			},
			'details': self.checks
		}
		
		return report


async def main():
	"""Main health check execution."""
	checker = HealthChecker()
	
	try:
		report = await checker.run_health_checks()
		
		print("\n" + "=" * 40)
		print(f"🏥 Health Check Summary")
		print(f"Status: {report['status'].upper()}")
		print(f"Checks: {report['checks']['passed']}/{report['checks']['total']} passed")
		print(f"Duration: {report['duration_ms']:.1f}ms")
		
		# Exit with appropriate code
		if report['status'] == 'healthy':
			print("✅ All health checks passed")
			sys.exit(0)
		else:
			print("❌ Some health checks failed")
			sys.exit(1)
			
	except Exception as e:
		print(f"💥 Health check system failed: {e}")
		sys.exit(2)


if __name__ == "__main__":
	asyncio.run(main())