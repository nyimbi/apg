"""
APG Event Streaming Bus - Security Audit Tests

Comprehensive security testing and vulnerability assessment for production validation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import aiohttp
import ssl
import socket
import json
import re
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import hmac
from urllib.parse import urljoin, urlparse
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
	"""Individual security finding."""
	severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
	category: str  # Authentication, Authorization, Encryption, etc.
	title: str
	description: str
	recommendation: str
	affected_endpoint: Optional[str] = None
	evidence: Optional[str] = None
	cvss_score: Optional[float] = None

@dataclass
class SecurityAuditReport:
	"""Complete security audit report."""
	target_url: str
	audit_time: datetime
	total_findings: int
	critical_findings: int
	high_findings: int
	medium_findings: int
	low_findings: int
	info_findings: int
	findings: List[SecurityFinding]
	passed_checks: List[str]
	overall_risk_level: str
	
	def get_findings_by_severity(self, severity: str) -> List[SecurityFinding]:
		"""Get findings by severity level."""
		return [f for f in self.findings if f.severity == severity]

class SecurityAuditor:
	"""Comprehensive security auditor for Event Streaming Bus."""
	
	def __init__(self, base_url: str, api_key: Optional[str] = None):
		self.base_url = base_url.rstrip('/')
		self.api_key = api_key
		self.session: Optional[aiohttp.ClientSession] = None
		self.findings: List[SecurityFinding] = []
		self.passed_checks: List[str] = []
	
	async def setup(self):
		"""Setup security audit environment."""
		logger.info("Setting up security audit environment...")
		
		# Create session with custom SSL context for testing
		ssl_context = ssl.create_default_context()
		ssl_context.check_hostname = False
		ssl_context.verify_mode = ssl.CERT_NONE
		
		connector = aiohttp.TCPConnector(ssl=ssl_context)
		timeout = aiohttp.ClientTimeout(total=30)
		
		headers = {}
		if self.api_key:
			headers['Authorization'] = f'Bearer {self.api_key}'
		
		self.session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers=headers
		)
		
		logger.info("Security audit environment setup completed")
	
	async def teardown(self):
		"""Cleanup audit environment."""
		if self.session:
			await self.session.close()
	
	def _add_finding(self, finding: SecurityFinding):
		"""Add a security finding."""
		self.findings.append(finding)
		logger.warning(f"Security finding: [{finding.severity}] {finding.title}")
	
	def _add_passed_check(self, check_name: str):
		"""Add a passed security check."""
		self.passed_checks.append(check_name)
		logger.info(f"Security check passed: {check_name}")
	
	async def _make_request(self, method: str, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
		"""Make HTTP request with error handling."""
		url = urljoin(self.base_url, endpoint)
		
		try:
			async with self.session.request(method, url, **kwargs) as response:
				return response
		except Exception as e:
			logger.error(f"Request failed: {method} {url} - {e}")
			raise
	
	async def test_ssl_tls_configuration(self):
		"""Test SSL/TLS configuration."""
		logger.info("Testing SSL/TLS configuration...")
		
		parsed_url = urlparse(self.base_url)
		hostname = parsed_url.hostname
		port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
		
		if parsed_url.scheme != 'https':
			self._add_finding(SecurityFinding(
				severity="HIGH",
				category="Encryption",
				title="Unencrypted HTTP Connection",
				description="The application is accessible over unencrypted HTTP",
				recommendation="Enforce HTTPS for all connections and redirect HTTP to HTTPS",
				affected_endpoint=self.base_url
			))
			return
		
		try:
			# Test SSL connection
			context = ssl.create_default_context()
			
			with socket.create_connection((hostname, port), timeout=10) as sock:
				with context.wrap_socket(sock, server_hostname=hostname) as ssock:
					# Check TLS version
					tls_version = ssock.version()
					cipher = ssock.cipher()
					
					if tls_version in ['TLSv1.2', 'TLSv1.3']:
						self._add_passed_check(f"TLS version check ({tls_version})")
					else:
						self._add_finding(SecurityFinding(
							severity="HIGH",
							category="Encryption",
							title="Weak TLS Version",
							description=f"Using TLS version: {tls_version}",
							recommendation="Upgrade to TLS 1.2 or 1.3",
							evidence=f"TLS version: {tls_version}"
						))
					
					# Check cipher suite
					if cipher and len(cipher) >= 3:
						cipher_name = cipher[0]
						if any(weak in cipher_name.lower() for weak in ['rc4', 'des', 'md5']):
							self._add_finding(SecurityFinding(
								severity="MEDIUM",
								category="Encryption",
								title="Weak Cipher Suite",
								description=f"Weak cipher suite detected: {cipher_name}",
								recommendation="Configure strong cipher suites only",
								evidence=f"Cipher: {cipher_name}"
							))
						else:
							self._add_passed_check("Strong cipher suite")
		
		except Exception as e:
			self._add_finding(SecurityFinding(
				severity="MEDIUM",
				category="Encryption",
				title="SSL/TLS Configuration Error",
				description=f"Unable to verify SSL/TLS configuration: {e}",
				recommendation="Verify SSL/TLS certificate and configuration"
			))
	
	async def test_authentication_mechanisms(self):
		"""Test authentication mechanisms."""
		logger.info("Testing authentication mechanisms...")
		
		# Test unauthenticated access to protected endpoints
		protected_endpoints = [
			'/api/v1/events',
			'/api/v1/streams',
			'/api/v1/subscriptions',
			'/api/v1/schemas',
			'/metrics'
		]
		
		for endpoint in protected_endpoints:
			try:
				# Remove auth headers for this test
				async with aiohttp.ClientSession() as unauth_session:
					async with unauth_session.get(f"{self.base_url}{endpoint}") as response:
						if response.status == 200:
							self._add_finding(SecurityFinding(
								severity="HIGH",
								category="Authentication",
								title="Unauthenticated Access to Protected Endpoint",
								description=f"Endpoint {endpoint} accessible without authentication",
								recommendation="Implement proper authentication for all protected endpoints",
								affected_endpoint=endpoint
							))
						elif response.status in [401, 403]:
							self._add_passed_check(f"Authentication required for {endpoint}")
			
			except Exception as e:
				logger.warning(f"Error testing endpoint {endpoint}: {e}")
		
		# Test JWT token validation (if using JWT)
		if self.api_key:
			await self._test_jwt_security()
	
	async def _test_jwt_security(self):
		"""Test JWT token security."""
		try:
			# Test with malformed JWT
			malformed_tokens = [
				"invalid.jwt.token",
				"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
				"",
				"Bearer "
			]
			
			for token in malformed_tokens:
				headers = {'Authorization': f'Bearer {token}'}
				
				async with aiohttp.ClientSession() as test_session:
					async with test_session.get(f"{self.base_url}/api/v1/events", headers=headers) as response:
						if response.status == 200:
							self._add_finding(SecurityFinding(
								severity="HIGH",
								category="Authentication",
								title="Malformed JWT Accepted",
								description="Application accepts malformed JWT tokens",
								recommendation="Implement proper JWT validation",
								evidence=f"Token: {token[:20]}..."
							))
						else:
							self._add_passed_check("JWT validation working")
							break
		
		except Exception as e:
			logger.warning(f"JWT security test error: {e}")
	
	async def test_authorization_controls(self):
		"""Test authorization and access controls."""
		logger.info("Testing authorization controls...")
		
		# Test administrative endpoints
		admin_endpoints = [
			'/api/v1/admin/users',
			'/api/v1/admin/system',
			'/api/v1/admin/metrics',
			'/admin',
			'/dashboard'
		]
		
		for endpoint in admin_endpoints:
			try:
				async with self.session.get(f"{self.base_url}{endpoint}") as response:
					if response.status == 200:
						content = await response.text()
						if any(keyword in content.lower() for keyword in ['admin', 'dashboard', 'system']):
							self._add_finding(SecurityFinding(
								severity="HIGH",
								category="Authorization",
								title="Unauthorized Admin Access",
								description=f"Admin endpoint {endpoint} accessible without proper authorization",
								recommendation="Implement role-based access control for admin functions",
								affected_endpoint=endpoint
							))
			
			except Exception as e:
				logger.debug(f"Admin endpoint {endpoint} not accessible: {e}")
	
	async def test_input_validation(self):
		"""Test input validation and injection vulnerabilities."""
		logger.info("Testing input validation...")
		
		# SQL Injection payloads
		sql_payloads = [
			"' OR '1'='1",
			"'; DROP TABLE users; --",
			"1' UNION SELECT * FROM information_schema.tables --",
			"admin'--",
			"1; WAITFOR DELAY '00:00:10'--"
		]
		
		# NoSQL Injection payloads
		nosql_payloads = [
			{"$ne": None},
			{"$regex": ".*"},
			{"$where": "this.password.length > 0"}
		]
		
		# XSS payloads
		xss_payloads = [
			"<script>alert('XSS')</script>",
			"javascript:alert('XSS')",
			"<img src=x onerror=alert('XSS')>",
			"'\"><script>alert('XSS')</script>"
		]
		
		# Test SQL injection in query parameters
		for payload in sql_payloads:
			try:
				params = {'limit': payload, 'offset': payload}
				async with self.session.get(f"{self.base_url}/api/v1/events", params=params) as response:
					if response.status == 500:
						content = await response.text()
						if any(error in content.lower() for error in ['sql', 'mysql', 'postgres', 'database']):
							self._add_finding(SecurityFinding(
								severity="CRITICAL",
								category="Input Validation",
								title="SQL Injection Vulnerability",
								description="Application vulnerable to SQL injection attacks",
								recommendation="Implement parameterized queries and input sanitization",
								affected_endpoint="/api/v1/events",
								evidence=f"Payload: {payload}"
							))
							break
				else:
					self._add_passed_check("SQL injection protection")
			
			except Exception as e:
				logger.debug(f"SQL injection test error: {e}")
		
		# Test XSS in event data
		xss_event = {
			"event_type": "test.xss",
			"source_capability": "security_test",
			"aggregate_id": "test_123",
			"aggregate_type": "Test",
			"payload": {"message": xss_payloads[0]}
		}
		
		try:
			async with self.session.post(f"{self.base_url}/api/v1/events", json=xss_event) as response:
				if response.status == 200:
					# Check if the data is reflected without encoding
					async with self.session.get(f"{self.base_url}/api/v1/events") as get_response:
						content = await get_response.text()
						if "<script>" in content:
							self._add_finding(SecurityFinding(
								severity="HIGH",
								category="Input Validation",
								title="Cross-Site Scripting (XSS)",
								description="Application vulnerable to XSS attacks",
								recommendation="Implement proper output encoding and input sanitization",
								affected_endpoint="/api/v1/events"
							))
						else:
							self._add_passed_check("XSS protection")
		
		except Exception as e:
			logger.debug(f"XSS test error: {e}")
	
	async def test_security_headers(self):
		"""Test security headers."""
		logger.info("Testing security headers...")
		
		required_headers = {
			'X-Content-Type-Options': 'nosniff',
			'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
			'X-XSS-Protection': '1; mode=block',
			'Strict-Transport-Security': None,  # Should exist for HTTPS
			'Content-Security-Policy': None,
			'Referrer-Policy': None
		}
		
		try:
			async with self.session.get(f"{self.base_url}/health") as response:
				headers = response.headers
				
				for header_name, expected_value in required_headers.items():
					if header_name not in headers:
						self._add_finding(SecurityFinding(
							severity="MEDIUM",
							category="Security Headers",
							title=f"Missing Security Header: {header_name}",
							description=f"Security header {header_name} is not set",
							recommendation=f"Add {header_name} header with appropriate value",
							affected_endpoint="/health"
						))
					else:
						if expected_value:
							if isinstance(expected_value, list):
								if headers[header_name] not in expected_value:
									self._add_finding(SecurityFinding(
										severity="LOW",
										category="Security Headers",
										title=f"Weak Security Header: {header_name}",
										description=f"Security header {header_name} has weak value: {headers[header_name]}",
										recommendation=f"Set {header_name} to one of: {expected_value}"
									))
								else:
									self._add_passed_check(f"Security header {header_name}")
							elif expected_value not in headers[header_name]:
								self._add_finding(SecurityFinding(
									severity="LOW",
									category="Security Headers",
									title=f"Weak Security Header: {header_name}",
									description=f"Security header {header_name} has unexpected value",
									recommendation=f"Set {header_name} to include: {expected_value}"
								))
							else:
								self._add_passed_check(f"Security header {header_name}")
						else:
							self._add_passed_check(f"Security header {header_name}")
		
		except Exception as e:
			logger.error(f"Security headers test error: {e}")
	
	async def test_information_disclosure(self):
		"""Test for information disclosure vulnerabilities."""
		logger.info("Testing information disclosure...")
		
		# Test error pages
		test_endpoints = [
			'/nonexistent',
			'/api/v1/nonexistent',
			'/admin/test',
			'/.env',
			'/config.json',
			'/swagger.json',
			'/api/docs'
		]
		
		for endpoint in test_endpoints:
			try:
				async with self.session.get(f"{self.base_url}{endpoint}") as response:
					content = await response.text()
					
					# Check for sensitive information in error pages
					sensitive_patterns = [
						r'stack trace',
						r'debug',
						r'exception',
						r'database.*error',
						r'connection.*string',
						r'password',
						r'secret',
						r'token',
						r'api.*key'
					]
					
					for pattern in sensitive_patterns:
						if re.search(pattern, content, re.IGNORECASE):
							self._add_finding(SecurityFinding(
								severity="MEDIUM",
								category="Information Disclosure",
								title="Sensitive Information in Error Page",
								description=f"Error page reveals sensitive information: {pattern}",
								recommendation="Implement generic error pages for production",
								affected_endpoint=endpoint,
								evidence=f"Pattern found: {pattern}"
							))
							break
					else:
						if endpoint == '/nonexistent':
							self._add_passed_check("Generic error pages")
			
			except Exception as e:
				logger.debug(f"Information disclosure test error for {endpoint}: {e}")
		
		# Test for exposed configuration
		config_endpoints = [
			'/.well-known/security.txt',
			'/robots.txt',
			'/sitemap.xml',
			'/health',
			'/metrics',
			'/status'
		]
		
		for endpoint in config_endpoints:
			try:
				async with self.session.get(f"{self.base_url}{endpoint}") as response:
					if response.status == 200:
						content = await response.text()
						
						# Check if sensitive info is exposed
						if endpoint == '/health' and 'database' in content.lower():
							if any(sensitive in content.lower() for sensitive in ['password', 'secret', 'token']):
								self._add_finding(SecurityFinding(
									severity="MEDIUM",
									category="Information Disclosure",
									title="Sensitive Information in Health Check",
									description="Health endpoint exposes sensitive configuration",
									recommendation="Remove sensitive information from health checks",
									affected_endpoint=endpoint
								))
						elif endpoint == '/metrics':
							self._add_passed_check("Metrics endpoint accessible")
			
			except Exception as e:
				logger.debug(f"Config endpoint test error for {endpoint}: {e}")
	
	async def test_rate_limiting(self):
		"""Test rate limiting mechanisms."""
		logger.info("Testing rate limiting...")
		
		# Test API rate limiting
		rapid_requests = []
		endpoint = '/api/v1/events'
		
		# Send rapid requests
		for i in range(50):
			task = asyncio.create_task(self.session.get(f"{self.base_url}{endpoint}"))
			rapid_requests.append(task)
		
		# Check responses
		rate_limited = False
		for task in asyncio.as_completed(rapid_requests):
			try:
				response = await task
				if response.status == 429:  # Too Many Requests
					rate_limited = True
					self._add_passed_check("Rate limiting implemented")
					break
			except Exception as e:
				logger.debug(f"Rate limiting test error: {e}")
		
		if not rate_limited:
			self._add_finding(SecurityFinding(
				severity="MEDIUM",
				category="Rate Limiting",
				title="No Rate Limiting Detected",
				description="API endpoints do not implement rate limiting",
				recommendation="Implement rate limiting to prevent abuse",
				affected_endpoint=endpoint
			))
	
	async def test_cors_configuration(self):
		"""Test CORS configuration."""
		logger.info("Testing CORS configuration...")
		
		# Test CORS headers
		cors_headers = {
			'Origin': 'https://malicious-site.com',
			'Access-Control-Request-Method': 'POST',
			'Access-Control-Request-Headers': 'Content-Type'
		}
		
		try:
			async with self.session.options(f"{self.base_url}/api/v1/events", headers=cors_headers) as response:
				cors_origin = response.headers.get('Access-Control-Allow-Origin')
				
				if cors_origin == '*':
					self._add_finding(SecurityFinding(
						severity="MEDIUM",
						category="CORS",
						title="Overly Permissive CORS Policy",
						description="CORS policy allows all origins (*)",
						recommendation="Restrict CORS to specific trusted origins",
						affected_endpoint="/api/v1/events"
					))
				elif cors_origin:
					self._add_passed_check("CORS policy configured")
				else:
					self._add_passed_check("CORS properly restricted")
		
		except Exception as e:
			logger.debug(f"CORS test error: {e}")
	
	async def run_security_audit(self) -> SecurityAuditReport:
		"""Run comprehensive security audit."""
		logger.info("Starting comprehensive security audit...")
		
		audit_start = datetime.now(timezone.utc)
		
		try:
			await self.setup()
			
			# Run all security tests
			await self.test_ssl_tls_configuration()
			await self.test_authentication_mechanisms()
			await self.test_authorization_controls()
			await self.test_input_validation()
			await self.test_security_headers()
			await self.test_information_disclosure()
			await self.test_rate_limiting()
			await self.test_cors_configuration()
			
		finally:
			await self.teardown()
		
		# Calculate risk level
		critical_count = len([f for f in self.findings if f.severity == "CRITICAL"])
		high_count = len([f for f in self.findings if f.severity == "HIGH"])
		medium_count = len([f for f in self.findings if f.severity == "MEDIUM"])
		low_count = len([f for f in self.findings if f.severity == "LOW"])
		info_count = len([f for f in self.findings if f.severity == "INFO"])
		
		if critical_count > 0:
			risk_level = "CRITICAL"
		elif high_count > 0:
			risk_level = "HIGH"
		elif medium_count > 2:
			risk_level = "MEDIUM"
		elif medium_count > 0 or low_count > 5:
			risk_level = "LOW"
		else:
			risk_level = "ACCEPTABLE"
		
		# Create report
		report = SecurityAuditReport(
			target_url=self.base_url,
			audit_time=audit_start,
			total_findings=len(self.findings),
			critical_findings=critical_count,
			high_findings=high_count,
			medium_findings=medium_count,
			low_findings=low_count,
			info_findings=info_count,
			findings=self.findings,
			passed_checks=self.passed_checks,
			overall_risk_level=risk_level
		)
		
		logger.info("Security audit completed")
		return report

async def run_security_audit(base_url: str, api_key: Optional[str] = None) -> SecurityAuditReport:
	"""Run security audit and generate report."""
	auditor = SecurityAuditor(base_url, api_key)
	report = await auditor.run_security_audit()
	
	# Save report
	report_data = asdict(report)
	report_data['findings'] = [asdict(f) for f in report.findings]
	
	with open(f"security_audit_report_{int(datetime.now().timestamp())}.json", "w") as f:
		json.dump(report_data, f, indent=2, default=str)
	
	# Print summary
	print("\n" + "="*60)
	print("SECURITY AUDIT RESULTS")
	print("="*60)
	print(f"Target: {report.target_url}")
	print(f"Audit Time: {report.audit_time}")
	print(f"Overall Risk Level: {report.overall_risk_level}")
	print(f"\nFindings Summary:")
	print(f"  Critical: {report.critical_findings}")
	print(f"  High: {report.high_findings}")
	print(f"  Medium: {report.medium_findings}")
	print(f"  Low: {report.low_findings}")
	print(f"  Info: {report.info_findings}")
	print(f"  Total: {report.total_findings}")
	print(f"\nPassed Checks: {len(report.passed_checks)}")
	
	if report.findings:
		print(f"\nTop Security Issues:")
		for finding in sorted(report.findings, key=lambda x: {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'INFO': 0}[x.severity], reverse=True)[:5]:
			print(f"  [{finding.severity}] {finding.title}")
			print(f"    {finding.description}")
			print(f"    Recommendation: {finding.recommendation}")
	
	print("="*60)
	
	return report

if __name__ == "__main__":
	import sys
	
	base_url = sys.argv[1] if len(sys.argv) > 1 else "https://localhost:8080"
	api_key = sys.argv[2] if len(sys.argv) > 2 else None
	
	asyncio.run(run_security_audit(base_url, api_key))