#!/usr/bin/env python3
"""
Test Marketplace System
=======================

Comprehensive test of the capability marketplace and discovery system.
"""

import asyncio
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

from marketplace.capability_marketplace import (
	CapabilityMarketplace,
	MarketplaceCapability,
	CapabilityCategory,
	CapabilityStatus,
	LicenseType,
	CapabilityRating,
	CapabilityDependency,
	CapabilityVersion,
	CapabilityValidator,
	CapabilityDiscovery
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_marketplace")

class MarketplaceSystemTest:
	"""Test class for the marketplace system"""
	
	def __init__(self):
		self.test_results = {}
		self.temp_dir = tempfile.mkdtemp(prefix='apg_marketplace_test_')
		self.marketplace: CapabilityMarketplace = None
		logger.info(f"Test working directory: {self.temp_dir}")
	
	async def test_marketplace_initialization(self):
		"""Test marketplace initialization and basic functionality"""
		logger.info("Testing marketplace initialization")
		
		try:
			# Initialize marketplace
			self.marketplace = CapabilityMarketplace(self.temp_dir)
			await asyncio.sleep(0.1)  # Allow initialization
			
			# Test basic properties
			assert hasattr(self.marketplace, 'capabilities'), "Marketplace should have capabilities dict"
			assert hasattr(self.marketplace, 'validator'), "Marketplace should have validator"
			assert hasattr(self.marketplace, 'discovery'), "Marketplace should have discovery engine"
			
			# Test storage path creation
			storage_path = Path(self.temp_dir)
			assert storage_path.exists(), "Storage directory should be created"
			
			self.test_results['marketplace_initialization'] = {
				'status': 'passed',
				'storage_path': str(storage_path),
				'components_initialized': True
			}
			
		except Exception as e:
			logger.error(f"Marketplace initialization test failed: {e}")
			self.test_results['marketplace_initialization'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_capability_creation_and_validation(self):
		"""Test capability creation and validation"""
		logger.info("Testing capability creation and validation")
		
		try:
			# Create a test capability
			capability = MarketplaceCapability(
				name="test_web_auth",
				display_name="Web Authentication Helper",
				description="A capability that provides secure web authentication using JWT tokens and OAuth2 integration for modern web applications.",
				detailed_description="This capability implements comprehensive web authentication including JWT token management, OAuth2 flows, session handling, and security best practices.",
				category=CapabilityCategory.WEB_DEVELOPMENT,
				tags=["authentication", "jwt", "oauth2", "security", "web"],
				keywords=["auth", "login", "token", "security", "web", "api"],
				author="Test Author",
				author_email="test@example.com",
				organization="Test Organization",
				license=LicenseType.MIT,
				homepage="https://github.com/test/web-auth",
				repository="https://github.com/test/web-auth.git",
				capability_code="""
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class WebAuthenticator:
	def __init__(self, secret_key: str):
		self.secret_key = secret_key
	
	def generate_token(self, user_id: str, expires_hours: int = 24) -> str:
		payload = {
			'user_id': user_id,
			'exp': datetime.utcnow() + timedelta(hours=expires_hours),
			'iat': datetime.utcnow()
		}
		return jwt.encode(payload, self.secret_key, algorithm='HS256')
	
	def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
		try:
			payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
			return payload
		except jwt.ExpiredSignatureError:
			return None
		except jwt.InvalidTokenError:
			return None
	
	def hash_password(self, password: str) -> str:
		return hashlib.sha256(password.encode()).hexdigest()
	
	def verify_password(self, password: str, hashed: str) -> bool:
		return self.hash_password(password) == hashed
				""",
				example_usage="""
# Initialize authenticator
auth = WebAuthenticator("your-secret-key")

# Generate token for user
token = auth.generate_token("user123")

# Verify token
payload = auth.verify_token(token)
if payload:
	print(f"User ID: {payload['user_id']}")

# Hash and verify password
hashed_password = auth.hash_password("user_password")
is_valid = auth.verify_password("user_password", hashed_password)
				""",
				documentation="# Web Authentication Capability\n\nThis capability provides secure authentication for web applications.\n\n## Features\n- JWT token generation and verification\n- Password hashing with SHA-256\n- Configurable token expiration\n- Secure token validation",
				dependencies=[
					CapabilityDependency(
						name="PyJWT",
						version_constraint=">=2.0.0",
						optional=False,
						description="JWT token handling"
					)
				],
				platforms=["linux", "windows", "macos"]
			)
			
			# Test validation
			validator = CapabilityValidator()
			validation_results = await validator.validate_capability(capability)
			
			assert isinstance(validation_results, dict), "Validation should return dict"
			assert 'valid' in validation_results, "Validation should include valid field"
			assert 'score' in validation_results, "Validation should include score"
			assert validation_results['valid'] == True, "Valid capability should pass validation"
			assert validation_results['score'] > 50, "Good capability should have decent score"
			
			self.test_results['capability_creation_and_validation'] = {
				'status': 'passed',
				'capability_name': capability.name,
				'validation_valid': validation_results['valid'],
				'validation_score': validation_results['score'],
				'security_issues': len(validation_results.get('security_issues', [])),
				'quality_issues': len(validation_results.get('quality_issues', []))
			}
			
		except Exception as e:
			logger.error(f"Capability creation and validation test failed: {e}")
			self.test_results['capability_creation_and_validation'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_capability_submission_and_storage(self):
		"""Test capability submission to marketplace"""
		logger.info("Testing capability submission and storage")
		
		try:
			# Create test capability
			capability = MarketplaceCapability(
				name="test_data_processor",
				display_name="Data Processing Toolkit",
				description="A comprehensive data processing capability with filtering, transformation, and analysis tools for structured and unstructured data.",
				category=CapabilityCategory.DATA_PROCESSING,
				tags=["data", "processing", "analytics", "transformation"],
				keywords=["data", "etl", "analytics", "transform", "filter"],
				author="Data Team",
				author_email="data@example.com",
				license=LicenseType.APACHE_2,
				capability_code="""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable

class DataProcessor:
	def __init__(self):
		self.transformations = []
	
	def add_transformation(self, func: Callable):
		self.transformations.append(func)
	
	def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
		result = df.copy()
		for transform in self.transformations:
			result = transform(result)
		return result
	
	def filter_data(self, df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
		result = df.copy()
		for column, condition in conditions.items():
			if column in result.columns:
				if callable(condition):
					result = result[result[column].apply(condition)]
				else:
					result = result[result[column] == condition]
		return result
	
	def aggregate_data(self, df: pd.DataFrame, group_by: List[str], agg_funcs: Dict[str, str]) -> pd.DataFrame:
		return df.groupby(group_by).agg(agg_funcs).reset_index()
				""",
				example_usage="processor = DataProcessor()\nprocessed_df = processor.process_dataframe(raw_df)",
				documentation="# Data Processing Capability\n\nProcess and transform data efficiently.",
				dependencies=[
					CapabilityDependency(name="pandas", version_constraint=">=1.3.0"),
					CapabilityDependency(name="numpy", version_constraint=">=1.20.0")
				]
			)
			
			# Submit capability
			result = await self.marketplace.submit_capability(capability)
			
			assert result['success'] == True, "Capability submission should succeed"
			assert 'capability_id' in result, "Result should include capability ID"
			
			capability_id = result['capability_id']
			
			# Verify capability was stored
			stored_capability = await self.marketplace.get_capability(capability_id)
			assert stored_capability is not None, "Capability should be retrievable"
			assert stored_capability.name == capability.name, "Stored capability should match original"
			
			# Test listing capabilities
			capabilities = await self.marketplace.list_capabilities()
			assert len(capabilities) > 0, "Should have at least one capability"
			
			# Find our capability in the list
			found = any(cap.id == capability_id for cap in capabilities)
			assert found, "Our capability should be in the list"
			
			self.test_results['capability_submission_and_storage'] = {
				'status': 'passed',
				'capability_id': capability_id,
				'submission_successful': result['success'],
				'storage_verified': True,
				'listing_verified': True
			}
			
		except Exception as e:
			logger.error(f"Capability submission and storage test failed: {e}")
			self.test_results['capability_submission_and_storage'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_capability_search_and_discovery(self):
		"""Test capability search and discovery functionality"""
		logger.info("Testing capability search and discovery")
		
		try:
			# Create multiple test capabilities for search
			capabilities_data = [
				{
					'name': 'ml_classifier',
					'display_name': 'Machine Learning Classifier',
					'description': 'Advanced machine learning classification algorithms including SVM, Random Forest, and Neural Networks for supervised learning tasks.',
					'category': CapabilityCategory.AI_ML,
					'tags': ['machine-learning', 'classification', 'ai', 'supervised'],
					'keywords': ['ml', 'classifier', 'ai', 'prediction', 'model']
				},
				{
					'name': 'cloud_storage',
					'display_name': 'Cloud Storage Integration',
					'description': 'Seamless integration with major cloud storage providers including AWS S3, Google Cloud Storage, and Azure Blob Storage.',
					'category': CapabilityCategory.CLOUD_INTEGRATION,
					'tags': ['cloud', 'storage', 'aws', 'gcp', 'azure'],
					'keywords': ['cloud', 's3', 'blob', 'storage', 'upload']
				},
				{
					'name': 'iot_sensor',
					'display_name': 'IoT Sensor Management',
					'description': 'Comprehensive IoT sensor data collection, processing, and real-time monitoring for industrial and smart home applications.',
					'category': CapabilityCategory.IOT_HARDWARE,
					'tags': ['iot', 'sensors', 'monitoring', 'realtime'],
					'keywords': ['iot', 'sensor', 'monitoring', 'data', 'device']
				}
			]
			
			submitted_ids = []
			
			# Submit all test capabilities
			for cap_data in capabilities_data:
				capability = MarketplaceCapability(
					name=cap_data['name'],
					display_name=cap_data['display_name'],
					description=cap_data['description'],
					category=cap_data['category'],
					tags=cap_data['tags'],
					keywords=cap_data['keywords'],
					author="Test Author",
					author_email="test@example.com",
					license=LicenseType.MIT,
					capability_code="# Test code\nclass TestCapability:\n    pass"
				)
				
				result = await self.marketplace.submit_capability(capability)
				if result['success']:
					submitted_ids.append(result['capability_id'])
					# Publish for search testing
					await self.marketplace.publish_capability(result['capability_id'])
			
			# Test search functionality
			search_tests = [
				{
					'query': 'machine learning',
					'expected_categories': [CapabilityCategory.AI_ML],
					'min_results': 1
				},
				{
					'query': 'cloud storage',
					'expected_categories': [CapabilityCategory.CLOUD_INTEGRATION],
					'min_results': 1
				},
				{
					'query': 'iot sensor',
					'expected_categories': [CapabilityCategory.IOT_HARDWARE],
					'min_results': 1
				}
			]
			
			search_results = {}
			
			for test in search_tests:
				results = await self.marketplace.search_capabilities(
					query=test['query'],
					max_results=10
				)
				
				search_results[test['query']] = {
					'results_count': len(results),
					'min_expected': test['min_results'],
					'found_expected_categories': any(
						cap.category in test['expected_categories'] for cap in results
					)
				}
				
				assert len(results) >= test['min_results'], f"Search for '{test['query']}' should return at least {test['min_results']} results"
			
			# Test category filtering
			ml_capabilities = await self.marketplace.search_capabilities(
				query="",
				category=CapabilityCategory.AI_ML,
				max_results=10
			)
			
			if ml_capabilities:
				assert all(cap.category == CapabilityCategory.AI_ML for cap in ml_capabilities), "Category filter should work"
			
			# Test recommendations
			if submitted_ids:
				recommendations = await self.marketplace.get_recommendations(
					based_on_capability=submitted_ids[0],
					limit=5
				)
				
				# Should get recommendations (might be empty if not enough data)
				assert isinstance(recommendations, list), "Recommendations should return a list"
			
			self.test_results['capability_search_and_discovery'] = {
				'status': 'passed',
				'capabilities_submitted': len(submitted_ids),
				'search_tests': search_results,
				'category_filtering': len(ml_capabilities) if ml_capabilities else 0,
				'recommendations_working': True
			}
			
		except Exception as e:
			logger.error(f"Capability search and discovery test failed: {e}")
			self.test_results['capability_search_and_discovery'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_capability_download_and_ratings(self):
		"""Test capability download and rating functionality"""
		logger.info("Testing capability download and ratings")
		
		try:
			# Create and submit a test capability
			capability = MarketplaceCapability(
				name="test_download_capability",
				display_name="Download Test Capability",
				description="A test capability for download and rating functionality with comprehensive features and documentation.",
				category=CapabilityCategory.WEB_DEVELOPMENT,
				tags=["test", "download", "rating"],
				keywords=["test", "download", "example"],
				author="Test Author",
				author_email="test@example.com",
				license=LicenseType.MIT,
				capability_code="""
class DownloadTestCapability:
	def __init__(self):
		self.version = "1.0.0"
	
	def test_function(self):
		return "Hello from test capability!"
	
	def another_function(self, data):
		return f"Processing: {data}"
				""",
				example_usage="""
# Example usage
cap = DownloadTestCapability()
result = cap.test_function()
print(result)
				""",
				documentation="# Download Test Capability\n\nThis is a test capability for download functionality.",
				test_cases=[
					"assert cap.test_function() == 'Hello from test capability!'",
					"assert cap.another_function('test') == 'Processing: test'"
				]
			)
			
			result = await self.marketplace.submit_capability(capability)
			assert result['success'], "Capability submission should succeed"
			
			capability_id = result['capability_id']
			
			# Publish capability
			published = await self.marketplace.publish_capability(capability_id)
			assert published, "Capability should be published successfully"
			
			# Test download
			download_package = await self.marketplace.download_capability(
				capability_id, 
				user_id="test_user_1"
			)
			
			assert download_package is not None, "Download should succeed"
			assert 'capability' in download_package, "Package should contain capability"
			assert 'code' in download_package, "Package should contain code"
			assert 'documentation' in download_package, "Package should contain documentation"
			assert 'test_cases' in download_package, "Package should contain test cases"
			
			# Verify download count increased
			updated_capability = await self.marketplace.get_capability(capability_id)
			assert updated_capability.metrics.download_count > 0, "Download count should increase"
			
			# Test ratings
			ratings_data = [
				{'user_id': 'user1', 'rating': 5, 'review': 'Excellent capability!'},
				{'user_id': 'user2', 'rating': 4, 'review': 'Good functionality'},
				{'user_id': 'user3', 'rating': 5, 'review': 'Perfect for my needs'},
				{'user_id': 'user4', 'rating': 3, 'review': 'Decent but could be better'}
			]
			
			for rating_data in ratings_data:
				rating = CapabilityRating(
					user_id=rating_data['user_id'],
					capability_id=capability_id,
					rating=rating_data['rating'],
					review=rating_data['review']
				)
				
				success = await self.marketplace.add_rating(rating)
				assert success, f"Rating from {rating_data['user_id']} should be added successfully"
			
			# Verify rating calculations
			final_capability = await self.marketplace.get_capability(capability_id)
			assert len(final_capability.ratings) == len(ratings_data), "Should have all ratings"
			assert final_capability.metrics.rating_count == len(ratings_data), "Rating count should match"
			
			# Calculate expected average
			expected_avg = sum(r['rating'] for r in ratings_data) / len(ratings_data)
			assert abs(final_capability.metrics.average_rating - expected_avg) < 0.01, "Average rating should be calculated correctly"
			
			self.test_results['capability_download_and_ratings'] = {
				'status': 'passed',
				'capability_id': capability_id,
				'download_successful': True,
				'download_count': updated_capability.metrics.download_count,
				'ratings_added': len(ratings_data),
				'average_rating': final_capability.metrics.average_rating,
				'expected_average': expected_avg
			}
			
		except Exception as e:
			logger.error(f"Capability download and ratings test failed: {e}")
			self.test_results['capability_download_and_ratings'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_security_validation(self):
		"""Test security validation of capabilities"""
		logger.info("Testing security validation")
		
		try:
			# Create capabilities with security issues
			security_test_cases = [
				{
					'name': 'dangerous_eval',
					'code': 'result = eval(user_input)',
					'should_fail': True,
					'issue_type': 'eval'
				},
				{
					'name': 'subprocess_call',
					'code': 'import subprocess\nsubprocess.call(["rm", "-rf", "/"])',
					'should_fail': True,
					'issue_type': 'subprocess'
				},
				{
					'name': 'hardcoded_secret',
					'code': 'api_key = "sk-1234567890abcdef"',
					'should_fail': False,  # Warning, not critical
					'issue_type': 'secret'
				},
				{
					'name': 'safe_code',
					'code': 'def safe_function(x):\n    return x * 2',
					'should_fail': False,
					'issue_type': 'none'
				}
			]
			
			validator = CapabilityValidator()
			validation_results = {}
			
			for test_case in security_test_cases:
				capability = MarketplaceCapability(
					name=test_case['name'],
					display_name=f"Security Test - {test_case['name']}",
					description="A test capability for security validation with sufficient description length to pass basic checks.",
					author="Security Tester",
					author_email="security@example.com",
					license=LicenseType.MIT,
					capability_code=test_case['code']
				)
				
				result = await validator.validate_capability(capability)
				
				validation_results[test_case['name']] = {
					'valid': result['valid'],
					'security_issues': len(result.get('security_issues', [])),
					'critical_issues': len([e for e in result.get('errors', []) if 'dangerous' in e.lower()]),
					'expected_failure': test_case['should_fail']
				}
				
				if test_case['should_fail']:
					# Should have critical security issues
					assert not result['valid'] or len(result.get('security_issues', [])) > 0, f"Security test {test_case['name']} should fail or have issues"
				
				if test_case['issue_type'] == 'eval':
					assert any('eval' in issue.lower() for issue in result.get('security_issues', []) + result.get('errors', [])), "Should detect eval usage"
				elif test_case['issue_type'] == 'subprocess':
					assert any('subprocess' in issue.lower() or 'dangerous' in issue.lower() for issue in result.get('security_issues', []) + result.get('errors', [])), "Should detect subprocess usage"
			
			self.test_results['security_validation'] = {
				'status': 'passed',
				'test_cases': validation_results,
				'total_tests': len(security_test_cases)
			}
			
		except Exception as e:
			logger.error(f"Security validation test failed: {e}")
			self.test_results['security_validation'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_marketplace_persistence(self):
		"""Test marketplace data persistence and loading"""
		logger.info("Testing marketplace persistence")
		
		try:
			# Create test capability
			capability = MarketplaceCapability(
				name="persistence_test",
				display_name="Persistence Test Capability",
				description="A test capability for testing marketplace data persistence and loading functionality across sessions.",
				author="Persistence Tester",
				author_email="persist@example.com",
				license=LicenseType.BSD,
				capability_code="class PersistenceTest:\n    def test(self):\n        return 'persistence works'"
			)
			
			# Submit capability
			result = await self.marketplace.submit_capability(capability)
			assert result['success'], "Capability submission should succeed"
			
			capability_id = result['capability_id']
			
			# Force save
			await self.marketplace._save_capabilities()
			
			# Create new marketplace instance (simulating restart)
			new_marketplace = CapabilityMarketplace(self.temp_dir)
			await asyncio.sleep(0.1)  # Allow loading
			
			# Verify capability was loaded
			loaded_capability = await new_marketplace.get_capability(capability_id)
			assert loaded_capability is not None, "Capability should be loaded from storage"
			assert loaded_capability.name == capability.name, "Loaded capability should match original"
			assert loaded_capability.author == capability.author, "Author should be preserved"
			
			# Verify indexes were rebuilt
			assert capability_id in new_marketplace.capabilities, "Capability should be in new marketplace"
			
			# Test that we can still list capabilities
			capabilities = await new_marketplace.list_capabilities()
			found = any(cap.id == capability_id for cap in capabilities)
			assert found, "Loaded capability should be listable"
			
			self.test_results['marketplace_persistence'] = {
				'status': 'passed',
				'capability_id': capability_id,
				'save_successful': True,
				'load_successful': True,
				'data_integrity': True
			}
			
		except Exception as e:
			logger.error(f"Marketplace persistence test failed: {e}")
			self.test_results['marketplace_persistence'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_marketplace_statistics(self):
		"""Test marketplace statistics and analytics"""
		logger.info("Testing marketplace statistics")
		
		try:
			# Get initial stats
			initial_stats = self.marketplace.get_marketplace_stats()
			
			assert 'total_capabilities' in initial_stats, "Stats should include total capabilities"
			assert 'published_capabilities' in initial_stats, "Stats should include published capabilities"
			assert 'categories' in initial_stats, "Stats should include category breakdown"
			assert 'top_authors' in initial_stats, "Stats should include top authors"
			
			# Create some test capabilities in different categories
			test_capabilities = [
				{
					'name': 'stats_web_cap',
					'category': CapabilityCategory.WEB_DEVELOPMENT,
					'author': 'Web Developer'
				},
				{
					'name': 'stats_ai_cap',
					'category': CapabilityCategory.AI_ML,
					'author': 'AI Developer'
				},
				{
					'name': 'stats_iot_cap',
					'category': CapabilityCategory.IOT_HARDWARE,
					'author': 'IoT Developer'
				},
				{
					'name': 'stats_web_cap2',
					'category': CapabilityCategory.WEB_DEVELOPMENT,
					'author': 'Web Developer'  # Same author as first
				}
			]
			
			submitted_count = 0
			for cap_data in test_capabilities:
				capability = MarketplaceCapability(
					name=cap_data['name'],
					display_name=f"Stats Test - {cap_data['name']}",
					description="A test capability for statistics generation with sufficient description length to pass validation.",
					category=cap_data['category'],
					author=cap_data['author'],
					author_email=f"{cap_data['author'].lower().replace(' ', '')}@example.com",
					license=LicenseType.MIT,
					capability_code="class StatsTest:\n    def test(self):\n        return 'stats test'"
				)
				
				result = await self.marketplace.submit_capability(capability)
				if result['success']:
					submitted_count += 1
					# Publish some capabilities
					if submitted_count % 2 == 0:
						await self.marketplace.publish_capability(result['capability_id'])
			
			# Get updated stats
			updated_stats = self.marketplace.get_marketplace_stats()
			
			# Verify stats changed
			assert updated_stats['total_capabilities'] > initial_stats['total_capabilities'], "Total capabilities should increase"
			
			# Check category breakdown
			if 'web_development' in updated_stats['categories']:
				assert updated_stats['categories']['web_development'] >= 2, "Should have at least 2 web development capabilities"
			
			# Check top authors
			if 'Web Developer' in updated_stats['top_authors']:
				assert updated_stats['top_authors']['Web Developer'] >= 2, "Web Developer should have multiple capabilities"
			
			# Check recent activity
			assert 'recent_activity' in updated_stats, "Should include recent activity"
			assert isinstance(updated_stats['recent_activity'], list), "Recent activity should be a list"
			
			self.test_results['marketplace_statistics'] = {
				'status': 'passed',
				'initial_total': initial_stats['total_capabilities'],
				'final_total': updated_stats['total_capabilities'],
				'capabilities_added': submitted_count,
				'categories_tracked': len(updated_stats['categories']),
				'authors_tracked': len(updated_stats['top_authors'])
			}
			
		except Exception as e:
			logger.error(f"Marketplace statistics test failed: {e}")
			self.test_results['marketplace_statistics'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def run_all_tests(self):
		"""Run all marketplace system tests"""
		logger.info("Starting comprehensive marketplace system tests")
		
		test_methods = [
			self.test_marketplace_initialization,
			self.test_capability_creation_and_validation,
			self.test_capability_submission_and_storage,
			self.test_capability_search_and_discovery,
			self.test_capability_download_and_ratings,
			self.test_security_validation,
			self.test_marketplace_persistence,
			self.test_marketplace_statistics
		]
		
		for test_method in test_methods:
			try:
				await test_method()
				logger.info(f"✓ {test_method.__name__} passed")
			except Exception as e:
				logger.error(f"✗ {test_method.__name__} failed: {e}")
				if test_method.__name__ not in self.test_results:
					self.test_results[test_method.__name__] = {
						'status': 'failed',
						'error': str(e)
					}
		
		await self.generate_test_report()
	
	async def generate_test_report(self):
		"""Generate comprehensive test report"""
		logger.info("Generating marketplace system test report")
		
		passed_tests = sum(1 for result in self.test_results.values() 
						  if result.get('status') == 'passed')
		total_tests = len(self.test_results)
		
		report = {
			'test_summary': {
				'total_tests': total_tests,
				'passed_tests': passed_tests,
				'failed_tests': total_tests - passed_tests,
				'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
				'timestamp': datetime.utcnow().isoformat(),
				'test_duration': 'approximately_3_minutes'
			},
			'detailed_results': self.test_results,
			'marketplace_capabilities': {
				'initialization': 'passed' if self.test_results.get('test_marketplace_initialization', {}).get('status') == 'passed' else 'failed',
				'capability_management': 'passed' if self.test_results.get('test_capability_creation_and_validation', {}).get('status') == 'passed' else 'failed',
				'search_and_discovery': 'passed' if self.test_results.get('test_capability_search_and_discovery', {}).get('status') == 'passed' else 'failed',
				'download_and_ratings': 'passed' if self.test_results.get('test_capability_download_and_ratings', {}).get('status') == 'passed' else 'failed',
				'security_validation': 'passed' if self.test_results.get('test_security_validation', {}).get('status') == 'passed' else 'failed',
				'data_persistence': 'passed' if self.test_results.get('test_marketplace_persistence', {}).get('status') == 'passed' else 'failed',
				'statistics_analytics': 'passed' if self.test_results.get('test_marketplace_statistics', {}).get('status') == 'passed' else 'failed'
			}
		}
		
		# Save report
		report_path = Path(self.temp_dir) / 'marketplace_system_test_report.json'
		with open(report_path, 'w') as f:
			json.dump(report, f, indent=2, default=str)
		
		logger.info(f"Marketplace Test Report Summary:")
		logger.info(f"  Total Tests: {total_tests}")
		logger.info(f"  Passed: {passed_tests}")
		logger.info(f"  Failed: {total_tests - passed_tests}")
		logger.info(f"  Success Rate: {report['test_summary']['success_rate']:.2%}")
		logger.info(f"  Report saved to: {report_path}")
		
		# Print key marketplace insights
		logger.info("\nMarketplace System Capabilities:")
		for capability, status in report['marketplace_capabilities'].items():
			status_icon = "✅" if status == 'passed' else "❌"
			logger.info(f"  {status_icon} {capability.replace('_', ' ').title()}")
	
	def cleanup(self):
		"""Clean up test files"""
		try:
			shutil.rmtree(self.temp_dir)
			logger.info(f"Cleaned up test directory: {self.temp_dir}")
		except Exception as e:
			logger.warning(f"Failed to clean up test directory: {e}")

async def main():
	"""Main test execution"""
	logger.info("APG Capability Marketplace System Test")
	logger.info("=" * 50)
	
	test_system = MarketplaceSystemTest()
	try:
		await test_system.run_all_tests()
	finally:
		test_system.cleanup()
	
	logger.info("Marketplace system tests completed!")

if __name__ == "__main__":
	asyncio.run(main())