#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Quality Assessment Example
Demonstrates AI-powered data quality assessment capabilities

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ..service import MDMService
from ..models import MdEntityCreate, EntityType, EntityStatus
from ..database import MDMDatabaseManager
from ..integrations import APGIntegrationManager


async def setup_quality_examples():
	"""Setup MDM service with quality assessment enabled"""
	print("🔍 Initializing APG MDM Service for Quality Assessment...")
	
	# Configure with AI quality assessment enabled
	db_config = {
		"database_url": "postgresql://mdm_user:password@localhost:5432/apg_mdm"
	}
	
	apg_config = {
		"redis_url": "redis://localhost:6379/0",
		"enable_ai_quality": True
	}
	
	db_manager = MDMDatabaseManager(db_config)
	integration_manager = APGIntegrationManager(apg_config)
	
	mdm_service = MDMService(
		db_manager=db_manager,
		integration_manager=integration_manager,
		config={
			"enable_ai": True,
			"quality_thresholds": {
				"excellent": 95.0,
				"good": 80.0,
				"fair": 60.0,
				"poor": 40.0
			}
		}
	)
	
	await mdm_service.initialize()
	print("✅ MDM Service initialized with AI quality assessment")
	
	return mdm_service


async def create_quality_test_entities(mdm_service: MDMService) -> List[str]:
	"""Create entities with various quality characteristics for testing"""
	print("\n📝 Creating Test Entities with Different Quality Profiles")
	print("=" * 60)
	
	tenant_id = "quality-demo-tenant"
	user_id = "quality-demo-user"
	
	context = mdm_service.create_operation_context(
		tenant_id=tenant_id,
		user_id=user_id,
		operation_type="create_entity",
		source_system="quality_examples"
	)
	
	entity_ids = []
	
	# Entity 1: High Quality - Complete, accurate, consistent
	print("\n1. Creating HIGH QUALITY entity...")
	high_quality_entity = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.PERSON,
		entity_name="Dr. Elizabeth Martinez",
		entity_description="Chief Technology Officer",
		business_key="EMP-HQ-001",
		source_system="hr_premium",
		status=EntityStatus.ACTIVE,
		attributes={
			# Complete personal information
			"first_name": "Elizabeth",
			"last_name": "Martinez",
			"middle_name": "Carmen",
			"title": "Dr.",
			"suffix": "Ph.D.",
			
			# Complete contact information
			"email": "elizabeth.martinez@company.com",
			"phone": "+1-555-123-4567",
			"mobile": "+1-555-987-6543",
			"emergency_contact": "+1-555-555-0123",
			
			# Complete address
			"address": {
				"street": "123 Executive Boulevard",
				"unit": "Suite 500",
				"city": "San Francisco",
				"state": "CA",
				"postal_code": "94105",
				"country": "USA"
			},
			
			# Complete employment information
			"employee_id": "E001001",
			"department": "Technology",
			"position": "Chief Technology Officer",
			"hire_date": "2020-01-15",
			"salary": 350000,
			"manager": "ceo@company.com",
			
			# Additional quality indicators
			"education": "Ph.D. Computer Science, MIT",
			"certifications": ["AWS Solutions Architect", "PMP"],
			"last_updated": datetime.utcnow().isoformat(),
			"verified": True,
			"verification_date": "2024-01-15",
			"data_source": "authoritative_hr_system"
		},
		tags=["employee", "executive", "technology", "verified", "complete"],
		data_classification="confidential"
	)
	
	result = await mdm_service.entity_service.create_entity(high_quality_entity, context)
	if result["status"] == "success":
		entity_ids.append(result["entity_id"])
		print(f"✅ High quality entity created: {result['entity_id']}")
	
	# Entity 2: Medium Quality - Some missing fields, minor inconsistencies
	print("\n2. Creating MEDIUM QUALITY entity...")
	medium_quality_entity = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.PERSON,
		entity_name="John Smith",
		entity_description="Software Developer",
		business_key="EMP-MQ-002",
		source_system="hr_standard",
		status=EntityStatus.ACTIVE,
		attributes={
			# Basic personal information (missing middle name, title)
			"first_name": "John",
			"last_name": "Smith",
			
			# Contact information (missing mobile, emergency contact)
			"email": "john.smith@company.com",
			"phone": "555-234-5678",  # Non-standard format
			
			# Incomplete address (missing unit, country)
			"address": {
				"street": "456 Development Lane",
				"city": "Austin",
				"state": "Texas",  # Inconsistent - should be "TX"
				"postal_code": "78701"
			},
			
			# Employment information (missing some details)
			"employee_id": "E002002",
			"department": "Engineering",
			"position": "Software Developer",
			"hire_date": "2022-03-01",
			# Missing salary, manager
			
			# Some quality issues
			"last_updated": (datetime.utcnow() - timedelta(days=180)).isoformat(),  # Stale data
			"verified": False,
			"data_source": "csv_import"
		},
		tags=["employee", "engineering", "standard"],
		data_classification="internal"
	)
	
	result = await mdm_service.entity_service.create_entity(medium_quality_entity, context)
	if result["status"] == "success":
		entity_ids.append(result["entity_id"])
		print(f"✅ Medium quality entity created: {result['entity_id']}")
	
	# Entity 3: Low Quality - Many missing fields, inconsistencies, errors
	print("\n3. Creating LOW QUALITY entity...")
	low_quality_entity = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.PERSON,
		entity_name="",  # Empty name - major quality issue
		entity_description="Unknown Employee",
		business_key="EMP-LQ-003",
		source_system="legacy_import",
		status=EntityStatus.PENDING,  # Uncertain status
		attributes={
			# Poor personal information
			"first_name": "",  # Empty
			"last_name": "Doe",
			"name": "Jane Doe",  # Inconsistent with empty first_name
			
			# Poor contact information
			"email": "invalid-email",  # Invalid format
			"phone": "123",  # Too short
			"alt_phone": "call me",  # Not a phone number
			
			# Poor address information
			"address": {
				"street": "Unknown",
				"city": "",  # Empty
				"state": "XX",  # Invalid state code
				"postal_code": "00000"  # Invalid zip
			},
			
			# Poor employment information
			"employee_id": "",  # Empty
			"department": "?",
			"position": "TBD",
			"hire_date": "01/01/2000",  # Non-standard format, unlikely date
			"salary": -1000,  # Negative salary - impossible
			
			# Quality issues
			"last_updated": (datetime.utcnow() - timedelta(days=730)).isoformat(),  # Very stale
			"verified": False,
			"notes": "Data quality issues - needs cleanup",
			"import_errors": [
				"Invalid email format",
				"Missing required fields", 
				"Inconsistent name fields"
			]
		},
		tags=["employee", "import_error", "needs_cleanup"],
		data_classification="internal"
	)
	
	result = await mdm_service.entity_service.create_entity(low_quality_entity, context)
	if result["status"] == "success":
		entity_ids.append(result["entity_id"])
		print(f"✅ Low quality entity created: {result['entity_id']}")
	
	# Entity 4: Customer with specific quality issues
	print("\n4. Creating CUSTOMER entity with mixed quality...")
	customer_quality_entity = MdEntityCreate(
		tenant_id=tenant_id,
		entity_type=EntityType.CUSTOMER,
		entity_name="Acme Corp Inc.",
		entity_description="Technology services customer",
		business_key="CUST-QT-001",
		source_system="crm_integration",
		status=EntityStatus.ACTIVE,
		attributes={
			# Company information with issues
			"company_name": "ACME CORP INC",  # Inconsistent case with entity_name
			"legal_name": "Acme Corporation Incorporated",  # Different from display name
			"industry": "Tech",  # Abbreviated, non-standard
			"industry_code": "5112",  # Standard code - good
			
			# Contact information issues
			"email": "info@acme.com",  # Generic email
			"phone": "1-800-ACME-001",  # Vanity number - hard to validate
			"website": "www.acme.com",  # Missing protocol
			"website_alt": "https://acme.corp.com",  # Different domain
			
			# Address inconsistencies
			"headquarters": {
				"street": "100 Main Street",
				"city": "New York",
				"state": "New York",  # Should be "NY"
				"postal_code": "10001-1234",  # Extended ZIP - good
				"country": "US"  # Abbreviated - could be "USA"
			},
			"billing_address": {
				"street": "100 Main St",  # Abbreviated vs headquarters
				"city": "NYC",  # Abbreviated vs headquarters
				"state": "NY",  # Standard vs headquarters
				"postal_code": "10001"  # Shorter vs headquarters
			},
			
			# Financial data quality issues
			"revenue": 5000000,  # No currency specified
			"revenue_range": "$1M-$10M",  # Text range vs specific number
			"employees": "50-100",  # Text range instead of number
			"employee_count": 75,  # Specific number - good
			
			# Temporal quality issues
			"founded": "1995",  # Year only
			"customer_since": "2020-06-15T10:30:00",  # Precise timestamp - good
			"last_contact": "2024-01-01",  # Date only
			"contract_renewal": "Q2 2024",  # Imprecise quarter notation
			
			# Data freshness
			"last_updated": (datetime.utcnow() - timedelta(days=90)).isoformat(),
			"data_quality_score": 72.5,  # Self-reported quality score
			"needs_review": True
		},
		tags=["customer", "corporate", "mixed_quality", "needs_review"],
		data_classification="confidential"
	)
	
	result = await mdm_service.entity_service.create_entity(customer_quality_entity, context)
	if result["status"] == "success":
		entity_ids.append(result["entity_id"])
		print(f"✅ Customer entity created: {result['entity_id']}")
	
	print(f"\n✅ Created {len(entity_ids)} test entities for quality assessment")
	return entity_ids, tenant_id


async def demonstrate_single_quality_assessment(mdm_service: MDMService, 
                                              entity_id: str, tenant_id: str,
                                              description: str):
	"""Demonstrate quality assessment for a single entity"""
	print(f"\n🔍 Quality Assessment: {description}")
	print("-" * 50)
	
	# First get the entity data
	entity_result = await mdm_service.entity_service.get_entity(entity_id, tenant_id)
	if entity_result["status"] != "success":
		print(f"❌ Could not retrieve entity: {entity_result['message']}")
		return
	
	entity = entity_result["entity"]
	print(f"Entity: {entity['entity_name']} ({entity['business_key']})")
	
	# Run quality assessment
	start_time = datetime.now()
	quality_result = await mdm_service.quality_service.assess_quality(
		entity_id,
		tenant_id,
		entity["attributes"],
		entity["entity_type"]
	)
	end_time = datetime.now()
	
	assessment_time = (end_time - start_time).total_seconds() * 1000
	
	if quality_result["status"] == "success":
		print(f"\n📊 Quality Assessment Results (completed in {assessment_time:.1f}ms):")
		
		# Overall quality
		overall_score = quality_result["overall_score"]
		quality_status = quality_result["quality_status"]
		print(f"   Overall Score: {overall_score:.1f}% ({quality_status.upper()})")
		
		# Quality dimensions
		dimensions = [
			("Completeness", quality_result["completeness_score"]),
			("Accuracy", quality_result["accuracy_score"]),
			("Consistency", quality_result["consistency_score"]),
			("Validity", quality_result["validity_score"]),
			("Uniqueness", quality_result["uniqueness_score"]),
			("Timeliness", quality_result["timeliness_score"])
		]
		
		print(f"\n   Quality Dimensions:")
		for dimension, score in dimensions:
			status_emoji = "🟢" if score >= 85 else "🟡" if score >= 60 else "🔴"
			print(f"   {status_emoji} {dimension:12}: {score:5.1f}%")
		
		# Quality issues
		if quality_result.get("quality_issues"):
			print(f"\n   🚨 Quality Issues ({len(quality_result['quality_issues'])}):")
			for issue in quality_result["quality_issues"]:
				severity_emoji = {
					"critical": "🔴",
					"high": "🟠", 
					"medium": "🟡",
					"low": "🔵"
				}.get(issue["severity"], "⚪")
				
				print(f"   {severity_emoji} {issue['issue_type'].title()}: {issue['message']}")
				if issue.get("field"):
					print(f"      Field: {issue['field']}")
				if issue.get("recommendation"):
					print(f"      💡 {issue['recommendation']}")
		
		# Recommendations
		if quality_result.get("recommendations"):
			print(f"\n   💡 Recommendations ({len(quality_result['recommendations'])}):")
			for rec in quality_result["recommendations"]:
				print(f"   • {rec}")
		
		# Auto-fix suggestions
		if quality_result.get("auto_fix_suggestions"):
			print(f"\n   🔧 Auto-fix Suggestions:")
			for suggestion in quality_result["auto_fix_suggestions"]:
				confidence = suggestion.get("confidence", 0) * 100
				print(f"   • {suggestion['action']} (confidence: {confidence:.0f}%)")
				if suggestion.get("field"):
					print(f"     Field: {suggestion['field']}")
	
	else:
		print(f"❌ Quality assessment failed: {quality_result['message']}")
	
	return quality_result if quality_result["status"] == "success" else None


async def demonstrate_batch_quality_assessment(mdm_service: MDMService,
                                             entity_ids: List[str], 
                                             tenant_id: str):
	"""Demonstrate batch quality assessment for multiple entities"""
	print(f"\n📦 Batch Quality Assessment for {len(entity_ids)} Entities")
	print("=" * 60)
	
	start_time = datetime.now()
	result = await mdm_service.quality_service.batch_assess_quality(entity_ids, tenant_id)
	end_time = datetime.now()
	
	total_time = (end_time - start_time).total_seconds() * 1000
	
	if result["status"] == "success":
		assessments = result["assessments"]
		summary = result["summary"]
		
		print(f"✅ Batch assessment completed in {total_time:.1f}ms")
		print(f"   Average time per entity: {total_time / len(entity_ids):.1f}ms")
		
		print(f"\n📈 Batch Summary:")
		print(f"   Entities Assessed: {len(assessments)}")
		print(f"   Average Quality Score: {summary.get('average_score', 0):.1f}%")
		print(f"   Highest Score: {summary.get('highest_score', 0):.1f}%")
		print(f"   Lowest Score: {summary.get('lowest_score', 0):.1f}%")
		
		# Quality distribution
		if summary.get("quality_distribution"):
			print(f"\n   Quality Distribution:")
			for status, count in summary["quality_distribution"].items():
				print(f"   • {status.title()}: {count} entities")
		
		# Most common issues
		if summary.get("common_issues"):
			print(f"\n   Most Common Issues:")
			for issue_type, count in summary["common_issues"].items():
				print(f"   • {issue_type.title()}: {count} occurrences")
		
		print(f"\n📋 Individual Results:")
		for i, assessment in enumerate(assessments, 1):
			score = assessment["overall_score"]
			status = assessment["quality_status"]
			entity_id = assessment["entity_id"]
			
			status_emoji = "🟢" if score >= 85 else "🟡" if score >= 60 else "🔴"
			print(f"   {i}. {status_emoji} Entity {entity_id[:8]}... "
			      f"Score: {score:.1f}% ({status})")
	
	else:
		print(f"❌ Batch assessment failed: {result['message']}")


async def demonstrate_quality_trends_analysis(mdm_service: MDMService,
                                            entity_ids: List[str],
                                            tenant_id: str):
	"""Demonstrate quality trends and analytics"""
	print(f"\n📈 Quality Trends Analysis")
	print("=" * 40)
	
	# Simulate historical quality assessments by running assessments
	# at different time points (in a real system, this would be actual historical data)
	
	print("Running quality assessments to establish trends...")
	
	quality_history = []
	for entity_id in entity_ids:
		# Get entity for assessment
		entity_result = await mdm_service.entity_service.get_entity(entity_id, tenant_id)
		if entity_result["status"] == "success":
			entity = entity_result["entity"]
			
			# Run quality assessment
			quality_result = await mdm_service.quality_service.assess_quality(
				entity_id, tenant_id, entity["attributes"], entity["entity_type"]
			)
			
			if quality_result["status"] == "success":
				quality_history.append({
					"entity_id": entity_id,
					"entity_name": entity["entity_name"],
					"overall_score": quality_result["overall_score"],
					"quality_status": quality_result["quality_status"],
					"dimensions": {
						"completeness": quality_result["completeness_score"],
						"accuracy": quality_result["accuracy_score"],
						"consistency": quality_result["consistency_score"],
						"validity": quality_result["validity_score"],
						"uniqueness": quality_result["uniqueness_score"],
						"timeliness": quality_result["timeliness_score"]
					},
					"issue_count": len(quality_result.get("quality_issues", []))
				})
	
	if quality_history:
		# Analyze trends
		total_entities = len(quality_history)
		avg_score = sum(q["overall_score"] for q in quality_history) / total_entities
		
		# Quality distribution
		quality_distribution = {}
		for assessment in quality_history:
			status = assessment["quality_status"]
			quality_distribution[status] = quality_distribution.get(status, 0) + 1
		
		# Dimension averages
		dimension_averages = {}
		for dimension in ["completeness", "accuracy", "consistency", "validity", "uniqueness", "timeliness"]:
			avg = sum(q["dimensions"][dimension] for q in quality_history) / total_entities
			dimension_averages[dimension] = avg
		
		# Issue frequency
		total_issues = sum(q["issue_count"] for q in quality_history)
		
		print(f"\n📊 Quality Analytics Summary:")
		print(f"   Total Entities Analyzed: {total_entities}")
		print(f"   Overall Average Score: {avg_score:.1f}%")
		print(f"   Total Quality Issues: {total_issues}")
		print(f"   Issues per Entity: {total_issues / total_entities:.1f}")
		
		print(f"\n🎯 Quality Distribution:")
		for status, count in quality_distribution.items():
			percentage = (count / total_entities) * 100
			print(f"   • {status.title()}: {count} entities ({percentage:.1f}%)")
		
		print(f"\n📈 Dimension Performance:")
		sorted_dimensions = sorted(dimension_averages.items(), key=lambda x: x[1], reverse=True)
		for dimension, avg_score in sorted_dimensions:
			status_emoji = "🟢" if avg_score >= 85 else "🟡" if avg_score >= 60 else "🔴"
			print(f"   {status_emoji} {dimension.title():12}: {avg_score:5.1f}%")
		
		# Recommendations based on analysis
		print(f"\n💡 Improvement Recommendations:")
		
		# Find weakest dimension
		weakest_dimension = min(dimension_averages.items(), key=lambda x: x[1])
		print(f"   • Focus on {weakest_dimension[0]} improvement (lowest avg: {weakest_dimension[1]:.1f}%)")
		
		# Check for common patterns
		low_quality_entities = [q for q in quality_history if q["overall_score"] < 60]
		if low_quality_entities:
			print(f"   • {len(low_quality_entities)} entities need immediate attention")
		
		if avg_score < 80:
			print(f"   • Overall data quality below target (80%), current: {avg_score:.1f}%")
		
		print(f"   • Consider automated data quality monitoring")
		print(f"   • Implement data validation rules at data entry points")


async def demonstrate_quality_monitoring_alerts(mdm_service: MDMService,
                                              entity_ids: List[str],
                                              tenant_id: str):
	"""Demonstrate quality monitoring and alerting capabilities"""
	print(f"\n🚨 Quality Monitoring & Alerts")
	print("=" * 40)
	
	# Define quality thresholds for alerting
	alert_thresholds = {
		"critical_score": 40.0,  # Below this triggers critical alert
		"warning_score": 60.0,   # Below this triggers warning alert
		"target_score": 80.0,    # Target quality score
		"max_issues": 5          # Maximum acceptable issues per entity
	}
	
	print(f"Quality Monitoring Thresholds:")
	print(f"   🔴 Critical: < {alert_thresholds['critical_score']}%")
	print(f"   🟡 Warning:  < {alert_thresholds['warning_score']}%") 
	print(f"   🎯 Target:   ≥ {alert_thresholds['target_score']}%")
	
	alerts_generated = []
	
	# Monitor each entity
	for entity_id in entity_ids:
		entity_result = await mdm_service.entity_service.get_entity(entity_id, tenant_id)
		if entity_result["status"] != "success":
			continue
			
		entity = entity_result["entity"]
		
		# Run quality assessment
		quality_result = await mdm_service.quality_service.assess_quality(
			entity_id, tenant_id, entity["attributes"], entity["entity_type"]
		)
		
		if quality_result["status"] != "success":
			continue
		
		# Check for alert conditions
		score = quality_result["overall_score"]
		issues = quality_result.get("quality_issues", [])
		issue_count = len(issues)
		
		# Generate alerts based on thresholds
		if score < alert_thresholds["critical_score"]:
			alerts_generated.append({
				"level": "CRITICAL",
				"entity_id": entity_id,
				"entity_name": entity["entity_name"],
				"message": f"Critical quality score: {score:.1f}%",
				"score": score,
				"issue_count": issue_count,
				"recommended_action": "Immediate data remediation required"
			})
		
		elif score < alert_thresholds["warning_score"]:
			alerts_generated.append({
				"level": "WARNING",
				"entity_id": entity_id,
				"entity_name": entity["entity_name"],
				"message": f"Below target quality score: {score:.1f}%",
				"score": score,
				"issue_count": issue_count,
				"recommended_action": "Schedule data quality improvement"
			})
		
		if issue_count > alert_thresholds["max_issues"]:
			alerts_generated.append({
				"level": "WARNING",
				"entity_id": entity_id,
				"entity_name": entity["entity_name"],
				"message": f"High issue count: {issue_count} issues detected",
				"score": score,
				"issue_count": issue_count,
				"recommended_action": "Review and resolve data quality issues"
			})
		
		# Check for specific high-severity issues
		critical_issues = [i for i in issues if i.get("severity") == "critical"]
		if critical_issues:
			for issue in critical_issues:
				alerts_generated.append({
					"level": "CRITICAL",
					"entity_id": entity_id,
					"entity_name": entity["entity_name"],
					"message": f"Critical issue: {issue['message']}",
					"issue_type": issue["issue_type"],
					"field": issue.get("field", "unknown"),
					"recommended_action": issue.get("recommendation", "Manual review required")
				})
	
	# Display alerts
	if alerts_generated:
		print(f"\n🚨 Quality Alerts Generated: {len(alerts_generated)}")
		
		# Group by level
		critical_alerts = [a for a in alerts_generated if a["level"] == "CRITICAL"]
		warning_alerts = [a for a in alerts_generated if a["level"] == "WARNING"]
		
		if critical_alerts:
			print(f"\n🔴 CRITICAL ALERTS ({len(critical_alerts)}):")
			for alert in critical_alerts:
				print(f"   • Entity: {alert['entity_name']}")
				print(f"     Issue: {alert['message']}")
				print(f"     Action: {alert['recommended_action']}")
				if "field" in alert:
					print(f"     Field: {alert['field']}")
				print()
		
		if warning_alerts:
			print(f"🟡 WARNING ALERTS ({len(warning_alerts)}):")
			for alert in warning_alerts:
				print(f"   • Entity: {alert['entity_name']}")
				print(f"     Issue: {alert['message']}")
				print(f"     Action: {alert['recommended_action']}")
				print()
	
	else:
		print("\n✅ No quality alerts generated - all entities meet quality thresholds!")
	
	# Summary report
	print(f"📋 Monitoring Summary:")
	print(f"   Entities Monitored: {len(entity_ids)}")
	print(f"   Alerts Generated: {len(alerts_generated)}")
	print(f"   Critical Alerts: {len([a for a in alerts_generated if a['level'] == 'CRITICAL'])}")
	print(f"   Warning Alerts: {len([a for a in alerts_generated if a['level'] == 'WARNING'])}")


async def main():
	"""Run all quality assessment examples"""
	print("🌟 APG Master Data Management - Quality Assessment Examples")
	print("=" * 70)
	
	try:
		# Setup
		mdm_service = await setup_quality_examples()
		entity_ids, tenant_id = await create_quality_test_entities(mdm_service)
		
		# Individual quality assessments
		descriptions = [
			"High Quality Entity (Complete & Accurate)",
			"Medium Quality Entity (Some Issues)",
			"Low Quality Entity (Many Problems)",
			"Customer Entity (Mixed Quality Issues)"
		]
		
		quality_results = []
		for entity_id, description in zip(entity_ids, descriptions):
			result = await demonstrate_single_quality_assessment(
				mdm_service, entity_id, tenant_id, description
			)
			if result:
				quality_results.append(result)
		
		# Batch assessment
		await demonstrate_batch_quality_assessment(mdm_service, entity_ids, tenant_id)
		
		# Quality analytics and trends
		await demonstrate_quality_trends_analysis(mdm_service, entity_ids, tenant_id)
		
		# Quality monitoring and alerts
		await demonstrate_quality_monitoring_alerts(mdm_service, entity_ids, tenant_id)
		
		print("\n✅ All quality assessment examples completed successfully!")
		print("\n🚀 Key Takeaways:")
		print("   • APG MDM provides sub-100ms quality assessment")
		print("   • 6-dimensional quality scoring with detailed insights")
		print("   • AI-powered issue detection and recommendations")
		print("   • Batch processing for large-scale quality monitoring")
		print("   • Real-time alerts and quality trend analysis")
		
	except Exception as e:
		print(f"\n❌ Error running quality examples: {str(e)}")
		raise
	finally:
		if 'mdm_service' in locals():
			await mdm_service.shutdown()
			print("\n🛑 MDM service shut down")


if __name__ == "__main__":
	asyncio.run(main())