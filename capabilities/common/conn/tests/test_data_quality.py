"""
Tests for APG Connection Management Data Quality functionality
Comprehensive testing of data quality assessment, validation, and monitoring

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from ..data_quality import (
	DataProfiler, DataQualityValidator, DataQualityAssessment, DataQualityMonitor,
	DataQualityRule, DataQualityDimension, IssueSeverity, IssueType,
	FieldProfile, DataQualityMetrics, DataQualityLevel, DataQualityIssue,
	assess_connection_data_quality, get_data_quality_report
)


@pytest.fixture
def sample_data():
	"""Sample data for testing"""
	return [
		{
			'id': 1,
			'name': 'John Doe',
			'email': 'john.doe@example.com',
			'age': 25,
			'created_at': '2025-01-08T10:00:00Z',
			'salary': 50000.0,
			'department': 'Engineering'
		},
		{
			'id': 2,
			'name': 'Jane Smith',
			'email': 'jane.smith@company.com',
			'age': 30,
			'created_at': '2025-01-07T15:30:00Z',
			'salary': 65000.0,
			'department': 'Marketing'
		},
		{
			'id': 3,
			'name': None,  # Missing value
			'email': 'invalid-email',  # Invalid format
			'age': 150,  # Outlier
			'created_at': '2024-01-01T00:00:00Z',  # Stale data
			'salary': -5000.0,  # Invalid value
			'department': 'Sales'
		},
		{
			'id': 2,  # Duplicate ID
			'name': 'Bob Johnson',
			'email': 'bob@example.com',
			'age': 35,
			'created_at': '2025-01-08T09:00:00Z',
			'salary': 70000.0,
			'department': None  # Missing value
		}
	]


@pytest.fixture
def data_profiler():
	"""Data profiler instance"""
	return DataProfiler()


@pytest.fixture
def data_validator():
	"""Data validator instance"""
	return DataQualityValidator()


@pytest.fixture
def quality_assessment():
	"""Data quality assessment instance"""
	return DataQualityAssessment()


@pytest.fixture
def quality_monitor():
	"""Data quality monitor instance"""
	return DataQualityMonitor()


class TestDataProfiler:
	"""Test data profiling functionality"""

	def test_profile_dataset_with_dict_list(self, data_profiler, sample_data):
		"""Test profiling list of dictionaries"""
		profiles = data_profiler.profile_dataset(sample_data, "test_dataset")

		assert len(profiles) == 7  # All fields
		assert 'id' in profiles
		assert 'name' in profiles
		assert 'email' in profiles

		# Check ID field profile
		id_profile = profiles['id']
		assert id_profile.field_name == 'id'
		assert id_profile.data_type == 'int'
		assert id_profile.total_count == 4
		assert id_profile.null_count == 0
		assert id_profile.unique_count == 3  # 1, 2, 3 (2 appears twice)
		assert id_profile.duplicate_count == 1
		assert id_profile.completeness_rate == 1.0
		assert id_profile.uniqueness_rate == 0.75  # 3/4

	def test_profile_numeric_field(self, data_profiler, sample_data):
		"""Test profiling numeric fields"""
		profiles = data_profiler.profile_dataset(sample_data)

		age_profile = profiles['age']
		assert age_profile.data_type == 'int'
		assert age_profile.min_value == 25
		assert age_profile.max_value == 150
		assert age_profile.mean_value is not None
		assert len(age_profile.outliers) > 0  # Should detect age 150 as outlier

	def test_profile_string_field(self, data_profiler, sample_data):
		"""Test profiling string fields"""
		profiles = data_profiler.profile_dataset(sample_data)

		email_profile = profiles['email']
		assert email_profile.data_type == 'string'
		assert 'email' in email_profile.patterns  # Should detect email pattern

	def test_profile_with_missing_values(self, data_profiler, sample_data):
		"""Test profiling fields with missing values"""
		profiles = data_profiler.profile_dataset(sample_data)

		name_profile = profiles['name']
		assert name_profile.null_count == 1
		assert name_profile.completeness_rate == 0.75  # 3/4 non-null

		department_profile = profiles['department']
		assert department_profile.null_count == 1
		assert department_profile.completeness_rate == 0.75

	def test_determine_data_type(self, data_profiler):
		"""Test data type detection"""
		assert data_profiler._determine_data_type([1, 2, 3]) == 'int'
		assert data_profiler._determine_data_type([1.1, 2.2, 3.3]) == 'float'
		assert data_profiler._determine_data_type(['a', 'b', 'c']) == 'string'
		assert data_profiler._determine_data_type([True, False, True]) == 'bool'

	def test_is_numeric(self, data_profiler):
		"""Test numeric value detection"""
		assert data_profiler._is_numeric(42) == True
		assert data_profiler._is_numeric('42') == True
		assert data_profiler._is_numeric('42.5') == True
		assert data_profiler._is_numeric('abc') == False
		assert data_profiler._is_numeric(None) == False

	def test_outlier_detection(self, data_profiler):
		"""Test outlier detection"""
		values = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
		outliers = data_profiler._detect_outliers(values, 'int')
		assert 100 in outliers

	def test_pattern_analysis(self, data_profiler):
		"""Test pattern analysis"""
		email_values = ['test@example.com', 'user@domain.org', 'admin@company.net']
		patterns = data_profiler._analyze_patterns(email_values, 'string')
		assert 'email' in patterns

		phone_values = ['1234567890', '9876543210', '5555555555']
		patterns = data_profiler._analyze_patterns(phone_values, 'string')
		assert 'phone' in patterns


class TestDataQualityValidator:
	"""Test data quality validation"""

	def test_default_rules_loaded(self, data_validator):
		"""Test that default rules are loaded"""
		assert len(data_validator.rules) > 0
		assert 'completeness_check' in data_validator.rules
		assert 'email_format' in data_validator.rules
		assert 'duplicate_check' in data_validator.rules

	def test_add_custom_rule(self, data_validator):
		"""Test adding custom validation rule"""
		custom_rule = DataQualityRule(
			rule_id='test_rule',
			name='Test Rule',
			description='Test validation rule',
			dimension=DataQualityDimension.VALIDITY,
			severity=IssueSeverity.MEDIUM,
			field_names=['test_field'],
			rule_type='regex',
			parameters={'pattern': r'^\d+$'}
		)

		data_validator.add_rule(custom_rule)
		assert 'test_rule' in data_validator.rules
		assert data_validator.rules['test_rule'].name == 'Test Rule'

	def test_remove_rule(self, data_validator):
		"""Test removing validation rule"""
		initial_count = len(data_validator.rules)
		data_validator.remove_rule('completeness_check')
		assert len(data_validator.rules) == initial_count - 1
		assert 'completeness_check' not in data_validator.rules

	def test_validate_completeness(self, data_validator, sample_data):
		"""Test completeness validation"""
		issues = data_validator.validate_data(sample_data, ['completeness_check'])

		# Should find missing values in name and department fields
		missing_issues = [i for i in issues if i.issue_type == IssueType.MISSING_VALUE]
		assert len(missing_issues) > 0

		# Check specific missing values
		name_issues = [i for i in missing_issues if i.field_name == 'name']
		assert len(name_issues) > 0

	def test_validate_email_format(self, data_validator, sample_data):
		"""Test email format validation"""
		issues = data_validator.validate_data(sample_data, ['email_format'])

		# Should find invalid email format
		format_issues = [i for i in issues if i.issue_type == IssueType.INVALID_FORMAT]
		assert len(format_issues) > 0
		assert any('invalid-email' in str(i.actual_value) for i in format_issues)

	def test_validate_uniqueness(self, data_validator, sample_data):
		"""Test uniqueness validation"""
		issues = data_validator.validate_data(sample_data, ['duplicate_check'])

		# Should find duplicate ID values
		duplicate_issues = [i for i in issues if i.issue_type == IssueType.DUPLICATE]
		assert len(duplicate_issues) > 0
		assert any(str(i.actual_value) == '2' for i in duplicate_issues)

	def test_validate_timeliness(self, data_validator, sample_data):
		"""Test timeliness validation"""
		issues = data_validator.validate_data(sample_data, ['data_freshness'])

		# Should find stale data (2024 timestamp)
		stale_issues = [i for i in issues if i.issue_type == IssueType.STALE_DATA]
		assert len(stale_issues) > 0

	def test_validate_all_rules(self, data_validator, sample_data):
		"""Test validation with all rules"""
		issues = data_validator.validate_data(sample_data)

		assert len(issues) > 0

		# Check that different types of issues are found
		issue_types = set(issue.issue_type for issue in issues)
		assert IssueType.MISSING_VALUE in issue_types
		assert IssueType.INVALID_FORMAT in issue_types
		assert IssueType.DUPLICATE in issue_types


class TestDataQualityAssessment:
	"""Test comprehensive data quality assessment"""

	@pytest.mark.asyncio
	async def test_assess_data_quality(self, quality_assessment, sample_data):
		"""Test complete data quality assessment"""
		metrics = await quality_assessment.assess_data_quality(sample_data, "test_dataset")

		assert isinstance(metrics, DataQualityMetrics)
		assert metrics.total_records == 4
		assert metrics.valid_records <= metrics.total_records
		assert 0 <= metrics.overall_score <= 100
		assert isinstance(metrics.quality_level, DataQualityLevel)
		assert len(metrics.issues) > 0
		assert len(metrics.profiling_stats) > 0

	@pytest.mark.asyncio
	async def test_quality_scores_calculation(self, quality_assessment, sample_data):
		"""Test quality scores calculation"""
		metrics = await quality_assessment.assess_data_quality(sample_data)

		# All scores should be percentages (0-100)
		assert 0 <= metrics.completeness_score <= 100
		assert 0 <= metrics.accuracy_score <= 100
		assert 0 <= metrics.consistency_score <= 100
		assert 0 <= metrics.validity_score <= 100
		assert 0 <= metrics.uniqueness_score <= 100
		assert 0 <= metrics.timeliness_score <= 100
		assert 0 <= metrics.integrity_score <= 100

	@pytest.mark.asyncio
	async def test_quality_level_determination(self, quality_assessment):
		"""Test quality level determination based on scores"""
		# Mock quality scores
		excellent_scores = {dim: 95.0 for dim in ['completeness', 'accuracy', 'consistency', 'validity', 'uniqueness', 'timeliness', 'integrity']}
		excellent_level = quality_assessment._determine_quality_level(quality_assessment._calculate_overall_score(excellent_scores))
		assert excellent_level == DataQualityLevel.EXCELLENT

		poor_scores = {dim: 30.0 for dim in ['completeness', 'accuracy', 'consistency', 'validity', 'uniqueness', 'timeliness', 'integrity']}
		poor_level = quality_assessment._determine_quality_level(quality_assessment._calculate_overall_score(poor_scores))
		assert poor_level == DataQualityLevel.POOR

	@pytest.mark.asyncio
	async def test_custom_rules_in_assessment(self, quality_assessment, sample_data):
		"""Test assessment with custom rules"""
		custom_rule = DataQualityRule(
			rule_id='salary_range',
			name='Salary Range Check',
			description='Check salary is in valid range',
			dimension=DataQualityDimension.VALIDITY,
			severity=IssueSeverity.HIGH,
			field_names=['salary'],
			rule_type='custom',
			parameters={'min_value': 0, 'max_value': 1000000}
		)

		metrics = await quality_assessment.assess_data_quality(sample_data, custom_rules=[custom_rule])

		assert 'salary_range' in quality_assessment.validator.rules
		assert len(metrics.issues) > 0

	@pytest.mark.asyncio
	async def test_profile_to_dict_conversion(self, quality_assessment, sample_data):
		"""Test conversion of field profiles to dictionaries"""
		metrics = await quality_assessment.assess_data_quality(sample_data)

		profiles = metrics.profiling_stats['profiles']
		assert isinstance(profiles, dict)
		assert len(profiles) > 0

		# Check profile structure
		for profile_dict in profiles.values():
			assert 'data_type' in profile_dict
			assert 'completeness_rate' in profile_dict
			assert 'uniqueness_rate' in profile_dict
			assert 'null_count' in profile_dict


class TestDataQualityMonitor:
	"""Test continuous data quality monitoring"""

	@pytest.mark.asyncio
	async def test_monitor_data_quality(self, quality_monitor, sample_data):
		"""Test data quality monitoring"""
		metrics = await quality_monitor.monitor_data_quality('test_conn_1', sample_data)

		assert isinstance(metrics, DataQualityMetrics)
		assert len(quality_monitor.quality_history) == 1
		assert quality_monitor.quality_history[0] == metrics

	@pytest.mark.asyncio
	async def test_quality_alerts(self, quality_monitor, sample_data):
		"""Test quality alert generation"""
		# Set low thresholds to trigger alerts
		quality_monitor.alert_thresholds = {
			'overall_score': 95,
			'completeness_score': 95,
			'validity_score': 95,
			'critical_issues': 0
		}

		alert_callback = AsyncMock()

		metrics = await quality_monitor.monitor_data_quality('test_conn_1', sample_data, alert_callback)

		# Should trigger alerts due to low thresholds
		if metrics.overall_score < 95:
			alert_callback.assert_called_once()

	def test_quality_trends(self, quality_monitor, sample_data):
		"""Test quality trends calculation"""
		# Add some mock history
		now = datetime.now(timezone.utc)
		for i in range(5):
			mock_metrics = DataQualityMetrics(
				total_records=4,
				valid_records=3,
				completeness_score=80.0 + i,
				accuracy_score=85.0,
				consistency_score=90.0,
				validity_score=75.0,
				uniqueness_score=70.0,
				timeliness_score=65.0,
				integrity_score=95.0,
				overall_score=80.0 + i,
				quality_level=DataQualityLevel.GOOD,
				assessment_timestamp=now - timedelta(hours=i)
			)
			quality_monitor.quality_history.append(mock_metrics)

		trends = quality_monitor.get_quality_trends(lookback_hours=6)

		assert 'overall_score' in trends
		assert 'completeness_trend' in trends
		assert 'issues_trend' in trends
		assert trends['assessments_count'] == 5
		assert trends['overall_score']['trend'] in ['improving', 'declining', 'stable']

	def test_history_size_limit(self, quality_monitor):
		"""Test that quality history is limited to 100 entries"""
		# Add 150 mock metrics
		now = datetime.now(timezone.utc)
		for i in range(150):
			mock_metrics = DataQualityMetrics(
				total_records=10,
				valid_records=9,
				completeness_score=80.0,
				accuracy_score=85.0,
				consistency_score=90.0,
				validity_score=75.0,
				uniqueness_score=70.0,
				timeliness_score=65.0,
				integrity_score=95.0,
				overall_score=80.0,
				quality_level=DataQualityLevel.GOOD,
				assessment_timestamp=now - timedelta(minutes=i)
			)
			quality_monitor.quality_history.append(mock_metrics)

		assert len(quality_monitor.quality_history) == 100


class TestConvenienceFunctions:
	"""Test convenience functions"""

	@pytest.mark.asyncio
	async def test_assess_connection_data_quality(self, sample_data):
		"""Test convenience function for connection data quality assessment"""
		metrics = await assess_connection_data_quality('test_connection', sample_data)

		assert isinstance(metrics, DataQualityMetrics)
		assert metrics.total_records == 4
		assert 0 <= metrics.overall_score <= 100

	@pytest.mark.asyncio
	async def test_get_data_quality_report(self):
		"""Test getting data quality report"""
		report = await get_data_quality_report('test_connection')

		assert isinstance(report, dict)
		assert 'connection_id' in report
		assert 'generated_at' in report
		assert 'quality_trends' in report
		assert 'alert_thresholds' in report
		assert report['connection_id'] == 'test_connection'


class TestDataQualityRules:
	"""Test data quality rule management"""

	def test_rule_creation(self):
		"""Test creating data quality rules"""
		rule = DataQualityRule(
			rule_id='test_rule',
			name='Test Rule',
			description='A test validation rule',
			dimension=DataQualityDimension.COMPLETENESS,
			severity=IssueSeverity.HIGH,
			field_names=['field1', 'field2'],
			rule_type='completeness'
		)

		assert rule.rule_id == 'test_rule'
		assert rule.name == 'Test Rule'
		assert rule.dimension == DataQualityDimension.COMPLETENESS
		assert rule.severity == IssueSeverity.HIGH
		assert rule.is_active == True
		assert isinstance(rule.created_at, datetime)

	def test_issue_creation(self):
		"""Test creating data quality issues"""
		issue = DataQualityIssue(
			issue_type=IssueType.MISSING_VALUE,
			severity=IssueSeverity.HIGH,
			dimension=DataQualityDimension.COMPLETENESS,
			description='Missing required field',
			field_name='required_field',
			record_id='123',
			expected_value='non-null',
			actual_value=None
		)

		assert issue.issue_type == IssueType.MISSING_VALUE
		assert issue.severity == IssueSeverity.HIGH
		assert issue.field_name == 'required_field'
		assert isinstance(issue.detected_at, datetime)


class TestEdgeCases:
	"""Test edge cases and error conditions"""

	@pytest.mark.asyncio
	async def test_empty_dataset(self, quality_assessment):
		"""Test assessment with empty dataset"""
		with pytest.raises(Exception):  # Should raise APGError
			await quality_assessment.assess_data_quality([])

	@pytest.mark.asyncio
	async def test_invalid_data_format(self, quality_assessment):
		"""Test assessment with invalid data format"""
		with pytest.raises(Exception):  # Should raise APGError
			await quality_assessment.assess_data_quality("invalid_data_format")

	def test_profile_empty_values(self, data_profiler):
		"""Test profiling with empty values"""
		empty_data = [{'field1': None}, {'field1': ''}, {'field1': 'null'}]
		profiles = data_profiler.profile_dataset(empty_data)

		field_profile = profiles['field1']
		assert field_profile.null_count == 3
		assert field_profile.completeness_rate == 0.0

	def test_validator_with_invalid_regex(self, data_validator):
		"""Test validator with invalid regex pattern"""
		invalid_rule = DataQualityRule(
			rule_id='invalid_regex',
			name='Invalid Regex Rule',
			description='Rule with invalid regex',
			dimension=DataQualityDimension.VALIDITY,
			severity=IssueSeverity.MEDIUM,
			field_names=['test_field'],
			rule_type='regex',
			parameters={'pattern': '[invalid_regex'}  # Invalid regex
		)

		data_validator.add_rule(invalid_rule)

		# Should not raise exception but return no issues
		issues = data_validator.validate_data([{'test_field': 'test'}], ['invalid_regex'])
		assert len(issues) == 0


# Integration test
class TestDataQualityIntegration:
	"""Integration tests for data quality system"""

	@pytest.mark.asyncio
	async def test_end_to_end_quality_assessment(self, sample_data):
		"""Test complete end-to-end data quality assessment workflow"""
		# 1. Profile the data
		profiler = DataProfiler()
		profiles = profiler.profile_dataset(sample_data, "integration_test")
		assert len(profiles) > 0

		# 2. Validate the data
		validator = DataQualityValidator()
		issues = validator.validate_data(sample_data)
		assert len(issues) > 0

		# 3. Perform comprehensive assessment
		assessment = DataQualityAssessment()
		metrics = await assessment.assess_data_quality(sample_data)
		assert isinstance(metrics, DataQualityMetrics)
		assert metrics.overall_score > 0

		# 4. Monitor quality
		monitor = DataQualityMonitor()
		monitored_metrics = await monitor.monitor_data_quality('test_integration', sample_data)
		assert monitored_metrics.overall_score == metrics.overall_score

		# 5. Get trends
		trends = monitor.get_quality_trends()
		assert 'overall_score' in trends

		print(f"Integration test completed successfully:")
		print(f"- Total records: {metrics.total_records}")
		print(f"- Valid records: {metrics.valid_records}")
		print(f"- Overall quality score: {metrics.overall_score}%")
		print(f"- Quality level: {metrics.quality_level.value}")
		print(f"- Issues found: {len(metrics.issues)}")


if __name__ == '__main__':
	pytest.main([__file__, '-v'])