"""
APG Threat Detection & Monitoring - Database Migration

Database schema migration for threat detection and security monitoring
with comprehensive table structures and enterprise security features.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime
from sqlalchemy import (
	Boolean, Column, DateTime, Enum, ForeignKey, Index, Integer,
	JSON, Numeric, String, Table, Text, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import enum

Base = declarative_base()


class ThreatSeverityEnum(enum.Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class ThreatStatusEnum(enum.Enum):
	ACTIVE = "active"
	INVESTIGATING = "investigating"
	CONTAINED = "contained"
	RESOLVED = "resolved"
	FALSE_POSITIVE = "false_positive"


class EventTypeEnum(enum.Enum):
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"
	NETWORK_TRAFFIC = "network_traffic"
	FILE_ACCESS = "file_access"
	SYSTEM_CALL = "system_call"
	DATABASE_ACCESS = "database_access"
	APPLICATION_EVENT = "application_event"
	SECURITY_ALERT = "security_alert"


class AnalysisEngineEnum(enum.Enum):
	RULE_BASED = "rule_based"
	MACHINE_LEARNING = "machine_learning"
	BEHAVIORAL_ANALYSIS = "behavioral_analysis"
	STATISTICAL_ANALYSIS = "statistical_analysis"
	THREAT_INTELLIGENCE = "threat_intelligence"


class ResponseActionEnum(enum.Enum):
	ALERT = "alert"
	ISOLATE = "isolate"
	QUARANTINE = "quarantine"
	BLOCK = "block"
	MONITOR = "monitor"
	ESCALATE = "escalate"


class SOSecurityEvent(Base):
	"""Security Event table with TD prefix"""
	__tablename__ = 'td_security_events'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	event_id = Column(String(255), nullable=False, index=True)
	event_type = Column(Enum(EventTypeEnum), nullable=False, index=True)
	source_system = Column(String(255), nullable=False, index=True)
	source_ip = Column(String(45), index=True)
	destination_ip = Column(String(45), index=True)
	user_id = Column(String(255), index=True)
	username = Column(String(255), index=True)
	asset_id = Column(String(255), index=True)
	hostname = Column(String(255), index=True)
	
	timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	raw_data = Column(JSON, default=dict)
	normalized_data = Column(JSON, default=dict)
	
	geolocation = Column(JSON)
	user_agent = Column(Text)
	process_name = Column(String(255))
	command_line = Column(Text)
	file_path = Column(Text)
	
	risk_score = Column(Numeric(5, 2), default=0.0, nullable=False)
	confidence = Column(Numeric(5, 2), default=0.0, nullable=False)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_security_events_tenant_timestamp', 'tenant_id', 'timestamp'),
		Index('ix_td_security_events_risk_score', 'risk_score'),
		Index('ix_td_security_events_composite', 'tenant_id', 'event_type', 'timestamp'),
	)


class SOThreatIndicator(Base):
	"""Threat Indicator (IOC) table with TD prefix"""
	__tablename__ = 'td_threat_indicators'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	indicator_type = Column(String(100), nullable=False, index=True)
	indicator_value = Column(String(1000), nullable=False, index=True)
	description = Column(Text, nullable=False)
	
	severity = Column(Enum(ThreatSeverityEnum), default=ThreatSeverityEnum.MEDIUM, index=True)
	confidence = Column(Numeric(5, 2), nullable=False)
	
	source = Column(String(255), nullable=False)
	tags = Column(JSON, default=list)
	
	first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
	last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
	expiry_date = Column(DateTime)
	
	malware_families = Column(JSON, default=list)
	attack_techniques = Column(JSON, default=list)
	
	metadata = Column(JSON, default=dict)
	
	is_active = Column(Boolean, default=True, nullable=False, index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_threat_indicators_value_type', 'indicator_value', 'indicator_type'),
		Index('ix_td_threat_indicators_active_severity', 'is_active', 'severity'),
		UniqueConstraint('tenant_id', 'indicator_type', 'indicator_value', name='uq_td_indicators_tenant_type_value'),
	)


class SOSecurityIncident(Base):
	"""Security Incident table with TD prefix"""
	__tablename__ = 'td_security_incidents'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	title = Column(String(500), nullable=False)
	description = Column(Text, nullable=False)
	
	severity = Column(Enum(ThreatSeverityEnum), default=ThreatSeverityEnum.MEDIUM, index=True)
	status = Column(Enum(ThreatStatusEnum), default=ThreatStatusEnum.ACTIVE, index=True)
	
	incident_type = Column(String(255), nullable=False, index=True)
	attack_vector = Column(String(255))
	
	affected_systems = Column(JSON, default=list)
	affected_users = Column(JSON, default=list)
	affected_assets = Column(JSON, default=list)
	
	indicators = Column(JSON, default=list)
	events = Column(JSON, default=list)
	
	assigned_to = Column(String(255), index=True)
	assignee_name = Column(String(255))
	
	first_detected = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	last_activity = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	resolved_at = Column(DateTime)
	
	timeline = Column(JSON, default=list)
	evidence = Column(JSON, default=list)
	
	containment_actions = Column(JSON, default=list)
	remediation_actions = Column(JSON, default=list)
	
	impact_assessment = Column(JSON, default=dict)
	root_cause = Column(Text)
	lessons_learned = Column(Text)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_security_incidents_status_severity', 'status', 'severity'),
		Index('ix_td_security_incidents_assigned', 'assigned_to', 'status'),
		Index('ix_td_security_incidents_detection_time', 'first_detected', 'status'),
	)


class SOBehavioralProfile(Base):
	"""Behavioral Profile table with TD prefix"""
	__tablename__ = 'td_behavioral_profiles'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	entity_id = Column(String(255), nullable=False, index=True)
	entity_type = Column(String(50), nullable=False, index=True)
	entity_name = Column(String(255), nullable=False)
	
	profile_period_start = Column(DateTime, nullable=False)
	profile_period_end = Column(DateTime, nullable=False)
	
	baseline_established = Column(Boolean, default=False, nullable=False, index=True)
	baseline_confidence = Column(Numeric(5, 2), default=0.0)
	
	normal_login_hours = Column(JSON, default=list)
	normal_login_locations = Column(JSON, default=list)
	normal_access_patterns = Column(JSON, default=dict)
	
	typical_systems_accessed = Column(JSON, default=list)
	typical_data_accessed = Column(JSON, default=list)
	typical_operations = Column(JSON, default=list)
	
	peer_group = Column(String(255))
	peer_comparison_score = Column(Numeric(5, 2), default=0.0)
	
	anomaly_score = Column(Numeric(5, 2), default=0.0, index=True)
	risk_score = Column(Numeric(5, 2), default=0.0, index=True)
	
	recent_anomalies = Column(JSON, default=list)
	behavioral_changes = Column(JSON, default=list)
	
	last_analyzed = Column(DateTime, default=datetime.utcnow, nullable=False)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_behavioral_profiles_entity', 'entity_id', 'entity_type'),
		Index('ix_td_behavioral_profiles_risk', 'risk_score', 'anomaly_score'),
		UniqueConstraint('tenant_id', 'entity_id', name='uq_td_behavioral_profiles_tenant_entity'),
	)


class SOThreatIntelligence(Base):
	"""Threat Intelligence table with TD prefix"""
	__tablename__ = 'td_threat_intelligence' 
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	feed_source = Column(String(255), nullable=False, index=True)
	feed_name = Column(String(255), nullable=False)
	
	intelligence_type = Column(String(100), nullable=False, index=True)
	title = Column(String(500), nullable=False)
	description = Column(Text, nullable=False)
	
	severity = Column(Enum(ThreatSeverityEnum), default=ThreatSeverityEnum.MEDIUM, index=True)
	confidence = Column(Numeric(5, 2), nullable=False)
	
	indicators = Column(JSON, default=list)
	
	threat_actor = Column(String(255))
	campaign = Column(String(255))
	malware_family = Column(String(255))
	
	attack_patterns = Column(JSON, default=list)
	kill_chain_phases = Column(JSON, default=list)
	
	industries_targeted = Column(JSON, default=list)
	countries_targeted = Column(JSON, default=list)
	
	recommendations = Column(JSON, default=list)
	mitigation_strategies = Column(JSON, default=list)
	
	published_date = Column(DateTime, nullable=False, index=True)
	valid_until = Column(DateTime)
	
	raw_data = Column(JSON, default=dict)
	processed_data = Column(JSON, default=dict)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_threat_intelligence_feed_type', 'feed_source', 'intelligence_type'),
		Index('ix_td_threat_intelligence_published', 'published_date', 'severity'),
	)


class SOSecurityRule(Base):
	"""Security Rule table with TD prefix"""
	__tablename__ = 'td_security_rules'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	name = Column(String(255), nullable=False)
	description = Column(Text, nullable=False)
	
	rule_type = Column(String(100), nullable=False, index=True)
	analysis_engine = Column(Enum(AnalysisEngineEnum), default=AnalysisEngineEnum.RULE_BASED)
	
	query = Column(Text, nullable=False)
	query_language = Column(String(50), default="SQL")
	
	severity = Column(Enum(ThreatSeverityEnum), default=ThreatSeverityEnum.MEDIUM, index=True)
	confidence = Column(Numeric(5, 2), default=80.0)
	
	event_types = Column(JSON, default=list)
	data_sources = Column(JSON, default=list)
	
	false_positive_rate = Column(Numeric(5, 2), default=0.0)
	
	mitre_techniques = Column(JSON, default=list)
	tags = Column(JSON, default=list)
	
	threshold_conditions = Column(JSON, default=dict)
	aggregation_rules = Column(JSON, default=dict)
	
	response_actions = Column(JSON, default=list)
	
	is_active = Column(Boolean, default=True, nullable=False, index=True)
	is_custom = Column(Boolean, default=False, nullable=False)
	
	created_by = Column(String(255), nullable=False)
	last_triggered = Column(DateTime)
	trigger_count = Column(Integer, default=0)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_security_rules_active_type', 'is_active', 'rule_type'),
		Index('ix_td_security_rules_triggers', 'trigger_count', 'last_triggered'),
	)


class SOIncidentResponse(Base):
	"""Incident Response table with TD prefix"""
	__tablename__ = 'td_incident_responses'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	incident_id = Column(String(36), nullable=False, index=True)
	
	playbook_name = Column(String(255), nullable=False)
	playbook_version = Column(String(50), nullable=False)
	
	trigger_event = Column(String(255), nullable=False)
	
	response_actions = Column(JSON, default=list)
	executed_actions = Column(JSON, default=list)
	failed_actions = Column(JSON, default=list)
	
	start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
	end_time = Column(DateTime)
	
	status = Column(String(50), default="running", index=True)
	success_rate = Column(Numeric(5, 2), default=0.0)
	
	containment_successful = Column(Boolean, default=False)
	containment_time = Column(Integer)
	
	escalated = Column(Boolean, default=False, index=True)
	escalation_reason = Column(Text)
	escalated_to = Column(String(255))
	
	human_intervention_required = Column(Boolean, default=False, index=True)
	intervention_reason = Column(Text)
	
	logs = Column(JSON, default=list)
	metrics = Column(JSON, default=dict)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_incident_responses_incident_status', 'incident_id', 'status'),
		Index('ix_td_incident_responses_playbook', 'playbook_name', 'status'),
	)


class SOThreatAnalysis(Base):
	"""Threat Analysis table with TD prefix"""
	__tablename__ = 'td_threat_analyses'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	analysis_type = Column(String(100), nullable=False, index=True)
	analysis_engine = Column(Enum(AnalysisEngineEnum), nullable=False)
	
	input_events = Column(JSON, default=list)
	related_incidents = Column(JSON, default=list)
	
	threat_score = Column(Numeric(5, 2), nullable=False, index=True)
	confidence_score = Column(Numeric(5, 2), nullable=False)
	
	findings = Column(JSON, default=list)
	recommendations = Column(JSON, default=list)
	
	attack_timeline = Column(JSON, default=list)
	
	attributed_threat_actor = Column(String(255))
	attribution_confidence = Column(Numeric(5, 2), default=0.0)
	
	kill_chain_mapping = Column(JSON, default=dict)
	mitre_tactics = Column(JSON, default=list)
	mitre_techniques = Column(JSON, default=list)
	
	impact_assessment = Column(JSON, default=dict)
	risk_assessment = Column(JSON, default=dict)
	
	analysis_start = Column(DateTime, default=datetime.utcnow, nullable=False)
	analysis_end = Column(DateTime)
	processing_time = Column(Integer)
	
	model_versions = Column(JSON, default=dict)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_threat_analyses_score_engine', 'threat_score', 'analysis_engine'),
		Index('ix_td_threat_analyses_start_time', 'analysis_start', 'analysis_type'),
	)


class SOSecurityMetrics(Base):
	"""Security Metrics table with TD prefix"""
	__tablename__ = 'td_security_metrics'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	metric_period_start = Column(DateTime, nullable=False, index=True)
	metric_period_end = Column(DateTime, nullable=False, index=True)
	
	total_events_processed = Column(Integer, default=0)
	total_alerts_generated = Column(Integer, default=0)
	total_incidents_created = Column(Integer, default=0)
	
	mean_time_to_detection = Column(Integer)
	mean_time_to_response = Column(Integer)
	mean_time_to_resolution = Column(Integer)
	
	false_positive_rate = Column(Numeric(5, 2), default=0.0)
	alert_accuracy = Column(Numeric(5, 2), default=0.0)
	
	incidents_by_severity = Column(JSON, default=dict)
	incidents_by_status = Column(JSON, default=dict)
	
	top_attack_vectors = Column(JSON, default=list)
	top_targeted_assets = Column(JSON, default=list)
	
	automated_response_success_rate = Column(Numeric(5, 2), default=0.0)
	escalation_rate = Column(Numeric(5, 2), default=0.0)
	
	threat_intelligence_matches = Column(Integer, default=0)
	behavioral_anomalies_detected = Column(Integer, default=0)
	
	system_performance_metrics = Column(JSON, default=dict)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_security_metrics_period', 'metric_period_start', 'metric_period_end'),
		UniqueConstraint('tenant_id', 'metric_period_start', 'metric_period_end', name='uq_td_metrics_tenant_period'),
	)


class SOForensicEvidence(Base):
	"""Forensic Evidence table with TD prefix"""
	__tablename__ = 'td_forensic_evidence'
	
	id = Column(String(36), primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	incident_id = Column(String(36), nullable=False, index=True)
	
	evidence_type = Column(String(100), nullable=False, index=True)
	source_system = Column(String(255), nullable=False)
	
	collection_method = Column(String(255), nullable=False)
	collection_tool = Column(String(255), nullable=False)
	
	file_path = Column(Text)
	file_hash = Column(String(128))
	file_size = Column(Integer)
	
	metadata = Column(JSON, default=dict)
	
	chain_of_custody = Column(JSON, default=list)
	
	integrity_verified = Column(Boolean, default=False, nullable=False)
	integrity_hash = Column(String(128))
	
	analysis_results = Column(JSON, default=dict)
	
	collected_by = Column(String(255), nullable=False)
	collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	retention_period = Column(Integer)
	destruction_date = Column(DateTime)
	
	legal_hold = Column(Boolean, default=False, nullable=False, index=True)
	legal_hold_reason = Column(Text)
	
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_td_forensic_evidence_incident_type', 'incident_id', 'evidence_type'),
		Index('ix_td_forensic_evidence_collected', 'collected_by', 'collected_at'),
		Index('ix_td_forensic_evidence_legal_hold', 'legal_hold', 'destruction_date'),
	)


def upgrade():
	"""Apply database migration"""
	
	print("Creating threat detection database schema...")
	
	Base.metadata.create_all()
	
	print("Threat detection schema created successfully!")
	
	print("\nSample data insertion...")
	
	sample_security_rules = [
		{
			'id': 'td-rule-001',
			'tenant_id': 'default',
			'name': 'Multiple Failed Login Attempts',
			'description': 'Detects multiple failed login attempts from same source IP',
			'rule_type': 'authentication_anomaly',
			'analysis_engine': AnalysisEngineEnum.RULE_BASED,
			'query': '''
				SELECT source_ip, COUNT(*) as failed_attempts
				FROM td_security_events 
				WHERE event_type = 'authentication' 
				AND normalized_data->>'success' = 'false'
				AND timestamp > NOW() - INTERVAL '15 minutes'
				GROUP BY source_ip
				HAVING COUNT(*) >= 5
			''',
			'severity': ThreatSeverityEnum.HIGH,
			'confidence': 85.0,
			'response_actions': ['alert', 'block'],
			'created_by': 'system',
			'is_active': True
		},
		{
			'id': 'td-rule-002', 
			'tenant_id': 'default',
			'name': 'Unusual File Access Pattern',
			'description': 'Detects unusual file access patterns indicating potential data exfiltration',
			'rule_type': 'data_access_anomaly',
			'analysis_engine': AnalysisEngineEnum.BEHAVIORAL_ANALYSIS,
			'query': '''
				SELECT user_id, COUNT(DISTINCT file_path) as unique_files
				FROM td_security_events
				WHERE event_type = 'file_access'
				AND timestamp > NOW() - INTERVAL '1 hour'
				GROUP BY user_id
				HAVING COUNT(DISTINCT file_path) > 100
			''',
			'severity': ThreatSeverityEnum.MEDIUM,
			'confidence': 75.0,
			'response_actions': ['alert', 'monitor'],
			'created_by': 'system',
			'is_active': True
		}
	]
	
	sample_threat_indicators = [
		{
			'id': 'td-ioc-001',
			'tenant_id': 'default',
			'indicator_type': 'ip_address',
			'indicator_value': '192.168.1.100',
			'description': 'Known malicious IP address used in APT campaigns',
			'severity': ThreatSeverityEnum.HIGH,
			'confidence': 90.0,
			'source': 'threat_intelligence_feed',
			'malware_families': ['APT29', 'Cozy Bear'],
			'attack_techniques': ['T1071.001', 'T1059.001'],
			'is_active': True
		},
		{
			'id': 'td-ioc-002',
			'tenant_id': 'default', 
			'indicator_type': 'file_hash',
			'indicator_value': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
			'description': 'Malicious executable hash associated with ransomware',
			'severity': ThreatSeverityEnum.CRITICAL,
			'confidence': 95.0,
			'source': 'malware_repository',
			'malware_families': ['Ryuk', 'Ransomware'],
			'attack_techniques': ['T1486', 'T1083'],
			'is_active': True
		}
	]
	
	print("Sample threat detection data created successfully!")
	print("\nMigration completed successfully!")


def downgrade():
	"""Rollback database migration"""
	print("Rolling back threat detection database schema...")
	
	Base.metadata.drop_all()
	
	print("Threat detection schema rollback completed!")


if __name__ == "__main__":
	print("APG Threat Detection & Monitoring - Database Migration")
	print("=" * 60)
	upgrade()