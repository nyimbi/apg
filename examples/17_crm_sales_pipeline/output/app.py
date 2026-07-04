"""
crm_sales - APG Python Application
==================================

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import importlib
import html
import json
import os
import sys
from flask import Flask as _FlaskApp, request as _flask_request, redirect as _flask_redirect, Response as _FlaskResponse
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote


MODULE_NAME = 'crm_sales'
MODULE_VERSION = '1.0.0'
MODULE_DESCRIPTION = None
LANDING_STYLE = 'default'
ENTITIES = [{'name': 'Lead', 'type': 'entity', 'properties': ['lead_id', 'first_name', 'last_name', 'email', 'phone', 'company', 'source', 'status', 'score', 'owner_id', 'created_at', 'notes'], 'fields': [{'name': 'lead_id', 'type': 'str', 'required': True}, {'name': 'first_name', 'type': 'str', 'required': True}, {'name': 'last_name', 'type': 'str', 'required': True}, {'name': 'email', 'type': 'str', 'required': True}, {'name': 'phone', 'type': 'str?', 'required': True}, {'name': 'company', 'type': 'str?', 'required': True}, {'name': 'source', 'type': 'str', 'required': False, 'default': '"website"'}, {'name': 'status', 'type': 'str', 'required': False, 'default': '"new"'}, {'name': 'score', 'type': 'float', 'required': False, 'default': '0.0'}, {'name': 'owner_id', 'type': 'str', 'required': True}, {'name': 'created_at', 'type': 'str', 'required': True}, {'name': 'notes', 'type': 'str?', 'required': True}], 'methods': []}, {'name': 'Opportunity', 'type': 'entity', 'properties': ['opportunity_id', 'lead_id', 'account_id', 'name', 'stage', 'amount', 'probability', 'expected_close', 'owner_id', 'competitor', 'loss_reason'], 'fields': [{'name': 'opportunity_id', 'type': 'str', 'required': True}, {'name': 'lead_id', 'type': 'str?', 'required': True}, {'name': 'account_id', 'type': 'str', 'required': True}, {'name': 'name', 'type': 'str', 'required': True}, {'name': 'stage', 'type': 'str', 'required': False, 'default': '"prospecting"'}, {'name': 'amount', 'type': 'float', 'required': True}, {'name': 'probability', 'type': 'float', 'required': False, 'default': '0.0'}, {'name': 'expected_close', 'type': 'str', 'required': True}, {'name': 'owner_id', 'type': 'str', 'required': True}, {'name': 'competitor', 'type': 'str?', 'required': True}, {'name': 'loss_reason', 'type': 'str?', 'required': True}], 'methods': []}, {'name': 'LeadManagement', 'type': 'capability', 'properties': [], 'fields': [], 'methods': []}, {'name': 'OpportunityPipeline', 'type': 'capability', 'properties': [], 'fields': [], 'methods': []}, {'name': 'CRMSalesPipeline', 'type': 'app', 'properties': [], 'fields': [], 'methods': []}]
ENTITY_NAMES = {entity["name"] for entity in ENTITIES}
RECORD_STORE: Dict[str, list[Dict[str, Any]]] = {entity["name"]: [] for entity in ENTITIES}
NEXT_RECORD_IDS: Dict[str, int] = {entity["name"]: 1 for entity in ENTITIES}
EVENT_LOG: list[Dict[str, Any]] = []
NEXT_EVENT_ID = 1
WORKFLOW_RUNS: Dict[str, Dict[str, Any]] = {}
NEXT_WORKFLOW_RUN_ID = 1
CIRCUIT_BREAKERS: Dict[str, Dict[str, Any]] = {}
APG_EVENT_SUBSCRIPTIONS: Dict[str, list[str]] = {}
APG_CONNECTOR_REGISTRY: list[Dict[str, Any]] = []
APG_ACTIVITY_LOG: Dict[str, list[Dict[str, Any]]] = {}
WORKFLOW_EVENT_JOURNAL: Dict[str, list[Dict[str, Any]]] = {}
WORKFLOW_SIGNALS: Dict[str, list[str]] = {}
TENANT_SCOPED_ENTITIES: set[str] = {
    e["name"] for e in ENTITIES
    if any(str(f.get("name")) == "tenant_id" for f in e.get("fields", []))
}
SEMANTIC_MODEL: Dict[str, Any] = {'format': 'apg.semantic-model.v1', 'ok': True, 'source_files': ['crm_sales.apg'], 'app': {'name': 'crm_sales', 'version': '1.0.0', 'description': None, 'entity_count': 5}, 'symbols': {'module.crm_sales': {'id': 'module.crm_sales', 'kind': 'module', 'name': 'crm_sales', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'table.Lead': {'id': 'table.Lead', 'kind': 'table', 'name': 'Lead', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.lead_id': {'id': 'field.Lead.lead_id', 'kind': 'field', 'name': 'Lead.lead_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.first_name': {'id': 'field.Lead.first_name', 'kind': 'field', 'name': 'Lead.first_name', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.last_name': {'id': 'field.Lead.last_name', 'kind': 'field', 'name': 'Lead.last_name', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.email': {'id': 'field.Lead.email', 'kind': 'field', 'name': 'Lead.email', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.phone': {'id': 'field.Lead.phone', 'kind': 'field', 'name': 'Lead.phone', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.company': {'id': 'field.Lead.company', 'kind': 'field', 'name': 'Lead.company', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.source': {'id': 'field.Lead.source', 'kind': 'field', 'name': 'Lead.source', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.status': {'id': 'field.Lead.status', 'kind': 'field', 'name': 'Lead.status', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.score': {'id': 'field.Lead.score', 'kind': 'field', 'name': 'Lead.score', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.owner_id': {'id': 'field.Lead.owner_id', 'kind': 'field', 'name': 'Lead.owner_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.created_at': {'id': 'field.Lead.created_at', 'kind': 'field', 'name': 'Lead.created_at', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Lead.notes': {'id': 'field.Lead.notes', 'kind': 'field', 'name': 'Lead.notes', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'table.Opportunity': {'id': 'table.Opportunity', 'kind': 'table', 'name': 'Opportunity', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.opportunity_id': {'id': 'field.Opportunity.opportunity_id', 'kind': 'field', 'name': 'Opportunity.opportunity_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.lead_id': {'id': 'field.Opportunity.lead_id', 'kind': 'field', 'name': 'Opportunity.lead_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.account_id': {'id': 'field.Opportunity.account_id', 'kind': 'field', 'name': 'Opportunity.account_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.name': {'id': 'field.Opportunity.name', 'kind': 'field', 'name': 'Opportunity.name', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.stage': {'id': 'field.Opportunity.stage', 'kind': 'field', 'name': 'Opportunity.stage', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.amount': {'id': 'field.Opportunity.amount', 'kind': 'field', 'name': 'Opportunity.amount', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.probability': {'id': 'field.Opportunity.probability', 'kind': 'field', 'name': 'Opportunity.probability', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.expected_close': {'id': 'field.Opportunity.expected_close', 'kind': 'field', 'name': 'Opportunity.expected_close', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.owner_id': {'id': 'field.Opportunity.owner_id', 'kind': 'field', 'name': 'Opportunity.owner_id', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.competitor': {'id': 'field.Opportunity.competitor', 'kind': 'field', 'name': 'Opportunity.competitor', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Opportunity.loss_reason': {'id': 'field.Opportunity.loss_reason', 'kind': 'field', 'name': 'Opportunity.loss_reason', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'capability.LeadManagement': {'id': 'capability.LeadManagement', 'kind': 'capability', 'name': 'LeadManagement', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'capability.OpportunityPipeline': {'id': 'capability.OpportunityPipeline', 'kind': 'capability', 'name': 'OpportunityPipeline', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'app.CRMSalesPipeline': {'id': 'app.CRMSalesPipeline', 'kind': 'app', 'name': 'CRMSalesPipeline', 'file': 'crm_sales.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}}, 'tables': {'Lead': {'name': 'Lead', 'fields': {'lead_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Lead', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'lead'}}, 'first_name': {'type': 'str', 'required': True, 'relationship': None}, 'last_name': {'type': 'str', 'required': True, 'relationship': None}, 'email': {'type': 'str', 'required': True, 'relationship': None}, 'phone': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'company': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'source': {'type': 'str', 'required': False, 'relationship': None}, 'status': {'type': 'str', 'required': False, 'relationship': None}, 'score': {'type': 'float', 'required': False, 'relationship': None}, 'owner_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Owner', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'owner'}}, 'created_at': {'type': 'str', 'required': True, 'relationship': None}, 'notes': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}}, 'lookup_paths': {'phone.id': {'chain': ['Lead.phone', 'str?.id'], 'valid': True}, 'company.id': {'chain': ['Lead.company', 'str?.id'], 'valid': True}, 'notes.id': {'chain': ['Lead.notes', 'str?.id'], 'valid': True}}}, 'Opportunity': {'name': 'Opportunity', 'fields': {'opportunity_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Opportunity', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'opportunity'}}, 'lead_id': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'account_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Account', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'account'}}, 'name': {'type': 'str', 'required': True, 'relationship': None}, 'stage': {'type': 'str', 'required': False, 'relationship': None}, 'amount': {'type': 'float', 'required': True, 'relationship': None}, 'probability': {'type': 'float', 'required': False, 'relationship': None}, 'expected_close': {'type': 'str', 'required': True, 'relationship': None}, 'owner_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Owner', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'owner'}}, 'competitor': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'loss_reason': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}}, 'lookup_paths': {'lead_id.id': {'chain': ['Opportunity.lead_id', 'str?.id'], 'valid': True}, 'competitor.id': {'chain': ['Opportunity.competitor', 'str?.id'], 'valid': True}, 'loss_reason.id': {'chain': ['Opportunity.loss_reason', 'str?.id'], 'valid': True}}}}, 'views': {}, 'flows': {}, 'operations': {}, 'rules': {'LeadManagement.duplicate_check': {'name': 'duplicate_check', 'when': 'email_exists == true', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'email_exists', 'op': '==', 'value': True}}, 'LeadManagement.score_threshold': {'name': 'score_threshold', 'when': 'score >= 75', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'score', 'op': '>=', 'value': 75}}, 'LeadManagement.stale_lead': {'name': 'stale_lead', 'when': 'days_since_contact > 30', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'days_since_contact', 'op': '>', 'value': 30}}, 'OpportunityPipeline.large_deal_review': {'name': 'large_deal_review', 'when': 'amount > 500000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'amount', 'op': '>', 'value': 500000}}, 'OpportunityPipeline.discount_limit': {'name': 'discount_limit', 'when': 'discount_pct > 30', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'discount_pct', 'op': '>', 'value': 30}}, 'OpportunityPipeline.stage_progression': {'name': 'stage_progression', 'when': 'next_stage missing', 'action': 'warn', 'when_ast': {'type': 'missing', 'field': 'next_stage', 'negated': False}}}, 'roles': {}, 'security': {}, 'agents': {}, 'llms': {}, 'capabilities': {'LeadManagement': {'name': 'LeadManagement', 'provides': ['lead_capture', 'lead_scoring', 'lead_conversion'], 'requires': [], 'configuration': {'tenant_id': 'default', 'lead_timeout_days': 30}, 'rules': [{'name': 'duplicate_check', 'when': 'email_exists == true', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'email_exists', 'op': '==', 'value': True}}, {'name': 'score_threshold', 'when': 'score >= 75', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'score', 'op': '>=', 'value': 75}}, {'name': 'stale_lead', 'when': 'days_since_contact > 30', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'days_since_contact', 'op': '>', 'value': 30}}], 'rule_engine': {}, 'ui': {'shell': 'python', 'routes': [{'name': 'Leads', 'path': '/crm/leads', 'component': 'LeadList', 'permission': 'crm:leads'}]}, 'theme': {'name': 'lead_theme', 'tokens': {'accent': '#00838F'}}, 'runtime': {}, 'erp_modules': [], 'components': {}, 'business_rules': [], 'approvals': {}, 'master_data': {}, 'i18n': {}, 'streaming': {}, 'screens': {}}, 'OpportunityPipeline': {'name': 'OpportunityPipeline', 'provides': ['deal_tracking', 'pipeline_analytics', 'forecast'], 'requires': ['lead_capture'], 'configuration': {'tenant_id': 'default', 'max_discount_pct': 30}, 'rules': [{'name': 'large_deal_review', 'when': 'amount > 500000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'amount', 'op': '>', 'value': 500000}}, {'name': 'discount_limit', 'when': 'discount_pct > 30', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'discount_pct', 'op': '>', 'value': 30}}, {'name': 'stage_progression', 'when': 'next_stage missing', 'action': 'warn', 'when_ast': {'type': 'missing', 'field': 'next_stage', 'negated': False}}], 'rule_engine': {}, 'ui': {'shell': 'python', 'routes': [{'name': 'Pipeline', 'path': '/crm/pipeline', 'component': 'PipelineView', 'permission': 'crm:pipeline'}, {'name': 'Forecast', 'path': '/crm/forecast', 'component': 'ForecastView', 'permission': 'crm:forecast'}, {'name': 'Analytics', 'path': '/crm/analytics', 'component': 'SalesAnalytics', 'permission': 'crm:analytics'}]}, 'theme': {'name': 'pipeline_theme', 'tokens': {'accent': '#E65100'}}, 'runtime': {}, 'erp_modules': [], 'components': {}, 'business_rules': [], 'approvals': {}, 'master_data': {}, 'i18n': {}, 'streaming': {}, 'screens': {'PipelineKanban': {'route': '/crm/kanban', 'title': 'Pipeline Board', 'layout': 'grid', 'contains': ['StageColumn', 'DealCard'], 'binds': ['opportunities.by_stage'], 'actions': ['move_stage', 'create_deal', 'filter']}}}}, 'composition': {'applications': {'CRMSalesPipeline': {'name': 'CRMSalesPipeline', 'description': 'CRM sales pipeline with lead and opportunity management', 'capabilities': ['LeadManagement', 'OpportunityPipeline'], 'agents': [], 'agent_teams': [], 'components': {'lead_desk': {'capability': 'lead_capture', 'route': '/crm/leads'}, 'deal_pipeline': {'capability': 'deal_tracking', 'route': '/crm/pipeline'}, 'forecast_console': {'capability': 'forecast', 'route': '/crm/forecast'}, 'analytics_hub': {'capability': 'pipeline_analytics', 'route': '/crm/analytics'}}, 'screens': {'SalesDashboard': {'route': '/crm', 'title': 'Sales Dashboard', 'layout': 'dashboard', 'contains': ['PipelineSummary', 'LeadFunnel', 'ForecastWidget', 'RecentActivity'], 'binds': ['pipeline.summary', 'leads.recent', 'forecast.current'], 'actions': ['create_lead', 'refresh', 'export']}}, 'routes': ['/crm', '/crm/leads', '/crm/pipeline', '/crm/forecast'], 'workflows': [], 'policies': {}, 'configuration': {}, 'theme': {'name': 'crm_theme', 'tokens': {'accent': '#FF6D00', 'border.radius': '4px'}}, 'runtime': {'target': 'python', 'deployment': 'container', 'streaming': {'processor': 'bytewax'}}, 'integrations': {}, 'deployments': {}}}, 'agent_teams': {}, 'capability_dependencies': {'LeadManagement': [], 'OpportunityPipeline': ['lead_capture']}}, 'contracts': {'LeadManagement': {'id': 'lead_management', 'provides': ['lead_capture', 'lead_scoring', 'lead_conversion'], 'configuration': {'tenant_id': 'default', 'lead_timeout_days': 30}, 'rules': [{'name': 'duplicate_check', 'when': 'email_exists == true', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'email_exists', 'op': '==', 'value': True}}, {'name': 'score_threshold', 'when': 'score >= 75', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'score', 'op': '>=', 'value': 75}}, {'name': 'stale_lead', 'when': 'days_since_contact > 30', 'action': 'warn', 'when_ast': {'type': 'compare', 'field': 'days_since_contact', 'op': '>', 'value': 30}}], 'ui': {'shell': 'python', 'routes': [{'name': 'Leads', 'path': '/crm/leads', 'component': 'LeadList', 'permission': 'crm:leads'}]}, 'theme': {'name': 'lead_theme', 'tokens': {'accent': '#00838F'}}}, 'OpportunityPipeline': {'id': 'opportunity_pipeline', 'provides': ['deal_tracking', 'pipeline_analytics', 'forecast'], 'requires': ['lead_capture'], 'configuration': {'tenant_id': 'default', 'max_discount_pct': 30}, 'rules': [{'name': 'large_deal_review', 'when': 'amount > 500000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'amount', 'op': '>', 'value': 500000}}, {'name': 'discount_limit', 'when': 'discount_pct > 30', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'discount_pct', 'op': '>', 'value': 30}}, {'name': 'stage_progression', 'when': 'next_stage missing', 'action': 'warn', 'when_ast': {'type': 'missing', 'field': 'next_stage', 'negated': False}}], 'ui': {'shell': 'python', 'routes': [{'name': 'Pipeline', 'path': '/crm/pipeline', 'component': 'PipelineView', 'permission': 'crm:pipeline'}, {'name': 'Forecast', 'path': '/crm/forecast', 'component': 'ForecastView', 'permission': 'crm:forecast'}, {'name': 'Analytics', 'path': '/crm/analytics', 'component': 'SalesAnalytics', 'permission': 'crm:analytics'}]}, 'theme': {'name': 'pipeline_theme', 'tokens': {'accent': '#E65100'}}}}, 'deployment': {'target': 'python', 'source': 'crm_sales.apg'}, 'packages': {}, 'graphs': {'er': {'kind': 'er', 'nodes': 25, 'edges': 26}, 'lookup': {'kind': 'lookup', 'nodes': 6, 'edges': 5}, 'workflow': {'kind': 'workflow', 'nodes': 6, 'edges': 5}, 'handler': {'kind': 'handler', 'nodes': 6, 'edges': 5}, 'capability': {'kind': 'capability', 'nodes': 4, 'edges': 3}, 'security': {'kind': 'security', 'nodes': 6, 'edges': 5}, 'agent': {'kind': 'agent', 'nodes': 0, 'edges': 0}, 'deployment': {'kind': 'deployment', 'nodes': 6, 'edges': 5}, 'package': {'kind': 'package', 'nodes': 6, 'edges': 5}}, 'diagnostics': []}
APG_UI_TEMPLATES: Dict[str, str] = {'entity_list.html.j2': '{# entity_list.html.j2 — APG entity list + create form\n   Variables: entity_name, entity_type, safe_entity, fields, records,\n              total, count, records_table, create_inputs, notice, query,\n              has_kanban (bool), q (search term) #}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">{{ entity_name }}</span>\n  <div class="ml-auto flex items-center gap-1">\n    {% if has_kanban %}\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">≡ List</span>\n    <a href="/ui/entities/{{ safe_entity }}?view=kanban"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ⊞ Kanban\n    </a>\n    {% endif %}\n    <a href="/ui/entities/{{ safe_entity }}?view=analytics"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      Analytics\n    </a>\n    <a href="/entities/{{ safe_entity }}/records.csv"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-500 hover:border-emerald-400 hover:text-emerald-600 transition-colors">\n      ↓ CSV\n    </a>\n    <a href="/entities/{{ safe_entity }}/records"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-500 hover:border-gray-300 transition-colors">\n      API JSON ↗\n    </a>\n    <button type="button" class="apg-btn" onclick="document.getElementById(\'apg-create-drawer\').showModal();">\n      New {{ entity_name }}\n    </button>\n  </div>\n</nav>\n\n{% if notice %}\n<div role="alert"\n     class="mb-4 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">\n  ⚠ {{ notice }}\n</div>\n{% endif %}\n\n{# Search bar #}\n<form method="get" action="/ui/entities/{{ safe_entity }}" class="mb-5">\n  <div class="relative max-w-sm">\n    <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-xs pointer-events-none">🔍</span>\n    <input type="text" name="q" value="{{ q or \'\' }}"\n           placeholder="Search {{ entity_name }} records…"\n           class="w-full pl-8 pr-8 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n    {% if q %}\n    <a href="/ui/entities/{{ safe_entity }}"\n       class="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-700 text-xs leading-none">✕</a>\n    {% endif %}\n  </div>\n</form>\n\n<dialog id="apg-create-drawer" class="apg-drawer" aria-labelledby="apg-create-title">\n  <form method="post" action="/ui/entities/{{ safe_entity }}/records" novalidate class="apg-drawer-panel">\n    <header class="apg-card-header">\n      <div>\n        <h2 id="apg-create-title" class="text-base font-semibold text-gray-900">New {{ entity_name }}</h2>\n        <p class="text-xs text-gray-400 mt-1">{{ entity_type }}</p>\n      </div>\n      <button type="button" class="apg-btn apg-btn-secondary" onclick="document.getElementById(\'apg-create-drawer\').close();">Close</button>\n    </header>\n    <div class="overflow-y-auto max-h-[70vh] space-y-3">\n      {{ create_inputs | safe }}\n    </div>\n    <footer class="flex items-center justify-end gap-2 pt-4 border-t border-gray-100 mt-4">\n      <button type="button" class="apg-btn apg-btn-secondary" onclick="document.getElementById(\'apg-create-drawer\').close();">Cancel</button>\n      <button type="submit" class="apg-btn">Create {{ entity_name }}</button>\n    </footer>\n  </form>\n</dialog>\n\n<div class="flex items-start gap-5 flex-col lg:flex-row">\n\n  {# ── Records section ─────────────────────────────────────────── #}\n  <section class="flex-1 min-w-0">\n    <div class="flex items-center gap-3 mb-3 flex-wrap">\n      <h1 class="text-lg font-semibold text-gray-900">{{ entity_name }}</h1>\n      <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n        {{ total }} record{{ \'s\' if total != 1 else \'\' }}\n      </span>\n      {% if count != total %}\n      <span class="text-xs text-apg-primary bg-blue-50 px-2 py-0.5 rounded-full">\n        {{ count }} match{% if q %} for "{{ q }}"{% endif %}\n      </span>\n      {% endif %}\n    </div>\n\n    <p class="text-xs text-gray-500 mb-2">Showing {{ count }} of {{ total }} matching records.</p>\n\n    {% if records %}\n      {{ records_table | safe }}\n    {% else %}\n      <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n        <div class="text-3xl mb-3 opacity-30">📋</div>\n        {% if q %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records match "{{ q }}".</p>\n        <p class="text-xs text-gray-400 mt-1">\n          <a href="/ui/entities/{{ safe_entity }}" class="text-apg-primary hover:underline">Clear search</a>\n        </p>\n        {% else %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records yet.</p>\n        <p class="text-xs text-gray-400 mt-1">Create the first record to get started.</p>\n        <button type="button" class="apg-btn mt-4" onclick="document.getElementById(\'apg-create-drawer\').showModal();">New {{ entity_name }}</button>\n        {% endif %}\n      </div>\n    {% endif %}\n\n    {# Pagination controls #}\n    {% if total_pages > 1 %}\n    <nav class="mt-4 flex items-center justify-between flex-wrap gap-3" aria-label="Pagination">\n      <div class="flex items-center gap-1 flex-wrap">\n        {% if page > 1 %}\n        <a href="/ui/entities/{{ safe_entity }}?page={{ page - 1 }}&per={{ per }}{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">← Prev</a>\n        {% else %}\n        <span class="px-3 py-1.5 text-sm border border-gray-100 rounded-lg text-gray-300 select-none">← Prev</span>\n        {% endif %}\n\n        {% if page > 3 %}\n        <a href="/ui/entities/{{ safe_entity }}?page=1&per={{ per }}{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">1</a>\n        {% if page > 4 %}<span class="px-1 text-xs text-gray-400">…</span>{% endif %}\n        {% endif %}\n\n        {% for p in range(1, total_pages + 1) %}\n        {% if p >= page - 2 and p <= page + 2 %}\n        <a href="/ui/entities/{{ safe_entity }}?page={{ p }}&per={{ per }}{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}"\n           class="px-3 py-1.5 text-sm rounded-lg {% if p == page %}bg-apg-primary text-white font-semibold{% else %}border border-gray-200 text-gray-600 hover:border-apg-primary hover:text-apg-primary{% endif %} transition-colors">{{ p }}</a>\n        {% endif %}\n        {% endfor %}\n\n        {% if page < total_pages - 2 %}\n        {% if page < total_pages - 3 %}<span class="px-1 text-xs text-gray-400">…</span>{% endif %}\n        <a href="/ui/entities/{{ safe_entity }}?page={{ total_pages }}&per={{ per }}{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">{{ total_pages }}</a>\n        {% endif %}\n\n        {% if page < total_pages %}\n        <a href="/ui/entities/{{ safe_entity }}?page={{ page + 1 }}&per={{ per }}{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">Next →</a>\n        {% else %}\n        <span class="px-3 py-1.5 text-sm border border-gray-100 rounded-lg text-gray-300 select-none">Next →</span>\n        {% endif %}\n      </div>\n      <div class="flex items-center gap-2 text-xs text-gray-400">\n        <span>Page {{ page }} of {{ total_pages }}</span>\n        <select onchange="location.href=\'/ui/entities/{{ safe_entity }}?page=1&per=\'+this.value+\'{% if sort_field %}&sort={{ sort_field }}&dir={{ sort_dir }}{% endif %}{% if q %}&q={{ q }}{% endif %}\'"\n                class="px-2 py-1 border border-gray-200 rounded text-xs text-gray-500 bg-white cursor-pointer focus:outline-none focus:ring-1 focus:ring-apg-primary">\n          {% for n in [10, 25, 50, 100, 200] %}\n          <option value="{{ n }}"{% if n == per %} selected{% endif %}>{{ n }} / page</option>\n          {% endfor %}\n        </select>\n      </div>\n    </nav>\n    {% endif %}\n  </section>\n\n</div>\n\n<details class="mt-4">\n  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 select-none">Advanced filter</summary>\n  <div class="mt-2">{{ query_form | safe }}</div>\n</details>\n\n<details class="mt-2">\n  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 select-none">Record JSON</summary>\n  <pre class="mt-2 text-xs bg-gray-50 border border-gray-200 rounded-lg p-3 overflow-auto max-h-64 font-mono">{{ records_json }}</pre>\n</details>\n', 'workflow_wizard.html.j2': '<section class="max-w-2xl mx-auto">\n  <p class="text-sm text-gray-500 mb-6">\n    <a href="/ui" class="hover:text-blue-600">Application</a> /\n    <a href="/ui/workflows" class="hover:text-blue-600">Workflows</a> /\n    <span class="font-semibold text-gray-900">{{ workflow.name }}</span>\n  </p>\n\n  {% if completed %}\n  <div class="apg-card text-center py-12">\n    <div class="text-5xl mb-4" aria-hidden="true">✓</div>\n    <h1 class="text-xl font-bold text-gray-900 mb-2">{{ workflow.name }} complete</h1>\n    <p class="text-gray-500 text-sm mb-6">Your {{ entity_name }} record has been created successfully.</p>\n    <div class="flex items-center justify-center gap-3 flex-wrap">\n      <a href="/ui/entities/{{ safe_entity }}" class="apg-btn">View all {{ entity_name }} records</a>\n      <a href="/ui/workflows/{{ safe_entity }}/{{ safe_workflow_id }}" class="apg-btn apg-btn-secondary">Start again</a>\n      <a href="/ui/workflows" class="apg-btn apg-btn-secondary">All workflows</a>\n    </div>\n  </div>\n  {% else %}\n  <div class="text-center mb-8">\n    <div class="text-4xl mb-3" aria-hidden="true">{{ workflow.icon }}</div>\n    <h1 class="text-xl font-bold text-gray-900">{{ workflow.name }}</h1>\n    <p class="text-sm text-gray-500 mt-1">{{ workflow.description }}</p>\n  </div>\n\n  <ol class="flex items-center gap-0 mb-8 px-2" aria-label="Workflow progress">\n    {% for item in progress %}\n    <li class="flex items-center gap-1.5 text-xs font-medium {{ item.class_name }}">\n      <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs {{ item.badge_class }}">{{ item.label }}</span>\n      <span class="hidden sm:block">{{ item.title }}</span>\n    </li>\n    {% if not loop.last %}\n    <li class="flex-1 h-px bg-gray-200 mx-1" aria-hidden="true"><span class="block h-px bg-blue-600" style="width:{{ \'100%\' if loop.index0 < step_index else \'0%\' }}"></span></li>\n    {% endif %}\n    {% endfor %}\n  </ol>\n\n  <article class="apg-card overflow-hidden">\n    <header class="px-6 py-4 border-b border-gray-100 bg-gray-50 -mx-4 -mt-4 mb-4">\n      <h2 class="font-semibold text-gray-900">Step {{ step_index + 1 }} of {{ total_steps }}: {{ step.title }}</h2>\n      <p class="text-sm text-gray-500 mt-0.5">{{ step.subtitle }}</p>\n    </header>\n    {% if error %}\n    <div role="alert" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{{ error }}</div>\n    {% endif %}\n    <form method="post" action="{{ next_url }}" class="space-y-4">\n      {{ hidden_fields | safe }}\n      {{ step_inputs | safe }}\n      <div class="flex items-center justify-between pt-4 border-t border-gray-100 mt-6">\n        {% if step_index > 0 %}\n        <a href="/ui/workflows/{{ safe_entity }}/{{ safe_workflow_id }}/step/{{ step_index - 1 }}" class="apg-btn apg-btn-secondary">Back</a>\n        {% else %}\n        <a href="/ui/workflows" class="apg-btn apg-btn-secondary">Cancel</a>\n        {% endif %}\n        <button type="submit" class="apg-btn">{{ next_label }}</button>\n      </div>\n    </form>\n  </article>\n  {% endif %}\n</section>\n', 'database_catalog.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Databases</span></p>\n  <h1 class="text-xl font-bold text-gray-900">Databases</h1>\n  <p class="text-sm text-gray-500 mt-1">{{ status.database_count }} database(s), {{ status.schema_count }} schema(s), {{ status.table_count }} table(s), {{ status.reference_count }} reference(s).</p>\n  <p class="text-sm text-gray-500 mt-1">Status: <strong>{{ status_label }}</strong></p>\n</section>\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">\n  {% for database in databases %}\n  <article class="apg-card">\n    <header class="apg-card-header">\n      <h2 class="text-base font-semibold text-gray-900">{{ database.name }}</h2>\n      <a class="text-sm hover:underline" href="/databases/{{ database.name }}/schemas">Schema JSON</a>\n    </header>\n    {% for schema in database.schemas %}\n    <section class="mb-4">\n      <h3 class="text-sm font-semibold text-gray-700">{{ schema.name }}</h3>\n      {% if schema.tables %}\n      <div class="space-y-3">\n        {% for table in schema.tables %}\n        <div class="border border-gray-200 rounded-lg p-3">\n          <p class="font-medium text-sm text-gray-900">{{ table.name }}</p>\n          {% if table.columns %}\n          <ul class="mt-2 space-y-1">\n            {% for column in table.columns %}\n            <li class="text-xs text-gray-500"><span class="font-mono text-gray-700">{{ column.name }}</span> · {{ column.type }}</li>\n            {% endfor %}\n          </ul>\n          {% else %}\n          <p class="text-xs text-gray-400 mt-2">No columns declared.</p>\n          {% endif %}\n        </div>\n        {% endfor %}\n      </div>\n      {% else %}\n      <p class="text-xs text-gray-400">No tables declared.</p>\n      {% endif %}\n    </section>\n    {% endfor %}\n  </article>\n  {% endfor %}\n</section>\n\n<section class="apg-card">\n  <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Relationships</h2></header>\n  {% if relationships %}\n  <ul class="space-y-2">\n    {% for relationship in relationships %}\n    <li class="text-sm text-gray-600"><span class="font-mono">{{ relationship.source }}</span> → <span class="font-mono">{{ relationship.target }}</span></li>\n    {% endfor %}\n  </ul>\n  {% else %}\n  <p class="text-sm text-gray-500">No relationships declared.</p>\n  {% endif %}\n</section>\n\n<section class="apg-card">\n  <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Validation</h2><span class="apg-badge {{ \'apg-badge-success\' if status.valid else \'apg-badge-danger\' }}">{{ status_label }}</span></header>\n  <pre>{{ validation_json }}</pre>\n</section>\n', 'kanban_view.html.j2': '{# kanban_view.html.j2 — Kanban board view for status-field entities\n   Variables: entity_name, safe_entity, columns, display_field, status_field, fields\n   columns: [{"label": str, "records": [dict]}]\n#}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Kanban</span>\n  <div class="ml-auto flex items-center gap-1">\n    <a href="/ui/entities/{{ safe_entity }}"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ≡ List\n    </a>\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">⊞ Kanban</span>\n  </div>\n</nav>\n\n<div class="flex items-center gap-4 mb-5">\n  <h1 class="text-xl font-bold text-gray-900">{{ entity_name }}</h1>\n  <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n    by {{ status_field | replace(\'_\', \' \') }}\n  </span>\n  <span class="text-xs text-gray-400">\n    {{ columns | sum(attribute=\'records\') | length }} total\n  </span>\n</div>\n\n{# Kanban board — horizontal scroll #}\n<div class="flex gap-4 overflow-x-auto pb-6 items-start -mx-1 px-1">\n  {% for col in columns %}\n  <div class="flex-shrink-0 w-72">\n    {# Column header #}\n    <div class="flex items-center justify-between mb-3 px-1">\n      <div class="flex items-center gap-2">\n        <span class="w-2.5 h-2.5 rounded-full\n          {% if col.label | lower in [\'active\', \'approved\', \'paid\', \'open\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-400\n          {% elif col.label | lower in [\'inactive\', \'rejected\', \'closed\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-400\n          {% elif col.label | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-400\n          {% else %}bg-gray-300{% endif %}"></span>\n        <h2 class="text-sm font-semibold text-gray-900">{{ col.label }}</h2>\n        <span class="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full font-medium">{{ col.records | length }}</span>\n      </div>\n    </div>\n\n    {# Card list — apg-kanban-col enables SortableJS drag-and-drop #}\n    <div class="space-y-2.5 apg-kanban-col min-h-16"\n         data-col-label="{{ col.label | e }}"\n         id="apg-col-{{ col.label | urlencode }}">\n      {% for record in col.records %}\n      {% set rec_id = record.get(\'id\', \'\') | string %}\n      <a href="/ui/entities/{{ safe_entity }}/{{ rec_id | urlencode }}"\n         data-record-id="{{ rec_id | e }}"\n         data-revision="{{ record.get(\'_revision\', \'\') | string | e }}"\n         class="block bg-white rounded-xl border border-gray-200 p-4 hover:border-apg-primary hover:shadow-md transition-all group/card cursor-grab active:cursor-grabbing">\n        <div class="flex items-start justify-between gap-2 mb-2.5">\n          <div class="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n               style="background: var(--apg-primary, #0ea5e9)">\n            {{ (record.get(display_field, \'\') | string)[:1] | upper or \'?\' }}\n          </div>\n          <span class="text-xs text-gray-300 font-mono mt-1">{{ rec_id[:8] }}</span>\n        </div>\n        <p class="text-sm font-semibold text-gray-900 group-hover/card:text-apg-primary transition-colors leading-tight mb-2">\n          {{ record.get(display_field, \'—\') | string | truncate(50) }}\n        </p>\n        {% for f in fields %}\n        {% if f.name not in [\'id\', \'_revision\', display_field, status_field] %}\n        {% set fval = record.get(f.name, \'\') %}\n        {% if fval and fval != \'\' and fval != \'None\' %}\n        <p class="text-xs text-gray-400 truncate mt-0.5">\n          <span class="font-medium text-gray-500">{{ (f.name[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if f.name.endswith(\'_id\') else (f.name | replace(\'_\', \' \') | title) }}:</span>\n          {{ fval | string | truncate(35) }}\n        </p>\n        {% if loop.index >= 2 %}{% break %}{% endif %}\n        {% endif %}\n        {% endif %}\n        {% endfor %}\n      </a>\n      {% endfor %}\n\n      {% if not col.records %}\n      <div class="bg-gray-50 rounded-xl border border-dashed border-gray-200 p-6 text-center apg-kanban-empty">\n        <p class="text-xs text-gray-300">Empty</p>\n      </div>\n      {% endif %}\n    </div>\n  </div>\n  {% endfor %}\n\n  {% if not columns %}\n  <div class="flex-1 text-center py-16 text-gray-400">\n    <div class="text-4xl mb-3 opacity-20">⊞</div>\n    <p class="text-sm">No records to display.</p>\n  </div>\n  {% endif %}\n</div>\n\n<script>\n(function() {\n  var entity = {{ safe_entity | tojson }};\n  var statusField = {{ status_field | tojson }};\n\n  document.querySelectorAll(\'.apg-kanban-col\').forEach(function(col) {\n    new Sortable(col, {\n      group: \'apg-kanban\',\n      animation: 150,\n      ghostClass: \'opacity-30\',\n      chosenClass: \'shadow-lg\',\n      dragClass: \'rotate-1\',\n      onEnd: function(evt) {\n        var card = evt.item;\n        var recordId = card.dataset.recordId;\n        var newCol = evt.to;\n        var newStatus = newCol.dataset.colLabel;\n        if (!recordId || !newStatus) return;\n\n        var body = {record: {}};\n        body.record[statusField] = newStatus;\n\n        fetch(\'/entities/\' + encodeURIComponent(entity) + \'/records/\' + encodeURIComponent(recordId), {\n          method: \'PUT\',\n          headers: {\'Content-Type\': \'application/json\'},\n          body: JSON.stringify(body)\n        }).then(function(r) {\n          if (r.ok) {\n            APGToast(\'Moved to \' + newStatus, \'success\');\n          } else {\n            APGToast(\'Move failed — \' + r.status, \'error\');\n            evt.from.insertBefore(card, evt.from.children[evt.oldIndex] || null);\n          }\n        }).catch(function() {\n          APGToast(\'Move failed\', \'error\');\n          evt.from.insertBefore(card, evt.from.children[evt.oldIndex] || null);\n        });\n      }\n    });\n  });\n})();\n</script>\n', 'landing.html.j2': '{# landing.html.j2 — APG application landing page\n   Variables: module_name, module_description, entities, capabilities,\n              theme_primary, theme_accent, landing_style, api_links\n   landing_style: "default" | "minimal" | "corporate" | "africa"\n#}\n<!doctype html>\n<html lang="en" class="h-full">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>{{ module_name | replace(\'_\', \' \') | title }}</title>\n  <link rel="stylesheet" href="/static/apg.css">\n  <style>\n    :root {\n      --brand: {{ theme_primary }};\n      --accent: {{ theme_accent }};\n    }\n    .hero-gradient {\n      background: linear-gradient(135deg, {{ theme_primary }} 0%, {{ theme_accent }} 100%);\n    }\n    {% if landing_style == \'minimal\' %}\n    .hero-gradient { background: {{ theme_primary }}; }\n    {% elif landing_style == \'africa\' %}\n    .hero-gradient {\n      background: linear-gradient(135deg, #8B1A1A 0%, {{ theme_primary }} 50%, #E9A84B 100%);\n    }\n    {% elif landing_style == \'corporate\' %}\n    .hero-gradient { background: linear-gradient(180deg, #0F172A 0%, {{ theme_primary }} 100%); }\n    {% endif %}\n  </style>\n</head>\n<body class="min-h-full bg-gray-50 font-sans antialiased">\n\n  {# ── Hero ──────────────────────────────────────────────────────── #}\n  <div class="hero-gradient text-white">\n    <div class="max-w-6xl mx-auto px-6">\n      {# Topnav #}\n      <nav class="flex items-center justify-between py-5">\n        <span class="text-lg font-bold tracking-tight">\n          {{ module_name | replace(\'_\', \' \') | title }}\n        </span>\n        <div class="flex items-center gap-3">\n          <a href="/ui"\n             class="px-4 py-2 text-sm font-medium bg-white/15 hover:bg-white/25 rounded-lg transition-colors">\n            Open App\n          </a>\n          <a href="/openapi.json"\n             class="px-4 py-2 text-sm font-medium border border-white/30 hover:bg-white/10 rounded-lg transition-colors">\n            API Docs\n          </a>\n        </div>\n      </nav>\n\n      {# Hero content #}\n      <div class="py-20 text-center max-w-3xl mx-auto">\n        <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/15 text-xs font-medium mb-6">\n          <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>\n          Powered by APG · Datacraft\n        </div>\n        <h1 class="text-4xl sm:text-5xl font-extrabold tracking-tight mb-5">\n          {{ module_name | replace(\'_\', \' \') | title }}\n        </h1>\n        <p class="text-lg text-white/80 mb-8 leading-relaxed">\n          {{ module_description or \'A fully generated application by Datacraft APG.\' }}\n        </p>\n        <div class="flex items-center justify-center gap-3 flex-wrap">\n          <a href="/ui"\n             class="px-6 py-3 bg-white text-gray-900 font-semibold rounded-xl hover:bg-white/90 transition-colors shadow-lg">\n            Open Application →\n          </a>\n          <a href="/manifest"\n             class="px-6 py-3 border border-white/40 text-white font-medium rounded-xl hover:bg-white/10 transition-colors">\n            View Manifest\n          </a>\n        </div>\n      </div>\n    </div>\n  </div>\n\n  {# ── Stats strip ────────────────────────────────────────────────── #}\n  <div class="bg-white border-b border-gray-200 shadow-sm">\n    <div class="max-w-6xl mx-auto px-6 py-5 grid grid-cols-2 sm:grid-cols-4 gap-6 divide-x divide-gray-100">\n      {% for stat in stats %}\n      <div class="px-6 first:pl-0 last:pr-0 text-center">\n        <p class="text-2xl font-bold text-gray-900">{{ stat.value }}</p>\n        <p class="text-xs font-medium uppercase tracking-wide text-gray-400 mt-0.5">{{ stat.label }}</p>\n      </div>\n      {% endfor %}\n    </div>\n  </div>\n\n  {# ── Entity cards ───────────────────────────────────────────────── #}\n  <main class="max-w-6xl mx-auto px-6 py-12">\n    <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-5">Data Entities</h2>\n    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">\n      {% for entity in entities %}\n      {% if entity.type != \'application\' %}\n      <a href="/ui/entities/{{ entity.name }}"\n         class="group bg-white rounded-xl border border-gray-200 p-5 hover:border-brand hover:shadow-md transition-all">\n        <div class="flex items-start justify-between mb-3">\n          <div class="w-9 h-9 rounded-lg flex items-center justify-center text-white text-sm font-bold"\n               style="background: var(--brand)">\n            {{ entity.name[0] | upper }}\n          </div>\n          <span class="text-xs font-medium text-gray-400 bg-gray-50 px-2 py-0.5 rounded uppercase tracking-wide">\n            {{ entity.type }}\n          </span>\n        </div>\n        <h3 class="font-semibold text-gray-900 group-hover:text-brand transition-colors">\n          {{ entity.name }}\n        </h3>\n        <p class="text-xs text-gray-400 mt-1">\n          {{ entity.fields | length if entity.fields else entity.properties | length }} fields\n        </p>\n      </a>\n      {% endif %}\n      {% endfor %}\n    </div>\n\n    {# ── API quick links ──────────────────────────────────────────── #}\n    <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-4">API Endpoints</h2>\n    <div class="flex flex-wrap gap-2">\n      {% for link in api_links %}\n      <a href="{{ link.url }}"\n         class="px-3 py-1.5 text-sm text-gray-600 bg-white border border-gray-200 rounded-lg hover:border-brand hover:text-brand transition-colors">\n        {{ link.label }}\n      </a>\n      {% endfor %}\n    </div>\n  </main>\n\n  {# ── Footer ─────────────────────────────────────────────────────── #}\n  <footer class="border-t border-gray-200 py-6 text-center text-xs text-gray-400">\n    Generated by <span class="font-medium text-gray-600">APG</span> ·\n    <span>Datacraft</span> ·\n    © 2025\n  </footer>\n\n</body>\n</html>\n', 'capability_console.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <a href="/capabilities" class="hover:text-blue-600">Capability catalog</a> / <span class="font-semibold text-gray-900">{{ name }}</span></p>\n  <h1 class="text-xl font-bold text-gray-900">{{ name }}</h1>\n</section>\n\n{% if error %}\n<div role="alert" class="bg-red-50 border border-red-200 text-red-700">{{ error }}</div>\n{% endif %}\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4">\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/rules/evaluate" class="space-y-4">\n    <h2 class="text-base font-semibold text-gray-900">Rules Evaluate</h2>\n    <label>Context JSON <textarea name="context_json" rows="8">{}</textarea></label>\n    <button type="submit" class="apg-btn">Evaluate</button>\n  </form>\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/configuration/resolve" class="space-y-4">\n    <h2 class="text-base font-semibold text-gray-900">Configuration Resolve</h2>\n    <label>Overrides JSON <textarea name="configuration_json" rows="8">{}</textarea></label>\n    <button type="submit" class="apg-btn">Resolve</button>\n  </form>\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/approval/plan" class="space-y-4">\n    <h2 class="text-base font-semibold text-gray-900">Approval Plan</h2>\n    <label>Context JSON <textarea name="context_json" rows="8">{}</textarea></label>\n    <button type="submit" class="apg-btn">Plan</button>\n  </form>\n</section>\n\n<section class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">\n  <article class="apg-card">\n    <h2 class="text-base font-semibold text-gray-900 mb-3">Description</h2>\n    <details open>\n      <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw capability JSON</summary>\n      <pre>{{ description_json }}</pre>\n    </details>\n  </article>\n  <article class="apg-card">\n    <h2 class="text-base font-semibold text-gray-900 mb-3">Result</h2>\n    {% if result %}\n    <dl class="space-y-2 mb-4">\n      {% for key, value in result_items %}\n      <div><dt class="text-xs font-semibold uppercase tracking-wide text-gray-400">{{ key }}</dt><dd class="text-sm text-gray-700 break-words">{{ value }}</dd></div>\n      {% endfor %}\n    </dl>\n    <details>\n      <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw result JSON</summary>\n      <pre>{{ result_json_html | safe }}</pre>\n    </details>\n    {% else %}\n    <p class="text-sm text-gray-500">Run a capability operation to view results.</p>\n    {% endif %}\n  </article>\n</section>\n', 'record_detail.html.j2': '{# record_detail.html.j2 — Salesforce-quality record detail page\n   Variables: entity_name, entity_type, safe_entity, safe_record_id,\n              record, fields, field_semantics, title, status_val, revision,\n              related_lists, has_kanban (bool)\n   related_lists: [{"entity": str, "fk_field": str, "records": [dict], "cols": [str]}]\n   field_semantics: {field_name: semantic_type}\n#}\n\n{# Breadcrumb #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900 truncate max-w-xs">{{ title }}</span>\n</nav>\n\n{# Record header card #}\n<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5 overflow-hidden">\n  <div class="h-1 bg-apg-primary"></div>\n  <div class="px-6 py-5 flex items-start gap-4">\n    <div class="w-14 h-14 rounded-xl flex items-center justify-center text-white text-2xl font-bold flex-shrink-0"\n         style="background: var(--apg-primary, #0ea5e9)">\n      {{ (title[:1] | upper) if title else (entity_name[:1] | upper) }}\n    </div>\n    <div class="flex-1 min-w-0">\n      <div class="flex items-center gap-3 flex-wrap">\n        <h1 class="text-xl font-bold text-gray-900 break-all">{{ title }}</h1>\n        {% if status_val %}\n        <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold\n          {% if status_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\'] %}bg-green-100 text-green-800\n          {% elif status_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n          {% elif status_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n          {% else %}bg-gray-100 text-gray-600{% endif %}">\n          {{ status_val }}\n        </span>\n        {% endif %}\n        <span class="text-xs font-medium text-gray-400 bg-gray-50 px-2 py-0.5 rounded uppercase tracking-wide">{{ entity_type }}</span>\n      </div>\n      {% set id_val = record.get(\'id\', \'\') %}\n      {% if id_val %}\n      <p class="text-xs text-gray-400 mt-1 font-mono truncate">{{ id_val | string }}</p>\n      {% endif %}\n    </div>\n    <div class="flex items-center gap-2 flex-shrink-0 flex-wrap">\n      <a href="/ui/workflows/{{ safe_entity }}/create_{{ safe_entity }}"\n         class="px-3 py-1.5 text-sm font-medium bg-apg-primary text-white rounded-lg hover:opacity-90 transition-opacity whitespace-nowrap">\n        ⚡ Workflow\n      </a>\n      <form method="post"\n            action="/ui/entities/{{ safe_entity }}/records/{{ safe_record_id }}/delete"\n            class="inline"\n            onsubmit="return apgConfirmSubmit(this, \'Delete this record? This cannot be undone.\')">\n        <input type="hidden" name="expected_revision" value="{{ revision }}">\n        <button type="submit"\n                class="px-3 py-1.5 text-sm font-medium border border-red-200 text-red-500 rounded-lg hover:bg-red-50 transition-colors">\n          Delete\n        </button>\n      </form>\n    </div>\n  </div>\n{# Highlights panel — top fields at a glance #}\n{% set highlight_fields = [] %}\n{% for f in fields %}\n  {% if f.name not in [\'id\', \'_revision\'] and not f.name.endswith(\'_id\') %}\n    {% if highlight_fields | length < 4 %}\n      {% set _ = highlight_fields.append(f) %}\n    {% endif %}\n  {% endif %}\n{% endfor %}\n{% if highlight_fields %}\n<div class="border-t border-gray-100 px-6 py-3 grid grid-cols-2 md:grid-cols-4 gap-4 bg-gray-50/50">\n  {% for f in highlight_fields %}\n  {% set fv = record.get(f.name, \'\') %}\n  {% set sem = field_semantics.get(f.name, \'text\') if field_semantics else \'text\' %}\n  <div class="min-w-0">\n    <p class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-0.5 truncate">\n      {{ (f.name[:-3] | replace(\'_\',\' \') | title ~ \' ID\') if f.name.endswith(\'_id\') else (f.name | replace(\'_\',\' \') | title) }}\n    </p>\n    <p class="text-sm font-medium text-gray-900 truncate">\n      {% if fv is none or fv == \'\' or fv | string == \'None\' %}\n      <span class="text-gray-300 italic text-xs">—</span>\n      {% elif sem == \'currency\' %}\n      <span class="tabular-nums">{{ fv | string }}</span>\n      {% elif sem == \'status\' %}\n      <span class="inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-semibold\n        {% if fv | string | lower in [\'active\',\'approved\',\'paid\',\'open\',\'enabled\',\'complete\',\'completed\',\'success\',\'done\'] %}bg-green-100 text-green-800\n        {% elif fv | string | lower in [\'inactive\',\'rejected\',\'closed\',\'disabled\',\'cancelled\',\'canceled\',\'failed\',\'expired\'] %}bg-red-100 text-red-800\n        {% else %}bg-yellow-100 text-yellow-800{% endif %}">{{ fv }}</span>\n      {% else %}\n      {{ fv | string | truncate(30) }}\n      {% endif %}\n    </p>\n  </div>\n  {% endfor %}\n</div>\n{% endif %}\n</div>\n\n{# Tab bar #}\n<div class="flex items-center gap-1 border-b border-gray-200 mb-6">\n  <button onclick="apgTab(\'details\')" id="apg-tab-details"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-apg-primary text-apg-primary -mb-px transition-colors">\n    Details\n  </button>\n  <button onclick="apgTab(\'related\')" id="apg-tab-related"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Related\n    {% if related_lists %}\n    <span class="ml-1 text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">\n      {{ related_lists | sum(attribute=\'records\') | length if related_lists else 0 }}\n    </span>\n    {% endif %}\n  </button>\n  <button onclick="apgTab(\'activity\')" id="apg-tab-activity"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Activity\n  </button>\n</div>\n\n{# Details panel #}\n<div id="apg-panel-details">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n      <h2 class="text-sm font-semibold text-gray-900">Record Details</h2>\n      <span class="text-xs text-gray-400 font-mono">rev. {{ revision }}</span>\n    </div>\n    <div class="p-5 grid grid-cols-1 md:grid-cols-2 gap-x-8">\n      {% for field in fields %}\n      {% if field.name != \'_revision\' %}\n      {% set field_val = record.get(field.name, \'\') %}\n      {% set fld_id = \'fld-\' ~ safe_entity ~ \'-\' ~ safe_record_id ~ \'-\' ~ field.name %}\n      <div id="{{ fld_id }}" class="py-3 border-b border-gray-50 last:border-0 group/field">\n        <dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">\n          {{ (field.name[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if field.name.endswith(\'_id\') else (field.name | replace(\'_\', \' \') | title) }}\n        </dt>\n        <dd class="flex items-center justify-between gap-2 min-h-6">\n          <span class="text-sm text-gray-900 break-words">\n            {% set semantic = field_semantics.get(field.name, \'text\') if field_semantics else \'text\' %}\n            {% include \'widgets/field_display.html.j2\' %}\n          </span>\n          <button\n            hx-get="/ui/entities/{{ safe_entity }}/{{ safe_record_id }}/fields/{{ field.name }}/edit"\n            hx-target="#{{ fld_id }}"\n            hx-swap="outerHTML"\n            class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"\n            title="Edit {{ field.name }}">\n            <svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">\n              <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zm-2.207 2.207L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>\n            </svg>\n          </button>\n        </dd>\n      </div>\n      {% endif %}\n      {% endfor %}\n    </div>\n  </div>\n</div>\n\n{# Related panel #}\n<div id="apg-panel-related" class="hidden">\n  {% if related_lists %}\n    {% for rel in related_lists %}\n    <div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-4">\n      <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n        <div class="flex items-center gap-2">\n          <h2 class="text-sm font-semibold text-gray-900">{{ rel.entity }}</h2>\n          <span class="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded-full font-medium">{{ rel.records | length }}</span>\n        </div>\n        <a href="/ui/entities/{{ rel.entity | urlencode }}"\n           class="text-xs text-apg-primary hover:underline">View all →</a>\n      </div>\n      {% if rel.records %}\n      <div class="overflow-x-auto">\n        <table class="w-full text-sm">\n          <thead>\n            <tr class="bg-gray-50 border-b border-gray-100">\n              {% for col in rel.cols %}\n              <th class="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">\n                {{ (col[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if col.endswith(\'_id\') else (col | replace(\'_\', \' \') | title) }}\n              </th>\n              {% endfor %}\n              <th class="px-4 py-2 w-16"></th>\n            </tr>\n          </thead>\n          <tbody class="divide-y divide-gray-50">\n            {% for row in rel.records[:5] %}\n            <tr class="hover:bg-gray-50 transition-colors">\n              {% for col in rel.cols %}\n              <td class="px-4 py-2.5 text-gray-700 max-w-xs truncate">\n                {{ row.get(col, \'\') | string | truncate(40) }}\n              </td>\n              {% endfor %}\n              <td class="px-4 py-2.5 text-right">\n                <a href="/ui/entities/{{ rel.entity | urlencode }}/{{ row.get(\'id\', \'\') | string | urlencode }}"\n                   class="text-xs text-apg-primary hover:underline font-medium">View →</a>\n              </td>\n            </tr>\n            {% endfor %}\n            {% if rel.records | length > 5 %}\n            <tr>\n              <td colspan="{{ rel.cols | length + 1 }}" class="px-4 py-2.5 text-center text-xs text-gray-400">\n                + {{ rel.records | length - 5 }} more —\n                <a href="/ui/entities/{{ rel.entity | urlencode }}" class="text-apg-primary hover:underline">view all</a>\n              </td>\n            </tr>\n            {% endif %}\n          </tbody>\n        </table>\n      </div>\n      {% else %}\n      <div class="px-4 py-8 text-center text-sm text-gray-400">No related {{ rel.entity }} records.</div>\n      {% endif %}\n    </div>\n    {% endfor %}\n  {% else %}\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n    <div class="text-4xl mb-3 opacity-20">🔗</div>\n    <p class="text-sm font-medium text-gray-500">No related records found.</p>\n    <p class="text-xs text-gray-400 mt-1">Other entities with FK fields pointing to {{ entity_name }} appear here.</p>\n  </div>\n  {% endif %}\n</div>\n\n{# Activity panel #}\n<div id="apg-panel-activity" class="hidden">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100">\n      <h2 class="text-sm font-semibold text-gray-900">Activity</h2>\n    </div>\n    <div class="p-5">\n      <ol class="relative border-l-2 border-gray-100 ml-4 space-y-5">\n        {% if activity_events %}\n          {% for ev in activity_events %}\n          <li class="ml-6">\n            <span class="absolute flex items-center justify-center w-8 h-8 rounded-full -left-4 text-sm ring-4 ring-white\n              {% if ev.type == \'create\' %}bg-blue-50\n              {% elif ev.type == \'update\' %}bg-purple-50\n              {% elif ev.type == \'delete\' %}bg-red-50\n              {% elif ev.type == \'note\' %}bg-yellow-50\n              {% else %}bg-gray-50{% endif %}">\n              {% if ev.type == \'create\' %}📋\n              {% elif ev.type == \'update\' %}✏️\n              {% elif ev.type == \'delete\' %}🗑️\n              {% elif ev.type == \'note\' %}💬\n              {% else %}⚡{% endif %}\n            </span>\n            <div class="pl-2">\n              <p class="text-sm font-medium text-gray-900">{{ ev.detail or (ev.type | title) }}</p>\n              <p class="text-xs text-gray-400 mt-0.5">\n                {{ ev.actor or \'APG\' }}\n                {% if ev.ts %} · {{ ev.ts }}{% endif %}\n              </p>\n            </div>\n          </li>\n          {% endfor %}\n        {% else %}\n          <li class="ml-6">\n            <span class="absolute flex items-center justify-center w-8 h-8 bg-blue-50 rounded-full -left-4 text-sm ring-4 ring-white">📋</span>\n            <div class="pl-2">\n              <p class="text-sm font-medium text-gray-900">Record created</p>\n              <p class="text-xs text-gray-400 mt-0.5">Revision {{ revision }} · via APG</p>\n            </div>\n          </li>\n        {% endif %}\n      </ol>\n      <form method="post"\n            action="/ui/entities/{{ safe_entity }}/records/{{ safe_record_id }}/note"\n            class="mt-8 flex gap-3">\n        <div class="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n             style="background: var(--apg-primary, #0ea5e9)">A</div>\n        <div class="flex-1">\n          <textarea name="note" placeholder="Add a note…" rows="2" required\n                    class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent placeholder-gray-300"></textarea>\n          <button type="submit"\n                  class="mt-1.5 px-3 py-1.5 bg-apg-primary text-white text-xs font-medium rounded-lg hover:opacity-90 transition-opacity">\n            Save Note\n          </button>\n        </div>\n      </form>\n    </div>\n  </div>\n</div>\n\n<script>\nfunction apgTab(name) {\n  document.querySelectorAll(\'.apg-tab-btn\').forEach(function(b) {\n    b.classList.remove(\'border-apg-primary\', \'text-apg-primary\');\n    b.classList.add(\'border-transparent\', \'text-gray-500\');\n  });\n  document.querySelectorAll(\'[id^="apg-panel-"]\').forEach(function(p) { p.classList.add(\'hidden\'); });\n  var btn = document.getElementById(\'apg-tab-\' + name);\n  if (btn) {\n    btn.classList.remove(\'border-transparent\', \'text-gray-500\');\n    btn.classList.add(\'border-apg-primary\', \'text-apg-primary\');\n  }\n  var panel = document.getElementById(\'apg-panel-\' + name);\n  if (panel) panel.classList.remove(\'hidden\');\n}\n</script>\n', 'entity_analytics.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-apg-primary">Application</a> / <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary">{{ entity_name }}</a> / <span class="font-semibold text-gray-900">Analytics</span></p>\n  <div class="flex items-center justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">{{ entity_name }} Analytics</h1>\n      <p class="text-sm text-gray-500 mt-1">{{ total }} record{{ \'s\' if total != 1 else \'\' }}</p>\n    </div>\n    <a class="apg-btn apg-btn-secondary" href="/ui/entities/{{ safe_entity }}">Table</a>\n  </div>\n</section>\n\n<section class="apg-grid-2 gap-6 mb-6">\n  <article class="apg-card">\n    <div class="apg-card-header"><h2 class="text-sm font-semibold text-gray-900">Records Over Time</h2></div>\n    <div class="apg-chart" data-apg-chart="{{ line_chart.id }}"></div>\n    <script id="{{ line_chart.id }}" type="application/json">{{ line_chart.spec_json | safe }}</script>\n  </article>\n  <article class="apg-card">\n    <div class="apg-card-header"><h2 class="text-sm font-semibold text-gray-900">Status Distribution</h2></div>\n    <div class="apg-chart" data-apg-chart="{{ status_chart.id }}"></div>\n    <script id="{{ status_chart.id }}" type="application/json">{{ status_chart.spec_json | safe }}</script>\n  </article>\n</section>\n\n<section class="apg-grid-3 gap-4">\n  {% for stat in numeric_stats %}\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 mb-3">{{ stat.field }}</h2>\n    <dl class="grid grid-cols-3 gap-3 text-center">\n      <div><dt class="text-xs text-gray-400">Min</dt><dd class="font-semibold">{{ stat.min }}</dd></div>\n      <div><dt class="text-xs text-gray-400">Avg</dt><dd class="font-semibold">{{ stat.avg }}</dd></div>\n      <div><dt class="text-xs text-gray-400">Max</dt><dd class="font-semibold">{{ stat.max }}</dd></div>\n    </dl>\n  </article>\n  {% else %}\n  <article class="apg-card"><div class="apg-chart-empty"><p>No numeric fields available</p></div></article>\n  {% endfor %}\n</section>\n', 'agent_console.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <a href="/agents" class="hover:text-blue-600">Agent catalog</a> / <span class="font-semibold text-gray-900">{{ name }}</span></p>\n  <h1 class="text-xl font-bold text-gray-900">{{ name }}</h1>\n  <p class="text-sm text-gray-500 mt-1">{{ \'Team console\' if team else \'Agent console\' }}</p>\n</section>\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4">\n  <article class="apg-card lg:col-span-2 flex flex-col min-h-16">\n    <div class="flex-1 space-y-3 mb-4">\n      <div class="rounded-lg bg-gray-50 border border-gray-200 p-4">\n        <p class="text-sm text-gray-600">Submit a prompt and optional JSON payload. Streaming output is added in the live UI work package.</p>\n      </div>\n      {% if error %}\n      <div role="alert" class="bg-red-50 border border-red-200 text-red-700">{{ error }}</div>\n      {% endif %}\n      {% if result %}\n      <div class="rounded-lg border border-gray-200 p-4 bg-white">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Response</p>\n        <pre>{{ result_json }}</pre>\n      </div>\n      {% endif %}\n    </div>\n    <form method="post" action="{{ action }}" class="space-y-4">\n      <label>Message <input name="message" type="text" autocomplete="off"></label>\n      <label>Payload JSON <textarea name="payload_json" rows="8">{}</textarea></label>\n      <button type="submit" class="apg-btn">Invoke</button>\n    </form>\n  </article>\n\n  <aside class="apg-card">\n    <h2 class="text-base font-semibold text-gray-900 mb-3">Configuration</h2>\n    <details open>\n      <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw description JSON</summary>\n      <pre>{{ description_json }}</pre>\n    </details>\n    {% if result %}\n    <details class="mt-4">\n      <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw response JSON</summary>\n      <pre>{{ result_json }}</pre>\n    </details>\n    {% endif %}\n  </aside>\n</section>\n', 'marketplace.html.j2': '{# marketplace.html.j2 — APG Connector Marketplace\n   Variables: connectors (list of manifest dicts), installed_count\n#}\n\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Connector Marketplace</span>\n</nav>\n\n<div class="flex items-center gap-4 mb-6">\n  <h1 class="text-xl font-bold text-gray-900">Connector Marketplace</h1>\n  <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n    {{ connectors | length }} connector{{ \'s\' if connectors | length != 1 else \'\' }}\n  </span>\n</div>\n\n{% if connectors %}\n<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">\n  {% for conn in connectors %}\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm hover:border-apg-primary hover:shadow-md transition-all p-5 group">\n    <div class="flex items-start gap-4 mb-3">\n      <div class="w-12 h-12 rounded-xl flex items-center justify-center text-white text-xl font-bold flex-shrink-0"\n           style="background: var(--apg-primary, #0ea5e9)">\n        {{ (conn.title or conn.name or \'?\')[:1] | upper }}\n      </div>\n      <div class="flex-1 min-w-0">\n        <h2 class="text-sm font-bold text-gray-900 group-hover:text-apg-primary transition-colors truncate">\n          {{ conn.title or conn.name }}\n        </h2>\n        <p class="text-xs text-gray-400 truncate">{{ conn.base_url or \'Custom connector\' }}</p>\n      </div>\n    </div>\n    <div class="flex items-center gap-3 text-xs text-gray-500 mb-4">\n      <span class="flex items-center gap-1">\n        <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span>\n        {{ (conn.operations or []) | length }} operations\n      </span>\n      {% if conn.version %}\n      <span class="text-gray-300">·</span>\n      <span>v{{ conn.version }}</span>\n      {% endif %}\n    </div>\n    <div class="flex items-center gap-2">\n      <span class="flex-1 text-xs text-gray-400 font-mono truncate">{{ conn.file or conn.name }}</span>\n      <a href="/entities/connectors/{{ conn.name | urlencode }}"\n         class="px-3 py-1.5 text-xs font-medium border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n        View API ↗\n      </a>\n    </div>\n  </div>\n  {% endfor %}\n</div>\n{% else %}\n<div class="bg-white rounded-xl border border-gray-200 shadow-sm p-16 text-center">\n  <div class="text-5xl mb-4 opacity-20">🔌</div>\n  <p class="text-sm font-medium text-gray-500 mb-1">No connectors installed</p>\n  <p class="text-xs text-gray-400 mb-6">Generate a connector from an OpenAPI spec to get started.</p>\n  <div class="bg-gray-50 rounded-lg border border-gray-200 px-4 py-3 text-left max-w-sm mx-auto">\n    <p class="text-xs font-mono text-gray-600">apg connector generate --spec openapi.yaml</p>\n  </div>\n</div>\n{% endif %}\n', 'workflow_list.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Workflows</span></p>\n  <div class="flex items-center justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">Workflows</h1>\n      <p class="text-sm text-gray-500 mt-1">{{ total }} guided workflows across {{ entity_count }} entities</p>\n    </div>\n  </div>\n</section>\n\n{% if workflows %}\n<section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">\n  {% for workflow in workflows %}\n  <article class="apg-card group">\n    <div class="flex items-start gap-3 mb-3">\n      <span class="text-2xl" aria-hidden="true">{{ workflow.icon }}</span>\n      <div class="min-w-0">\n        <h2 class="font-semibold text-gray-900 group-hover:text-blue-600 text-sm mb-1">{{ workflow.name }}</h2>\n        <p class="text-xs text-gray-400">{{ workflow.entity }} · {{ workflow.step_count }} steps</p>\n      </div>\n    </div>\n    <p class="text-xs text-gray-500 leading-relaxed mb-4">{{ workflow.description }}</p>\n    <div class="flex items-center gap-1 mb-4" aria-hidden="true">\n      {% for step in workflow.steps %}\n      <div class="h-1.5 flex-1 rounded-full {{ \'bg-blue-500\' if loop.first else \'bg-gray-100\' }}"></div>\n      {% endfor %}\n    </div>\n    <a href="{{ workflow.href }}" class="apg-btn">Start</a>\n  </article>\n  {% endfor %}\n</section>\n{% else %}\n<section class="apg-card text-center py-10">\n  <h2 class="text-base font-semibold text-gray-900 mb-2">No workflows available</h2>\n  <p class="text-sm text-gray-500">Declare entities with fields to generate guided workflows.</p>\n</section>\n{% endif %}\n', 'debug_console.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Debug</span></p>\n  <h1 class="text-xl font-bold text-gray-900">Flow Debugger</h1>\n  <p class="text-sm text-gray-500 mt-1">Workflow runs, event journals, and circuit breaker state.</p>\n</section>\n\n{% if selected_run %}\n<article class="apg-card">\n  <header class="apg-card-header">\n    <h2 class="text-base font-semibold text-gray-900">Run: {{ selected_run.id }}</h2>\n    <span class="apg-badge apg-badge-neutral">{{ selected_run.workflow }}</span>\n  </header>\n  {% if selected_run.trace %}\n  <ol class="space-y-3">\n    {% for step in selected_run.trace %}\n    <li class="border border-gray-200 rounded-lg p-3">\n      <div class="flex items-center justify-between gap-3">\n        <p class="text-sm font-medium text-gray-900">{{ step.step }}</p>\n        <span class="apg-badge {{ step.badge_class }}">{{ step.status }}</span>\n      </div>\n      <p class="text-xs text-gray-500 mt-1">Step {{ step.index }}{% if step.notes %} · {{ step.notes }}{% endif %}</p>\n    </li>\n    {% endfor %}\n  </ol>\n  {% else %}\n  <p class="text-sm text-gray-500">No steps recorded for this run.</p>\n  {% endif %}\n</article>\n{% endif %}\n\n<section class="grid grid-cols-1 lg:grid-cols-2 gap-4">\n  <article class="apg-card">\n    <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Recent Runs</h2></header>\n    {% if runs %}\n    <div class="apg-table-wrap">\n      <table class="apg-table">\n        <thead><tr><th>Run</th><th>Workflow</th><th>Status</th><th>Steps</th></tr></thead>\n        <tbody>\n          {% for run in runs %}\n          <tr>\n            <td><a href="/ui/debug/{{ run.id }}" class="font-mono hover:underline">{{ run.id }}</a></td>\n            <td>{{ run.workflow }}</td>\n            <td><span class="apg-badge {{ run.badge_class }}">{{ run.status }}</span></td>\n            <td>{{ run.step_count }}</td>\n          </tr>\n          {% endfor %}\n        </tbody>\n      </table>\n    </div>\n    {% else %}\n    <p class="text-sm text-gray-500">No workflow runs yet.</p>\n    {% endif %}\n  </article>\n\n  <article class="apg-card">\n    <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Circuit Breakers</h2></header>\n    {% if circuit_breakers %}\n    <dl class="space-y-3">\n      {% for item in circuit_breakers %}\n      <div class="flex items-center justify-between gap-3 border-b border-gray-100 pb-3">\n        <dt class="font-mono text-xs text-gray-600">{{ item.key }}</dt>\n        <dd><span class="apg-badge {{ item.badge_class }}">{{ item.state }}</span> <span class="text-xs text-gray-400">{{ item.failures }} failures</span></dd>\n      </div>\n      {% endfor %}\n    </dl>\n    {% else %}\n    <p class="text-sm text-gray-500">No circuit breakers tripped.</p>\n    {% endif %}\n  </article>\n</section>\n\n<section class="apg-card">\n  <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Event Subscriptions</h2></header>\n  {% if subscriptions %}\n  <ul class="space-y-2">\n    {% for item in subscriptions %}\n    <li class="text-sm text-gray-600"><span class="font-mono">{{ item.event }}</span> → {{ item.workflows }}</li>\n    {% endfor %}\n  </ul>\n  {% else %}\n  <p class="text-sm text-gray-500">No event subscriptions declared.</p>\n  {% endif %}\n</section>\n', 'app_index.html.j2': '<!--- app_index.html.j2 — APG application home page --->\n{# Variables: module_name, module_description, entities, capabilities, databases,\n              application_routes, ui_routes, agents, agent_teams #}\n<div class="mb-6">\n  <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">{{ module_name }}</h1>\n  <p class="text-gray-500 mt-1">{{ module_description or \'Generated APG application\' }}</p>\n</div>\n\n{# Quick nav #}\n<nav class="flex flex-wrap gap-2 mb-8 text-sm" aria-label="API navigation">\n  <a href="/ui/workflows"\n     class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-apg-primary text-white hover:opacity-90 transition-opacity font-medium">\n    ⚡ Workflows\n  </a>\n  {% for link in api_links %}\n  <a href="{{ link.url }}"\n     class="inline-flex items-center px-3 py-1.5 rounded border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-apg-primary transition-colors">\n    {{ link.label }}\n  </a>\n  {% endfor %}\n</nav>\n\n{# Stats row #}\n<div class="apg-grid-4 mb-8">\n  {% for stat in dashboard_stats[:4] %}\n  <article class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ stat.value }}</span>\n      <span class="apg-stat-label">{{ stat.label }}</span>\n      <span class="apg-stat-delta">{{ stat.delta }}</span>\n    </div>\n    <div class="apg-chart mt-3" data-apg-chart="{{ stat.chart_id }}"></div>\n    <script id="{{ stat.chart_id }}" type="application/json">{{ stat.spec_json | safe }}</script>\n  </article>\n  {% endfor %}\n  <article class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ capabilities | length }}</span>\n      <span class="apg-stat-label">Capabilities</span>\n    </div>\n  </article>\n  <article class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ workflow_summary.workflow_count }}</span>\n      <span class="apg-stat-label">Workflows</span>\n    </div>\n  </article>\n</div>\n\n{% if status_charts %}\n<section class="apg-grid-2 gap-6 mb-8">\n  {% for chart in status_charts %}\n  <article class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">{{ chart.entity }} by {{ chart.field }}</h2>\n      <a class="text-xs hover:underline" href="/ui/entities/{{ chart.entity | urlencode }}?view=analytics">Analytics</a>\n    </div>\n    <div class="apg-chart" data-apg-chart="{{ chart.chart_id }}"></div>\n    <script id="{{ chart.chart_id }}" type="application/json">{{ chart.spec_json | safe }}</script>\n  </article>\n  {% endfor %}\n</section>\n{% endif %}\n\n<section class="apg-grid-3 gap-6 mb-8">\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Recent Activity</h2>\n    {% if recent_activity %}\n    <ul class="space-y-2">\n      {% for event in recent_activity %}\n      <li class="text-xs text-gray-500">{{ event.get(\'type\', \'event\') }} · {{ event.get(\'entity\', \'\') }}</li>\n      {% endfor %}\n    </ul>\n    {% else %}\n    <div class="apg-chart-empty"><p>No activity yet</p></div>\n    {% endif %}\n  </article>\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Workflow Summary</h2>\n    <p class="text-sm text-gray-500">{{ workflow_summary.workflow_count }} workflow(s), {{ workflow_summary.run_count }} run(s)</p>\n  </article>\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Agent Summary</h2>\n    <p class="text-sm text-gray-500">{{ agent_summary.agent_count }} agent(s), {{ agent_summary.team_count }} team(s)</p>\n  </article>\n</section>\n\n<div class="apg-grid-2 gap-6">\n  {% if entities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Entities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for entity in entities %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="/ui/entities/{{ entity.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ entity.name }}\n        </a>\n        <span class="apg-badge apg-badge-neutral">{{ entity.type }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if capabilities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Capabilities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for cap in capabilities %}\n      <li class="py-2">\n        <a href="/ui/capabilities/{{ cap | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ cap }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if ui_routes %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Application Screens</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for route, screen in ui_routes.items() %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="{{ route }}" class="text-sm text-apg-primary hover:underline">{{ route }}</a>\n        <span class="text-xs text-gray-400">{{ screen.get(\'application\', \'\') }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if application_routes %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Application Routes</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for route, screen in application_routes.items() %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="{{ route }}" class="text-sm text-apg-primary hover:underline">{{ route }}</a>\n        <span class="text-xs text-gray-400">{{ screen.get(\'application\', \'\') }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if agents %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">AI Agents</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for agent_name in agents %}\n      <li class="py-2">\n        <a href="/ui/agents/{{ agent_name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ agent_name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if agent_teams %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">AI Agent Teams</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for team_name in agent_teams %}\n      <li class="py-2">\n        <a href="/ui/agent-teams/{{ team_name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ team_name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n</div>\n', 'widgets/breadcrumbs.html.j2': '<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap" aria-label="Breadcrumb">\n  {% for item in breadcrumbs %}\n    {% if item.href and not loop.last %}\n    <a href="{{ item.href }}" class="hover:text-apg-primary transition-colors">{{ item.label }}</a>\n    {% else %}\n    <span class="font-semibold text-gray-900" {% if loop.last %}aria-current="page"{% endif %}>{{ item.label }}</span>\n    {% endif %}\n    {% if not loop.last %}<span aria-hidden="true">/</span>{% endif %}\n  {% endfor %}\n</nav>\n', 'widgets/field_display.html.j2': '{# field_display.html.j2 — semantic field rendering for record detail\n   Included by record_detail.html.j2 for individual field value rendering.\n   Variables: field (dict), field_val (any), semantic (str)\n#}\n{% if semantic == \'email\' and field_val %}\n  <a href="mailto:{{ field_val }}" class="text-apg-primary hover:underline text-sm">{{ field_val }}</a>\n{% elif semantic == \'phone\' and field_val %}\n  <a href="tel:{{ field_val }}" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1">\n    <svg class="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z"/></svg>\n    {{ field_val }}\n  </a>\n{% elif semantic == \'url\' and field_val %}\n  <a href="{{ field_val }}" target="_blank" rel="noopener" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1 truncate max-w-xs">\n    <svg class="w-3 h-3 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/><path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/></svg>\n    {{ field_val | string | truncate(40) }}\n  </a>\n{% elif semantic == \'image_url\' and field_val %}\n  <img src="{{ field_val }}" alt="{{ field.name }}" class="w-12 h-12 rounded-lg object-cover border border-gray-100">\n{% elif semantic == \'currency\' and field_val %}\n  <span class="text-sm font-semibold text-gray-900 tabular-nums">{{ field_val | string | float | round(2) }}</span>\n{% elif semantic == \'percent\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="flex-1 bg-gray-100 rounded-full h-1.5 max-w-24">\n      <div class="bg-apg-primary h-1.5 rounded-full" style="width: {{ [field_val | float, 100] | min }}%"></div>\n    </div>\n    <span class="text-sm text-gray-700 tabular-nums">{{ field_val }}%</span>\n  </div>\n{% elif semantic == \'rating\' and field_val %}\n  <div class="flex items-center gap-0.5">\n    {% set stars = field_val | float | round | int %}\n    {% for i in range(5) %}\n    <svg class="w-4 h-4 {{ \'text-amber-400\' if i < stars else \'text-gray-200\' }}" viewBox="0 0 20 20" fill="currentColor">\n      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>\n    </svg>\n    {% endfor %}\n    <span class="text-xs text-gray-400 ml-1">{{ field_val }}/5</span>\n  </div>\n{% elif semantic == \'color\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="w-5 h-5 rounded-full border border-gray-200 flex-shrink-0" style="background-color: {{ field_val }}"></div>\n    <span class="text-sm text-gray-700 font-mono">{{ field_val }}</span>\n  </div>\n{% elif semantic == \'json\' and field_val %}\n  <details class="max-w-xs">\n    <summary class="text-xs text-apg-primary cursor-pointer hover:underline">View JSON</summary>\n    <pre class="mt-1 text-xs bg-gray-50 rounded-lg p-2 overflow-auto max-h-40 border border-gray-100">{{ field_val | string }}</pre>\n  </details>\n{% elif semantic == \'status\' and field_val %}\n  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold\n    {% if field_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-100 text-green-800\n    {% elif field_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n    {% elif field_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n    {% else %}bg-gray-100 text-gray-600{% endif %}">\n    {{ field_val }}\n  </span>\n{% elif semantic == \'boolean\' %}\n  {% if field_val | string | lower in [\'true\', \'1\', \'yes\'] %}\n  <span class="inline-flex items-center gap-1 text-green-600 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>Yes</span>\n  {% else %}\n  <span class="inline-flex items-center gap-1 text-gray-400 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>No</span>\n  {% endif %}\n{% else %}\n  {% if field_val is none or field_val == \'\' or field_val | string == \'None\' %}\n  <span class="text-gray-300 italic text-xs">—</span>\n  {% else %}\n  {{ field_val | string | truncate(200) }}\n  {% endif %}\n{% endif %}\n'}


def _optional_module(name: str) -> Optional[Any]:
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            package_import_failed = True
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _log_activity(entity_name: str, record_id: str, event_type: str, actor: str = "system", detail: str = "") -> None:
    key = f"{entity_name}:{record_id}"
    if key not in APG_ACTIVITY_LOG:
        APG_ACTIVITY_LOG[key] = []
    import datetime
    APG_ACTIVITY_LOG[key].append({
        "type": event_type,
        "actor": actor,
        "detail": detail,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
    })
    if len(APG_ACTIVITY_LOG[key]) > 50:
        APG_ACTIVITY_LOG[key] = APG_ACTIVITY_LOG[key][-50:]


def _get_activity(entity_name: str, record_id: str) -> list[Dict[str, Any]]:
    return list(reversed(APG_ACTIVITY_LOG.get(f"{entity_name}:{record_id}", [])))


AI_AGENTS = _optional_module("ai_agents")
APG_APPLICATIONS = _optional_module("apg_application")
APG_CAPABILITIES = _optional_module("apg_capabilities")

import hashlib as _hashlib


def _journal_append(run_id: str, event_type: str, step: str, data: Dict[str, Any]) -> None:
    import datetime
    if run_id not in WORKFLOW_EVENT_JOURNAL:
        WORKFLOW_EVENT_JOURNAL[run_id] = []
    prev_hash = WORKFLOW_EVENT_JOURNAL[run_id][-1]["hash"] if WORKFLOW_EVENT_JOURNAL[run_id] else "0" * 64
    entry = {
        "seq": len(WORKFLOW_EVENT_JOURNAL[run_id]),
        "run_id": run_id,
        "event_type": event_type,
        "step": step,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "data": data,
    }
    raw = f"{prev_hash}{entry['seq']}{entry['event_type']}{entry['step']}{entry['ts']}"
    entry["hash"] = _hashlib.sha256(raw.encode()).hexdigest()
    WORKFLOW_EVENT_JOURNAL[run_id].append(entry)
    if _APG_PG_URL:
        _pg_save_journal_entry(entry)


def _pg_save_journal_entry(entry: Dict[str, Any]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_workflow_journal ("
                "  id SERIAL PRIMARY KEY,"
                "  run_id TEXT NOT NULL,"
                "  seq INTEGER NOT NULL,"
                "  module_name TEXT NOT NULL,"
                "  event_type TEXT NOT NULL,"
                "  step TEXT NOT NULL,"
                "  ts TIMESTAMPTZ NOT NULL,"
                "  data TEXT NOT NULL,"
                "  hash TEXT NOT NULL,"
                "  UNIQUE(run_id, seq)"
                ")"
            )
            cur.execute(
                "INSERT INTO apg_workflow_journal (run_id, seq, module_name, event_type, step, ts, data, hash)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (
                    entry["run_id"], entry["seq"], MODULE_NAME,
                    entry["event_type"], entry["step"],
                    entry["ts"], json.dumps(entry.get("data", {}), default=str),
                    entry["hash"]
                )
            )
        conn.commit()
    except Exception:
        pass  # best-effort
    finally:
        conn.close()


def _get_journal(run_id: str) -> list[Dict[str, Any]]:
    return WORKFLOW_EVENT_JOURNAL.get(run_id, [])


def list_agents() -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        return AI_AGENTS.list_agents()
    return []


def list_agent_teams() -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        return AI_AGENTS.list_agent_teams()
    return []


def invoke_agent(name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "invoke_agent"):
        return AI_AGENTS.invoke_agent(name, payload)
    return {"agent": name, "status": "unavailable", "error": "agents_unavailable"}


def invoke_team(name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "invoke_team"):
        return AI_AGENTS.invoke_team(name, payload)
    return {"team": name, "status": "unavailable", "error": "agents_unavailable"}


def runtime_adapter_environment_keys(runtime: str, agent_name: str | None = None) -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "runtime_adapter_environment_keys"):
        return AI_AGENTS.runtime_adapter_environment_keys(runtime, agent_name)
    return []


def runtime_adapter_command_candidates(runtime: str) -> list[list[str]]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "runtime_adapter_command_candidates"):
        return AI_AGENTS.runtime_adapter_command_candidates(runtime)
    return []


def validate_agent_runtimes(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        return AI_AGENTS.validate_agent_runtimes(available_agent_runtimes)
    return {"errors": [], "warnings": []}


def list_capabilities() -> list[str]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities"):
        return APG_CAPABILITIES.list_capabilities()
    return []


def capability_health(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health"):
        return APG_CAPABILITIES.capability_health(capability_name)
    return {"capability": capability_name, "status": "unavailable", "healthy": False, "errors": ["capability_health_unavailable"], "warnings": []}


def capability_health_report() -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health_report"):
        return APG_CAPABILITIES.capability_health_report()
    return {"healthy": True, "errors": [], "warnings": [], "capabilities": {}}


def describe_capability(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capability"):
        return APG_CAPABILITIES.describe_capability(capability_name)
    return {"name": capability_name, "available": False, "error": "capabilities_unavailable"}


def describe_capabilities() -> Dict[str, Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities"):
        return APG_CAPABILITIES.describe_capabilities()
    return {}


def capability_rules(capability_name: str) -> list[Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_rules"):
        return APG_CAPABILITIES.capability_rules(capability_name)
    return []


def evaluate_capability_rules(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "evaluate_capability_rules"):
        return APG_CAPABILITIES.evaluate_capability_rules(capability_name, context or {})
    return {"decision": "allow", "matched_rules": [], "actions": [], "context": context or {}, "warning": "capability_rules_unavailable"}


def capability_configuration(capability_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_configuration"):
        return APG_CAPABILITIES.capability_configuration(capability_name, overrides)
    return dict(overrides or {})


def validate_capability_configuration(
    capability_name: str,
    configuration: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "validate_capability_configuration"):
        return APG_CAPABILITIES.validate_capability_configuration(capability_name, configuration)
    return {"errors": ["capability_configuration_unavailable"], "warnings": []}


def approval_plan(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "approval_plan"):
        return APG_CAPABILITIES.approval_plan(capability_name, context or {})
    return {"capability": capability_name, "required": False, "approvers": [], "context": context or {}}


def capability_theme(capability_name: str, tenant_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_theme"):
        return APG_CAPABILITIES.capability_theme(capability_name, tenant_overrides)
    return {"name": capability_name, "tokens": dict(tenant_overrides or {})}


def theme_token(capability_name: str, token: str, default: Any = None) -> Any:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "theme_token"):
        return APG_CAPABILITIES.theme_token(capability_name, token, default)
    return capability_theme(capability_name).get("tokens", {}).get(token, default)


def capability_languages(capability_name: str) -> list[str]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_languages"):
        return APG_CAPABILITIES.capability_languages(capability_name)
    return []


def capability_screens(capability_name: str) -> list[Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_screens"):
        return APG_CAPABILITIES.capability_screens(capability_name)
    return []


def capability_streaming(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_streaming"):
        return APG_CAPABILITIES.capability_streaming(capability_name)
    return {}


def list_entities() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES]


def list_databases() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES if entity.get("type") == "database"]


def list_workflows() -> list[str]:
    names = {
        str(entity["name"])
        for entity in ENTITIES
        if entity.get("type") in {"workflow", "flow"}
    }
    names.update(str(name) for name in SEMANTIC_MODEL.get("flows", {}))
    return sorted(names)


def _workflow_entity(workflow_name: str) -> Dict[str, Any] | None:
    for entity in ENTITIES:
        if entity.get("type") in {"workflow", "flow"} and str(entity.get("name")) == workflow_name:
            return dict(entity)
    return None


def _workflow_defaults(entity: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for field in entity.get("fields", []):
        if isinstance(field, dict) and "default" in field:
            defaults[str(field.get("name"))] = field.get("default")
    return defaults


def _split_workflow_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    delimiter = "->" if "->" in text else ","
    parts: list[str] = []
    for part in text.split(delimiter):
        item = part.strip()
        if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
            item = item[1:-1].strip()
        if item:
            parts.append(item)
    return parts


def _workflow_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, list):
        mapping: Dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                step = item.get("step") or item.get("name") or item.get("from")
                if step not in (None, ""):
                    mapping[str(step)] = dict(item)
            elif isinstance(item, str):
                mapping.update(_workflow_mapping(item))
        return mapping
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return {str(key): item for key, item in loaded.items()}
    if isinstance(loaded, list):
        return _workflow_mapping(loaded)
    mapping: Dict[str, Any] = {}
    for item in text.split(";"):
        part = item.strip()
        if not part:
            continue
        separator = ":" if ":" in part else "=" if "=" in part else None
        if separator is None:
            continue
        key, raw_value = part.split(separator, 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key:
            mapping[key] = raw_value
    return mapping


def _workflow_step_metadata(workflow: Dict[str, Any], step: str) -> Dict[str, Any]:
    step = str(step)
    metadata: Dict[str, Any] = {}
    guards = workflow.get("guards", {})
    assignments = workflow.get("assignments", {})
    timers = workflow.get("timers", {})
    waits = workflow.get("waits", {})
    retry_policy = workflow.get("retry_policy", {})
    compensation = workflow.get("compensation", {})
    human_tasks = set(str(item) for item in workflow.get("human_tasks", []))
    if step in guards:
        metadata["guard"] = guards[step]
    if step in assignments:
        metadata["assignee"] = assignments[step]
        metadata["task_type"] = "human"
    elif step in human_tasks:
        metadata["task_type"] = "human"
    if step in timers:
        metadata["timer"] = timers[step]
    if step in waits:
        metadata["wait_for"] = waits[step]
    if step in retry_policy:
        metadata["retry_policy"] = retry_policy[step]
    if step in compensation:
        metadata["compensation"] = compensation[step]
    return metadata


def _compensation_actions(workflow: Dict[str, Any], completed_steps: list[str]) -> list[Dict[str, Any]]:
    compensation = workflow.get("compensation", {})
    actions: list[Dict[str, Any]] = []
    if not isinstance(compensation, dict):
        return actions
    for step in reversed(completed_steps):
        if step in compensation:
            actions.append({"step": step, "action": compensation[step]})
    return actions


def _retry_limit(policy: Any) -> int:
    if isinstance(policy, dict):
        for key in ("attempts", "max_attempts", "retries", "limit"):
            if key in policy:
                return _retry_limit(policy[key])
        return 1
    try:
        parsed = int(policy)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def _step_failure_budget(step: str, payload: Dict[str, Any]) -> int:
    failures = payload.get("step_failures", payload.get("failures", {}))
    if isinstance(failures, dict) and step in failures:
        try:
            return max(0, int(failures[step]))
        except (TypeError, ValueError):
            return 0
    fail_steps = payload.get("fail_steps", [])
    if isinstance(fail_steps, str):
        fail_steps = [part.strip() for part in fail_steps.split(",") if part.strip()]
    if isinstance(fail_steps, list) and step in [str(item) for item in fail_steps]:
        return 999999
    return 0


def _available_workflow_events(payload: Dict[str, Any]) -> set[str]:
    raw_events = payload.get("events", payload.get("completed_events", payload.get("signals", [])))
    if isinstance(raw_events, str):
        return {part.strip() for part in raw_events.split(",") if part.strip()}
    if isinstance(raw_events, list):
        return {str(item) for item in raw_events}
    if isinstance(raw_events, dict):
        return {str(key) for key, value in raw_events.items() if value}
    return set()


def _context_value(path: str, context: Dict[str, Any]) -> Any:
    current: Any = context
    for part in str(path).split("."):
        key = part.strip()
        if not key:
            continue
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _literal_or_context(value: str, context: Dict[str, Any]) -> Any:
    text = str(value).strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        numeric_value = float(text) if "." in text else int(text)
    except ValueError:
        numeric_value = None
    if numeric_value is not None:
        return numeric_value
    context_value = _context_value(text, context)
    if context_value is not None:
        return context_value
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def _compare_values(left: Any, operator: str, right: Any) -> bool:
    if operator in {"in", "not in"}:
        if isinstance(right, str):
            candidates = [part.strip() for part in right.split(",") if part.strip()]
        else:
            candidates = right
        try:
            result = left in candidates
        except TypeError:
            result = False
        return not result if operator == "not in" else result
    if operator == "contains":
        try:
            return right in left
        except TypeError:
            return False
    if operator in {"==", "!="}:
        result = left == right
        return not result if operator == "!=" else result
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        left_value = str(left)
        right_value = str(right)
    if operator == ">=":
        return left_value >= right_value
    if operator == "<=":
        return left_value <= right_value
    if operator == ">":
        return left_value > right_value
    if operator == "<":
        return left_value < right_value
    return False


def _evaluate_workflow_condition(condition: Any, context: Dict[str, Any]) -> bool:
    if condition in (None, ""):
        return True
    if isinstance(condition, bool):
        return condition
    text = str(condition).strip()
    lowered = text.lower()
    if lowered in {"always", "true", "allow"}:
        return True
    if lowered in {"never", "false", "deny"}:
        return False
    if " or " in lowered:
        return any(_evaluate_workflow_condition(part, context) for part in text.split(" or "))
    if " and " in lowered:
        return all(_evaluate_workflow_condition(part, context) for part in text.split(" and "))
    if lowered.endswith(" present"):
        field = text[: -len(" present")].strip()
        return _context_value(field, context) is not None
    if lowered.endswith(" missing"):
        field = text[: -len(" missing")].strip()
        return _context_value(field, context) is None
    for operator in (" not in ", " contains ", ">=", "<=", "==", "!=", ">", "<", " in "):
        if operator in text:
            left_text, right_text = text.split(operator, 1)
            normalized_operator = operator.strip()
            left = _context_value(left_text.strip(), context)
            right = _literal_or_context(right_text, context)
            return _compare_values(left, normalized_operator, right)
    return bool(_context_value(text, context))


def describe_workflow(workflow_name: str) -> Dict[str, Any]:
    flows = SEMANTIC_MODEL.get("flows", {})
    flow = dict(flows.get(workflow_name, {})) if isinstance(flows, dict) else {}
    entity = _workflow_entity(workflow_name) or {"name": workflow_name, "type": flow.get("type", "workflow"), "fields": [], "methods": []}
    defaults = _workflow_defaults(entity)
    steps = _split_workflow_sequence(defaults.get("steps") or flow.get("steps"))
    stages = _split_workflow_sequence(defaults.get("stages") or flow.get("stages"))
    guards = _workflow_mapping(defaults.get("guards") or flow.get("guards") or defaults.get("guard_rules") or defaults.get("conditions"))
    assignments = _workflow_mapping(defaults.get("assignments") or flow.get("assignments") or defaults.get("assignees") or defaults.get("owners"))
    timers = _workflow_mapping(defaults.get("timers") or flow.get("timers") or defaults.get("sla") or defaults.get("deadlines"))
    waits = _workflow_mapping(defaults.get("waits") or flow.get("waits") or defaults.get("event_waits") or defaults.get("wait_for"))
    retry_policy = _workflow_mapping(defaults.get("retry_policy") or flow.get("retry_policy") or defaults.get("retries"))
    compensation = _workflow_mapping(defaults.get("compensation") or flow.get("compensation") or defaults.get("compensations"))
    human_tasks = _split_workflow_sequence(defaults.get("human_tasks") or flow.get("human_tasks") or defaults.get("manual_steps"))
    transitions = [
        {
            "from": steps[index],
            "to": steps[index + 1],
            **({"guard": guards.get(steps[index + 1])} if steps[index + 1] in guards else {}),
        }
        for index in range(max(0, len(steps) - 1))
    ]
    return {
        "name": workflow_name,
        "type": entity.get("type", flow.get("type", "workflow")),
        "properties": dict(flow.get("properties", {})),
        "defaults": defaults,
        "methods": list(entity.get("methods", flow.get("methods", []))),
        "steps": steps,
        "stages": stages,
        "guards": guards,
        "assignments": assignments,
        "human_tasks": human_tasks,
        "timers": timers,
        "waits": waits,
        "retry_policy": retry_policy,
        "compensation": compensation,
        "transitions": transitions,
    }


def describe_workflows() -> Dict[str, Dict[str, Any]]:
    return {
        workflow_name: describe_workflow(workflow_name)
        for workflow_name in list_workflows()
    }


def _trigger_saga_compensation(workflow: Dict[str, Any], completed_steps: list[str]) -> None:
    comp = workflow.get("compensation", {})
    if not isinstance(comp, dict):
        return
    for step in reversed(completed_steps):
        action = comp.get(step)
        if action:
            try:
                _record_event("saga.compensate", str(workflow.get("name", "workflow")), after={"step": step, "action": str(action)})
            except Exception:
                pass  # best-effort


def _execute_workflow_steps(
    workflow: Dict[str, Any],
    steps: list[str],
    start_index: int,
    payload: Dict[str, Any],
    pause_at: str | None = None,
    existing_trace: list[Dict[str, Any]] | None = None,
    existing_completed_steps: list[str] | None = None,
    run_id: str = "",
) -> Dict[str, Any]:
    selected_steps = steps[start_index:]
    if pause_at is not None and pause_at not in selected_steps:
        return {
            "status": "error",
            "error": "unknown_pause_step",
            "pause_at": pause_at,
            "steps": selected_steps,
            "payload": payload,
        }
    trace = list(existing_trace or [])
    completed_steps = list(existing_completed_steps or [])
    guards = workflow.get("guards", {})
    retry_policy = workflow.get("retry_policy", {})
    waits = workflow.get("waits", {})
    available_events = _available_workflow_events(payload)
    for offset, step in enumerate(selected_steps):
        index = start_index + offset
        entry: Dict[str, Any] = {
            "index": index,
            "step": step,
            **_workflow_step_metadata(workflow, step),
        }
        if run_id:
            _journal_append(run_id, "step_started", step, {})
        guard = guards.get(step)
        if guard is not None:
            guard_passed = _evaluate_workflow_condition(guard, payload)
            entry["guard"] = guard
            entry["guard_passed"] = guard_passed
            if not guard_passed:
                entry["status"] = "blocked"
                trace.append(entry)
                return {
                    "status": "blocked",
                    "current_step": step,
                    "completed_at": None,
                    "steps": selected_steps,
                    "completed_steps": completed_steps,
                    "pending_steps": selected_steps[offset:],
                    "trace": trace,
                    "payload": payload,
                    "blocked_at": step,
                    "blocked_reason": "guard_failed",
                    "guard": guard,
                    "compensations": _compensation_actions(workflow, completed_steps),
                }
        wait_for = waits.get(step)
        if wait_for is not None:
            event_name = str(wait_for)
            entry["wait_for"] = event_name
            if event_name not in available_events:
                entry["status"] = "waiting"
                trace.append(entry)
                return {
                    "status": "waiting",
                    "current_step": step,
                    "completed_at": None,
                    "steps": selected_steps,
                    "completed_steps": completed_steps,
                    "pending_steps": selected_steps[offset:],
                    "trace": trace,
                    "payload": payload,
                    "waiting_at": step,
                    "waiting_for": event_name,
                    "compensations": [],
                }
            entry["event_received"] = event_name
        failure_budget = _step_failure_budget(step, payload)
        retry_limit = _retry_limit(retry_policy.get(step)) if isinstance(retry_policy, dict) and step in retry_policy else 1
        # Circuit breaker: fail fast if open
        cb_k = _cb_key(workflow.get("name", "wf"), step)
        # Check workflow-level circuit_breaker config for this step
        wf_circuit_breakers = workflow.get("circuit_breakers", {})
        step_cb_spec = wf_circuit_breakers.get(step, {}) if isinstance(wf_circuit_breakers, dict) else {}
        _raw_step_policy = retry_policy.get(step) if isinstance(retry_policy, dict) else None
        step_policy = _raw_step_policy if isinstance(_raw_step_policy, dict) else {}
        cb_threshold = int(step_cb_spec.get("threshold", step_policy.get("circuit_threshold", 5)) if isinstance(step_cb_spec, dict) else step_policy.get("circuit_threshold", 5))
        cb_reset = int(step_cb_spec.get("reset_timeout", step_policy.get("reset_timeout", 60)) if isinstance(step_cb_spec, dict) else step_policy.get("reset_timeout", 60))
        if _cb_is_open(cb_k, cb_threshold, cb_reset):
            entry["status"] = "circuit_open"
            trace.append(entry)
            return {
                "status": "failed",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset:],
                "trace": trace,
                "payload": payload,
                "failed_at": step,
                "failure_reason": "circuit_open",
                "compensations": _compensation_actions(workflow, completed_steps),
            }
        # Step timeout metadata (from timers dict)
        timers = workflow.get("timers", {})
        if isinstance(timers, dict) and step in timers:
            entry["timeout_spec"] = timers[step]
        attempts: list[Dict[str, Any]] = []
        for attempt_number in range(1, retry_limit + 1):
            failed = failure_budget >= attempt_number
            attempts.append({
                "attempt": attempt_number,
                "status": "failed" if failed else "completed",
            })
            if not failed:
                break
        entry["attempts"] = attempts
        if attempts and attempts[-1]["status"] == "failed":
            _cb_fail(cb_k, cb_threshold, cb_reset)
            # Saga: auto-trigger compensation for completed steps
            is_saga = bool(workflow.get("is_saga", False))
            if is_saga and completed_steps:
                _trigger_saga_compensation(workflow, completed_steps)
                if run_id:
                    comp = workflow.get("compensation", {})
                    comp_action = str(comp.get(step, "")) if isinstance(comp, dict) else ""
                    _journal_append(run_id, "saga_compensating", step, {"compensation": comp_action})
            if run_id:
                _journal_append(run_id, "step_failed", step, {"error": "step_failed_after_retries"})
            entry["status"] = "failed"
            trace.append(entry)
            return {
                "status": "failed",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset:],
                "trace": trace,
                "payload": payload,
                "failed_at": step,
                "failure_reason": "step_failed",
                "attempts": attempts,
                "compensations": _compensation_actions(workflow, completed_steps),
            }
        _cb_success(cb_k)
        entry["status"] = "completed"
        trace.append(entry)
        completed_steps.append(step)
        if run_id:
            _journal_append(run_id, "step_completed", step, {"attempts": len(attempts)})
        if pause_at == step and offset < len(selected_steps) - 1:
            return {
                "status": "paused",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset + 1:],
                "trace": trace,
                "payload": payload,
                "compensations": [],
            }
    return {
        "status": "completed",
        "current_step": selected_steps[-1],
        "completed_at": selected_steps[-1],
        "steps": selected_steps,
        "completed_steps": completed_steps,
        "pending_steps": [],
        "trace": trace,
        "payload": payload,
        "compensations": [],
    }


def run_workflow(workflow_name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global NEXT_WORKFLOW_RUN_ID
    if workflow_name not in list_workflows():
        raise KeyError(workflow_name)
    payload = dict(payload or {})
    workflow = describe_workflow(workflow_name)
    steps = list(workflow.get("steps", []))
    if not steps:
        steps = list(workflow.get("stages", []))
    if not steps:
        steps = ["start", "complete"]
    start_at = str(payload.get("start_at") or steps[0])
    if start_at not in steps:
        return {
            "workflow": workflow_name,
            "status": "error",
            "error": "unknown_start_step",
            "start_at": start_at,
            "steps": steps,
            "payload": payload,
        }
    start_index = steps.index(start_at)
    selected_steps = steps[start_index:]
    pause_at = payload.get("pause_at", payload.get("stop_after"))
    pause_at = str(pause_at) if pause_at is not None else None
    run_id = f"workflow-run-{NEXT_WORKFLOW_RUN_ID}"
    NEXT_WORKFLOW_RUN_ID += 1
    execution = _execute_workflow_steps(workflow, steps, start_index, payload, pause_at, run_id=run_id)
    if execution.get("status") == "error":
        return {
            "workflow": workflow_name,
            **execution,
        }
    result = {
        "id": run_id,
        "workflow": workflow_name,
        "started_at": start_at,
        **execution,
    }
    event = _record_event("workflow.run", workflow_name, after=result)
    result["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(result)
    # PostgreSQL persistence for durable workflows
    if _APG_PG_URL:
        _pg_save_workflow_run(result)
    persistence_error = _persist_record_store()
    if persistence_error:
        result["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(result)
    # Emit declared completion events
    emit_events = workflow.get("emit_events") or workflow.get("events", {}).get("emit", [])
    if isinstance(emit_events, str):
        emit_events = [emit_events]
    for ev_name in (emit_events or []):
        try:
            emit_apg_event(str(ev_name), {"workflow": workflow_name, "run_id": run_id, "status": execution.get("status")})
        except Exception:
            pass  # best-effort
    # Register subscriptions declared on this workflow
    subscribe_events = workflow.get("subscribe_events") or workflow.get("events", {}).get("subscribe", [])
    if isinstance(subscribe_events, str):
        subscribe_events = [subscribe_events]
    for ev_name in (subscribe_events or []):
        _subscribe_workflow_event(str(ev_name), workflow_name)
    return dict(result)


def list_workflow_runs(workflow_name: str | None = None) -> list[Dict[str, Any]]:
    runs = [dict(run) for run in WORKFLOW_RUNS.values()]
    if workflow_name is not None:
        runs = [run for run in runs if run.get("workflow") == workflow_name]
    return runs


def get_workflow_run(run_id: str) -> Dict[str, Any]:
    run = WORKFLOW_RUNS.get(str(run_id))
    if run is None:
        raise KeyError(run_id)
    return dict(run)


def resume_workflow(run_id: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    run_id = str(run_id)
    existing = WORKFLOW_RUNS.get(run_id)
    if existing is None:
        raise KeyError(run_id)
    if existing.get("status") == "completed":
        result = dict(existing)
        result["resumed"] = False
        return result
    workflow_name = str(existing.get("workflow"))
    if workflow_name not in list_workflows():
        raise KeyError(workflow_name)
    payload_update = dict(payload or {})
    merged_payload = dict(existing.get("payload", {}))
    merged_payload.update(payload_update)
    steps = list(existing.get("steps") or describe_workflow(workflow_name).get("steps", []))
    if not steps:
        steps = ["start", "complete"]
    current_step = str(existing.get("current_step") or existing.get("started_at") or steps[0])
    if current_step in steps:
        start_index = steps.index(current_step) + 1
    else:
        start_index = 0
    if start_index >= len(steps):
        existing["status"] = "completed"
        existing["completed_at"] = steps[-1]
        existing["pending_steps"] = []
        WORKFLOW_RUNS[run_id] = dict(existing)
        return dict(existing)

    selected_steps = steps[start_index:]
    pause_at = payload_update.get("pause_at", payload_update.get("stop_after"))
    pause_at = str(pause_at) if pause_at is not None else None
    workflow = describe_workflow(workflow_name)
    execution = _execute_workflow_steps(
        workflow,
        steps,
        start_index,
        merged_payload,
        pause_at,
        existing_trace=list(existing.get("trace", [])),
        existing_completed_steps=list(existing.get("completed_steps", [])),
        run_id=run_id,
    )
    if execution.get("status") == "error":
        return {
            "id": run_id,
            "workflow": workflow_name,
            **execution,
        }
    updated = dict(existing)
    updated.update({
        **execution,
        "resumed": True,
    })
    event = _record_event("workflow.resume", workflow_name, before=existing, after=updated)
    updated["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(updated)
    persistence_error = _persist_record_store()
    if persistence_error:
        updated["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(updated)
    return dict(updated)


def execute_workflow_compensations(
    run_id: str,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_id = str(run_id)
    existing = WORKFLOW_RUNS.get(run_id)
    if existing is None:
        raise KeyError(run_id)
    payload = dict(payload or {})
    actions = [
        dict(action)
        for action in existing.get("compensations", [])
        if isinstance(action, dict)
    ]
    if existing.get("compensation_status") == "completed":
        return {
            "id": run_id,
            "workflow": existing.get("workflow"),
            "status": "completed",
            "already_executed": True,
            "actions": existing.get("compensation_results", []),
            "run": dict(existing),
        }
    results: list[Dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        result = dict(action)
        result.update({
            "index": index,
            "status": "completed",
            "mode": "generated",
        })
        if payload:
            result["payload"] = dict(payload)
        results.append(result)
    updated = dict(existing)
    updated.update({
        "compensation_status": "completed" if actions else "skipped",
        "compensation_results": results,
    })
    event = _record_event("workflow.compensate", str(existing.get("workflow")), before=existing, after=updated)
    updated["compensation_event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(updated)
    persistence_error = _persist_record_store()
    if persistence_error:
        updated["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(updated)
    return {
        "id": run_id,
        "workflow": updated.get("workflow"),
        "status": updated["compensation_status"],
        "already_executed": False,
        "actions": results,
        "event_id": event["id"],
        "run": dict(updated),
    }


import threading as _apg_threading
_CB_LOCK = _apg_threading.Lock()
_ES_LOCK = _apg_threading.Lock()
try:
    import jwt as _jwt_lib
except ImportError:
    _jwt_lib = None


def _cb_key(workflow_name: str, step: str) -> str:
    return f"{workflow_name}:{step}"


def _cb_is_open(key: str, threshold: int = 5, reset: int = 60) -> bool:
    import time as _t
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.get(key)
        if cb is None:
            return False
        if cb["state"] == "open":
            if _t.time() - cb.get("opened_at", 0.0) > reset:
                cb["state"] = "half_open"
                return False
            return True
        return False


def _cb_fail(key: str, threshold: int = 5, reset: int = 60) -> None:
    import time as _t
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.setdefault(key, {"state": "closed", "failures": 0, "opened_at": 0.0})
        cb["failures"] += 1
        if cb["failures"] >= threshold:
            cb["state"] = "open"
            cb["opened_at"] = _t.time()


def _cb_success(key: str) -> None:
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.get(key)
        if cb:
            cb.update({"state": "closed", "failures": 0, "opened_at": 0.0})


def circuit_breaker_status() -> Dict[str, Any]:
    with _CB_LOCK:
        return {k: dict(v) for k, v in CIRCUIT_BREAKERS.items()}


_TENANT_LOCAL = _apg_threading.local()


def _tenant_id() -> str | None:
    return getattr(_TENANT_LOCAL, "tenant_id", None)


def _subscribe_workflow_event(event_name: str, workflow_name: str) -> None:
    with _ES_LOCK:
        APG_EVENT_SUBSCRIPTIONS.setdefault(event_name, [])
        if workflow_name not in APG_EVENT_SUBSCRIPTIONS[event_name]:
            APG_EVENT_SUBSCRIPTIONS[event_name].append(workflow_name)


def emit_apg_event(event_name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global NEXT_EVENT_ID
    import time as _t
    ev: Dict[str, Any] = {
        "id": NEXT_EVENT_ID,
        "name": event_name,
        "payload": payload or {},
        "ts": _t.time(),
        "triggered": [],
    }
    with _ES_LOCK:
        NEXT_EVENT_ID += 1
        EVENT_LOG.append(ev)
    subs = list(APG_EVENT_SUBSCRIPTIONS.get(event_name, []))
    for wf_name in subs:
        try:
            run_workflow(wf_name, {"trigger_event": event_name, **(payload or {})})
            ev["triggered"].append(wf_name)
        except Exception:
            pass  # best-effort
    return dict(ev)


def semantic_model() -> Dict[str, Any]:
    return json.loads(json.dumps(SEMANTIC_MODEL))


def database_status() -> Dict[str, Any]:
    databases = list_databases()
    schema_count = sum(len(database.get("schemas", [])) for database in databases)
    table_count = sum(
        len(schema.get("tables", []))
        for database in databases
        for schema in database.get("schemas", [])
    )
    reference_count = sum(
        1
        for database in databases
        for schema in database.get("schemas", [])
        for table in schema.get("tables", [])
        for column in table.get("columns", [])
        if isinstance(column, dict) and isinstance(column.get("reference"), dict)
    )
    validation = validate_database_schema_contracts()
    return {
        "valid": not validation["errors"],
        "database_count": len(databases),
        "schema_count": schema_count,
        "table_count": table_count,
        "reference_count": reference_count,
        "validation": validation,
    }


def list_records(entity_name: str | None = None) -> Dict[str, list[Dict[str, Any]]] | list[Dict[str, Any]]:
    if entity_name is None:
        return {
            name: [dict(record) for record in records]
            for name, records in RECORD_STORE.items()
    }
    return [dict(record) for record in RECORD_STORE[entity_name]]


def query_records(entity_name: str, query: Dict[str, list[str]] | None = None) -> Dict[str, Any]:
    query = query or {}
    records = list_records(entity_name)
    filters = {
        key.removeprefix("filter."): values[-1]
        for key, values in query.items()
        if values and key not in {"limit", "offset", "sort", "order"}
    }
    # Tenant routing: auto-scope to current tenant when entity has tenant_id field
    tid = _tenant_id()
    if tid and entity_name in TENANT_SCOPED_ENTITIES and "tenant_id" not in filters:
        filters["tenant_id"] = tid
    records = [
        record
        for record in records
        if all(str(record.get(field, "")) == str(expected) for field, expected in filters.items())
    ]
    sort_field = query.get("sort", [None])[-1]
    if sort_field:
        reverse = query.get("order", ["asc"])[-1].lower() == "desc"
        records = sorted(records, key=lambda record: str(record.get(sort_field, "")), reverse=reverse)
    total = len(records)
    try:
        offset = max(0, int(query.get("offset", ["0"])[-1]))
    except (TypeError, ValueError):
        offset = 0
    limit = query.get("limit", [None])[-1]
    try:
        parsed_limit = int(limit) if limit not in (None, "") else None
    except (TypeError, ValueError):
        parsed_limit = None
    if parsed_limit is not None:
        records = records[offset:offset + max(0, parsed_limit)]
    elif offset:
        records = records[offset:]
    return {
        "entity": entity_name,
        "records": records,
        "count": len(records),
        "total": total,
        "offset": offset,
        "limit": parsed_limit,
        "filters": filters,
        "sort": sort_field,
        "order": query.get("order", ["asc"])[-1],
    }


def get_record(entity_name: str, record_id: Any) -> tuple[int, Dict[str, Any]]:
    return _records_payload(f"/entities/{entity_name}/records/{record_id}")


def create_record(entity_name: str, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    return _create_record_payload(f"/entities/{entity_name}/records", {"record": record})


def update_record(
    entity_name: str,
    record_id: Any,
    record: Dict[str, Any],
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    payload: Dict[str, Any] = {"record": record}
    if expected_revision is not None:
        payload["expected_revision"] = expected_revision
    return _update_record_payload(f"/entities/{entity_name}/records/{record_id}", payload)


def delete_record(
    entity_name: str,
    record_id: Any,
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    path = f"/entities/{entity_name}/records/{record_id}"
    if expected_revision is not None:
        path = f"{path}?expected_revision={expected_revision}"
    return _delete_record_payload(path)


def _data_path() -> Path | None:
    raw_path = os.environ.get("APG_DATA_FILE") or os.environ.get("APG_DATA_PATH")
    if not raw_path:
        return None
    return Path(raw_path)


def _record_numeric_id(record: Dict[str, Any]) -> int | None:
    value = record.get("id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _sync_next_record_ids() -> None:
    for entity_name in ENTITY_NAMES:
        numeric_ids = [
            numeric_id
            for record in RECORD_STORE[entity_name]
            for numeric_id in [_record_numeric_id(record)]
            if numeric_id is not None
        ]
        NEXT_RECORD_IDS[entity_name] = max(numeric_ids, default=0) + 1


def _sync_next_event_id() -> None:
    global NEXT_EVENT_ID
    numeric_ids = [
        numeric_id
        for event in EVENT_LOG
        for numeric_id in [_record_numeric_id(event)]
        if numeric_id is not None
    ]
    NEXT_EVENT_ID = max(numeric_ids, default=0) + 1


def _workflow_run_numeric_id(run: Dict[str, Any]) -> int | None:
    value = run.get("id")
    if isinstance(value, str) and value.startswith("workflow-run-"):
        suffix = value.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    if isinstance(value, int):
        return value
    return None


def _sync_next_workflow_run_id() -> None:
    global NEXT_WORKFLOW_RUN_ID
    numeric_ids = [
        numeric_id
        for run in WORKFLOW_RUNS.values()
        for numeric_id in [_workflow_run_numeric_id(run)]
        if numeric_id is not None
    ]
    NEXT_WORKFLOW_RUN_ID = max(numeric_ids, default=0) + 1


def _load_record_store() -> None:
    path = _data_path()
    if path is None or not path.exists():
        return
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"APG could not load record data from {path}: {error}", file=sys.stderr)
        return
    if not isinstance(loaded, dict):
        return
    raw_records = loaded.get("records", loaded)
    if not isinstance(raw_records, dict):
        return
    for entity_name in ENTITY_NAMES:
        entity_records = raw_records.get(entity_name, [])
        if isinstance(entity_records, list):
            RECORD_STORE[entity_name] = [
                dict(record)
                for record in entity_records
                if isinstance(record, dict)
            ]
    raw_events = loaded.get("events", [])
    if isinstance(raw_events, list):
        EVENT_LOG.clear()
        EVENT_LOG.extend(dict(event) for event in raw_events if isinstance(event, dict))
    raw_workflow_runs = loaded.get("workflow_runs", {})
    if isinstance(raw_workflow_runs, list):
        raw_workflow_runs = {
            str(run.get("id")): run
            for run in raw_workflow_runs
            if isinstance(run, dict) and run.get("id") not in (None, "")
        }
    if isinstance(raw_workflow_runs, dict):
        WORKFLOW_RUNS.clear()
        for run_id, run in raw_workflow_runs.items():
            if isinstance(run, dict):
                normalized = dict(run)
                normalized.setdefault("id", str(run_id))
                WORKFLOW_RUNS[str(normalized["id"])] = normalized
    _sync_next_record_ids()
    _sync_next_event_id()
    _sync_next_workflow_run_id()
    # Merge from PostgreSQL if available
    if _APG_PG_URL:
        for run in _pg_load_workflow_runs():
            rid = str(run.get("id", ""))
            if rid and rid not in WORKFLOW_RUNS:
                WORKFLOW_RUNS[rid] = run
        for entity_name in list(RECORD_STORE.keys()):
            pg_records = _pg_load_entity_records(entity_name)
            if pg_records:
                RECORD_STORE[entity_name] = pg_records


def _persist_record_store() -> str | None:
    if _APG_PG_URL:
        for entity_name, records in list_records().items():
            _pg_save_entity_records(entity_name, records)
    path = _data_path()
    if path is None:
        return None
    payload = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "records": list_records(),
        "events": list_events(),
        "workflow_runs": {run_id: dict(run) for run_id, run in WORKFLOW_RUNS.items()},
        "next_record_ids": dict(NEXT_RECORD_IDS),
        "next_event_id": NEXT_EVENT_ID,
        "next_workflow_run_id": NEXT_WORKFLOW_RUN_ID,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError as error:
        return str(error)
    return None


def storage_status(include_records: bool = False) -> Dict[str, Any]:
    path = _data_path()
    status: Dict[str, Any] = {
        "mode": "file" if path is not None else "memory",
        "path": str(path) if path is not None else None,
    }
    if include_records:
        status["records"] = list_records()
        status["events"] = list_events()
        status["workflow_runs"] = list_workflow_runs()
    return status


def metrics_snapshot() -> Dict[str, Any]:
    record_counts = {
        entity_name: len(RECORD_STORE[entity_name])
        for entity_name in sorted(ENTITY_NAMES)
    }
    event_counts: Dict[str, int] = {}
    for event in EVENT_LOG:
        action = str(event.get("action", "unknown"))
        event_counts[action] = event_counts.get(action, 0) + 1
    return {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "entity_count": len(ENTITIES),
        "workflow_count": len(list_workflows()),
        "workflow_run_count": len(WORKFLOW_RUNS),
        "database_status": database_status(),
        "record_counts": record_counts,
        "total_records": sum(record_counts.values()),
        "event_count": len(EVENT_LOG),
        "event_counts": event_counts,
        "relationship_count": len(relationship_graph()["edges"]),
        "storage": storage_status(),
        "auth": auth_status(),
    }


def self_test() -> Dict[str, Any]:
    validation = validate_application()
    openapi = openapi_document()
    routes = sorted(openapi["paths"])
    metrics = metrics_snapshot()
    checks: Dict[str, Any] = {
        "validation": validation,
        "metrics": metrics,
        "route_count": len(routes),
        "entity_count": metrics["entity_count"],
    }
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health_report"):
        checks["capability_health"] = APG_CAPABILITIES.capability_health_report()
    return {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "passed": validation["valid"],
        "status": "ok" if validation["valid"] else "warning",
        "checks": checks,
        "routes": routes,
    }


def component_manifest() -> Dict[str, Any]:
    app = describe_application()
    openapi = openapi_document()
    return {
        "kind": "apg.application",
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "target": "python",
        "composable": True,
        "interfaces": {
            "http": {
                "openapi": "/openapi.json",
                "paths": sorted(openapi["paths"]),
            },
            "python": {
                "package": MODULE_NAME,
                "exports": [
                    "auth_status",
                    "approval_plan",
                    "capability_configuration",
                    "coerce_record_types",
                    "component_manifest",
                    "create_record",
                    "database_status",
                    "delete_record",
                    "describe_capabilities",
                    "describe_application",
                    "describe_capability",
                    "describe_workflow",
                    "describe_workflows",
                    "evaluate_capability_rules",
                    "execute_workflow_compensations",
                    "get_record",
                    "get_workflow_run",
                    "invoke_agent",
                    "invoke_team",
                    "list_agent_teams",
                    "list_agents",
                    "list_capabilities",
                    "list_databases",
                    "list_entities",
                    "list_events",
                    "list_records",
                    "list_workflow_runs",
                    "list_workflows",
                    "main",
                    "metrics_snapshot",
                    "openapi_document",
                    "query_records",
                    "relationship_graph",
                    "resume_workflow",
                    "run_workflow",
                    "runtime_adapter_command_candidates",
                    "runtime_adapter_environment_keys",
                    "self_test",
                    "semantic_model",
                    "storage_status",
                    "capability_health",
                    "capability_health_report",
                    "capability_languages",
                    "capability_rules",
                    "capability_screens",
                    "capability_streaming",
                    "capability_theme",
                    "theme_token",
                    "update_record",
                    "validate_agent_runtimes",
                    "validate_application",
                    "validate_capability_configuration",
                    "validate_component_manifest_contract",
                    "validate_openapi_contract",
                    "validate_route_dispatch_contract",
                    "validate_record",
                ],
            },
            "records": sorted(ENTITY_NAMES),
            "theme": "/theme.css",
            "semantic_model": "/semantic-model.json",
        },
        "entities": list_entities(),
        "databases": list_databases(),
        "workflows": describe_workflows(),
        "ai_agents": app.get("ai_agents", []),
        "ai_agent_teams": app.get("ai_agent_teams", []),
        "application_compositions": app.get("application_compositions", []),
        "application_dependency_graph": app.get("application_dependency_graph", {}),
        "application_routes": app.get("application_routes", {}),
        "capabilities": app.get("capabilities", []),
        "ui_routes": app.get("ui_routes", {}),
        "streaming_processors": app.get("streaming_processors", {}),
        "deployment": {
            "artifacts": [
                "app.py",
                "__init__.py",
                "README.md",
                "semantic_model.json",
                "requirements.txt",
                "Dockerfile",
                ".dockerignore",
                ".env.example",
                "smoke_test.py",
            ],
            "commands": {
                "run": "python app.py",
                "describe": "python app.py --describe",
                "semantic_model": "python app.py --semantic-model",
                "validate": "python app.py --validate",
                "self_test": "python app.py --self-test",
                "smoke_test": "python smoke_test.py",
            },
            "environment": ["APG_HOST", "APG_PORT", "APG_DATA_FILE", "APG_API_KEY", "APG_DEBUG"],
        },
    }


def auth_status() -> Dict[str, Any]:
    return {
        "mode": "api_key" if os.environ.get("APG_API_KEY") else "open",
        "header": "Authorization: Bearer <key> or X-APG-API-Key" if os.environ.get("APG_API_KEY") else None,
    }


def _authorized(headers: Any) -> bool:
    authorization = headers.get("Authorization", "")
    supplied_key = headers.get("X-APG-API-Key")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        jwt_secret = os.environ.get("APG_JWT_SECRET")
        jwt_pubkey = os.environ.get("APG_JWT_PUBLIC_KEY")
        if (jwt_secret or jwt_pubkey) and _jwt_lib is not None:
            try:
                key = jwt_pubkey or jwt_secret
                alg = "RS256" if jwt_pubkey else "HS256"
                _jwt_lib.decode(token, key, algorithms=[alg])
                return True
            except Exception:
                return False
        supplied_key = token
    required_key = os.environ.get("APG_API_KEY")
    if required_key:
        return supplied_key == required_key
    return True


def _auth_failure_payload() -> tuple[int, Dict[str, Any]]:
    return 401, {
        "error": "unauthorized",
        "message": "Set Authorization: Bearer <key> or X-APG-API-Key to mutate this APG app.",
    }


def list_events(entity_name: str | None = None) -> list[Dict[str, Any]]:
    events = [dict(event) for event in EVENT_LOG]
    if entity_name is None:
        return events
    return [event for event in events if event.get("entity") == entity_name]


def _record_event(
    action: str,
    entity_name: str,
    before: Dict[str, Any] | None = None,
    after: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    global NEXT_EVENT_ID
    record = after if after is not None else before if before is not None else {}
    event = {
        "id": NEXT_EVENT_ID,
        "action": action,
        "entity": entity_name,
        "record_id": record.get("id"),
    }
    if before is not None:
        event["before"] = dict(before)
    if after is not None:
        event["after"] = dict(after)
    NEXT_EVENT_ID += 1
    EVENT_LOG.append(event)
    return dict(event)


def _prepare_new_record(record: Dict[str, Any], entity_name: str = "") -> Dict[str, Any]:
    prepared = dict(record)
    prepared.setdefault("_revision", 1)
    # Auto-inject tenant_id for tenant-scoped entities
    tid = _tenant_id()
    if tid and entity_name in TENANT_SCOPED_ENTITIES:
        prepared.setdefault("tenant_id", tid)
    return prepared


def _expected_revision(payload: Dict[str, Any]) -> int | None:
    value = payload.get("expected_revision")
    if value is None and isinstance(payload.get("record"), dict):
        value = payload["record"].get("_revision")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _revision_conflict(existing: Dict[str, Any], expected_revision: int | None) -> Dict[str, Any] | None:
    current_revision = existing.get("_revision")
    if expected_revision is None or current_revision == expected_revision:
        return None
    return {
        "error": "revision_conflict",
        "expected_revision": expected_revision,
        "current_revision": current_revision,
        "record": dict(existing),
    }


def _record_schema(entity: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    fields = _field_specs(str(entity["name"]))
    if not fields:
        return {"type": "object", "additionalProperties": True}
    schema_properties: Dict[str, Any] = {
        "id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "_revision": {"type": "integer"},
    }
    required_fields: list[str] = []
    for field in fields:
        field_name = str(field["name"])
        schema_properties[field_name] = {"type": _json_schema_type(str(field.get("type", "any")))}
        if not partial and field.get("required", False):
            required_fields.append(field_name)
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": schema_properties,
    }
    if required_fields:
        schema["required"] = required_fields
    return schema


def _schema_ref(name: str) -> Dict[str, Any]:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_media(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"application/json": {"schema": schema}}


def _record_body_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "record": _schema_ref(schema_name),
        },
        "required": ["record"],
    }


def _record_import_body_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "records": {"type": "array", "items": _schema_ref(schema_name)},
        },
        "required": ["records"],
    }


def _record_list_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "records": {"type": "array", "items": _schema_ref(schema_name)},
            "count": {"type": "integer"},
            "total": {"type": "integer"},
            "filters": {"type": "object", "additionalProperties": {"type": "string"}},
            "sort": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "order": {"type": "string"},
        },
        "required": ["entity", "records", "count"],
    }


def _record_item_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "record": _schema_ref(schema_name),
        },
        "required": ["entity", "record"],
    }


def _record_mutation_response_schema(schema_name: str, record_key: str = "record") -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            record_key: _schema_ref(schema_name),
            "event": _schema_ref("EventRecord"),
        },
        "required": [record_key],
    }


def _record_export_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "records": {"type": "array", "items": _schema_ref(schema_name)},
            "count": {"type": "integer"},
        },
        "required": ["entity", "records", "count"],
    }


def _record_import_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "imported": {"type": "array", "items": _schema_ref(schema_name)},
            "errors": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "events": {"type": "array", "items": _schema_ref("EventRecord")},
            "count": {"type": "integer"},
            "failed": {"type": "integer"},
        },
        "required": ["entity", "imported", "errors", "count", "failed"],
    }


def _database_openapi_schemas() -> Dict[str, Any]:
    nullable_string = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    generic_object = {"type": "object", "additionalProperties": True}
    return {
        "ApplicationDescription": generic_object,
        "SemanticModel": generic_object,
        "ComponentManifest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "kind": {"const": "apg.application"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "target": {"const": "python"},
                "composable": {"type": "boolean"},
                "interfaces": generic_object,
                "entities": {"type": "array", "items": generic_object},
                "databases": {"type": "array", "items": _schema_ref("DatabaseCatalogEntry")},
                "deployment": generic_object,
            },
            "required": ["kind", "name", "version", "target", "composable", "interfaces"],
        },
        "EntityCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entities": {"type": "array", "items": generic_object},
            },
            "required": ["entities"],
        },
        "WorkflowSpec": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
                "stages": {"type": "array", "items": {"type": "string"}},
                "guards": generic_object,
                "assignments": generic_object,
                "human_tasks": {"type": "array", "items": {"type": "string"}},
                "timers": generic_object,
                "waits": generic_object,
                "retry_policy": generic_object,
                "compensation": generic_object,
                "transitions": {"type": "array", "items": generic_object},
                "methods": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "type", "steps", "stages", "transitions"],
        },
        "WorkflowCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "workflows": {"type": "object", "additionalProperties": _schema_ref("WorkflowSpec")},
            },
            "required": ["workflows"],
        },
        "WorkflowRunRequest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "payload": generic_object,
                "start_at": {"type": "string"},
                "pause_at": {"type": "string"},
                "stop_after": {"type": "string"},
            },
        },
        "WorkflowRunResult": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "string"},
                "workflow": {"type": "string"},
                "status": {"type": "string"},
                "started_at": {"type": "string"},
                "current_step": {"type": "string"},
                "completed_at": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "steps": {"type": "array", "items": {"type": "string"}},
                "completed_steps": {"type": "array", "items": {"type": "string"}},
                "pending_steps": {"type": "array", "items": {"type": "string"}},
                "trace": {"type": "array", "items": generic_object},
                "payload": generic_object,
                "event_id": {"type": "integer"},
                "blocked_at": {"type": "string"},
                "blocked_reason": {"type": "string"},
                "waiting_at": {"type": "string"},
                "waiting_for": {"type": "string"},
                "failed_at": {"type": "string"},
                "failure_reason": {"type": "string"},
                "compensations": {"type": "array", "items": generic_object},
                "guard": {"oneOf": [{"type": "string"}, {"type": "boolean"}, generic_object]},
            },
            "required": ["id", "workflow", "status", "steps", "trace", "payload"],
        },
        "WorkflowRunCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "runs": {"type": "array", "items": _schema_ref("WorkflowRunResult")},
            },
            "required": ["runs"],
        },
        "WorkflowCompensationRequest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "payload": generic_object,
                "context": generic_object,
            },
        },
        "WorkflowCompensationResult": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "string"},
                "workflow": {"type": "string"},
                "status": {"type": "string"},
                "already_executed": {"type": "boolean"},
                "actions": {"type": "array", "items": generic_object},
                "event_id": {"type": "integer"},
                "run": _schema_ref("WorkflowRunResult"),
            },
            "required": ["id", "status", "already_executed", "actions", "run"],
        },
        "RecordsByEntity": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "records": {"type": "object", "additionalProperties": {"type": "array", "items": generic_object}},
            },
            "required": ["records"],
        },
        "AuthStatus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string"},
                "header": nullable_string,
            },
            "required": ["mode", "header"],
        },
        "StorageStatus": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "mode": {"type": "string"},
                "path": nullable_string,
                "records": {"type": "object", "additionalProperties": {"type": "array", "items": generic_object}},
                "events": {"type": "array", "items": _schema_ref("EventRecord")},
            },
            "required": ["mode", "path"],
        },
        "ValidationReport": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "valid": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "checks": generic_object,
            },
            "required": ["name", "valid", "errors", "warnings", "checks"],
        },
        "HealthReport": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "valid": {"type": "boolean"},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "name", "version", "valid", "storage", "auth", "warnings"],
        },
        "EventLog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "events": {"type": "array", "items": _schema_ref("EventRecord")},
            },
            "required": ["events"],
        },
        "MetricsSnapshot": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "entity_count": {"type": "integer"},
                "database_status": _schema_ref("DatabaseStatus"),
                "record_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                "total_records": {"type": "integer"},
                "event_count": {"type": "integer"},
                "event_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                "relationship_count": {"type": "integer"},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
            },
            "required": ["name", "version", "entity_count", "record_counts", "total_records", "event_count"],
        },
        "SelfTestReport": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "passed": {"type": "boolean"},
                "status": {"type": "string"},
                "checks": _schema_ref("SelfTestChecks"),
                "routes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "version", "passed", "status", "checks", "routes"],
        },
        "SelfTestChecks": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "validation": _schema_ref("ValidationReport"),
                "metrics": _schema_ref("MetricsSnapshot"),
                "route_count": {"type": "integer"},
                "entity_count": {"type": "integer"},
                "capability_health": _schema_ref("CapabilityHealthReport"),
            },
            "required": ["validation", "metrics", "route_count", "entity_count"],
        },
        "RelationshipNode": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["id", "name", "type"],
        },
        "RelationshipEdge": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "field": {"type": "string"},
                "relationship": {"type": "string"},
            },
            "required": ["from", "to", "relationship"],
        },
        "RelationshipGraph": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nodes": {"type": "array", "items": _schema_ref("RelationshipNode")},
                "edges": {"type": "array", "items": _schema_ref("RelationshipEdge")},
            },
            "required": ["nodes", "edges"],
        },
        "AgentCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "agents": generic_object,
                "teams": generic_object,
            },
            "required": ["agents", "teams"],
        },
        "ApplicationCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "applications": generic_object,
                "dependency_graph": generic_object,
                "components": generic_object,
            },
            "required": ["applications", "dependency_graph", "components"],
        },
        "CapabilityCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capabilities": generic_object,
                "by_erp_module": generic_object,
                "dependency_graph": generic_object,
                "load_order": {"oneOf": [generic_object, {"type": "array", "items": {"type": "string"}}]},
            },
            "required": ["capabilities", "by_erp_module", "dependency_graph", "load_order"],
        },
        "CapabilityHealth": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "status": {"type": "string"},
                "healthy": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "configuration": generic_object,
                "rules": generic_object,
                "approvals": generic_object,
                "ui": generic_object,
                "theme": generic_object,
                "streaming": generic_object,
                "master_data": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "components": generic_object,
            },
            "required": ["capability", "status", "healthy", "errors", "warnings"],
        },
        "CapabilityHealthReport": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "healthy": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "capabilities": {"type": "object", "additionalProperties": _schema_ref("CapabilityHealth")},
            },
            "required": ["healthy", "errors", "warnings", "capabilities"],
        },
        "RouteCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "routes": generic_object,
            },
            "required": ["routes"],
        },
        "AgentInvocationRequest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "message": {"type": "string"},
                "payload": generic_object,
                "context": generic_object,
            },
        },
        "AgentInvocationResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "agent": {"type": "string"},
                "team": {"type": "string"},
                "runtime": {"type": "string"},
                "status": {"type": "string"},
                "result": {"oneOf": [generic_object, {"type": "string"}, {"type": "null"}]},
                "payload": generic_object,
            },
        },
        "RuleEvaluationRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "context": generic_object,
            },
            "required": ["context"],
        },
        "RuleEvaluationResult": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "decision": {"type": "string"},
                "matched_rules": {"type": "array", "items": {"type": "string"}},
                "actions": {"type": "array", "items": generic_object},
                "context": generic_object,
            },
        },
        "CapabilityConfigurationRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "configuration": generic_object,
                "overrides": generic_object,
            },
        },
        "CapabilityConfigurationResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "configuration": generic_object,
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        "ApprovalPlanRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "context": generic_object,
            },
            "required": ["context"],
        },
        "ApprovalPlanResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "required": {"type": "boolean"},
                "levels": {"type": "integer"},
                "approvers": {"type": "array", "items": {"type": "string"}},
                "thresholds": generic_object,
                "segregation_of_duties": {"type": "boolean"},
                "escalation": {"oneOf": [{"type": "string"}, generic_object, {"type": "null"}]},
            },
        },
        "StreamingTopology": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "processor": {"type": "string"},
                "processors": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                "states": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                "streams": {"type": "object", "additionalProperties": generic_object},
            },
            "required": ["processor", "processors", "states", "streams"],
        },
        "CapabilityStreamingContract": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "processor": {"type": "string"},
                "state": {"type": "string"},
                "input": generic_object,
                "output": generic_object,
            },
        },
        "EventRecord": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "integer"},
                "entity": {"type": "string"},
                "action": {"type": "string"},
                "record_id": {"oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
                "before": {"oneOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}]},
                "after": {"oneOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}]},
            },
            "required": ["id", "entity", "action"],
        },
        "DatabaseReference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string"},
                "relationship": {"type": "string"},
                "schema": {"type": "string"},
                "table": {"type": "string"},
                "column": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["table", "column"],
        },
        "DatabaseColumn": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "primary_key": {"type": "boolean"},
                "nullable": {"type": "boolean"},
                "default": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "integer"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "constraints": {"type": "array", "items": {"type": "string"}},
                "reference": {"oneOf": [_schema_ref("DatabaseReference"), {"type": "null"}]},
            },
            "required": ["name", "type", "primary_key", "nullable", "constraints"],
        },
        "DatabaseIndex": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": nullable_string,
                "columns": {"type": "array", "items": {"type": "string"}},
                "unique": {"type": "boolean"},
                "type": nullable_string,
            },
            "required": ["columns", "unique"],
        },
        "DatabaseTable": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "columns": {"type": "array", "items": _schema_ref("DatabaseColumn")},
                "indexes": {"type": "array", "items": _schema_ref("DatabaseIndex")},
            },
            "required": ["name", "columns", "indexes"],
        },
        "DatabaseSchema": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "tables": {"type": "array", "items": _schema_ref("DatabaseTable")},
            },
            "required": ["name", "tables"],
        },
        "DatabaseCatalogEntry": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "type": {"const": "database"},
                "properties": {"type": "array", "items": {"type": "string"}},
                "connection_config": {"type": "object", "additionalProperties": True},
                "schemas": {"type": "array", "items": _schema_ref("DatabaseSchema")},
            },
            "required": ["name", "type", "schemas"],
        },
        "DatabaseCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "databases": {"type": "array", "items": _schema_ref("DatabaseCatalogEntry")},
            },
            "required": ["databases"],
        },
        "DatabaseSchemaCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "database": {"type": "string"},
                "schemas": {"type": "array", "items": _schema_ref("DatabaseSchema")},
            },
            "required": ["database", "schemas"],
        },
        "DatabaseValidation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "validated_databases": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["errors", "warnings", "validated_databases"],
        },
        "DatabaseStatus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "valid": {"type": "boolean"},
                "database_count": {"type": "integer"},
                "schema_count": {"type": "integer"},
                "table_count": {"type": "integer"},
                "reference_count": {"type": "integer"},
                "validation": _schema_ref("DatabaseValidation"),
            },
            "required": [
                "valid",
                "database_count",
                "schema_count",
                "table_count",
                "reference_count",
                "validation",
            ],
        },
    }


def _api_operation(
    summary: str,
    description: str,
    status: str = "200",
    request_body: bool = False,
    request_schema: Dict[str, Any] | None = None,
    response_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {"description": description}
    if response_schema is not None:
        response["content"] = _json_media(response_schema)
    operation: Dict[str, Any] = {
        "summary": summary,
        "responses": {status: response},
    }
    if request_body:
        operation["requestBody"] = {"required": True}
        if request_schema is not None:
            operation["requestBody"]["content"] = _json_media(request_schema)
    return operation


def openapi_document() -> Dict[str, Any]:
    paths: Dict[str, Any] = {
        "/health": {"get": _api_operation("Application health", "Health report", response_schema=_schema_ref("HealthReport"))},
        "/component.json": {"get": _api_operation("Composable component manifest", "APG component manifest", response_schema=_schema_ref("ComponentManifest"))},
        "/manifest": {"get": _api_operation("Application manifest", "APG manifest", response_schema=_schema_ref("ApplicationDescription"))},
        "/semantic-model.json": {"get": _api_operation("Semantic model", "APG semantic model", response_schema=_schema_ref("SemanticModel"))},
        "/openapi.json": {"get": _api_operation("OpenAPI contract", "OpenAPI 3.1 contract", response_schema={"type": "object", "additionalProperties": True})},
        "/validate": {"get": _api_operation("Application validation", "Validation report", response_schema=_schema_ref("ValidationReport"))},
        "/events": {"get": _api_operation("Record mutation events", "Event log", response_schema=_schema_ref("EventLog"))},
        "/auth": {"get": _api_operation("Authentication status", "Authentication mode", response_schema=_schema_ref("AuthStatus"))},
        "/metrics": {"get": _api_operation("Application metrics", "Runtime metrics", response_schema=_schema_ref("MetricsSnapshot"))},
        "/applications": {"get": _api_operation("Application compositions", "Application composition catalog", response_schema=_schema_ref("ApplicationCatalog"))},
        "/self-test": {"get": _api_operation("Application self-test", "Self-test report", response_schema=_schema_ref("SelfTestReport"))},
        "/theme.css": {"get": _api_operation("Generated visual theme stylesheet", "CSS theme stylesheet")},
        "/records": {"get": _api_operation("All entity records", "Records by entity", response_schema=_schema_ref("RecordsByEntity"))},
        "/entities": {"get": _api_operation("Entity catalog", "Generated entity metadata", response_schema=_schema_ref("EntityCatalog"))},
        "/workflows": {"get": _api_operation("Workflow catalog", "Generated workflow metadata", response_schema=_schema_ref("WorkflowCatalog"))},
        "/workflows/runs": {"get": _api_operation("Workflow run catalog", "Generated workflow run state", response_schema=_schema_ref("WorkflowRunCatalog"))},
        "/workflows/runs/{id}": {"get": _api_operation("Workflow run detail", "Generated workflow run state", response_schema=_schema_ref("WorkflowRunResult"))},
        "/workflows/runs/{id}/resume": {"post": _api_operation("Resume workflow run", "Workflow resume result", request_body=True, request_schema=_schema_ref("WorkflowRunRequest"), response_schema=_schema_ref("WorkflowRunResult"))},
        "/workflows/runs/{id}/compensate": {"post": _api_operation("Execute workflow compensations", "Workflow compensation result", request_body=True, request_schema=_schema_ref("WorkflowCompensationRequest"), response_schema=_schema_ref("WorkflowCompensationResult"))},
        "/databases": {"get": _api_operation("Database catalog", "Database schema and connection metadata", response_schema=_schema_ref("DatabaseCatalog"))},
        "/databases/status": {"get": _api_operation("Database validation status", "Database schema validation and counts", response_schema=_schema_ref("DatabaseStatus"))},
        "/relationships": {"get": _api_operation("Entity relationship graph", "Relationship graph", response_schema=_schema_ref("RelationshipGraph"))},
        "/storage": {"get": _api_operation("Record storage status", "Storage status", response_schema=_schema_ref("StorageStatus"))},
        "/agents": {"get": _api_operation("Agent catalog", "AI agent and team catalog", response_schema=_schema_ref("AgentCatalog"))},
        "/capabilities": {"get": _api_operation("Capability catalog", "Capability catalog", response_schema=_schema_ref("CapabilityCatalog"))},
        "/capabilities/health": {"get": _api_operation("Capability health report", "Capability health report", response_schema=_schema_ref("CapabilityHealthReport"))},
        "/routes": {"get": _api_operation("Generated UI route catalog", "UI route catalog", response_schema=_schema_ref("RouteCatalog"))},
        "/composition": {"get": _api_operation("Composition graph", "Composition graph", response_schema=_schema_ref("RelationshipGraph"))},
        "/ui": {"get": _api_operation("Generated application UI", "HTML application index")},
        "/ui/databases": {"get": _api_operation("Generated database catalog UI", "HTML database catalog")},
    }
    schemas: Dict[str, Any] = _database_openapi_schemas()
    for entity in ENTITIES:
        entity_name = str(entity["name"])
        schema_name = f"{entity_name}Record"
        patch_schema_name = f"{entity_name}RecordPatch"
        schemas[schema_name] = _record_schema(entity)
        schemas[patch_schema_name] = _record_schema(entity, partial=True)
        paths[f"/entities/{entity_name}/records"] = {
            "get": _api_operation(
                f"List {entity_name} records",
                "Record list",
                response_schema=_record_list_response_schema(schema_name),
            ),
            "post": _api_operation(
                f"Create {entity_name} record",
                "Created record",
                status="201",
                request_body=True,
                request_schema=_record_body_schema(schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records"]["get"]["parameters"] = [
            {"name": "filter.<field>", "in": "query", "required": False, "description": "Exact field filter"},
            {"name": "sort", "in": "query", "required": False, "description": "Field to sort by"},
            {"name": "order", "in": "query", "required": False, "description": "asc or desc"},
            {"name": "limit", "in": "query", "required": False, "description": "Maximum records to return"},
            {"name": "offset", "in": "query", "required": False, "description": "Records to skip"},
        ]
        paths[f"/entities/{entity_name}/records/export"] = {
            "get": _api_operation(
                f"Export {entity_name} records",
                "Record export",
                response_schema=_record_export_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records/import"] = {
            "post": _api_operation(
                f"Import {entity_name} records",
                "Record import",
                request_body=True,
                request_schema=_record_import_body_schema(schema_name),
                response_schema=_record_import_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records/{{id}}"] = {
            "get": _api_operation(
                f"Fetch {entity_name} record",
                "Record",
                response_schema=_record_item_response_schema(schema_name),
            ),
            "put": _api_operation(
                f"Update {entity_name} record",
                "Updated record",
                request_body=True,
                request_schema=_record_body_schema(patch_schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
            "delete": _api_operation(
                f"Delete {entity_name} record",
                "Deleted record",
                response_schema=_record_mutation_response_schema(schema_name, record_key="deleted"),
            ),
        }
        paths[f"/ui/entities/{entity_name}"] = {
            "get": _api_operation(f"Generated {entity_name} UI", "HTML entity screen"),
        }
        if entity.get("type") == "database":
            paths[f"/databases/{entity_name}/schemas"] = {
                "get": _api_operation(f"{entity_name} database schemas", "Database schema metadata", response_schema=_schema_ref("DatabaseSchemaCatalog")),
            }
    for workflow_name in list_workflows():
        paths[f"/workflows/{workflow_name}"] = {
            "get": _api_operation(f"Describe {workflow_name} workflow", "Workflow description", response_schema=_schema_ref("WorkflowSpec")),
        }
        paths[f"/workflows/{workflow_name}/run"] = {
            "post": _api_operation(f"Run {workflow_name} workflow", "Workflow run result", request_body=True, request_schema=_schema_ref("WorkflowRunRequest"), response_schema=_schema_ref("WorkflowRunResult")),
        }
    if APG_CAPABILITIES is not None:
        paths["/rules/evaluate"] = {"post": _api_operation("Evaluate capability rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult"))}
        paths["/configuration/resolve"] = {"post": _api_operation("Resolve capability configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}
        paths["/configuration/validate"] = {"post": _api_operation("Validate capability configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}
        paths["/approval/plan"] = {"post": _api_operation("Plan capability approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse"))}
        paths["/streaming"] = {"get": _api_operation("Streaming topology", "ByteWax streaming topology", response_schema=_schema_ref("StreamingTopology"))}
        if hasattr(APG_CAPABILITIES, "list_capabilities"):
            for capability_name in APG_CAPABILITIES.list_capabilities():
                paths[f"/capabilities/{capability_name}/streaming"] = {
                    "get": _api_operation(f"{capability_name} streaming contract", "Capability streaming contract", response_schema=_schema_ref("CapabilityStreamingContract")),
                }
                paths[f"/capabilities/{capability_name}/health"] = {
                    "get": _api_operation(f"{capability_name} health", "Capability health", response_schema=_schema_ref("CapabilityHealth")),
                }
                paths[f"/capabilities/{capability_name}/rules/evaluate"] = {
                    "post": _api_operation(f"Evaluate {capability_name} rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult")),
                }
                paths[f"/capabilities/{capability_name}/configuration/resolve"] = {
                    "post": _api_operation(f"Resolve {capability_name} configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }
                paths[f"/capabilities/{capability_name}/configuration/validate"] = {
                    "post": _api_operation(f"Validate {capability_name} configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }
                paths[f"/capabilities/{capability_name}/approval/plan"] = {
                    "post": _api_operation(f"Plan {capability_name} approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse")),
                }
        route_index = getattr(APG_CAPABILITIES, "ui_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {"get": _api_operation(f"Capability screen {route}", "Generated capability screen")}
    if AI_AGENTS is not None:
        for agent_name in describe_application().get("ai_agents", []):
            paths[f"/agents/{agent_name}/invoke"] = {
                "post": _api_operation(f"Invoke agent {agent_name}", "Agent invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }
        for team_name in describe_application().get("ai_agent_teams", []):
            paths[f"/agent-teams/{team_name}/invoke"] = {
                "post": _api_operation(f"Invoke agent team {team_name}", "Agent team invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }
    if APG_APPLICATIONS is not None:
        route_index = getattr(APG_APPLICATIONS, "application_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {"get": _api_operation(f"Application route {route}", "Generated application composition screen")}
    return {
        "openapi": "3.1.0",
        "info": {
            "title": MODULE_NAME,
            "version": MODULE_VERSION,
            "description": MODULE_DESCRIPTION,
        },
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-APG-API-Key"},
                "BearerAuth": {"type": "http", "scheme": "bearer"},
            },
        },
    }


def validate_component_manifest_contract() -> Dict[str, Any]:
    manifest = component_manifest()
    openapi = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    interfaces = manifest.get("interfaces", {})
    http = interfaces.get("http", {}) if isinstance(interfaces, dict) else {}
    python = interfaces.get("python", {}) if isinstance(interfaces, dict) else {}
    http_paths = sorted(http.get("paths", [])) if isinstance(http, dict) else []
    expected_paths = sorted(openapi.get("paths", {}))
    if http.get("openapi") != "/openapi.json":
        errors.append("component manifest HTTP interface must point to /openapi.json")
    if http_paths != expected_paths:
        errors.append("component manifest HTTP paths do not match OpenAPI paths")
    exports = python.get("exports", []) if isinstance(python, dict) else []
    if not isinstance(exports, list) or not exports:
        errors.append("component manifest Python interface does not declare exports")
        exports = []
    export_names: list[str] = []
    for export_name in exports:
        if not isinstance(export_name, str):
            errors.append("component manifest Python exports must be strings")
            continue
        export_names.append(export_name)
    missing_exports = [
        export_name
        for export_name in export_names
        if export_name not in globals() or not callable(globals()[export_name])
    ]
    for export_name in missing_exports:
        errors.append(f"component manifest Python export {export_name} is not callable")
    expected_record_names = sorted(ENTITY_NAMES)
    manifest_record_names = sorted(interfaces.get("records", [])) if isinstance(interfaces, dict) else []
    if manifest_record_names != expected_record_names:
        errors.append("component manifest record interface does not match generated entities")
    if interfaces.get("theme") != "/theme.css":
        errors.append("component manifest theme interface must point to /theme.css")
    if interfaces.get("semantic_model") != "/semantic-model.json":
        errors.append("component manifest semantic model interface must point to /semantic-model.json")
    deployment = manifest.get("deployment", {})
    expected_artifacts = ["app.py", "__init__.py", "README.md", "semantic_model.json", "requirements.txt", "Dockerfile", ".dockerignore", ".env.example", "smoke_test.py"]
    raw_artifacts = deployment.get("artifacts", []) if isinstance(deployment, dict) else []
    artifacts: set[str] = set()
    if not isinstance(raw_artifacts, list):
        errors.append("component manifest deployment artifacts must be an array")
        raw_artifacts = []
    for artifact in raw_artifacts:
        if not isinstance(artifact, str):
            errors.append("component manifest deployment artifacts must be strings")
            continue
        artifacts.add(artifact)
    unexpected_artifacts = sorted(artifacts.difference(expected_artifacts))
    for artifact in unexpected_artifacts:
        errors.append(f"component manifest deployment has unexpected artifact {artifact}")
    artifact_root = Path(__file__).resolve().parent if "__file__" in globals() else None
    for artifact in expected_artifacts:
        if artifact not in artifacts:
            errors.append(f"component manifest deployment is missing artifact {artifact}")
            continue
        if artifact_root is not None and not (artifact_root / artifact).exists():
            errors.append(f"component manifest deployment artifact {artifact} does not exist")
    commands = deployment.get("commands", {}) if isinstance(deployment, dict) else {}
    expected_commands = {
        "run": "python app.py",
        "describe": "python app.py --describe",
        "semantic_model": "python app.py --semantic-model",
        "validate": "python app.py --validate",
        "self_test": "python app.py --self-test",
        "smoke_test": "python smoke_test.py",
    }
    if not isinstance(commands, dict):
        errors.append("component manifest deployment commands must be an object")
        commands = {}
    for command_name, expected_command in expected_commands.items():
        actual_command = commands.get(command_name)
        if actual_command is None:
            errors.append(f"component manifest deployment is missing command {command_name}")
        elif actual_command != expected_command:
            errors.append(
                f"component manifest deployment command {command_name} must be {expected_command!r}"
            )
    environment = deployment.get("environment", []) if isinstance(deployment, dict) else []
    expected_environment = ["APG_HOST", "APG_PORT", "APG_DATA_FILE", "APG_API_KEY", "APG_DEBUG"]
    if environment != expected_environment:
        errors.append("component manifest deployment environment does not match generated runtime variables")
    return {
        "errors": errors,
        "warnings": warnings,
        "http_path_count": len(http_paths),
        "python_exports": sorted(export_names),
        "artifact_count": len(artifacts),
        "command_count": len(commands),
    }


def _walk_openapi_refs(value: Any, path: str = "$") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        raw_ref = value.get("$ref")
        if isinstance(raw_ref, str):
            refs.append((path + ".$ref", raw_ref))
        for key, child in value.items():
            if key == "$ref":
                continue
            refs.extend(_walk_openapi_refs(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            refs.extend(_walk_openapi_refs(child, f"{path}[{index}]"))
    return refs


def validate_openapi_contract() -> Dict[str, Any]:
    document = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    paths = document.get("paths", {})
    schemas = document.get("components", {}).get("schemas", {})
    if not isinstance(paths, dict) or not paths:
        errors.append("OpenAPI document does not declare paths")
        paths = {}
    if not isinstance(schemas, dict):
        errors.append("OpenAPI document components.schemas must be an object")
        schemas = {}
    for schema_name, schema in sorted(schemas.items()):
        if not isinstance(schema, dict):
            errors.append(f"OpenAPI schema {schema_name} must be an object")
            continue
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if required and not isinstance(required, list):
            errors.append(f"OpenAPI schema {schema_name} required must be an array")
            continue
        if required and not isinstance(properties, dict):
            errors.append(f"OpenAPI schema {schema_name} declares required fields without object properties")
            continue
        for field_name in required:
            if not isinstance(field_name, str):
                errors.append(f"OpenAPI schema {schema_name} required field names must be strings")
            elif field_name not in properties:
                errors.append(f"OpenAPI schema {schema_name} requires missing property {field_name}")
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            errors.append(f"OpenAPI path {route} must be an object")
            continue
        for method, operation in sorted(path_item.items()):
            if method.lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            if not isinstance(operation, dict):
                errors.append(f"OpenAPI operation {method.upper()} {route} must be an object")
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict) or not responses:
                errors.append(f"OpenAPI operation {method.upper()} {route} does not declare responses")
    referenced_schemas: set[str] = set()
    for ref_path, ref in _walk_openapi_refs(document):
        prefix = "#/components/schemas/"
        if not ref.startswith(prefix):
            errors.append(f"OpenAPI reference {ref} at {ref_path} is not an internal component schema reference")
            continue
        schema_name = ref[len(prefix):]
        referenced_schemas.add(schema_name)
        if schema_name not in schemas:
            errors.append(f"OpenAPI reference {ref} at {ref_path} does not resolve")
    return {
        "errors": sorted(errors),
        "warnings": warnings,
        "path_count": len(paths),
        "schema_count": len(schemas),
        "referenced_schemas": sorted(referenced_schemas),
    }


def _route_dispatch_target(route: str, method: str) -> str | None:
    method = method.lower()
    route = route.rstrip("/") or "/"
    if method == "get":
        if route == "/theme.css":
            return "theme_stylesheet"
        if route == "/ui" or route.startswith("/ui/"):
            return "_ui_payload"
        if _capability_screen(route) is not None:
            return "_capability_screen_payload"
        if _application_screen(route) is not None:
            return "_application_screen_payload"
        if route in {
            "/",
            "/manifest",
            "/application",
            "/component.json",
            "/semantic-model.json",
            "/health",
            "/validate",
            "/openapi.json",
            "/entities",
            "/workflows",
            "/workflows/runs",
            "/databases",
            "/databases/status",
            "/auth",
            "/events",
            "/metrics",
            "/self-test",
            "/records",
            "/relationships",
            "/storage",
            "/agents",
            "/applications",
            "/capabilities",
            "/streaming",
            "/routes",
            "/composition",
        }:
            return "_route_payload"
        if route.startswith("/databases/") and route.endswith("/schemas"):
            return "_route_payload"
        if route.startswith("/workflows/runs/"):
            return "_route_payload"
        if route.startswith("/workflows/"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/streaming"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/health"):
            return "_route_payload"
        if route.startswith("/entities/") and "/records" in route:
            return "_records_payload_with_query"
        return None
    if method == "post":
        if route.startswith("/agents/") and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if (route.startswith("/agent-teams/") or route.startswith("/teams/")) and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if route.startswith("/entities/") and (route.endswith("/records") or route.endswith("/records/import")):
            return "_create_record_payload"
        if route in {"/rules/evaluate", "/capabilities/rules/evaluate"} or (
            route.startswith("/capabilities/") and route.endswith("/rules/evaluate")
        ):
            return "_rule_evaluation_payload"
        if route in {"/configuration/resolve", "/capabilities/configuration/resolve"} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/resolve")
        ):
            return "_configuration_payload"
        if route in {"/configuration/validate", "/capabilities/configuration/validate"} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/validate")
        ):
            return "_configuration_payload"
        if route in {"/approval/plan", "/capabilities/approval/plan"} or (
            route.startswith("/capabilities/") and route.endswith("/approval/plan")
        ):
            return "_approval_plan_payload"
        if route.startswith("/workflows/runs/") and route.endswith("/compensate"):
            return "_workflow_compensation_payload"
        if route.startswith("/workflows/runs/") and route.endswith("/resume"):
            return "_workflow_resume_payload"
        if route.startswith("/workflows/") and route.endswith("/run"):
            return "_workflow_run_payload"
        return None
    if method == "put":
        if route.startswith("/entities/") and "/records/{id}" in route:
            return "_update_record_payload"
        return None
    if method == "delete":
        if route.startswith("/entities/") and "/records/{id}" in route:
            return "_delete_record_payload"
        return None
    return None


def validate_route_dispatch_contract() -> Dict[str, Any]:
    document = openapi_document()
    paths = document.get("paths", {})
    errors: list[str] = []
    warnings: list[str] = []
    route_targets: Dict[str, list[Dict[str, str]]] = {}
    method_count = 0
    if not isinstance(paths, dict):
        return {
            "errors": ["OpenAPI paths must be an object before dispatch validation"],
            "warnings": warnings,
            "route_count": 0,
            "method_count": 0,
            "routes": route_targets,
        }
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            continue
        for method in sorted(path_item):
            method_name = str(method).lower()
            if method_name not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            method_count += 1
            target = _route_dispatch_target(str(route), method_name)
            if target is None:
                errors.append(f"OpenAPI route {method_name.upper()} {route} has no generated dispatcher")
                continue
            route_targets.setdefault(str(route), []).append({"method": method_name.upper(), "target": target})
    return {
        "errors": errors,
        "warnings": warnings,
        "route_count": len(paths),
        "method_count": method_count,
        "routes": route_targets,
    }


def describe_application() -> Dict[str, Any]:
    _entity_summary_keys = {"name", "type", "properties", "methods"}
    description: Dict[str, Any] = {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "entities": [
            {k: v for k, v in entity.items() if k in _entity_summary_keys}
            for entity in list_entities()
        ],
        "databases": list_databases(),
    }
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agents"] = AI_AGENTS.list_agents()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_agent") and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agent_descriptions"] = {
            name: AI_AGENTS.describe_agent(name)
            for name in AI_AGENTS.list_agents()
        }
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_teams"] = AI_AGENTS.list_agent_teams()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_team") and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_team_descriptions"] = {
            name: AI_AGENTS.describe_team(name)
            for name in AI_AGENTS.list_agent_teams()
        }
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "list_applications"):
        description["application_compositions"] = APG_APPLICATIONS.list_applications()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "describe_application_compositions"):
        description["application_composition_descriptions"] = APG_APPLICATIONS.describe_application_compositions()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_dependency_graph"):
        description["application_dependency_graph"] = APG_APPLICATIONS.application_dependency_graph()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_component_catalog"):
        description["application_component_catalog"] = APG_APPLICATIONS.application_component_catalog()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_route_index"):
        description["application_routes"] = APG_APPLICATIONS.application_route_index()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities"):
        description["capabilities"] = APG_CAPABILITIES.list_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities"):
        description["capability_descriptions"] = APG_CAPABILITIES.describe_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities_by_erp_module"):
        description["capability_descriptions_by_erp_module"] = APG_CAPABILITIES.describe_capabilities_by_erp_module()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_dependency_graph"):
        description["capability_dependency_graph"] = APG_CAPABILITIES.capability_dependency_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_load_order"):
        description["capability_load_order"] = APG_CAPABILITIES.capability_load_order()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "ui_route_index"):
        description["ui_routes"] = APG_CAPABILITIES.ui_route_index()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "composition_graph"):
        description["composition_graph"] = APG_CAPABILITIES.composition_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "streaming_processor_index"):
        description["streaming_processors"] = APG_CAPABILITIES.streaming_processor_index()
    return description


def _record_validation(report: Dict[str, Any], name: str, validation: Dict[str, Any]) -> None:
    check = dict(validation)
    errors = [str(error) for error in check.get("errors", [])]
    warnings = [str(warning) for warning in check.get("warnings", [])]
    report["checks"][name] = check
    report["errors"].extend(f"{name}: {error}" for error in errors)
    report["warnings"].extend(f"{name}: {warning}" for warning in warnings)


def validate_database_schema_contracts() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    validated: list[str] = []
    for database in list_databases():
        database_name = str(database.get("name", "database"))
        validated.append(database_name)
        schemas = database.get("schemas", [])
        if not schemas:
            warnings.append(f"{database_name} does not declare schemas")
            continue
        table_index: Dict[str, list[Dict[str, Any]]] = {}
        seen_schemas: set[str] = set()
        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            schema_key = schema_name.lower()
            if schema_key in seen_schemas:
                errors.append(f"{database_name} declares duplicate schema {schema_name}")
            seen_schemas.add(schema_key)
            seen_tables: set[str] = set()
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    errors.append(f"{database_name}.{schema_name} declares a table without a name")
                    continue
                table_key = table_name.lower()
                qualified_key = f"{schema_name}.{table_name}".lower()
                if table_key in seen_tables:
                    errors.append(f"{database_name}.{schema_name} declares duplicate table {table_name}")
                seen_tables.add(table_key)
                table_index.setdefault(table_key, []).append(table)
                table_index.setdefault(qualified_key, []).append(table)

        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                columns = table.get("columns", [])
                column_names = [str(column.get("name", "")) for column in columns if isinstance(column, dict)]
                known_columns = {column_name.lower() for column_name in column_names if column_name}
                if len(known_columns) != len([column_name for column_name in column_names if column_name]):
                    errors.append(f"{database_name}.{schema_name}.{table_name} declares duplicate columns")
                if columns and not any(bool(column.get("primary_key")) for column in columns if isinstance(column, dict)):
                    warnings.append(f"{database_name}.{schema_name}.{table_name} does not declare a primary key")
                for index in table.get("indexes", []):
                    for indexed_column in index.get("columns", []):
                        if str(indexed_column).lower() not in known_columns:
                            errors.append(
                                f"{database_name}.{schema_name}.{table_name} index references unknown column {indexed_column}"
                            )
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    reference = column.get("reference")
                    if not isinstance(reference, dict):
                        continue
                    target_table_name = str(reference.get("table", ""))
                    target_column_name = str(reference.get("column", ""))
                    target_schema_name = str(reference.get("schema", ""))
                    target_label = (
                        f"{target_schema_name}.{target_table_name}"
                        if target_schema_name
                        else target_table_name
                    )
                    if target_schema_name:
                        candidates = table_index.get(f"{target_schema_name}.{target_table_name}".lower(), [])
                    else:
                        candidates = table_index.get(f"{schema_name}.{target_table_name}".lower(), [])
                        if not candidates:
                            candidates = table_index.get(target_table_name.lower(), [])
                    if not candidates:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references unknown table {target_label}"
                        )
                        continue
                    if len(candidates) > 1:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references ambiguous table {target_label}; use schema-qualified target"
                        )
                        continue
                    target_table = candidates[0]
                    target_columns = {
                        str(target_column.get("name", "")).lower()
                        for target_column in target_table.get("columns", [])
                        if isinstance(target_column, dict)
                    }
                    if target_column_name.lower() not in target_columns:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references unknown column {target_label}.{target_column_name}"
                        )
    return {"errors": errors, "warnings": warnings, "validated_databases": sorted(validated)}


def validate_workflow_contracts() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for workflow_name in list_workflows():
        workflow = describe_workflow(workflow_name)
        steps = workflow.get("steps", [])
        step_set = set(str(step) for step in steps)
        if not steps:
            warnings.append(f"{workflow_name} does not declare executable steps")
        transitions = workflow.get("transitions", [])
        if len(steps) > 1 and len(transitions) != len(steps) - 1:
            errors.append(f"{workflow_name} transition count does not match step chain")
        for section in ("guards", "assignments", "timers", "waits", "retry_policy", "compensation"):
            mapping = workflow.get(section, {})
            if not isinstance(mapping, dict):
                errors.append(f"{workflow_name} {section} metadata must be an object")
                continue
            for step in mapping:
                if str(step) not in step_set:
                    errors.append(f"{workflow_name} {section} references unknown step {step}")
        assignments = workflow.get("assignments", {})
        for step in workflow.get("human_tasks", []):
            if str(step) not in step_set:
                errors.append(f"{workflow_name} human task references unknown step {step}")
            elif str(step) not in assignments:
                warnings.append(f"{workflow_name} human task {step} has no assignee")
    return {"errors": errors, "warnings": warnings, "validated_workflows": list_workflows()}


def validate_application(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "name": MODULE_NAME,
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }
    _record_validation(report, "openapi_contract", validate_openapi_contract())
    _record_validation(report, "component_manifest", validate_component_manifest_contract())
    _record_validation(report, "route_dispatch", validate_route_dispatch_contract())
    _record_validation(report, "database_schemas", validate_database_schema_contracts())
    _record_validation(report, "workflows", validate_workflow_contracts())
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        _record_validation(
            report,
            "ai_agent_runtimes",
            AI_AGENTS.validate_agent_runtimes(available_agent_runtimes),
        )
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "validate_application_compositions"):
        available_capabilities = APG_CAPABILITIES.list_capabilities() if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") else []
        available_agents = AI_AGENTS.list_agents() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents") else []
        available_teams = AI_AGENTS.list_agent_teams() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams") else []
        _record_validation(
            report,
            "application_compositions",
            APG_APPLICATIONS.validate_application_compositions(
                available_capabilities=available_capabilities,
                available_agents=available_agents,
                available_teams=available_teams,
            ),
        )
    if APG_CAPABILITIES is not None:
        for check_name, function_name in (
            ("capability_contracts", "validate_capability_contracts"),
            ("capability_dependencies", "validate_capability_dependencies"),
            ("component_contracts", "validate_component_contracts"),
            ("master_data_contracts", "validate_master_data_contracts"),
            ("capability_i18n", "validate_capability_i18n"),
            ("streaming_contracts", "validate_streaming_contracts"),
        ):
            validator = getattr(APG_CAPABILITIES, function_name, None)
            if validator is not None:
                _record_validation(report, check_name, validator())
    report["valid"] = not report["errors"]
    return report


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _css_name(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "value"


def theme_stylesheet() -> str:
    lines = [
        ":root {",
        "  --apg-primary: #1E5B5A;",
        "  --apg-accent: #D97706;",
        "  --apg-surface: #ffffff;",
        "  --apg-border: #d0d7de;",
        "  --apg-text: #1f2328;",
        "  --apg-muted: #59636e;",
        "  --apg-bg-canvas: #f6f8fa;",
        "  --apg-bg-card: var(--apg-surface);",
        "  --apg-bg-hover: rgba(0,0,0,0.04);",
        "}",
        "@media (prefers-color-scheme: dark) { :root:not([data-theme='light']) { --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); } }",
        ":root[data-theme='dark'], :root.dark { --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); }",
    ]
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_theme"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            try:
                theme = APG_CAPABILITIES.capability_theme(capability_name)
            except KeyError:
                continue
            theme_name = _css_name(str(theme.get("name") or capability_name))
            tokens = theme.get("tokens", {})
            if isinstance(tokens, dict):
                for token_name, token_value in sorted(tokens.items()):
                    css_var = f"--apg-theme-{theme_name}-{_css_name(str(token_name))}"
                    lines.append(":root { " + css_var + ": " + str(token_value) + "; }")
                    if str(token_name).lower() in {"accent", "primary", "brand"}:
                        lines.append(":root { --apg-accent: var(" + css_var + "); }")
    return "\n".join(lines) + "\n"
    lines.extend([
        # Extended spacing + radius + shadow tokens
        ":root { --apg-radius: 8px; --apg-radius-sm: 4px; --apg-radius-full: 9999px; }",
        ":root { --apg-shadow-sm: 0 1px 2px rgba(0,0,0,0.08); --apg-shadow-md: 0 4px 6px rgba(0,0,0,0.10); --apg-shadow-lg: 0 10px 15px rgba(0,0,0,0.12); }",
        ":root { --apg-sidebar-width: 240px; --apg-topbar-height: 56px; }",
        ":root { --apg-font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; --apg-font-mono: ui-monospace, 'Cascadia Code', 'Fira Mono', monospace; }",
        ":root { --apg-space-1: 4px; --apg-space-2: 8px; --apg-space-3: 12px; --apg-space-4: 16px; --apg-space-6: 24px; --apg-space-8: 32px; }",
        ":root { --apg-duration-fast: 150ms; --apg-duration-base: 200ms; }",
        ":root { --apg-bg-canvas: #f6f8fa; --apg-bg-card: var(--apg-surface); --apg-bg-hover: rgba(0,0,0,0.04); }",
        # Dark mode
        "@media (prefers-color-scheme: dark) { :root { --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); } }",
        # Base styles
        "*, *::before, *::after { box-sizing: border-box; }",
        "body { margin: 0; font-family: var(--apg-font-sans); color: var(--apg-text); background: var(--apg-bg-canvas); line-height: 1.5; font-size: 14px; }",
        "h1 { margin: 0 0 var(--apg-space-4); font-size: 1.5rem; font-weight: 600; color: var(--apg-text); }",
        "h2 { margin: var(--apg-space-6) 0 var(--apg-space-3); font-size: 1.125rem; font-weight: 600; color: var(--apg-text); }",
        "h3 { margin: var(--apg-space-4) 0 var(--apg-space-2); font-size: 1rem; font-weight: 600; color: var(--apg-text); }",
        "a { color: var(--apg-accent); text-decoration: none; transition: opacity var(--apg-duration-fast); }",
        "a:hover { text-decoration: underline; opacity: 0.85; }",
        "p { margin: 0 0 var(--apg-space-3); }",
        # Topbar layout shell
        ".apg-topbar { position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: var(--apg-space-4); height: var(--apg-topbar-height); padding: 0 var(--apg-space-6); border-bottom: 1px solid var(--apg-border); background: var(--apg-surface); box-shadow: var(--apg-shadow-sm); }",
        ".apg-logo { font-weight: 700; font-size: 1rem; color: var(--apg-accent) !important; text-decoration: none !important; letter-spacing: -0.02em; }",
        ".apg-topnav { display: flex; align-items: center; gap: var(--apg-space-1); flex: 1; }",
        ".apg-content { max-width: 1280px; margin: 0 auto; padding: var(--apg-space-6); }",
        # Nav links
        ".apg-nav-link { display: inline-flex; align-items: center; padding: var(--apg-space-2) var(--apg-space-3); border-radius: var(--apg-radius-sm); font-size: 0.875rem; color: var(--apg-text); text-decoration: none !important; transition: background var(--apg-duration-fast); white-space: nowrap; }",
        ".apg-nav-link:hover { background: var(--apg-bg-hover); text-decoration: none !important; opacity: 1; }",
        ".apg-nav-link.active { background: var(--apg-bg-hover); font-weight: 500; }",
        # Card
        ".apg-card { background: var(--apg-bg-card); border: 1px solid var(--apg-border); border-radius: var(--apg-radius); box-shadow: var(--apg-shadow-sm); padding: var(--apg-space-4); margin-bottom: var(--apg-space-4); }",
        ".apg-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--apg-space-3); padding-bottom: var(--apg-space-3); border-bottom: 1px solid var(--apg-border); }",
        # Table
        ".apg-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }",
        ".apg-table thead { background: var(--apg-bg-canvas); }",
        ".apg-table th { padding: var(--apg-space-2) var(--apg-space-3); text-align: left; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--apg-muted); border-bottom: 2px solid var(--apg-border); white-space: nowrap; }",
        ".apg-table td { padding: var(--apg-space-2) var(--apg-space-3); border-bottom: 1px solid var(--apg-border); vertical-align: middle; }",
        ".apg-table tbody tr:hover { background: var(--apg-bg-hover); }",
        ".apg-table-wrap { overflow-x: auto; border: 1px solid var(--apg-border); border-radius: var(--apg-radius); background: var(--apg-bg-card); }",
        # Badge
        ".apg-badge { display: inline-flex; align-items: center; padding: 2px var(--apg-space-2); border-radius: var(--apg-radius-full); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; line-height: 1.6; }",
        ".apg-badge-success { background: #dcfce7; color: #166534; }",
        ".apg-badge-warning { background: #fef9c3; color: #854d0e; }",
        ".apg-badge-danger { background: #fee2e2; color: #991b1b; }",
        ".apg-badge-info { background: #dbeafe; color: #1e40af; }",
        ".apg-badge-neutral { background: var(--apg-bg-hover); color: var(--apg-muted); }",
        # Form
        "form, .apg-form { padding: var(--apg-space-4); background: var(--apg-bg-card); border: 1px solid var(--apg-border); border-radius: var(--apg-radius); box-shadow: var(--apg-shadow-sm); }",
        "label { display: block; margin-bottom: var(--apg-space-1); font-size: 0.875rem; font-weight: 500; color: var(--apg-text); }",
        "input, select, textarea { width: 100%; max-width: 480px; padding: var(--apg-space-2) var(--apg-space-3); border: 1px solid var(--apg-border); border-radius: var(--apg-radius-sm); background: var(--apg-surface); color: var(--apg-text); font-family: var(--apg-font-sans); font-size: 0.875rem; transition: border-color var(--apg-duration-fast); outline: none; }",
        "input:focus, select:focus, textarea:focus { border-color: var(--apg-accent); box-shadow: 0 0 0 3px rgba(18,110,130,0.12); }",
        ".apg-field { margin-bottom: var(--apg-space-4); }",
        # Button
        "button, .apg-btn { display: inline-flex; align-items: center; gap: var(--apg-space-2); padding: var(--apg-space-2) var(--apg-space-4); border: 1px solid var(--apg-accent); border-radius: var(--apg-radius-sm); background: var(--apg-accent); color: white; font-family: var(--apg-font-sans); font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: opacity var(--apg-duration-fast); line-height: 1.5; }",
        "button:hover, .apg-btn:hover { opacity: 0.88; }",
        ".apg-btn-secondary { background: var(--apg-surface); color: var(--apg-text); border-color: var(--apg-border); }",
        ".apg-btn-danger { background: #dc2626; border-color: #dc2626; }",
        # Alert / notice
        "[role=alert] { padding: var(--apg-space-3) var(--apg-space-4); background: #fef9c3; border: 1px solid #fde68a; border-radius: var(--apg-radius-sm); margin-bottom: var(--apg-space-4); font-size: 0.875rem; }",
        # Code / pre
        "pre { padding: var(--apg-space-4); overflow: auto; background: var(--apg-bg-canvas); border: 1px solid var(--apg-border); border-left: 3px solid var(--apg-accent); border-radius: var(--apg-radius); font-family: var(--apg-font-mono); font-size: 0.8rem; line-height: 1.6; }",
        "code { font-family: var(--apg-font-mono); font-size: 0.85em; color: var(--apg-accent); background: var(--apg-bg-hover); padding: 1px 5px; border-radius: 3px; }",
        "pre code { background: transparent; padding: 0; color: inherit; }",
        # Stat card
        ".apg-stat { display: flex; flex-direction: column; gap: var(--apg-space-1); }",
        ".apg-stat-value { font-size: 1.75rem; font-weight: 700; color: var(--apg-text); line-height: 1; }",
        ".apg-stat-label { font-size: 0.75rem; color: var(--apg-muted); text-transform: uppercase; letter-spacing: 0.05em; }",
        ".apg-stat-delta { font-size: 0.8rem; font-weight: 500; }",
        ".apg-stat-delta.up { color: #16a34a; } .apg-stat-delta.down { color: #dc2626; }",
        # Grid helpers
        ".apg-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--apg-space-4); }",
        ".apg-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--apg-space-4); }",
        ".apg-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--apg-space-4); }",
        "@media (max-width: 768px) { .apg-grid-2, .apg-grid-3, .apg-grid-4 { grid-template-columns: 1fr; } }",
        # Utility
        ".apg-flex { display: flex; align-items: center; } .apg-flex-between { justify-content: space-between; }",
        ".apg-mt-4 { margin-top: var(--apg-space-4); } .apg-mb-4 { margin-bottom: var(--apg-space-4); }",
        ".apg-text-muted { color: var(--apg-muted); } .apg-text-sm { font-size: 0.875rem; }",
        ".apg-sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }",
    ])
    return "\n".join(lines) + "\n"


def _html_page(title: str, body: str) -> str:
    safe_title = html.escape(title)
    safe_module = html.escape(MODULE_NAME)
    entity_nav = "".join(
        f'<a class="apg-sidebar-link" href="/ui/entities/{html.escape(quote(str(entity["name"]), safe=""), quote=True)}">{html.escape(str(entity["name"]))}</a>'
        for entity in ENTITIES
        if entity.get("type") not in {"application"}
    ) or '<span class="apg-sidebar-empty">No entities</span>'
    app = describe_application()
    agent_nav = "".join(
        f'<a class="apg-sidebar-link" href="/ui/agents/{html.escape(quote(str(name), safe=""), quote=True)}">{html.escape(str(name))}</a>'
        for name in sorted(app.get("ai_agent_descriptions", {}))
    )
    team_nav = "".join(
        f'<a class="apg-sidebar-link" href="/ui/agent-teams/{html.escape(quote(str(name), safe=""), quote=True)}">{html.escape(str(name))}</a>'
        for name in sorted(app.get("ai_agent_team_descriptions", {}))
    )
    sidebar_html = (
        '<aside id="apg-sidebar" class="apg-sidebar" aria-label="Application navigation">'
        '<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Navigate</p>'
        '<a class="apg-sidebar-link" href="/ui">Dashboard</a>'
        '<a class="apg-sidebar-link" href="/ui/workflows">Workflows</a>'
        '<a class="apg-sidebar-link" href="/ui/databases">Databases</a>'
        '<a class="apg-sidebar-link" href="/ui/marketplace">Marketplace</a></div>'
        f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Entities</p>{entity_nav}</div>'
        + (f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Agents</p>{agent_nav}{team_nav}</div>' if agent_nav or team_nav else "")
        + '</aside><div id="apg-sidebar-backdrop" class="apg-sidebar-backdrop" onclick="apgCloseSidebar()"></div>'
    )
    head_extras = (
        '<script>(function(){try{var m=localStorage.getItem("apg-theme")||"system";var d=document.documentElement;if(m==="dark"||m==="light")d.setAttribute("data-theme",m);else d.removeAttribute("data-theme");d.dataset.themeMode=m;}catch(e){}})();</script>'
        '<link rel="stylesheet" href="/static/apg.css">'
        '<link rel="stylesheet" href="/static/uplot.min.css">'
        '<script defer src="/static/htmx.min.js"></script>'
        '<script defer src="/static/sortable.min.js"></script>'
        '<script defer src="/static/uplot.min.js"></script>'
        '<script defer src="/static/apg-charts.js"></script>'
    )
    toast_js = (
        '<div id="apg-toast-root" class="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none"></div>'
        '<dialog id="apg-confirm-dialog" class="apg-dialog">'
        '<form method="dialog" class="apg-dialog-panel">'
        '<h2 id="apg-confirm-title">Confirm action</h2>'
        '<p id="apg-confirm-message" class="text-sm text-gray-600">Are you sure?</p>'
        '<div class="flex items-center justify-end gap-2 mt-4">'
        '<button value="cancel" class="apg-btn apg-btn-secondary" type="submit">Cancel</button>'
        '<button value="confirm" class="apg-btn apg-btn-danger" type="submit">Delete</button>'
        '</div></form></dialog>'
        '<script>'
        'function apgToast(m,t){'
        'var c=t==="error"?"bg-red-600":"bg-gray-900";'
        'var el=document.createElement("div");'
        'el.className=c+" text-white text-sm font-medium px-4 py-2.5 rounded-xl shadow-lg pointer-events-auto transition-all duration-300 opacity-0 translate-y-2";'
        'el.textContent=m;'
        'document.getElementById("apg-toast-root").appendChild(el);'
        'requestAnimationFrame(function(){el.classList.remove("opacity-0","translate-y-2");});'
        'setTimeout(function(){el.classList.add("opacity-0");setTimeout(function(){el.remove();},300);},3000);'
        '}'
        'document.addEventListener("htmx:afterOnLoad",function(e){'
        'var t=e.detail.xhr.getResponseHeader("HX-Trigger");'
        'if(!t)return;'
        'try{var d=JSON.parse(t);if(d.apgToast)apgToast(d.apgToast.msg,d.apgToast.type||"success");}catch(ex){}'
        '});'
        'function apgApplyTheme(m){var d=document.documentElement;if(m==="dark"||m==="light")d.setAttribute("data-theme",m);else d.removeAttribute("data-theme");d.dataset.themeMode=m;var b=document.getElementById("apg-theme-toggle");if(b){b.setAttribute("aria-label","Theme: "+m);b.textContent=m==="dark"?"Dark":m==="light"?"Light":"System";}}'
        'function apgCycleTheme(){var order=["system","light","dark"];var cur=localStorage.getItem("apg-theme")||"system";var next=order[(order.indexOf(cur)+1)%order.length];localStorage.setItem("apg-theme",next);apgApplyTheme(next);}'
        'document.addEventListener("DOMContentLoaded",function(){apgApplyTheme(localStorage.getItem("apg-theme")||"system");});'
        'function apgConfirm(message,ok){var d=document.getElementById("apg-confirm-dialog");if(!d||!d.showModal){var nativeConfirm=window["confirm"];if(nativeConfirm&&nativeConfirm(message))ok();return;}document.getElementById("apg-confirm-message").textContent=message;var done=false;function close(){if(done)return;done=true;d.removeEventListener("close",onclose);}function onclose(){var v=d.returnValue;close();if(v==="confirm")ok();}d.addEventListener("close",onclose);d.showModal();}'
        'function apgConfirmSubmit(form,message){apgConfirm(message||"Delete this record?",function(){form.dataset.apgConfirmed="1";form.requestSubmit();});return false;}'
        'document.addEventListener("DOMContentLoaded",function(){document.querySelectorAll(".apg-topnav a").forEach(function(a){if(a.getAttribute("href")===location.pathname){a.classList.add("active");a.setAttribute("aria-current","page");}});});'
        'function apgSetSidebar(collapsed){document.documentElement.classList.toggle("apg-sidebar-collapsed",collapsed);try{localStorage.setItem("apg-sidebar-collapsed",collapsed?"1":"0");}catch(e){}}'
        'function apgToggleSidebar(){if(matchMedia("(max-width: 767px)").matches){document.documentElement.classList.toggle("apg-sidebar-open");}else{apgSetSidebar(!document.documentElement.classList.contains("apg-sidebar-collapsed"));}}'
        'function apgCloseSidebar(){document.documentElement.classList.remove("apg-sidebar-open");}'
        'try{if(localStorage.getItem("apg-sidebar-collapsed")==="1")document.documentElement.classList.add("apg-sidebar-collapsed");}catch(e){}'
        'document.addEventListener("keydown",function(e){if(e.key==="Escape")apgCloseSidebar();});'
        '</script>'
    )
    skeleton_css = (
        '<style>'
        '.apg-skeleton{'
        '  background:linear-gradient(90deg,#f0f0f0 25%,#e0e0e0 50%,#f0f0f0 75%);'
        '  background-size:200% 100%;'
        '  animation:apg-shimmer 1.5s infinite;'
        '  border-radius:4px;'
        '}'
        '@keyframes apg-shimmer{'
        '  0%{background-position:200% 0}'
        '  100%{background-position:-200% 0}'
        '}'
        '.apg-loading .apg-skeleton-row{height:40px;margin-bottom:8px;}'
        '.htmx-request .apg-content-area{opacity:0.6;transition:opacity 0.2s;}'
        '</style>'
    )
    cmd_palette_html = '<div id="apg-cmd" class="hidden fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" onclick="if(event.target===this)apgCmdClose()"><div class="mx-auto mt-[15vh] max-w-xl bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden"><div class="flex items-center gap-3 px-4 py-3 border-b border-gray-100"><svg class="w-4 h-4 text-gray-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9 a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clip-rule="evenodd"/></svg><input id="apg-cmd-input" type="text" placeholder="Search records, entities..." autocomplete="off" class="flex-1 text-sm outline-none placeholder-gray-400" oninput="apgCmdSearch(this.value)"><kbd class="text-xs text-gray-400 border border-gray-200 rounded px-1.5 py-0.5">Esc</kbd></div><div id="apg-cmd-results" class="max-h-80 overflow-y-auto py-2"><p class="text-xs text-gray-400 text-center py-8">Type to search...</p></div></div></div><script>document.addEventListener("keydown",function(e){if((e.metaKey||e.ctrlKey)&&e.key==="k"){e.preventDefault();apgCmdOpen();}if(e.key==="Escape")apgCmdClose();});function apgCmdOpen(){document.getElementById("apg-cmd").classList.remove("hidden");document.getElementById("apg-cmd-input").focus();}function apgCmdClose(){document.getElementById("apg-cmd").classList.add("hidden");document.getElementById("apg-cmd-input").value="";document.getElementById("apg-cmd-results").innerHTML=\'<p class="text-xs text-gray-400 text-center py-8">Type to search...</p>\';}var _cmdTimer;function apgCmdSearch(q){clearTimeout(_cmdTimer);if(!q.trim()){document.getElementById("apg-cmd-results").innerHTML=\'<p class="text-xs text-gray-400 text-center py-8">Type to search...</p>\';return;}_cmdTimer=setTimeout(function(){fetch("/api/search?q="+encodeURIComponent(q)).then(function(r){return r.json();}).then(function(d){var el=document.getElementById("apg-cmd-results");if(!d.results||!d.results.length){el.innerHTML=\'<p class="text-xs text-gray-400 text-center py-8">No results</p>\';return;}el.innerHTML=d.results.map(function(r){return \'<a href="/ui/entities/\'+encodeURIComponent(r.entity)+\'/\'+encodeURIComponent(r.id)+\'"\'+\'  onclick="apgCmdClose()"\'+\'  class="flex items-center gap-3 px-4 py-2.5 hover:bg-gray-50 transition-colors group">\'+\'<span class="w-6 h-6 rounded-md bg-blue-50 flex items-center justify-center text-xs font-bold text-blue-600 flex-shrink-0">\'+r.entity.charAt(0).toUpperCase()+\'</span>\'+\'<div class="min-w-0"><p class="text-sm font-medium text-gray-900 truncate">\'+r.label+\'</p>\'+\'<p class="text-xs text-gray-400 truncate">\'+r.entity+\'</p></div>\'+\'</a>\';}).join("");});},200);}</script>'
    return (
        "<!doctype html>"
        '<html lang="en" class="h-full"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{head_extras}"
        f"{skeleton_css}"
        '<link rel="stylesheet" href="/theme.css">'
        f"<title>{safe_title} — {safe_module}</title>"
        "</head>"
        '<body class="min-h-full bg-gray-50 text-gray-900">'
        '<a class="apg-skip-link" href="#content">Skip to content</a>'
        f'<header class="apg-topbar sticky top-0 z-50" role="banner">'
        f'  <button class="apg-icon-btn" type="button" onclick="apgToggleSidebar()" aria-label="Toggle navigation">☰</button>'
        f'  <a class="apg-logo" href="/ui">{safe_module}</a>'
        f'  <nav class="apg-topnav ml-4">'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui">Home</a>'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui/workflows">⚡ Workflows</a>'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui/marketplace">Marketplace</a>'
        f'  </nav>'
        f'  <button id="apg-theme-toggle" class="apg-btn apg-btn-secondary apg-theme-toggle" type="button" onclick="apgCycleTheme()" aria-label="Theme: system">System</button>'
        f'</header>'
        f'{sidebar_html}'
        f'<main class="apg-content apg-shell-content" id="content" tabindex="-1">{body}</main>'
        f"{toast_js}"
        f"{cmd_palette_html}"
        "</body></html>"
    )


def _jinja_required_page(title: str = "Application UI") -> str:
    safe_title = html.escape(title)
    return (
        f'<section class="apg-card">'
        f'<h1>{safe_title}</h1>'
        f'<p>This application requires Jinja2 — pip install -r requirements.txt.</p>'
        f'</section>'
    )


def _render_template(template_name: str, **context: Any) -> str | None:
    """Render a Jinja2 template from APG_UI_TEMPLATES dict if Jinja2 is available.

    Returns None when Jinja2 is not installed — callers fall back to the existing
    f-string builder so the generated app works with zero extra dependencies.

    APG_UI_TEMPLATES is injected at module level when the compiler embeds templates
    as string literals. In standalone mode (running code_generator.py directly),
    templates are loaded from compiler/templates/*.j2 relative to this file.
    """
    try:
        from jinja2 import Environment, DictLoader, BaseLoader, FileSystemLoader, ChoiceLoader  # type: ignore[import]
    except ImportError:
        return None
    try:
        # APG_UI_TEMPLATES injected at compile time takes priority
        templates: dict[str, str] = globals().get("APG_UI_TEMPLATES", {})
        if templates:
            env = Environment(loader=DictLoader(templates), autoescape=True)
        else:
            # Standalone: load from compiler/templates/ directory
            import pathlib
            tmpl_dir = pathlib.Path(__file__).parent / "templates"
            if not tmpl_dir.exists():
                return None
            env = Environment(loader=FileSystemLoader(str(tmpl_dir)), autoescape=True)
            # Adjust template name for standalone (files have .j2 extension, no nested path)
            if not template_name.endswith(".j2"):
                template_name = template_name.replace(".html", ".html.j2") if ".html" in template_name else template_name + ".j2"
        # Add url encode filter
        env.filters["urlencode"] = lambda s: __import__("urllib.parse", fromlist=["quote"]).quote(str(s), safe="")
        tmpl = env.get_template(template_name)
        return tmpl.render(**context)
    except Exception:
        return None


def _entity_spec(entity_name: str) -> Dict[str, Any] | None:
    for entity in ENTITIES:
        if entity["name"] == entity_name:
            return dict(entity)
    return None


def _field_specs(entity_name: str) -> list[Dict[str, Any]]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return []
    fields = entity.get("fields") or []
    if fields:
        return [dict(field) for field in fields if isinstance(field, dict)]
    return [
        {"name": property_name, "type": "any", "required": True}
        for property_name in entity.get("properties", [])
    ]


def _json_schema_type(apg_type: str) -> str:
    normalized = apg_type.lower()
    if normalized in {"str", "string", "text", "varchar", "char", "email", "uuid", "date", "datetime", "timestamp"}:
        return "string"
    if normalized in {"int", "integer", "serial", "bigint", "smallint"}:
        return "integer"
    if normalized in {"float", "double", "decimal", "number", "numeric", "money"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"list", "array", "set"}:
        return "array"
    if normalized in {"dict", "map", "object", "json", "jsonb"}:
        return "object"
    return "string"


def _value_matches_type(value: Any, apg_type: str) -> bool:
    expected = _json_schema_type(apg_type)
    if value is None:
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _coerce_value_for_type(value: Any, apg_type: str) -> Any:
    if not isinstance(value, str):
        return value
    expected = _json_schema_type(apg_type)
    if expected == "integer":
        try:
            return int(value.strip())
        except ValueError:
            return value
    if expected == "number":
        try:
            return float(value.strip())
        except ValueError:
            return value
    if expected == "boolean":
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return value


def coerce_record_types(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(record)
    for field in _field_specs(entity_name):
        field_name = str(field["name"])
        if field_name in coerced:
            coerced[field_name] = _coerce_value_for_type(
                coerced[field_name],
                str(field.get("type", "any")),
            )
    return coerced


def validate_record(entity_name: str, record: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    errors: list[str] = []
    fields = _field_specs(entity_name)
    for field in fields:
        field_name = str(field["name"])
        if not partial and field.get("required", False) and field_name not in record:
            errors.append(f"{field_name} is required")
            continue
        if field_name in record and not _value_matches_type(record[field_name], str(field.get("type", "any"))):
            errors.append(f"{field_name} must be {_json_schema_type(str(field.get('type', 'any')))}")
    return {
        "valid": not errors,
        "entity": entity_name,
        "errors": errors,
    }


def relationship_graph() -> Dict[str, Any]:
    nodes = [
        {"id": str(entity["name"]), "name": str(entity["name"]), "type": str(entity["type"])}
        for entity in ENTITIES
    ]
    table_nodes_by_name: Dict[str, list[str]] = {}
    for entity in ENTITIES:
        database_name = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                node_id = f"{database_name}.{schema_name}.{table_name}"
                nodes.append({
                    "id": node_id,
                    "name": table_name,
                    "type": "database_table",
                    "database": database_name,
                    "schema": schema_name,
                })
                table_nodes_by_name.setdefault(table_name.lower(), []).append(node_id)
                table_nodes_by_name.setdefault(f"{schema_name}.{table_name}".lower(), []).append(node_id)
    entity_names = {str(entity["name"]) for entity in ENTITIES}
    entity_names_by_lower = {name.lower(): name for name in entity_names}
    edges: list[Dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for entity in ENTITIES:
        source = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                table_node = f"{source}.{schema_name}.{table_name}"
                contains_key = (source, table_node, schema_name, "contains_table")
                if contains_key not in seen_edges:
                    edges.append({
                        "from": source,
                        "to": table_node,
                        "field": schema_name,
                        "relationship": "contains_table",
                    })
                    seen_edges.add(contains_key)
                for column in table.get("columns", []):
                    reference = column.get("reference") if isinstance(column, dict) else None
                    if not isinstance(reference, dict):
                        continue
                    target_table = str(reference.get("table", ""))
                    target_schema = str(reference.get("schema", ""))
                    if target_schema:
                        targets = table_nodes_by_name.get(f"{target_schema}.{target_table}".lower(), [])
                    else:
                        targets = table_nodes_by_name.get(f"{schema_name}.{target_table}".lower(), [])
                        if not targets:
                            targets = table_nodes_by_name.get(target_table.lower(), [])
                    target = targets[0] if len(targets) == 1 else None
                    if not target:
                        continue
                    edge_key = (
                        table_node,
                        target,
                        str(column.get("name", "")),
                        str(reference.get("relationship", "db_ref")),
                    )
                    if edge_key not in seen_edges:
                        edges.append({
                            "from": table_node,
                            "to": target,
                            "field": str(column.get("name", "")),
                            "relationship": str(reference.get("relationship", "db_ref")),
                            "target_column": str(reference.get("column", "")),
                        })
                        seen_edges.add(edge_key)
        for field in _field_specs(source):
            field_name = str(field["name"])
            field_type = str(field.get("type", ""))
            target = None
            relationship = "references"
            if field_type in entity_names:
                target = field_type
                relationship = "typed_as"
            elif field_type.lower() in entity_names_by_lower:
                target = entity_names_by_lower[field_type.lower()]
                relationship = "typed_as"
            elif field_name.endswith("_id"):
                candidate = field_name[:-3]
                target = entity_names_by_lower.get(candidate.lower())
            if target and target != source:
                edge_key = (source, target, field_name, relationship)
                if edge_key not in seen_edges:
                    edges.append({
                        "from": source,
                        "to": target,
                        "field": field_name,
                        "relationship": relationship,
                    })
                    seen_edges.add(edge_key)
    return {"nodes": nodes, "edges": edges}


# ── Workflow engine ─────────────────────────────────────────────────────────

_WORKFLOW_PATTERNS: list[tuple[list[str], str, str, str]] = [
    # (name_keywords, workflow_name_fmt, description_fmt, icon)
    (["loan", "credit", "lending"], "Apply for {entity_name}", "Step-by-step {entity_name} application and approval", "💳"),
    (["repayment", "payment", "installment"], "Record {entity_name}", "Capture payment details and update balances", "💰"),
    (["member", "customer", "client", "subscriber"], "Register {entity_name}", "Complete {entity_name} onboarding and KYC", "👤"),
    (["patient", "beneficiary", "recipient"], "Enroll {entity_name}", "Register and profile the {entity_name}", "🏥"),
    (["ticket", "incident", "issue", "fault"], "Log {entity_name}", "Capture incident details and assign for resolution", "🎫"),
    (["change", "request", "order"], "Submit {entity_name}", "Prepare and route the {entity_name} for approval", "📋"),
    (["asset", "equipment", "device"], "Register {entity_name}", "Record asset details, location and assignment", "🖥️"),
    (["grant", "award", "fund"], "Register {entity_name}", "Document {entity_name} details and donor linkage", "🌍"),
    (["contribution", "deposit", "saving"], "Record {entity_name}", "Capture and confirm the {entity_name}", "🏦"),
    (["farmer", "supplier", "vendor"], "Onboard {entity_name}", "Complete {entity_name} registration and verification", "🌱"),
    (["produce", "product", "item", "listing"], "List {entity_name}", "Create a new {entity_name} listing with pricing", "📦"),
    (["appointment", "booking", "schedule"], "Book {entity_name}", "Select date, time and details for the {entity_name}", "📅"),
    (["prescription", "medication", "drug"], "Issue {entity_name}", "Document prescribed treatment and dosage", "💊"),
    (["invoice", "bill", "charge"], "Generate {entity_name}", "Prepare and issue the {entity_name}", "🧾"),
    (["score", "assessment", "evaluation", "rating"], "Run {entity_name}", "Collect inputs and compute the {entity_name}", "📊"),
]
_DEFAULT_WORKFLOW = ("Create {entity_name}", "Fill in all required fields to create a new {entity_name}", "➕")

def _workflow_meta(entity_name: str) -> tuple[str, str, str]:
    lower = entity_name.lower()
    for keywords, name_fmt, desc_fmt, icon in _WORKFLOW_PATTERNS:
        if any(kw in lower for kw in keywords):
            return name_fmt.format(entity_name=entity_name), desc_fmt.format(entity_name=entity_name), icon
    name_fmt, desc_fmt, icon = _DEFAULT_WORKFLOW
    return name_fmt.format(entity_name=entity_name), desc_fmt.format(entity_name=entity_name), icon


def _group_fields_into_steps(entity_name: str, fields: list[dict]) -> list[dict]:
    """Group entity fields into logical wizard steps."""
    # Categorise fields
    id_fields, ref_fields, core_fields, numeric_fields, date_fields, other_fields = [], [], [], [], [], []
    tables = SEMANTIC_MODEL.get("tables", {})
    table_fields = tables.get(entity_name, {}).get("fields", {})

    for f in fields:
        fname = str(f["name"])
        ftype = str(f.get("type", "")).lower()
        rel = table_fields.get(fname, {}).get("relationship")
        real_rel = rel and rel.get("target_table") and rel["target_table"] in {e["name"] for e in ENTITIES}

        if fname in {"id", "_revision"}:
            id_fields.append(f)
        elif real_rel:
            ref_fields.append(f)
        elif ftype in {"float", "double", "decimal", "money", "int", "integer", "number"}:
            numeric_fields.append(f)
        elif ftype in {"date", "datetime", "timestamp"}:
            date_fields.append(f)
        elif any(fname.endswith(sfx) for sfx in ("_id", "_code", "_number", "_ref", "_key")):
            core_fields.append(f)
        else:
            other_fields.append(f)

    steps = []
    # Step 1: Identity (own ID + code/number fields)
    s1 = id_fields + core_fields
    if s1:
        steps.append({"title": "Identity", "subtitle": f"Enter the unique identifiers for this {entity_name}", "fields": s1})
    # Step 2: Core details (name/title/description/type/status/category)
    priority = ["name", "full_name", "title", "description", "type", "category", "status",
                "gender", "email", "phone", "nationality", "country"]
    prio_fields = [f for f in other_fields if str(f["name"]) in priority]
    rest_other = [f for f in other_fields if str(f["name"]) not in priority]
    if prio_fields:
        steps.append({"title": "Core Details", "subtitle": "Enter the primary descriptive information", "fields": prio_fields})
    # Step 3: Relationships (FK dropdowns)
    if ref_fields:
        steps.append({"title": "Relationships", "subtitle": "Link to related records", "fields": ref_fields})
    # Step 4: Financial / numeric
    if numeric_fields:
        steps.append({"title": "Amounts & Rates", "subtitle": "Enter financial and numeric values", "fields": numeric_fields})
    # Step 5: Dates
    if date_fields:
        steps.append({"title": "Dates & Schedule", "subtitle": "Set relevant dates and deadlines", "fields": date_fields})
    # Step 6: Remaining details
    if rest_other:
        # Split into chunks of max 5 fields per step
        for i in range(0, len(rest_other), 5):
            chunk = rest_other[i:i+5]
            steps.append({"title": "Additional Details" if i == 0 else "More Details", "subtitle": "Provide any additional information", "fields": chunk})
    # Ensure at least one step
    if not steps:
        steps.append({"title": "Details", "subtitle": f"Enter information for this {entity_name}", "fields": fields})
    return steps


def _build_app_workflows() -> dict[str, list[dict]]:
    result = {}
    for entity in ENTITIES:
        if entity.get("type") in {"application"}:
            continue
        name = entity["name"]
        fields = entity.get("fields") or []
        wf_name, wf_desc, wf_icon = _workflow_meta(name)
        steps = _group_fields_into_steps(name, fields)
        result[name] = [{
            "id": f"create_{name.lower()}",
            "name": wf_name,
            "description": wf_desc,
            "icon": wf_icon,
            "entity": name,
            "action": "create",
            "steps": steps,
        }]
    return result

APP_WORKFLOWS: dict[str, list[dict]] = _build_app_workflows()


def _ui_workflow_list_html() -> tuple[int, str]:
    """Render the list of all available workflows across all entities."""
    total = sum(len(wfs) for wfs in APP_WORKFLOWS.values())
    workflow_items = []
    for entity_name, workflows in APP_WORKFLOWS.items():
        for wf in workflows:
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(wf["id"], safe=""), quote=True)
            workflow_items.append({
                "id": wf["id"],
                "name": wf["name"],
                "description": wf["description"],
                "icon": wf["icon"],
                "entity": entity_name,
                "step_count": len(wf["steps"]),
                "steps": wf["steps"],
                "href": f"/ui/workflows/{safe_entity}/{safe_wf_id}",
            })
    tmpl_body = _render_template(
        "workflow_list.html.j2",
        workflows=workflow_items,
        total=total,
        entity_count=len(APP_WORKFLOWS),
    )
    return 200, _html_page("Workflows", tmpl_body if tmpl_body is not None else _jinja_required_page("Workflows"))


def _ui_workflow_wizard_html(
    entity_name: str,
    workflow_id: str,
    step_index: int = 0,
    accumulated: dict | None = None,
    error: str = "",
) -> tuple[int, str]:
    """Render one step of the multi-step workflow wizard."""
    entity_workflows = APP_WORKFLOWS.get(entity_name, [])
    wf = next((w for w in entity_workflows if w["id"] == workflow_id), None)
    if wf is None:
        return 404, _html_page("Workflow not found", f"<h1>Workflow not found</h1>")

    steps = wf["steps"]
    total_steps = len(steps)
    accumulated = accumulated or {}

    # Final step: show summary and create record
    if step_index >= total_steps:
        record_data = dict(accumulated)
        result = create_record(entity_name, record_data)
        if result.get("ok"):
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)
            tmpl_body = _render_template(
                "workflow_wizard.html.j2",
                completed=True,
                workflow=wf,
                entity_name=entity_name,
                safe_entity=safe_entity,
                safe_workflow_id=safe_wf_id,
            )
            return 200, _html_page(wf["name"], tmpl_body if tmpl_body is not None else _jinja_required_page(wf["name"]))
        else:
            error = result.get("error") or "Failed to create record"
            step_index = total_steps - 1  # Stay on last step

    step = steps[min(step_index, total_steps - 1)]
    step_fields = step.get("fields", [])
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)

    progress = []
    for i, item in enumerate(steps):
        complete = i < step_index
        current = i == step_index
        progress.append({
            "title": item["title"],
            "label": "✓" if complete else str(i + 1),
            "class_name": "text-blue-600" if current or complete else "text-gray-400 opacity-60",
            "badge_class": "bg-blue-600 text-white" if current or complete else "bg-gray-200 text-gray-500",
        })

    # Hidden fields to carry accumulated data through steps
    hidden_fields = "".join(
        f'<input type="hidden" name="__acc_{html.escape(k, quote=True)}" value="{html.escape(str(v), quote=True)}">'
        for k, v in accumulated.items()
    )

    # Current step fields
    step_inputs = "".join(_ui_field_input_html(f, entity_name) for f in step_fields)

    # Navigation buttons
    is_last = step_index == total_steps - 1
    next_label = "Create Record ✓" if is_last else "Next →"
    next_url = f"/ui/workflows/{safe_entity}/{safe_wf_id}/step/{step_index + 1}"

    error_html = (
        f'<div role="alert" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">⚠ {html.escape(error)}</div>'
        if error else ""
    )

    tmpl_body = _render_template(
        "workflow_wizard.html.j2",
        completed=False,
        workflow=wf,
        entity_name=entity_name,
        safe_entity=safe_entity,
        safe_workflow_id=safe_wf_id,
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        progress=progress,
        hidden_fields=hidden_fields,
        step_inputs=step_inputs,
        next_url=next_url,
        next_label=next_label,
        error=error,
    )
    return 200, _html_page(wf["name"], tmpl_body if tmpl_body is not None else _jinja_required_page(wf["name"]))


def _landing_page_html() -> str:
    """Render the application landing page using landing.html.j2."""
    theme = APG_CAPABILITIES.capability_theme(MODULE_NAME) if APG_CAPABILITIES and hasattr(APG_CAPABILITIES, "capability_theme") else {}
    tokens = theme.get("tokens", {}) if isinstance(theme, dict) else {}
    theme_primary = tokens.get("color.primary") or "#1E5B5A"
    theme_accent = tokens.get("color.accent") or "#D97706"
    landing_style = os.environ.get("APG_LANDING_STYLE", LANDING_STYLE)
    api_links = [
        {"url": "/ui",            "label": "Open App"},
        {"url": "/manifest",      "label": "Manifest"},
        {"url": "/openapi.json",  "label": "OpenAPI"},
        {"url": "/capabilities",  "label": "Capabilities"},
        {"url": "/metrics",       "label": "Metrics"},
        {"url": "/self-test",     "label": "Self-Test"},
    ]
    stats = [
        {"value": len([e for e in ENTITIES if e.get("type") not in {"application"}]), "label": "Entities"},
        {"value": len(describe_application().get("capabilities", [])), "label": "Capabilities"},
        {"value": len(describe_application().get("ai_agents", [])), "label": "AI Agents"},
        {"value": sum(len(list_records(e["name"])) for e in ENTITIES if e.get("type") not in {"application"}), "label": "Records"},
    ]
    rendered = _render_template(
        "landing.html.j2",
        module_name=MODULE_NAME,
        module_description=MODULE_DESCRIPTION or "",
        entities=ENTITIES,
        theme_primary=theme_primary,
        theme_accent=theme_accent,
        landing_style=landing_style,
        api_links=api_links,
        stats=stats,
    )
    if rendered is not None:
        return rendered
    # Fallback: redirect to /ui
    return (
        "<!doctype html><html><head>"
        f'<meta http-equiv="refresh" content="0; url=/ui">'
        f"<title>{html.escape(MODULE_NAME)}</title>"
        "</head><body></body></html>"
    )


def _ui_index_html() -> str:
    app = describe_application()
    entity_links = "".join(
        f'<li><a href="/ui/entities/{html.escape(entity["name"], quote=True)}">'
        f'{html.escape(entity["name"])}</a> '
        f'<code>{html.escape(entity["type"])}</code></li>'
        for entity in ENTITIES
    )
    if not entity_links:
        entity_links = "<li>No APG entities declared.</li>"
    database_links = "".join(
        f'<li><a href="/ui/databases">{html.escape(database["name"])}</a> '
        f'<code>{len(database.get("schemas", []))} schema(s)</code></li>'
        for database in app.get("databases", [])
    )
    if not database_links:
        database_links = "<li>No databases declared.</li>"
    application_route_links = "".join(
        f'<li><a href="{html.escape(route, quote=True)}">{html.escape(route)}</a> '
        f'<code>{html.escape(str(screen.get("application", "application")))}</code></li>'
        for route, screen in sorted(app.get("application_routes", {}).items())
    )
    if not application_route_links:
        application_route_links = "<li>No application routes declared.</li>"
    capability_route_links = "".join(
        f'<li><a href="{html.escape(route, quote=True)}">{html.escape(route)}</a> '
        f'<code>{html.escape(str(screen.get("capability", "capability")))}</code></li>'
        for route, screen in sorted(app.get("ui_routes", {}).items())
    )
    if not capability_route_links:
        capability_route_links = "<li>No capability screens declared.</li>"
    capability_links = "".join(
        f'<li><a href="/ui/capabilities/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("capabilities", [])
    )
    if not capability_links:
        capability_links = "<li>No capabilities declared.</li>"
    agent_links = "".join(
        f'<li><a href="/ui/agents/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("ai_agents", [])
    )
    if not agent_links:
        agent_links = "<li>No AI agents declared.</li>"
    team_links = "".join(
        f'<li><a href="/ui/agent-teams/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("ai_agent_teams", [])
    )
    if not team_links:
        team_links = "<li>No AI agent teams declared.</li>"

    # Prefer Jinja2 template; fall back to f-string for zero-dep mode
    api_links = [
        {"url": "/manifest",       "label": "Manifest JSON"},
        {"url": "/component.json", "label": "Component JSON"},
        {"url": "/capabilities",   "label": "Capabilities"},
        {"url": "/agents",         "label": "Agents"},
        {"url": "/events",         "label": "Events"},
        {"url": "/metrics",        "label": "Metrics"},
        {"url": "/self-test",      "label": "Self-Test"},
        {"url": "/openapi.json",   "label": "API Contract"},
        {"url": "/ui/databases",   "label": "Databases"},
    ]
    dashboard = _ui_dashboard_context(app)
    tmpl_body = _render_template(
        "app_index.html.j2",
        module_name=html.escape(MODULE_NAME),
        module_description=html.escape(MODULE_DESCRIPTION or "Generated APG application"),
        entities=ENTITIES,
        capabilities=app.get("capabilities", []),
        databases=app.get("databases", []),
        application_routes=app.get("application_routes", {}),
        ui_routes=app.get("ui_routes", {}),
        agents=app.get("ai_agents", []),
        agent_teams=app.get("ai_agent_teams", []),
        api_links=api_links,
        dashboard_stats=dashboard["stats"],
        status_charts=dashboard["status_charts"],
        recent_activity=dashboard["recent_activity"],
        workflow_summary=dashboard["workflow_summary"],
        agent_summary=dashboard["agent_summary"],
    )
    if tmpl_body is not None:
        return _html_page(MODULE_NAME, tmpl_body)

    # Fallback: original f-string builder
    body = (
        f"<h1>{html.escape(MODULE_NAME)}</h1>"
        f"<p>{html.escape(MODULE_DESCRIPTION or 'Generated APG application')}</p>"
        '<nav><a href="/manifest">Manifest JSON</a> | '
        '<a href="/component.json">Component JSON</a> | '
        '<a href="/capabilities">Capabilities</a> | '
        '<a href="/agents">Agents</a> | '
        '<a href="/events">Events</a> | '
        '<a href="/metrics">Metrics</a> | '
        '<a href="/self-test">Self-Test</a> | '
        '<a href="/ui/databases">Databases</a> | '
        '<a href="/openapi.json">API Contract</a></nav>'
        "<h2>Application Routes</h2>"
        f"<ul>{application_route_links}</ul>"
        "<h2>Capability Screens</h2>"
        f"<ul>{capability_route_links}</ul>"
        "<h2>Entities</h2>"
        f"<ul>{entity_links}</ul>"
        "<h2>Databases</h2>"
        f"<ul>{database_links}</ul>"
        "<h2>Capabilities</h2>"
        f"<ul>{capability_links}</ul>"
        "<h2>AI Agents</h2>"
        f"<ul>{agent_links}</ul>"
        "<h2>AI Agent Teams</h2>"
        f"<ul>{team_links}</ul>"
    )
    return _html_page(MODULE_NAME, body)


def _status_field_name(fields: list[Dict[str, Any]]) -> str | None:
    for candidate in ("status", "state", "stage", "phase"):
        for field in fields:
            if str(field.get("name", "")).lower() == candidate:
                return str(field.get("name"))
    return None


def _chart_json(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True)


def _ui_dashboard_context(app: Dict[str, Any]) -> Dict[str, Any]:
    stats = []
    status_charts = []
    for entity in ENTITIES:
        if entity.get("type") in {"application"}:
            continue
        entity_name = str(entity["name"])
        records = list_records(entity_name)
        spark = {"type": "sparkline", "title": f"{entity_name} records", "data": [{"x": i, "y": len(records)} for i in range(30)], "empty": "No records yet"}
        stats.append({
            "label": entity_name,
            "value": len(records),
            "delta": "0%",
            "chart_id": f"chart-stat-{_css_name(entity_name)}",
            "spec_json": _chart_json(spark),
        })
        status_field = _status_field_name(_field_specs(entity_name))
        if status_field:
            counts: Dict[str, int] = {}
            for record in records:
                key = str(record.get(status_field) or "Unspecified")
                counts[key] = counts.get(key, 0) + 1
            status_charts.append({
                "entity": entity_name,
                "field": status_field,
                "chart_id": f"chart-status-{_css_name(entity_name)}",
                "spec_json": _chart_json({
                    "type": "donut",
                    "title": f"{entity_name} by {status_field}",
                    "data": [{"label": key, "value": value} for key, value in sorted(counts.items())],
                    "empty": f"No {status_field} data yet",
                }),
            })
    return {
        "stats": stats,
        "status_charts": status_charts,
        "recent_activity": EVENT_LOG[-8:],
        "workflow_summary": {"workflow_count": sum(len(items) for items in APP_WORKFLOWS.values()), "run_count": len(WORKFLOW_RUNS)},
        "agent_summary": {"agent_count": len(app.get("ai_agent_descriptions", {})), "team_count": len(app.get("ai_agent_team_descriptions", {}))},
    }


def _ui_database_catalog_html() -> tuple[int, str]:
    status = database_status()
    status_code = 200 if status["valid"] else 422
    status_label = "valid" if status["valid"] else "invalid"
    databases = list_databases()
    graph = relationship_graph()
    relationships = [
        {"source": edge.get("source", ""), "target": edge.get("target", "")}
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
    ]
    tmpl_body = _render_template(
        "database_catalog.html.j2",
        status=status,
        status_label=status_label,
        databases=databases,
        relationships=relationships,
        validation_json=json.dumps(status["validation"], indent=2, sort_keys=True),
    )
    return status_code, _html_page("Databases", tmpl_body if tmpl_body is not None else _jinja_required_page("Databases"))


def _field_relationship(entity_name: str, field_name: str) -> Dict[str, Any] | None:
    """Return relationship metadata for a field from SEMANTIC_MODEL, or None."""
    tables = SEMANTIC_MODEL.get("tables", {})
    table = tables.get(entity_name, {})
    field_info = table.get("fields", {}).get(field_name, {})
    rel = field_info.get("relationship")
    if not rel or not rel.get("target_table"):
        return None
    # Skip relationships to synthetic types like 'date' that aren't real entities
    target = rel["target_table"]
    if target not in {e["name"] for e in ENTITIES}:
        return None
    return rel


def _best_display_field(target_entity: str) -> str:
    """Return the best human-readable field name for a FK select option label."""
    priority = ["name", "full_name", "title", "label", "description",
                "company_name", "display_name", "username", "email",
                "first_name", "code", "number", "reference"]
    fields = _field_specs(target_entity)
    field_names = [str(f["name"]) for f in fields]
    for candidate in priority:
        if candidate in field_names:
            return candidate
    # Fall back to first non-id string field
    for f in fields:
        if str(f["name"]) not in {"id", "_revision", "_created_at"} and _json_schema_type(str(f.get("type", ""))) == "string":
            return str(f["name"])
    return "id"


def _fk_select_options(target_entity: str, current_value: str = "", form_id: str = "") -> str:
    """Render <option> elements for a foreign key select, populated from live records."""
    records = list_records(target_entity)
    display_field = _best_display_field(target_entity)
    blank_label = html.escape(f"— select {target_entity} —")
    options = [f'<option value="">{blank_label}</option>']
    for rec in records:
        val = str(rec.get("id", ""))
        label_val = rec.get(display_field) or val
        display = html.escape(str(label_val))
        sel = ' selected' if val == current_value else ''
        options.append(f'<option value="{html.escape(val, quote=True)}"{sel}>{display}</option>')
    return "".join(options)


def _ui_field_semantic(field_name: str, field_type: str) -> str:
    name = field_name.lower()
    ft = field_type.lower()
    if "email" in name: return "email"
    if any(x in name for x in ("phone", "mobile", "tel")): return "phone"
    if any(x in name for x in ("url", "website", "link", "href")): return "url"
    if any(x in name for x in ("avatar", "photo", "image", "thumbnail", "picture", "logo")): return "image_url"
    if any(x in name for x in ("amount", "price", "cost", "fee", "salary", "balance", "revenue", "total")): return "currency"
    if any(x in name for x in ("percent", "progress", "completion")): return "percent"
    if any(x in name for x in ("rating", "score", "stars", "grade")): return "rating"
    if any(x in name for x in ("color", "colour", "hex")): return "color"
    if any(x in name for x in ("config", "metadata", "settings", "payload", "extra")) or ft in ("json", "jsonb"): return "json"
    if any(x in name for x in ("status", "state", "stage", "phase")): return "status"
    if ft in ("bool", "boolean"): return "boolean"
    return "text"


_INPUT_CLS = 'class="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white placeholder-gray-300"'
_LABEL_CLS = 'class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1"'
_SELECT_CLS = 'class="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary bg-white"'
_CHECKBOX_CLS = 'class="w-4 h-4 text-apg-primary rounded border-gray-300"'


def _humanize_label(field_name: str) -> str:
    if field_name.endswith("_id"):
        base = field_name[:-3].replace("_", " ").strip()
        return " ".join(w.capitalize() for w in base.split()) + " ID"
    return " ".join(w.capitalize() for w in field_name.replace("_", " ").split())


def _ui_field_input_html(field: Dict[str, Any], entity_name: str = "") -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    human_label = html.escape(_humanize_label(field_name))
    expected = _json_schema_type(str(field.get("type", "any")))

    # Foreign key → styled dropdown
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target)
        return (
            f'<div class="space-y-1">'
            f'<label {_LABEL_CLS}>{human_label}</label>'
            f'<select name="{safe_name}" {_SELECT_CLS}>{opts}</select>'
            f'</div>'
        )

    if expected == "boolean":
        return (
            f'<div class="flex items-center gap-2">'
            f'<input type="hidden" name="{safe_name}" value="false">'
            f'<input type="checkbox" name="{safe_name}" value="true" {_CHECKBOX_CLS}>'
            f'<label {_LABEL_CLS} style="margin-bottom:0">{human_label}</label>'
            f'</div>'
        )
    if expected == "integer":
        type_attr = 'type="number" step="1"'
    elif expected == "number":
        type_attr = 'type="number" step="any"'
    elif field.get("type", "").lower() in {"date", "datetime", "timestamp"}:
        type_attr = 'type="date"'
    else:
        type_attr = 'type="text"'
    placeholder = f'placeholder="{human_label}"'
    return (
        f'<div class="space-y-1">'
        f'<label {_LABEL_CLS}>{human_label}</label>'
        f'<input name="{safe_name}" {type_attr} {placeholder} {_INPUT_CLS}>'
        f'</div>'
    )


def _ui_entity_location(entity_name: str) -> str:
    return f"/ui/entities/{quote(entity_name, safe='')}"


def _ui_record_display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value)
    return str(value)


def _ui_record_editor_input_html(
    field: Dict[str, Any], record: Dict[str, Any], form_id: str, entity_name: str = ""
) -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    safe_form_id = html.escape(form_id, quote=True)
    expected = _json_schema_type(str(field.get("type", "any")))
    value = record.get(field_name)

    # Foreign key → dropdown showing related entity records
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target, current_value=str(value or ""), form_id=form_id)
        return f'<select form="{safe_form_id}" name="{safe_name}">{opts}</select>'

    if expected == "boolean":
        checked = " checked" if value is True else ""
        return (
            f'<input form="{safe_form_id}" type="hidden" name="{safe_name}" value="false">'
            f'<input form="{safe_form_id}" type="checkbox" name="{safe_name}" value="true"{checked}>'
        )
    if expected == "integer":
        attributes = 'type="number" step="1"'
    elif expected == "number":
        attributes = 'type="number" step="any"'
    elif field.get("type", "").lower() in {"date", "datetime", "timestamp"}:
        attributes = 'type="date"'
    else:
        attributes = 'type="text"'
    safe_value = html.escape(_ui_record_display_value(value), quote=True)
    return f'<input form="{safe_form_id}" name="{safe_name}" value="{safe_value}" {attributes}>'


def _ui_query_value(query: Dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    return str(values[-1]) if values else ""


def _ui_records_query_form_html(entity_name: str, query: Dict[str, list[str]]) -> str:
    safe_entity_path = html.escape(quote(entity_name, safe=""), quote=True)
    fields = _field_specs(entity_name)
    filter_inputs = []
    for field in fields:
        field_name = str(field["name"])
        input_name = f"filter.{field_name}"
        safe_input_name = html.escape(input_name, quote=True)
        safe_label = html.escape(field_name)
        safe_value = html.escape(_ui_query_value(query, input_name), quote=True)
        filter_inputs.append(
            f'<label>{safe_label} <input type="text" name="{safe_input_name}" value="{safe_value}"></label>'
        )
    sort_options = ["", "id", "_revision"] + [
        str(field["name"]) for field in fields if str(field["name"]) not in {"id", "_revision"}
    ]
    selected_sort = _ui_query_value(query, "sort")
    sort_select = "".join(
        f'<option value="{html.escape(option, quote=True)}"{" selected" if option == selected_sort else ""}>'
        f'{html.escape(option or "none")}</option>'
        for option in sort_options
    )
    selected_order = (_ui_query_value(query, "order") or "asc").lower()
    order_select = "".join(
        f'<option value="{option}"{" selected" if option == selected_order else ""}>{option}</option>'
        for option in ["asc", "desc"]
    )
    limit_value = html.escape(_ui_query_value(query, "limit"), quote=True)
    offset_value = html.escape(_ui_query_value(query, "offset"), quote=True)
    filters = "".join(filter_inputs) or "<span>No fields available.</span>"
    return (
        f'<form method="get" action="/ui/entities/{safe_entity_path}">'
        f'<fieldset><legend>Query records</legend>'
        f"{filters}"
        f'<label>Sort <select name="sort">{sort_select}</select></label>'
        f'<label>Order <select name="order">{order_select}</select></label>'
        f'<label>Limit <input type="number" min="0" step="1" name="limit" value="{limit_value}"></label>'
        f'<label>Offset <input type="number" min="0" step="1" name="offset" value="{offset_value}"></label>'
        '<button type="submit">Apply</button> '
        f'<a href="/ui/entities/{safe_entity_path}">Reset</a>'
        '</fieldset></form>'
    )


def _ui_create_form_html(entity_name: str, fields: list[Dict[str, Any]]) -> str:
    """Return the HTML for the create-record form fields (used by the Jinja2 template)."""
    _SKIP = {"id", "_revision"}
    parts = []
    for field in fields:
        if str(field.get("name", "")) in _SKIP:
            continue
        parts.append(_ui_field_input_html(field, entity_name))
    return '<div class="space-y-3">' + "".join(parts) + "</div>"


def _ui_records_table_html(entity_name: str, records: list[Dict[str, Any]] | None = None, sort_field: str = "", sort_dir: str = "asc", q: str = "") -> str:
    records = records if records is not None else list_records(entity_name)
    if not records:
        return "<p>No records yet.</p>"
    fields = _field_specs(entity_name)
    field_names = [str(f["name"]) for f in fields if str(f["name"]) not in {"_revision"}]
    # Show at most 6 columns to keep table readable; id always first
    display_cols = ["id"] + [c for c in field_names if c != "id"][:5]
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    q_part = f"&q={html.escape(quote(q, safe=''), quote=True)}" if q else ""
    header_cells = []
    for col in display_cols:
        label = html.escape((col[:-3].replace("_", " ").title() + " ID") if col.endswith("_id") else col.replace("_", " ").title())
        next_dir = "desc" if sort_field == col and sort_dir == "asc" else "asc"
        sort_icon = ""
        if sort_field == col:
            sort_icon = " ▼" if sort_dir == "desc" else " ▲"
        header_cells.append(
            f'<th class="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">'
            f'<a href="/ui/entities/{safe_entity}?sort={html.escape(col)}&dir={next_dir}{q_part}"'
            f' class="hover:text-gray-900 transition-colors">{label}{sort_icon}</a>'
            f'</th>'
        )
    header = "".join(header_cells)
    rows: list[str] = []
    for record in records:
        raw_record_id = str(record.get("id", ""))
        record_id = html.escape(quote(raw_record_id, safe=""), quote=True)
        revision = html.escape(str(record.get("_revision", "")), quote=True)
        cb_cell = (
            f'<td class="pl-3 pr-1 py-2.5 w-8">'
            f'<input type="checkbox" class="apg-row-cb w-4 h-4 rounded border-gray-300 text-apg-primary"'
            f' data-row-id="{raw_record_id}" data-rev="{revision}">'
            f'</td>'
        )
        cells = [cb_cell]
        for col in display_cols:
            val = html.escape(_ui_record_display_value(record.get(col)))
            if col == "id":
                cells.append(
                    f'<td class="px-4 py-2.5">'
                    f'<a href="/ui/entities/{safe_entity}/{record_id}"'
                    f' class="text-xs font-mono text-apg-primary hover:underline truncate block max-w-24">{val[:16]}</a>'
                    f'</td>'
                )
            else:
                cells.append(f'<td class="px-4 py-2.5 text-sm text-gray-700 max-w-xs truncate">{val}</td>')
        edit_hidden = "".join(
            f'<input type="hidden" name="{html.escape(str(f["name"]), quote=True)}" value="{html.escape(str(record.get(str(f["name"]), "") or ""), quote=True)}">'
            for f in fields if str(f.get("name")) not in {"id", "_revision"}
        )
        action = (
            f'<div class="flex items-center gap-3 justify-end opacity-0 group-hover/row:opacity-100 transition-opacity">'
            f'<form method="post" action="/ui/entities/{safe_entity}/records/{record_id}" class="inline">'
            f'<input type="hidden" name="expected_revision" value="{revision}">'
            f'{edit_hidden}'
            f'<button type="submit"'
            f' class="text-xs font-medium text-apg-primary hover:underline whitespace-nowrap">Edit</button>'
            f'</form>'
            f'<form method="post" action="/ui/entities/{safe_entity}/records/{record_id}/delete" class="inline">'
            f'<input type="hidden" name="expected_revision" value="{revision}">'
            f'<button type="submit" onclick="return apgConfirmSubmit(this.form, this.dataset.msg)" data-msg="Delete this record?"'
            f' class="text-xs text-red-400 hover:text-red-600 transition-colors">Delete</button>'
            f'</form>'
            f'</div>'
        )
        rows.append(
            f'<tr class="hover:bg-gray-50 transition-colors group/row border-b border-gray-50 last:border-0">'
            f'{"".join(cells)}'
            f'<td class="px-4 py-2.5 text-right">{action}</td>'
            f'</tr>'
        )
    bulk_bar = (
        f'<div id="apg-bulk-bar" data-entity="{safe_entity}"'
        f' class="hidden fixed bottom-20 left-1/2 -translate-x-1/2 z-50'
        f' bg-gray-900 text-white rounded-2xl shadow-2xl px-5 py-3 flex items-center gap-3 text-sm">'
        f'<span id="apg-bulk-cnt" class="font-semibold tabular-nums"></span>'
        f'<button onclick="apgBulkDelete()"'
        f' class="px-3 py-1.5 bg-red-500 hover:bg-red-600 text-white text-xs font-medium rounded-lg transition-colors">Delete</button>'
        f'<a id="apg-csv-link" href="/entities/{safe_entity}/records.csv"'
        f' class="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-lg transition-colors">Export CSV</a>'
        f'<button onclick="apgBulkClear()" class="ml-1 text-gray-400 hover:text-white leading-none text-base">✕</button>'
        f'</div>'
    )
    bulk_js = (
        '<script>'
        '(function(){'
        'function upd(){'
        'var cc=document.querySelectorAll(".apg-row-cb:checked");'
        'var bar=document.getElementById("apg-bulk-bar");'
        'if(!bar)return;'
        'var cnt=document.getElementById("apg-bulk-cnt");'
        'if(cc.length>0){bar.classList.remove("hidden");cnt.textContent=cc.length+" selected";}else{bar.classList.add("hidden");}'
        '}'
        'window.apgBulkClear=function(){'
        'document.querySelectorAll(".apg-row-cb").forEach(function(c){c.checked=false;});'
        'upd();'
        '};'
        'window.apgBulkDelete=function(){'
        'var cc=document.querySelectorAll(".apg-row-cb:checked");'
        'if(!cc.length)return;'
        'apgConfirm("Delete "+cc.length+" record(s)? This cannot be undone.",function(){'
        'var ids=Array.from(cc).map(function(c){return c.dataset.rowId;}).join(",");'
        'var entity=document.getElementById("apg-bulk-bar").dataset.entity;'
        'var fd=new FormData();fd.append("ids",ids);'
        'fetch("/ui/entities/"+entity+"/records/bulk_delete",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body:"ids="+encodeURIComponent(ids)})'
        '.then(function(r){if(r.redirected||r.ok)window.location.reload();});'
        '});'
        '};'
        'document.addEventListener("change",function(e){if(e.target.classList.contains("apg-row-cb"))upd();});'
        'document.addEventListener("click",function(e){'
        'var allCb=e.target.closest(".apg-select-all");'
        'if(allCb){document.querySelectorAll(".apg-row-cb").forEach(function(c){c.checked=allCb.checked;});upd();}'
        '});'
        '})()'
        '</script>'
    )
    return (
        bulk_bar
        + f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">'
        + f'<div class="overflow-x-auto">'
        + f'<table class="w-full">'
        + f'<thead class="bg-gray-50 border-b border-gray-100">'
        + f'<tr>'
        + f'<th class="pl-3 pr-1 py-2.5 w-8"><input type="checkbox" class="apg-select-all w-4 h-4 rounded border-gray-300"></th>'
        + f'{header}<th class="px-4 py-2.5 w-28"></th></tr>'
        + f'</thead>'
        + f'<tbody>{"".join(rows)}</tbody>'
        + f'</table>'
        + f'</div>'
        + f'</div>'
        + bulk_js
    )


def _ui_entity_html(entity_name: str, notice: str = "", query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Unknown entity", f"<h1>Unknown entity: {html.escape(entity_name)}</h1>")
    query = query or {}
    safe_entity = html.escape(entity_name, quote=True)
    fields = _field_specs(entity_name) or [{"name": "value", "type": "string", "required": True}]

    # Full-text search: filter records where any string field contains q
    q = query.get("q", [""])[0].strip() if "q" in query else ""
    sort_field = query.get("sort", [""])[0].strip()
    sort_dir = query.get("dir", ["asc"])[0].strip().lower()
    if sort_dir not in ("asc", "desc"):
        sort_dir = "asc"
    # Pagination
    try:
        page = max(1, int(query.get("page", ["1"])[0]))
    except (ValueError, TypeError):
        page = 1
    try:
        per = max(5, min(200, int(query.get("per", ["50"])[0])))
    except (ValueError, TypeError):
        per = 50

    # Build query for sort/pagination and field filters
    base_query: Dict[str, list[str]] = {}
    if sort_field:
        base_query["sort"] = [sort_field]
        base_query["order"] = [sort_dir]
    for _k, _v in query.items():
        if _k.startswith("filter."):
            base_query[_k] = _v
    query_result = query_records(entity_name, base_query)
    all_records = query_result["records"]

    # Full-text search filter
    if q:
        q_low = q.lower()
        filtered = [
            r for r in all_records
            if any(q_low in str(v).lower() for v in r.values() if v is not None)
        ]
    else:
        filtered = all_records

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + per - 1) // per)
    page = min(page, total_pages)
    offset = (page - 1) * per
    paginated = filtered[offset:offset + per]

    # Detect kanban-eligible status field
    status_field_names = {"status", "state", "stage", "phase"}
    has_kanban = any(str(f.get("name", "")).lower() in status_field_names for f in fields)

    records_table = _ui_records_table_html(entity_name, paginated, sort_field=sort_field, sort_dir=sort_dir, q=q)

    # Prefer Jinja2 template for rich UI; fall back to f-string builder for zero-dep mode
    create_inputs = _ui_create_form_html(entity_name, fields)
    query_form = _ui_records_query_form_html(entity_name, query)
    tmpl_body = _render_template(
        "entity_list.html.j2",
        entity_name=html.escape(entity_name),
        entity_type=html.escape(entity.get("type", "entity")),
        safe_entity=safe_entity,
        fields=fields,
        records=paginated,
        total=query_result["total"],
        count=total_filtered,
        records_table=records_table,
        create_inputs=create_inputs,
        notice=html.escape(notice) if notice else "",
        query=query,
        has_kanban=has_kanban,
        q=html.escape(q) if q else "",
        sort_field=sort_field,
        sort_dir=sort_dir,
        page=page,
        per=per,
        total_pages=total_pages,
        records_json=html.escape(json.dumps(paginated, indent=2, sort_keys=True)),
        query_form=query_form,
    )
    if tmpl_body is not None:
        return 200, _html_page(entity_name, tmpl_body)

    # Fallback: original f-string builder
    inputs = _ui_create_form_html(entity_name, fields)
    query_form = _ui_records_query_form_html(entity_name, query)
    result_summary = f'<p>Showing {query_result["count"]} of {query_result["total"]} matching records.</p>'
    notice_html = f'<section role="alert"><strong>{html.escape(notice)}</strong></section>' if notice else ""
    body = (
        f'<nav><a href="/ui">Application</a> | '
        f'<a href="/entities/{safe_entity}/records">Record JSON</a></nav>'
        f"<h1>{html.escape(entity_name)}</h1>"
        f"<p><code>{html.escape(entity.get('type', 'entity'))}</code></p>"
        f"{notice_html}"
        f'<form method="post" action="/ui/entities/{safe_entity}/records">'
        f"{inputs}"
        '<button type="submit">Create record</button>'
        "</form>"
        "<h2>Records</h2>"
        f"{query_form}"
        f"{result_summary}"
        f"{records_table}"
        "<details><summary>Record JSON</summary>"
        f"<pre>{records_json}</pre>"
        "</details>"
    )
    return 200, _html_page(entity_name, body)


def _ui_entity_analytics_html(entity_name: str) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Unknown entity", f"<h1>Unknown entity: {html.escape(entity_name)}</h1>")
    fields = _field_specs(entity_name)
    records = list_records(entity_name)
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    line_data = [{"x": i, "y": len(records)} for i in range(30)]
    line_chart = {
        "id": f"analytics-line-{_css_name(entity_name)}",
        "spec_json": _chart_json({"type": "line", "title": f"{entity_name} records over time", "data": line_data, "empty": "No records yet"}),
    }
    status_field = _status_field_name(fields)
    counts: Dict[str, int] = {}
    if status_field:
        for record in records:
            key = str(record.get(status_field) or "Unspecified")
            counts[key] = counts.get(key, 0) + 1
    status_chart = {
        "id": f"analytics-status-{_css_name(entity_name)}",
        "spec_json": _chart_json({
            "type": "donut",
            "title": f"{entity_name} status distribution",
            "data": [{"label": key, "value": value} for key, value in sorted(counts.items())],
            "empty": "No status data yet",
        }),
    }
    numeric_stats = []
    for field in fields:
        field_name = str(field.get("name", ""))
        if _json_schema_type(str(field.get("type", ""))) not in {"integer", "number"}:
            continue
        values = []
        for record in records:
            value = record.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            numeric_stats.append({
                "field": field_name,
                "min": round(min(values), 2),
                "avg": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
            })
    tmpl_body = _render_template(
        "entity_analytics.html.j2",
        entity_name=entity_name,
        safe_entity=safe_entity,
        total=len(records),
        line_chart=line_chart,
        status_chart=status_chart,
        numeric_stats=numeric_stats,
    )
    return 200, _html_page(f"{entity_name} Analytics", tmpl_body if tmpl_body is not None else _jinja_required_page(f"{entity_name} Analytics"))


def _ui_error_message(response: Dict[str, Any]) -> str:
    errors = response.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)
    if response.get("error") == "revision_conflict":
        return (
            "Revision conflict: record has revision "
            f"{response.get('current_revision')} but form submitted revision {response.get('expected_revision')}"
        )
    if "message" in response:
        return str(response["message"])
    if "error" in response:
        return str(response["error"])
    return "The submitted form could not be applied."


def _ui_error_payload(path: str, response: Dict[str, Any]) -> str:
    parts = [part for part in path.split("/") if part]
    message = _ui_error_message(response)
    if len(parts) >= 3 and parts[0] == "ui" and parts[1] == "entities":
        _status, body = _ui_entity_html(parts[2], notice=message)
        return body
    details = html.escape(json.dumps(response, indent=2, sort_keys=True))
    return _html_page("Form error", f"<h1>Form error</h1><p>{html.escape(message)}</p><pre>{details}</pre>")


def _extract_accumulated(form: dict) -> dict:
    """Pull __acc_FIELD hidden fields from a step POST into an accumulated dict."""
    return {
        k[6:]: v  # strip '__acc_' prefix
        for k, v in form.items()
        if k.startswith("__acc_")
    }


def _ui_workflow_step_post(
    entity_name: str, workflow_id: str, step_index: int, form: dict
) -> tuple[int, str]:
    """Handle POST to a workflow step: accumulate data and advance."""
    accumulated = _extract_accumulated(form)
    step_fields_data = {k: v for k, v in form.items() if not k.startswith("__acc_") and k != "expected_revision"}
    accumulated.update(step_fields_data)

    entity_workflows = APP_WORKFLOWS.get(entity_name, [])
    wf = next((w for w in entity_workflows if w["id"] == workflow_id), None)
    if wf is None:
        return 404, _html_page("Workflow not found", "<h1>Workflow not found</h1>")

    next_step = step_index + 1
    return _ui_workflow_wizard_html(entity_name, workflow_id, next_step, accumulated)


def _ui_field_view_fragment(entity_name: str, record_id: str, field: Dict[str, Any], record: Dict[str, Any]) -> str:
    """Return the view-mode div for one field (used after save or cancel)."""
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)
    field_name = str(field.get("name", ""))
    fld_id = f"fld-{safe_entity}-{safe_record_id}-{field_name}"
    field_val = record.get(field_name, "")
    if field_val is None or field_val == "" or str(field_val) == "None":
        display = '<span class="text-gray-300 italic text-xs">—</span>'
    elif str(field_val).lower() == "true":
        display = '<span class="inline-flex items-center gap-1 text-green-600"><span class="text-xs">✓</span> Yes</span>'
    elif str(field_val).lower() == "false":
        display = '<span class="inline-flex items-center gap-1 text-gray-400"><span class="text-xs">✕</span> No</span>'
    else:
        display = html.escape(str(field_val)[:200])
    label = html.escape((field_name[:-3].replace("_", " ").title() + " ID") if field_name.endswith("_id") else field_name.replace("_", " ").title())
    edit_url = f"/ui/entities/{safe_entity}/{safe_record_id}/fields/{html.escape(field_name)}/edit"
    return (
        f'<div id="{fld_id}" class="py-3 border-b border-gray-50 last:border-0 group/field">'
        f'<dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">{label}</dt>'
        f'<dd class="flex items-center justify-between gap-2 min-h-6">'
        f'<span class="text-sm text-gray-900 break-words">{display}</span>'
        f'<button hx-get="{edit_url}" hx-target="#{fld_id}" hx-swap="outerHTML"'
        f' class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"'
        f' title="Edit {html.escape(field_name)}">'
        f'<svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">'
        f'<path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zm-2.207 2.207L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>'
        f'</svg></button>'
        f'</dd></div>'
    )


def _ui_record_detail_html(entity_name: str, record_id: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, _html_page("Not found", f"<h1>Record not found</h1><p>{html.escape(entity_name)}/{html.escape(record_id)}</p>")
    record = response.get("record", response)
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Not found", f"<h1>Unknown entity: {html.escape(entity_name)}</h1>")
    fields = _field_specs(entity_name) or []
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)

    # Pick a good display title (first non-id string field value, or id prefix)
    title_field = next(
        (f for f in fields if str(f.get("type", "")).lower() in {"str", "string", "text", "email", "varchar"} and str(f.get("name")) not in {"id", "_revision"}),
        None,
    )
    title = str(record.get(title_field["name"], record_id) if title_field else record_id)[:80]

    # Status badge value
    status_field = next(
        (f for f in fields if str(f.get("name", "")).lower() in {"status", "state", "stage", "phase"}),
        None,
    )
    status_val = str(record.get(status_field["name"], "")) if status_field else ""

    # Related lists: find entities with FK fields pointing to this entity
    related_lists: list[Dict[str, Any]] = []
    for ent in sorted(ENTITY_NAMES):
        if ent == entity_name:
            continue
        ent_fields = _field_specs(ent) or []
        fk_field = next(
            (f for f in ent_fields if str(f.get("name", "")).endswith("_id") and str(f.get("name", ""))[:-3] == entity_name.lower()),
            None,
        )
        if fk_field is None:
            # Try FK by entity name convention: field name == entity_name + "_id"
            fk_candidates = [f for f in ent_fields if str(f.get("name", "")).lower().replace("_id", "") == entity_name.lower()]
            fk_field = fk_candidates[0] if fk_candidates else None
        if fk_field:
            fk_name = str(fk_field["name"])
            rel_result = query_records(ent, {f"filter.{fk_name}": [record_id]})
            if rel_result.get("records"):
                rel_cols = ["id"] + [str(f["name"]) for f in ent_fields if str(f.get("name")) not in {"id", "_revision", fk_name}][:4]
                related_lists.append({"entity": ent, "fk_field": fk_name, "records": rel_result["records"], "cols": rel_cols})

    has_kanban = any(str(f.get("name", "")).lower() in {"status", "state", "stage", "phase"} for f in fields)
    revision = html.escape(str(record.get("_revision", "")))

    display_fields = [f for f in fields if str(f.get("name")) != "_revision"]
    field_semantics = {
        str(f.get("name", "")): _ui_field_semantic(str(f.get("name", "")), str(f.get("type", "")))
        for f in display_fields
    }
    tmpl_body = _render_template(
        "record_detail.html.j2",
        entity_name=html.escape(entity_name),
        entity_type=html.escape(entity.get("type", "entity")),
        safe_entity=safe_entity,
        safe_record_id=safe_record_id,
        record=record,
        fields=display_fields,
        field_semantics=field_semantics,
        title=html.escape(title),
        status_val=html.escape(status_val),
        revision=revision,
        related_lists=related_lists,
        has_kanban=has_kanban,
        activity_events=_get_activity(entity_name, record_id),
    )
    if tmpl_body is not None:
        return 200, _html_page(title or entity_name, tmpl_body)
    return 200, _html_page(entity_name, f"<h1>{html.escape(title)}</h1><pre>{html.escape(json.dumps(record, indent=2))}</pre>")


def _ui_field_edit_html(entity_name: str, record_id: str, field_name: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, "{}"
    record = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, "{}"
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)
    safe_field_name = html.escape(field_name)
    fld_id = f"fld-{safe_entity}-{safe_record_id}-{safe_field_name}"
    current_val = html.escape(str(record.get(field_name, "") or ""), quote=True)
    label = html.escape((field_name[:-3].replace("_", " ").title() + " ID") if field_name.endswith("_id") else field_name.replace("_", " ").title())
    patch_url = f"/ui/entities/{safe_entity}/{safe_record_id}/fields/{safe_field_name}/patch"
    cancel_url = f"/ui/entities/{safe_entity}/{safe_record_id}/fields/{safe_field_name}/view"
    field_type = str(field.get("type", "string"))
    if field_type in {"text", "markdown"}:
        input_html = (
            f'<textarea name="{safe_field_name}" rows="3"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary resize-none">'
            f'{current_val}</textarea>'
        )
    elif field_type == "boolean":
        checked = "checked" if str(record.get(field_name, "")).lower() == "true" else ""
        input_html = f'<input type="checkbox" name="{safe_field_name}" value="true" {checked} class="w-4 h-4 text-apg-primary rounded">'
    elif field_type in {"integer", "number", "float"}:
        input_html = (
            f'<input type="number" name="{safe_field_name}" value="{current_val}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    else:
        input_html = (
            f'<input type="text" name="{safe_field_name}" value="{current_val}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    revision = html.escape(str(record.get("_revision", "")), quote=True)
    fragment = (
        f'<div id="{fld_id}" class="py-3 border-b border-gray-50 last:border-0">'
        f'<dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">{label}</dt>'
        f'<dd>'
        f'<form hx-post="{patch_url}" hx-target="#{fld_id}" hx-swap="outerHTML" class="flex flex-col gap-1.5">'
        f'<input type="hidden" name="expected_revision" value="{revision}">'
        f'{input_html}'
        f'<div class="flex gap-2">'
        f'<button type="submit" class="px-2.5 py-1 bg-apg-primary text-white text-xs font-medium rounded-lg hover:opacity-90">Save</button>'
        f'<button type="button" hx-get="{cancel_url}" hx-target="#{fld_id}" hx-swap="outerHTML"'
        f' class="px-2.5 py-1 text-xs text-gray-500 hover:text-gray-700 border border-gray-200 rounded-lg">Cancel</button>'
        f'</div>'
        f'</form>'
        f'</dd></div>'
    )
    return 200, fragment


def _ui_field_view_html(entity_name: str, record_id: str, field_name: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, "{}"
    record = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, "{}"
    return 200, _ui_field_view_fragment(entity_name, record_id, field, record)


def _ui_field_patch_post(entity_name: str, record_id: str, field_name: str, form: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, {"error": "record not found"}
    current = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, {"error": "field not found"}
    new_val = form.get(field_name, "")
    field_type = str(field.get("type", "string"))
    if field_type == "boolean":
        new_val = "true" if new_val == "true" else "false"
    elif field_type == "integer":
        try:
            new_val = str(int(new_val))
        except (ValueError, TypeError):
            new_val = "0"
    updated = dict(current)
    updated[field_name] = new_val
    expected_revision_raw = form.get("expected_revision")
    try:
        expected_revision_int: int | None = int(expected_revision_raw) if expected_revision_raw is not None else None
    except (TypeError, ValueError):
        expected_revision_int = None
    save_status, save_result = update_record(entity_name, record_id, updated, expected_revision_int)
    if save_status not in (200, 201, 204):
        err_msg = html.escape(str(save_result.get("error") or save_result.get("message") or "Save failed"))
        fragment = (
            f'<div class="py-3 border-b border-gray-50">'
            f'<p class="text-xs text-red-500">{err_msg}</p>'
            f'</div>'
        )
        return save_status, {"html": fragment}
    _status2, refreshed_resp = get_record(entity_name, record_id)
    refreshed = refreshed_resp.get("record", refreshed_resp) if isinstance(refreshed_resp, dict) else {}
    rec = refreshed if refreshed else updated
    label = str(field.get("name", "")).replace("_", " ").title()
    return 200, {"html": _ui_field_view_fragment(entity_name, record_id, field, rec), "hx_trigger": {"apgToast": {"msg": f"{label} saved", "type": "success"}}}


def _ui_kanban_html(entity_name: str) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Not found", f"<h1>Unknown entity: {html.escape(entity_name)}</h1>")
    fields = _field_specs(entity_name) or []
    status_field_names = {"status", "state", "stage", "phase"}
    status_field = next((f for f in fields if str(f.get("name", "")).lower() in status_field_names), None)
    if status_field is None:
        return _ui_entity_html(entity_name)
    status_fname = str(status_field["name"])
    all_records = query_records(entity_name, {}).get("records", [])
    # Gather unique status values preserving insertion order
    seen: list[str] = []
    for r in all_records:
        v = str(r.get(status_fname, "") or "")
        if v and v not in seen:
            seen.append(v)
    if not seen:
        seen = ["active", "inactive"]
    columns = [{"label": v, "records": [r for r in all_records if str(r.get(status_fname, "")) == v]} for v in seen]
    # Choose display field: first non-id, non-status string field
    display_field_obj = next(
        (f for f in fields if str(f.get("type", "")).lower() in {"str", "string", "text", "email", "varchar"} and str(f.get("name")) not in {"id", "_revision", status_fname}),
        None,
    )
    display_field = str(display_field_obj["name"]) if display_field_obj else "id"
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    tmpl_body = _render_template(
        "kanban_view.html.j2",
        entity_name=html.escape(entity_name),
        safe_entity=safe_entity,
        columns=columns,
        display_field=display_field,
        status_field=status_fname,
        fields=fields,
    )
    if tmpl_body is not None:
        return 200, _html_page(f"{entity_name} — Kanban", tmpl_body)
    return _ui_entity_html(entity_name)


def _ui_debug_html(run_id: str | None = None) -> tuple[int, str]:
    runs = list_workflow_runs()
    cb_status = circuit_breaker_status()
    subs = dict(APG_EVENT_SUBSCRIPTIONS)
    def _badge(status: str) -> str:
        if status in {"completed", "closed", "success"}:
            return "apg-badge-success"
        if status in {"failed", "open", "circuit_open"}:
            return "apg-badge-danger"
        return "apg-badge-warning"

    selected_run = None
    if run_id:
        try:
            raw_run = get_workflow_run(run_id)
        except KeyError:
            raw_run = None
        if raw_run:
            selected_run = {
                "id": str(raw_run.get("id", run_id)),
                "workflow": str(raw_run.get("workflow", "")),
                "trace": [
                    {
                        "index": str(step.get("index", "")),
                        "step": str(step.get("step", "")),
                        "status": str(step.get("status", "")),
                        "notes": str(step.get("timeout_spec", "")),
                        "badge_class": _badge(str(step.get("status", ""))),
                    }
                    for step in raw_run.get("trace", [])
                    if isinstance(step, dict)
                ],
            }
    run_items = [
        {
            "id": str(run.get("id", "")),
            "workflow": str(run.get("workflow", "")),
            "status": str(run.get("status", "")),
            "badge_class": _badge(str(run.get("status", ""))),
            "step_count": len(run.get("trace", [])),
        }
        for run in sorted(runs, key=lambda item: str(item.get("id", "")), reverse=True)[:50]
        if isinstance(run, dict)
    ]
    breaker_items = [
        {
            "key": str(key),
            "state": str(value.get("state", "closed")) if isinstance(value, dict) else "closed",
            "failures": value.get("failures", 0) if isinstance(value, dict) else 0,
            "badge_class": _badge(str(value.get("state", "closed")) if isinstance(value, dict) else "closed"),
        }
        for key, value in sorted(cb_status.items())
    ]
    subscription_items = [
        {"event": str(event), "workflows": ", ".join(str(item) for item in workflows)}
        for event, workflows in sorted(subs.items())
    ]
    tmpl_body = _render_template(
        "debug_console.html.j2",
        selected_run=selected_run,
        runs=run_items,
        circuit_breakers=breaker_items,
        subscriptions=subscription_items,
    )
    if tmpl_body is not None:
        return 200, _html_page("Flow Debugger", tmpl_body)
    return 200, _html_page("Flow Debugger", _jinja_required_page("Flow Debugger"))

def _ui_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    parts = [part for part in path.split("/") if part]
    if parts == ["ui"]:
        return 200, _ui_index_html()
    if parts == ["ui", "databases"]:
        return _ui_database_catalog_html()
    if parts == ["ui", "workflows"]:
        return _ui_workflow_list_html()
    # /ui/workflows/ENTITY/WORKFLOW_ID  or  /ui/workflows/ENTITY/WORKFLOW_ID/step/N
    if len(parts) >= 4 and parts[0] == "ui" and parts[1] == "workflows":
        entity_name = parts[2]
        workflow_id = parts[3]
        step_index = 0
        if len(parts) == 6 and parts[4] == "step":
            try:
                step_index = int(parts[5])
            except ValueError:
                step_index = 0
        return _ui_workflow_wizard_html(entity_name, workflow_id, step_index)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "entities":
        if query and query.get("view", [""])[0] == "kanban":
            return _ui_kanban_html(parts[2])
        if query and query.get("view", [""])[0] == "analytics":
            return _ui_entity_analytics_html(parts[2])
        return _ui_entity_html(parts[2], query=query)
    # /ui/entities/ENTITY/RECORD_ID
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities":
        return _ui_record_detail_html(parts[2], parts[3])
    # /ui/entities/ENTITY/RECORD_ID/fields/FIELD_NAME/edit|view
    if (len(parts) == 7 and parts[0] == "ui" and parts[1] == "entities"
            and parts[4] == "fields" and parts[6] in {"edit", "view"}):
        if parts[6] == "edit":
            status, fragment = _ui_field_edit_html(parts[2], parts[3], parts[5])
        else:
            status, fragment = _ui_field_view_html(parts[2], parts[3], parts[5])
        return status, fragment
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "agents":
        return _ui_agent_console_html(parts[2])
    if len(parts) == 3 and parts[0] == "ui" and parts[1] in {"agent-teams", "teams"}:
        return _ui_agent_console_html(parts[2], team=True)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "capabilities":
        return _ui_capability_console_html(parts[2])
    if parts[:2] == ["ui", "debug"]:
        return _ui_debug_html(parts[2] if len(parts) > 2 else None)
    if parts == ["ui", "marketplace"]:
        try:
            from compiler.connector_generator import scan_connectors
            connectors = scan_connectors("connectors")
        except Exception:
            connectors = list(APG_CONNECTOR_REGISTRY)
        tmpl_body = _render_template("marketplace.html.j2",
            connectors=connectors,
            installed_count=len(connectors),
        )
        if tmpl_body is not None:
            return 200, _html_page("Connector Marketplace", tmpl_body)
        return 200, _html_page("Connector Marketplace", "<h1>Connector Marketplace</h1>")
    return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")


def _parse_json_object_field(form_record: Dict[str, Any], field_name: str) -> tuple[Dict[str, Any] | None, str | None]:
    raw_value = str(form_record.get(field_name) or "{}").strip() or "{}"
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        return None, f"{field_name} is invalid JSON: {error}"
    if not isinstance(value, dict):
        return None, f"{field_name} must be a JSON object"
    return value, None


def _result_section(result: Dict[str, Any] | None = None, error: str = "") -> str:
    if error:
        return f'<section role="alert"><strong>{html.escape(error)}</strong></section>'
    if result is None:
        return ""
    return "<h2>Result</h2><pre>" + html.escape(json.dumps(result, indent=2, sort_keys=True)) + "</pre>"


def _ui_agent_console_html(name: str, result: Dict[str, Any] | None = None, error: str = "", team: bool = False) -> tuple[int, str]:
    app = describe_application()
    catalog_key = "ai_agent_team_descriptions" if team else "ai_agent_descriptions"
    catalog = app.get(catalog_key, {})
    if name not in catalog:
        title = "Unknown agent team" if team else "Unknown agent"
        return 404, _html_page(title, f"<h1>{title}</h1><p>{html.escape(name)}</p>")
    action = f"/ui/{'agent-teams' if team else 'agents'}/{html.escape(name, quote=True)}/invoke"
    tmpl_body = _render_template(
        "agent_console.html.j2",
        name=name,
        team=team,
        action=action,
        description_json=json.dumps(catalog[name], indent=2, sort_keys=True),
        result=result,
        result_json=json.dumps(result, indent=2, sort_keys=True) if result is not None else "",
        error=error,
    )
    return 200, _html_page(name, tmpl_body if tmpl_body is not None else _jinja_required_page(name))


def _ui_capability_console_html(name: str, result: Dict[str, Any] | None = None, error: str = "") -> tuple[int, str]:
    app = describe_application()
    capabilities = app.get("capability_descriptions", {})
    if name not in capabilities:
        return 404, _html_page("Unknown capability", f"<h1>Unknown capability</h1><p>{html.escape(name)}</p>")
    safe_name = html.escape(name, quote=True)
    result_items = []
    if isinstance(result, dict):
        for key, value in sorted(result.items()):
            if isinstance(value, (dict, list)):
                result_items.append((str(key), json.dumps(value, sort_keys=True)))
            else:
                result_items.append((str(key), str(value)))
    tmpl_body = _render_template(
        "capability_console.html.j2",
        name=name,
        safe_name=safe_name,
        description_json=json.dumps(capabilities[name], indent=2, sort_keys=True),
        result=result,
        result_items=result_items,
        result_json=json.dumps(result, indent=2, sort_keys=True) if result is not None else "",
        result_json_html=html.escape(json.dumps(result, indent=2, sort_keys=True)) if result is not None else "",
        error=error,
    )
    return 200, _html_page(name, tmpl_body if tmpl_body is not None else _jinja_required_page(name))


def _ui_post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    raw_form_record = payload.get("record", payload)
    form_record = dict(raw_form_record) if isinstance(raw_form_record, dict) else {}

    # Field patch POST: /ui/entities/ENTITY/RECORD_ID/fields/FIELD_NAME/patch
    if (len(parts) == 7 and parts[0] == "ui" and parts[1] == "entities"
            and parts[4] == "fields" and parts[6] == "patch"):
        return _ui_field_patch_post(parts[2], parts[3], parts[5], form_record)

    # Workflow step POST: /ui/workflows/ENTITY/WORKFLOW_ID/step/N
    if (len(parts) == 6 and parts[0] == "ui" and parts[1] == "workflows" and parts[4] == "step"):
        entity_name, workflow_id = parts[2], parts[3]
        try:
            step_index = int(parts[5])
        except ValueError:
            step_index = 0
        _status, html_payload = _ui_workflow_step_post(entity_name, workflow_id, step_index, form_record)
        return _status, {"html": html_payload}

    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "agents" and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error)
            return 400, {"html": html_payload}
        message = form_record.get("message")
        if message:
            request_payload["message"] = message
        status, result = _agent_invocation_payload(f"/agents/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(parts[2], result=result if status == 200 else None, error="" if status == 200 else result.get("error", "agent invocation failed"))
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] in {"agent-teams", "teams"} and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, team=True)
            return 400, {"html": html_payload}
        message = form_record.get("message")
        if message:
            request_payload["message"] = message
        status, result = _agent_invocation_payload(f"/agent-teams/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(parts[2], result=result if status == 200 else None, error="" if status == 200 else result.get("error", "team invocation failed"), team=True)
        return status, {"html": html_payload}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "capabilities":
        capability_name = parts[2]
        operation = "/".join(parts[3:])
        if operation == "rules/evaluate":
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _rule_evaluation_payload(f"/capabilities/{capability_name}/rules/evaluate", {"context": context})
        elif operation == "configuration/resolve":
            configuration, error = _parse_json_object_field(form_record, "configuration_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _configuration_payload(f"/capabilities/{capability_name}/configuration/resolve", {"overrides": configuration})
        elif operation == "approval/plan":
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _approval_plan_payload(f"/capabilities/{capability_name}/approval/plan", {"context": context})
        else:
            return 404, {"error": "not_found", "path": path}
        _status, html_payload = _ui_capability_console_html(
            capability_name,
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "capability operation failed"),
        )
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        status, response = _create_record_payload(f"/entities/{entity_name}/records", payload)
        if status == 201:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    if (len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities"
            and parts[3] == "records" and parts[4] == "bulk_delete"):
        entity_name = parts[2]
        ids_raw = form_record.get("ids", "")
        ids = [i.strip() for i in ids_raw.split(",") if i.strip()]
        for rid in ids:
            try:
                delete_record(entity_name, rid)
            except Exception:
                pass  # best-effort
        return 303, {"location": _ui_entity_location(entity_name)}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        record_id = parts[4]
        expected_revision = form_record.pop("expected_revision", None)
        status, response = _update_record_payload(
            f"/entities/{entity_name}/records/{record_id}",
            {"record": form_record, "expected_revision": expected_revision},
        )
        if status == 200:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    if (
        len(parts) == 6
        and parts[0] == "ui"
        and parts[1] == "entities"
        and parts[3] == "records"
        and parts[5] == "delete"
    ):
        entity_name = parts[2]
        record_id = parts[4]
        delete_path = f"/entities/{entity_name}/records/{record_id}"
        expected_revision = form_record.get("expected_revision")
        if expected_revision not in (None, ""):
            delete_path = f"{delete_path}?expected_revision={quote(str(expected_revision), safe='')}"
        status, response = _delete_record_payload(delete_path)
        if status == 200:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    if (len(parts) == 6 and parts[0] == "ui" and parts[1] == "entities"
            and parts[3] == "records" and parts[5] == "note"):
        entity_name = parts[2]
        record_id = parts[4]
        note = str(form_record.get("note", "")).strip()
        if note:
            _log_activity(entity_name, record_id, "note", detail=note[:200])
        return 303, {"location": f"/ui/entities/{entity_name}/{record_id}"}
    return 404, {"error": "not_found", "path": path}


def _capability_screen(path: str) -> Dict[str, Any] | None:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "ui_route_index"):
        return None
    routes = APG_CAPABILITIES.ui_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _capability_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Capability screen")
    capability = str(screen.get("capability") or "")
    component = str(screen.get("component") or title)
    theme_name = str(screen.get("theme") or "")
    theme_tokens: Dict[str, Any] = {}
    if capability and APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_theme"):
        try:
            theme_tokens = APG_CAPABILITIES.capability_theme(capability).get("tokens", {})
        except KeyError:
            theme_tokens = {}
    actions = "".join(
        f"<li>{html.escape(str(action))}</li>"
        for action in screen.get("actions", [])
    ) or "<li>No actions declared.</li>"
    relationships = html.escape(json.dumps(screen.get("relationships", []), indent=2, sort_keys=True))
    tokens = html.escape(json.dumps(theme_tokens, indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>Capability:</strong> {html.escape(capability)}</p>"
        f"<p><strong>Component:</strong> {html.escape(component)}</p>"
        f"<p><strong>Theme:</strong> {html.escape(theme_name)}</p>"
        f"<h2>Actions</h2><ul>{actions}</ul>"
        f"<h2>Relationships</h2><pre>{relationships}</pre>"
        f"<h2>Theme Tokens</h2><pre>{tokens}</pre>"
    )
    return _html_page(title, body)


def _capability_screen_payload(path: str) -> tuple[int, str]:
    screen = _capability_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")
    return 200, _capability_screen_html(screen)


def _application_screen(path: str) -> Dict[str, Any] | None:
    if APG_APPLICATIONS is None or not hasattr(APG_APPLICATIONS, "application_route_index"):
        return None
    routes = APG_APPLICATIONS.application_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _application_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Application route")
    application = str(screen.get("application") or "")
    route = str(screen.get("route") or screen.get("path") or "")
    capabilities = html.escape(json.dumps(screen.get("capabilities", []), indent=2, sort_keys=True))
    agents = html.escape(json.dumps(screen.get("agents", []), indent=2, sort_keys=True))
    component = html.escape(json.dumps(screen.get("component"), indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/applications">Applications</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>Application:</strong> {html.escape(application)}</p>"
        f"<p><strong>Route:</strong> {html.escape(route)}</p>"
        f"<h2>Capabilities</h2><pre>{capabilities}</pre>"
        f"<h2>Agents</h2><pre>{agents}</pre>"
        f"<h2>Component</h2><pre>{component}</pre>"
    )
    return _html_page(title, body)


def _application_screen_payload(path: str) -> tuple[int, str]:
    screen = _application_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")
    return 200, _application_screen_html(screen)


def _record_route(path: str) -> Dict[str, str | None] | None:
    parts = [part for part in path.split("/") if part]
    if parts == ["records"]:
        return {"entity": None, "record_id": None, "operation": None}
    if len(parts) in {2, 3} and parts[0] == "records":
        return {
            "entity": parts[1],
            "record_id": parts[2] if len(parts) == 3 else None,
            "operation": None,
        }
    if len(parts) in {3, 4} and parts[0] == "entities" and parts[2] == "records":
        operation = parts[3] if len(parts) == 4 and parts[3] in {"export", "import"} else None
        return {
            "entity": parts[1],
            "record_id": None if operation else parts[3] if len(parts) == 4 else None,
            "operation": operation,
        }
    return None


def _record_by_id(entity_name: str, record_id: str) -> Dict[str, Any] | None:
    for record in RECORD_STORE[entity_name]:
        if str(record.get("id")) == str(record_id):
            return dict(record)
    return None


def _records_payload(path: str) -> tuple[int, Dict[str, Any]]:
    return _records_payload_with_query(path, {})


def _records_payload_with_query(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    operation = route.get("operation")
    if entity_name is None:
        return 200, {"records": list_records()}
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    if operation == "export":
        return 200, {
            "entity": entity_name,
            "records": list_records(entity_name),
            "count": len(list_records(entity_name)),
        }
    if operation is not None:
        return 405, {"error": "method_not_allowed", "operation": operation}
    if record_id is None:
        return 200, query_records(entity_name, query)
    record = _record_by_id(entity_name, record_id)
    if record is None:
        return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}
    return 200, {"entity": entity_name, "record": record}


def _route_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path in {"/", "/manifest", "/application"}:
        return 200, describe_application()
    if path == "/component.json":
        return 200, component_manifest()
    if path == "/semantic-model.json":
        return 200, semantic_model()
    if path == "/health":
        validation = validate_application()
        return 200, {
            "status": "ok" if validation["valid"] else "warning",
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "valid": validation["valid"],
            "storage": storage_status(),
            "auth": auth_status(),
            "warnings": validation["warnings"],
        }
    if path == "/validate":
        validation = validate_application()
        return (200 if validation["valid"] else 422), validation
    if path == "/openapi.json":
        return 200, openapi_document()
    if path == "/entities":
        return 200, {"entities": list_entities()}
    if path == "/workflows":
        return 200, {"workflows": describe_workflows()}
    if path == "/workflows/runs":
        return 200, {"runs": list_workflow_runs()}
    if path.startswith("/workflows/runs/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 4 and parts[3] == "journal":
            return 200, {"run_id": parts[2], "events": _get_journal(parts[2])}
        if len(parts) == 3:
            try:
                return 200, get_workflow_run(parts[2])
            except KeyError:
                return 404, {"error": "workflow_run_not_found", "id": parts[2]}
    if path.startswith("/workflows/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 2:
            try:
                return 200, describe_workflow(parts[1])
            except KeyError:
                return 404, {"error": "unknown_workflow", "workflow": parts[1]}
    if path == "/databases":
        return 200, {"databases": list_databases()}
    if path == "/databases/status":
        status = database_status()
        return (200 if status["valid"] else 422), status
    if path.startswith("/databases/") and path.endswith("/schemas"):
        database_name = path.strip("/").split("/")[1]
        for database in list_databases():
            if str(database.get("name")) == database_name:
                return 200, {
                    "database": database_name,
                    "schemas": database.get("schemas", []),
                }
        return 404, {"error": "unknown_database", "database": database_name}
    if path == "/auth":
        return 200, auth_status()
    if path == "/events":
        return 200, {"events": list_events()}
    if path == "/events/subscriptions":
        return 200, {"subscriptions": dict(APG_EVENT_SUBSCRIPTIONS)}
    if path == "/api/search":
        q = str((query or {}).get("q", [""])[0]).strip().lower() if query else ""
        results: list[Dict[str, Any]] = []
        if q:
            for ent in ENTITIES:
                ename = str(ent["name"])
                for rec in list_records(ename)[:200]:
                    for v in rec.values():
                        if q in str(v).lower():
                            label_field = next(
                                (f["name"] for f in ent.get("fields", [])
                                 if f["name"] not in ["id", "_revision"]),
                                "id",
                            )
                            results.append({
                                "entity": ename,
                                "id": str(rec.get("id", "")),
                                "label": str(rec.get(label_field, rec.get("id", "")))[:60],
                            })
                            break
        results = results[:20]
        return 200, {"results": results, "query": q, "count": len(results)}
    if path == "/circuit-breakers":
        return 200, {"circuit_breakers": circuit_breaker_status()}
    if path == "/connectors":
        return 200, {"connectors": APG_CONNECTOR_REGISTRY}
    if path == "/metrics":
        return 200, metrics_snapshot()
    if path == "/self-test":
        report = self_test()
        return (200 if report["passed"] else 422), report
    if path == "/records" or path.startswith("/records/") or (
        path.startswith("/entities/") and "/records" in path
    ):
        return _records_payload_with_query(path, query)
    if path == "/relationships":
        return 200, relationship_graph()
    if path == "/storage":
        return 200, storage_status(include_records=True)
    if path == "/agents":
        return 200, {
            "agents": describe_application().get("ai_agent_descriptions", {}),
            "teams": describe_application().get("ai_agent_team_descriptions", {}),
        }
    if path == "/applications":
        app = describe_application()
        return 200, {
            "applications": app.get("application_composition_descriptions", {}),
            "dependency_graph": app.get("application_dependency_graph", {}),
            "components": app.get("application_component_catalog", {}),
        }
    if path == "/capabilities":
        app = describe_application()
        return 200, {
            "capabilities": app.get("capability_descriptions", {}),
            "by_erp_module": app.get("capability_descriptions_by_erp_module", {}),
            "dependency_graph": app.get("capability_dependency_graph", {}),
            "load_order": app.get("capability_load_order", {}),
        }
    if path == "/capabilities/health":
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health_report"):
            return 404, {"error": "capability_health_unavailable"}
        health = APG_CAPABILITIES.capability_health_report()
        return (200 if health.get("healthy") else 422), health
    if path.startswith("/capabilities/") and path.endswith("/health"):
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health"):
            return 404, {"error": "capability_health_unavailable"}
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            try:
                health = APG_CAPABILITIES.capability_health(parts[1])
            except KeyError:
                return 404, {"error": "unknown_capability", "capability": parts[1]}
            return (200 if health.get("healthy") else 422), health
    if path == "/streaming":
        return _streaming_payload()
    if path.startswith("/capabilities/") and path.endswith("/streaming"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            return _capability_streaming_payload(parts[1])
    if path == "/routes":
        return 200, {"routes": describe_application().get("ui_routes", {})}
    if path == "/composition":
        return 200, describe_application().get("composition_graph", {"nodes": [], "edges": []})
    return 404, {"error": "not_found", "path": path}


def _rule_evaluation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if path.startswith("/capabilities/") and path.endswith("/rules/evaluate"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3:
            capability_name = parts[1]
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "evaluate_capability_rules"):
        return 404, {"error": "capability_rules_unavailable"}
    context = payload.get("context", {})
    if not isinstance(context, dict):
        return 400, {"error": "context_must_be_object"}
    try:
        return 200, APG_CAPABILITIES.evaluate_capability_rules(str(capability_name), context)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _capability_name_from_payload_or_path(path: str, payload: Dict[str, Any]) -> str | None:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if capability_name:
        return str(capability_name)
    if path.startswith("/capabilities/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            return parts[1]
    return None


def _configuration_payload(path: str, payload: Dict[str, Any], validate: bool = False) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None:
        return 404, {"error": "capabilities_unavailable"}
    configuration = payload.get("configuration", payload.get("overrides"))
    if configuration is not None and not isinstance(configuration, dict):
        return 400, {"error": "configuration_must_be_object"}
    try:
        if validate:
            validator = getattr(APG_CAPABILITIES, "validate_capability_configuration", None)
            if validator is None:
                return 404, {"error": "configuration_validation_unavailable"}
            return 200, validator(str(capability_name), configuration)
        resolver = getattr(APG_CAPABILITIES, "capability_configuration", None)
        if resolver is None:
            return 404, {"error": "configuration_resolution_unavailable"}
        return 200, {
            "capability": str(capability_name),
            "configuration": resolver(str(capability_name), configuration),
        }
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _approval_plan_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "approval_plan"):
        return 404, {"error": "approval_planning_unavailable"}
    context = payload.get("context", {})
    if not isinstance(context, dict):
        return 400, {"error": "context_must_be_object"}
    try:
        return 200, APG_CAPABILITIES.approval_plan(str(capability_name), context)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _workflow_run_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    workflow_name = payload.get("workflow") or payload.get("workflow_name")
    if len(parts) >= 2:
        workflow_name = parts[1]
    if not workflow_name:
        return 400, {"error": "missing_workflow"}
    context = payload.get("payload", payload.get("context", {}))
    if not isinstance(context, dict):
        return 400, {"error": "payload_must_be_object"}
    if "start_at" in payload and "start_at" not in context:
        context = dict(context)
        context["start_at"] = payload["start_at"]
    try:
        return 200, run_workflow(str(workflow_name), context)
    except KeyError:
        return 404, {"error": "unknown_workflow", "workflow": str(workflow_name)}


def _workflow_resume_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 4:
        return 404, {"error": "not_found", "path": path}
    context = payload.get("payload", payload.get("context", {}))
    if not isinstance(context, dict):
        return 400, {"error": "payload_must_be_object"}
    if "pause_at" in payload and "pause_at" not in context:
        context = dict(context)
        context["pause_at"] = payload["pause_at"]
    if "stop_after" in payload and "stop_after" not in context:
        context = dict(context)
        context["stop_after"] = payload["stop_after"]
    try:
        return 200, resume_workflow(parts[2], context)
    except KeyError:
        return 404, {"error": "workflow_run_not_found", "id": parts[2]}


def _workflow_compensation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 4:
        return 404, {"error": "not_found", "path": path}
    context = payload.get("payload", payload.get("context", {}))
    if not isinstance(context, dict):
        return 400, {"error": "payload_must_be_object"}
    try:
        return 200, execute_workflow_compensations(parts[2], context)
    except KeyError:
        return 404, {"error": "workflow_run_not_found", "id": parts[2]}


def _streaming_payload() -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None:
        return 404, {"error": "capabilities_unavailable"}
    processor_index = getattr(APG_CAPABILITIES, "streaming_processor_index", lambda: {})()
    state_index = getattr(APG_CAPABILITIES, "streaming_state_index", lambda: {})()
    streams: Dict[str, Any] = {}
    if hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_streaming"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            streams[capability_name] = APG_CAPABILITIES.capability_streaming(capability_name)
    return 200, {
        "processor": "bytewax",
        "processors": processor_index,
        "states": state_index,
        "streams": streams,
    }


def _capability_streaming_payload(capability_name: str) -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_streaming"):
        return 404, {"error": "capability_streaming_unavailable"}
    try:
        return 200, APG_CAPABILITIES.capability_streaming(capability_name)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": capability_name}


def _agent_invocation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if AI_AGENTS is None:
        return 404, {"error": "agents_unavailable"}
    parts = [part for part in path.split("/") if part]
    try:
        if len(parts) == 3 and parts[0] == "agents" and parts[2] in {"invoke", "run"}:
            invoker = getattr(AI_AGENTS, "invoke_agent", None)
            if invoker is None:
                return 404, {"error": "agent_invocation_unavailable"}
            return 200, invoker(parts[1], payload)
        if len(parts) == 3 and parts[0] in {"agent-teams", "teams"} and parts[2] in {"invoke", "run"}:
            invoker = getattr(AI_AGENTS, "invoke_team", None)
            if invoker is None:
                return 404, {"error": "team_invocation_unavailable"}
            return 200, invoker(parts[1], payload)
    except KeyError as error:
        return 404, {"error": "unknown_agent_composition", "name": str(error).strip("'")}
    return 404, {"error": "not_found", "path": path}


def _create_record_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is not None and route.get("operation") == "import":
        return _import_records_payload(str(route["entity"]), payload)
    if route is None or route["entity"] is None or route["record_id"] is not None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {"error": "record_must_be_object"}
    record = coerce_record_types(entity_name, dict(raw_record))
    validation = validate_record(entity_name, record)
    if not validation["valid"]:
        return 422, {"error": "record_validation_failed", **validation}
    if record.get("id") in (None, ""):
        record["id"] = NEXT_RECORD_IDS[entity_name]
        NEXT_RECORD_IDS[entity_name] += 1
    elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
        return 409, {"error": "duplicate_record_id", "entity": entity_name, "id": record["id"]}
    record = _prepare_new_record(record, entity_name)
    RECORD_STORE[entity_name].append(record)
    event = _record_event("create", entity_name, after=record)
    _log_activity(entity_name, str(record.get("id", "")), "created", detail=f"Record created with {len(record)} fields")
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {"error": "persistence_failed", "message": persistence_error}
    return 201, {
        "entity": entity_name,
        "record": dict(record),
        "event": event,
        "count": len(RECORD_STORE[entity_name]),
    }


def _import_records_payload(entity_name: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return 400, {"error": "records_must_be_array"}
    imported: list[Dict[str, Any]] = []
    events: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            errors.append({"index": index, "errors": ["record must be object"]})
            continue
        record = coerce_record_types(entity_name, dict(raw_record))
        validation = validate_record(entity_name, record)
        if not validation["valid"]:
            errors.append({"index": index, "errors": validation["errors"]})
            continue
        if record.get("id") in (None, ""):
            record["id"] = NEXT_RECORD_IDS[entity_name]
            NEXT_RECORD_IDS[entity_name] += 1
        elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
            errors.append({"index": index, "errors": [f"duplicate id {record['id']}"]})
            continue
        record = _prepare_new_record(record)
        RECORD_STORE[entity_name].append(record)
        imported.append(dict(record))
        events.append(_record_event("import", entity_name, after=record))
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {"error": "persistence_failed", "message": persistence_error}
    return (201 if imported else 422), {
        "entity": entity_name,
        "imported": imported,
        "events": events,
        "errors": errors,
        "count": len(imported),
        "failed": len(errors),
    }


def _update_record_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {"error": "record_must_be_object"}
    record_update = coerce_record_types(entity_name, dict(raw_record))
    validation = validate_record(entity_name, record_update, partial=True)
    if not validation["valid"]:
        return 422, {"error": "record_validation_failed", **validation}
    for index, existing in enumerate(RECORD_STORE[entity_name]):
        if str(existing.get("id")) == str(record_id):
            conflict = _revision_conflict(existing, _expected_revision(payload))
            if conflict is not None:
                return 409, conflict
            updated = dict(existing)
            updated.update(record_update)
            updated["id"] = existing.get("id")
            updated["_revision"] = int(existing.get("_revision", 1)) + 1
            RECORD_STORE[entity_name][index] = updated
            event = _record_event("update", entity_name, before=existing, after=updated)
            _log_activity(entity_name, str(record_id), "updated", detail="Fields updated")
            persistence_error = _persist_record_store()
            if persistence_error:
                return 500, {"error": "persistence_failed", "message": persistence_error}
            return 200, {"entity": entity_name, "record": dict(updated), "event": event}
    return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}


def _delete_record_payload(path: str) -> tuple[int, Dict[str, Any]]:
    raw_path = path
    path = path.split("?", 1)[0]
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    for index, existing in enumerate(RECORD_STORE[entity_name]):
        if str(existing.get("id")) == str(record_id):
            expected_revision = None
            if "?" in raw_path:
                query = parse_qs(raw_path.split("?", 1)[1], keep_blank_values=True)
                value = query.get("expected_revision", [None])[-1]
                try:
                    expected_revision = int(value) if value is not None else None
                except (TypeError, ValueError):
                    expected_revision = None
            conflict = _revision_conflict(existing, expected_revision)
            if conflict is not None:
                return 409, conflict
            _log_activity(entity_name, str(record_id), "deleted", detail="Record deleted")
            deleted = RECORD_STORE[entity_name].pop(index)
            event = _record_event("delete", entity_name, before=deleted)
            persistence_error = _persist_record_store()
            if persistence_error:
                return 500, {"error": "persistence_failed", "message": persistence_error}
            return 200, {
                "entity": entity_name,
                "deleted": dict(deleted),
                "event": event,
                "count": len(RECORD_STORE[entity_name]),
            }
    return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}


def _post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path == "/events/emit":
        event_name = payload.get("name") or payload.get("event") or ""
        if not event_name:
            return 422, {"error": "missing_field", "field": "name"}
        ev = emit_apg_event(str(event_name), payload.get("payload") or {})
        return 200, {"event": ev}
    if (
        path.startswith("/agents/") and path.endswith(("/invoke", "/run"))
    ) or (
        (path.startswith("/agent-teams/") or path.startswith("/teams/")) and path.endswith(("/invoke", "/run"))
    ):
        return _agent_invocation_payload(path, payload)
    if path.startswith("/records/") or path.endswith("/records/import") or (
        path.startswith("/entities/") and path.endswith("/records")
    ):
        return _create_record_payload(path, payload)
    if path in {"/rules/evaluate", "/capabilities/rules/evaluate"} or (
        path.startswith("/capabilities/") and path.endswith("/rules/evaluate")
    ):
        return _rule_evaluation_payload(path, payload)
    if path in {"/configuration/resolve", "/capabilities/configuration/resolve"} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/resolve")
    ):
        return _configuration_payload(path, payload)
    if path in {"/configuration/validate", "/capabilities/configuration/validate"} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/validate")
    ):
        return _configuration_payload(path, payload, validate=True)
    if path in {"/approval/plan", "/capabilities/approval/plan"} or (
        path.startswith("/capabilities/") and path.endswith("/approval/plan")
    ):
        return _approval_plan_payload(path, payload)
    if path.startswith("/workflows/runs/") and "/signal/" in path:
        parts = [part for part in path.split("/") if part]
        if len(parts) == 5 and parts[0] == "workflows" and parts[1] == "runs" and parts[3] == "signal":
            sig_run_id = parts[2]
            signal_name = parts[4]
            if sig_run_id not in WORKFLOW_SIGNALS:
                WORKFLOW_SIGNALS[sig_run_id] = []
            WORKFLOW_SIGNALS[sig_run_id].append(signal_name)
            _journal_append(sig_run_id, "signal_received", signal_name, {"from": "external"})
            return 200, {"status": "signal_received", "run_id": sig_run_id, "signal": signal_name}
    if path.startswith("/workflows/runs/") and path.endswith("/compensate"):
        return _workflow_compensation_payload(path, payload)
    if path.startswith("/workflows/runs/") and path.endswith("/resume"):
        return _workflow_resume_payload(path, payload)
    if path.startswith("/workflows/") and path.endswith("/run"):
        return _workflow_run_payload(path, payload)
    return 404, {"error": "not_found", "path": path}


def _put_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path.startswith("/records/") or (
        path.startswith("/entities/") and "/records/" in path
    ):
        return _update_record_payload(path, payload)
    return 404, {"error": "not_found", "path": path}


def _csv_export_body(entity_name: str) -> bytes:
    records = list_records(entity_name)
    if not records:
        return b""
    import io, csv as _csv
    fields = _field_specs(entity_name)
    cols = [str(f["name"]) for f in fields if str(f["name"]) != "_revision"] or list(records[0].keys())
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(cols)
    for rec in records:
        w.writerow([str(rec.get(c, "")) for c in cols])
    return buf.getvalue().encode("utf-8")


import os as _os_env
_APG_PG_URL: str | None = _os_env.environ.get("APG_DATABASE_URL") or _os_env.environ.get("APG_PG_URL") or _os_env.environ.get("DATABASE_URL") or None


def _pg_connection():
    if not _APG_PG_URL:
        return None
    try:
        import psycopg2  # type: ignore
        return psycopg2.connect(_APG_PG_URL)
    except Exception:
        return None


def _pg_ensure_runs_table(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_workflow_runs ("
                "  run_id TEXT PRIMARY KEY,"
                "  module_name TEXT NOT NULL,"
                "  data TEXT NOT NULL,"
                "  updated_at TIMESTAMPTZ DEFAULT NOW()"
                ")"
            )
        conn.commit()
    except Exception:
        pass  # best-effort


def _pg_save_workflow_run(run: Dict[str, Any]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        _pg_ensure_runs_table(conn)
        rid = str(run.get("id", ""))
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO apg_workflow_runs (run_id, module_name, data)"
                " VALUES (%s, %s, %s)"
                " ON CONFLICT (run_id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()",
                (rid, MODULE_NAME, json.dumps(run, default=str))
            )
        conn.commit()
    except Exception:
        pass  # best-effort
    finally:
        conn.close()


def _pg_load_workflow_runs() -> list[Dict[str, Any]]:
    conn = _pg_connection()
    if not conn:
        return []
    try:
        _pg_ensure_runs_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM apg_workflow_runs WHERE module_name = %s", (MODULE_NAME,))
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _pg_ensure_records_table(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_records ("
                "  id TEXT NOT NULL,"
                "  collection TEXT NOT NULL,"
                "  tenant_id TEXT NOT NULL DEFAULT 'default',"
                "  data JSONB NOT NULL,"
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  PRIMARY KEY (collection, id)"
                ")"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_apg_records_tenant"
                " ON apg_records (collection, tenant_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_apg_records_gin"
                " ON apg_records USING gin (data)"
            )
        conn.commit()
    except Exception:
        pass  # best-effort


def _pg_save_entity_records(entity_name: str, records: list[Dict[str, Any]]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        _pg_ensure_records_table(conn)
        with conn.cursor() as cur:
            for record in records:
                rid = str(record.get("id", ""))
                if not rid:
                    continue
                cur.execute(
                    "INSERT INTO apg_records (id, collection, tenant_id, data)"
                    " VALUES (%s, %s, %s, %s::jsonb)"
                    " ON CONFLICT (collection, id)"
                    " DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()",
                    (rid, entity_name.lower(), "default", json.dumps(record, default=str))
                )
        conn.commit()
    except Exception:
        pass  # best-effort
    finally:
        conn.close()


def _pg_load_entity_records(entity_name: str) -> list[Dict[str, Any]]:
    conn = _pg_connection()
    if not conn:
        return []
    try:
        _pg_ensure_records_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM apg_records WHERE collection = %s ORDER BY created_at",
                (entity_name.lower(),)
            )
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


_load_record_store()

_flask_app = _FlaskApp("app", root_path=os.path.abspath(os.path.dirname(globals().get("__file__", None) or ".")))


@_flask_app.before_request
def _setup_tenant() -> None:
    tid = _flask_request.headers.get("X-APG-Tenant") or _flask_request.headers.get("X-Tenant-ID")
    _TENANT_LOCAL.tenant_id = tid or None


def _check_mutation_auth():
    if _authorized(_flask_request.headers):
        return None
    status, response = _auth_failure_payload()
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/", methods=["GET"])
@_flask_app.route("/home", methods=["GET"])
def _flask_home():
    return _FlaskResponse(_landing_page_html(), content_type="text/html; charset=utf-8")


@_flask_app.route("/theme.css", methods=["GET"])
def _flask_theme():
    return _FlaskResponse(theme_stylesheet(), content_type="text/css; charset=utf-8")


@_flask_app.route("/entities/<entity_name>/records.csv", methods=["GET"])
def _flask_csv_export(entity_name):
    return _FlaskResponse(_csv_export_body(entity_name), content_type="text/csv; charset=utf-8")


@_flask_app.route("/ui", methods=["GET"])
@_flask_app.route("/ui/", methods=["GET"])
@_flask_app.route("/ui/<path:subpath>", methods=["GET"])
def _flask_ui_get(subpath=""):
    path = "/ui/" + subpath if subpath else "/ui"
    query = {k: v for k, v in _flask_request.args.lists()}
    status, html_payload = _ui_payload(path, query)
    return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")


@_flask_app.route("/ui", methods=["POST"])
@_flask_app.route("/ui/", methods=["POST"])
@_flask_app.route("/ui/<path:subpath>", methods=["POST"])
def _flask_ui_post(subpath=""):
    path = "/ui/" + subpath if subpath else "/ui"
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    if _flask_request.content_type and "application/x-www-form-urlencoded" in _flask_request.content_type:
        payload = {"record": _flask_request.form.to_dict(flat=True)}
    else:
        try:
            payload = _flask_request.get_json(force=True, silent=False) or {}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as _e:
            return _FlaskResponse(
                json.dumps({"error": "invalid_json", "message": str(_e)}),
                status=400, content_type="application/json; charset=utf-8",
            )
    status, response = _ui_post_payload(path, payload)
    if status in {302, 303}:
        return _flask_redirect(str(response["location"]), code=status)
    if "html" in response:
        _r = _FlaskResponse(str(response["html"]), status=status, content_type="text/html; charset=utf-8")
        if response.get("hx_trigger"):
            _r.headers["HX-Trigger"] = json.dumps(response["hx_trigger"])
        return _r
    return _FlaskResponse(_ui_error_payload(path, response), status=status, content_type="text/html; charset=utf-8")


_APG_GET_PUBLIC = frozenset({"/health", "/auth", "/openapi.json", "/metrics", "/describe"})


@_flask_app.route("/<path:api_path>", methods=["GET"])
def _flask_api_get(api_path):
    path = "/" + api_path
    if path not in _APG_GET_PUBLIC:
        auth_err = _check_mutation_auth()
        if auth_err:
            return auth_err
    if _capability_screen(path) is not None:
        status, html_payload = _capability_screen_payload(path)
        return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")
    if _application_screen(path) is not None:
        status, html_payload = _application_screen_payload(path)
        return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")
    query = {k: v for k, v in _flask_request.args.lists()}
    status, payload = _route_payload(path, query)
    return _FlaskResponse(json.dumps(payload), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/<path:api_path>", methods=["POST"])
def _flask_api_post(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    ct = _flask_request.content_type or ""
    if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
        payload = _flask_request.form.to_dict(flat=True)
    else:
        try:
            payload = _flask_request.get_json(force=True, silent=False) or {}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as _e:
            return _FlaskResponse(
                json.dumps({"error": "invalid_json", "message": str(_e)}),
                status=400, content_type="application/json; charset=utf-8",
            )
    status, response = _post_payload(path, payload)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/<path:api_path>", methods=["PUT"])
def _flask_api_put(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    try:
        payload = _flask_request.get_json(force=True, silent=False) or {}
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
    except Exception as _e:
        return _FlaskResponse(
            json.dumps({"error": "invalid_json", "message": str(_e)}),
            status=400, content_type="application/json; charset=utf-8",
        )
    status, response = _put_payload(path, payload)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/<path:api_path>", methods=["DELETE"])
def _flask_api_delete(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    status, response = _delete_record_payload(path)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


def _arg_value(argv: list[str], name: str, default: str) -> str:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def run_server(host: str | None = None, port: int | str | None = None) -> None:
    resolved_host = host or os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1"
    resolved_port = int(port or os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    debug = os.environ.get("APG_DEBUG") == "1"
    print(f"{MODULE_NAME} listening on {resolved_host}:{resolved_port}", flush=True)
    _flask_app.run(host=resolved_host, port=resolved_port, debug=debug, use_reloader=False)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--describe" in args:
        print(json.dumps(describe_application(), indent=2, sort_keys=True))
        return
    if "--semantic-model" in args:
        print(json.dumps(semantic_model(), indent=2, sort_keys=True))
        return
    if "--validate" in args:
        report = validate_application()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["valid"] else 1)
    if "--self-test" in args:
        report = self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)
    host = _arg_value(args, "--host", os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1")
    port = _arg_value(args, "--port", os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    run_server(host, port)


if __name__ == "__main__":
    main()
