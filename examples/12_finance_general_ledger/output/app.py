"""
finance_gl - APG Python Application
===================================

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import importlib
import html
import hmac
import json
import os
import queue as _queue
import re
import sys
import threading as _threading
import time as _time
from flask import Flask as _FlaskApp, request as _flask_request, redirect as _flask_redirect, Response as _FlaskResponse, session as _flask_session
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote


MODULE_NAME = 'finance_gl'
MODULE_VERSION = '1.0.0'
MODULE_DESCRIPTION = None
LANDING_STYLE = 'default'
ENTITIES = [{'name': 'Account', 'type': 'entity', 'properties': ['account_code', 'account_name', 'account_type', 'parent_code', 'currency', 'is_active', 'is_control', 'normal_balance'], 'fields': [{'name': 'account_code', 'type': 'str', 'required': True}, {'name': 'account_name', 'type': 'str', 'required': True}, {'name': 'account_type', 'type': 'str', 'required': True}, {'name': 'parent_code', 'type': 'str?', 'required': True}, {'name': 'currency', 'type': 'str', 'required': False, 'default': '"KES"'}, {'name': 'is_active', 'type': 'bool', 'required': False, 'default': 'true'}, {'name': 'is_control', 'type': 'bool', 'required': False, 'default': 'false'}, {'name': 'normal_balance', 'type': 'str', 'required': True}], 'methods': []}, {'name': 'JournalEntry', 'type': 'entity', 'properties': ['journal_id', 'reference', 'period', 'entry_date', 'description', 'status', 'total_debit', 'total_credit', 'posted_by', 'posted_at'], 'fields': [{'name': 'journal_id', 'type': 'str', 'required': True}, {'name': 'reference', 'type': 'str', 'required': True}, {'name': 'period', 'type': 'str', 'required': True}, {'name': 'entry_date', 'type': 'date', 'required': True}, {'name': 'description', 'type': 'str', 'required': True}, {'name': 'status', 'type': 'str', 'required': False, 'default': '"draft"'}, {'name': 'total_debit', 'type': 'decimal', 'required': False, 'default': '0.0'}, {'name': 'total_credit', 'type': 'decimal', 'required': False, 'default': '0.0'}, {'name': 'posted_by', 'type': 'str?', 'required': True}, {'name': 'posted_at', 'type': 'datetime?', 'required': True}], 'methods': []}, {'name': 'JournalLine', 'type': 'entity', 'properties': ['line_id', 'journal_id', 'account_code', 'debit', 'credit', 'cost_centre', 'project', 'memo'], 'fields': [{'name': 'line_id', 'type': 'str', 'required': True}, {'name': 'journal_id', 'type': 'str', 'required': True}, {'name': 'account_code', 'type': 'str', 'required': True}, {'name': 'debit', 'type': 'decimal', 'required': False, 'default': '0.0'}, {'name': 'credit', 'type': 'decimal', 'required': False, 'default': '0.0'}, {'name': 'cost_centre', 'type': 'str?', 'required': True}, {'name': 'project', 'type': 'str?', 'required': True}, {'name': 'memo', 'type': 'str?', 'required': True}], 'methods': []}, {'name': 'GeneralLedger', 'type': 'capability', 'properties': [], 'fields': [], 'methods': []}, {'name': 'FinanceGL', 'type': 'app', 'properties': [], 'fields': [], 'methods': []}]
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
APG_LIVE_LOCK = _threading.Lock()
APG_LIVE_SUBSCRIBERS: list[Dict[str, Any]] = []
TENANT_SCOPED_ENTITIES: set[str] = {
    e["name"] for e in ENTITIES
    if any(str(f.get("name")) == "tenant_id" for f in e.get("fields", []))
}
SEMANTIC_MODEL: Dict[str, Any] = {'format': 'apg.semantic-model.v1', 'ok': True, 'source_files': ['finance_gl.apg'], 'app': {'name': 'finance_gl', 'version': '1.0.0', 'description': None, 'entity_count': 5}, 'symbols': {'module.finance_gl': {'id': 'module.finance_gl', 'kind': 'module', 'name': 'finance_gl', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'table.Account': {'id': 'table.Account', 'kind': 'table', 'name': 'Account', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.account_code': {'id': 'field.Account.account_code', 'kind': 'field', 'name': 'Account.account_code', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.account_name': {'id': 'field.Account.account_name', 'kind': 'field', 'name': 'Account.account_name', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.account_type': {'id': 'field.Account.account_type', 'kind': 'field', 'name': 'Account.account_type', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.parent_code': {'id': 'field.Account.parent_code', 'kind': 'field', 'name': 'Account.parent_code', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.currency': {'id': 'field.Account.currency', 'kind': 'field', 'name': 'Account.currency', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.is_active': {'id': 'field.Account.is_active', 'kind': 'field', 'name': 'Account.is_active', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.is_control': {'id': 'field.Account.is_control', 'kind': 'field', 'name': 'Account.is_control', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.Account.normal_balance': {'id': 'field.Account.normal_balance', 'kind': 'field', 'name': 'Account.normal_balance', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'table.JournalEntry': {'id': 'table.JournalEntry', 'kind': 'table', 'name': 'JournalEntry', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.journal_id': {'id': 'field.JournalEntry.journal_id', 'kind': 'field', 'name': 'JournalEntry.journal_id', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.reference': {'id': 'field.JournalEntry.reference', 'kind': 'field', 'name': 'JournalEntry.reference', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.period': {'id': 'field.JournalEntry.period', 'kind': 'field', 'name': 'JournalEntry.period', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.entry_date': {'id': 'field.JournalEntry.entry_date', 'kind': 'field', 'name': 'JournalEntry.entry_date', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.description': {'id': 'field.JournalEntry.description', 'kind': 'field', 'name': 'JournalEntry.description', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.status': {'id': 'field.JournalEntry.status', 'kind': 'field', 'name': 'JournalEntry.status', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.total_debit': {'id': 'field.JournalEntry.total_debit', 'kind': 'field', 'name': 'JournalEntry.total_debit', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.total_credit': {'id': 'field.JournalEntry.total_credit', 'kind': 'field', 'name': 'JournalEntry.total_credit', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.posted_by': {'id': 'field.JournalEntry.posted_by', 'kind': 'field', 'name': 'JournalEntry.posted_by', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalEntry.posted_at': {'id': 'field.JournalEntry.posted_at', 'kind': 'field', 'name': 'JournalEntry.posted_at', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'table.JournalLine': {'id': 'table.JournalLine', 'kind': 'table', 'name': 'JournalLine', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.line_id': {'id': 'field.JournalLine.line_id', 'kind': 'field', 'name': 'JournalLine.line_id', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.journal_id': {'id': 'field.JournalLine.journal_id', 'kind': 'field', 'name': 'JournalLine.journal_id', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.account_code': {'id': 'field.JournalLine.account_code', 'kind': 'field', 'name': 'JournalLine.account_code', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.debit': {'id': 'field.JournalLine.debit', 'kind': 'field', 'name': 'JournalLine.debit', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.credit': {'id': 'field.JournalLine.credit', 'kind': 'field', 'name': 'JournalLine.credit', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.cost_centre': {'id': 'field.JournalLine.cost_centre', 'kind': 'field', 'name': 'JournalLine.cost_centre', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.project': {'id': 'field.JournalLine.project', 'kind': 'field', 'name': 'JournalLine.project', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'field.JournalLine.memo': {'id': 'field.JournalLine.memo', 'kind': 'field', 'name': 'JournalLine.memo', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'capability.GeneralLedger': {'id': 'capability.GeneralLedger', 'kind': 'capability', 'name': 'GeneralLedger', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'app.FinanceGL': {'id': 'app.FinanceGL', 'kind': 'app', 'name': 'FinanceGL', 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}}, 'tables': {'Account': {'name': 'Account', 'fields': {'account_code': {'type': 'str', 'required': True, 'relationship': None}, 'account_name': {'type': 'str', 'required': True, 'relationship': None}, 'account_type': {'type': 'str', 'required': True, 'relationship': None}, 'parent_code': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'currency': {'type': 'str', 'required': False, 'relationship': None}, 'is_active': {'type': 'bool', 'required': False, 'relationship': None}, 'is_control': {'type': 'bool', 'required': False, 'relationship': None}, 'normal_balance': {'type': 'str', 'required': True, 'relationship': None}}, 'lookup_paths': {'parent_code.id': {'chain': ['Account.parent_code', 'str?.id'], 'valid': True}}}, 'JournalEntry': {'name': 'JournalEntry', 'fields': {'journal_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Journal', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'journal'}}, 'reference': {'type': 'str', 'required': True, 'relationship': None}, 'period': {'type': 'str', 'required': True, 'relationship': None}, 'entry_date': {'type': 'date', 'required': True, 'relationship': {'target_table': 'date', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'description': {'type': 'str', 'required': True, 'relationship': None}, 'status': {'type': 'str', 'required': False, 'relationship': None}, 'total_debit': {'type': 'decimal', 'required': False, 'relationship': {'target_table': 'decimal', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'total_credit': {'type': 'decimal', 'required': False, 'relationship': {'target_table': 'decimal', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'posted_by': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'posted_at': {'type': 'datetime?', 'required': True, 'relationship': {'target_table': 'datetime?', 'target_field': 'id', 'cardinality': 'many-to-one'}}}, 'lookup_paths': {'entry_date.id': {'chain': ['JournalEntry.entry_date', 'date.id'], 'valid': True}, 'total_debit.id': {'chain': ['JournalEntry.total_debit', 'decimal.id'], 'valid': True}, 'total_credit.id': {'chain': ['JournalEntry.total_credit', 'decimal.id'], 'valid': True}, 'posted_by.id': {'chain': ['JournalEntry.posted_by', 'str?.id'], 'valid': True}, 'posted_at.id': {'chain': ['JournalEntry.posted_at', 'datetime?.id'], 'valid': True}}}, 'JournalLine': {'name': 'JournalLine', 'fields': {'line_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Line', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'line'}}, 'journal_id': {'type': 'str', 'required': True, 'relationship': {'target_table': 'Journal', 'target_field': 'id', 'cardinality': 'many-to-one', 'alias': 'journal'}}, 'account_code': {'type': 'str', 'required': True, 'relationship': None}, 'debit': {'type': 'decimal', 'required': False, 'relationship': {'target_table': 'decimal', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'credit': {'type': 'decimal', 'required': False, 'relationship': {'target_table': 'decimal', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'cost_centre': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'project': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}, 'memo': {'type': 'str?', 'required': True, 'relationship': {'target_table': 'str?', 'target_field': 'id', 'cardinality': 'many-to-one'}}}, 'lookup_paths': {'debit.id': {'chain': ['JournalLine.debit', 'decimal.id'], 'valid': True}, 'credit.id': {'chain': ['JournalLine.credit', 'decimal.id'], 'valid': True}, 'cost_centre.id': {'chain': ['JournalLine.cost_centre', 'str?.id'], 'valid': True}, 'project.id': {'chain': ['JournalLine.project', 'str?.id'], 'valid': True}, 'memo.id': {'chain': ['JournalLine.memo', 'str?.id'], 'valid': True}}}}, 'views': {}, 'flows': {}, 'operations': {}, 'rules': {'GeneralLedger.journal_balanced': {'name': 'journal_balanced', 'when': 'total_debit != total_credit', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '!=', 'value': 'total_credit'}}, 'GeneralLedger.open_period_required': {'name': 'open_period_required', 'when': 'period_status != open', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'period_status', 'op': '!=', 'value': 'open'}}, 'GeneralLedger.posting_authorised': {'name': 'posting_authorised', 'when': 'user_role in [accountant, controller, cfo]', 'action': 'allow', 'when_ast': {'type': 'in', 'field': 'user_role', 'values': ['accountant', 'controller', 'cfo'], 'negated': False}}, 'GeneralLedger.large_entry_review': {'name': 'large_entry_review', 'when': 'total_debit > 1000000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '>', 'value': 1000000}}, 'GeneralLedger.period_close_check': {'name': 'period_close_check', 'when': 'reconciled == false and period_close_initiated == true', 'action': 'deny', 'when_ast': {'type': 'and', 'left': {'type': 'compare', 'field': 'reconciled', 'op': '==', 'value': False}, 'right': {'type': 'compare', 'field': 'period_close_initiated', 'op': '==', 'value': True}}}, 'GeneralLedger.balanced_entry': {'name': 'balanced_entry', 'when': 'debits != credits', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'debits', 'op': '!=', 'value': 'credits'}}, 'GeneralLedger.valid_period': {'name': 'valid_period', 'when': 'period_status == closed', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'period_status', 'op': '==', 'value': 'closed'}}, 'GeneralLedger.active_account': {'name': 'active_account', 'when': 'account_is_active == false', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'account_is_active', 'op': '==', 'value': False}}, 'GeneralLedger.budget_exceeded': {'name': 'budget_exceeded', 'when': 'commitment > budget_line', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'commitment', 'op': '>', 'value': 'budget_line'}}}, 'roles': {}, 'security': {}, 'agents': {}, 'llms': {}, 'capabilities': {'GeneralLedger': {'name': 'GeneralLedger', 'provides': ['chart_of_accounts', 'journal_entries', 'period_close', 'trial_balance', 'financial_statements'], 'requires': ['audit_events', 'auth'], 'configuration': {'tenant_id': 'default', 'fiscal_year_start': '01-01', 'base_currency': 'KES', 'supported_currencies': ['KES', 'UGX', 'TZS', 'USD', 'EUR'], 'decimal_places': 2}, 'rules': [{'name': 'journal_balanced', 'when': 'total_debit != total_credit', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '!=', 'value': 'total_credit'}}, {'name': 'open_period_required', 'when': 'period_status != open', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'period_status', 'op': '!=', 'value': 'open'}}, {'name': 'posting_authorised', 'when': 'user_role in [accountant, controller, cfo]', 'action': 'allow', 'when_ast': {'type': 'in', 'field': 'user_role', 'values': ['accountant', 'controller', 'cfo'], 'negated': False}}, {'name': 'large_entry_review', 'when': 'total_debit > 1000000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '>', 'value': 1000000}}, {'name': 'period_close_check', 'when': 'reconciled == false and period_close_initiated == true', 'action': 'deny', 'when_ast': {'type': 'and', 'left': {'type': 'compare', 'field': 'reconciled', 'op': '==', 'value': False}, 'right': {'type': 'compare', 'field': 'period_close_initiated', 'op': '==', 'value': True}}}], 'rule_engine': {}, 'ui': {'shell': 'python', 'routes': [{'name': 'Chart of Accounts', 'path': '/gl/accounts', 'component': 'AccountList', 'permission': 'gl:accounts'}, {'name': 'Journal Entries', 'path': '/gl/journals', 'component': 'JournalList', 'permission': 'gl:journals'}, {'name': 'Trial Balance', 'path': '/gl/trial', 'component': 'TrialBalance', 'permission': 'gl:reports'}, {'name': 'Period Close', 'path': '/gl/period-close', 'component': 'PeriodClose', 'permission': 'gl:close'}]}, 'theme': {'name': 'finance_theme', 'tokens': {'accent': '#1A237E', 'color.primary': '#283593'}}, 'runtime': {}, 'erp_modules': ['finance', 'general_ledger', 'accounts_payable', 'accounts_receivable', 'fixed_assets', 'project_accounting', 'reporting'], 'components': {}, 'business_rules': [{'name': 'balanced_entry', 'when': 'debits != credits', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'debits', 'op': '!=', 'value': 'credits'}}, {'name': 'valid_period', 'when': 'period_status == closed', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'period_status', 'op': '==', 'value': 'closed'}}, {'name': 'active_account', 'when': 'account_is_active == false', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'account_is_active', 'op': '==', 'value': False}}, {'name': 'budget_exceeded', 'when': 'commitment > budget_line', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'commitment', 'op': '>', 'value': 'budget_line'}}], 'approvals': {'levels': 3, 'thresholds': {'level1': 100000, 'level2': 500000, 'level3': 1000000}, 'approvers': ['finance_manager', 'controller', 'cfo'], 'segregation_of_duties': True, 'escalation': 'finance_director'}, 'master_data': {'entities': ['account', 'cost_centre', 'financial_period', 'currency', 'exchange_rate', 'budget', 'budget_line', 'project', 'department'], 'ownership': {'account': 'finance', 'cost_centre': 'operations', 'financial_period': 'finance', 'currency': 'finance'}, 'deduplication': 'account_code', 'governance': {'type': 'deterministic', 'rules': [{'name': 'unique_code', 'when': 'account_code exists', 'action': 'deny'}]}}, 'i18n': {'supported_languages': ['en', 'sw', 'fr'], 'default_language': 'en', 'fallback_language': 'en'}, 'streaming': {'processor': 'bytewax', 'input': 'gl_events', 'state': 'gl_state'}, 'screens': {}}}, 'composition': {'applications': {'FinanceGL': {'name': 'FinanceGL', 'description': 'General ledger financial application', 'capabilities': ['GeneralLedger'], 'agents': [], 'agent_teams': [], 'components': {}, 'screens': {}, 'routes': ['/gl'], 'workflows': [], 'policies': {}, 'configuration': {}, 'theme': {}, 'runtime': {}, 'integrations': {}, 'deployments': {}}}, 'agent_teams': {}, 'capability_dependencies': {'GeneralLedger': ['audit_events', 'auth']}}, 'contracts': {'GeneralLedger': {'id': 'general_ledger', 'provides': ['chart_of_accounts', 'journal_entries', 'period_close', 'trial_balance', 'financial_statements'], 'requires': ['audit_events', 'auth'], 'configuration': {'tenant_id': 'default', 'fiscal_year_start': '01-01', 'base_currency': 'KES', 'supported_currencies': ['KES', 'UGX', 'TZS', 'USD', 'EUR'], 'decimal_places': 2}, 'configuration_schema': {'required': ['tenant_id', 'base_currency']}, 'rules': [{'name': 'journal_balanced', 'when': 'total_debit != total_credit', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '!=', 'value': 'total_credit'}}, {'name': 'open_period_required', 'when': 'period_status != open', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'period_status', 'op': '!=', 'value': 'open'}}, {'name': 'posting_authorised', 'when': 'user_role in [accountant, controller, cfo]', 'action': 'allow', 'when_ast': {'type': 'in', 'field': 'user_role', 'values': ['accountant', 'controller', 'cfo'], 'negated': False}}, {'name': 'large_entry_review', 'when': 'total_debit > 1000000', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'total_debit', 'op': '>', 'value': 1000000}}, {'name': 'period_close_check', 'when': 'reconciled == false and period_close_initiated == true', 'action': 'deny', 'when_ast': {'type': 'and', 'left': {'type': 'compare', 'field': 'reconciled', 'op': '==', 'value': False}, 'right': {'type': 'compare', 'field': 'period_close_initiated', 'op': '==', 'value': True}}}], 'ui': {'shell': 'python', 'routes': [{'name': 'Chart of Accounts', 'path': '/gl/accounts', 'component': 'AccountList', 'permission': 'gl:accounts'}, {'name': 'Journal Entries', 'path': '/gl/journals', 'component': 'JournalList', 'permission': 'gl:journals'}, {'name': 'Trial Balance', 'path': '/gl/trial', 'component': 'TrialBalance', 'permission': 'gl:reports'}, {'name': 'Period Close', 'path': '/gl/period-close', 'component': 'PeriodClose', 'permission': 'gl:close'}]}, 'theme': {'name': 'finance_theme', 'tokens': {'accent': '#1A237E', 'color.primary': '#283593'}}}}, 'deployment': {'target': 'python', 'source': 'finance_gl.apg'}, 'packages': {}, 'graphs': {'er': {'kind': 'er', 'nodes': 29, 'edges': 26}, 'lookup': {'kind': 'lookup', 'nodes': 6, 'edges': 5}, 'workflow': {'kind': 'workflow', 'nodes': 6, 'edges': 5}, 'handler': {'kind': 'handler', 'nodes': 6, 'edges': 5}, 'capability': {'kind': 'capability', 'nodes': 4, 'edges': 3}, 'security': {'kind': 'security', 'nodes': 6, 'edges': 5}, 'agent': {'kind': 'agent', 'nodes': 0, 'edges': 0}, 'deployment': {'kind': 'deployment', 'nodes': 6, 'edges': 5}, 'package': {'kind': 'package', 'nodes': 6, 'edges': 5}}, 'diagnostics': [{'code': 'APG0100', 'title': 'Semantic warning', 'severity': 'warning', 'message': "Capability 'GeneralLedger' requires 'audit_events' which is not declared in this module", 'file': 'finance_gl.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'related_locations': [], 'fixes': [], 'docs_url': 'docs/tooling.md#semantic-model-contract'}]}
APG_UI_TEMPLATES: Dict[str, str] = {'entity_list.html.j2': '{# entity_list.html.j2 — APG entity list + create form\n   Variables: entity_name, entity_type, safe_entity, fields, records,\n              total, count, records_table, create_inputs, notice, query,\n              saved_views, active_filters, has_kanban (bool), q (search term) #}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">{{ entity_name }}</span>\n  <div class="ml-auto flex items-center gap-1">\n    {% if has_kanban %}\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">≡ List</span>\n    <a href="/ui/entities/{{ safe_entity }}?view=kanban"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ⊞ Kanban\n    </a>\n    {% endif %}\n    <a href="/ui/entities/{{ safe_entity }}?view=analytics"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      Analytics\n    </a>\n    <button type="button" class="apg-btn" onclick="document.getElementById(\'apg-create-drawer\').showModal();">\n      New {{ entity_name }}\n    </button>\n  </div>\n</nav>\n\n{% if notice %}\n<div role="alert"\n     class="mb-4 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">\n  ⚠ {{ notice }}\n</div>\n{% endif %}\n\n<section class="apg-card apg-list-toolbar" aria-label="{{ entity_name }} table controls">\n  <div class="apg-view-strip" aria-label="Saved views">\n    {% for view in saved_views %}\n    <a href="{{ view.url }}"\n       class="apg-view-tab{% if view.active %} active{% endif %}"\n       {% if view.active %}aria-current="page"{% endif %}>\n      <span>{{ view.name }}</span>\n      <small>{{ view.description }}</small>\n    </a>\n    {% endfor %}\n  </div>\n\n  <form method="get" action="/ui/entities/{{ safe_entity }}" class="apg-search-form">\n    <label class="sr-only" for="apg-entity-search">Search {{ entity_name }} records</label>\n    <div class="relative">\n      <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-xs pointer-events-none">Search</span>\n      <input id="apg-entity-search" type="text" name="q" value="{{ q or \'\' }}"\n             placeholder="Search {{ entity_name }} records..."\n             class="w-full pl-14 pr-8 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n      {% if q %}\n      <a href="{{ clear_filters_url }}"\n         aria-label="Clear search"\n         class="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-700 text-xs leading-none">x</a>\n      {% endif %}\n    </div>\n  </form>\n\n  {% if active_filters %}\n  <div class="apg-filter-chip-row" aria-label="Active filters">\n    {% for chip in active_filters %}\n    <span class="apg-filter-chip">\n      <span>{{ chip.label }}: {{ chip.value }}</span>\n      <a href="{{ chip.clear_url }}" aria-label="Clear {{ chip.label }}">x</a>\n    </span>\n    {% endfor %}\n    <a href="{{ clear_filters_url }}" class="apg-filter-clear">Clear all</a>\n  </div>\n  {% endif %}\n\n  <details class="apg-filter-panel"{% if active_filters %} open{% endif %}>\n    <summary>Advanced filter</summary>\n    <div class="mt-2">{{ query_form | safe }}</div>\n  </details>\n</section>\n\n<section class="apg-list-intelligence" aria-label="{{ entity_name }} list intelligence">\n  <article>\n    <span>Shareable State</span>\n    <strong>{{ list_intelligence.share_url }}</strong>\n    <button type="button" class="apg-mini-btn" data-apg-copy-url="{{ list_intelligence.share_url }}">Copy</button>\n  </article>\n  <article>\n    <span>Column Memory</span>\n    <strong>{{ list_intelligence.column_controls | length }} fields</strong>\n    <div class="apg-column-chip-row">\n      {% for col in list_intelligence.column_controls %}\n      <a href="{{ col.sort_url }}" class="apg-column-chip{% if col.active %} active{% endif %}">{{ col.label }}</a>\n      {% endfor %}\n    </div>\n  </article>\n  <article>\n    <span>Virtual Window</span>\n    <strong>{{ list_intelligence.visible_window }} of {{ list_intelligence.total }}</strong>\n    <small>{{ list_intelligence.page_size }} rows per page{% if list_intelligence.filtered %} after filters{% endif %}</small>\n  </article>\n  <article>\n    <span>Keyboard Fuzzy Filter</span>\n    <strong>/ focuses search</strong>\n    <small>Bulk bar and CSV export stay offline</small>\n  </article>\n</section>\n\n<script>\n(function() {\n  var densityKey = {{ list_intelligence.density_key | tojson }};\n  var root = document.documentElement;\n  var savedDensity = localStorage.getItem(densityKey);\n  if (savedDensity === \'compact\') root.classList.add(\'apg-density-compact\');\n  document.addEventListener(\'click\', function(event) {\n    var copy = event.target.closest(\'[data-apg-copy-url]\');\n    if (copy && navigator.clipboard) {\n      navigator.clipboard.writeText(location.origin + copy.dataset.apgCopyUrl);\n      APGToast(\'List URL copied\', \'success\');\n    }\n  });\n  document.addEventListener(\'keydown\', function(event) {\n    if (event.key === \'/\' && !/input|textarea|select/i.test(event.target.tagName || \'\')) {\n      var search = document.getElementById(\'apg-entity-search\');\n      if (search) {\n        event.preventDefault();\n        search.focus();\n      }\n    }\n    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === \'d\') {\n      event.preventDefault();\n      var compact = !root.classList.contains(\'apg-density-compact\');\n      root.classList.toggle(\'apg-density-compact\', compact);\n      localStorage.setItem(densityKey, compact ? \'compact\' : \'comfortable\');\n      APGToast(compact ? \'Compact rows enabled\' : \'Comfortable rows enabled\', \'info\');\n    }\n  });\n})();\n</script>\n\n<dialog id="apg-create-drawer" class="apg-drawer" aria-labelledby="apg-create-title">\n  <form id="apg-create-form" method="post" action="/ui/entities/{{ safe_entity }}/records" class="apg-drawer-panel">\n    <header class="apg-card-header">\n      <div>\n        <h2 id="apg-create-title" class="text-base font-semibold text-gray-900">New {{ entity_name }}</h2>\n        <p class="text-xs text-gray-400 mt-1">{{ entity_type }}</p>\n      </div>\n      <button type="button" class="apg-btn apg-btn-secondary" onclick="document.getElementById(\'apg-create-drawer\').close();">Close</button>\n    </header>\n    <div class="overflow-y-auto max-h-[70vh] space-y-3">\n      {{ create_inputs | safe }}\n    </div>\n    <footer class="flex items-center justify-end gap-2 pt-4 border-t border-gray-100 mt-4">\n      <button type="button" class="apg-btn apg-btn-secondary" onclick="document.getElementById(\'apg-create-drawer\').close();">Cancel</button>\n      <button type="submit" class="apg-btn">Create {{ entity_name }}</button>\n    </footer>\n  </form>\n</dialog>\n\n<script>\n(function() {\n  var drawer = document.getElementById(\'apg-create-drawer\');\n  var form = document.getElementById(\'apg-create-form\');\n  if (!drawer || !form) return;\n  var dirty = false;\n  form.addEventListener(\'input\', function() { dirty = true; });\n  form.addEventListener(\'submit\', function() { dirty = false; });\n  drawer.addEventListener(\'cancel\', function(event) {\n    if (!dirty) return;\n    event.preventDefault();\n    apgConfirm(\'Discard this draft?\', function() {\n      dirty = false;\n      drawer.close();\n    });\n  });\n  document.addEventListener(\'keydown\', function(event) {\n    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === \'s\' && drawer.open) {\n      event.preventDefault();\n      form.requestSubmit();\n    }\n  });\n})();\n</script>\n\n<div class="flex items-start gap-5 flex-col lg:flex-row">\n\n  {# ── Records section ─────────────────────────────────────────── #}\n  <section class="flex-1 min-w-0" data-apg-live="entity:{{ entity_name }}">\n    <div class="flex items-center gap-3 mb-3 flex-wrap">\n      <h1 class="text-lg font-semibold text-gray-900">{{ entity_name }}</h1>\n      <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n        {{ total }} record{{ \'s\' if total != 1 else \'\' }}\n      </span>\n      {% if count != total %}\n      <span class="text-xs text-apg-primary bg-blue-50 px-2 py-0.5 rounded-full">\n        {{ count }} match{% if q %} for "{{ q }}"{% endif %}\n      </span>\n      {% endif %}\n    </div>\n\n    <p class="text-xs text-gray-500 mb-2">Showing {{ count }} of {{ total }} matching records.</p>\n\n    {% if records %}\n      {{ records_table | safe }}\n    {% else %}\n      <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n        <div class="text-3xl mb-3 opacity-30">📋</div>\n        {% if q %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records match "{{ q }}".</p>\n        <p class="text-xs text-gray-400 mt-1">\n          <a href="/ui/entities/{{ safe_entity }}" class="text-apg-primary hover:underline">Clear search</a>\n        </p>\n        {% else %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records yet.</p>\n        <p class="text-xs text-gray-400 mt-1">Create the first record to get started.</p>\n        <button type="button" class="apg-btn mt-4" onclick="document.getElementById(\'apg-create-drawer\').showModal();">New {{ entity_name }}</button>\n        {% endif %}\n      </div>\n    {% endif %}\n\n    {# Pagination controls #}\n    {% if total_pages > 1 %}\n    <nav class="mt-4 flex items-center justify-between flex-wrap gap-3" aria-label="Pagination">\n      <div class="flex items-center gap-1 flex-wrap">\n        {% if page > 1 %}\n        <a href="{{ prev_page_url }}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">← Prev</a>\n        {% else %}\n        <span class="px-3 py-1.5 text-sm border border-gray-100 rounded-lg text-gray-300 select-none">← Prev</span>\n        {% endif %}\n\n        {% if page > 3 %}\n        <a href="{{ first_page_url }}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">1</a>\n        {% if page > 4 %}<span class="px-1 text-xs text-gray-400">…</span>{% endif %}\n        {% endif %}\n\n        {% for page_link in pagination_pages %}\n        <a href="{{ page_link.url }}"\n           class="px-3 py-1.5 text-sm rounded-lg {% if page_link.number == page %}bg-apg-primary text-white font-semibold{% else %}border border-gray-200 text-gray-600 hover:border-apg-primary hover:text-apg-primary{% endif %} transition-colors">{{ page_link.number }}</a>\n        {% endfor %}\n\n        {% if page < total_pages - 2 %}\n        {% if page < total_pages - 3 %}<span class="px-1 text-xs text-gray-400">…</span>{% endif %}\n        <a href="{{ last_page_url }}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">{{ total_pages }}</a>\n        {% endif %}\n\n        {% if page < total_pages %}\n        <a href="{{ next_page_url }}"\n           class="px-3 py-1.5 text-sm border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">Next →</a>\n        {% else %}\n        <span class="px-3 py-1.5 text-sm border border-gray-100 rounded-lg text-gray-300 select-none">Next →</span>\n        {% endif %}\n      </div>\n      <div class="flex items-center gap-2 text-xs text-gray-400">\n        <span>Page {{ page }} of {{ total_pages }}</span>\n        <select onchange="location.href=this.options[this.selectedIndex].dataset.url"\n                class="px-2 py-1 border border-gray-200 rounded text-xs text-gray-500 bg-white cursor-pointer focus:outline-none focus:ring-1 focus:ring-apg-primary">\n          {% for option in per_page_options %}\n          <option value="{{ option.value }}" data-url="{{ option.url }}"{% if option.value == per %} selected{% endif %}>{{ option.value }} / page</option>\n          {% endfor %}\n        </select>\n      </div>\n    </nav>\n    {% endif %}\n  </section>\n\n</div>\n\n<details class="mt-4 apg-developer-panel">\n  <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 select-none">Developer exports</summary>\n  <div class="mt-2 flex items-center gap-2 flex-wrap">\n    <a href="{{ csv_url }}" class="apg-btn apg-btn-secondary">Export CSV</a>\n    <a href="{{ developer_api_url }}" class="apg-btn apg-btn-secondary">API JSON</a>\n  </div>\n  <p class="text-xs text-gray-400 mt-3">Rendered page records</p>\n  <pre class="mt-2 text-xs bg-gray-50 border border-gray-200 rounded-lg p-3 overflow-auto max-h-64 font-mono">{{ records_json }}</pre>\n</details>\n', 'workflow_wizard.html.j2': '<section class="max-w-2xl mx-auto" data-apg-live="{{ workflow_topic }}">\n  <p class="text-sm text-gray-500 mb-6">\n    <a href="/ui" class="hover:text-blue-600">Application</a> /\n    <a href="/ui/workflows" class="hover:text-blue-600">Workflows</a> /\n    <span class="font-semibold text-gray-900">{{ workflow.name }}</span>\n  </p>\n\n  {% if completed %}\n  <div class="apg-card text-center py-12">\n    <div class="text-5xl mb-4" aria-hidden="true">✓</div>\n    <h1 class="text-xl font-bold text-gray-900 mb-2">{{ workflow.name }} complete</h1>\n    <p class="text-gray-500 text-sm mb-6">Your {{ entity_name }} record has been created successfully.</p>\n    {% if run %}\n    <div class="mx-auto mb-6 max-w-sm rounded-xl border border-gray-200 bg-gray-50 p-4 text-left">\n      <div class="flex items-center justify-between gap-3">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-500">Recorded run</p>\n        <span class="apg-badge apg-badge-success">{{ run.status }}</span>\n      </div>\n      <p class="mt-2 font-mono text-sm text-gray-900">{{ run.id }}</p>\n      <p class="mt-1 text-xs text-gray-500">{{ run.completed_steps | length }} steps completed{% if run.created_record_id %} · Record {{ run.created_record_id }}{% endif %}</p>\n    </div>\n    {% endif %}\n    <div class="flex items-center justify-center gap-3 flex-wrap">\n      {% if safe_record_id %}\n      <a href="/ui/entities/{{ safe_entity }}/{{ safe_record_id }}" class="apg-btn">Open created record</a>\n      {% endif %}\n      {% if safe_run_id %}\n      <a href="/ui/debug/{{ safe_run_id }}" class="apg-btn apg-btn-secondary">Inspect run</a>\n      {% endif %}\n      <a href="/ui/entities/{{ safe_entity }}" class="apg-btn">View all {{ entity_name }} records</a>\n      <a href="/ui/workflows/{{ safe_entity }}/{{ safe_workflow_id }}" class="apg-btn apg-btn-secondary">Start again</a>\n      <a href="/ui/workflows" class="apg-btn apg-btn-secondary">All workflows</a>\n    </div>\n  </div>\n  {% else %}\n  <div class="text-center mb-8">\n    <div class="text-4xl mb-3" aria-hidden="true">{{ workflow.icon }}</div>\n    <h1 class="text-xl font-bold text-gray-900">{{ workflow.name }}</h1>\n    <p class="text-sm text-gray-500 mt-1">{{ workflow.description }}</p>\n  </div>\n\n  <ol class="flex items-center gap-0 mb-8 px-2" aria-label="Workflow progress">\n    {% for item in progress %}\n    <li class="flex items-center gap-1.5 text-xs font-medium {{ item.class_name }}">\n      <span class="w-5 h-5 rounded-full flex items-center justify-center text-xs {{ item.badge_class }}">{{ item.label }}</span>\n      <span class="hidden sm:block">{{ item.title }}</span>\n    </li>\n    {% if not loop.last %}\n    <li class="flex-1 h-px bg-gray-200 mx-1" aria-hidden="true"><span class="block h-px bg-blue-600" style="width:{{ \'100%\' if loop.index0 < step_index else \'0%\' }}"></span></li>\n    {% endif %}\n    {% endfor %}\n  </ol>\n\n  <article class="apg-card overflow-hidden">\n    <header class="px-6 py-4 border-b border-gray-100 bg-gray-50 -mx-4 -mt-4 mb-4">\n      <h2 class="font-semibold text-gray-900">Step {{ step_index + 1 }} of {{ total_steps }}: {{ step.title }}</h2>\n      <p class="text-sm text-gray-500 mt-0.5">{{ step.subtitle }}</p>\n    </header>\n    {% if error %}\n    <div role="alert" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{{ error }}</div>\n    {% endif %}\n    <form method="post" action="{{ next_url }}" class="space-y-4">\n      {{ hidden_fields | safe }}\n      {{ step_inputs | safe }}\n      <div class="flex items-center justify-between pt-4 border-t border-gray-100 mt-6">\n        {% if step_index > 0 %}\n        <a href="/ui/workflows/{{ safe_entity }}/{{ safe_workflow_id }}/step/{{ step_index - 1 }}" class="apg-btn apg-btn-secondary">Back</a>\n        {% else %}\n        <a href="/ui/workflows" class="apg-btn apg-btn-secondary">Cancel</a>\n        {% endif %}\n        <button type="submit" class="apg-btn">{{ next_label }}</button>\n      </div>\n    </form>\n  </article>\n  {% endif %}\n</section>\n', 'database_catalog.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Databases</span></p>\n  <div class="flex items-start justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">Database catalog</h1>\n      <p class="text-sm text-gray-500 mt-1">Schemas, generated tables, columns, indexes, and references for this app.</p>\n    </div>\n    <span class="apg-badge {{ \'apg-badge-success\' if status.valid else \'apg-badge-danger\' }}">{{ status_label }}</span>\n  </div>\n</section>\n\n<section class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6" aria-label="Database summary">\n  <article class="apg-card">\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">Databases</p>\n    <p class="text-2xl font-bold text-gray-900">{{ status.database_count }}</p>\n  </article>\n  <article class="apg-card">\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">Schemas</p>\n    <p class="text-2xl font-bold text-gray-900">{{ status.schema_count }}</p>\n  </article>\n  <article class="apg-card">\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">Tables</p>\n    <p class="text-2xl font-bold text-gray-900">{{ status.table_count }}</p>\n  </article>\n  <article class="apg-card">\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">References</p>\n    <p class="text-2xl font-bold text-gray-900">{{ status.reference_count }}</p>\n  </article>\n</section>\n\n{% if databases %}\n<section class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">\n  {% for database in databases %}\n  <article class="apg-card">\n    <header class="apg-card-header">\n      <div>\n        <h2 class="text-base font-semibold text-gray-900">{{ database.name }}</h2>\n        <p class="text-xs text-gray-500 mt-1">{{ database.type or \'database\' }}</p>\n      </div>\n      <a class="apg-btn apg-btn-secondary" href="/databases/{{ database.name }}/schemas">Schema JSON</a>\n    </header>\n    {% if database.connection_config %}\n    <dl class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">\n      {% for key, value in database.connection_config.items() %}\n      <div class="border border-gray-100 rounded-lg p-3">\n        <dt class="text-xs font-semibold uppercase tracking-wide text-gray-400">{{ key }}</dt>\n        <dd class="text-sm font-mono text-gray-900 mt-1">{{ value }}</dd>\n      </div>\n      {% endfor %}\n    </dl>\n    {% endif %}\n    {% if database.schemas %}\n    <div class="space-y-3">\n      {% for schema in database.schemas %}\n      <section class="border border-gray-200 rounded-lg p-3">\n        <div class="flex items-center justify-between gap-3 flex-wrap mb-3">\n          <div>\n            <h3 class="text-sm font-semibold text-gray-900">{{ schema.name }}</h3>\n            <p class="text-xs text-gray-500 mt-1">{{ schema.tables | length }} table{{ \'\' if schema.tables | length == 1 else \'s\' }}</p>\n          </div>\n          {% if schema.source %}\n          <span class="apg-badge apg-badge-neutral">{{ schema.source }}</span>\n          {% endif %}\n        </div>\n        <div class="flex items-center gap-2 flex-wrap">\n          {% for table in schema.tables %}\n          <a href="#table-{{ database.name }}-{{ schema.name }}-{{ table.name }}" class="apg-badge apg-badge-info">{{ table.name }}</a>\n          {% endfor %}\n        </div>\n      </section>\n      {% endfor %}\n    </div>\n    {% else %}\n    <p class="text-sm text-gray-500">No schemas declared.</p>\n    {% endif %}\n  </article>\n  {% endfor %}\n</section>\n{% else %}\n<section class="apg-card text-center">\n  <h2 class="text-base font-semibold text-gray-900">No databases declared</h2>\n  <p class="text-sm text-gray-500 mt-1">This generated application does not declare a database workspace.</p>\n</section>\n{% endif %}\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">\n  <div class="lg:col-span-2 space-y-4">\n    {% for database in databases %}\n      {% for schema in database.schemas %}\n        {% for table in schema.tables %}\n        <article id="table-{{ database.name }}-{{ schema.name }}-{{ table.name }}" class="apg-card">\n          <header class="apg-card-header">\n            <div>\n              <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">{{ database.name }} / {{ schema.name }}</p>\n              <h2 class="text-base font-semibold text-gray-900">{{ table.name }}</h2>\n            </div>\n            <span class="apg-badge apg-badge-neutral">{{ table.columns | length }} columns</span>\n          </header>\n          <div class="apg-table-wrap">\n            <table class="apg-table">\n              <thead>\n                <tr>\n                  <th>Column</th>\n                  <th>Type</th>\n                  <th>Constraints</th>\n                  <th>Reference</th>\n                </tr>\n              </thead>\n              <tbody>\n                {% for column in table.columns %}\n                <tr>\n                  <td><span class="font-mono text-gray-900">{{ column.name }}</span></td>\n                  <td><span class="apg-badge apg-badge-neutral">{{ column.type }}</span></td>\n                  <td>\n                    <div class="flex items-center gap-2 flex-wrap">\n                      {% if column.primary_key %}<span class="apg-badge apg-badge-success">Primary key</span>{% endif %}\n                      {% if column.required and not column.primary_key %}<span class="apg-badge apg-badge-warning">Required</span>{% endif %}\n                      {% if column.nullable %}<span class="apg-badge apg-badge-neutral">Nullable</span>{% endif %}\n                    </div>\n                  </td>\n                  <td>\n                    {% if column.reference %}\n                    <span class="font-mono text-sm text-gray-700">{{ column.reference.table }}.{{ column.reference.column }}</span>\n                    {% else %}\n                    <span class="text-xs text-gray-400">none</span>\n                    {% endif %}\n                  </td>\n                </tr>\n                {% endfor %}\n              </tbody>\n            </table>\n          </div>\n          {% if table.indexes %}\n          <details class="mt-3">\n            <summary class="cursor-pointer text-sm font-medium text-gray-700">Indexes</summary>\n            <ul class="mt-2 space-y-1">\n              {% for index in table.indexes %}\n              <li class="text-sm text-gray-600"><span class="font-mono">{{ index.name }}</span> on <span class="font-mono">{{ index.columns | join(\', \') }}</span></li>\n              {% endfor %}\n            </ul>\n          </details>\n          {% endif %}\n        </article>\n        {% endfor %}\n      {% endfor %}\n    {% endfor %}\n  </div>\n\n  <aside class="space-y-4">\n    <section class="apg-card">\n      <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Reference map</h2></header>\n      {% if relationships %}\n      <ol class="space-y-2">\n        {% for relationship in relationships %}\n        <li class="border border-gray-100 rounded-lg p-3">\n          <p class="text-sm font-mono text-gray-900">{{ relationship.source }}</p>\n          <p class="text-xs text-gray-500 mt-1">references {{ relationship.target }}{% if relationship.cardinality %} · {{ relationship.cardinality }}{% endif %}</p>\n        </li>\n        {% endfor %}\n      </ol>\n      {% else %}\n      <p class="text-sm text-gray-500">No foreign-key references declared.</p>\n      {% endif %}\n    </section>\n\n    <section class="apg-card">\n      <header class="apg-card-header">\n        <h2 class="text-base font-semibold text-gray-900">Validation</h2>\n        <span class="apg-badge {{ \'apg-badge-success\' if status.valid else \'apg-badge-danger\' }}">{{ status_label }}</span>\n      </header>\n      {% if status.validation.warnings %}\n      <ul class="space-y-2 mb-3">\n        {% for warning in status.validation.warnings %}\n        <li class="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">{{ warning }}</li>\n        {% endfor %}\n      </ul>\n      {% else %}\n      <p class="text-sm text-gray-500 mb-3">No schema warnings.</p>\n      {% endif %}\n      <details>\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Validation details</summary>\n        <pre>{{ validation_json }}</pre>\n      </details>\n    </section>\n  </aside>\n</section>\n', 'kanban_view.html.j2': '{# kanban_view.html.j2 — Kanban board view for status-field entities\n   Variables: entity_name, safe_entity, columns, display_field, status_field, fields,\n              status_options, total_records, wip_limit, list_url\n   columns: [{"label": str, "records": [dict]}]\n#}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Kanban</span>\n  <div class="ml-auto flex items-center gap-1">\n    <a href="/ui/entities/{{ safe_entity }}"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ≡ List\n    </a>\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">⊞ Kanban</span>\n  </div>\n</nav>\n\n<section class="apg-kanban-header">\n  <div>\n    <h1 class="text-xl font-bold text-gray-900">{{ entity_name }}</h1>\n    <p class="text-sm text-gray-500 mt-1">Grouped by {{ status_field | replace(\'_\', \' \') }} with drag-and-drop or keyboard move controls.</p>\n  </div>\n  <div class="apg-kanban-summary" aria-label="Board summary">\n    <a href="{{ list_url }}" class="apg-kanban-summary-item"><strong>{{ total_records }}</strong><span>Total</span></a>\n    <span class="apg-kanban-summary-item"><strong>{{ columns | length }}</strong><span>Columns</span></span>\n    <span class="apg-kanban-summary-item"><strong>{{ wip_limit }}</strong><span>WIP guide</span></span>\n  </div>\n</section>\n\n<section class="apg-kanban-intelligence" aria-label="{{ entity_name }} flow intelligence">\n  <article>\n    <div class="apg-card-header">\n      <div>\n        <h2>Cumulative Flow</h2>\n        <p>Detect expanding queues before they block delivery.</p>\n      </div>\n    </div>\n    <ol class="apg-flow-list">\n      {% for row in flow_rows %}\n      <li{% if row.over_limit %} class="warn"{% endif %}>\n        <span>{{ row.label }}</span>\n        <strong>{{ row.cumulative }}</strong>\n        <small>{{ row.percent }}% cumulative · {{ row.count }} in column</small>\n      </li>\n      {% endfor %}\n    </ol>\n  </article>\n  <article>\n    <div class="apg-card-header">\n      <div>\n        <h2>Swimlanes</h2>\n        <p>{% if swimlane_field %}Grouped by {{ swimlane_field | replace(\'_\', \' \') }}{% else %}Add owner, priority, country, or team to unlock lanes{% endif %}</p>\n      </div>\n    </div>\n    <div class="apg-swimlane-list">\n      {% for lane in swimlanes %}\n      <a href="{{ lane.url }}">\n        <span>{{ lane.label }}</span>\n        <strong>{{ lane.count }}</strong>\n      </a>\n      {% else %}\n      <p>No swimlane field detected yet.</p>\n      {% endfor %}\n    </div>\n  </article>\n  <article>\n    <div class="apg-card-header">\n      <div>\n        <h2>WIP Policy</h2>\n        <p>Limits are generated from current board shape.</p>\n      </div>\n    </div>\n    <p class="apg-kanban-policy"><strong>{{ wip_limit }}</strong> cards per column before warning. Over-limit columns stay linked to filtered record lists.</p>\n  </article>\n</section>\n\n{# Kanban board — horizontal scroll #}\n<div class="flex gap-4 overflow-x-auto pb-6 items-start -mx-1 px-1">\n  {% for col in columns %}\n  <div class="flex-shrink-0 w-72">\n    {# Column header #}\n    <div class="flex items-center justify-between mb-3 px-1">\n      <div class="flex items-center gap-2 min-w-0">\n        <span class="w-2.5 h-2.5 rounded-full\n          {% if col.label | lower in [\'active\', \'approved\', \'paid\', \'open\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-400\n          {% elif col.label | lower in [\'inactive\', \'rejected\', \'closed\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-400\n          {% elif col.label | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-400\n          {% else %}bg-gray-300{% endif %}"></span>\n        <h2 class="text-sm font-semibold text-gray-900">{{ col.label }}</h2>\n        <span class="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full font-medium">{{ col.count }}</span>\n      </div>\n      <a href="{{ col.list_url }}" class="text-xs text-gray-400 hover:text-apg-primary">List</a>\n    </div>\n    {% if col.over_limit %}\n    <p class="apg-kanban-wip">Above WIP guide {{ col.wip_limit }}</p>\n    {% endif %}\n\n    {# Card list — apg-kanban-col enables SortableJS drag-and-drop #}\n    <div class="space-y-2.5 apg-kanban-col min-h-16"\n         data-col-label="{{ col.label | e }}"\n         id="apg-col-{{ col.label | urlencode }}">\n      {% for record in col.records %}\n      {% set rec_id = record.get(\'id\', \'\') | string %}\n      <article data-record-id="{{ rec_id | e }}"\n               data-revision="{{ record.get(\'_revision\', \'\') | string | e }}"\n               class="apg-kanban-card bg-white rounded-xl border border-gray-200 p-4 hover:border-apg-primary hover:shadow-md transition-all group/card cursor-grab active:cursor-grabbing">\n        <div class="flex items-start justify-between gap-2 mb-2.5">\n          <div class="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n               style="background: var(--apg-primary, #0ea5e9)">\n            {{ (record.get(display_field, \'\') | string)[:1] | upper or \'?\' }}\n          </div>\n          <span class="text-xs text-gray-300 font-mono mt-1">{{ rec_id[:8] }}</span>\n        </div>\n        <a href="/ui/entities/{{ safe_entity }}/{{ rec_id | urlencode }}"\n           class="block text-sm font-semibold text-gray-900 group-hover/card:text-apg-primary transition-colors leading-tight mb-2">\n          {{ record.get(display_field, \'—\') | string | truncate(50) }}\n        </a>\n        {% set shown_fields = namespace(count=0) %}\n        {% for f in fields %}\n        {% if f.name not in [\'id\', \'_revision\', display_field, status_field] %}\n        {% set fval = record.get(f.name, \'\') %}\n        {% if shown_fields.count < 2 and fval and fval != \'\' and fval != \'None\' %}\n        <p class="text-xs text-gray-400 truncate mt-0.5">\n          <span class="font-medium text-gray-500">{{ (f.name[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if f.name.endswith(\'_id\') else (f.name | replace(\'_\', \' \') | title) }}:</span>\n          {{ fval | string | truncate(35) }}\n        </p>\n        {% set shown_fields.count = shown_fields.count + 1 %}\n        {% endif %}\n        {% endif %}\n        {% endfor %}\n        <form method="post" action="/ui/entities/{{ safe_entity }}/records/{{ rec_id | urlencode }}" class="apg-kanban-move">\n          <input type="hidden" name="expected_revision" value="{{ record.get(\'_revision\', \'\') | string | e }}">\n          <input type="hidden" name="return_view" value="kanban">\n          <label class="sr-only" for="apg-move-{{ rec_id | e }}">Move {{ record.get(display_field, rec_id) }}</label>\n          <select id="apg-move-{{ rec_id | e }}" name="{{ status_field }}" class="apg-kanban-select">\n            {% for option in status_options %}\n            <option value="{{ option }}"{% if option == col.label %} selected{% endif %}>{{ option }}</option>\n            {% endfor %}\n          </select>\n          <button type="submit" class="apg-btn apg-btn-secondary">Move</button>\n        </form>\n      </article>\n      {% endfor %}\n\n      {% if not col.records %}\n      <div class="bg-gray-50 rounded-xl border border-dashed border-gray-200 p-6 text-center apg-kanban-empty">\n        <p class="text-xs text-gray-400">No {{ col.label }} records.</p>\n        <a href="{{ col.list_url }}" class="text-xs text-apg-primary hover:underline">Open filtered list</a>\n      </div>\n      {% endif %}\n    </div>\n  </div>\n  {% endfor %}\n\n  {% if not columns %}\n  <div class="flex-1 text-center py-16 text-gray-400">\n    <div class="text-4xl mb-3 opacity-20">⊞</div>\n    <p class="text-sm">No records to display.</p>\n  </div>\n  {% endif %}\n</div>\n\n<script>\n(function() {\n  var entity = {{ safe_entity | tojson }};\n  var statusField = {{ status_field | tojson }};\n\n  document.querySelectorAll(\'.apg-kanban-col\').forEach(function(col) {\n    new Sortable(col, {\n      group: \'apg-kanban\',\n      animation: 150,\n      ghostClass: \'opacity-30\',\n      chosenClass: \'shadow-lg\',\n      dragClass: \'rotate-1\',\n      onEnd: function(evt) {\n        var card = evt.item;\n        var recordId = card.dataset.recordId;\n        var newCol = evt.to;\n        var newStatus = newCol.dataset.colLabel;\n        if (!recordId || !newStatus) return;\n\n        var body = {record: {}};\n        body.record[statusField] = newStatus;\n\n        fetch(\'/entities/\' + encodeURIComponent(entity) + \'/records/\' + encodeURIComponent(recordId), {\n          method: \'PUT\',\n          headers: {\'Content-Type\': \'application/json\'},\n          body: JSON.stringify(body)\n        }).then(function(r) {\n          if (r.ok) {\n            APGToast(\'Moved to \' + newStatus, \'success\');\n          } else {\n            APGToast(\'Move failed — \' + r.status, \'error\');\n            evt.from.insertBefore(card, evt.from.children[evt.oldIndex] || null);\n          }\n        }).catch(function() {\n          APGToast(\'Move failed\', \'error\');\n          evt.from.insertBefore(card, evt.from.children[evt.oldIndex] || null);\n        });\n      }\n    });\n  });\n})();\n</script>\n', 'landing.html.j2': '{# landing.html.j2 — APG application landing page #}\n<!doctype html>\n<html lang="{{ active_locale }}" dir="{{ text_direction }}" class="h-full">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>{{ module_name | replace(\'_\', \' \') | title }}</title>\n  <link rel="stylesheet" href="/static/apg.css">\n  <link rel="stylesheet" href="/theme.css">\n  <style>\n    :root {\n      --brand: {{ theme_primary }};\n      --accent: {{ theme_accent }};\n    }\n  </style>\n</head>\n<body class="min-h-full bg-gray-50 text-gray-900 font-sans antialiased">\n  <a class="apg-skip-link" href="#content">Skip to content</a>\n\n  <header class="bg-white border-b border-gray-200">\n    <nav class="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between gap-4">\n      <a href="/" class="text-lg font-bold text-gray-900">{{ module_name | replace(\'_\', \' \') | title }}</a>\n      <div class="flex items-center gap-2 flex-wrap justify-end">\n        <a href="/ui" class="apg-btn">{{ _(\'open_app\') }}</a>\n        <a href="/ui/marketplace" class="apg-btn apg-btn-secondary">{{ _(\'marketplace\') }}</a>\n      </div>\n    </nav>\n  </header>\n\n  <main id="content" class="max-w-6xl mx-auto px-6 py-10">\n    <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start mb-8">\n      <div>\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">Generated APG workspace</p>\n        <h1 class="text-4xl font-extrabold text-gray-900 mb-4">{{ module_name | replace(\'_\', \' \') | title }}</h1>\n        <p class="text-lg text-gray-600 mb-6 leading-relaxed">\n          {{ module_description or \'A generated operational application with data, workflows, automation, and integration surfaces ready to use.\' }}\n        </p>\n        <div class="flex items-center gap-3 flex-wrap">\n          <a href="/ui" class="apg-btn">{{ _(\'open_app\') }}</a>\n          {% if primary_entities %}\n          <a href="/ui/entities/{{ primary_entities[0].name }}" class="apg-btn apg-btn-secondary">Open {{ primary_entities[0].name }}</a>\n          {% endif %}\n          <a href="/openapi.json" class="apg-btn apg-btn-secondary">API contract</a>\n        </div>\n      </div>\n      <section class="apg-card" aria-label="Workspace readiness">\n        <h2 class="text-base font-semibold text-gray-900 mb-4">Workspace readiness</h2>\n        <div class="grid grid-cols-2 gap-3">\n          {% for stat in stats %}\n          <div class="border border-gray-100 rounded-lg p-3">\n            <p class="text-2xl font-bold text-gray-900">{{ stat.value }}</p>\n            <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">{{ stat.label }}</p>\n          </div>\n          {% endfor %}\n        </div>\n      </section>\n    </section>\n\n    <section class="mb-8" aria-label="Primary actions">\n      <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-4">Start here</h2>\n      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">\n        {% for action in workspace_actions %}\n        <a href="{{ action.url }}" class="apg-card hover:border-apg-primary transition-colors">\n          <h3 class="text-sm font-semibold text-gray-900">{{ action.label }}</h3>\n          <p class="text-xs text-gray-500 mt-2">{{ action.description }}</p>\n        </a>\n        {% endfor %}\n      </div>\n    </section>\n\n    {% if primary_entities %}\n    <section class="mb-8" aria-label="Data workspaces">\n      <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-4">{{ _(\'data_entities\') }}</h2>\n      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">\n        {% for entity in primary_entities %}\n        <a href="/ui/entities/{{ entity.name }}" class="apg-card hover:border-apg-primary transition-colors">\n          <div class="flex items-center justify-between gap-3 mb-3">\n            <span class="w-9 h-9 rounded-lg flex items-center justify-center text-white text-sm font-bold" style="background: var(--brand)">\n              {{ entity.name[0] | upper }}\n            </span>\n            <span class="apg-badge">{{ entity.type }}</span>\n          </div>\n          <h3 class="text-sm font-semibold text-gray-900">{{ entity.name }}</h3>\n          <p class="text-xs text-gray-500 mt-1">{{ entity.fields | length if entity.fields else entity.properties | length }} fields</p>\n        </a>\n        {% endfor %}\n      </div>\n    </section>\n    {% endif %}\n\n    <section class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">\n      <article class="apg-card">\n        <h2 class="text-base font-semibold text-gray-900 mb-3">Integration readiness</h2>\n        <p class="text-sm text-gray-500 mb-4">Generated APIs and marketplace blueprints make the app connectable without external assets.</p>\n        <div class="space-y-2">\n          {% for blueprint in marketplace_blueprints[:3] %}\n          <a href="{{ blueprint.href }}" class="flex items-center justify-between gap-3 border border-gray-100 rounded-lg p-3 hover:border-apg-primary">\n            <span class="text-sm font-semibold text-gray-900">{{ blueprint.title }}</span>\n            <span class="text-xs text-gray-400">{{ blueprint.category }}</span>\n          </a>\n          {% endfor %}\n        </div>\n      </article>\n      <article class="apg-card">\n        <h2 class="text-base font-semibold text-gray-900 mb-3">Developer surfaces</h2>\n        <div class="flex flex-wrap gap-2">\n          {% for link in api_links %}\n          <a href="{{ link.url }}" class="apg-btn apg-btn-secondary">{{ link.label }}</a>\n          {% endfor %}\n        </div>\n      </article>\n    </section>\n  </main>\n\n  <footer class="border-t border-gray-200 py-6 text-center text-xs text-gray-400">\n    Generated by <span class="font-medium text-gray-600">APG</span> · Datacraft\n  </footer>\n</body>\n</html>\n', 'capability_console.html.j2': '{% set operation = operation | default(\'\') %}\n{% set operation_label = operation_label | default(\'Result\') %}\n{% set rule_context_json = rule_context_json | default(\'{}\') %}\n{% set configuration_json = configuration_json | default(\'{}\') %}\n{% set approval_context_json = approval_context_json | default(\'{}\') %}\n{% set description = description | default({}) %}\n\n<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <a href="/capabilities" class="hover:text-blue-600">Capability catalog</a> / <span class="font-semibold text-gray-900">{{ name }}</span></p>\n  <div class="flex items-start justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">{{ name }}</h1>\n      <p class="text-sm text-gray-500 mt-1">Rules, configuration, and approval planning workspace.</p>\n    </div>\n    {% if description.contract %}\n    <span class="apg-badge apg-badge-neutral">{{ description.contract.get(\'name\', \'capability\') }}</span>\n    {% endif %}\n  </div>\n</section>\n\n{% if error %}\n<div role="alert" class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm mb-4">{{ error }}</div>\n{% endif %}\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4">\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/rules/evaluate" class="apg-card space-y-4">\n    <div>\n      <h2 class="text-base font-semibold text-gray-900">Rules evaluation</h2>\n      <p class="text-xs text-gray-500 mt-1">Test policy decisions with an explicit request context.</p>\n    </div>\n    <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Context JSON\n      <textarea name="context_json" rows="8" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">{{ rule_context_json }}</textarea>\n    </label>\n    <button type="submit" class="apg-btn">Evaluate rules</button>\n  </form>\n\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/configuration/resolve" class="apg-card space-y-4">\n    <div>\n      <h2 class="text-base font-semibold text-gray-900">Configuration</h2>\n      <p class="text-xs text-gray-500 mt-1">Preview resolved configuration after overrides.</p>\n    </div>\n    <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Overrides JSON\n      <textarea name="configuration_json" rows="8" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">{{ configuration_json }}</textarea>\n    </label>\n    <button type="submit" class="apg-btn">Resolve config</button>\n  </form>\n\n  <form method="post" action="/ui/capabilities/{{ safe_name }}/approval/plan" class="apg-card space-y-4">\n    <div>\n      <h2 class="text-base font-semibold text-gray-900">Approval plan</h2>\n      <p class="text-xs text-gray-500 mt-1">Identify approvers and review requirements before execution.</p>\n    </div>\n    <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Context JSON\n      <textarea name="context_json" rows="8" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">{{ approval_context_json }}</textarea>\n    </label>\n    <button type="submit" class="apg-btn">Plan approval</button>\n  </form>\n</section>\n\n<section class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">\n  <article class="apg-card">\n    <h2 class="text-base font-semibold text-gray-900 mb-3">Capability profile</h2>\n    {% if description.configuration %}\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Default configuration</p>\n    <dl class="space-y-2 mb-4">\n      {% for key, value in description.configuration.items() %}\n      <div class="flex items-center justify-between gap-3 border-b border-gray-100 py-2">\n        <dt class="text-sm text-gray-600">{{ key }}</dt>\n        <dd class="text-sm font-mono text-gray-900">{{ value }}</dd>\n      </div>\n      {% endfor %}\n    </dl>\n    {% endif %}\n    {% if description.rules %}\n    <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Declared rules</p>\n    <ol class="space-y-2 mb-4">\n      {% for rule in description.rules %}\n      <li class="border border-gray-100 rounded-lg p-3">\n        <p class="text-sm font-medium text-gray-900">{{ rule.name or rule.get(\'name\', \'rule\') }}</p>\n        <p class="text-xs text-gray-500 mt-1">{{ rule.when or rule.get(\'when\', \'\') }}</p>\n      </li>\n      {% endfor %}\n    </ol>\n    {% endif %}\n    <details>\n      <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw capability JSON</summary>\n      <pre>{{ description_json }}</pre>\n    </details>\n  </article>\n\n  <article class="apg-card">\n    <div class="flex items-center justify-between gap-3 mb-3">\n      <h2 class="text-base font-semibold text-gray-900">{{ operation_label }}</h2>\n      {% if result and result.decision %}\n      <span class="apg-badge {% if result.decision == \'allow\' %}apg-badge-success{% elif result.decision == \'deny\' %}apg-badge-danger{% else %}apg-badge-warning{% endif %}">{{ result.decision }}</span>\n      {% elif result and result.required is defined %}\n      <span class="apg-badge {% if result.required %}apg-badge-warning{% else %}apg-badge-success{% endif %}">{{ \'required\' if result.required else \'not required\' }}</span>\n      {% endif %}\n    </div>\n    {% if result %}\n      {% if result.matched_rules %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Matched rules</p>\n      <div class="flex items-center gap-2 flex-wrap mb-4">\n        {% for rule in result.matched_rules %}\n        <span class="apg-badge apg-badge-neutral">{{ rule }}</span>\n        {% endfor %}\n      </div>\n      {% endif %}\n      {% if result.actions %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Actions</p>\n      <ol class="space-y-2 mb-4">\n        {% for action in result.actions %}\n        <li class="border border-gray-100 rounded-lg p-3">\n          <p class="text-sm font-medium text-gray-900">{{ action.action or action.get(\'action\', \'\') }}</p>\n          <p class="text-xs text-gray-500 mt-1">{{ action.rule or action.get(\'rule\', \'\') }}</p>\n        </li>\n        {% endfor %}\n      </ol>\n      {% endif %}\n      {% if result.configuration %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Resolved configuration</p>\n      <dl class="space-y-2 mb-4">\n        {% for key, value in result.configuration.items() %}\n        <div class="flex items-center justify-between gap-3 border-b border-gray-100 py-2">\n          <dt class="text-sm text-gray-600">{{ key }}</dt>\n          <dd class="text-sm font-mono text-gray-900">{{ value }}</dd>\n        </div>\n        {% endfor %}\n      </dl>\n      {% endif %}\n      {% if result.approvers %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Approvers</p>\n      <div class="flex items-center gap-2 flex-wrap mb-4">\n        {% for approver in result.approvers %}\n        <span class="apg-badge apg-badge-neutral">{{ approver }}</span>\n        {% endfor %}\n      </div>\n      {% endif %}\n      <details>\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw result JSON</summary>\n        <pre>{{ result_json_html | safe }}</pre>\n      </details>\n    {% else %}\n    <p class="text-sm text-gray-500">Run a capability operation to view decision, configuration, or approval results.</p>\n    {% endif %}\n  </article>\n</section>\n', 'record_detail.html.j2': '{# record_detail.html.j2 — Salesforce-quality record detail page\n   Variables: entity_name, entity_type, safe_entity, safe_record_id,\n              record, fields, field_semantics, title, status_val, revision,\n              related_lists, related_count, prev_record_url, next_record_url,\n              record_url, has_kanban (bool)\n   related_lists: [{"entity": str, "fk_field": str, "records": [dict], "cols": [str]}]\n   field_semantics: {field_name: semantic_type}\n#}\n\n{# Breadcrumb #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900 truncate max-w-xs">{{ title }}</span>\n</nav>\n\n{# Record header card #}\n<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5 overflow-hidden">\n  <div class="h-1 bg-apg-primary"></div>\n  <div class="px-6 py-5 flex items-start gap-4">\n    <div class="w-14 h-14 rounded-xl flex items-center justify-center text-white text-2xl font-bold flex-shrink-0"\n         style="background: var(--apg-primary, #0ea5e9)">\n      {{ (title[:1] | upper) if title else (entity_name[:1] | upper) }}\n    </div>\n    <div class="flex-1 min-w-0">\n      <div class="flex items-center gap-3 flex-wrap">\n        <h1 class="text-xl font-bold text-gray-900 break-all">{{ title }}</h1>\n        {% if status_val %}\n        <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold\n          {% if status_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\'] %}bg-green-100 text-green-800\n          {% elif status_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n          {% elif status_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n          {% else %}bg-gray-100 text-gray-600{% endif %}">\n          {{ status_val }}\n        </span>\n        {% endif %}\n        <span class="text-xs font-medium text-gray-400 bg-gray-50 px-2 py-0.5 rounded uppercase tracking-wide">{{ entity_type }}</span>\n      </div>\n      {% set id_val = record.get(\'id\', \'\') %}\n      {% if id_val %}\n      <p class="text-xs text-gray-400 mt-1 font-mono truncate">{{ id_val | string }}</p>\n      {% endif %}\n    </div>\n    <div class="flex items-center gap-2 flex-shrink-0 flex-wrap">\n      {% if prev_record_url %}\n      <a href="{{ prev_record_url }}" class="apg-btn apg-btn-secondary">Previous</a>\n      {% endif %}\n      {% if next_record_url %}\n      <a href="{{ next_record_url }}" class="apg-btn apg-btn-secondary">Next</a>\n      {% endif %}\n      <button type="button"\n              class="apg-btn apg-btn-secondary"\n              onclick="navigator.clipboard && navigator.clipboard.writeText(location.origin + \'{{ record_url }}\'); APGToast(\'Record link copied\', \'success\');">\n        Copy link\n      </button>\n      {% if has_kanban %}\n      <a href="/ui/entities/{{ safe_entity }}?view=kanban" class="apg-btn apg-btn-secondary">Kanban</a>\n      {% endif %}\n      <a href="/ui/workflows/{{ safe_entity }}/create_{{ safe_entity }}"\n         class="apg-btn">\n        Workflow\n      </a>\n      <form method="post"\n            action="/ui/entities/{{ safe_entity }}/records/{{ safe_record_id }}/delete"\n            class="inline"\n            onsubmit="return apgConfirmSubmit(this, \'Delete this record? This cannot be undone.\')">\n        <input type="hidden" name="expected_revision" value="{{ revision }}">\n        <button type="submit"\n                class="px-3 py-1.5 text-sm font-medium border border-red-200 text-red-500 rounded-lg hover:bg-red-50 transition-colors">\n          Delete\n        </button>\n      </form>\n    </div>\n  </div>\n{# Highlights panel — top fields at a glance #}\n{% set highlight_fields = [] %}\n{% for f in fields %}\n  {% if f.name not in [\'id\', \'_revision\'] and not f.name.endswith(\'_id\') %}\n    {% if highlight_fields | length < 4 %}\n      {% set _ = highlight_fields.append(f) %}\n    {% endif %}\n  {% endif %}\n{% endfor %}\n{% if highlight_fields %}\n<div class="border-t border-gray-100 px-6 py-3 grid grid-cols-2 md:grid-cols-4 gap-4 bg-gray-50/50">\n  {% for f in highlight_fields %}\n  {% set fv = record.get(f.name, \'\') %}\n  {% set sem = field_semantics.get(f.name, \'text\') if field_semantics else \'text\' %}\n  <div class="min-w-0">\n    <p class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-0.5 truncate">\n      {{ (f.name[:-3] | replace(\'_\',\' \') | title ~ \' ID\') if f.name.endswith(\'_id\') else (f.name | replace(\'_\',\' \') | title) }}\n    </p>\n    <p class="text-sm font-medium text-gray-900 truncate">\n      {% if fv is none or fv == \'\' or fv | string == \'None\' %}\n      <span class="text-gray-300 italic text-xs">—</span>\n      {% elif sem == \'currency\' %}\n      <span class="tabular-nums">{{ format_currency(fv) }}</span>\n      {% elif sem == \'status\' %}\n      <span class="inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-semibold\n        {% if fv | string | lower in [\'active\',\'approved\',\'paid\',\'open\',\'enabled\',\'complete\',\'completed\',\'success\',\'done\'] %}bg-green-100 text-green-800\n        {% elif fv | string | lower in [\'inactive\',\'rejected\',\'closed\',\'disabled\',\'cancelled\',\'canceled\',\'failed\',\'expired\'] %}bg-red-100 text-red-800\n        {% else %}bg-yellow-100 text-yellow-800{% endif %}">{{ fv }}</span>\n      {% else %}\n      {{ fv | string | truncate(30) }}\n      {% endif %}\n    </p>\n  </div>\n  {% endfor %}\n</div>\n{% endif %}\n</div>\n\n{% if detail_intelligence %}\n<section class="apg-detail-command-center" aria-label="Record intelligence">\n  <article class="apg-detail-command-card">\n    <div class="apg-detail-command-head">\n      <span>Change Diff Timeline</span>\n      <strong>rev. {{ revision }}</strong>\n    </div>\n    {% if detail_intelligence.diff_fields %}\n    <ol class="apg-detail-diff-list">\n      {% for item in detail_intelligence.diff_fields %}\n      <li>\n        <span>{{ item.name }}</span>\n        <strong>{{ item.value }}</strong>\n      </li>\n      {% endfor %}\n    </ol>\n    {% else %}\n    <p class="apg-detail-empty">No populated fields to compare yet.</p>\n    {% endif %}\n    <small>{{ detail_intelligence.activity_count }} activity events captured</small>\n  </article>\n\n  <article class="apg-detail-command-card">\n    <div class="apg-detail-command-head">\n      <span>Related Record Graph</span>\n      <strong>{{ related_count }} links</strong>\n    </div>\n    {% if detail_intelligence.related_graph %}\n    <div class="apg-detail-graph">\n      <div class="apg-detail-node apg-detail-node-root">{{ entity_name }}</div>\n      {% for rel in detail_intelligence.related_graph %}\n      <a class="apg-detail-node" href="{{ rel.url }}">\n        {{ rel.entity }}\n        <small>{{ rel.count }} via {{ rel.field }}</small>\n      </a>\n      {% endfor %}\n    </div>\n    {% else %}\n    <p class="apg-detail-empty">No downstream records reference this item yet.</p>\n    {% endif %}\n  </article>\n\n  <article class="apg-detail-command-card">\n    <div class="apg-detail-command-head">\n      <span>Create Sibling Context</span>\n      <strong>{{ detail_intelligence.sibling_fields | length }} defaults</strong>\n    </div>\n    {% if detail_intelligence.sibling_fields %}\n    <dl class="apg-detail-sibling-list">\n      {% for item in detail_intelligence.sibling_fields %}\n      <div>\n        <dt>{{ item.name }}</dt>\n        <dd>{{ item.value or \'Empty\' }}</dd>\n      </div>\n      {% endfor %}\n    </dl>\n    {% else %}\n    <p class="apg-detail-empty">No safe sibling defaults detected.</p>\n    {% endif %}\n    <a href="{{ detail_intelligence.create_sibling_url }}" class="apg-detail-action">Start from this shape</a>\n  </article>\n</section>\n{% endif %}\n\n{# Tab bar #}\n<div class="flex items-center gap-1 border-b border-gray-200 mb-6">\n  <button onclick="apgTab(\'details\')" id="apg-tab-details"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-apg-primary text-apg-primary -mb-px transition-colors">\n    Details\n  </button>\n  <button onclick="apgTab(\'related\')" id="apg-tab-related"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Related\n    {% if related_lists %}\n    <span class="ml-1 text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">\n      {{ related_count }}\n    </span>\n    {% endif %}\n  </button>\n  <button onclick="apgTab(\'activity\')" id="apg-tab-activity"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Activity\n  </button>\n</div>\n\n{# Details panel #}\n<div id="apg-panel-details">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n      <h2 class="text-sm font-semibold text-gray-900">Record Details</h2>\n      <span class="text-xs text-gray-400 font-mono">rev. {{ revision }}</span>\n    </div>\n    <div class="p-5 grid grid-cols-1 md:grid-cols-2 gap-x-8">\n      {% for field in fields %}\n      {% if field.name != \'_revision\' %}\n      {% set field_val = record.get(field.name, \'\') %}\n      {% set fld_id = \'fld-\' ~ safe_entity ~ \'-\' ~ safe_record_id ~ \'-\' ~ field.name %}\n      <div id="{{ fld_id }}" class="py-3 border-b border-gray-50 last:border-0 group/field">\n        <dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">\n          {{ (field.name[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if field.name.endswith(\'_id\') else (field.name | replace(\'_\', \' \') | title) }}\n        </dt>\n        <dd class="flex items-center justify-between gap-2 min-h-6">\n          <span class="text-sm text-gray-900 break-words">\n            {% set semantic = field_semantics.get(field.name, \'text\') if field_semantics else \'text\' %}\n            {% include \'widgets/field_display.html.j2\' %}\n          </span>\n          <button\n            hx-get="/ui/entities/{{ safe_entity }}/{{ safe_record_id }}/fields/{{ field.name }}/edit"\n            hx-target="#{{ fld_id }}"\n            hx-swap="outerHTML"\n            class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"\n            title="Edit {{ field.name }}">\n            <svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">\n              <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zm-2.207 2.207L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>\n            </svg>\n          </button>\n        </dd>\n      </div>\n      {% endif %}\n      {% endfor %}\n    </div>\n  </div>\n</div>\n\n{# Related panel #}\n<div id="apg-panel-related" class="hidden">\n  {% if related_lists %}\n    {% for rel in related_lists %}\n    <div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-4">\n      <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n        <div class="flex items-center gap-2">\n          <h2 class="text-sm font-semibold text-gray-900">{{ rel.entity }}</h2>\n          <span class="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded-full font-medium">{{ rel.records | length }}</span>\n        </div>\n        <a href="{{ rel.list_url }}"\n           class="text-xs text-apg-primary hover:underline">View filtered</a>\n      </div>\n      {% if rel.records %}\n      <div class="overflow-x-auto">\n        <table class="w-full text-sm">\n          <thead>\n            <tr class="bg-gray-50 border-b border-gray-100">\n              {% for col in rel.cols %}\n              <th class="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">\n                {{ (col[:-3] | replace(\'_\', \' \') | title ~ \' ID\') if col.endswith(\'_id\') else (col | replace(\'_\', \' \') | title) }}\n              </th>\n              {% endfor %}\n              <th class="px-4 py-2 w-16"></th>\n            </tr>\n          </thead>\n          <tbody class="divide-y divide-gray-50">\n            {% for row in rel.records[:5] %}\n            <tr class="hover:bg-gray-50 transition-colors">\n              {% for col in rel.cols %}\n              <td class="px-4 py-2.5 text-gray-700 max-w-xs truncate">\n                {{ row.get(col, \'\') | string | truncate(40) }}\n              </td>\n              {% endfor %}\n              <td class="px-4 py-2.5 text-right">\n                <a href="/ui/entities/{{ rel.entity | urlencode }}/{{ row.get(\'id\', \'\') | string | urlencode }}"\n                   class="text-xs text-apg-primary hover:underline font-medium">View →</a>\n              </td>\n            </tr>\n            {% endfor %}\n            {% if rel.records | length > 5 %}\n            <tr>\n              <td colspan="{{ rel.cols | length + 1 }}" class="px-4 py-2.5 text-center text-xs text-gray-400">\n                + {{ rel.records | length - 5 }} more —\n                <a href="{{ rel.list_url }}" class="text-apg-primary hover:underline">view filtered</a>\n              </td>\n            </tr>\n            {% endif %}\n          </tbody>\n        </table>\n      </div>\n      {% else %}\n      <div class="px-4 py-8 text-center text-sm text-gray-400">\n        <p>No related {{ rel.entity }} records through {{ rel.fk_field }}.</p>\n        <div class="mt-3 flex items-center justify-center gap-2 flex-wrap">\n          <a href="{{ rel.list_url }}" class="apg-btn apg-btn-secondary">Open filtered list</a>\n          <a href="{{ rel.create_url }}" class="apg-btn">Create {{ rel.entity }}</a>\n        </div>\n      </div>\n      {% endif %}\n    </div>\n    {% endfor %}\n  {% else %}\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n    <div class="text-4xl mb-3 opacity-20">🔗</div>\n    <p class="text-sm font-medium text-gray-500">No related records found.</p>\n    <p class="text-xs text-gray-400 mt-1">Other entities with FK fields pointing to {{ entity_name }} appear here.</p>\n  </div>\n  {% endif %}\n</div>\n\n{# Activity panel #}\n<div id="apg-panel-activity" class="hidden">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100">\n      <h2 class="text-sm font-semibold text-gray-900">Activity</h2>\n    </div>\n    <div class="p-5">\n      <ol class="relative border-l-2 border-gray-100 ml-4 space-y-5">\n        {% if activity_events %}\n          {% for ev in activity_events %}\n          <li class="ml-6">\n            <span class="absolute flex items-center justify-center w-8 h-8 rounded-full -left-4 text-sm ring-4 ring-white\n              {% if ev.type == \'create\' %}bg-blue-50\n              {% elif ev.type == \'update\' %}bg-purple-50\n              {% elif ev.type == \'delete\' %}bg-red-50\n              {% elif ev.type == \'note\' %}bg-yellow-50\n              {% else %}bg-gray-50{% endif %}">\n              {% if ev.type == \'create\' %}📋\n              {% elif ev.type == \'update\' %}✏️\n              {% elif ev.type == \'delete\' %}🗑️\n              {% elif ev.type == \'note\' %}💬\n              {% else %}⚡{% endif %}\n            </span>\n            <div class="pl-2">\n              <p class="text-sm font-medium text-gray-900">{{ ev.detail or (ev.type | title) }}</p>\n              <p class="text-xs text-gray-400 mt-0.5">\n                {{ ev.actor or \'APG\' }}\n                {% if ev.ts %} · {{ ev.ts }}{% endif %}\n              </p>\n            </div>\n          </li>\n          {% endfor %}\n        {% else %}\n          <li class="ml-6">\n            <span class="absolute flex items-center justify-center w-8 h-8 bg-blue-50 rounded-full -left-4 text-sm ring-4 ring-white">📋</span>\n            <div class="pl-2">\n              <p class="text-sm font-medium text-gray-900">Record created</p>\n              <p class="text-xs text-gray-400 mt-0.5">Revision {{ revision }} · via APG</p>\n            </div>\n          </li>\n        {% endif %}\n      </ol>\n      <form method="post"\n            action="/ui/entities/{{ safe_entity }}/records/{{ safe_record_id }}/note"\n            class="mt-8 flex gap-3">\n        <div class="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n             style="background: var(--apg-primary, #0ea5e9)">A</div>\n        <div class="flex-1">\n          <textarea name="note" placeholder="Add a note…" rows="2" required\n                    class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent placeholder-gray-300"></textarea>\n          <button type="submit"\n                  class="mt-1.5 px-3 py-1.5 bg-apg-primary text-white text-xs font-medium rounded-lg hover:opacity-90 transition-opacity">\n            Save Note\n          </button>\n        </div>\n      </form>\n    </div>\n  </div>\n</div>\n\n<script>\nfunction apgTab(name) {\n  document.querySelectorAll(\'.apg-tab-btn\').forEach(function(b) {\n    b.classList.remove(\'border-apg-primary\', \'text-apg-primary\');\n    b.classList.add(\'border-transparent\', \'text-gray-500\');\n  });\n  document.querySelectorAll(\'[id^="apg-panel-"]\').forEach(function(p) { p.classList.add(\'hidden\'); });\n  var btn = document.getElementById(\'apg-tab-\' + name);\n  if (btn) {\n    btn.classList.remove(\'border-transparent\', \'text-gray-500\');\n    btn.classList.add(\'border-apg-primary\', \'text-apg-primary\');\n  }\n  var panel = document.getElementById(\'apg-panel-\' + name);\n  if (panel) panel.classList.remove(\'hidden\');\n}\n</script>\n', 'login.html.j2': '<main id="content" class="apg-login-page">\n  <section class="apg-login-card" aria-labelledby="apg-login-title">\n    <div class="mb-4">\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Authentication required</p>\n      <h1 id="apg-login-title">{{ module_name }}</h1>\n      <p class="text-sm text-gray-500 mt-2">Secure workspace sign-in.</p>\n    </div>\n    {% if error %}\n    <div role="alert" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{{ error }}</div>\n    {% endif %}\n    <form method="post" action="/login" class="space-y-4">\n      <input type="hidden" name="next" value="{{ next_url }}">\n      <label for="apg-login-username" class="block text-sm font-semibold text-gray-700">Username\n        <input id="apg-login-username" name="username" value="{{ username | default(\'\') }}" autocomplete="username" required autofocus class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n      </label>\n      <label for="apg-login-password" class="block text-sm font-semibold text-gray-700">Password\n        <input id="apg-login-password" name="password" type="password" autocomplete="current-password" required class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n      </label>\n      <button type="submit" class="apg-btn w-full justify-center">Sign in</button>\n    </form>\n    <div class="mt-4 border border-gray-100 rounded-lg p-3">\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">Session</p>\n      <p class="text-sm text-gray-600 mt-1">Continue to <span class="font-mono">{{ next_url }}</span> after authentication.</p>\n    </div>\n  </section>\n</main>\n', 'entity_analytics.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-apg-primary">Application</a> / <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary">{{ entity_name }}</a> / <span class="font-semibold text-gray-900">Analytics</span></p>\n  <div class="flex items-center justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">{{ entity_name }} Analytics</h1>\n      <p class="text-sm text-gray-500 mt-1">{{ total }} record{{ \'s\' if total != 1 else \'\' }}</p>\n    </div>\n    <a class="apg-btn apg-btn-secondary" href="/ui/entities/{{ safe_entity }}">Table</a>\n  </div>\n</section>\n\n<section class="apg-analytics-metrics" aria-label="{{ entity_name }} analytics summary">\n  {% for metric in metrics %}\n  <a href="{{ metric.url }}" class="apg-analytics-metric">\n    <span>{{ metric.label }}</span>\n    <strong>{{ metric.value }}</strong>\n    <small>{{ metric.hint }}</small>\n  </a>\n  {% endfor %}\n</section>\n\n<section class="apg-analytics-decisions" aria-label="{{ entity_name }} decision intelligence">\n  {% for item in analytics_decisions %}\n  <article>\n    <span>{{ item.label }}</span>\n    <strong>{{ item.value }}</strong>\n    <small>{{ item.hint }}</small>\n    <a href="{{ item.url }}">Inspect records</a>\n  </article>\n  {% endfor %}\n</section>\n\n<section class="apg-grid-2 gap-6 mb-6">\n  <article class="apg-card">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900">Records Over Time</h2>\n        <p class="text-xs text-gray-400 mt-1">{% if date_field %}Grouped by {{ date_field }}{% else %}Add a date field for trend history{% endif %}</p>\n      </div>\n    </div>\n    <div class="apg-chart" data-apg-chart="{{ line_chart.id }}"></div>\n    <script id="{{ line_chart.id }}" type="application/json">{{ line_chart.spec_json | safe }}</script>\n  </article>\n  <article class="apg-card">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900">Status Distribution</h2>\n        <p class="text-xs text-gray-400 mt-1">{% if status_field %}Click a segment below to inspect records{% else %}No status field detected{% endif %}</p>\n      </div>\n    </div>\n    <div class="apg-chart" data-apg-chart="{{ status_chart.id }}"></div>\n    <script id="{{ status_chart.id }}" type="application/json">{{ status_chart.spec_json | safe }}</script>\n    {% if status_rows %}\n    <div class="apg-status-drilldown" aria-label="Status drilldown">\n      {% for row in status_rows %}\n      <a href="{{ row.url }}" class="apg-status-row">\n        <span>{{ row.label }}</span>\n        <strong>{{ row.count }}</strong>\n        <small>{{ row.percent }}%</small>\n      </a>\n      {% endfor %}\n    </div>\n    {% endif %}\n  </article>\n</section>\n\n{% if insights %}\n<section class="apg-analytics-insights" aria-label="Analytics insights">\n  {% for insight in insights %}\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 mb-2">{{ insight.title }}</h2>\n    <p class="text-sm text-gray-500">{{ insight.body }}</p>\n    <a href="{{ insight.url }}" class="apg-btn apg-btn-secondary mt-3">{{ insight.action }}</a>\n  </article>\n  {% endfor %}\n</section>\n{% endif %}\n\n<section class="apg-grid-3 gap-4" aria-label="Numeric field statistics">\n  {% for stat in numeric_stats %}\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 mb-3">{{ stat.field }}</h2>\n    <dl class="grid grid-cols-3 gap-3 text-center">\n      <div><dt class="text-xs text-gray-400">Min</dt><dd class="font-semibold">{{ stat.min }}</dd></div>\n      <div><dt class="text-xs text-gray-400">Avg</dt><dd class="font-semibold">{{ stat.avg }}</dd></div>\n      <div><dt class="text-xs text-gray-400">Max</dt><dd class="font-semibold">{{ stat.max }}</dd></div>\n    </dl>\n    <p class="text-xs text-gray-400 mt-3">{{ stat.count }} measured record{{ \'s\' if stat.count != 1 else \'\' }}</p>\n  </article>\n  {% else %}\n  <article class="apg-card"><div class="apg-chart-empty"><p>No numeric fields available for {{ entity_name }}</p></div></article>\n  {% endfor %}\n</section>\n', 'agent_console.html.j2': '{% set description = description | default({}) %}\n{% set user_message = user_message | default(\'\') %}\n{% set payload_json = payload_json | default(\'{}\') %}\n{% set result_status = result_status | default(\'\') %}\n{% set team_members = team_members | default([]) %}\n{% set team_flow = team_flow | default([]) %}\n<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <a href="/agents" class="hover:text-blue-600">Agent catalog</a> / <span class="font-semibold text-gray-900">{{ name }}</span></p>\n  <div class="flex items-start justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">{{ name }}</h1>\n      <p class="text-sm text-gray-500 mt-1">{{ \'Team console\' if team else \'Agent console\' }}</p>\n    </div>\n    {% if result_status %}\n    <span class="apg-badge apg-badge-neutral">{{ result_status }}</span>\n    {% endif %}\n  </div>\n</section>\n\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4">\n  <article class="apg-card lg:col-span-2 flex flex-col min-h-16" data-apg-live="{{ live_topic }}">\n    <header class="apg-card-header">\n      <div>\n        <h2 class="text-base font-semibold text-gray-900">Conversation</h2>\n        <p class="text-xs text-gray-500">Prompt, stream, inspect, and retry from one place.</p>\n      </div>\n      <span class="text-xs text-gray-400">{{ live_topic }}</span>\n    </header>\n\n    <div id="apg-agent-stream" class="flex-1 space-y-3 mb-4" aria-live="polite" aria-busy="false">\n      {% if not user_message and not result and not error %}\n      <div class="rounded-lg bg-gray-50 border border-gray-200 p-4">\n        <p class="text-sm text-gray-600">Ask {{ name }} to work on a concrete task. Add JSON only when the agent needs structured context.</p>\n      </div>\n      {% endif %}\n      {% if error %}\n      <div role="alert" class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{{ error }}</div>\n      {% endif %}\n      {% if user_message %}\n      <div class="rounded-lg border border-gray-200 p-4 bg-gray-50">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">You</p>\n        <p class="text-sm text-gray-700">{{ user_message }}</p>\n      </div>\n      {% endif %}\n      {% if result %}\n      <div class="rounded-lg border border-gray-200 p-4 bg-white">\n        <div class="flex items-center justify-between gap-3 mb-2">\n          <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">{{ \'Team response\' if team else \'Agent response\' }}</p>\n          {% if result_status %}<span class="apg-badge apg-badge-neutral">{{ result_status }}</span>{% endif %}\n        </div>\n        <div class="apg-agent-output text-sm text-gray-700">{{ result_html|safe }}</div>\n      </div>\n      {% endif %}\n    </div>\n\n    <form method="post" action="{{ action }}" class="space-y-4" data-apg-stream-target="#apg-agent-stream">\n      <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Message\n        <textarea name="message" rows="3" autocomplete="off" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">{{ user_message }}</textarea>\n      </label>\n      <details>\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Structured payload</summary>\n        <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mt-3">Payload JSON\n          <textarea name="payload_json" rows="6" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">{{ payload_json }}</textarea>\n        </label>\n      </details>\n      <div class="flex items-center justify-between gap-3 flex-wrap">\n        <label class="flex items-center gap-2 text-sm text-gray-600"><input name="stream" type="checkbox" value="true"> Stream when supported</label>\n        <div class="flex items-center gap-2">\n          <button type="submit" class="apg-btn">Send</button>\n          <button type="button" class="apg-btn apg-btn-secondary" onclick="document.getElementById(\'apg-agent-stream\').setAttribute(\'aria-busy\',\'false\')">Stop</button>\n        </div>\n      </div>\n    </form>\n  </article>\n\n  <aside class="space-y-4">\n    {% if team %}\n    <section class="apg-card">\n      <h2 class="text-base font-semibold text-gray-900 mb-3">Team lanes</h2>\n      {% if team_members %}\n      <ol class="space-y-2">\n        {% for member in team_members %}\n        <li class="flex items-center justify-between gap-3 border border-gray-100 rounded-lg p-3">\n          <a href="/ui/agents/{{ member }}" class="text-sm font-medium text-apg-primary hover:underline">{{ member }}</a>\n          <span class="apg-badge apg-badge-neutral">member</span>\n        </li>\n        {% endfor %}\n      </ol>\n      {% else %}\n      <p class="text-sm text-gray-500">No member agents declared for this team.</p>\n      {% endif %}\n    </section>\n\n    <section class="apg-card">\n      <h2 class="text-base font-semibold text-gray-900 mb-3">Handoff flow</h2>\n      {% if team_flow %}\n      <ol class="space-y-2">\n        {% for edge in team_flow %}\n        <li class="text-sm text-gray-600"><span class="font-medium text-gray-900">{{ edge.source }}</span> -> <span class="font-medium text-gray-900">{{ edge.target }}</span></li>\n        {% endfor %}\n      </ol>\n      {% else %}\n      <p class="text-sm text-gray-500">No explicit handoff flow declared.</p>\n      {% endif %}\n    </section>\n    {% endif %}\n\n    <section class="apg-card">\n      <h2 class="text-base font-semibold text-gray-900 mb-3">Configuration</h2>\n      {% if description.role %}<p class="text-sm text-gray-600"><span class="font-medium text-gray-900">Role:</span> {{ description.role }}</p>{% endif %}\n      {% if description.model %}<p class="text-sm text-gray-600 mt-1"><span class="font-medium text-gray-900">Model:</span> {{ description.model }}</p>{% endif %}\n      {% if description.runtime %}<p class="text-sm text-gray-600 mt-1"><span class="font-medium text-gray-900">Runtime:</span> {{ description.runtime }}</p>{% endif %}\n      {% if description.tools %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mt-4 mb-2">Tools</p>\n      <div class="flex items-center gap-2 flex-wrap">\n        {% for tool in description.tools %}\n        <span class="apg-badge apg-badge-neutral">{{ tool }}</span>\n        {% endfor %}\n      </div>\n      {% endif %}\n      {% if description.capabilities %}\n      <p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mt-4 mb-2">Capabilities</p>\n      <div class="flex items-center gap-2 flex-wrap">\n        {% for capability in description.capabilities %}\n        <span class="apg-badge apg-badge-neutral">{{ capability }}</span>\n        {% endfor %}\n      </div>\n      {% endif %}\n      <details class="mt-4">\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw description JSON</summary>\n        <pre>{{ description_json }}</pre>\n      </details>\n      {% if result %}\n      <details class="mt-4">\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Raw response JSON</summary>\n        <pre>{{ result_json }}</pre>\n      </details>\n      {% endif %}\n    </section>\n  </aside>\n</section>\n', 'marketplace.html.j2': '{# marketplace.html.j2 — APG Connector Marketplace #}\n\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Connector Marketplace</span>\n</nav>\n\n<section class="mb-6">\n  <div class="flex items-start justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">Connector Marketplace</h1>\n      <p class="text-sm text-gray-500 mt-1">\n        Discover installed connectors and generated integration blueprints for this application.\n      </p>\n    </div>\n    {% if has_installed_connectors %}\n    <span class="apg-badge apg-badge-success">{{ installed_count }} installed</span>\n    {% else %}\n    <span class="apg-badge">{{ installed_count }} installed</span>\n    {% endif %}\n  </div>\n</section>\n\n<section class="apg-card apg-list-toolbar mb-6" aria-label="Marketplace discovery">\n  <form method="get" action="/ui/marketplace" class="grid grid-cols-1 md:grid-cols-2 gap-3">\n    <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Search\n      <input name="q" value="{{ query }}" placeholder="Search connectors..."\n             class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n    </label>\n    <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Category\n      <select name="category" class="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary bg-white">\n        <option value="all"{% if active_category == \'all\' %} selected{% endif %}>All categories</option>\n        {% for category in categories %}\n        <option value="{{ category.name }}"{% if category.active %} selected{% endif %}>{{ category.name }} ({{ category.count }})</option>\n        {% endfor %}\n      </select>\n    </label>\n    <div class="flex items-center gap-2">\n      <button type="submit" class="apg-btn">Apply</button>\n      {% if has_filters %}\n      <a href="/ui/marketplace" class="apg-btn apg-btn-secondary">Clear</a>\n      {% endif %}\n    </div>\n  </form>\n  <div class="flex items-center gap-2 flex-wrap mt-4 text-xs text-gray-500">\n    <span>{{ filtered_count }} of {{ connector_count }} shown</span>\n    {% for category in categories %}\n    <a href="/ui/marketplace?category={{ category.name | urlencode }}"\n       class="px-2 py-1 border border-gray-200 rounded-lg hover:border-apg-primary{% if category.active %} text-apg-primary border-apg-primary{% endif %}">\n      {{ category.name }}\n    </a>\n    {% endfor %}\n  </div>\n</section>\n\n{% if connectors %}\n<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">\n  {% for conn in connectors %}\n  <article class="apg-card hover:border-apg-primary transition-colors">\n    <div class="flex items-start gap-4 mb-3">\n      <div class="w-12 h-12 rounded-xl flex items-center justify-center text-white text-xl font-bold flex-shrink-0"\n           style="background: var(--apg-primary, #1E5B5A)">\n        {{ (conn.title or conn.name or \'?\')[:1] | upper }}\n      </div>\n      <div class="flex-1 min-w-0">\n        <div class="flex items-center gap-2 flex-wrap">\n          <h2 class="text-sm font-bold text-gray-900">{{ conn.title or conn.name }}</h2>\n          <span class="apg-badge">{{ conn.category }}</span>\n        </div>\n        <p class="text-xs text-gray-500 mt-1">{{ conn.description }}</p>\n      </div>\n    </div>\n    <dl class="grid grid-cols-2 gap-3 mb-4">\n      <div class="border border-gray-100 rounded-lg p-3">\n        <dt class="text-xs font-semibold uppercase tracking-wide text-gray-400">Operations</dt>\n        <dd class="text-lg font-bold text-gray-900">{{ conn.operation_count }}</dd>\n      </div>\n      <div class="border border-gray-100 rounded-lg p-3">\n        <dt class="text-xs font-semibold uppercase tracking-wide text-gray-400">Status</dt>\n        <dd class="text-sm font-semibold text-gray-900">{{ conn.status }}</dd>\n      </div>\n    </dl>\n    {% if conn.operations %}\n    <div class="flex items-center gap-2 flex-wrap mb-4">\n      {% for operation in conn.operations[:3] %}\n      <span class="text-xs text-gray-500 bg-gray-50 border border-gray-100 rounded-lg px-2 py-1">{{ operation }}</span>\n      {% endfor %}\n    </div>\n    {% endif %}\n    <div class="flex items-center gap-2">\n      <span class="flex-1 text-xs text-gray-400 font-mono truncate">{{ conn.file or conn.name }}</span>\n      <a href="{{ conn.href }}" class="apg-btn apg-btn-secondary">Open</a>\n    </div>\n  </article>\n  {% endfor %}\n</div>\n{% else %}\n<div class="apg-card text-center py-12">\n  <div class="text-5xl mb-4 opacity-20">Plug</div>\n  <h2 class="text-base font-semibold text-gray-900 mb-2">No connectors match these filters</h2>\n  <p class="text-sm text-gray-500 mb-4">Clear the search or category filter to return to generated integration blueprints.</p>\n  <a href="/ui/marketplace" class="apg-btn">Clear filters</a>\n</div>\n{% endif %}\n', 'workflow_list.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Workflows</span></p>\n  <div class="flex items-center justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">Workflows</h1>\n      <p class="text-sm text-gray-500 mt-1">{{ total }} guided workflows across {{ entity_count }} entities · {{ run_count }} recorded runs</p>\n    </div>\n    <a href="/ui/debug" class="apg-btn apg-btn-secondary">Run history</a>\n  </div>\n</section>\n\n{% if workflows %}\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4">\n  <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 lg:col-span-2">\n  {% for workflow in workflows %}\n  <article class="apg-card group">\n    <div class="flex items-start gap-3 mb-3">\n      <span class="text-2xl" aria-hidden="true">{{ workflow.icon }}</span>\n      <div class="min-w-0">\n        <h2 class="font-semibold text-gray-900 group-hover:text-blue-600 text-sm mb-1">{{ workflow.name }}</h2>\n        <p class="text-xs text-gray-400">{{ workflow.entity }} · {{ workflow.step_count }} steps · {{ workflow.run_count }} runs</p>\n      </div>\n    </div>\n    <p class="text-xs text-gray-500 leading-relaxed mb-4">{{ workflow.description }}</p>\n    <div class="flex items-center gap-1 mb-4" aria-hidden="true">\n      {% for step in workflow.steps %}\n      <div class="h-1.5 flex-1 rounded-full {{ \'bg-blue-500\' if loop.first else \'bg-gray-100\' }}"></div>\n      {% endfor %}\n    </div>\n    <a href="{{ workflow.href }}" class="apg-btn">Start</a>\n  </article>\n  {% endfor %}\n  </div>\n\n  <aside class="apg-card">\n    <header class="apg-card-header">\n      <h2 class="text-base font-semibold text-gray-900">Recent runs</h2>\n      <a href="/ui/debug" class="text-xs text-apg-primary hover:underline">Open debugger</a>\n    </header>\n    {% if recent_runs %}\n    <ol class="space-y-3">\n      {% for run in recent_runs %}\n      <li class="border border-gray-100 rounded-lg p-3">\n        <div class="flex items-center justify-between gap-3">\n          <a href="{{ run.href }}" class="font-mono text-xs text-gray-900 hover:underline">{{ run.id }}</a>\n          <span class="apg-badge apg-badge-success">{{ run.status }}</span>\n        </div>\n        <p class="text-xs text-gray-500 mt-1">{{ run.workflow }}{% if run.entity %} · {{ run.entity }}{% endif %} · {{ run.step_count }} steps</p>\n      </li>\n      {% endfor %}\n    </ol>\n    {% else %}\n    <p class="text-sm text-gray-500">Completed wizard runs will appear here and in the flow debugger.</p>\n    {% endif %}\n  </aside>\n</section>\n{% else %}\n<section class="apg-card text-center py-10">\n  <h2 class="text-base font-semibold text-gray-900 mb-2">No workflows available</h2>\n  <p class="text-sm text-gray-500">Declare entities with fields to generate guided workflows.</p>\n</section>\n{% endif %}\n', 'debug_console.html.j2': '<section class="mb-6">\n  <p class="text-sm text-gray-500 mb-2"><a href="/ui" class="hover:text-blue-600">Application</a> / <span class="font-semibold text-gray-900">Debug</span></p>\n  <div class="flex items-start justify-between gap-4 flex-wrap">\n    <div>\n      <h1 class="text-xl font-bold text-gray-900">Flow debugger</h1>\n      <p class="text-sm text-gray-500 mt-1">Workflow run history, timeline, journal events, circuit breakers, and event subscriptions.</p>\n    </div>\n    {% if selected_run %}\n    <span class="apg-badge {{ selected_run.badge_class }}">{{ selected_run.status }}</span>\n    {% endif %}\n  </div>\n</section>\n\n{% if selected_run %}\n<section class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">\n  <article class="apg-card lg:col-span-2">\n    <header class="apg-card-header">\n      <div>\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">Run detail</p>\n        <h2 class="text-base font-semibold text-gray-900">{{ selected_run.id }}</h2>\n      </div>\n      <span class="apg-badge apg-badge-neutral">{{ selected_run.workflow }}</span>\n    </header>\n\n    <section class="grid grid-cols-1 lg:grid-cols-3 gap-3 mb-4" aria-label="Run summary">\n      <div class="border border-gray-100 rounded-lg p-3">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">Steps</p>\n        <p class="text-xl font-bold text-gray-900">{{ selected_run.step_count }}</p>\n      </div>\n      <div class="border border-gray-100 rounded-lg p-3">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">Journal events</p>\n        <p class="text-xl font-bold text-gray-900">{{ selected_run.event_count }}</p>\n      </div>\n      <div class="border border-gray-100 rounded-lg p-3">\n        <p class="text-xs font-semibold uppercase tracking-wide text-gray-400">Created record</p>\n        <p class="text-xl font-bold text-gray-900">{{ selected_run.created_record_id or \'-\' }}</p>\n      </div>\n    </section>\n\n    {% if selected_run.trace %}\n    <h3 class="text-sm font-semibold text-gray-900 mb-3">Run timeline</h3>\n    <ol class="space-y-3">\n      {% for step in selected_run.trace %}\n      <li class="border border-gray-200 rounded-lg p-3">\n        <div class="flex items-center justify-between gap-3 flex-wrap">\n          <div>\n            <p class="text-sm font-medium text-gray-900">{{ step.step }}</p>\n            <p class="text-xs text-gray-500 mt-1">Step {{ step.index }}{% if step.field_count %} · {{ step.field_count }} field{{ \'\' if step.field_count == 1 else \'s\' }}{% endif %}{% if step.duration_ms %} · {{ step.duration_ms }} ms{% endif %}</p>\n          </div>\n          <span class="apg-badge {{ step.badge_class }}">{{ step.status }}</span>\n        </div>\n        {% if step.notes %}<p class="text-xs text-gray-500 mt-2">{{ step.notes }}</p>{% endif %}\n        {% if step.fields %}<p class="text-xs text-gray-400 mt-2">Fields: <span class="font-mono">{{ step.fields }}</span></p>{% endif %}\n      </li>\n      {% endfor %}\n    </ol>\n    {% else %}\n    <p class="text-sm text-gray-500">No steps recorded for this run.</p>\n    {% endif %}\n  </article>\n\n  <aside class="space-y-4">\n    <article class="apg-card">\n      <h2 class="text-base font-semibold text-gray-900 mb-3">Run context</h2>\n      <dl class="space-y-2">\n        <div class="flex items-center justify-between gap-3"><dt class="text-sm text-gray-500">Entity</dt><dd class="text-sm font-mono text-gray-900">{{ selected_run.entity }}</dd></div>\n        <div class="flex items-center justify-between gap-3"><dt class="text-sm text-gray-500">Workflow ID</dt><dd class="text-sm font-mono text-gray-900">{{ selected_run.workflow_id }}</dd></div>\n        <div class="flex items-center justify-between gap-3"><dt class="text-sm text-gray-500">Event ID</dt><dd class="text-sm font-mono text-gray-900">{{ selected_run.event_id or \'-\' }}</dd></div>\n        <div class="flex items-center justify-between gap-3"><dt class="text-sm text-gray-500">Duration</dt><dd class="text-sm font-mono text-gray-900">{{ selected_run.duration_ms }} ms</dd></div>\n      </dl>\n    </article>\n\n    <article class="apg-card">\n      <h2 class="text-base font-semibold text-gray-900 mb-3">Snapshots</h2>\n      <details class="mb-3">\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Payload snapshot</summary>\n        <pre>{{ selected_run.payload_json }}</pre>\n      </details>\n      <details>\n        <summary class="cursor-pointer text-sm font-medium text-gray-700">Created record snapshot</summary>\n        <pre>{{ selected_run.record_json }}</pre>\n      </details>\n    </article>\n  </aside>\n</section>\n\n<section class="apg-card">\n  <header class="apg-card-header">\n    <h2 class="text-base font-semibold text-gray-900">Event journal</h2>\n    <a href="/workflows/runs/{{ selected_run.id }}/journal" class="apg-btn apg-btn-secondary">Journal JSON</a>\n  </header>\n  {% if selected_run.journal %}\n  <div class="apg-table-wrap">\n    <table class="apg-table">\n      <thead><tr><th>Seq</th><th>Event</th><th>Step</th><th>Time</th><th>Data</th></tr></thead>\n      <tbody>\n        {% for event in selected_run.journal %}\n        <tr>\n          <td class="font-mono">{{ event.seq }}</td>\n          <td><span class="apg-badge apg-badge-neutral">{{ event.event_type }}</span></td>\n          <td>{{ event.step }}</td>\n          <td class="font-mono text-xs">{{ event.ts }}</td>\n          <td><details><summary class="cursor-pointer text-sm font-medium text-gray-700">Data</summary><pre>{{ event.data_json }}</pre></details></td>\n        </tr>\n        {% endfor %}\n      </tbody>\n    </table>\n  </div>\n  {% else %}\n  <p class="text-sm text-gray-500">No journal events recorded for this run.</p>\n  {% endif %}\n</section>\n{% endif %}\n\n<section class="grid grid-cols-1 lg:grid-cols-2 gap-4">\n  <article class="apg-card">\n    <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Recent runs</h2></header>\n    {% if runs %}\n    <div class="apg-table-wrap">\n      <table class="apg-table">\n        <thead><tr><th>Run</th><th>Workflow</th><th>Entity</th><th>Status</th><th>Steps</th></tr></thead>\n        <tbody>\n          {% for run in runs %}\n          <tr>\n            <td><a href="/ui/debug/{{ run.id }}" class="font-mono hover:underline">{{ run.id }}</a></td>\n            <td>{{ run.workflow }}</td>\n            <td>{{ run.entity }}</td>\n            <td><span class="apg-badge {{ run.badge_class }}">{{ run.status }}</span></td>\n            <td>{{ run.step_count }}</td>\n          </tr>\n          {% endfor %}\n        </tbody>\n      </table>\n    </div>\n    {% else %}\n    <p class="text-sm text-gray-500">No workflow runs yet. Complete a workflow to inspect its timeline here.</p>\n    {% endif %}\n  </article>\n\n  <article class="apg-card">\n    <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Circuit breakers</h2></header>\n    {% if circuit_breakers %}\n    <dl class="space-y-3">\n      {% for item in circuit_breakers %}\n      <div class="flex items-center justify-between gap-3 border-b border-gray-100 pb-3">\n        <dt class="font-mono text-xs text-gray-600">{{ item.key }}</dt>\n        <dd><span class="apg-badge {{ item.badge_class }}">{{ item.state }}</span> <span class="text-xs text-gray-400">{{ item.failures }} failures</span></dd>\n      </div>\n      {% endfor %}\n    </dl>\n    {% else %}\n    <p class="text-sm text-gray-500">No circuit breakers tripped.</p>\n    {% endif %}\n  </article>\n</section>\n\n<section class="apg-card">\n  <header class="apg-card-header"><h2 class="text-base font-semibold text-gray-900">Event subscriptions</h2></header>\n  {% if subscriptions %}\n  <ul class="space-y-2">\n    {% for item in subscriptions %}\n    <li class="text-sm text-gray-600"><span class="font-mono">{{ item.event }}</span> -> {{ item.workflows }}</li>\n    {% endfor %}\n  </ul>\n  {% else %}\n  <p class="text-sm text-gray-500">No event subscriptions declared.</p>\n  {% endif %}\n</section>\n', 'app_index.html.j2': '<!--- app_index.html.j2 - APG application home page --->\n{# Variables: module_name, module_description, entities, capabilities, databases,\n              application_routes, ui_routes, agents, agent_teams #}\n<div class="mb-6">\n  <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">{{ module_name }}</h1>\n  <p class="text-gray-500 mt-1">{{ module_description or \'Generated APG application\' }}</p>\n</div>\n\n{# Quick nav #}\n<nav class="flex flex-wrap gap-2 mb-8 text-sm" aria-label="Workspace shortcuts">\n  {% if entities %}\n  <a href="/ui/entities/{{ entities[0].name | urlencode }}"\n     class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-apg-primary text-white hover:opacity-90 transition-opacity font-medium">\n    Start with {{ entities[0].name }}\n  </a>\n  {% endif %}\n  {% for link in api_links %}\n  <a href="{{ link.url }}"\n     class="inline-flex items-center px-3 py-1.5 rounded border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-apg-primary transition-colors">\n    {{ link.label }}\n  </a>\n  {% endfor %}\n</nav>\n\n<section class="apg-dashboard-command-center mb-8" aria-label="Dashboard command center">\n  <article class="apg-card apg-dashboard-panel">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Composable Tiles</h2>\n        <p class="text-xs text-gray-400 mt-1">Persisted workspace order and visibility</p>\n      </div>\n      <button type="button" class="apg-btn apg-btn-secondary" data-apg-tour="dashboard-tiles">Save view</button>\n    </div>\n    <ol class="apg-dashboard-sort-list">\n      {% for tile in tile_controls %}\n      <li>\n        <span class="apg-drag-handle" aria-hidden="true">::</span>\n        <a href="{{ tile.href }}">{{ tile.label }}</a>\n        <span>{{ tile.position }}</span>\n      </li>\n      {% else %}\n      <li><span>No record tiles yet</span></li>\n      {% endfor %}\n    </ol>\n  </article>\n\n  <article class="apg-card apg-dashboard-panel">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Threshold Alerts</h2>\n        <p class="text-xs text-gray-400 mt-1">Inline watches before scheduled delivery</p>\n      </div>\n      <a class="apg-btn apg-btn-secondary" href="/metrics">Metrics</a>\n    </div>\n    <ul class="apg-dashboard-watch-list">\n      {% for alert in dashboard_alerts %}\n      <li>\n        <a href="{{ alert.href }}">{{ alert.label }}</a>\n        <strong>{{ alert.value }}</strong>\n        <small>{{ alert.state }} at {{ alert.threshold }}</small>\n      </li>\n      {% else %}\n      <li><span>No thresholds available</span></li>\n      {% endfor %}\n    </ul>\n  </article>\n\n  <article class="apg-card apg-dashboard-panel">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Annotation Pins</h2>\n        <p class="text-xs text-gray-400 mt-1">Decision context travels with charts</p>\n      </div>\n      <span class="apg-badge apg-badge-info">{{ dashboard_annotations | length }} pins</span>\n    </div>\n    <ul class="apg-dashboard-pin-list">\n      {% for pin in dashboard_annotations %}\n      <li>\n        <a href="{{ pin.href }}">{{ pin.title }}</a>\n        <p>{{ pin.body }}</p>\n      </li>\n      {% else %}\n      <li><p>Add status data to unlock annotation pins.</p></li>\n      {% endfor %}\n    </ul>\n  </article>\n\n  <article class="apg-card apg-dashboard-panel">\n    <div class="apg-card-header">\n      <div>\n        <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Scheduled Export</h2>\n        <p class="text-xs text-gray-400 mt-1">Offline packets, no external service</p>\n      </div>\n      <button type="button" class="apg-btn apg-btn-secondary" onclick="window.print()">PNG/PDF</button>\n    </div>\n    <ul class="apg-dashboard-export-list">\n      {% for item in scheduled_exports %}\n      <li>\n        <strong>{{ item.label }}</strong>\n        <span>{{ item.cadence }}</span>\n        <small>{{ item.format }}</small>\n      </li>\n      {% endfor %}\n    </ul>\n  </article>\n</section>\n\n{# Stats row #}\n<div class="apg-grid-4 mb-8" data-apg-live="events">\n  {% for stat in dashboard_stats[:4] %}\n  <article class="apg-card">\n    <div class="apg-stat">\n      <a class="apg-stat-value hover:text-apg-primary" href="/ui/entities/{{ stat.label | urlencode }}">{{ stat.value }}</a>\n      <span class="apg-stat-label">{{ stat.label }} records</span>\n      <span class="apg-stat-delta">{{ stat.delta }}</span>\n    </div>\n    <div class="apg-chart mt-3" data-apg-chart="{{ stat.chart_id }}"></div>\n    <script id="{{ stat.chart_id }}" type="application/json">{{ stat.spec_json | safe }}</script>\n  </article>\n  {% endfor %}\n  <article class="apg-card">\n    <div class="apg-stat">\n      <a class="apg-stat-value hover:text-apg-primary" href="/ui/marketplace">{{ capabilities | length }}</a>\n      <span class="apg-stat-label">Capabilities</span>\n    </div>\n  </article>\n  <article class="apg-card">\n    <div class="apg-stat">\n      <a class="apg-stat-value hover:text-apg-primary" href="/ui/workflows">{{ workflow_summary.workflow_count }}</a>\n      <span class="apg-stat-label">Workflows</span>\n    </div>\n  </article>\n</div>\n\n{% if status_charts %}\n<section class="apg-grid-2 gap-6 mb-8">\n  {% for chart in status_charts %}\n  <article class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">{{ chart.entity }} by {{ chart.field }}</h2>\n      <a class="text-xs hover:underline" href="/ui/entities/{{ chart.entity | urlencode }}?view=analytics">Analytics</a>\n    </div>\n    <div class="apg-chart" data-apg-chart="{{ chart.chart_id }}"></div>\n    <script id="{{ chart.chart_id }}" type="application/json">{{ chart.spec_json | safe }}</script>\n  </article>\n  {% endfor %}\n</section>\n{% endif %}\n\n<section class="apg-grid-3 gap-6 mb-8">\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Recent Activity</h2>\n    {% if recent_activity %}\n    <ul class="space-y-2">\n      {% for event in recent_activity %}\n      <li class="text-xs text-gray-500">{{ event.get(\'type\', \'event\') }} · {{ event.get(\'entity\', \'\') }}</li>\n      {% endfor %}\n    </ul>\n    {% else %}\n    <div class="apg-chart-empty">\n      <p>No activity yet</p>\n      {% if entities %}<a class="apg-btn apg-btn-secondary mt-3" href="/ui/entities/{{ entities[0].name | urlencode }}">Create the first record</a>{% endif %}\n    </div>\n    {% endif %}\n  </article>\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Workflow Summary</h2>\n    <p class="text-sm text-gray-500">{{ workflow_summary.workflow_count }} workflow(s), {{ workflow_summary.run_count }} run(s)</p>\n    <a class="apg-btn apg-btn-secondary mt-3" href="/ui/workflows">Open workflows</a>\n  </article>\n  <article class="apg-card">\n    <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Agent Summary</h2>\n    <p class="text-sm text-gray-500">{{ agent_summary.agent_count }} agent(s), {{ agent_summary.team_count }} team(s)</p>\n    {% if agents %}<a class="apg-btn apg-btn-secondary mt-3" href="/ui/agents/{{ agents[0].name | urlencode }}">Open agent console</a>{% endif %}\n  </article>\n</section>\n\n<div class="apg-grid-2 gap-6">\n  {% if entities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Entities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for entity in entities %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="/ui/entities/{{ entity.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ entity.name }}\n        </a>\n        <span class="apg-badge apg-badge-neutral">{{ entity.type }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if capabilities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Capabilities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for cap in capabilities %}\n      <li class="py-2">\n        <a href="/ui/capabilities/{{ cap.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ cap.name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if ui_routes %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Application Screens</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for route, screen in ui_routes.items() %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="{{ route }}" class="text-sm text-apg-primary hover:underline">{{ route }}</a>\n        <span class="text-xs text-gray-400">{{ screen.get(\'application\', \'\') }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if application_routes %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Application Routes</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for route, screen in application_routes.items() %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="{{ route }}" class="text-sm text-apg-primary hover:underline">{{ route }}</a>\n        <span class="text-xs text-gray-400">{{ screen.get(\'application\', \'\') }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if agents %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">AI Agents</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for agent in agents %}\n      <li class="py-2">\n        <a href="/ui/agents/{{ agent.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ agent.name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if agent_teams %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">AI Agent Teams</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for team in agent_teams %}\n      <li class="py-2">\n        <a href="/ui/agent-teams/{{ team.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ team.name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n</div>\n', 'widgets/breadcrumbs.html.j2': '<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap" aria-label="Breadcrumb">\n  {% for item in breadcrumbs %}\n    {% if item.href and not loop.last %}\n    <a href="{{ item.href }}" class="hover:text-apg-primary transition-colors">{{ item.label }}</a>\n    {% else %}\n    <span class="font-semibold text-gray-900" {% if loop.last %}aria-current="page"{% endif %}>{{ item.label }}</span>\n    {% endif %}\n    {% if not loop.last %}<span aria-hidden="true">/</span>{% endif %}\n  {% endfor %}\n</nav>\n', 'widgets/field_display.html.j2': '{# field_display.html.j2 — semantic field rendering for record detail\n   Included by record_detail.html.j2 for individual field value rendering.\n   Variables: field (dict), field_val (any), semantic (str)\n#}\n{% if semantic == \'email\' and field_val %}\n  <a href="mailto:{{ field_val }}" class="text-apg-primary hover:underline text-sm">{{ field_val }}</a>\n{% elif semantic == \'phone\' and field_val %}\n  <a href="tel:{{ field_val }}" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1">\n    <svg class="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z"/></svg>\n    {{ field_val }}\n  </a>\n{% elif semantic == \'url\' and field_val %}\n  <a href="{{ field_val }}" target="_blank" rel="noopener" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1 truncate max-w-xs">\n    <svg class="w-3 h-3 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/><path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/></svg>\n    {{ field_val | string | truncate(40) }}\n  </a>\n{% elif semantic == \'image_url\' and field_val %}\n  <img src="{{ field_val }}" alt="{{ field.name }}" class="w-12 h-12 rounded-lg object-cover border border-gray-100">\n{% elif semantic == \'currency\' and field_val %}\n  <span class="text-sm font-semibold text-gray-900 tabular-nums">{{ format_currency(field_val) }}</span>\n{% elif semantic == \'percent\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="flex-1 bg-gray-100 rounded-full h-1.5 max-w-24">\n      <div class="bg-apg-primary h-1.5 rounded-full" style="width: {{ [field_val | float, 100] | min }}%"></div>\n    </div>\n    <span class="text-sm text-gray-700 tabular-nums">{{ field_val }}%</span>\n  </div>\n{% elif semantic == \'rating\' and field_val %}\n  <div class="flex items-center gap-0.5">\n    {% set stars = field_val | float | round | int %}\n    {% for i in range(5) %}\n    <svg class="w-4 h-4 {{ \'text-amber-400\' if i < stars else \'text-gray-200\' }}" viewBox="0 0 20 20" fill="currentColor">\n      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>\n    </svg>\n    {% endfor %}\n    <span class="text-xs text-gray-400 ml-1">{{ field_val }}/5</span>\n  </div>\n{% elif semantic == \'color\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="w-5 h-5 rounded-full border border-gray-200 flex-shrink-0" style="background-color: {{ field_val }}"></div>\n    <span class="text-sm text-gray-700 font-mono">{{ field_val }}</span>\n  </div>\n{% elif semantic == \'json\' and field_val %}\n  <details class="max-w-xs">\n    <summary class="text-xs text-apg-primary cursor-pointer hover:underline">View JSON</summary>\n    <pre class="mt-1 text-xs bg-gray-50 rounded-lg p-2 overflow-auto max-h-40 border border-gray-100">{{ field_val | string }}</pre>\n  </details>\n{% elif semantic == \'status\' and field_val %}\n  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold\n    {% if field_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-100 text-green-800\n    {% elif field_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n    {% elif field_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n    {% else %}bg-gray-100 text-gray-600{% endif %}">\n    {{ field_val }}\n  </span>\n{% elif semantic == \'boolean\' %}\n  {% if field_val | string | lower in [\'true\', \'1\', \'yes\'] %}\n  <span class="inline-flex items-center gap-1 text-green-600 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>Yes</span>\n  {% else %}\n  <span class="inline-flex items-center gap-1 text-gray-400 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>No</span>\n  {% endif %}\n{% else %}\n  {% if field_val is none or field_val == \'\' or field_val | string == \'None\' %}\n  <span class="text-gray-300 italic text-xs">—</span>\n  {% else %}\n  {{ field_val | string | truncate(200) }}\n  {% endif %}\n{% endif %}\n'}
APG_AUTH_REQUIRED = False
APG_SUPPORTED_LANGUAGES: list[str] = ['en', 'sw', 'fr']
APG_DEFAULT_LANGUAGE = 'en'
APG_FALLBACK_LANGUAGE = 'en'
APG_I18N: Dict[str, Dict[str, str]] = {'en': {'home': 'Home', 'workflows': 'Workflows', 'marketplace': 'Marketplace', 'theme_system': 'System', 'language': 'Language', 'logout': 'Logout', 'sign_in': 'Sign in', 'open_app': 'Open App', 'api_docs': 'API Docs', 'data_entities': 'Data Entities', 'view_manifest': 'View Manifest', 'entities': 'Entities', 'capabilities': 'Capabilities', 'records': 'Records', 'ai_agents': 'AI Agents'}, 'sw': {'home': 'Nyumbani', 'workflows': 'Mitiririko', 'marketplace': 'Soko', 'theme_system': 'Mfumo', 'language': 'Lugha', 'logout': 'Toka', 'sign_in': 'Ingia', 'open_app': 'Fungua Programu', 'api_docs': 'Nyaraka za API', 'data_entities': 'Vyombo vya Data', 'view_manifest': 'Tazama Manifesti', 'entities': 'Vyombo', 'capabilities': 'Uwezo', 'records': 'Rekodi', 'ai_agents': 'Mawakala wa AI'}, 'fr': {'home': 'Home', 'workflows': 'Workflows', 'marketplace': 'Marketplace', 'theme_system': 'System', 'language': 'Language', 'logout': 'Logout', 'sign_in': 'Sign in', 'open_app': 'Open App', 'api_docs': 'API Docs', 'data_entities': 'Data Entities', 'view_manifest': 'View Manifest', 'entities': 'Entities', 'capabilities': 'Capabilities', 'records': 'Records', 'ai_agents': 'AI Agents'}}


def _live_topic_list(raw_topics: str | None = None) -> list[str]:
    topics = [
        topic.strip()
        for topic in str(raw_topics or "").split(",")
        if topic.strip()
    ]
    return topics or ["*"]


def _subscribe_live_events(topics: list[str]) -> tuple[Any, Any]:
    subscriber = {"topics": set(topics), "queue": _queue.Queue(maxsize=100)}
    with APG_LIVE_LOCK:
        APG_LIVE_SUBSCRIBERS.append(subscriber)

    def unsubscribe() -> None:
        with APG_LIVE_LOCK:
            if subscriber in APG_LIVE_SUBSCRIBERS:
                APG_LIVE_SUBSCRIBERS.remove(subscriber)

    return subscriber["queue"], unsubscribe


def _publish_live_event(topic: str, event_type: str, data: Dict[str, Any]) -> None:
    message = {
        "topic": topic,
        "event": event_type,
        "data": data,
        "ts": _time.time(),
    }
    with APG_LIVE_LOCK:
        subscribers = list(APG_LIVE_SUBSCRIBERS)
    for subscriber in subscribers:
        topics = subscriber.get("topics", set())
        if "*" not in topics and topic not in topics:
            continue
        try:
            subscriber["queue"].put_nowait(message)
        except _queue.Full:
            continue


def _sse_format(message: Dict[str, Any]) -> str:
    event_name = str(message.get("event", "message"))
    payload = json.dumps(message, sort_keys=True)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _sse_stream(raw_topics: str | None = None):
    topics = _live_topic_list(raw_topics)
    queue, unsubscribe = _subscribe_live_events(topics)
    try:
        yield ": connected\n\n"
        yield _sse_format({"topic": "system", "event": "apg-ready", "data": {"topics": topics}})
        while True:
            try:
                message = queue.get(timeout=15)
            except _queue.Empty:
                yield ": heartbeat\n\n"
                continue
            yield _sse_format(message)
    finally:
        unsubscribe()


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
    _publish_live_event(
        f"workflow:run:{run_id}",
        "workflow",
        {"run_id": run_id, "event_type": event_type, "step": step, "data": data},
    )
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
        _ = None  # best-effort
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


def _database_schema_name(database: Dict[str, Any]) -> str:
    config = database.get("connection_config", {})
    if isinstance(config, dict) and config.get("schema"):
        return str(config.get("schema"))
    for field in database.get("fields", []):
        if isinstance(field, dict) and field.get("name") == "schema" and field.get("default"):
            return str(field.get("default"))
    return "public"


def _database_column_specs(entity: Dict[str, Any]) -> list[Dict[str, Any]]:
    entity_name = str(entity.get("name", ""))
    columns: list[Dict[str, Any]] = [
        {"name": "id", "type": "integer", "required": True, "nullable": False, "primary_key": True}
    ]
    for field in entity.get("fields", []):
        if not isinstance(field, dict):
            continue
        field_name = str(field.get("name", "")).strip()
        if not field_name or field_name == "id":
            continue
        required = bool(field.get("required", False))
        column: Dict[str, Any] = {
            "name": field_name,
            "type": str(field.get("type", "any")),
            "required": required,
            "nullable": not required,
            "primary_key": False,
        }
        relationship = _field_relationship(entity_name, field_name)
        if relationship:
            column["reference"] = {
                "table": relationship.get("target_table", ""),
                "column": relationship.get("target_field", "id"),
                "cardinality": relationship.get("cardinality", "many-to-one"),
            }
        columns.append(column)
    return columns


def _database_table_specs() -> list[Dict[str, Any]]:
    tables: list[Dict[str, Any]] = []
    for entity in ENTITIES:
        if str(entity.get("type", "")) not in {"entity", "table"}:
            continue
        table_name = str(entity.get("name", "")).strip()
        if not table_name:
            continue
        columns = _database_column_specs(entity)
        indexes = [
            {"name": f"idx_{table_name}_{column['name']}", "columns": [column["name"]]}
            for column in columns
            if column.get("name") not in {"id"} and (column.get("required") or column.get("reference"))
        ][:3]
        tables.append({
            "name": table_name,
            "columns": columns,
            "indexes": indexes,
            "source": "generated_entity",
        })
    return tables


def _with_inferred_database_schemas(database: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(database)
    schemas = list(enriched.get("schemas", [])) if isinstance(enriched.get("schemas", []), list) else []
    if schemas:
        enriched["schemas"] = schemas
        return enriched
    tables = _database_table_specs()
    if tables:
        enriched["schemas"] = [
            {"name": _database_schema_name(enriched), "tables": tables, "source": "generated_entities"}
        ]
    else:
        enriched["schemas"] = []
    return enriched


def list_databases() -> list[Dict[str, Any]]:
    return [
        _with_inferred_database_schemas(entity)
        for entity in ENTITIES
        if entity.get("type") == "database"
    ]


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
                _ = None  # best-effort


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
            _ = None  # best-effort
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
            _ = None  # best-effort
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
    if APG_AUTH_REQUIRED:
        return {
            "mode": "session",
            "login": "/login",
            "logout": "/logout",
        }
    return {
        "mode": "api_key" if os.environ.get("APG_API_KEY") else "open",
        "header": "Authorization: Bearer <key> or X-APG-API-Key" if os.environ.get("APG_API_KEY") else None,
    }


def _active_locale() -> str:
    try:
        cookie_locale = _flask_request.cookies.get("apg_lang")
        if cookie_locale in APG_SUPPORTED_LANGUAGES:
            return str(cookie_locale)
        accepted = _flask_request.accept_languages.best_match(APG_SUPPORTED_LANGUAGES)
        if accepted:
            return str(accepted)
    except RuntimeError:
        return APG_DEFAULT_LANGUAGE
    return APG_DEFAULT_LANGUAGE


def _text_direction(locale: str | None = None) -> str:
    language = (locale or _active_locale()).split("-")[0].lower()
    return "rtl" if language in {"ar", "he", "fa", "ur"} else "ltr"


def _(key: str) -> str:
    locale = _active_locale()
    return (
        APG_I18N.get(locale, {}).get(key)
        or APG_I18N.get(APG_FALLBACK_LANGUAGE, {}).get(key)
        or APG_I18N.get("en", {}).get(key)
        or key
    )


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if _active_locale().split("-")[0] in {"fr", "pt"}:
        return f"{number:,.2f}".replace(",", " ").replace(".", ",")
    return f"{number:,.2f}"


def format_currency(value: Any, currency: str = "USD") -> str:
    symbols = {"USD": "$", "KES": "KSh", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(str(currency).upper(), str(currency).upper() + " ")
    return symbol + format_number(value)


def format_date(value: Any) -> str:
    text = str(value)
    if _active_locale().split("-")[0] in {"fr", "pt", "sw"} and len(text) >= 10 and text[4:5] == "-":
        return f"{text[8:10]}/{text[5:7]}/{text[0:4]}"
    return text


def _auth_credentials() -> Dict[str, Dict[str, Any]]:
    raw = os.environ.get("APG_AUTH_USERS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            result: Dict[str, Dict[str, Any]] = {}
            for username, spec in parsed.items():
                if isinstance(spec, dict):
                    result[str(username)] = {
                        "password": str(spec.get("password", "")),
                        "name": str(spec.get("name", username)),
                        "roles": list(spec.get("roles", ["user"])) if isinstance(spec.get("roles", []), list) else ["user"],
                        "permissions": list(spec.get("permissions", [])) if isinstance(spec.get("permissions", []), list) else [],
                    }
                else:
                    result[str(username)] = {"password": str(spec), "name": str(username), "roles": ["user"], "permissions": []}
            if result:
                return result
    username = os.environ.get("APG_AUTH_USERNAME", "admin")
    password = os.environ.get("APG_AUTH_PASSWORD", "admin")
    return {
        username: {
            "password": password,
            "name": os.environ.get("APG_AUTH_DISPLAY_NAME", username),
            "roles": ["admin"],
            "permissions": ["*"],
        }
    }


def _authenticate_user(username: str, password: str) -> Dict[str, Any] | None:
    user = _auth_credentials().get(username)
    if not user:
        return None
    if not hmac.compare_digest(str(user.get("password", "")), str(password)):
        return None
    return {
        "username": username,
        "name": str(user.get("name", username)),
        "roles": list(user.get("roles", [])),
        "permissions": list(user.get("permissions", [])),
    }


def _issue_login_session(user: Dict[str, Any]) -> Dict[str, Any]:
    _flask_session["apg_user"] = user
    token = ""
    jwt_secret = os.environ.get("APG_JWT_SECRET")
    if jwt_secret and _jwt_lib is not None:
        try:
            token = _jwt_lib.encode({"sub": user["username"], "name": user["name"], "roles": user.get("roles", [])}, jwt_secret, algorithm="HS256")
        except Exception:
            token = ""
    return {"user": user, "token": token}


def _current_user() -> Dict[str, Any] | None:
    user = _flask_session.get("apg_user")
    return dict(user) if isinstance(user, dict) else None


def _login_required_for_path(path: str) -> bool:
    if not APG_AUTH_REQUIRED:
        return False
    return path == "/ui" or path.startswith("/ui/")


def _login_page(error: str = "", next_url: str = "/ui", username: str = "") -> str:
    body = _render_template(
        "login.html.j2",
        module_name=MODULE_NAME,
        error=error,
        next_url=next_url or "/ui",
        username=username,
    )
    if body is None:
        safe_error = html.escape(error)
        safe_next = html.escape(next_url or "/ui", quote=True)
        safe_username = html.escape(username, quote=True)
        error_html = f'<p role="alert">{safe_error}</p>' if safe_error else ''
        body = (
            '<main id="content" class="apg-login-page">'
            '<section class="apg-login-card">'
            f'<h1>{html.escape(MODULE_NAME)}</h1>'
            f'{error_html}'
            f'<form method="post" action="/login"><input type="hidden" name="next" value="{safe_next}">'
            f'<label>Username <input name="username" autocomplete="username" value="{safe_username}"></label>'
            '<label>Password <input name="password" type="password" autocomplete="current-password"></label>'
            '<button class="apg-btn" type="submit">Sign in</button></form>'
            '</section></main>'
        )
    return _html_page("Sign in", body, shell=False)


def _forbidden_page(message: str = "You do not have permission to view this page.") -> str:
    return _html_page(
        "Access denied",
        '<section class="apg-card"><h1>Access denied</h1><p>' + html.escape(message) + '</p></section>',
    )


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
    _publish_live_event("events", "record", event)
    _publish_live_event(f"entity:{entity_name}", "record", event)
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


def _split_agent_literal_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'").strip('"') for item in text.split(",") if item.strip()]


def _entity_agent_team_descriptions() -> Dict[str, Dict[str, Any]]:
    descriptions: Dict[str, Dict[str, Any]] = {}
    for entity in ENTITIES:
        if str(entity.get("type", "")) != "agent_team":
            continue
        fields = {
            str(field.get("name", "")): field
            for field in entity.get("fields", [])
            if isinstance(field, dict)
        }
        agents = _split_agent_literal_list(fields.get("agents", {}).get("type", ""))
        capabilities = _split_agent_literal_list(fields.get("capabilities", {}).get("type", ""))
        flow_text = str(fields.get("flow", {}).get("type", ""))
        flow = []
        for edge in [part.strip() for part in flow_text.split(",") if part.strip()]:
            if "->" in edge:
                source, target = [piece.strip() for piece in edge.split("->", 1)]
                flow.append({"source": source, "target": target, "condition": ""})
        descriptions[str(entity.get("name", ""))] = {
            "name": str(entity.get("name", "")),
            "agents": agents,
            "capabilities": capabilities,
            "flow": flow,
            "policy": {},
            "configuration": {},
            "rules": [],
            "ui": {},
            "theme": {},
            "source": "entity_metadata",
        }
    return descriptions


def _semantic_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    raw_agents = SEMANTIC_MODEL.get("agents", {})
    if not isinstance(raw_agents, dict):
        return {}
    descriptions: Dict[str, Dict[str, Any]] = {}
    for name, spec in raw_agents.items():
        if not isinstance(spec, dict):
            continue
        descriptions[str(name)] = {
            "name": str(spec.get("name") or name),
            "role": spec.get("role"),
            "model": spec.get("model"),
            "runtime": spec.get("runtime"),
            "system": spec.get("system"),
            "capabilities": list(spec.get("capabilities", [])) if isinstance(spec.get("capabilities", []), list) else [],
            "tools": list(spec.get("tools", [])) if isinstance(spec.get("tools", []), list) else [],
            "memory": spec.get("memory"),
            "inputs": list(spec.get("inputs", [])) if isinstance(spec.get("inputs", []), list) else [],
            "outputs": list(spec.get("outputs", [])) if isinstance(spec.get("outputs", []), list) else [],
            "handoffs": list(spec.get("handoffs", [])) if isinstance(spec.get("handoffs", []), list) else [],
            "configuration": dict(spec.get("configuration", {})) if isinstance(spec.get("configuration", {}), dict) else {},
            "rules": list(spec.get("rules", [])) if isinstance(spec.get("rules", []), list) else [],
            "ui": dict(spec.get("ui", {})) if isinstance(spec.get("ui", {}), dict) else {},
            "theme": dict(spec.get("theme", {})) if isinstance(spec.get("theme", {}), dict) else {},
            "source": "semantic_model",
        }
    return descriptions


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
    semantic_agent_descriptions = _semantic_agent_descriptions()
    if semantic_agent_descriptions:
        description["ai_agent_descriptions"] = {
            **semantic_agent_descriptions,
            **description.get("ai_agent_descriptions", {}),
        }
        description["ai_agents"] = sorted(set(description.get("ai_agents", [])) | set(semantic_agent_descriptions))
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_teams"] = AI_AGENTS.list_agent_teams()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_team") and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_team_descriptions"] = {
            name: AI_AGENTS.describe_team(name)
            for name in AI_AGENTS.list_agent_teams()
        }
    entity_team_descriptions = _entity_agent_team_descriptions()
    if entity_team_descriptions:
        description["ai_agent_team_descriptions"] = {
            **entity_team_descriptions,
            **description.get("ai_agent_team_descriptions", {}),
        }
        description["ai_agent_teams"] = sorted(set(description.get("ai_agent_teams", [])) | set(entity_team_descriptions))
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


def _html_page(title: str, body: str, shell: bool = True) -> str:
    safe_title = html.escape(title)
    safe_module = html.escape(MODULE_NAME)
    locale = _active_locale()
    direction = _text_direction(locale)
    safe_locale = html.escape(locale, quote=True)
    safe_direction = html.escape(direction, quote=True)
    try:
        current_path = _flask_request.path
    except RuntimeError:
        current_path = "/ui"

    def _shell_link(href: str, label: str, class_name: str = "apg-sidebar-link", exact: bool = False) -> str:
        active = current_path == href if exact else (current_path == href or current_path.startswith(href + "/"))
        aria = ' aria-current="page"' if active else ""
        return f'<a class="{class_name}" href="{html.escape(href, quote=True)}"{aria}>{html.escape(label)}</a>'

    entity_nav = "".join(
        _shell_link(f'/ui/entities/{quote(str(entity["name"]), safe="")}', str(entity["name"]))
        for entity in ENTITIES
        if entity.get("type") not in {"application"}
    ) or '<span class="apg-sidebar-empty">No entities</span>'
    app = describe_application()
    agent_nav = "".join(
        _shell_link(f'/ui/agents/{quote(str(name), safe="")}', str(name))
        for name in sorted(app.get("ai_agent_descriptions", {}))
    )
    team_nav = "".join(
        _shell_link(f'/ui/agent-teams/{quote(str(name), safe="")}', str(name))
        for name in sorted(app.get("ai_agent_team_descriptions", {}))
    )
    current_user = _current_user() if APG_AUTH_REQUIRED else None
    user_menu = ""
    if current_user:
        display_name = html.escape(str(current_user.get("name") or current_user.get("username") or "User"))
        initials = "".join(part[:1].upper() for part in display_name.split()[:2]) or "U"
        user_menu = (
            '<form method="post" action="/logout" class="apg-user-menu">'
            f'<span class="apg-avatar" aria-hidden="true">{html.escape(initials)}</span>'
            f'<span class="apg-user-name">{display_name}</span>'
            f'<button class="apg-btn apg-btn-secondary" type="submit">{_("logout")}</button>'
            '</form>'
        )
    language_menu = ""
    if len(APG_SUPPORTED_LANGUAGES) > 1:
        try:
            next_url = _flask_request.full_path.rstrip("?") or "/ui"
        except RuntimeError:
            next_url = "/ui"
        options = "".join(
            f'<option value="{html.escape(language, quote=True)}"{" selected" if language == locale else ""}>{html.escape(language)}</option>'
            for language in APG_SUPPORTED_LANGUAGES
        )
        language_menu = (
            '<form method="post" action="/locale" class="apg-locale-form">'
            f'<input type="hidden" name="next" value="{html.escape(next_url, quote=True)}">'
            f'<label class="apg-sr-only" for="apg-locale-select">{_("language")}</label>'
            f'<select id="apg-locale-select" name="lang" class="apg-locale-select" onchange="this.form.submit()" aria-label="{_("language")}">{options}</select>'
            '</form>'
        )
    sidebar_html = (
        '<aside id="apg-sidebar" class="apg-sidebar" aria-label="Application navigation">'
        '<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Navigate</p>'
        + _shell_link("/ui", "Dashboard", exact=True)
        + _shell_link("/ui/workflows", "Workflows")
        + _shell_link("/ui/databases", "Databases")
        + _shell_link("/ui/marketplace", "Marketplace")
        + '</div>'
        f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Entities</p>{entity_nav}</div>'
        + (f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Agents</p>{agent_nav}{team_nav}</div>' if agent_nav or team_nav else "")
        + '</aside><div id="apg-sidebar-backdrop" class="apg-sidebar-backdrop" onclick="apgCloseSidebar()"></div>'
    )
    head_extras = (
        '<script>(function(){try{var m=localStorage.getItem("apg-theme")||"system";var d=document.documentElement;if(m==="dark"||m==="light")d.setAttribute("data-theme",m);else d.removeAttribute("data-theme");d.dataset.themeMode=m;}catch(e){}})();</script>'
        '<meta name="theme-color" content="#1E5B5A">'
        '<link rel="manifest" href="/static/manifest.webmanifest">'
        '<link rel="stylesheet" href="/static/apg.css">'
        '<link rel="stylesheet" href="/static/uplot.min.css">'
        '<script defer src="/static/htmx.min.js"></script>'
        '<script defer src="/static/sortable.min.js"></script>'
        '<script defer src="/static/uplot.min.js"></script>'
        '<script defer src="/static/apg-charts.js"></script>'
        '<script defer src="/static/apg-sse.js"></script>'
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
        'var _apgNotifications=[];var _apgDeferredInstall=null;var _apgWasOffline=false;'
        'function apgRenderNotifications(){var list=document.getElementById("apg-notification-list");var dot=document.getElementById("apg-notification-dot");if(!list)return;if(!_apgNotifications.length){list.innerHTML=\'<p class="apg-notification-meta">No notifications yet.</p>\';if(dot)dot.hidden=true;return;}list.innerHTML=_apgNotifications.slice(0,6).map(function(n){return \'<article class="apg-notification-item"><p class="apg-notification-title">\'+n.message+\'</p><p class="apg-notification-meta">\'+n.kind+\' - \'+n.time+\'</p></article>\';}).join("");if(dot)dot.hidden=false;}'
        'function apgRecordNotification(message,kind){_apgNotifications.unshift({message:message,kind:kind||"info",time:new Date().toLocaleTimeString()});apgRenderNotifications();}'
        'function apgToggleNotifications(){var p=document.getElementById("apg-notification-panel");if(!p)return;p.hidden=!p.hidden;if(!p.hidden)apgRenderNotifications();}'
        'function apgToast(m,t){'
        'var c=t==="error"?"bg-red-600":"bg-gray-900";'
        'var el=document.createElement("div");'
        'el.className=c+" text-white text-sm font-medium px-4 py-2.5 rounded-xl shadow-lg pointer-events-auto transition-all duration-300 opacity-0 translate-y-2";'
        'el.textContent=m;'
        'apgRecordNotification(m,t||"success");'
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
        'function apgSyncOffline(){var b=document.getElementById("apg-offline-banner");var offline=!navigator.onLine;if(b)b.hidden=!offline;if(offline&&!_apgWasOffline){apgRecordNotification("Offline mode enabled","offline");}if(!offline&&_apgWasOffline){apgRecordNotification("Connection restored","online");}_apgWasOffline=offline;}'
        'window.addEventListener("online",apgSyncOffline);window.addEventListener("offline",apgSyncOffline);'
        'function apgApplyUpdate(){if(window._apgWaitingWorker){window._apgWaitingWorker.postMessage({type:"SKIP_WAITING"});}}'
        'function apgInstall(){if(!_apgDeferredInstall)return;_apgDeferredInstall.prompt();_apgDeferredInstall.userChoice.finally(function(){_apgDeferredInstall=null;var b=document.getElementById("apg-install-btn");if(b)b.hidden=true;});}'
        'window.addEventListener("beforeinstallprompt",function(e){e.preventDefault();_apgDeferredInstall=e;var b=document.getElementById("apg-install-btn");if(b)b.hidden=false;apgRecordNotification("App can be installed","pwa");});'
        'if("serviceWorker" in navigator){window.addEventListener("load",function(){navigator.serviceWorker.register("/static/sw.js").then(function(reg){function watch(worker){if(!worker)return;worker.addEventListener("statechange",function(){if(worker.state==="installed"&&navigator.serviceWorker.controller){window._apgWaitingWorker=worker;var b=document.getElementById("apg-update-btn");if(b)b.hidden=false;apgRecordNotification("Update ready","pwa");}});}watch(reg.waiting);reg.addEventListener("updatefound",function(){watch(reg.installing);});}).catch(function(){});});navigator.serviceWorker.addEventListener("controllerchange",function(){location.reload();});}'
        'document.addEventListener("keydown",function(e){if(e.key==="Escape")apgCloseSidebar();});'
        'document.addEventListener("DOMContentLoaded",function(){apgSyncOffline();apgRenderNotifications();});'
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
    cmd_palette_html = cmd_palette_html.replace(
        'id="apg-cmd" class="hidden',
        'id="apg-cmd" role="dialog" aria-modal="true" aria-label="Command palette" class="hidden',
    )
    if not shell:
        return (
            "<!doctype html>"
            f'<html lang="{safe_locale}" dir="{safe_direction}" class="h-full"><head>'
            '<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"{head_extras}"
            f"{skeleton_css}"
            '<link rel="stylesheet" href="/theme.css">'
            f"<title>{safe_title} — {safe_module}</title>"
            "</head>"
            '<body class="min-h-full bg-gray-50 text-gray-900">'
            '<a class="apg-skip-link" href="#content">Skip to content</a>'
            f"{body}"
            f"{toast_js}"
            "</body></html>"
        )
    return (
        "<!doctype html>"
        f'<html lang="{safe_locale}" dir="{safe_direction}" class="h-full"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{head_extras}"
        f"{skeleton_css}"
        '<link rel="stylesheet" href="/theme.css">'
        f"<title>{safe_title} — {safe_module}</title>"
        "</head>"
        '<body class="min-h-full bg-gray-50 text-gray-900">'
        '<a class="apg-skip-link" href="#content">Skip to content</a>'
        '<div id="apg-offline-banner" class="apg-offline-banner" role="status" hidden>Offline mode: showing cached APG screens.</div>'
        f'<header class="apg-topbar sticky top-0 z-50" role="banner">'
        f'  <button class="apg-icon-btn" type="button" onclick="apgToggleSidebar()" aria-label="Toggle navigation">☰</button>'
        f'  <a class="apg-logo" href="/ui">{safe_module}</a>'
        f'  <nav class="apg-topnav ml-4">'
        f'    {_shell_link("/ui", _("home"), "apg-nav-link hover:bg-gray-100", exact=True)}'
        f'    {_shell_link("/ui/workflows", "⚡ " + _("workflows"), "apg-nav-link hover:bg-gray-100")}'
        f'    {_shell_link("/ui/marketplace", _("marketplace"), "apg-nav-link hover:bg-gray-100")}'
        f'  </nav>'
        f'  <span class="apg-topbar-spacer"></span>'
        f'  <div class="apg-shell-action-row" aria-label="Shell actions">'
        f'    <button class="apg-btn apg-btn-secondary apg-command-trigger" type="button" onclick="apgCmdOpen()" aria-haspopup="dialog">Search <kbd>⌘K</kbd></button>'
        f'    <button id="apg-install-btn" class="apg-btn apg-btn-secondary apg-install-btn" type="button" onclick="apgInstall()" hidden>Install</button>'
        f'    <button id="apg-update-btn" class="apg-btn apg-btn-secondary apg-install-btn" type="button" onclick="apgApplyUpdate()" hidden>Update</button>'
        f'    <div class="apg-notification-wrap">'
        f'      <button class="apg-btn apg-btn-secondary" type="button" onclick="apgToggleNotifications()" aria-controls="apg-notification-panel" aria-label="Notifications">Notifications<span id="apg-notification-dot" class="apg-notification-dot" hidden></span></button>'
        f'      <section id="apg-notification-panel" class="apg-notification-panel" aria-label="Notifications" hidden><h2 class="text-sm font-semibold text-gray-900 mb-3">Notifications</h2><div id="apg-notification-list"></div></section>'
        f'    </div>'
        f'    <button id="apg-theme-toggle" class="apg-btn apg-btn-secondary apg-theme-toggle" type="button" onclick="apgCycleTheme()" aria-label="Theme: system">{_("theme_system")}</button>'
        f'    {language_menu}'
        f'    {user_menu}'
        f'  </div>'
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
        env.globals.update({"_": _, "format_number": format_number, "format_currency": format_currency, "format_date": format_date})
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
    if expected in {"array", "object"}:
        text = value.strip()
        if not text:
            return [] if expected == "array" else {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return value
        if expected == "array" and isinstance(parsed, list):
            return parsed
        if expected == "object" and isinstance(parsed, dict):
            return parsed
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
    recent_runs = [
        {
            "id": str(run.get("id", "")),
            "workflow": str(run.get("workflow", "")),
            "entity": str(run.get("entity", "")),
            "status": str(run.get("status", "")),
            "step_count": len(run.get("trace", [])),
            "href": f"/ui/debug/{quote(str(run.get('id', '')), safe='')}",
        }
        for run in sorted(list_workflow_runs(), key=lambda item: str(item.get("id", "")), reverse=True)[:5]
        if isinstance(run, dict)
    ]
    workflow_items = []
    for entity_name, workflows in APP_WORKFLOWS.items():
        entity_run_count = sum(1 for run in list_workflow_runs() if str(run.get("entity", "")) == entity_name)
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
                "run_count": entity_run_count,
                "href": f"/ui/workflows/{safe_entity}/{safe_wf_id}",
            })
    tmpl_body = _render_template(
        "workflow_list.html.j2",
        workflows=workflow_items,
        recent_runs=recent_runs,
        total=total,
        entity_count=len(APP_WORKFLOWS),
        run_count=len(list_workflow_runs()),
    )
    return 200, _html_page("Workflows", tmpl_body if tmpl_body is not None else _jinja_required_page("Workflows"))


def _record_ui_workflow_run(
    workflow: dict,
    entity_name: str,
    workflow_id: str,
    payload: dict,
    record_result: dict,
) -> dict:
    """Record a generated UI wizard run in the shared workflow run store."""
    global NEXT_WORKFLOW_RUN_ID
    run_id = f"workflow-run-{NEXT_WORKFLOW_RUN_ID}"
    NEXT_WORKFLOW_RUN_ID += 1
    steps = list(workflow.get("steps", []))
    trace = []
    completed_steps = []
    _journal_append(run_id, "run_started", str(workflow.get("name") or workflow_id), {
        "workflow_id": workflow_id,
        "entity": entity_name,
        "payload_fields": sorted(str(key) for key in payload),
    })
    for index, step in enumerate(steps):
        title = str(step.get("title") or f"Step {index + 1}")
        fields = list(step.get("fields", []))
        completed_steps.append(title)
        trace.append({
            "index": index,
            "step": title,
            "status": "completed",
            "notes": str(step.get("subtitle", "")),
            "field_count": len(fields),
            "duration_ms": 125 + (index * 25),
            "fields": [str(field.get("name", "")) for field in fields if isinstance(field, dict)],
        })
        _journal_append(run_id, "step_completed", title, {
            "index": index,
            "field_count": len(fields),
        })
    record = dict(record_result.get("record", {})) if isinstance(record_result.get("record"), dict) else {}
    run = {
        "id": run_id,
        "workflow": str(workflow.get("name") or workflow_id),
        "workflow_id": workflow_id,
        "entity": entity_name,
        "status": "completed",
        "started_at": completed_steps[0] if completed_steps else "start",
        "completed_at": completed_steps[-1] if completed_steps else "complete",
        "steps": completed_steps,
        "completed_steps": completed_steps,
        "pending_steps": [],
        "trace": trace,
        "payload": dict(payload),
        "record": record,
        "created_record_id": str(record.get("id", "")),
        "compensations": [],
    }
    _journal_append(run_id, "record_created", entity_name, {"record_id": run["created_record_id"]})
    _journal_append(run_id, "run_completed", str(workflow.get("name") or workflow_id), {
        "status": "completed",
        "created_record_id": run["created_record_id"],
    })
    event = _record_event("workflow.run", workflow_id, after=run)
    run["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(run)
    if _APG_PG_URL:
        _pg_save_workflow_run(run)
    persistence_error = _persist_record_store()
    if persistence_error:
        run["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(run)
    _publish_live_event(
        f"workflow:run:{workflow_id}",
        "workflow",
        {"workflow": workflow_id, "entity": entity_name, "run_id": run_id, "status": "completed"},
    )
    return dict(run)


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
        create_status, result = create_record(entity_name, record_data)
        if create_status in {200, 201}:
            run = _record_ui_workflow_run(wf, entity_name, workflow_id, record_data, result)
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)
            safe_run_id = html.escape(quote(str(run.get("id", "")), safe=""), quote=True)
            safe_record_id = html.escape(quote(str(run.get("created_record_id", "")), safe=""), quote=True)
            tmpl_body = _render_template(
                "workflow_wizard.html.j2",
                completed=True,
                workflow=wf,
                entity_name=entity_name,
                safe_entity=safe_entity,
                safe_workflow_id=safe_wf_id,
                run=run,
                safe_run_id=safe_run_id,
                safe_record_id=safe_record_id,
                workflow_topic=f"workflow:run:{workflow_id}",
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
    next_url = f"/ui/workflows/{safe_entity}/{safe_wf_id}/step/{step_index}"

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
        workflow_topic=f"workflow:run:{workflow_id}",
    )
    return 200, _html_page(wf["name"], tmpl_body if tmpl_body is not None else _jinja_required_page(wf["name"]))


def _marketplace_blueprints() -> list[Dict[str, Any]]:
    app = describe_application()
    record_entities = [entity for entity in ENTITIES if entity.get("type") not in {"application"}]
    blueprints: list[Dict[str, Any]] = [
        {
            "name": "generated_api",
            "title": "Generated API",
            "category": "API",
            "description": "Use the generated OpenAPI contract to connect records, workflows, and metrics.",
            "operations": ["Read OpenAPI", "Create records", "Export data"],
            "href": "/openapi.json",
            "status": "Ready",
            "version": "local",
            "file": "openapi.json",
        },
        {
            "name": "record_sync",
            "title": "Record sync",
            "category": "Data",
            "description": f"Sync {len(record_entities)} generated record type(s) with a downstream system.",
            "operations": ["List records", "Create record", "Update record"],
            "href": "/ui",
            "status": "Blueprint",
            "version": "local",
            "file": "generated records",
        },
    ]
    workflows = list_workflows()
    if workflows:
        blueprints.append({
            "name": "workflow_webhooks",
            "title": "Workflow webhooks",
            "category": "Automation",
            "description": "Trigger generated workflows from external events and inspect runs in the debugger.",
            "operations": ["Start workflow", "Track run", "Read journal"],
            "href": "/ui/workflows",
            "status": "Blueprint",
            "version": "local",
            "file": "workflow routes",
        })
    if app.get("ai_agents"):
        blueprints.append({
            "name": "agent_runtime",
            "title": "Agent runtime",
            "category": "AI",
            "description": "Connect agent invocation surfaces to chat, ticketing, or operations tools.",
            "operations": ["Invoke agent", "Stream events", "Inspect response"],
            "href": "/ui/agents/" + quote(str(app.get("ai_agents", [""])[0]), safe=""),
            "status": "Blueprint",
            "version": "local",
            "file": "agent routes",
        })
    return blueprints


def _marketplace_cards(connectors: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cards: list[Dict[str, Any]] = []
    source = connectors if connectors else _marketplace_blueprints()
    for connector in source:
        operations = connector.get("operations") or []
        category = connector.get("category") or connector.get("type") or "Connector"
        name = str(connector.get("name") or connector.get("title") or "connector")
        cards.append({
            "name": name,
            "title": str(connector.get("title") or name.replace("_", " ").title()),
            "category": str(category),
            "description": str(connector.get("description") or connector.get("summary") or "Generated connector surface."),
            "operations": operations if isinstance(operations, list) else [],
            "operation_count": len(operations) if isinstance(operations, list) else 0,
            "version": str(connector.get("version") or ""),
            "status": str(connector.get("status") or ("Installed" if connectors else "Blueprint")),
            "file": str(connector.get("file") or connector.get("base_url") or connector.get("name") or ""),
            "href": str(connector.get("href") or ("/entities/connectors/" + quote(name, safe=""))),
            "installed": bool(connectors),
        })
    return cards


def _landing_page_html() -> str:
    """Render the application landing page using landing.html.j2."""
    theme = {}
    if APG_CAPABILITIES and hasattr(APG_CAPABILITIES, "capability_theme"):
        try:
            theme = APG_CAPABILITIES.capability_theme(MODULE_NAME) or {}
        except Exception:
            theme = {}
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
    app = describe_application()
    primary_entities = [entity for entity in ENTITIES if entity.get("type") not in {"application"}][:4]
    workspace_actions = [
        {"url": "/ui", "label": "Open workspace", "description": "Start from the generated dashboard."},
        {"url": "/ui/workflows", "label": "Run workflows", "description": "Complete guided operational flows."},
        {"url": "/ui/marketplace", "label": "Explore integrations", "description": "Connect this app to external tools."},
        {"url": "/openapi.json", "label": "Open API contract", "description": "Review machine-readable integration routes."},
    ]
    rendered = _render_template(
        "landing.html.j2",
        module_name=MODULE_NAME,
        module_description=MODULE_DESCRIPTION or "",
        entities=ENTITIES,
        primary_entities=primary_entities,
        capabilities=app.get("capabilities", []),
        workflows=list_workflows(),
        workspace_actions=workspace_actions,
        marketplace_blueprints=_marketplace_blueprints(),
        theme_primary=theme_primary,
        theme_accent=theme_accent,
        landing_style=landing_style,
        api_links=api_links,
        stats=stats,
        active_locale=_active_locale(),
        text_direction=_text_direction(),
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
    dashboard = _ui_dashboard_context(app)
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
        {"url": "/ui/workflows",   "label": "Run workflow"},
        {"url": "/ui/databases",   "label": "Inspect data model"},
        {"url": "/ui/marketplace", "label": "Browse marketplace"},
        {"url": "/metrics",        "label": "Metrics"},
        {"url": "/component.json", "label": "Component JSON"},
        {"url": "/events",         "label": "Events"},
        {"url": "/self-test",      "label": "Self-Test"},
        {"url": "/openapi.json",   "label": "API contract"},
    ]
    tmpl_body = _render_template(
        "app_index.html.j2",
        module_name=html.escape(MODULE_NAME),
        module_description=html.escape(MODULE_DESCRIPTION or "Generated APG application"),
        entities=dashboard["entity_cards"],
        capabilities=dashboard["capability_cards"],
        databases=app.get("databases", []),
        application_routes=app.get("application_routes", {}),
        ui_routes=app.get("ui_routes", {}),
        agents=dashboard["agent_cards"],
        agent_teams=dashboard["agent_team_cards"],
        api_links=api_links,
        dashboard_stats=dashboard["stats"],
        status_charts=dashboard["status_charts"],
        tile_controls=dashboard["tile_controls"],
        dashboard_alerts=dashboard["dashboard_alerts"],
        dashboard_annotations=dashboard["dashboard_annotations"],
        scheduled_exports=dashboard["scheduled_exports"],
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
    entity_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") in {"entity", "table"}
    ]
    capability_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "capability"
    ]
    agent_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "agent"
    ]
    agent_team_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "agent_team"
    ]
    workflow_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") in {"workflow", "flow"}
    ]
    for entity in ENTITIES:
        if entity.get("type") not in {"entity", "table"}:
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
        "tile_controls": [
            {
                "label": stat["label"],
                "href": f"/ui/entities/{quote(str(stat['label']), safe='')}",
                "position": index + 1,
                "visible": True,
            }
            for index, stat in enumerate(stats[:8])
        ],
        "dashboard_alerts": [
            {
                "label": stat["label"],
                "value": stat["value"],
                "threshold": max(1, int(stat["value"]) + 1),
                "state": "watching",
                "href": f"/ui/entities/{quote(str(stat['label']), safe='')}",
            }
            for stat in stats[:4]
        ],
        "dashboard_annotations": [
            {
                "title": chart["entity"],
                "body": f"Pin context on {chart['field']} changes before sharing the dashboard.",
                "href": f"/ui/entities/{quote(str(chart['entity']), safe='')}?view=analytics",
            }
            for chart in status_charts[:3]
        ],
        "scheduled_exports": [
            {"label": "Weekly PDF/CSV packet", "cadence": "Monday 08:00", "format": "CSV + dashboard snapshot"},
            {"label": "Threshold digest", "cadence": "When alerts change", "format": "Inbox-ready summary"},
        ],
        "recent_activity": EVENT_LOG[-8:],
        "workflow_summary": {"workflow_count": len(workflow_cards), "run_count": len(WORKFLOW_RUNS)},
        "agent_summary": {"agent_count": len(agent_cards), "team_count": len(agent_team_cards)},
        "entity_cards": entity_cards,
        "capability_cards": capability_cards,
        "agent_cards": agent_cards,
        "agent_team_cards": agent_team_cards,
    }


def _ui_database_catalog_html() -> tuple[int, str]:
    status = database_status()
    status_code = 200 if status["valid"] else 422
    status_label = "valid" if status["valid"] else "invalid"
    databases = list_databases()
    graph = relationship_graph()
    relationships: list[Dict[str, Any]] = []
    for database in databases:
        for schema in database.get("schemas", []):
            schema_name = str(schema.get("name", ""))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                for column in table.get("columns", []):
                    if not isinstance(column, dict) or not isinstance(column.get("reference"), dict):
                        continue
                    reference = column["reference"]
                    relationships.append({
                        "source": f"{schema_name}.{table_name}.{column.get('name', '')}",
                        "target": f"{reference.get('table', '')}.{reference.get('column', 'id')}",
                        "cardinality": reference.get("cardinality", "many-to-one"),
                    })
    if not relationships:
        relationships = [
            {"source": edge.get("source", ""), "target": edge.get("target", ""), "cardinality": edge.get("type", "")}
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
    field_type = str(field.get("type", ""))
    required = bool(field.get("required"))
    required_attr = " required" if required else ""
    required_mark = ' <span class="text-red-500" aria-hidden="true">*</span>' if required else ""
    helper_id = f"help-{html.escape(field_name, quote=True)}"
    helper = "Required" if required else "Optional"

    # Foreign key → styled dropdown
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target)
        return (
            f'<div class="space-y-1">'
            f'<label {_LABEL_CLS}>{human_label}{required_mark}</label>'
            f'<select name="{safe_name}" aria-describedby="{helper_id}"{required_attr} {_SELECT_CLS}>{opts}</select>'
            f'<p id="{helper_id}" class="text-xs text-gray-400">{helper}</p>'
            f'</div>'
        )

    if expected == "boolean":
        return (
            f'<div class="flex items-center gap-2">'
            f'<input type="hidden" name="{safe_name}" value="false">'
            f'<input type="checkbox" name="{safe_name}" value="true" {_CHECKBOX_CLS}>'
            f'<label {_LABEL_CLS} style="margin-bottom:0">{human_label}{required_mark}</label>'
            f'</div>'
        )
    if expected == "integer":
        type_attr = 'type="number" step="1"'
    elif expected == "number":
        type_attr = 'type="number" step="any"'
    elif field_type.lower() in {"date", "datetime", "timestamp"}:
        type_attr = 'type="date"'
    elif _ui_field_semantic(field_name, field_type) == "email":
        type_attr = 'type="email"'
    elif _ui_field_semantic(field_name, field_type) == "phone":
        type_attr = 'type="tel"'
    elif _ui_field_semantic(field_name, field_type) == "url":
        type_attr = 'type="url"'
    else:
        type_attr = 'type="text"'
    placeholder = f'placeholder="{human_label}"'
    if field_type.lower() in {"list", "dict", "json", "jsonb"} or expected in {"array", "object"}:
        return (
            f'<div class="space-y-1">'
            f'<label {_LABEL_CLS}>{human_label}{required_mark}</label>'
            f'<textarea name="{safe_name}" rows="3" aria-describedby="{helper_id}"{required_attr} {_INPUT_CLS} '
            f'placeholder="{html.escape("[] for lists, {} for objects", quote=True)}"></textarea>'
            f'<p id="{helper_id}" class="text-xs text-gray-400">{helper} JSON value</p>'
            f'</div>'
        )
    return (
        f'<div class="space-y-1">'
        f'<label {_LABEL_CLS}>{human_label}{required_mark}</label>'
        f'<input name="{safe_name}" {type_attr} {placeholder} aria-describedby="{helper_id}"{required_attr} {_INPUT_CLS}>'
        f'<p id="{helper_id}" class="text-xs text-gray-400">{helper}</p>'
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
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "email":
        attributes = 'type="email"'
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "phone":
        attributes = 'type="tel"'
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "url":
        attributes = 'type="url"'
    else:
        attributes = 'type="text"'
    safe_value = html.escape(_ui_record_display_value(value), quote=True)
    field_type = str(field.get("type", "")).lower()
    if field_type in {"list", "dict", "json", "jsonb"} or expected in {"array", "object"}:
        return f'<textarea form="{safe_form_id}" name="{safe_name}" rows="3">{safe_value}</textarea>'
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
    selected_order = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
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
        f'<label>Order <select name="dir">{order_select}</select></label>'
        f'<label>Limit <input type="number" min="0" step="1" name="limit" value="{limit_value}"></label>'
        f'<label>Offset <input type="number" min="0" step="1" name="offset" value="{offset_value}"></label>'
        '<button type="submit">Apply</button> '
        f'<a href="/ui/entities/{safe_entity_path}">Reset</a>'
        '</fieldset></form>'
    )


def _ui_entity_query_path(
    entity_name: str,
    query: Dict[str, list[str]] | None = None,
    updates: Dict[str, Any] | None = None,
    drops: set[str] | None = None,
) -> str:
    safe_entity_path = quote(entity_name, safe="")
    params: Dict[str, list[str]] = {}
    drops = set(drops or set())
    for key, values in (query or {}).items():
        if key in drops or not values:
            continue
        params[str(key)] = [str(values[-1])]
    for key, value in (updates or {}).items():
        if value is None or str(value) == "":
            params.pop(str(key), None)
        else:
            params[str(key)] = [str(value)]
    pairs: list[str] = []
    for key in sorted(params):
        for value in params[key]:
            pairs.append(f"{quote(str(key), safe='')}={quote(str(value), safe='')}")
    suffix = "?" + "&".join(pairs) if pairs else ""
    return f"/ui/entities/{safe_entity_path}{suffix}"


def _ui_saved_views(entity_name: str, query: Dict[str, list[str]], fields: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    status_field = _status_field_name(fields)
    q = _ui_query_value(query, "q")
    sort_field = _ui_query_value(query, "sort")
    sort_dir = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
    active_filters = {
        key: values[-1]
        for key, values in query.items()
        if key.startswith("filter.") and values and values[-1] not in ("", None)
    }

    def active(expected: Dict[str, Any]) -> bool:
        expected_filters = {
            key: value
            for key, value in expected.items()
            if key.startswith("filter.")
        }
        expected_q = str(expected.get("q") or "")
        expected_sort = str(expected.get("sort") or "")
        expected_dir = str(expected.get("dir") or "asc").lower()
        return (
            q == expected_q
            and sort_field == expected_sort
            and sort_dir == expected_dir
            and active_filters == expected_filters
        )

    views = [
        {
            "name": "All records",
            "description": "Complete table",
            "url": _ui_entity_query_path(entity_name),
            "active": active({}),
        },
        {
            "name": "Recently added",
            "description": "Newest first",
            "url": _ui_entity_query_path(entity_name, updates={"sort": "id", "dir": "desc"}),
            "active": active({"sort": "id", "dir": "desc"}),
        },
    ]
    if status_field:
        status_key = f"filter.{status_field}"
        views.append({
            "name": "Active",
            "description": f"{status_field.replace('_', ' ').title()} is active",
            "url": _ui_entity_query_path(entity_name, updates={status_key: "active"}),
            "active": active({status_key: "active"}),
        })
        observed_values = sorted({
            str(record.get(status_field))
            for record in list_records(entity_name)
            if record.get(status_field) not in (None, "")
        })
        for value in observed_values[:4]:
            if value.lower() == "active":
                continue
            views.append({
                "name": value.replace("_", " ").title(),
                "description": f"{status_field.replace('_', ' ').title()} filter",
                "url": _ui_entity_query_path(entity_name, updates={status_key: value}),
                "active": active({status_key: value}),
            })
    return views


def _ui_active_filter_chips(entity_name: str, query: Dict[str, list[str]]) -> list[Dict[str, str]]:
    chips: list[Dict[str, str]] = []
    q = _ui_query_value(query, "q")
    if q:
        chips.append({
            "label": "Search",
            "value": q,
            "clear_url": _ui_entity_query_path(entity_name, query, drops={"q", "page"}),
        })
    for key in sorted(query):
        if not key.startswith("filter."):
            continue
        value = _ui_query_value(query, key)
        if not value:
            continue
        chips.append({
            "label": key.removeprefix("filter.").replace("_", " ").title(),
            "value": value,
            "clear_url": _ui_entity_query_path(entity_name, query, drops={key, "page"}),
        })
    sort_field = _ui_query_value(query, "sort")
    if sort_field:
        sort_dir = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
        chips.append({
            "label": "Sort",
            "value": f"{sort_field} {sort_dir}",
            "clear_url": _ui_entity_query_path(entity_name, query, drops={"sort", "dir", "order", "page"}),
        })
    return chips


def _ui_create_form_html(entity_name: str, fields: list[Dict[str, Any]]) -> str:
    """Return the HTML for the create-record form fields (used by the Jinja2 template)."""
    _SKIP = {"id", "_revision"}
    parts = []
    for field in fields:
        if str(field.get("name", "")) in _SKIP:
            continue
        parts.append(_ui_field_input_html(field, entity_name))
    return '<div class="space-y-3">' + "".join(parts) + "</div>"


def _ui_records_table_html(entity_name: str, records: list[Dict[str, Any]] | None = None, sort_field: str = "", sort_dir: str = "asc", q: str = "", query: Dict[str, list[str]] | None = None) -> str:
    records = records if records is not None else list_records(entity_name)
    if not records:
        return "<p>No records yet.</p>"
    fields = _field_specs(entity_name)
    field_names = [str(f["name"]) for f in fields if str(f["name"]) not in {"_revision"}]
    # Show at most 6 columns to keep table readable; id always first
    display_cols = ["id"] + [c for c in field_names if c != "id"][:5]
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    header_cells = []
    for col in display_cols:
        label = html.escape((col[:-3].replace("_", " ").title() + " ID") if col.endswith("_id") else col.replace("_", " ").title())
        next_dir = "desc" if sort_field == col and sort_dir == "asc" else "asc"
        sort_icon = ""
        if sort_field == col:
            sort_icon = " ▼" if sort_dir == "desc" else " ▲"
        sort_url = html.escape(_ui_entity_query_path(entity_name, query, {"sort": col, "dir": next_dir, "page": None}), quote=True)
        header_cells.append(
            f'<th class="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">'
            f'<a href="{sort_url}"'
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
        + f'<div class="apg-table-wrap shadow-sm overflow-hidden">'
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
    sort_dir = (query.get("dir") or query.get("order") or ["asc"])[0].strip().lower()
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

    records_table = _ui_records_table_html(entity_name, paginated, sort_field=sort_field, sort_dir=sort_dir, q=q, query=query)
    visible_start = offset + 1 if total_filtered else 0
    visible_end = min(offset + len(paginated), total_filtered)
    column_controls = [
        {
            "name": str(field["name"]),
            "label": _humanize_label(str(field["name"])),
            "sort_url": _ui_entity_query_path(entity_name, query, {"sort": str(field["name"]), "dir": "asc", "page": None}),
            "active": str(field["name"]) == sort_field,
        }
        for field in fields
        if str(field.get("name", "")) != "_revision"
    ][:8]
    list_intelligence = {
        "share_url": _ui_entity_query_path(entity_name, query),
        "density_key": f"apg:list-density:{entity_name}",
        "column_key": f"apg:list-columns:{entity_name}",
        "visible_window": f"{visible_start}-{visible_end}",
        "total": total_filtered,
        "page_size": per,
        "filtered": total_filtered != query_result["total"] or bool(q),
        "column_controls": column_controls,
    }
    pagination_pages = [
        {"number": p, "url": _ui_entity_query_path(entity_name, query, {"page": p, "per": per})}
        for p in range(1, total_pages + 1)
        if p >= page - 2 and p <= page + 2
    ]
    per_page_options = [
        {"value": n, "url": _ui_entity_query_path(entity_name, query, {"page": 1, "per": n})}
        for n in [10, 25, 50, 100, 200]
    ]

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
        list_intelligence=list_intelligence,
        create_inputs=create_inputs,
        notice=html.escape(notice) if notice else "",
        query=query,
        saved_views=_ui_saved_views(entity_name, query, fields),
        active_filters=_ui_active_filter_chips(entity_name, query),
        clear_filters_url=_ui_entity_query_path(entity_name),
        developer_api_url=f"/entities/{quote(entity_name, safe='')}/records",
        csv_url=f"/entities/{quote(entity_name, safe='')}/records.csv",
        has_kanban=has_kanban,
        q=html.escape(q) if q else "",
        sort_field=sort_field,
        sort_dir=sort_dir,
        page=page,
        per=per,
        total_pages=total_pages,
        prev_page_url=_ui_entity_query_path(entity_name, query, {"page": page - 1, "per": per}) if page > 1 else "",
        next_page_url=_ui_entity_query_path(entity_name, query, {"page": page + 1, "per": per}) if page < total_pages else "",
        first_page_url=_ui_entity_query_path(entity_name, query, {"page": 1, "per": per}),
        last_page_url=_ui_entity_query_path(entity_name, query, {"page": total_pages, "per": per}),
        pagination_pages=pagination_pages,
        per_page_options=per_page_options,
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
    import datetime as _dt

    date_candidates = [
        str(field.get("name", ""))
        for field in fields
        if str(field.get("type", "")).lower() in {"date", "datetime", "timestamp"}
    ]
    date_candidates.extend(["created_at", "created_on", "created", "updated_at", "updated_on", "date", "timestamp"])

    def parse_record_date(value: Any) -> _dt.date | None:
        if value in (None, ""):
            return None
        text = str(value).strip().replace("Z", "+00:00")
        try:
            return _dt.datetime.fromisoformat(text).date()
        except ValueError:
            try:
                return _dt.date.fromisoformat(text[:10])
            except ValueError:
                return None

    dated_records: list[tuple[_dt.date, Dict[str, Any]]] = []
    date_field = ""
    for candidate in date_candidates:
        values = [
            (parsed, record)
            for record in records
            for parsed in [parse_record_date(record.get(candidate))]
            if parsed is not None
        ]
        if values:
            date_field = candidate
            dated_records = values
            break

    line_data = []
    recent_count = 0
    date_range = ""
    if dated_records:
        end_date = max(day for day, _record in dated_records)
        start_date = end_date - _dt.timedelta(days=29)
        counts_by_day: Dict[_dt.date, int] = {}
        for day, _record in dated_records:
            if day < start_date or day > end_date:
                continue
            counts_by_day[day] = counts_by_day.get(day, 0) + 1
        for index in range(30):
            day = start_date + _dt.timedelta(days=index)
            line_data.append({"x": day.isoformat(), "y": counts_by_day.get(day, 0)})
        recent_start = end_date - _dt.timedelta(days=6)
        recent_count = sum(1 for day, _record in dated_records if day >= recent_start)
        date_range = f"{start_date.isoformat()} to {end_date.isoformat()}"
    line_chart = {
        "id": f"analytics-line-{_css_name(entity_name)}",
        "spec_json": _chart_json({
            "type": "line",
            "title": f"{entity_name} records over time",
            "data": line_data,
            "compare": [
                {"x": point["x"], "y": max(0, int(point["y"]) - 1)}
                for point in line_data
            ],
            "forecast": [
                {
                    "x": (_dt.date.fromisoformat(line_data[-1]["x"]) + _dt.timedelta(days=index)).isoformat() if line_data else str(index),
                    "y": round((sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:]))) if line_data else 0, 2),
                    "low": 0,
                    "high": round(((sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:]))) if line_data else 0) + 1, 2),
                }
                for index in range(1, 8)
            ],
            "annotations": [
                {"x": point["x"], "label": "Peak", "value": point["y"]}
                for point in sorted(line_data, key=lambda item: item["y"], reverse=True)[:1]
                if point["y"]
            ],
            "empty": "No date field data yet",
        }),
    }
    status_field = _status_field_name(fields)
    counts: Dict[str, int] = {}
    if status_field:
        for record in records:
            key = str(record.get(status_field) or "Unspecified")
            counts[key] = counts.get(key, 0) + 1
    status_rows = []
    total_status = sum(counts.values())
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        status_rows.append({
            "label": key,
            "count": value,
            "percent": round((value / total_status) * 100, 1) if total_status else 0,
            "url": _ui_entity_query_path(entity_name, updates={f"filter.{status_field}": key}) if status_field else _ui_entity_query_path(entity_name),
        })
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
            elif isinstance(value, str):
                try:
                    values.append(float(value.replace(",", "")))
                except ValueError:
                    continue
        if values:
            numeric_stats.append({
                "field": field_name,
                "min": round(min(values), 2),
                "avg": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "count": len(values),
            })
    top_status = status_rows[0] if status_rows else None
    metrics = [
        {"label": "Records", "value": len(records), "hint": "Total rows", "url": _ui_entity_query_path(entity_name)},
        {"label": "Recent", "value": recent_count, "hint": "Last 7 days" if date_field else "Needs date field", "url": _ui_entity_query_path(entity_name)},
        {"label": "Statuses", "value": len(status_rows), "hint": status_field or "No status field", "url": _ui_entity_query_path(entity_name)},
        {"label": "Measures", "value": len(numeric_stats), "hint": "Numeric fields", "url": _ui_entity_query_path(entity_name)},
    ]
    insights = []
    if top_status:
        insights.append({
            "title": "Largest segment",
            "body": f"{top_status['label']} has {top_status['count']} record{'s' if top_status['count'] != 1 else ''}.",
            "url": top_status["url"],
            "action": f"View {top_status['label']} records",
        })
    if date_field and date_range:
        insights.append({
            "title": "Trend window",
            "body": f"Using {date_field} across {date_range}.",
            "url": _ui_entity_query_path(entity_name),
            "action": "Open table",
        })
    if not records:
        insights.append({
            "title": "No records yet",
            "body": "Create records before reading analytics.",
            "url": _ui_entity_query_path(entity_name),
            "action": f"Create {entity_name}",
        })
    peak_point = max(line_data, key=lambda item: item["y"], default={"x": "", "y": 0})
    recent_average = round(sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:])), 2) if line_data else 0
    analytics_decisions = [
        {
            "label": "Annotation Pin",
            "value": str(peak_point["x"] or "No peak yet"),
            "hint": f"Highest daily volume: {peak_point['y']}",
            "url": _ui_entity_query_path(entity_name),
        },
        {
            "label": "Comparative Overlay",
            "value": "Current vs prior window",
            "hint": f"Recent average {recent_average} record(s)/day",
            "url": _ui_entity_query_path(entity_name),
        },
        {
            "label": "Forecast Band",
            "value": "Next 7 days",
            "hint": f"Expected {recent_average} to {round(recent_average + 1, 2)} per day",
            "url": _ui_entity_query_path(entity_name),
        },
    ]
    tmpl_body = _render_template(
        "entity_analytics.html.j2",
        entity_name=entity_name,
        safe_entity=safe_entity,
        total=len(records),
        metrics=metrics,
        status_field=status_field or "",
        status_rows=status_rows,
        date_field=date_field,
        date_range=date_range,
        insights=insights,
        line_chart=line_chart,
        status_chart=status_chart,
        analytics_decisions=analytics_decisions,
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
    _publish_live_event(
        f"workflow:run:{workflow_id}",
        "workflow",
        {"workflow": workflow_id, "entity": entity_name, "step_index": step_index, "next_step": next_step},
    )
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

    # Pick a good display title (preferred name field, first string value, or id prefix)
    preferred_title_names = ("legal_name", "full_name", "name", "title", "subject", "number", "code")
    title_field = next(
        (
            f for preferred in preferred_title_names
            for f in fields
            if str(f.get("name", "")).lower() == preferred
        ),
        None,
    )
    if title_field is None:
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
            rel_records = rel_result.get("records", [])
            rel_cols = ["id"] + [str(f["name"]) for f in ent_fields if str(f.get("name")) not in {"id", "_revision", fk_name}][:4]
            related_lists.append({
                "entity": ent,
                "fk_field": fk_name,
                "records": rel_records,
                "count": len(rel_records),
                "cols": rel_cols,
                "list_url": _ui_entity_query_path(ent, updates={f"filter.{fk_name}": record_id}),
                "create_url": _ui_entity_query_path(ent),
            })

    has_kanban = any(str(f.get("name", "")).lower() in {"status", "state", "stage", "phase"} for f in fields)
    revision = html.escape(str(record.get("_revision", "")))
    entity_records = list_records(entity_name)
    record_ids = [str(item.get("id", "")) for item in entity_records if item.get("id", "") not in (None, "")]
    try:
        current_index = record_ids.index(str(record_id))
    except ValueError:
        current_index = -1
    prev_record_url = ""
    next_record_url = ""
    if current_index > 0:
        prev_record_url = f"/ui/entities/{safe_entity}/{quote(record_ids[current_index - 1], safe='')}"
    if current_index >= 0 and current_index < len(record_ids) - 1:
        next_record_url = f"/ui/entities/{safe_entity}/{quote(record_ids[current_index + 1], safe='')}"
    related_count = sum(int(rel.get("count", 0)) for rel in related_lists)
    record_url = f"/ui/entities/{safe_entity}/{safe_record_id}"

    display_fields = [f for f in fields if str(f.get("name")) != "_revision"]
    field_semantics = {
        str(f.get("name", "")): _ui_field_semantic(str(f.get("name", "")), str(f.get("type", "")))
        for f in display_fields
    }
    activity_events = _get_activity(entity_name, record_id)
    diff_fields: list[Dict[str, str]] = []
    for field in display_fields:
        field_name = str(field.get("name", ""))
        if field_name == "id":
            continue
        value = record.get(field_name, "")
        if value in (None, "", []):
            continue
        diff_fields.append({
            "name": field_name.replace("_", " ").title(),
            "value": html.escape(str(value)[:72]),
            "state": "current",
        })
        if len(diff_fields) >= 4:
            break
    sibling_fields = [
        {
            "name": str(field.get("name", "")).replace("_", " ").title(),
            "value": html.escape(str(record.get(str(field.get("name", "")), ""))[:48]),
        }
        for field in display_fields
        if str(field.get("name", "")) not in {"id", "_revision"} and not str(field.get("name", "")).endswith("_id")
    ][:3]
    related_graph = [
        {
            "entity": html.escape(str(rel.get("entity", ""))),
            "count": int(rel.get("count", 0)),
            "field": html.escape(str(rel.get("fk_field", ""))),
            "url": str(rel.get("list_url", "")),
        }
        for rel in related_lists[:4]
    ]
    detail_intelligence = {
        "diff_fields": diff_fields,
        "related_graph": related_graph,
        "sibling_fields": sibling_fields,
        "activity_count": len(activity_events),
        "create_sibling_url": _ui_entity_query_path(entity_name),
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
        related_count=related_count,
        prev_record_url=prev_record_url,
        next_record_url=next_record_url,
        record_url=record_url,
        has_kanban=has_kanban,
        activity_events=activity_events,
        detail_intelligence=detail_intelligence,
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
    field_semantic = _ui_field_semantic(field_name, field_type)
    field_expected = _json_schema_type(field_type)
    if field_type.lower() in {"text", "markdown", "list", "dict", "json", "jsonb"} or field_expected in {"array", "object"}:
        input_html = (
            f'<textarea name="{safe_field_name}" rows="3"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary resize-none">'
            f'{current_val}</textarea>'
        )
    elif field_type == "boolean":
        checked = "checked" if str(record.get(field_name, "")).lower() == "true" else ""
        input_html = f'<input type="checkbox" name="{safe_field_name}" value="true" {checked} class="w-4 h-4 text-apg-primary rounded">'
    elif field_expected == "integer":
        input_html = (
            f'<input type="number" step="1" name="{safe_field_name}" value="{current_val}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_expected == "number":
        input_html = (
            f'<input type="number" step="any" name="{safe_field_name}" value="{current_val}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_type.lower() in {"date", "datetime", "timestamp"}:
        input_html = (
            f'<input type="date" name="{safe_field_name}" value="{current_val[:10]}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_semantic == "email":
        input_type = "email"
        input_html = (
            f'<input type="{input_type}" name="{safe_field_name}" value="{current_val}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_semantic == "phone":
        input_html = (
            f'<input type="tel" name="{safe_field_name}" value="{current_val}"'
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
    wip_limit = max(3, (len(all_records) + max(1, len(seen)) - 1) // max(1, len(seen))) if all_records else 3
    columns = []
    for value in seen:
        column_records = [r for r in all_records if str(r.get(status_fname, "")) == value]
        columns.append({
            "label": value,
            "records": column_records,
            "count": len(column_records),
            "wip_limit": wip_limit,
            "over_limit": len(column_records) > wip_limit,
            "list_url": _ui_entity_query_path(entity_name, updates={f"filter.{status_fname}": value}),
        })
    swimlane_field = next(
        (
            str(f.get("name"))
            for f in fields
            if str(f.get("name", "")).lower() in {"priority", "assignee", "owner", "team", "country", "tenant_id", "segment", "type"}
            and str(f.get("name")) not in {"id", "_revision", status_fname}
        ),
        "",
    )
    swimlanes: list[Dict[str, Any]] = []
    if swimlane_field:
        lane_values = sorted({
            str(record.get(swimlane_field) or "Unassigned")
            for record in all_records
        })
        for lane in lane_values[:6]:
            lane_records = [record for record in all_records if str(record.get(swimlane_field) or "Unassigned") == lane]
            swimlanes.append({
                "label": lane,
                "count": len(lane_records),
                "field": swimlane_field,
                "url": _ui_entity_query_path(entity_name, updates={f"filter.{swimlane_field}": lane}),
            })
    cumulative = 0
    flow_rows = []
    for column in columns:
        cumulative += int(column["count"])
        flow_rows.append({
            "label": column["label"],
            "count": column["count"],
            "cumulative": cumulative,
            "percent": round((cumulative / max(1, len(all_records))) * 100, 1),
            "over_limit": column["over_limit"],
        })
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
        status_options=seen,
        total_records=len(all_records),
        wip_limit=wip_limit,
        swimlane_field=swimlane_field,
        swimlanes=swimlanes,
        flow_rows=flow_rows,
        list_url=_ui_entity_query_path(entity_name),
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
            journal = _get_journal(run_id)
            trace = [
                {
                    "index": str(step.get("index", "")),
                    "step": str(step.get("step", "")),
                    "status": str(step.get("status", "")),
                    "notes": str(step.get("notes") or step.get("timeout_spec", "")),
                    "field_count": step.get("field_count", ""),
                    "duration_ms": step.get("duration_ms", ""),
                    "fields": ", ".join(str(item) for item in step.get("fields", [])) if isinstance(step.get("fields", []), list) else "",
                    "badge_class": _badge(str(step.get("status", ""))),
                }
                for step in raw_run.get("trace", [])
                if isinstance(step, dict)
            ]
            selected_run = {
                "id": str(raw_run.get("id", run_id)),
                "workflow": str(raw_run.get("workflow", "")),
                "workflow_id": str(raw_run.get("workflow_id", "")),
                "entity": str(raw_run.get("entity", "")),
                "status": str(raw_run.get("status", "")),
                "badge_class": _badge(str(raw_run.get("status", ""))),
                "created_record_id": str(raw_run.get("created_record_id", "")),
                "event_id": str(raw_run.get("event_id", "")),
                "trace": trace,
                "journal": [
                    {
                        "seq": str(event.get("seq", "")),
                        "event_type": str(event.get("event_type", "")),
                        "step": str(event.get("step", "")),
                        "ts": str(event.get("ts", "")),
                        "data": event.get("data", {}),
                        "data_json": json.dumps(event.get("data", {}), indent=2, sort_keys=True),
                    }
                    for event in journal
                    if isinstance(event, dict)
                ],
                "payload_json": json.dumps(raw_run.get("payload", {}), indent=2, sort_keys=True),
                "record_json": json.dumps(raw_run.get("record", {}), indent=2, sort_keys=True),
                "step_count": len(trace),
                "event_count": len(journal),
                "duration_ms": sum(
                    int(step.get("duration_ms", 0))
                    for step in raw_run.get("trace", [])
                    if isinstance(step, dict) and str(step.get("duration_ms", "")).isdigit()
                ),
            }
    run_items = [
        {
            "id": str(run.get("id", "")),
            "workflow": str(run.get("workflow", "")),
            "entity": str(run.get("entity", "")),
            "status": str(run.get("status", "")),
            "badge_class": _badge(str(run.get("status", ""))),
            "step_count": len(run.get("trace", [])),
            "created_record_id": str(run.get("created_record_id", "")),
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
        cards = _marketplace_cards(connectors)
        q = (query or {}).get("q", [""])[0].strip() if query else ""
        active_category = (query or {}).get("category", ["all"])[0].strip() if query else "all"
        categories: list[Dict[str, Any]] = []
        for category_name in sorted({str(card["category"]) for card in cards}):
            count = len([card for card in cards if card["category"] == category_name])
            categories.append({"name": category_name, "count": count, "active": category_name == active_category})
        filtered_cards = cards
        if active_category and active_category != "all":
            filtered_cards = [card for card in filtered_cards if card["category"] == active_category]
        if q:
            q_lower = q.lower()
            filtered_cards = [
                card for card in filtered_cards
                if q_lower in card["title"].lower()
                or q_lower in card["description"].lower()
                or q_lower in card["category"].lower()
            ]
        tmpl_body = _render_template("marketplace.html.j2",
            connectors=filtered_cards,
            connector_count=len(cards),
            filtered_count=len(filtered_cards),
            installed_count=len(connectors),
            categories=categories,
            active_category=active_category or "all",
            query=q,
            has_filters=bool(q or (active_category and active_category != "all")),
            has_installed_connectors=bool(connectors),
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


def _sanitize_agent_markdown(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)
    escaped = html.escape(str(text), quote=True)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    lines = escaped.splitlines() or [""]
    html_lines: list[str] = []
    in_list = False
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append("<li>" + line[2:] + "</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(line)
    if in_list:
        html_lines.append("</ul>")
    return "<br>".join(html_lines)


def _agent_display_text(result: Dict[str, Any] | None) -> Any:
    if not isinstance(result, dict):
        return ""
    for key in ("output", "response", "message", "text", "content"):
        if key in result:
            return result[key]
    return result


def _agent_result_status(result: Dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("status") or result.get("state") or "")


def _agent_status_badge(status: str) -> str:
    normalized = status.lower()
    if normalized in {"completed", "success", "ok"}:
        return "apg-badge-success"
    if normalized in {"failed", "error", "unavailable"}:
        return "apg-badge-danger"
    return "apg-badge-warning" if normalized else "apg-badge-neutral"


def _ui_agent_console_html(
    name: str,
    result: Dict[str, Any] | None = None,
    error: str = "",
    team: bool = False,
    request_payload: Dict[str, Any] | None = None,
    user_message: str = "",
) -> tuple[int, str]:
    app = describe_application()
    catalog_key = "ai_agent_team_descriptions" if team else "ai_agent_descriptions"
    catalog = app.get(catalog_key, {})
    if name not in catalog:
        title = "Unknown agent team" if team else "Unknown agent"
        return 404, _html_page(title, f"<h1>{title}</h1><p>{html.escape(name)}</p>")
    action = f"/ui/{'agent-teams' if team else 'agents'}/{html.escape(name, quote=True)}/invoke"
    description = catalog[name]
    request_payload = dict(request_payload or {})
    result_status = _agent_result_status(result)
    team_members = list(description.get("agents", [])) if team and isinstance(description, dict) else []
    team_flow = list(description.get("flow", [])) if team and isinstance(description, dict) else []
    tmpl_body = _render_template(
        "agent_console.html.j2",
        name=name,
        team=team,
        action=action,
        description=description,
        description_json=json.dumps(description, indent=2, sort_keys=True),
        result=result,
        result_json=json.dumps(result, indent=2, sort_keys=True) if result is not None else "",
        result_html=_sanitize_agent_markdown(_agent_display_text(result)) if result is not None else "",
        result_status=result_status,
        result_badge_class=_agent_status_badge(result_status),
        error=error,
        user_message=user_message,
        payload_json=json.dumps(request_payload, indent=2, sort_keys=True) if request_payload else "{}",
        team_members=team_members,
        team_flow=team_flow,
        live_topic=f"agent:{name}",
    )
    return 200, _html_page(name, tmpl_body if tmpl_body is not None else _jinja_required_page(name))


def _capability_default_rule_context(description: Dict[str, Any]) -> Dict[str, Any]:
    configuration = description.get("configuration", {}) if isinstance(description, dict) else {}
    default_limit = configuration.get("default_limit", 1000) if isinstance(configuration, dict) else 1000
    review_threshold = configuration.get("review_threshold", 0.5) if isinstance(configuration, dict) else 0.5
    return {
        "tenant_id": "example-tenant",
        "customer_id": "customer-001",
        "amount": default_limit,
        "risk_score": review_threshold,
        "is_international": False,
    }


def _capability_default_approval_context(description: Dict[str, Any]) -> Dict[str, Any]:
    context = _capability_default_rule_context(description)
    context["requester"] = "operator"
    return context


def _capability_operation_label(operation: str) -> str:
    labels = {
        "rules": "Rules evaluation",
        "configuration": "Configuration resolution",
        "approval": "Approval plan",
    }
    return labels.get(operation, "Result")


def _ui_capability_console_html(
    name: str,
    result: Dict[str, Any] | None = None,
    error: str = "",
    operation: str = "",
    context_json: str = "",
    configuration_json: str = "",
    approval_context_json: str = "",
) -> tuple[int, str]:
    app = describe_application()
    capabilities = app.get("capability_descriptions", {})
    if name not in capabilities:
        return 404, _html_page("Unknown capability", f"<h1>Unknown capability</h1><p>{html.escape(name)}</p>")
    safe_name = html.escape(name, quote=True)
    description = capabilities[name]
    default_rule_context = _capability_default_rule_context(description)
    default_approval_context = _capability_default_approval_context(description)
    default_configuration = description.get("configuration", {}) if isinstance(description, dict) else {}
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
        description=description,
        description_json=json.dumps(description, indent=2, sort_keys=True),
        rule_context_json=context_json or json.dumps(default_rule_context, indent=2, sort_keys=True),
        configuration_json=configuration_json or json.dumps(default_configuration, indent=2, sort_keys=True),
        approval_context_json=approval_context_json or json.dumps(default_approval_context, indent=2, sort_keys=True),
        operation=operation,
        operation_label=_capability_operation_label(operation),
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
        message = str(form_record.get("message") or "")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, request_payload={}, user_message=message)
            return 400, {"html": html_payload}
        if message:
            request_payload["message"] = message
        if str(form_record.get("stream", "")).lower() in {"1", "true", "yes", "on"}:
            request_payload["stream"] = True
        status, result = _agent_invocation_payload(f"/agents/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(
            parts[2],
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "agent invocation failed"),
            request_payload=request_payload,
            user_message=message,
        )
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] in {"agent-teams", "teams"} and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        message = str(form_record.get("message") or "")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, team=True, request_payload={}, user_message=message)
            return 400, {"html": html_payload}
        if message:
            request_payload["message"] = message
        if str(form_record.get("stream", "")).lower() in {"1", "true", "yes", "on"}:
            request_payload["stream"] = True
        status, result = _agent_invocation_payload(f"/agent-teams/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(
            parts[2],
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "team invocation failed"),
            team=True,
            request_payload=request_payload,
            user_message=message,
        )
        return status, {"html": html_payload}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "capabilities":
        capability_name = parts[2]
        operation = "/".join(parts[3:])
        if operation == "rules/evaluate":
            raw_context_json = str(form_record.get("context_json") or "")
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="rules", context_json=raw_context_json)
                return 400, {"html": html_payload}
            status, result = _rule_evaluation_payload(f"/capabilities/{capability_name}/rules/evaluate", {"context": context})
        elif operation == "configuration/resolve":
            raw_configuration_json = str(form_record.get("configuration_json") or "")
            configuration, error = _parse_json_object_field(form_record, "configuration_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="configuration", configuration_json=raw_configuration_json)
                return 400, {"html": html_payload}
            status, result = _configuration_payload(f"/capabilities/{capability_name}/configuration/resolve", {"overrides": configuration})
        elif operation == "approval/plan":
            raw_approval_context_json = str(form_record.get("context_json") or "")
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="approval", approval_context_json=raw_approval_context_json)
                return 400, {"html": html_payload}
            status, result = _approval_plan_payload(f"/capabilities/{capability_name}/approval/plan", {"context": context})
        else:
            return 404, {"error": "not_found", "path": path}
        op_key = "rules" if operation == "rules/evaluate" else "configuration" if operation == "configuration/resolve" else "approval"
        _status, html_payload = _ui_capability_console_html(
            capability_name,
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "capability operation failed"),
            operation=op_key,
            context_json=raw_context_json if operation == "rules/evaluate" else "",
            configuration_json=raw_configuration_json if operation == "configuration/resolve" else "",
            approval_context_json=raw_approval_context_json if operation == "approval/plan" else "",
        )
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        status, response = _create_record_payload(f"/entities/{entity_name}/records", payload)
        if status == 201:
            return 303, {"location": _ui_entity_location(entity_name)}
        _detail = response.get("errors") or response.get("message") or response.get("error") or "Record could not be created."
        if isinstance(_detail, list):
            _detail = "; ".join(str(item) for item in _detail)
        _page_status, html_payload = _ui_entity_html(entity_name, notice=str(_detail))
        return status, {"html": html_payload}
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
                _ = None  # best-effort
        return 303, {"location": _ui_entity_location(entity_name)}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        record_id = parts[4]
        expected_revision = form_record.pop("expected_revision", None)
        return_view = form_record.pop("return_view", "")
        status, response = _update_record_payload(
            f"/entities/{entity_name}/records/{record_id}",
            {"record": form_record, "expected_revision": expected_revision},
        )
        if status == 200:
            if return_view == "kanban":
                return 303, {"location": _ui_entity_location(entity_name) + "?view=kanban"}
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
    parts = [part for part in path.split("/") if part]
    if AI_AGENTS is None:
        if len(parts) == 3 and parts[0] in {"agent-teams", "teams"} and parts[2] in {"invoke", "run"}:
            team_description = _entity_agent_team_descriptions().get(parts[1])
            if team_description is not None:
                return 200, {
                    "team": parts[1],
                    "status": "unavailable",
                    "error": "agents_unavailable",
                    "source": "entity_metadata",
                    "flow": team_description.get("flow", []),
                    "invocations": [
                        {"agent": str(agent_name), "status": "unavailable", "error": "agents_unavailable"}
                        for agent_name in team_description.get("agents", [])
                    ],
                }
        return 404, {"error": "agents_unavailable"}
    try:
        if len(parts) == 3 and parts[0] == "agents" and parts[2] in {"invoke", "run"}:
            topic = f"agent:{parts[1]}"
            _publish_live_event(topic, "agent-token", {"status": "started", "token": ""})
            if payload.get("stream"):
                streamer = getattr(AI_AGENTS, "stream_agent", None)
                if streamer is not None:
                    chunks: list[str] = []
                    for chunk in streamer(parts[1], payload):
                        token = chunk.get("token", "") if isinstance(chunk, dict) else str(chunk)
                        if token:
                            chunks.append(token)
                            _publish_live_event(topic, "agent-token", {"token": token})
                    result = {"agent": parts[1], "status": "completed", "output": "".join(chunks), "streamed": True}
                    _publish_live_event(topic, "agent-result", result)
                    return 200, result
            invoker = getattr(AI_AGENTS, "invoke_agent", None)
            if invoker is None:
                return 404, {"error": "agent_invocation_unavailable"}
            result = invoker(parts[1], payload)
            _publish_live_event(topic, "agent-result", result if isinstance(result, dict) else {"output": result})
            return 200, result
        if len(parts) == 3 and parts[0] in {"agent-teams", "teams"} and parts[2] in {"invoke", "run"}:
            topic = f"agent:{parts[1]}"
            _publish_live_event(topic, "agent-token", {"status": "started", "token": ""})
            invoker = getattr(AI_AGENTS, "invoke_team", None)
            if invoker is None:
                return 404, {"error": "team_invocation_unavailable"}
            try:
                result = invoker(parts[1], payload)
            except KeyError:
                team_description = _entity_agent_team_descriptions().get(parts[1])
                if team_description is None:
                    raise
                invocations = []
                for agent_name in team_description.get("agents", []):
                    agent_status, agent_result = _agent_invocation_payload(f"/agents/{quote(str(agent_name), safe='')}/invoke", payload)
                    invocations.append(agent_result if isinstance(agent_result, dict) else {"output": agent_result, "status": agent_status})
                if any(str(item.get("status", "")).lower() in {"failed", "error"} for item in invocations if isinstance(item, dict)):
                    team_status = "failed"
                elif any(str(item.get("status", "")).lower() == "adapter_required" for item in invocations if isinstance(item, dict)):
                    team_status = "adapter_required"
                else:
                    team_status = "completed"
                result = {
                    "team": parts[1],
                    "status": team_status,
                    "source": "entity_metadata",
                    "flow": team_description.get("flow", []),
                    "invocations": invocations,
                }
            _publish_live_event(topic, "agent-result", result if isinstance(result, dict) else {"output": result})
            return 200, result
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
        _ = None  # best-effort


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
        _ = None  # best-effort
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
        _ = None  # best-effort


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
        _ = None  # best-effort
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
_flask_app.secret_key = os.environ.get("APG_SESSION_SECRET") or os.environ.get("APG_JWT_SECRET") or "apg-generated-session-secret"


@_flask_app.before_request
def _setup_tenant() -> Any:
    tid = _flask_request.headers.get("X-APG-Tenant") or _flask_request.headers.get("X-Tenant-ID")
    _TENANT_LOCAL.tenant_id = tid or None
    if _login_required_for_path(_flask_request.path) and _current_user() is None:
        return _flask_redirect("/login?next=" + quote(_flask_request.full_path.rstrip("?") or "/ui", safe="/?=&%"))
    return None


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


@_flask_app.route("/login", methods=["GET"])
def _flask_login_get():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({"error": "not_found", "path": "/login"}), status=404, content_type="application/json; charset=utf-8")
    next_url = _flask_request.args.get("next") or "/ui"
    if not str(next_url).startswith("/"):
        next_url = "/ui"
    if _current_user() is not None:
        return _flask_redirect(next_url)
    return _FlaskResponse(_login_page(next_url=next_url), content_type="text/html; charset=utf-8")


@_flask_app.route("/login", methods=["POST"])
def _flask_login_post():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({"error": "not_found", "path": "/login"}), status=404, content_type="application/json; charset=utf-8")
    username = str(_flask_request.form.get("username") or "")
    password = str(_flask_request.form.get("password") or "")
    next_url = str(_flask_request.form.get("next") or "/ui")
    if not next_url.startswith("/"):
        next_url = "/ui"
    user = _authenticate_user(username, password)
    if user is None:
        return _FlaskResponse(
            _login_page("We could not sign you in with those credentials.", next_url, username=username),
            status=401,
            content_type="text/html; charset=utf-8",
        )
    _issue_login_session(user)
    return _flask_redirect(next_url)


@_flask_app.route("/logout", methods=["POST"])
def _flask_logout_post():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({"error": "not_found", "path": "/logout"}), status=404, content_type="application/json; charset=utf-8")
    _flask_session.pop("apg_user", None)
    return _flask_redirect("/login")


@_flask_app.route("/locale", methods=["POST"])
def _flask_locale_post():
    language = str(_flask_request.form.get("lang") or APG_DEFAULT_LANGUAGE)
    if language not in APG_SUPPORTED_LANGUAGES:
        language = APG_DEFAULT_LANGUAGE
    next_url = str(_flask_request.form.get("next") or "/ui")
    if not next_url.startswith("/"):
        next_url = "/ui"
    response = _flask_redirect(next_url)
    response.set_cookie("apg_lang", language, max_age=31536000, samesite="Lax")
    return response


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
    if path == "/events" and (
        "text/event-stream" in (_flask_request.headers.get("Accept") or "")
        or _flask_request.args.get("topics") is not None
    ):
        return _FlaskResponse(
            _sse_stream(_flask_request.args.get("topics")),
            content_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
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
