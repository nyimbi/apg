"""
main - APG Python Application
=============================

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import importlib
import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote


MODULE_NAME = 'main'
MODULE_VERSION = '1.0.0'
MODULE_DESCRIPTION = None
LANDING_STYLE = 'default'
ENTITIES = [{'name': 'FreelanceHub', 'type': 'marketplace', 'properties': ['description', 'version', 'own_data', 'freelancer_profiles', 'project_data', 'payment_history', 'email', 'phone', 'payment_method', 'identity', 'completion_incentive', 'layout', 'payment_methods', 'payment_processing_fee', 'platform_fee', 'own_data_1', 'client_profiles', 'project_data_1', 'earnings_data', 'email_1', 'phone_1', 'identity_1', 'professional_credentials', 'background_check', 'portfolio_review', 'profile_completeness', 'skills_tests_passed', 'portfolio_items', 'layout_1', 'payment_methods_1', 'platform_fee_1', 'payment_processing_fee_1', 'withdrawal_fee', 'all_data', 'financial_data', 'user_communications', 'employee_verification', 'background_check_1', 'two_factor_auth', 'layout_2', 'user_content', 'reported_content', 'user_communications_1', 'employee_verification_1', 'background_check_2', 'content_moderation_training', 'layout_3', 'escrow_enabled', 'name', 'api_key', 'webhook_secret', 'connect_enabled', 'supported_methods', 'supported_currencies', 'name_1', 'client_id', 'client_secret', 'mode', 'supported_methods_1', 'supported_currencies_1', 'name_2', 'supported_coins', 'wallet_provider', 'confirmation_blocks', 'supported_currencies_2', 'payment_processing', 'platform_fee_2', 'platform_commission', 'payment_processing_1', 'bank_transfer', 'paypal', 'cryptocurrency', 'currency_conversion', 'auto_resolution_enabled', 'stages', 'timeouts', 'communication_logs', 'work_samples', 'time_tracking', 'third_party_validation', 'evidence_review_period', 'decision_timeline', 'enabled', 'scenario', 'freelancer', 'platform', 'scenario_1', 'lead_freelancer', 'team_members', 'platform_1', 'scenario_2', 'freelancer_1', 'referring_user', 'platform_2', 'ml_models', 'low_risk', 'medium_risk', 'high_risk', 'critical_risk', 'requirements', 'verification_methods', 'requirements_1', 'verification_methods_1', 'requirements_2', 'verification_methods_2', 'additional_checks', 'transaction_volume', 'project_completion_rate', 'positive_reviews', 'scale', 'recent_reviews_weight', 'older_reviews_weight', 'minimum_reviews', 'reviewer_reliability', 'review_authenticity_check', 'bias_detection', 'timing', 'grace_period', 'mutual_review', 'anonymous_option', 'minimum_length', 'profanity_filter', 'spam_detection', 'authenticity_verification', 'review_completion_bonus', 'detailed_review_bonus', 'profanity_detection', 'hate_speech_detection', 'spam_detection_1', 'personal_info_detection', 'inappropriate_content', 'copyright_infringement', 'personal_documents', 'flagged_content_review', 'user_reports', 'automated_confidence', 'review_sla', 'two_factor_auth_1', 'login_anomaly_detection', 'device_tracking', 'session_management', 'unusual_activity_patterns', 'velocity_checks', 'cross_platform_correlation', 'models', 'name_3', 'data_protection', 'consent_management', 'data_portability', 'right_to_deletion', 'name_4', 'privacy_disclosures', 'opt_out_mechanisms', 'data_sale_restrictions', 'name_5', 'secure_payment_processing', 'data_encryption', 'access_controls', 'regular_security_testing', 'enabled_1', 'providers', 'criminal_background', 'professional_verification', 'education_verification', 'employment_history', 'refresh_frequency', 'provider', 'version_1', 'cluster_name', 'nodes', 'shards', 'replicas', 'type', 'rating', 'completion_rate', 'response_time', 'recent_activity', 'type_1', 'budget', 'urgency', 'client_rating', 'update_frequency', 'similarity_metric', 'neighbors', 'weight', 'similarity_metric_1', 'weight_1', 'feature_extraction', 'similarity_threshold', 'weight_2', 'collaborative', 'content_based', 'popularity', 'trending', 'implicit_feedback', 'explicit_feedback', 'decay_factor', 'enabled_2', 'precision', 'default', 'remote_work_preference', 'local_only', 'timezone_matching', 'name_6', 'type_2', 'aggregation', 'size', 'name_7', 'type_3', 'name_8', 'type_4', 'options', 'name_9', 'type_5', 'options_1', 'name_10', 'type_6', 'levels', 'enabled_3', 'sources', 'suggestions_count', 'fuzzy_matching', 'typo_tolerance', 'search_queries', 'result_clicks', 'zero_results', 'abandonment_rate', 'query_expansion', 'result_reranking', 'a_b_testing', 'platform_3', 'real_time_chat', 'file_sharing', 'video_calls', 'screen_sharing', 'message_encryption', 'message_history', 'search_messages', 'participants', 'encryption', 'participants_1', 'features', 'participants_2', 'features_1', 'automated_filtering', 'reporting_system', 'admin_override', 'type_7', 'platforms', 'urgency_levels', 'type_8', 'templates', 'personalization', 'type_9', 'use_cases', 'international', 'user_customizable', 'frequency_controls', 'technology', 'typing_indicators', 'read_receipts', 'message_reactions', 'thread_replies', 'connection_limit', 'horizontal_scaling', 'load_balancing', 'provider_1', 'one_on_one', 'group_calls', 'screen_sharing_1', 'recording', 'max_resolution', 'adaptive_bitrate', 'noise_suppression', 'stages_1', 'auto_expiry', 'stages_2', 'history_tracking', 'proposal_templates', 'auto_calculations', 'comparison_tools', 'deadline_tracking', 'name_11', 'type_10', 'path', 'methods', 'authentication', 'path_1', 'methods_1', 'rate_limit', 'type_11', 'name_12', 'tables', 'dependencies', 'strategy', 'min_instances', 'max_instances', 'cpu_threshold', 'memory_threshold', 'image', 'cpu', 'memory', 'type_12', 'constraints', 'preferences', 'name_13', 'type_13', 'path_2', 'methods_2', 'caching', 'path_3', 'methods_3', 'async_processing', 'type_14', 'name_14', 'tables_1', 'dependencies_1', 'strategy_1', 'min_instances_1', 'max_instances_1', 'cpu_threshold_1', 'custom_metrics', 'name_15', 'type_15', 'path_4', 'methods_4', 'security', 'encryption_1', 'type_16', 'name_16', 'encryption_2', 'backup_frequency', 'dependencies_2', 'strategy_2', 'instances', 'vault_integration', 'secret_rotation', 'audit_logging', 'name_17', 'type_17', 'path_5', 'methods_5', 'caching_1', 'type_18', 'cluster', 'indices', 'dependencies_3', 'strategy_3', 'min_instances_2', 'max_instances_2', 'cpu_threshold_2', 'name_18', 'type_19', 'path_6', 'methods_6', 'websocket', 'type_20', 'name_19', 'collections', 'dependencies_4', 'strategy_4', 'min_instances_3', 'max_instances_3', 'connection_based', 'name_20', 'type_21', 'path_7', 'methods_7', 'rate_limit_1', 'type_22', 'name_21', 'partitioning', 'dependencies_5', 'strategy_5', 'min_instances_4', 'max_instances_4', 'schedule_based', 'provider_2', 'rate_limiting', 'authentication_1', 'request_transformation', 'response_caching', 'load_balancing_1', 'versioning', 'ssl_termination', 'cors_handling', 'provider_3', 'traffic_management', 'security_policies', 'observability', 'circuit_breaking', 'mutual_tls', 'authorization', 'traffic_splitting', 'provider_4', 'health_checking', 'service_registration', 'dns_interface', 'default_1', 'cpu_intensive', 'session_sticky', 'interval', 'timeout', 'retries', 'failure_threshold', 'recovery_timeout', 'half_open_max_calls', 'provider_5', 'sampling_rate', 'trace_retention', 'method', 'fallback', 'quality_assurance', 'date_formats', 'number_formats', 'address_formats', 'payment_methods_2', 'provider_6', 'update_frequency_1', 'caching_2', 'fallback_rates', 'providers_1', 'vat_handling', 'sales_tax', 'gst_handling', 'cost_calculation', 'delivery_options', 'cost_calculation_1', 'customs_handling', 'duty_calculation', 'provider_7', 'compute_clusters', 'storage_size', 'auto_scaling', 'frequency', 'transformation_engine', 'data_quality_checks', 'stream_processing', 'operational', 'business', 'executive', 'tools', 'metrics', 'frequency_1', 'stakeholders', 'metrics_1', 'frequency_2', 'stakeholders_1', 'metrics_2', 'frequency_3', 'stakeholders_2', 'name_22', 'type_23', 'features_2', 'update_frequency_2', 'name_23', 'type_24', 'features_3', 'update_frequency_3', 'name_24', 'type_25', 'features_4', 'update_frequency_4', 'training_platform', 'model_serving', 'feature_store', 'platform_4', 'significance_threshold', 'minimum_sample_size', 'test_duration', 'name_25', 'steps', 'name_26', 'steps_1', 'models_1', 'lookback_window', 'metric', 'method_1', 'forecast_horizon', 'metric_1', 'method_2', 'forecast_horizon_1', 'user_id', 'user_type', 'registration_source', 'timestamp', 'triggers', 'exchange', 'routing_key', 'project_id', 'client_id_1', 'project_type', 'budget_range', 'skills_required', 'timestamp_1', 'triggers_1', 'transaction_id', 'payer_id', 'recipient_id', 'amount', 'currency', 'timestamp_2', 'triggers_2', 'dispute_id', 'project_id_1', 'initiator_id', 'dispute_type', 'severity', 'timestamp_3', 'triggers_3'], 'fields': [{'name': 'description', 'type': '"Professional freelance services marketplace connecting clients with service providers"', 'required': True}, {'name': 'version', 'type': '"3.2.0"', 'required': True}, {'name': 'own_data', 'type': '"full_access"', 'required': True}, {'name': 'freelancer_profiles', 'type': '"public_only"', 'required': True}, {'name': 'project_data', 'type': '"own_projects_only"', 'required': True}, {'name': 'payment_history', 'type': '"own_transactions"', 'required': True}, {'name': 'email', 'type': 'true', 'required': True}, {'name': 'phone', 'type': 'true', 'required': True}, {'name': 'payment_method', 'type': 'true', 'required': True}, {'name': 'identity', 'type': 'false', 'required': True}, {'name': 'completion_incentive', 'type': '"$50 credit for first project"', 'required': True}, {'name': 'layout', 'type': '"client_dashboard"', 'required': True}, {'name': 'payment_methods', 'type': '["credit_card", "bank_transfer", "paypal", "cryptocurrency"]', 'required': True}, {'name': 'payment_processing_fee', 'type': '2.9%', 'required': True}, {'name': 'platform_fee', 'type': '0%', 'required': True}, {'name': 'own_data_1', 'type': '"full_access"', 'required': True}, {'name': 'client_profiles', 'type': '"basic_info_only"', 'required': True}, {'name': 'project_data_1', 'type': '"applied_projects_and_hired"', 'required': True}, {'name': 'earnings_data', 'type': '"own_earnings_only"', 'required': True}, {'name': 'email_1', 'type': 'true', 'required': True}, {'name': 'phone_1', 'type': 'true', 'required': True}, {'name': 'identity_1', 'type': 'true', 'required': True}, {'name': 'professional_credentials', 'type': 'true', 'required': True}, {'name': 'background_check', 'type': '"optional_premium"', 'required': True}, {'name': 'portfolio_review', 'type': 'true', 'required': True}, {'name': 'profile_completeness', 'type': '90%', 'required': True}, {'name': 'skills_tests_passed', 'type': '3', 'required': True}, {'name': 'portfolio_items', 'type': '5', 'required': True}, {'name': 'layout_1', 'type': '"freelancer_dashboard"', 'required': True}, {'name': 'payment_methods_1', 'type': '["bank_transfer", "paypal", "stripe_connect", "cryptocurrency"]', 'required': True}, {'name': 'platform_fee_1', 'type': '8%', 'required': True}, {'name': 'payment_processing_fee_1', 'type': '2.9%', 'required': True}, {'name': 'withdrawal_fee', 'type': '"$1 per withdrawal"', 'required': True}, {'name': 'all_data', 'type': '"full_access"', 'required': True}, {'name': 'financial_data', 'type': '"full_access"', 'required': True}, {'name': 'user_communications', 'type': '"moderation_access"', 'required': True}, {'name': 'employee_verification', 'type': 'true', 'required': True}, {'name': 'background_check_1', 'type': 'true', 'required': True}, {'name': 'two_factor_auth', 'type': 'true', 'required': True}, {'name': 'layout_2', 'type': '"admin_dashboard"', 'required': True}, {'name': 'user_content', 'type': '"moderation_access"', 'required': True}, {'name': 'reported_content', 'type': '"full_access"', 'required': True}, {'name': 'user_communications_1', 'type': '"flagged_only"', 'required': True}, {'name': 'employee_verification_1', 'type': 'true', 'required': True}, {'name': 'background_check_2', 'type': 'true', 'required': True}, {'name': 'content_moderation_training', 'type': 'true', 'required': True}, {'name': 'layout_3', 'type': '"moderator_dashboard"', 'required': True}, {'name': 'escrow_enabled', 'type': 'true', 'required': True}, {'name': 'name', 'type': '"stripe"', 'required': True}, {'name': 'api_key', 'type': 'env("STRIPE_SECRET_KEY")', 'required': True}, {'name': 'webhook_secret', 'type': 'env("STRIPE_WEBHOOK_SECRET")', 'required': True}, {'name': 'connect_enabled', 'type': 'true', 'required': True}, {'name': 'supported_methods', 'type': '["credit_card", "debit_card", "bank_transfer"]', 'required': True}, {'name': 'supported_currencies', 'type': '["USD", "EUR", "GBP", "AUD", "CAD"]', 'required': True}, {'name': 'name_1', 'type': '"paypal"', 'required': True}, {'name': 'client_id', 'type': 'env("PAYPAL_CLIENT_ID")', 'required': True}, {'name': 'client_secret', 'type': 'env("PAYPAL_CLIENT_SECRET")', 'required': True}, {'name': 'mode', 'type': 'env("PAYPAL_MODE", "sandbox")', 'required': True}, {'name': 'supported_methods_1', 'type': '["paypal_account", "credit_card"]', 'required': True}, {'name': 'supported_currencies_1', 'type': '["USD", "EUR", "GBP"]', 'required': True}, {'name': 'name_2', 'type': '"cryptocurrency"', 'required': True}, {'name': 'supported_coins', 'type': '["BTC", "ETH", "USDC", "USDT"]', 'required': True}, {'name': 'wallet_provider', 'type': '"coinbase_commerce"', 'required': True}, {'name': 'confirmation_blocks', 'type': '3', 'required': True}, {'name': 'supported_currencies_2', 'type': '["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]', 'required': True}, {'name': 'payment_processing', 'type': '2.9% + $0.30', 'required': True}, {'name': 'platform_fee_2', 'type': '0%', 'required': True}, {'name': 'platform_commission', 'type': '8%', 'required': True}, {'name': 'payment_processing_1', 'type': '2.9% + $0.30', 'required': True}, {'name': 'bank_transfer', 'type': '$1.00', 'required': True}, {'name': 'paypal', 'type': '1%', 'required': True}, {'name': 'cryptocurrency', 'type': '2%', 'required': True}, {'name': 'currency_conversion', 'type': '1.5%', 'required': True}, {'name': 'auto_resolution_enabled', 'type': 'true', 'required': True}, {'name': 'stages', 'type': '["automated_review", "peer_mediation", "admin_arbitration"]', 'required': True}, {'name': 'timeouts', 'type': '["48h", "7d", "14d"]', 'required': True}, {'name': 'communication_logs', 'type': '"required"', 'required': True}, {'name': 'work_samples', 'type': '"required"', 'required': True}, {'name': 'time_tracking', 'type': '"optional"', 'required': True}, {'name': 'third_party_validation', 'type': '"optional"', 'required': True}, {'name': 'evidence_review_period', 'type': '7d', 'required': True}, {'name': 'decision_timeline', 'type': '14d', 'required': True}, {'name': 'enabled', 'type': 'true', 'required': True}, {'name': 'scenario', 'type': '"standard_project"', 'required': True}, {'name': 'freelancer', 'type': '92%', 'required': True}, {'name': 'platform', 'type': '8%', 'required': True}, {'name': 'scenario_1', 'type': '"team_project"', 'required': True}, {'name': 'lead_freelancer', 'type': '60%', 'required': True}, {'name': 'team_members', 'type': '32%', 'required': True}, {'name': 'platform_1', 'type': '8%', 'required': True}, {'name': 'scenario_2', 'type': '"referral_bonus"', 'required': True}, {'name': 'freelancer_1', 'type': '90%', 'required': True}, {'name': 'referring_user', 'type': '2%', 'required': True}, {'name': 'platform_2', 'type': '8%', 'required': True}, {'name': 'ml_models', 'type': '["payment_fraud_detector_v2.1", "account_takeover_detector"]', 'required': True}, {'name': 'low_risk', 'type': '"allow"', 'required': True}, {'name': 'medium_risk', 'type': '"additional_verification"', 'required': True}, {'name': 'high_risk', 'type': '"manual_review"', 'required': True}, {'name': 'critical_risk', 'type': '"block_and_investigate"', 'required': True}, {'name': 'requirements', 'type': '["email", "phone"]', 'required': True}, {'name': 'verification_methods', 'type': '["email_link", "sms_code"]', 'required': True}, {'name': 'requirements_1', 'type': '["email", "phone", "government_id"]', 'required': True}, {'name': 'verification_methods_1', 'type': '["jumio_id_verification", "onfido"]', 'required': True}, {'name': 'requirements_2', 'type': '["email", "phone", "government_id", "address_proof"]', 'required': True}, {'name': 'verification_methods_2', 'type': '["jumio_id_verification", "address_verification"]', 'required': True}, {'name': 'additional_checks', 'type': '["professional_license_verification"]', 'required': True}, {'name': 'transaction_volume', 'type': '"$5000"', 'required': True}, {'name': 'project_completion_rate', 'type': '95%', 'required': True}, {'name': 'positive_reviews', 'type': '50', 'required': True}, {'name': 'scale', 'type': '1..5', 'required': True}, {'name': 'recent_reviews_weight', 'type': '0.6', 'required': True}, {'name': 'older_reviews_weight', 'type': '0.4', 'required': True}, {'name': 'minimum_reviews', 'type': '5', 'required': True}, {'name': 'reviewer_reliability', 'type': 'true', 'required': True}, {'name': 'review_authenticity_check', 'type': 'true', 'required': True}, {'name': 'bias_detection', 'type': 'true', 'required': True}, {'name': 'timing', 'type': '"after_project_completion"', 'required': True}, {'name': 'grace_period', 'type': '14d', 'required': True}, {'name': 'mutual_review', 'type': 'true', 'required': True}, {'name': 'anonymous_option', 'type': 'false', 'required': True}, {'name': 'minimum_length', 'type': '50', 'required': True}, {'name': 'profanity_filter', 'type': 'true', 'required': True}, {'name': 'spam_detection', 'type': 'true', 'required': True}, {'name': 'authenticity_verification', 'type': 'true', 'required': True}, {'name': 'review_completion_bonus', 'type': '"$2 credit"', 'required': True}, {'name': 'detailed_review_bonus', 'type': '"$5 credit"', 'required': True}, {'name': 'profanity_detection', 'type': 'true', 'required': True}, {'name': 'hate_speech_detection', 'type': 'true', 'required': True}, {'name': 'spam_detection_1', 'type': 'true', 'required': True}, {'name': 'personal_info_detection', 'type': 'true', 'required': True}, {'name': 'inappropriate_content', 'type': 'true', 'required': True}, {'name': 'copyright_infringement', 'type': 'true', 'required': True}, {'name': 'personal_documents', 'type': 'true', 'required': True}, {'name': 'flagged_content_review', 'type': 'true', 'required': True}, {'name': 'user_reports', 'type': '3', 'required': True}, {'name': 'automated_confidence', 'type': '0.7', 'required': True}, {'name': 'review_sla', 'type': '"4 hours"', 'required': True}, {'name': 'two_factor_auth_1', 'type': '"optional_recommended"', 'required': True}, {'name': 'login_anomaly_detection', 'type': 'true', 'required': True}, {'name': 'device_tracking', 'type': 'true', 'required': True}, {'name': 'session_management', 'type': 'true', 'required': True}, {'name': 'unusual_activity_patterns', 'type': 'true', 'required': True}, {'name': 'velocity_checks', 'type': 'true', 'required': True}, {'name': 'cross_platform_correlation', 'type': 'true', 'required': True}, {'name': 'models', 'type': '["fraud_detection_ensemble_v3.0"]', 'required': True}, {'name': 'name_3', 'type': '"gdpr"', 'required': True}, {'name': 'data_protection', 'type': 'true', 'required': True}, {'name': 'consent_management', 'type': 'true', 'required': True}, {'name': 'data_portability', 'type': 'true', 'required': True}, {'name': 'right_to_deletion', 'type': 'true', 'required': True}, {'name': 'name_4', 'type': '"ccpa"', 'required': True}, {'name': 'privacy_disclosures', 'type': 'true', 'required': True}, {'name': 'opt_out_mechanisms', 'type': 'true', 'required': True}, {'name': 'data_sale_restrictions', 'type': 'true', 'required': True}, {'name': 'name_5', 'type': '"pci_dss"', 'required': True}, {'name': 'secure_payment_processing', 'type': 'true', 'required': True}, {'name': 'data_encryption', 'type': 'true', 'required': True}, {'name': 'access_controls', 'type': 'true', 'required': True}, {'name': 'regular_security_testing', 'type': 'true', 'required': True}, {'name': 'enabled_1', 'type': 'true', 'required': True}, {'name': 'providers', 'type': '["checkr", "sterling"]', 'required': True}, {'name': 'criminal_background', 'type': '"premium_users"', 'required': True}, {'name': 'professional_verification', 'type': '"all_freelancers"', 'required': True}, {'name': 'education_verification', 'type': '"optional"', 'required': True}, {'name': 'employment_history', 'type': '"premium_users"', 'required': True}, {'name': 'refresh_frequency', 'type': '"annually"', 'required': True}, {'name': 'provider', 'type': '"elasticsearch"', 'required': True}, {'name': 'version_1', 'type': '"8.0"', 'required': True}, {'name': 'cluster_name', 'type': '"freelancehub_search"', 'required': True}, {'name': 'nodes', 'type': '3', 'required': True}, {'name': 'shards', 'type': '5', 'required': True}, {'name': 'replicas', 'type': '1', 'required': True}, {'name': 'type', 'type': '"freelancer_profiles"', 'required': True}, {'name': 'rating', 'type': '2.0', 'required': True}, {'name': 'completion_rate', 'type': '1.5', 'required': True}, {'name': 'response_time', 'type': '1.3', 'required': True}, {'name': 'recent_activity', 'type': '1.2', 'required': True}, {'name': 'type_1', 'type': '"projects"', 'required': True}, {'name': 'budget', 'type': '1.5', 'required': True}, {'name': 'urgency', 'type': '1.8', 'required': True}, {'name': 'client_rating', 'type': '1.3', 'required': True}, {'name': 'update_frequency', 'type': '"real_time"', 'required': True}, {'name': 'similarity_metric', 'type': '"cosine"', 'required': True}, {'name': 'neighbors', 'type': '50', 'required': True}, {'name': 'weight', 'type': '0.3', 'required': True}, {'name': 'similarity_metric_1', 'type': '"jaccard"', 'required': True}, {'name': 'weight_1', 'type': '0.2', 'required': True}, {'name': 'feature_extraction', 'type': '"tfidf + embeddings"', 'required': True}, {'name': 'similarity_threshold', 'type': '0.7', 'required': True}, {'name': 'weight_2', 'type': '0.3', 'required': True}, {'name': 'collaborative', 'type': '0.4', 'required': True}, {'name': 'content_based', 'type': '0.3', 'required': True}, {'name': 'popularity', 'type': '0.2', 'required': True}, {'name': 'trending', 'type': '0.1', 'required': True}, {'name': 'implicit_feedback', 'type': '["views", "saves", "applications", "hires"]', 'required': True}, {'name': 'explicit_feedback', 'type': '["ratings", "favorites", "preferences"]', 'required': True}, {'name': 'decay_factor', 'type': '0.95', 'required': True}, {'name': 'enabled_2', 'type': 'true', 'required': True}, {'name': 'precision', 'type': '"city_level"', 'required': True}, {'name': 'default', 'type': '"50km"', 'required': True}, {'name': 'remote_work_preference', 'type': '"global"', 'required': True}, {'name': 'local_only', 'type': '"25km"', 'required': True}, {'name': 'timezone_matching', 'type': 'true', 'required': True}, {'name': 'name_6', 'type': '"skills"', 'required': True}, {'name': 'type_2', 'type': '"multi_select"', 'required': True}, {'name': 'aggregation', 'type': '"terms"', 'required': True}, {'name': 'size', 'type': '100', 'required': True}, {'name': 'name_7', 'type': '"hourly_rate"', 'required': True}, {'name': 'type_3', 'type': '"range"', 'required': True}, {'name': 'name_8', 'type': '"experience_level"', 'required': True}, {'name': 'type_4', 'type': '"single_select"', 'required': True}, {'name': 'options', 'type': '["entry", "intermediate", "expert"]', 'required': True}, {'name': 'name_9', 'type': '"availability"', 'required': True}, {'name': 'type_5', 'type': '"single_select"', 'required': True}, {'name': 'options_1', 'type': '["immediate", "within_week", "within_month"]', 'required': True}, {'name': 'name_10', 'type': '"location"', 'required': True}, {'name': 'type_6', 'type': '"hierarchical"', 'required': True}, {'name': 'levels', 'type': '["country", "state", "city"]', 'required': True}, {'name': 'enabled_3', 'type': 'true', 'required': True}, {'name': 'sources', 'type': '["skills", "job_titles", "company_names", "locations"]', 'required': True}, {'name': 'suggestions_count', 'type': '8', 'required': True}, {'name': 'fuzzy_matching', 'type': 'true', 'required': True}, {'name': 'typo_tolerance', 'type': '2', 'required': True}, {'name': 'search_queries', 'type': 'true', 'required': True}, {'name': 'result_clicks', 'type': 'true', 'required': True}, {'name': 'zero_results', 'type': 'true', 'required': True}, {'name': 'abandonment_rate', 'type': 'true', 'required': True}, {'name': 'query_expansion', 'type': 'true', 'required': True}, {'name': 'result_reranking', 'type': 'true', 'required': True}, {'name': 'a_b_testing', 'type': 'true', 'required': True}, {'name': 'platform_3', 'type': '"custom_websocket + redis"', 'required': True}, {'name': 'real_time_chat', 'type': 'true', 'required': True}, {'name': 'file_sharing', 'type': 'true', 'required': True}, {'name': 'video_calls', 'type': 'true', 'required': True}, {'name': 'screen_sharing', 'type': 'true', 'required': True}, {'name': 'message_encryption', 'type': '"end_to_end"', 'required': True}, {'name': 'message_history', 'type': '"unlimited"', 'required': True}, {'name': 'search_messages', 'type': 'true', 'required': True}, {'name': 'participants', 'type': '2', 'required': True}, {'name': 'encryption', 'type': '"end_to_end"', 'required': True}, {'name': 'participants_1', 'type': '"unlimited"', 'required': True}, {'name': 'features', 'type': '["file_sharing", "task_management", "milestone_tracking"]', 'required': True}, {'name': 'participants_2', 'type': '["user", "support_agent"]', 'required': True}, {'name': 'features_1', 'type': '["ticket_integration", "knowledge_base_search"]', 'required': True}, {'name': 'automated_filtering', 'type': 'true', 'required': True}, {'name': 'reporting_system', 'type': 'true', 'required': True}, {'name': 'admin_override', 'type': 'true', 'required': True}, {'name': 'type_7', 'type': '"push_notification"', 'required': True}, {'name': 'platforms', 'type': '["ios", "android", "web"]', 'required': True}, {'name': 'urgency_levels', 'type': '["low", "normal", "high", "critical"]', 'required': True}, {'name': 'type_8', 'type': '"email"', 'required': True}, {'name': 'templates', 'type': '["project_invitation", "payment_received", "dispute_alert"]', 'required': True}, {'name': 'personalization', 'type': 'true', 'required': True}, {'name': 'type_9', 'type': '"sms"', 'required': True}, {'name': 'use_cases', 'type': '["security_alerts", "payment_confirmations"]', 'required': True}, {'name': 'international', 'type': 'true', 'required': True}, {'name': 'user_customizable', 'type': 'true', 'required': True}, {'name': 'frequency_controls', 'type': '["immediate", "daily_digest", "weekly_summary"]', 'required': True}, {'name': 'technology', 'type': '"websocket + socketio"', 'required': True}, {'name': 'typing_indicators', 'type': 'true', 'required': True}, {'name': 'read_receipts', 'type': 'true', 'required': True}, {'name': 'message_reactions', 'type': 'true', 'required': True}, {'name': 'thread_replies', 'type': 'true', 'required': True}, {'name': 'connection_limit', 'type': '100000', 'required': True}, {'name': 'horizontal_scaling', 'type': 'true', 'required': True}, {'name': 'load_balancing', 'type': '"sticky_sessions"', 'required': True}, {'name': 'provider_1', 'type': '"agora.io"', 'required': True}, {'name': 'one_on_one', 'type': 'true', 'required': True}, {'name': 'group_calls', 'type': 'true', 'required': True}, {'name': 'screen_sharing_1', 'type': 'true', 'required': True}, {'name': 'recording', 'type': '"optional"', 'required': True}, {'name': 'max_resolution', 'type': '"1080p"', 'required': True}, {'name': 'adaptive_bitrate', 'type': 'true', 'required': True}, {'name': 'noise_suppression', 'type': 'true', 'required': True}, {'name': 'stages_1', 'type': '["initial_proposal", "negotiation", "final_terms", "acceptance"]', 'required': True}, {'name': 'auto_expiry', 'type': '7d', 'required': True}, {'name': 'stages_2', 'type': '["rate_proposal", "counter_offer", "agreement"]', 'required': True}, {'name': 'history_tracking', 'type': 'true', 'required': True}, {'name': 'proposal_templates', 'type': 'true', 'required': True}, {'name': 'auto_calculations', 'type': 'true', 'required': True}, {'name': 'comparison_tools', 'type': 'true', 'required': True}, {'name': 'deadline_tracking', 'type': 'true', 'required': True}, {'name': 'name_11', 'type': '"user_service"', 'required': True}, {'name': 'type_10', 'type': '"core_service"', 'required': True}, {'name': 'path', 'type': '"/api/users"', 'required': True}, {'name': 'methods', 'type': '["GET", "POST", "PUT", "DELETE"]', 'required': True}, {'name': 'authentication', 'type': '"jwt_required"', 'required': True}, {'name': 'path_1', 'type': '"/api/auth"', 'required': True}, {'name': 'methods_1', 'type': '["POST"]', 'required': True}, {'name': 'rate_limit', 'type': '"10/minute"', 'required': True}, {'name': 'type_11', 'type': '"postgresql"', 'required': True}, {'name': 'name_12', 'type': '"users_db"', 'required': True}, {'name': 'tables', 'type': '["users", "profiles", "verifications", "preferences"]', 'required': True}, {'name': 'dependencies', 'type': '["notification_service", "verification_service"]', 'required': True}, {'name': 'strategy', 'type': '"horizontal"', 'required': True}, {'name': 'min_instances', 'type': '2', 'required': True}, {'name': 'max_instances', 'type': '10', 'required': True}, {'name': 'cpu_threshold', 'type': '70%', 'required': True}, {'name': 'memory_threshold', 'type': '75%', 'required': True}, {'name': 'image', 'type': '"freelancehub/user-service:v2.1.0"', 'required': True}, {'name': 'cpu', 'type': '"500m"', 'required': True}, {'name': 'memory', 'type': '"1Gi"', 'required': True}, {'name': 'type_12', 'type': '"spread"', 'required': True}, {'name': 'constraints', 'type': '["node.role', 'required': True, 'default': '=worker"]'}, {'name': 'preferences', 'type': '["zone', 'required': True, 'default': '=us-west-2a"]'}, {'name': 'name_13', 'type': '"project_service"', 'required': True}, {'name': 'type_13', 'type': '"business_service"', 'required': True}, {'name': 'path_2', 'type': '"/api/projects"', 'required': True}, {'name': 'methods_2', 'type': '["GET", "POST", "PUT", "DELETE"]', 'required': True}, {'name': 'caching', 'type': '"redis:300s"', 'required': True}, {'name': 'path_3', 'type': '"/api/proposals"', 'required': True}, {'name': 'methods_3', 'type': '["GET", "POST", "PUT"]', 'required': True}, {'name': 'async_processing', 'type': 'true', 'required': True}, {'name': 'type_14', 'type': '"postgresql"', 'required': True}, {'name': 'name_14', 'type': '"projects_db"', 'required': True}, {'name': 'tables_1', 'type': '["projects", "proposals", "milestones", "contracts"]', 'required': True}, {'name': 'dependencies_1', 'type': '["user_service", "search_service", "notification_service"]', 'required': True}, {'name': 'strategy_1', 'type': '"horizontal"', 'required': True}, {'name': 'min_instances_1', 'type': '3', 'required': True}, {'name': 'max_instances_1', 'type': '15', 'required': True}, {'name': 'cpu_threshold_1', 'type': '65%', 'required': True}, {'name': 'custom_metrics', 'type': '["active_projects_count"]', 'required': True}, {'name': 'name_15', 'type': '"payment_service"', 'required': True}, {'name': 'type_15', 'type': '"financial_service"', 'required': True}, {'name': 'path_4', 'type': '"/api/payments"', 'required': True}, {'name': 'methods_4', 'type': '["GET", "POST"]', 'required': True}, {'name': 'security', 'type': '"pci_compliant"', 'required': True}, {'name': 'encryption_1', 'type': '"tls_1_3"', 'required': True}, {'name': 'type_16', 'type': '"postgresql"', 'required': True}, {'name': 'name_16', 'type': '"payments_db"', 'required': True}, {'name': 'encryption_2', 'type': '"field_level"', 'required': True}, {'name': 'backup_frequency', 'type': '"continuous"', 'required': True}, {'name': 'dependencies_2', 'type': '["user_service", "notification_service"]', 'required': True}, {'name': 'strategy_2', 'type': '"vertical"', 'required': True}, {'name': 'instances', 'type': '3', 'required': True}, {'name': 'vault_integration', 'type': 'true', 'required': True}, {'name': 'secret_rotation', 'type': '"weekly"', 'required': True}, {'name': 'audit_logging', 'type': '"comprehensive"', 'required': True}, {'name': 'name_17', 'type': '"search_service"', 'required': True}, {'name': 'type_17', 'type': '"data_service"', 'required': True}, {'name': 'path_5', 'type': '"/api/search"', 'required': True}, {'name': 'methods_5', 'type': '["GET", "POST"]', 'required': True}, {'name': 'caching_1', 'type': '"elasticsearch"', 'required': True}, {'name': 'type_18', 'type': '"elasticsearch"', 'required': True}, {'name': 'cluster', 'type': '"search_cluster"', 'required': True}, {'name': 'indices', 'type': '["freelancers", "projects", "skills"]', 'required': True}, {'name': 'dependencies_3', 'type': '["user_service", "project_service"]', 'required': True}, {'name': 'strategy_3', 'type': '"horizontal"', 'required': True}, {'name': 'min_instances_2', 'type': '2', 'required': True}, {'name': 'max_instances_2', 'type': '8', 'required': True}, {'name': 'cpu_threshold_2', 'type': '80%', 'required': True}, {'name': 'name_18', 'type': '"communication_service"', 'required': True}, {'name': 'type_19', 'type': '"communication_service"', 'required': True}, {'name': 'path_6', 'type': '"/api/messages"', 'required': True}, {'name': 'methods_6', 'type': '["GET", "POST", "PUT"]', 'required': True}, {'name': 'websocket', 'type': 'true', 'required': True}, {'name': 'type_20', 'type': '"mongodb"', 'required': True}, {'name': 'name_19', 'type': '"communications_db"', 'required': True}, {'name': 'collections', 'type': '["messages", "channels", "notifications"]', 'required': True}, {'name': 'dependencies_4', 'type': '["user_service"]', 'required': True}, {'name': 'strategy_4', 'type': '"horizontal"', 'required': True}, {'name': 'min_instances_3', 'type': '2', 'required': True}, {'name': 'max_instances_3', 'type': '20', 'required': True}, {'name': 'connection_based', 'type': 'true', 'required': True}, {'name': 'name_20', 'type': '"analytics_service"', 'required': True}, {'name': 'type_21', 'type': '"analytics_service"', 'required': True}, {'name': 'path_7', 'type': '"/api/analytics"', 'required': True}, {'name': 'methods_7', 'type': '["GET", "POST"]', 'required': True}, {'name': 'rate_limit_1', 'type': '"100/minute"', 'required': True}, {'name': 'type_22', 'type': '"clickhouse"', 'required': True}, {'name': 'name_21', 'type': '"analytics_db"', 'required': True}, {'name': 'partitioning', 'type': '"by_date"', 'required': True}, {'name': 'dependencies_5', 'type': '["all_services"]', 'required': True}, {'name': 'strategy_5', 'type': '"horizontal"', 'required': True}, {'name': 'min_instances_4', 'type': '1', 'required': True}, {'name': 'max_instances_4', 'type': '5', 'required': True}, {'name': 'schedule_based', 'type': 'true', 'required': True}, {'name': 'provider_2', 'type': '"kong"', 'required': True}, {'name': 'rate_limiting', 'type': 'true', 'required': True}, {'name': 'authentication_1', 'type': '"jwt + oauth2"', 'required': True}, {'name': 'request_transformation', 'type': 'true', 'required': True}, {'name': 'response_caching', 'type': 'true', 'required': True}, {'name': 'load_balancing_1', 'type': '"round_robin"', 'required': True}, {'name': 'versioning', 'type': '"url_path"', 'required': True}, {'name': 'ssl_termination', 'type': 'true', 'required': True}, {'name': 'cors_handling', 'type': 'true', 'required': True}, {'name': 'provider_3', 'type': '"istio"', 'required': True}, {'name': 'traffic_management', 'type': 'true', 'required': True}, {'name': 'security_policies', 'type': 'true', 'required': True}, {'name': 'observability', 'type': 'true', 'required': True}, {'name': 'circuit_breaking', 'type': 'true', 'required': True}, {'name': 'mutual_tls', 'type': '"strict"', 'required': True}, {'name': 'authorization', 'type': '"rbac"', 'required': True}, {'name': 'traffic_splitting', 'type': '"canary_deployments"', 'required': True}, {'name': 'provider_4', 'type': '"consul"', 'required': True}, {'name': 'health_checking', 'type': 'true', 'required': True}, {'name': 'service_registration', 'type': '"automatic"', 'required': True}, {'name': 'dns_interface', 'type': 'true', 'required': True}, {'name': 'default_1', 'type': '"least_connections"', 'required': True}, {'name': 'cpu_intensive', 'type': '"least_response_time"', 'required': True}, {'name': 'session_sticky', 'type': '"ip_hash"', 'required': True}, {'name': 'interval', 'type': '"30s"', 'required': True}, {'name': 'timeout', 'type': '"5s"', 'required': True}, {'name': 'retries', 'type': '3', 'required': True}, {'name': 'failure_threshold', 'type': '50%', 'required': True}, {'name': 'recovery_timeout', 'type': '"30s"', 'required': True}, {'name': 'half_open_max_calls', 'type': '3', 'required': True}, {'name': 'provider_5', 'type': '"jaeger"', 'required': True}, {'name': 'sampling_rate', 'type': '0.1', 'required': True}, {'name': 'trace_retention', 'type': '"7d"', 'required': True}, {'name': 'method', 'type': '"human_professional"', 'required': True}, {'name': 'fallback', 'type': '"machine_translation"', 'required': True}, {'name': 'quality_assurance', 'type': '"native_speaker_review"', 'required': True}, {'name': 'date_formats', 'type': '"region_specific"', 'required': True}, {'name': 'number_formats', 'type': '"locale_specific"', 'required': True}, {'name': 'address_formats', 'type': '"country_specific"', 'required': True}, {'name': 'payment_methods_2', 'type': '"region_preferred"', 'required': True}, {'name': 'provider_6', 'type': '"currencylayer"', 'required': True}, {'name': 'update_frequency_1', 'type': '"hourly"', 'required': True}, {'name': 'caching_2', 'type': '"redis:1h"', 'required': True}, {'name': 'fallback_rates', 'type': '"ecb"', 'required': True}, {'name': 'providers_1', 'type': '["avalara", "taxjar"]', 'required': True}, {'name': 'vat_handling', 'type': '"eu_regions"', 'required': True}, {'name': 'sales_tax', 'type': '"us_states"', 'required': True}, {'name': 'gst_handling', 'type': '"applicable_countries"', 'required': True}, {'name': 'cost_calculation', 'type': '"distance_based"', 'required': True}, {'name': 'delivery_options', 'type': '["standard", "express", "overnight"]', 'required': True}, {'name': 'cost_calculation_1', 'type': '"weight_zone_based"', 'required': True}, {'name': 'customs_handling', 'type': '"automated"', 'required': True}, {'name': 'duty_calculation', 'type': '"integrated"', 'required': True}, {'name': 'provider_7', 'type': '"snowflake"', 'required': True}, {'name': 'compute_clusters', 'type': '3', 'required': True}, {'name': 'storage_size', 'type': '"10TB"', 'required': True}, {'name': 'auto_scaling', 'type': 'true', 'required': True}, {'name': 'frequency', 'type': '"real_time + daily_batch"', 'required': True}, {'name': 'transformation_engine', 'type': '"dbt"', 'required': True}, {'name': 'data_quality_checks', 'type': 'true', 'required': True}, {'name': 'stream_processing', 'type': '"bytewax + flink"', 'required': True}, {'name': 'operational', 'type': '["system_health", "user_activity", "financial_metrics"]', 'required': True}, {'name': 'business', 'type': '["growth_metrics", "user_engagement", "revenue_analytics"]', 'required': True}, {'name': 'executive', 'type': '["kpi_summary", "trend_analysis", "forecast_modeling"]', 'required': True}, {'name': 'tools', 'type': '"tableau + looker"', 'required': True}, {'name': 'metrics', 'type': '["acquisition", "activation", "retention", "revenue", "referral"]', 'required': True}, {'name': 'frequency_1', 'type': '"daily"', 'required': True}, {'name': 'stakeholders', 'type': '["product", "marketing", "growth"]', 'required': True}, {'name': 'metrics_1', 'type': '["revenue", "commission", "transaction_volume", "payment_success_rate"]', 'required': True}, {'name': 'frequency_2', 'type': '"hourly"', 'required': True}, {'name': 'stakeholders_1', 'type': '["finance", "operations", "executive"]', 'required': True}, {'name': 'metrics_2', 'type': '["supply_demand_ratio", "matching_efficiency", "user_satisfaction"]', 'required': True}, {'name': 'frequency_3', 'type': '"weekly"', 'required': True}, {'name': 'stakeholders_2', 'type': '["product", "operations", "executive"]', 'required': True}, {'name': 'name_22', 'type': '"user_lifetime_value"', 'required': True}, {'name': 'type_23', 'type': '"regression"', 'required': True}, {'name': 'features_2', 'type': '["user_activity", "transaction_history", "engagement_metrics"]', 'required': True}, {'name': 'update_frequency_2', 'type': '"weekly"', 'required': True}, {'name': 'name_23', 'type': '"churn_prediction"', 'required': True}, {'name': 'type_24', 'type': '"classification"', 'required': True}, {'name': 'features_3', 'type': '["usage_patterns", "support_interactions", "satisfaction_scores"]', 'required': True}, {'name': 'update_frequency_3', 'type': '"daily"', 'required': True}, {'name': 'name_24', 'type': '"price_optimization"', 'required': True}, {'name': 'type_25', 'type': '"reinforcement_learning"', 'required': True}, {'name': 'features_4', 'type': '["market_demand", "competitor_pricing", "success_rates"]', 'required': True}, {'name': 'update_frequency_4', 'type': '"real_time"', 'required': True}, {'name': 'training_platform', 'type': '"kubeflow"', 'required': True}, {'name': 'model_serving', 'type': '"seldon"', 'required': True}, {'name': 'feature_store', 'type': '"feast"', 'required': True}, {'name': 'platform_4', 'type': '"optimizely"', 'required': True}, {'name': 'significance_threshold', 'type': '0.05', 'required': True}, {'name': 'minimum_sample_size', 'type': '1000', 'required': True}, {'name': 'test_duration', 'type': '"minimum_2_weeks"', 'required': True}, {'name': 'name_25', 'type': '"client_onboarding"', 'required': True}, {'name': 'steps', 'type': '["signup", "profile_complete", "first_project_post", "first_hire"]', 'required': True}, {'name': 'name_26', 'type': '"freelancer_onboarding"', 'required': True}, {'name': 'steps_1', 'type': '["signup", "profile_complete", "first_proposal", "first_job"]', 'required': True}, {'name': 'models_1', 'type': '["first_touch", "last_touch", "linear", "time_decay"]', 'required': True}, {'name': 'lookback_window', 'type': '"30d"', 'required': True}, {'name': 'metric', 'type': '"monthly_revenue"', 'required': True}, {'name': 'method_1', 'type': '"arima + prophet"', 'required': True}, {'name': 'forecast_horizon', 'type': '"12_months"', 'required': True}, {'name': 'metric_1', 'type': '"user_growth"', 'required': True}, {'name': 'method_2', 'type': '"regression + seasonality"', 'required': True}, {'name': 'forecast_horizon_1', 'type': '"6_months"', 'required': True}, {'name': 'user_id', 'type': '"uuid"', 'required': True}, {'name': 'user_type', 'type': '"enum[client, freelancer]"', 'required': True}, {'name': 'registration_source', 'type': '"string"', 'required': True}, {'name': 'timestamp', 'type': '"datetime"', 'required': True}, {'name': 'triggers', 'type': '["registration_complete"]', 'required': True}, {'name': 'exchange', 'type': '"user_events"', 'required': True}, {'name': 'routing_key', 'type': '"user.registered"', 'required': True}, {'name': 'project_id', 'type': '"uuid"', 'required': True}, {'name': 'client_id_1', 'type': '"uuid"', 'required': True}, {'name': 'project_type', 'type': '"string"', 'required': True}, {'name': 'budget_range', 'type': '"string"', 'required': True}, {'name': 'skills_required', 'type': '"array[string]"', 'required': True}, {'name': 'timestamp_1', 'type': '"datetime"', 'required': True}, {'name': 'triggers_1', 'type': '["project_publish"]', 'required': True}, {'name': 'transaction_id', 'type': '"uuid"', 'required': True}, {'name': 'payer_id', 'type': '"uuid"', 'required': True}, {'name': 'recipient_id', 'type': '"uuid"', 'required': True}, {'name': 'amount', 'type': '"decimal"', 'required': True}, {'name': 'currency', 'type': '"string"', 'required': True}, {'name': 'timestamp_2', 'type': '"datetime"', 'required': True}, {'name': 'triggers_2', 'type': '["payment_success"]', 'required': True}, {'name': 'dispute_id', 'type': '"uuid"', 'required': True}, {'name': 'project_id_1', 'type': '"uuid"', 'required': True}, {'name': 'initiator_id', 'type': '"uuid"', 'required': True}, {'name': 'dispute_type', 'type': '"string"', 'required': True}, {'name': 'severity', 'type': '"enum[low, medium, high, critical]"', 'required': True}, {'name': 'timestamp_3', 'type': '"datetime"', 'required': True}, {'name': 'triggers_3', 'type': '["dispute_initiated"]', 'required': True}], 'methods': []}, {'name': 'EcommerceHub', 'type': 'marketplace', 'properties': ['description', 'version', 'email', 'phone', 'payment_method', 'layout', 'payment_processing_fee', 'email_1', 'phone_1', 'business_license', 'tax_id', 'layout_1', 'platform_fee', 'payment_processing_fee_1', 'listing_fee', 'email_2', 'phone_2', 'professional_license', 'insurance_proof', 'background_check', 'platform_fee_1', 'booking_fee', 'brand_authentication', 'exclusive_agreements', 'platform_fee_2', 'marketing_co_op', 'levels', 'dynamic_attributes', 'ai_categorization', 'real_time_updates', 'multi_location_support', 'reserved_inventory', 'low_stock_alerts', 'automated_reordering', 'algorithms', 'frequency', 'types', 'duration_options', 'quantity_breaks', 'volume_discounts', 'attribute_types', 'variant_pricing', 'inventory_tracking', 'workflow', 'automation_level', 'processing_sla', 'shipping_integration', 'label_generation', 'tracking_integration', 'warehouse_network', 'same_day_delivery', 'two_day_shipping', 'supplier_integration', 'inventory_sync', 'order_routing', 'partner_network', 'delivery_windows', 'real_time_tracking', 'return_window', 'self_service_portal', 'prepaid_labels', 'quality_inspection', 'refund_processing', 'like_new', 'minor_damage', 'damaged', 'image_analysis', 'pattern_recognition', 'supplier_verification', 'business_license_1', 'tax_registration', 'bank_account_verification', 'money_back_guarantee', 'coverage', 'claim_window', 'automated_mediation', 'human_arbitration', 'evidence_collection', 'natural_language', 'voice_search', 'multi_language', 'image_upload', 'camera_search', 'similarity_matching', 'upc_ean_support', 'qr_code_support', 'mobile_integration', 'collaborative_filtering', 'content_based', 'deep_learning', 'seasonal_trending', 'weather_based', 'event_based', 'location_based', 'frequently_bought_together', 'complementary_products', 'bundle_recommendations', 'algorithms_1', 'update_frequency', 'flash_sales', 'lightning_deals', 'clearance_sections', 'price_drop_alerts', 'user_generated_content', 'influencer_partnerships', 'social_media_integration', 'algorithm', 'rebalancing_frequency', 'seasonal_adjustments', 'picking', 'packing', 'sorting', 'loading', 'algorithm_1', 'factors', 'last_mile_efficiency', 'consolidation_opportunities', 'delivery_windows_1', 'customs_automation', 'duty_calculation', 'restricted_items_checking', 'standard_delivery', 'expedited_delivery', 'same_day_delivery_1', 'scheduled_delivery', 'pickup_points'], 'fields': [{'name': 'description', 'type': '"Multi-sided e-commerce platform with buyers, sellers, and service providers"', 'required': True}, {'name': 'version', 'type': '"4.1.0"', 'required': True}, {'name': 'email', 'type': 'true', 'required': True}, {'name': 'phone', 'type': 'false', 'required': True}, {'name': 'payment_method', 'type': 'true', 'required': True}, {'name': 'layout', 'type': '"buyer_dashboard"', 'required': True}, {'name': 'payment_processing_fee', 'type': '2.9%', 'required': True}, {'name': 'email_1', 'type': 'true', 'required': True}, {'name': 'phone_1', 'type': 'true', 'required': True}, {'name': 'business_license', 'type': 'true', 'required': True}, {'name': 'tax_id', 'type': 'true', 'required': True}, {'name': 'layout_1', 'type': '"seller_dashboard"', 'required': True}, {'name': 'platform_fee', 'type': '6%', 'required': True}, {'name': 'payment_processing_fee_1', 'type': '2.9%', 'required': True}, {'name': 'listing_fee', 'type': '"$0.35 per item"', 'required': True}, {'name': 'email_2', 'type': 'true', 'required': True}, {'name': 'phone_2', 'type': 'true', 'required': True}, {'name': 'professional_license', 'type': 'true', 'required': True}, {'name': 'insurance_proof', 'type': 'true', 'required': True}, {'name': 'background_check', 'type': 'true', 'required': True}, {'name': 'platform_fee_1', 'type': '12%', 'required': True}, {'name': 'booking_fee', 'type': '"$2 per service"', 'required': True}, {'name': 'brand_authentication', 'type': 'true', 'required': True}, {'name': 'exclusive_agreements', 'type': 'true', 'required': True}, {'name': 'platform_fee_2', 'type': '4%', 'required': True}, {'name': 'marketing_co_op', 'type': '2%', 'required': True}, {'name': 'levels', 'type': '5', 'required': True}, {'name': 'dynamic_attributes', 'type': 'true', 'required': True}, {'name': 'ai_categorization', 'type': 'true', 'required': True}, {'name': 'real_time_updates', 'type': 'true', 'required': True}, {'name': 'multi_location_support', 'type': 'true', 'required': True}, {'name': 'reserved_inventory', 'type': 'true', 'required': True}, {'name': 'low_stock_alerts', 'type': 'true', 'required': True}, {'name': 'automated_reordering', 'type': 'true', 'required': True}, {'name': 'algorithms', 'type': '["demand_based", "competitor_based", "inventory_based"]', 'required': True}, {'name': 'frequency', 'type': '"hourly"', 'required': True}, {'name': 'types', 'type': '["english", "dutch", "sealed_bid"]', 'required': True}, {'name': 'duration_options', 'type': '["1d", "3d", "7d", "10d"]', 'required': True}, {'name': 'quantity_breaks', 'type': 'true', 'required': True}, {'name': 'volume_discounts', 'type': 'true', 'required': True}, {'name': 'attribute_types', 'type': '["size", "color", "material", "style"]', 'required': True}, {'name': 'variant_pricing', 'type': '"independent"', 'required': True}, {'name': 'inventory_tracking', 'type': '"per_variant"', 'required': True}, {'name': 'workflow', 'type': '"order_received -> payment_verified -> inventory_reserved -> picking -> packing -> shipping"', 'required': True}, {'name': 'automation_level', 'type': '85%', 'required': True}, {'name': 'processing_sla', 'type': '"same_day"', 'required': True}, {'name': 'shipping_integration', 'type': '["ups", "fedex", "dhl", "usps"]', 'required': True}, {'name': 'label_generation', 'type': '"automated"', 'required': True}, {'name': 'tracking_integration', 'type': 'true', 'required': True}, {'name': 'warehouse_network', 'type': '15', 'required': True}, {'name': 'same_day_delivery', 'type': '"major_cities"', 'required': True}, {'name': 'two_day_shipping', 'type': '"nationwide"', 'required': True}, {'name': 'supplier_integration', 'type': '"api_based"', 'required': True}, {'name': 'inventory_sync', 'type': '"real_time"', 'required': True}, {'name': 'order_routing', 'type': '"automated"', 'required': True}, {'name': 'partner_network', 'type': '"gig_economy_drivers"', 'required': True}, {'name': 'delivery_windows', 'type': '["2h", "4h", "same_day"]', 'required': True}, {'name': 'real_time_tracking', 'type': 'true', 'required': True}, {'name': 'return_window', 'type': '30', 'required': True}, {'name': 'self_service_portal', 'type': 'true', 'required': True}, {'name': 'prepaid_labels', 'type': 'true', 'required': True}, {'name': 'quality_inspection', 'type': '"automated + manual"', 'required': True}, {'name': 'refund_processing', 'type': '"within_5_business_days"', 'required': True}, {'name': 'like_new', 'type': '"resell_full_price"', 'required': True}, {'name': 'minor_damage', 'type': '"resell_discounted"', 'required': True}, {'name': 'damaged', 'type': '"liquidation_channels"', 'required': True}, {'name': 'image_analysis', 'type': '"ai_powered"', 'required': True}, {'name': 'pattern_recognition', 'type': 'true', 'required': True}, {'name': 'supplier_verification', 'type': 'true', 'required': True}, {'name': 'business_license_1', 'type': '"required"', 'required': True}, {'name': 'tax_registration', 'type': '"required"', 'required': True}, {'name': 'bank_account_verification', 'type': '"required"', 'required': True}, {'name': 'money_back_guarantee', 'type': 'true', 'required': True}, {'name': 'coverage', 'type': '"item_not_received + significantly_not_as_described"', 'required': True}, {'name': 'claim_window', 'type': '60', 'required': True}, {'name': 'automated_mediation', 'type': 'true', 'required': True}, {'name': 'human_arbitration', 'type': '"complex_cases"', 'required': True}, {'name': 'evidence_collection', 'type': '"integrated"', 'required': True}, {'name': 'natural_language', 'type': 'true', 'required': True}, {'name': 'voice_search', 'type': 'true', 'required': True}, {'name': 'multi_language', 'type': 'true', 'required': True}, {'name': 'image_upload', 'type': 'true', 'required': True}, {'name': 'camera_search', 'type': 'true', 'required': True}, {'name': 'similarity_matching', 'type': 'true', 'required': True}, {'name': 'upc_ean_support', 'type': 'true', 'required': True}, {'name': 'qr_code_support', 'type': 'true', 'required': True}, {'name': 'mobile_integration', 'type': 'true', 'required': True}, {'name': 'collaborative_filtering', 'type': 'true', 'required': True}, {'name': 'content_based', 'type': 'true', 'required': True}, {'name': 'deep_learning', 'type': '"neural_collaborative_filtering"', 'required': True}, {'name': 'seasonal_trending', 'type': 'true', 'required': True}, {'name': 'weather_based', 'type': 'true', 'required': True}, {'name': 'event_based', 'type': 'true', 'required': True}, {'name': 'location_based', 'type': 'true', 'required': True}, {'name': 'frequently_bought_together', 'type': 'true', 'required': True}, {'name': 'complementary_products', 'type': 'true', 'required': True}, {'name': 'bundle_recommendations', 'type': 'true', 'required': True}, {'name': 'algorithms_1', 'type': '["velocity", "social_signals", "engagement"]', 'required': True}, {'name': 'update_frequency', 'type': '"hourly"', 'required': True}, {'name': 'flash_sales', 'type': 'true', 'required': True}, {'name': 'lightning_deals', 'type': 'true', 'required': True}, {'name': 'clearance_sections', 'type': 'true', 'required': True}, {'name': 'price_drop_alerts', 'type': 'true', 'required': True}, {'name': 'user_generated_content', 'type': 'true', 'required': True}, {'name': 'influencer_partnerships', 'type': 'true', 'required': True}, {'name': 'social_media_integration', 'type': 'true', 'required': True}, {'name': 'algorithm', 'type': '"demand_prediction_based"', 'required': True}, {'name': 'rebalancing_frequency', 'type': '"weekly"', 'required': True}, {'name': 'seasonal_adjustments', 'type': 'true', 'required': True}, {'name': 'picking', 'type': '70%', 'required': True}, {'name': 'packing', 'type': '85%', 'required': True}, {'name': 'sorting', 'type': '95%', 'required': True}, {'name': 'loading', 'type': '60%', 'required': True}, {'name': 'algorithm_1', 'type': '"cost_speed_reliability_optimization"', 'required': True}, {'name': 'factors', 'type': '["cost", "delivery_time", "reliability", "sustainability"]', 'required': True}, {'name': 'last_mile_efficiency', 'type': 'true', 'required': True}, {'name': 'consolidation_opportunities', 'type': 'true', 'required': True}, {'name': 'delivery_windows_1', 'type': '"flexible"', 'required': True}, {'name': 'customs_automation', 'type': 'true', 'required': True}, {'name': 'duty_calculation', 'type': '"integrated"', 'required': True}, {'name': 'restricted_items_checking', 'type': 'true', 'required': True}, {'name': 'standard_delivery', 'type': '"3-5 business days"', 'required': True}, {'name': 'expedited_delivery', 'type': '"1-2 business days"', 'required': True}, {'name': 'same_day_delivery_1', 'type': '"select_metro_areas"', 'required': True}, {'name': 'scheduled_delivery', 'type': '"customer_preferred_time"', 'required': True}, {'name': 'pickup_points', 'type': '"partner_network"', 'required': True}], 'methods': []}]
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
SEMANTIC_MODEL: Dict[str, Any] = {'format': 'apg.semantic-model.v1', 'ok': True, 'source_files': ['main.apg'], 'app': {'name': 'main', 'version': '1.0.0', 'description': None, 'entity_count': 2}, 'symbols': {'module.main': {'id': 'module.main', 'kind': 'module', 'name': 'main', 'file': 'main.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'marketplace.FreelanceHub': {'id': 'marketplace.FreelanceHub', 'kind': 'marketplace', 'name': 'FreelanceHub', 'file': 'main.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}, 'marketplace.EcommerceHub': {'id': 'marketplace.EcommerceHub', 'kind': 'marketplace', 'name': 'EcommerceHub', 'file': 'main.apg', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}}, 'references': []}}, 'tables': {}, 'views': {}, 'flows': {}, 'operations': {}, 'rules': {}, 'roles': {}, 'security': {}, 'agents': {}, 'llms': {}, 'capabilities': {}, 'composition': {'applications': {}, 'agent_teams': {}, 'capability_dependencies': {}}, 'contracts': {}, 'deployment': {'target': 'python', 'source': 'main.apg'}, 'packages': {}, 'graphs': {'er': {'kind': 'er', 'nodes': 0, 'edges': 0}, 'lookup': {'kind': 'lookup', 'nodes': 3, 'edges': 2}, 'workflow': {'kind': 'workflow', 'nodes': 3, 'edges': 2}, 'handler': {'kind': 'handler', 'nodes': 3, 'edges': 2}, 'capability': {'kind': 'capability', 'nodes': 0, 'edges': 0}, 'security': {'kind': 'security', 'nodes': 3, 'edges': 2}, 'agent': {'kind': 'agent', 'nodes': 0, 'edges': 0}, 'deployment': {'kind': 'deployment', 'nodes': 3, 'edges': 2}, 'package': {'kind': 'package', 'nodes': 3, 'edges': 2}}, 'diagnostics': []}
APG_UI_TEMPLATES: Dict[str, str] = {'entity_list.html.j2': '{# entity_list.html.j2 — APG entity list + create form\n   Variables: entity_name, entity_type, safe_entity, fields, records,\n              total, count, records_table, create_inputs, notice, query,\n              has_kanban (bool), q (search term) #}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">{{ entity_name }}</span>\n  <div class="ml-auto flex items-center gap-1">\n    {% if has_kanban %}\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">≡ List</span>\n    <a href="/ui/entities/{{ safe_entity }}?view=kanban"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ⊞ Kanban\n    </a>\n    {% endif %}\n    <a href="/entities/{{ safe_entity }}/records"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-500 hover:border-gray-300 transition-colors">\n      API JSON ↗\n    </a>\n  </div>\n</nav>\n\n{% if notice %}\n<div role="alert"\n     class="mb-4 px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">\n  ⚠ {{ notice }}\n</div>\n{% endif %}\n\n{# Search bar #}\n<form method="get" action="/ui/entities/{{ safe_entity }}" class="mb-5">\n  <div class="relative max-w-sm">\n    <span class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-xs pointer-events-none">🔍</span>\n    <input type="text" name="q" value="{{ q or \'\' }}"\n           placeholder="Search {{ entity_name }} records…"\n           class="w-full pl-8 pr-8 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white">\n    {% if q %}\n    <a href="/ui/entities/{{ safe_entity }}"\n       class="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-700 text-xs leading-none">✕</a>\n    {% endif %}\n  </div>\n</form>\n\n<div class="flex items-start gap-5 flex-col xl:flex-row">\n\n  {# ── Create form ─────────────────────────────────────────────── #}\n  <aside class="w-full xl:w-72 shrink-0">\n    <div class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">\n      <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n        <h2 class="text-sm font-semibold text-gray-900">New {{ entity_name }}</h2>\n        <span class="text-xs font-medium uppercase tracking-wide text-gray-400 bg-gray-100 px-2 py-0.5 rounded">{{ entity_type }}</span>\n      </div>\n      <form method="post" action="/ui/entities/{{ safe_entity }}/records"\n            class="p-4 space-y-3" novalidate>\n        {{ create_inputs | safe }}\n        <button type="submit"\n                class="w-full mt-1 px-3 py-2 bg-apg-primary text-white text-sm font-medium rounded-lg hover:opacity-90 active:opacity-80 transition-opacity focus:outline-none focus:ring-2 focus:ring-apg-primary focus:ring-offset-2">\n          Create {{ entity_name }}\n        </button>\n      </form>\n    </div>\n  </aside>\n\n  {# ── Records section ─────────────────────────────────────────── #}\n  <section class="flex-1 min-w-0">\n    <div class="flex items-center gap-3 mb-3 flex-wrap">\n      <h1 class="text-lg font-semibold text-gray-900">{{ entity_name }}</h1>\n      <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n        {{ total }} record{{ \'s\' if total != 1 else \'\' }}\n      </span>\n      {% if count != total %}\n      <span class="text-xs text-apg-primary bg-blue-50 px-2 py-0.5 rounded-full">\n        {{ count }} match{% if q %} for "{{ q }}"{% endif %}\n      </span>\n      {% endif %}\n    </div>\n\n    {% if records %}\n      {{ records_table | safe }}\n    {% else %}\n      <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n        <div class="text-3xl mb-3 opacity-30">📋</div>\n        {% if q %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records match "{{ q }}".</p>\n        <p class="text-xs text-gray-400 mt-1">\n          <a href="/ui/entities/{{ safe_entity }}" class="text-apg-primary hover:underline">Clear search</a>\n        </p>\n        {% else %}\n        <p class="text-sm font-medium text-gray-500">No {{ entity_name }} records yet.</p>\n        <p class="text-xs text-gray-400 mt-1">Use the form on the left to create the first one.</p>\n        {% endif %}\n      </div>\n    {% endif %}\n  </section>\n\n</div>\n', 'kanban_view.html.j2': '{# kanban_view.html.j2 — Kanban board view for status-field entities\n   Variables: entity_name, safe_entity, columns, display_field, status_field, fields\n   columns: [{"label": str, "records": [dict]}]\n#}\n\n{# Breadcrumb + view toggle #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Kanban</span>\n  <div class="ml-auto flex items-center gap-1">\n    <a href="/ui/entities/{{ safe_entity }}"\n       class="px-3 py-1 text-xs border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n      ≡ List\n    </a>\n    <span class="px-3 py-1 text-xs bg-apg-primary text-white rounded-lg font-medium">⊞ Kanban</span>\n  </div>\n</nav>\n\n<div class="flex items-center gap-4 mb-5">\n  <h1 class="text-xl font-bold text-gray-900">{{ entity_name }}</h1>\n  <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n    by {{ status_field | replace(\'_\', \' \') }}\n  </span>\n  <span class="text-xs text-gray-400">\n    {{ columns | sum(attribute=\'records\') | length }} total\n  </span>\n</div>\n\n{# Kanban board — horizontal scroll #}\n<div class="flex gap-4 overflow-x-auto pb-6 items-start -mx-1 px-1">\n  {% for col in columns %}\n  <div class="flex-shrink-0 w-72">\n    {# Column header #}\n    <div class="flex items-center justify-between mb-3 px-1">\n      <div class="flex items-center gap-2">\n        <span class="w-2.5 h-2.5 rounded-full\n          {% if col.label | lower in [\'active\', \'approved\', \'paid\', \'open\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-400\n          {% elif col.label | lower in [\'inactive\', \'rejected\', \'closed\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-400\n          {% elif col.label | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-400\n          {% else %}bg-gray-300{% endif %}"></span>\n        <h2 class="text-sm font-semibold text-gray-900">{{ col.label }}</h2>\n        <span class="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full font-medium">{{ col.records | length }}</span>\n      </div>\n    </div>\n\n    {# Card list #}\n    <div class="space-y-2.5">\n      {% for record in col.records %}\n      {% set rec_id = record.get(\'id\', \'\') | string %}\n      <a href="/ui/entities/{{ safe_entity }}/{{ rec_id | urlencode }}"\n         class="block bg-white rounded-xl border border-gray-200 p-4 hover:border-apg-primary hover:shadow-md transition-all group/card">\n        <div class="flex items-start justify-between gap-2 mb-2.5">\n          <div class="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n               style="background: var(--apg-primary, #0ea5e9)">\n            {{ (record.get(display_field, \'\') | string)[:1] | upper or \'?\' }}\n          </div>\n          <span class="text-xs text-gray-300 font-mono mt-1">{{ rec_id[:8] }}</span>\n        </div>\n        <p class="text-sm font-semibold text-gray-900 group-hover/card:text-apg-primary transition-colors leading-tight mb-2">\n          {{ record.get(display_field, \'—\') | string | truncate(50) }}\n        </p>\n        {% for f in fields %}\n        {% if f.name not in [\'id\', \'_revision\', display_field, status_field] %}\n        {% set fval = record.get(f.name, \'\') %}\n        {% if fval and fval != \'\' and fval != \'None\' %}\n        <p class="text-xs text-gray-400 truncate mt-0.5">\n          <span class="font-medium text-gray-500">{{ f.name | replace(\'_id\', \'\') | replace(\'_\', \' \') | title }}:</span>\n          {{ fval | string | truncate(35) }}\n        </p>\n        {% if loop.index >= 2 %}{% break %}{% endif %}\n        {% endif %}\n        {% endif %}\n        {% endfor %}\n      </a>\n      {% endfor %}\n\n      {% if not col.records %}\n      <div class="bg-gray-50 rounded-xl border border-dashed border-gray-200 p-6 text-center">\n        <p class="text-xs text-gray-300">Empty</p>\n      </div>\n      {% endif %}\n    </div>\n  </div>\n  {% endfor %}\n\n  {% if not columns %}\n  <div class="flex-1 text-center py-16 text-gray-400">\n    <div class="text-4xl mb-3 opacity-20">⊞</div>\n    <p class="text-sm">No records to display.</p>\n  </div>\n  {% endif %}\n</div>\n', 'landing.html.j2': '{# landing.html.j2 — APG application landing page\n   Variables: module_name, module_description, entities, capabilities,\n              theme_primary, theme_accent, landing_style, api_links\n   landing_style: "default" | "minimal" | "corporate" | "africa"\n#}\n<!doctype html>\n<html lang="en" class="h-full">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>{{ module_name | replace(\'_\', \' \') | title }}</title>\n  <link rel="preconnect" href="https://fonts.googleapis.com">\n  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">\n  <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>\n  <script>\n    tailwind.config = {\n      theme: {\n        extend: {\n          fontFamily: { sans: [\'Inter\', \'system-ui\', \'sans-serif\'] },\n          colors: {\n            brand: { DEFAULT: \'{{ theme_primary }}\', light: \'{{ theme_accent }}\' }\n          }\n        }\n      }\n    }\n  </script>\n  <style>\n    :root {\n      --brand: {{ theme_primary }};\n      --accent: {{ theme_accent }};\n    }\n    .hero-gradient {\n      background: linear-gradient(135deg, {{ theme_primary }} 0%, {{ theme_accent }} 100%);\n    }\n    {% if landing_style == \'minimal\' %}\n    .hero-gradient { background: {{ theme_primary }}; }\n    {% elif landing_style == \'africa\' %}\n    .hero-gradient {\n      background: linear-gradient(135deg, #8B1A1A 0%, {{ theme_primary }} 50%, #E9A84B 100%);\n    }\n    {% elif landing_style == \'corporate\' %}\n    .hero-gradient { background: linear-gradient(180deg, #0F172A 0%, {{ theme_primary }} 100%); }\n    {% endif %}\n  </style>\n</head>\n<body class="min-h-full bg-gray-50 font-sans antialiased">\n\n  {# ── Hero ──────────────────────────────────────────────────────── #}\n  <div class="hero-gradient text-white">\n    <div class="max-w-6xl mx-auto px-6">\n      {# Topnav #}\n      <nav class="flex items-center justify-between py-5">\n        <span class="text-lg font-bold tracking-tight">\n          {{ module_name | replace(\'_\', \' \') | title }}\n        </span>\n        <div class="flex items-center gap-3">\n          <a href="/ui"\n             class="px-4 py-2 text-sm font-medium bg-white/15 hover:bg-white/25 rounded-lg transition-colors">\n            Open App\n          </a>\n          <a href="/openapi.json"\n             class="px-4 py-2 text-sm font-medium border border-white/30 hover:bg-white/10 rounded-lg transition-colors">\n            API Docs\n          </a>\n        </div>\n      </nav>\n\n      {# Hero content #}\n      <div class="py-20 text-center max-w-3xl mx-auto">\n        <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/15 text-xs font-medium mb-6">\n          <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>\n          Powered by APG · Datacraft\n        </div>\n        <h1 class="text-4xl sm:text-5xl font-extrabold tracking-tight mb-5">\n          {{ module_name | replace(\'_\', \' \') | title }}\n        </h1>\n        <p class="text-lg text-white/80 mb-8 leading-relaxed">\n          {{ module_description or \'A fully generated application by Datacraft APG.\' }}\n        </p>\n        <div class="flex items-center justify-center gap-3 flex-wrap">\n          <a href="/ui"\n             class="px-6 py-3 bg-white text-gray-900 font-semibold rounded-xl hover:bg-white/90 transition-colors shadow-lg">\n            Open Application →\n          </a>\n          <a href="/manifest"\n             class="px-6 py-3 border border-white/40 text-white font-medium rounded-xl hover:bg-white/10 transition-colors">\n            View Manifest\n          </a>\n        </div>\n      </div>\n    </div>\n  </div>\n\n  {# ── Stats strip ────────────────────────────────────────────────── #}\n  <div class="bg-white border-b border-gray-200 shadow-sm">\n    <div class="max-w-6xl mx-auto px-6 py-5 grid grid-cols-2 sm:grid-cols-4 gap-6 divide-x divide-gray-100">\n      {% for stat in stats %}\n      <div class="px-6 first:pl-0 last:pr-0 text-center">\n        <p class="text-2xl font-bold text-gray-900">{{ stat.value }}</p>\n        <p class="text-xs font-medium uppercase tracking-wide text-gray-400 mt-0.5">{{ stat.label }}</p>\n      </div>\n      {% endfor %}\n    </div>\n  </div>\n\n  {# ── Entity cards ───────────────────────────────────────────────── #}\n  <main class="max-w-6xl mx-auto px-6 py-12">\n    <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-5">Data Entities</h2>\n    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">\n      {% for entity in entities %}\n      {% if entity.type != \'application\' %}\n      <a href="/ui/entities/{{ entity.name }}"\n         class="group bg-white rounded-xl border border-gray-200 p-5 hover:border-brand hover:shadow-md transition-all">\n        <div class="flex items-start justify-between mb-3">\n          <div class="w-9 h-9 rounded-lg flex items-center justify-center text-white text-sm font-bold"\n               style="background: var(--brand)">\n            {{ entity.name[0] | upper }}\n          </div>\n          <span class="text-xs font-medium text-gray-400 bg-gray-50 px-2 py-0.5 rounded uppercase tracking-wide">\n            {{ entity.type }}\n          </span>\n        </div>\n        <h3 class="font-semibold text-gray-900 group-hover:text-brand transition-colors">\n          {{ entity.name }}\n        </h3>\n        <p class="text-xs text-gray-400 mt-1">\n          {{ entity.fields | length if entity.fields else entity.properties | length }} fields\n        </p>\n      </a>\n      {% endif %}\n      {% endfor %}\n    </div>\n\n    {# ── API quick links ──────────────────────────────────────────── #}\n    <h2 class="text-sm font-semibold uppercase tracking-wide text-gray-400 mb-4">API Endpoints</h2>\n    <div class="flex flex-wrap gap-2">\n      {% for link in api_links %}\n      <a href="{{ link.url }}"\n         class="px-3 py-1.5 text-sm text-gray-600 bg-white border border-gray-200 rounded-lg hover:border-brand hover:text-brand transition-colors">\n        {{ link.label }}\n      </a>\n      {% endfor %}\n    </div>\n  </main>\n\n  {# ── Footer ─────────────────────────────────────────────────────── #}\n  <footer class="border-t border-gray-200 py-6 text-center text-xs text-gray-400">\n    Generated by <span class="font-medium text-gray-600">APG</span> ·\n    <a href="https://www.datacraft.co.ke" class="hover:underline">Datacraft</a> ·\n    © 2025\n  </footer>\n\n</body>\n</html>\n', 'record_detail.html.j2': '{# record_detail.html.j2 — Salesforce-quality record detail page\n   Variables: entity_name, entity_type, safe_entity, safe_record_id,\n              record, fields, field_semantics, title, status_val, revision,\n              related_lists, has_kanban (bool)\n   related_lists: [{"entity": str, "fk_field": str, "records": [dict], "cols": [str]}]\n   field_semantics: {field_name: semantic_type}\n#}\n\n{# Breadcrumb #}\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <a href="/ui/entities/{{ safe_entity }}" class="hover:text-apg-primary transition-colors">{{ entity_name }}</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900 truncate max-w-xs">{{ title }}</span>\n</nav>\n\n{# Record header card #}\n<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5 overflow-hidden">\n  <div class="h-1 bg-apg-primary"></div>\n  <div class="px-6 py-5 flex items-start gap-4">\n    <div class="w-14 h-14 rounded-xl flex items-center justify-center text-white text-2xl font-bold flex-shrink-0"\n         style="background: var(--apg-primary, #0ea5e9)">\n      {{ (title[:1] | upper) if title else (entity_name[:1] | upper) }}\n    </div>\n    <div class="flex-1 min-w-0">\n      <div class="flex items-center gap-3 flex-wrap">\n        <h1 class="text-xl font-bold text-gray-900 break-all">{{ title }}</h1>\n        {% if status_val %}\n        <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold\n          {% if status_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\'] %}bg-green-100 text-green-800\n          {% elif status_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n          {% elif status_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n          {% else %}bg-gray-100 text-gray-600{% endif %}">\n          {{ status_val }}\n        </span>\n        {% endif %}\n        <span class="text-xs font-medium text-gray-400 bg-gray-50 px-2 py-0.5 rounded uppercase tracking-wide">{{ entity_type }}</span>\n      </div>\n      {% set id_val = record.get(\'id\', \'\') %}\n      {% if id_val %}\n      <p class="text-xs text-gray-400 mt-1 font-mono truncate">{{ id_val | string }}</p>\n      {% endif %}\n    </div>\n    <div class="flex items-center gap-2 flex-shrink-0 flex-wrap">\n      <a href="/ui/workflows/{{ safe_entity }}/create_{{ safe_entity }}"\n         class="px-3 py-1.5 text-sm font-medium bg-apg-primary text-white rounded-lg hover:opacity-90 transition-opacity whitespace-nowrap">\n        ⚡ Workflow\n      </a>\n      <form method="post"\n            action="/ui/entities/{{ safe_entity }}/records/{{ safe_record_id }}/delete"\n            class="inline"\n            onsubmit="return confirm(\'Delete this record? This cannot be undone.\')">\n        <input type="hidden" name="expected_revision" value="{{ revision }}">\n        <button type="submit"\n                class="px-3 py-1.5 text-sm font-medium border border-red-200 text-red-500 rounded-lg hover:bg-red-50 transition-colors">\n          Delete\n        </button>\n      </form>\n    </div>\n  </div>\n{# Highlights panel — top fields at a glance #}\n{% set highlight_fields = [] %}\n{% for f in fields %}\n  {% if f.name not in [\'id\', \'_revision\'] and not f.name.endswith(\'_id\') %}\n    {% if highlight_fields | length < 4 %}\n      {% set _ = highlight_fields.append(f) %}\n    {% endif %}\n  {% endif %}\n{% endfor %}\n{% if highlight_fields %}\n<div class="border-t border-gray-100 px-6 py-3 grid grid-cols-2 md:grid-cols-4 gap-4 bg-gray-50/50">\n  {% for f in highlight_fields %}\n  {% set fv = record.get(f.name, \'\') %}\n  {% set sem = field_semantics.get(f.name, \'text\') if field_semantics else \'text\' %}\n  <div class="min-w-0">\n    <p class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-0.5 truncate">\n      {{ f.name | replace(\'_id\',\'\') | replace(\'_\',\' \') | title }}\n    </p>\n    <p class="text-sm font-medium text-gray-900 truncate">\n      {% if fv is none or fv == \'\' or fv | string == \'None\' %}\n      <span class="text-gray-300 italic text-xs">—</span>\n      {% elif sem == \'currency\' %}\n      <span class="tabular-nums">{{ fv | string }}</span>\n      {% elif sem == \'status\' %}\n      <span class="inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-semibold\n        {% if fv | string | lower in [\'active\',\'approved\',\'paid\',\'open\',\'enabled\',\'complete\',\'completed\',\'success\',\'done\'] %}bg-green-100 text-green-800\n        {% elif fv | string | lower in [\'inactive\',\'rejected\',\'closed\',\'disabled\',\'cancelled\',\'canceled\',\'failed\',\'expired\'] %}bg-red-100 text-red-800\n        {% else %}bg-yellow-100 text-yellow-800{% endif %}">{{ fv }}</span>\n      {% else %}\n      {{ fv | string | truncate(30) }}\n      {% endif %}\n    </p>\n  </div>\n  {% endfor %}\n</div>\n{% endif %}\n</div>\n\n{# Tab bar #}\n<div class="flex items-center gap-1 border-b border-gray-200 mb-6">\n  <button onclick="apgTab(\'details\')" id="apg-tab-details"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-apg-primary text-apg-primary -mb-px transition-colors">\n    Details\n  </button>\n  <button onclick="apgTab(\'related\')" id="apg-tab-related"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Related\n    {% if related_lists %}\n    <span class="ml-1 text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">\n      {{ related_lists | sum(attribute=\'records\') | length if related_lists else 0 }}\n    </span>\n    {% endif %}\n  </button>\n  <button onclick="apgTab(\'activity\')" id="apg-tab-activity"\n          class="apg-tab-btn px-4 py-2.5 text-sm font-medium border-b-2 border-transparent text-gray-500 hover:text-gray-900 -mb-px transition-colors">\n    Activity\n  </button>\n</div>\n\n{# Details panel #}\n<div id="apg-panel-details">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n      <h2 class="text-sm font-semibold text-gray-900">Record Details</h2>\n      <span class="text-xs text-gray-400 font-mono">rev. {{ revision }}</span>\n    </div>\n    <div class="p-5 grid grid-cols-1 md:grid-cols-2 gap-x-8">\n      {% for field in fields %}\n      {% if field.name != \'_revision\' %}\n      {% set field_val = record.get(field.name, \'\') %}\n      {% set fld_id = \'fld-\' ~ safe_entity ~ \'-\' ~ safe_record_id ~ \'-\' ~ field.name %}\n      <div id="{{ fld_id }}" class="py-3 border-b border-gray-50 last:border-0 group/field">\n        <dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">\n          {{ field.name | replace(\'_id\', \'\') | replace(\'_\', \' \') | title }}\n        </dt>\n        <dd class="flex items-center justify-between gap-2 min-h-6">\n          <span class="text-sm text-gray-900 break-words">\n            {% set semantic = field_semantics.get(field.name, \'text\') if field_semantics else \'text\' %}\n            {% include \'widgets/field_display.html.j2\' %}\n          </span>\n          <button\n            hx-get="/ui/entities/{{ safe_entity }}/{{ safe_record_id }}/fields/{{ field.name }}/edit"\n            hx-target="#{{ fld_id }}"\n            hx-swap="outerHTML"\n            class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"\n            title="Edit {{ field.name }}">\n            <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">\n              <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zm-2.207 2.207L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>\n            </svg>\n          </button>\n        </dd>\n      </div>\n      {% endif %}\n      {% endfor %}\n    </div>\n  </div>\n</div>\n\n{# Related panel #}\n<div id="apg-panel-related" class="hidden">\n  {% if related_lists %}\n    {% for rel in related_lists %}\n    <div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-4">\n      <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">\n        <div class="flex items-center gap-2">\n          <h2 class="text-sm font-semibold text-gray-900">{{ rel.entity }}</h2>\n          <span class="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded-full font-medium">{{ rel.records | length }}</span>\n        </div>\n        <a href="/ui/entities/{{ rel.entity | urlencode }}"\n           class="text-xs text-apg-primary hover:underline">View all →</a>\n      </div>\n      {% if rel.records %}\n      <div class="overflow-x-auto">\n        <table class="w-full text-sm">\n          <thead>\n            <tr class="bg-gray-50 border-b border-gray-100">\n              {% for col in rel.cols %}\n              <th class="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">\n                {{ col | replace(\'_id\', \'\') | replace(\'_\', \' \') | title }}\n              </th>\n              {% endfor %}\n              <th class="px-4 py-2 w-16"></th>\n            </tr>\n          </thead>\n          <tbody class="divide-y divide-gray-50">\n            {% for row in rel.records[:5] %}\n            <tr class="hover:bg-gray-50 transition-colors">\n              {% for col in rel.cols %}\n              <td class="px-4 py-2.5 text-gray-700 max-w-xs truncate">\n                {{ row.get(col, \'\') | string | truncate(40) }}\n              </td>\n              {% endfor %}\n              <td class="px-4 py-2.5 text-right">\n                <a href="/ui/entities/{{ rel.entity | urlencode }}/{{ row.get(\'id\', \'\') | string | urlencode }}"\n                   class="text-xs text-apg-primary hover:underline font-medium">View →</a>\n              </td>\n            </tr>\n            {% endfor %}\n            {% if rel.records | length > 5 %}\n            <tr>\n              <td colspan="{{ rel.cols | length + 1 }}" class="px-4 py-2.5 text-center text-xs text-gray-400">\n                + {{ rel.records | length - 5 }} more —\n                <a href="/ui/entities/{{ rel.entity | urlencode }}" class="text-apg-primary hover:underline">view all</a>\n              </td>\n            </tr>\n            {% endif %}\n          </tbody>\n        </table>\n      </div>\n      {% else %}\n      <div class="px-4 py-8 text-center text-sm text-gray-400">No related {{ rel.entity }} records.</div>\n      {% endif %}\n    </div>\n    {% endfor %}\n  {% else %}\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center">\n    <div class="text-4xl mb-3 opacity-20">🔗</div>\n    <p class="text-sm font-medium text-gray-500">No related records found.</p>\n    <p class="text-xs text-gray-400 mt-1">Other entities with FK fields pointing to {{ entity_name }} appear here.</p>\n  </div>\n  {% endif %}\n</div>\n\n{# Activity panel #}\n<div id="apg-panel-activity" class="hidden">\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm">\n    <div class="px-4 py-3 border-b border-gray-100">\n      <h2 class="text-sm font-semibold text-gray-900">Activity</h2>\n    </div>\n    <div class="p-5">\n      <ol class="relative border-l-2 border-gray-100 ml-4 space-y-5">\n        <li class="ml-6">\n          <span class="absolute flex items-center justify-center w-8 h-8 bg-blue-50 rounded-full -left-4 text-sm ring-4 ring-white">📋</span>\n          <div class="pl-2">\n            <p class="text-sm font-medium text-gray-900">Record created</p>\n            <p class="text-xs text-gray-400 mt-0.5">Revision {{ revision }} · via APG</p>\n          </div>\n        </li>\n      </ol>\n      <div class="mt-8 flex gap-3">\n        <div class="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0"\n             style="background: var(--apg-primary, #0ea5e9)">A</div>\n        <div class="flex-1">\n          <textarea placeholder="Add a note…" rows="2"\n                    class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent placeholder-gray-300"></textarea>\n          <button class="mt-1.5 px-3 py-1.5 bg-apg-primary text-white text-xs font-medium rounded-lg hover:opacity-90 transition-opacity">\n            Save Note\n          </button>\n        </div>\n      </div>\n    </div>\n  </div>\n</div>\n\n<script>\nfunction apgTab(name) {\n  document.querySelectorAll(\'.apg-tab-btn\').forEach(function(b) {\n    b.classList.remove(\'border-apg-primary\', \'text-apg-primary\');\n    b.classList.add(\'border-transparent\', \'text-gray-500\');\n  });\n  document.querySelectorAll(\'[id^="apg-panel-"]\').forEach(function(p) { p.classList.add(\'hidden\'); });\n  var btn = document.getElementById(\'apg-tab-\' + name);\n  if (btn) {\n    btn.classList.remove(\'border-transparent\', \'text-gray-500\');\n    btn.classList.add(\'border-apg-primary\', \'text-apg-primary\');\n  }\n  var panel = document.getElementById(\'apg-panel-\' + name);\n  if (panel) panel.classList.remove(\'hidden\');\n}\n</script>\n', 'marketplace.html.j2': '{# marketplace.html.j2 — APG Connector Marketplace\n   Variables: connectors (list of manifest dicts), installed_count\n#}\n\n<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500 flex-wrap">\n  <a href="/ui" class="hover:text-apg-primary transition-colors">Application</a>\n  <span>/</span>\n  <span class="font-semibold text-gray-900">Connector Marketplace</span>\n</nav>\n\n<div class="flex items-center gap-4 mb-6">\n  <h1 class="text-xl font-bold text-gray-900">Connector Marketplace</h1>\n  <span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full font-medium">\n    {{ connectors | length }} connector{{ \'s\' if connectors | length != 1 else \'\' }}\n  </span>\n</div>\n\n{% if connectors %}\n<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">\n  {% for conn in connectors %}\n  <div class="bg-white rounded-xl border border-gray-200 shadow-sm hover:border-apg-primary hover:shadow-md transition-all p-5 group">\n    <div class="flex items-start gap-4 mb-3">\n      <div class="w-12 h-12 rounded-xl flex items-center justify-center text-white text-xl font-bold flex-shrink-0"\n           style="background: var(--apg-primary, #0ea5e9)">\n        {{ (conn.title or conn.name or \'?\')[:1] | upper }}\n      </div>\n      <div class="flex-1 min-w-0">\n        <h2 class="text-sm font-bold text-gray-900 group-hover:text-apg-primary transition-colors truncate">\n          {{ conn.title or conn.name }}\n        </h2>\n        <p class="text-xs text-gray-400 truncate">{{ conn.base_url or \'Custom connector\' }}</p>\n      </div>\n    </div>\n    <div class="flex items-center gap-3 text-xs text-gray-500 mb-4">\n      <span class="flex items-center gap-1">\n        <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span>\n        {{ (conn.operations or []) | length }} operations\n      </span>\n      {% if conn.version %}\n      <span class="text-gray-300">·</span>\n      <span>v{{ conn.version }}</span>\n      {% endif %}\n    </div>\n    <div class="flex items-center gap-2">\n      <span class="flex-1 text-xs text-gray-400 font-mono truncate">{{ conn.file or conn.name }}</span>\n      <a href="/entities/connectors/{{ conn.name | urlencode }}"\n         class="px-3 py-1.5 text-xs font-medium border border-gray-200 rounded-lg text-gray-600 hover:border-apg-primary hover:text-apg-primary transition-colors">\n        View API ↗\n      </a>\n    </div>\n  </div>\n  {% endfor %}\n</div>\n{% else %}\n<div class="bg-white rounded-xl border border-gray-200 shadow-sm p-16 text-center">\n  <div class="text-5xl mb-4 opacity-20">🔌</div>\n  <p class="text-sm font-medium text-gray-500 mb-1">No connectors installed</p>\n  <p class="text-xs text-gray-400 mb-6">Generate a connector from an OpenAPI spec to get started.</p>\n  <div class="bg-gray-50 rounded-lg border border-gray-200 px-4 py-3 text-left max-w-sm mx-auto">\n    <p class="text-xs font-mono text-gray-600">apg connector generate --spec openapi.yaml</p>\n  </div>\n</div>\n{% endif %}\n', 'app_index.html.j2': '<!--- app_index.html.j2 — APG application home page --->\n{# Variables: module_name, module_description, entities, capabilities, databases,\n              application_routes, ui_routes, agents, agent_teams #}\n<div class="mb-6">\n  <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">{{ module_name }}</h1>\n  <p class="text-gray-500 mt-1">{{ module_description or \'Generated APG application\' }}</p>\n</div>\n\n{# Quick nav #}\n<nav class="flex flex-wrap gap-2 mb-8 text-sm" aria-label="API navigation">\n  <a href="/ui/workflows"\n     class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-apg-primary text-white hover:opacity-90 transition-opacity font-medium">\n    ⚡ Workflows\n  </a>\n  {% for link in api_links %}\n  <a href="{{ link.url }}"\n     class="inline-flex items-center px-3 py-1.5 rounded border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-apg-primary transition-colors">\n    {{ link.label }}\n  </a>\n  {% endfor %}\n</nav>\n\n{# Stats row #}\n<div class="apg-grid-4 mb-8">\n  <div class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ entities | length }}</span>\n      <span class="apg-stat-label">Entities</span>\n    </div>\n  </div>\n  <div class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ capabilities | length }}</span>\n      <span class="apg-stat-label">Capabilities</span>\n    </div>\n  </div>\n  <div class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ agents | length }}</span>\n      <span class="apg-stat-label">AI Agents</span>\n    </div>\n  </div>\n  <div class="apg-card">\n    <div class="apg-stat">\n      <span class="apg-stat-value">{{ ui_routes | length }}</span>\n      <span class="apg-stat-label">Screens</span>\n    </div>\n  </div>\n</div>\n\n<div class="apg-grid-2 gap-6">\n  {% if entities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Entities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for entity in entities %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="/ui/entities/{{ entity.name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ entity.name }}\n        </a>\n        <span class="apg-badge apg-badge-neutral">{{ entity.type }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if capabilities %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Capabilities</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for cap in capabilities %}\n      <li class="py-2">\n        <a href="/ui/capabilities/{{ cap | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ cap }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if ui_routes %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">Application Screens</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for route, screen in ui_routes.items() %}\n      <li class="py-2 flex items-center justify-between">\n        <a href="{{ route }}" class="text-sm text-apg-primary hover:underline">{{ route }}</a>\n        <span class="text-xs text-gray-400">{{ screen.get(\'application\', \'\') }}</span>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n\n  {% if agents %}\n  <div class="apg-card">\n    <div class="apg-card-header">\n      <h2 class="text-sm font-semibold text-gray-900 dark:text-gray-100">AI Agents</h2>\n    </div>\n    <ul class="divide-y divide-gray-100 dark:divide-gray-700">\n      {% for agent_name in agents %}\n      <li class="py-2">\n        <a href="/ui/agents/{{ agent_name | urlencode }}"\n           class="text-sm text-apg-primary hover:underline font-medium">\n          {{ agent_name }}\n        </a>\n      </li>\n      {% endfor %}\n    </ul>\n  </div>\n  {% endif %}\n</div>\n', 'widgets/field_display.html.j2': '{# field_display.html.j2 — semantic field rendering for record detail\n   Included by record_detail.html.j2 for individual field value rendering.\n   Variables: field (dict), field_val (any), semantic (str)\n#}\n{% if semantic == \'email\' and field_val %}\n  <a href="mailto:{{ field_val }}" class="text-apg-primary hover:underline text-sm">{{ field_val }}</a>\n{% elif semantic == \'phone\' and field_val %}\n  <a href="tel:{{ field_val }}" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1">\n    <svg class="w-3 h-3" viewBox="0 0 20 20" fill="currentColor"><path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z"/></svg>\n    {{ field_val }}\n  </a>\n{% elif semantic == \'url\' and field_val %}\n  <a href="{{ field_val }}" target="_blank" rel="noopener" class="text-apg-primary hover:underline text-sm inline-flex items-center gap-1 truncate max-w-xs">\n    <svg class="w-3 h-3 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/><path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/></svg>\n    {{ field_val | string | truncate(40) }}\n  </a>\n{% elif semantic == \'image_url\' and field_val %}\n  <img src="{{ field_val }}" alt="{{ field.name }}" class="w-12 h-12 rounded-lg object-cover border border-gray-100">\n{% elif semantic == \'currency\' and field_val %}\n  <span class="text-sm font-semibold text-gray-900 tabular-nums">{{ field_val | string | float | round(2) }}</span>\n{% elif semantic == \'percent\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="flex-1 bg-gray-100 rounded-full h-1.5 max-w-24">\n      <div class="bg-apg-primary h-1.5 rounded-full" style="width: {{ [field_val | float, 100] | min }}%"></div>\n    </div>\n    <span class="text-sm text-gray-700 tabular-nums">{{ field_val }}%</span>\n  </div>\n{% elif semantic == \'rating\' and field_val %}\n  <div class="flex items-center gap-0.5">\n    {% set stars = field_val | float | round | int %}\n    {% for i in range(5) %}\n    <svg class="w-4 h-4 {{ \'text-amber-400\' if i < stars else \'text-gray-200\' }}" viewBox="0 0 20 20" fill="currentColor">\n      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>\n    </svg>\n    {% endfor %}\n    <span class="text-xs text-gray-400 ml-1">{{ field_val }}/5</span>\n  </div>\n{% elif semantic == \'color\' and field_val %}\n  <div class="flex items-center gap-2">\n    <div class="w-5 h-5 rounded-full border border-gray-200 flex-shrink-0" style="background-color: {{ field_val }}"></div>\n    <span class="text-sm text-gray-700 font-mono">{{ field_val }}</span>\n  </div>\n{% elif semantic == \'json\' and field_val %}\n  <details class="max-w-xs">\n    <summary class="text-xs text-apg-primary cursor-pointer hover:underline">View JSON</summary>\n    <pre class="mt-1 text-xs bg-gray-50 rounded-lg p-2 overflow-auto max-h-40 border border-gray-100">{{ field_val | string }}</pre>\n  </details>\n{% elif semantic == \'status\' and field_val %}\n  <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold\n    {% if field_val | lower in [\'active\', \'approved\', \'paid\', \'open\', \'enabled\', \'complete\', \'completed\', \'success\', \'done\'] %}bg-green-100 text-green-800\n    {% elif field_val | lower in [\'inactive\', \'rejected\', \'closed\', \'disabled\', \'cancelled\', \'canceled\', \'failed\', \'expired\'] %}bg-red-100 text-red-800\n    {% elif field_val | lower in [\'pending\', \'draft\', \'processing\', \'review\', \'in_progress\', \'waiting\'] %}bg-yellow-100 text-yellow-800\n    {% else %}bg-gray-100 text-gray-600{% endif %}">\n    {{ field_val }}\n  </span>\n{% elif semantic == \'boolean\' %}\n  {% if field_val | string | lower in [\'true\', \'1\', \'yes\'] %}\n  <span class="inline-flex items-center gap-1 text-green-600 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>Yes</span>\n  {% else %}\n  <span class="inline-flex items-center gap-1 text-gray-400 text-sm"><svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>No</span>\n  {% endif %}\n{% else %}\n  {% if field_val is none or field_val == \'\' or field_val | string == \'None\' %}\n  <span class="text-gray-300 italic text-xs">—</span>\n  {% else %}\n  {{ field_val | string | truncate(200) }}\n  {% endif %}\n{% endif %}\n'}


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
        pass
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
                pass


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
        step_policy = retry_policy.get(step, {}) if isinstance(retry_policy, dict) else {}
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
            pass
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
            pass
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


def _persist_record_store() -> str | None:
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
    required_key = os.environ.get("APG_API_KEY")
    if not required_key:
        return True
    supplied_key = headers.get("X-APG-API-Key")
    authorization = headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        supplied_key = authorization.removeprefix("Bearer ").strip()
    return supplied_key == required_key


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
        if not partial and field.get("required", True):
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
        "  --apg-accent: #126e82;",
        "  --apg-surface: #ffffff;",
        "  --apg-border: #d0d7de;",
        "  --apg-text: #1f2328;",
        "  --apg-muted: #59636e;",
        "}",
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
    lines.extend([
        # Extended spacing + radius + shadow tokens
        ":root { --apg-radius: 8px; --apg-radius-sm: 4px; --apg-radius-full: 9999px; }",
        ":root { --apg-shadow-sm: 0 1px 2px rgba(0,0,0,0.08); --apg-shadow-md: 0 4px 6px rgba(0,0,0,0.10); --apg-shadow-lg: 0 10px 15px rgba(0,0,0,0.12); }",
        ":root { --apg-sidebar-width: 240px; --apg-topbar-height: 56px; }",
        ":root { --apg-font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; --apg-font-mono: 'JetBrains Mono', ui-monospace, monospace; }",
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
    head_extras = (
        # Inter font
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
        # Tailwind CDN — enables utility classes in Jinja2 templates
        '<script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>'
        '<script>tailwind.config={theme:{extend:{fontFamily:{sans:["Inter","system-ui","sans-serif"],mono:["JetBrains Mono","ui-monospace","monospace"]},colors:{apg:{primary:"#1E5B5A",accent:"#D97706"}}}}}</script>'
        # htmx — progressive enhancement for partial updates
        '<script defer src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>'
        # SortableJS — drag-and-drop for kanban
        '<script defer src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.3/Sortable.min.js"></script>'
    )
    toast_js = (
        '<div id="apg-toast-root" class="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none"></div>'
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
        f'<header class="apg-topbar sticky top-0 z-50" role="banner">'
        f'  <a class="apg-logo" href="/ui">{safe_module}</a>'
        f'  <nav class="apg-topnav ml-4">'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui">Home</a>'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui/workflows">⚡ Workflows</a>'
        f'    <a class="apg-nav-link hover:bg-gray-100" href="/ui/marketplace">Marketplace</a>'
        f'  </nav>'
        f'</header>'
        f'<main class="apg-content" id="main-content">{body}</main>'
        f"{toast_js}"
        f"{cmd_palette_html}"
        "</body></html>"
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
        if not partial and field.get("required", True) and field_name not in record:
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
    cards = []
    for entity_name, workflows in APP_WORKFLOWS.items():
        for wf in workflows:
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(wf["id"], safe=""), quote=True)
            cards.append(
                f'<a href="/ui/workflows/{safe_entity}/{safe_wf_id}"'
                f'   class="group block bg-white rounded-xl border border-gray-200 p-5 hover:border-blue-400 hover:shadow-md transition-all">'
                f'<div class="flex items-start gap-3 mb-3">'
                f'  <span class="text-2xl" aria-hidden="true">{html.escape(wf["icon"])}</span>'
                f'  <div>'
                f'    <h3 class="font-semibold text-gray-900 group-hover:text-blue-600 text-sm">{html.escape(wf["name"])}</h3>'
                f'    <p class="text-xs text-gray-400 mt-0.5">{html.escape(entity_name)} · {len(wf["steps"])} steps</p>'
                f'  </div>'
                f'</div>'
                f'<p class="text-xs text-gray-500 leading-relaxed">{html.escape(wf["description"])}</p>'
                f'<div class="mt-3 flex items-center gap-1">'
                + "".join(
                    f'<div class="h-1.5 flex-1 rounded-full bg-gray-100 first:bg-blue-400"></div>'
                    for _ in wf["steps"]
                )
                + f'</div>'
                f'</a>'
            )
    grid = f'<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">{"".join(cards)}</div>' if cards else "<p>No workflows available.</p>"
    body = (
        '<nav class="flex items-center gap-2 text-sm mb-6 text-gray-500">'
        '<a href="/ui" class="hover:text-blue-600">Application</a>'
        '<span>/</span><span class="font-semibold text-gray-900">Workflows</span></nav>'
        f'<div class="flex items-center justify-between mb-6">'
        f'<div><h1 class="text-xl font-bold text-gray-900">Workflows</h1>'
        f'<p class="text-sm text-gray-500 mt-1">{total} guided workflows across {len(APP_WORKFLOWS)} entities</p></div>'
        f'</div>'
        + grid
    )
    return 200, _html_page("Workflows", body)


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
            record_id = result.get("record", {}).get("id", "")
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            body = (
                f'<div class="max-w-lg mx-auto text-center py-12">'
                f'<div class="text-5xl mb-4">✅</div>'
                f'<h1 class="text-xl font-bold text-gray-900 mb-2">{html.escape(wf["name"])} complete!</h1>'
                f'<p class="text-gray-500 text-sm mb-6">Your {html.escape(entity_name)} record has been created successfully.</p>'
                f'<div class="flex items-center justify-center gap-3 flex-wrap">'
                f'<a href="/ui/entities/{safe_entity}" class="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">View all {html.escape(entity_name)} records →</a>'
                f'<a href="/ui/workflows/{safe_entity}/{html.escape(quote(workflow_id, safe=""), quote=True)}" class="px-5 py-2.5 border border-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-50 transition-colors">Start again</a>'
                f'<a href="/ui/workflows" class="px-5 py-2.5 border border-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-50 transition-colors">All workflows</a>'
                f'</div></div>'
            )
            return 200, _html_page(wf["name"], body)
        else:
            error = result.get("error") or "Failed to create record"
            step_index = total_steps - 1  # Stay on last step

    step = steps[min(step_index, total_steps - 1)]
    step_fields = step.get("fields", [])
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)

    # Progress bar
    pct = int((step_index / total_steps) * 100)
    step_indicators = "".join(
        f'<div class="flex items-center gap-1.5 text-xs font-medium '
        f'{("text-blue-600" if i == step_index else "text-gray-400 opacity-60")}">'
        f'<span class="w-5 h-5 rounded-full flex items-center justify-center text-white text-xs '
        f'{("bg-blue-600" if i < step_index else "bg-blue-600" if i == step_index else "bg-gray-200 text-gray-500")}">'
        f'{("✓" if i < step_index else str(i + 1))}</span>'
        f'<span class="hidden sm:block">{html.escape(steps[i]["title"])}</span></div>'
        + (f'<div class="flex-1 h-px bg-gray-200 mx-1"><div class="h-px bg-blue-600 transition-all" style="width:{("100%" if i < step_index else "0%")}"></div></div>'
           if i < total_steps - 1 else "")
        for i in range(total_steps)
    )

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

    body = (
        # Breadcrumb
        f'<nav class="flex items-center gap-2 text-sm mb-6 text-gray-500">'
        f'<a href="/ui" class="hover:text-blue-600">Application</a><span>/</span>'
        f'<a href="/ui/workflows" class="hover:text-blue-600">Workflows</a><span>/</span>'
        f'<span class="font-semibold text-gray-900">{html.escape(wf["name"])}</span></nav>'
        # Header
        f'<div class="max-w-2xl mx-auto">'
        f'<div class="text-center mb-8">'
        f'<div class="text-4xl mb-3">{html.escape(wf["icon"])}</div>'
        f'<h1 class="text-xl font-bold text-gray-900">{html.escape(wf["name"])}</h1>'
        f'<p class="text-sm text-gray-500 mt-1">{html.escape(wf["description"])}</p>'
        f'</div>'
        # Step progress
        f'<div class="flex items-center gap-0 mb-8 px-2">{step_indicators}</div>'
        # Step card
        f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">'
        f'<div class="px-6 py-4 border-b border-gray-100 bg-gray-50">'
        f'<h2 class="font-semibold text-gray-900">Step {step_index + 1} of {total_steps}: {html.escape(step["title"])}</h2>'
        f'<p class="text-sm text-gray-500 mt-0.5">{html.escape(step.get("subtitle", ""))}</p>'
        f'</div>'
        f'<div class="p-6">'
        f'{error_html}'
        f'<form method="post" action="{next_url}" class="space-y-4">'
        f'{hidden_fields}'
        f'{step_inputs}'
        f'<div class="flex items-center justify-between pt-4 border-t border-gray-100 mt-6">'
        + (f'<a href="/ui/workflows/{safe_entity}/{safe_wf_id}/step/{step_index - 1}" class="px-4 py-2 text-sm text-gray-500 hover:text-gray-700 transition-colors">← Back</a>'
           if step_index > 0 else f'<a href="/ui/workflows" class="px-4 py-2 text-sm text-gray-500 hover:text-gray-700 transition-colors">← Cancel</a>')
        + f'<button type="submit" class="px-6 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 active:bg-blue-800 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">{next_label}</button>'
        f'</div></form></div></div></div>'
    )
    return 200, _html_page(wf["name"], body)


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


def _ui_database_catalog_html() -> tuple[int, str]:
    status = database_status()
    status_code = 200 if status["valid"] else 422
    status_label = "valid" if status["valid"] else "invalid"
    database_items: list[str] = []
    for database in list_databases():
        database_name = str(database.get("name", "database"))
        schema_rows: list[str] = []
        for schema in database.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            table_names = ", ".join(
                html.escape(str(table.get("name", "table")))
                for table in schema.get("tables", [])
            ) or "no tables"
            schema_rows.append(
                f"<li><strong>{html.escape(schema_name)}</strong>: {table_names}</li>"
            )
        schemas_html = "".join(schema_rows) or "<li>No schemas declared.</li>"
        database_items.append(
            f"<section><h2>{html.escape(database_name)}</h2>"
            f'<p><a href="/databases/{html.escape(database_name, quote=True)}/schemas">'
            "Schema JSON</a></p>"
            f"<ul>{schemas_html}</ul></section>"
        )
    databases_html = "".join(database_items) or "<p>No databases declared.</p>"
    validation_html = html.escape(json.dumps(status["validation"], indent=2, sort_keys=True))
    body = (
        "<h1>Databases</h1>"
        f"<p>Status: <strong>{html.escape(status_label)}</strong>; "
        f"{status['database_count']} database(s), "
        f"{status['schema_count']} schema(s), "
        f"{status['table_count']} table(s), "
        f"{status['reference_count']} reference(s).</p>"
        '<nav><a href="/ui">Application UI</a> | '
        '<a href="/databases">Database JSON</a> | '
        '<a href="/databases/status">Status JSON</a> | '
        '<a href="/relationships">Relationships</a></nav>'
        f"{databases_html}"
        f"<h2>Validation</h2><pre>{validation_html}</pre>"
    )
    return status_code, _html_page("Databases", body)


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


def _ui_field_input_html(field: Dict[str, Any], entity_name: str = "") -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    safe_label = html.escape(field_name)
    expected = _json_schema_type(str(field.get("type", "any")))

    # Foreign key → dropdown populated from the related entity's records
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target)
        return (
            f'<label>{safe_label}</label>'
            f'<select name="{safe_name}">'
            f'{opts}'
            f'</select><br>'
        )

    if expected == "boolean":
        return (
            f'<input type="hidden" name="{safe_name}" value="false">'
            f'<label>{safe_label} '
            f'<input type="checkbox" name="{safe_name}" value="true"></label><br>'
        )
    if expected == "integer":
        attributes = 'type="number" step="1"'
    elif expected == "number":
        attributes = 'type="number" step="any"'
    elif field.get("type", "").lower() in {"date", "datetime", "timestamp"}:
        attributes = 'type="date"'
    else:
        attributes = 'type="text"'
    return f'<label>{safe_label} <input name="{safe_name}" {attributes}></label><br>'


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
    return "".join(_ui_field_input_html(field, entity_name) for field in fields)


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
        label = html.escape(col.replace("_id", "").replace("_", " ").title())
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
        action = (
            f'<div class="flex items-center gap-3 justify-end opacity-0 group-hover/row:opacity-100 transition-opacity">'
            f'<a href="/ui/entities/{safe_entity}/{record_id}"'
            f' class="text-xs font-medium text-apg-primary hover:underline whitespace-nowrap">View →</a>'
            f'<form method="post" action="/ui/entities/{safe_entity}/records/{record_id}/delete" class="inline">'
            f'<input type="hidden" name="expected_revision" value="{revision}">'
            f'<button type="submit" onclick="return confirm(this.dataset.msg)" data-msg="Delete this record?"'
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
        'if(!confirm("Delete "+cc.length+" record(s)? This cannot be undone."))return;'
        'var ids=Array.from(cc).map(function(c){return c.dataset.rowId;}).join(",");'
        'var entity=document.getElementById("apg-bulk-bar").dataset.entity;'
        'var fd=new FormData();fd.append("ids",ids);'
        'fetch("/ui/entities/"+entity+"/records/bulk_delete",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body:"ids="+encodeURIComponent(ids)})'
        '.then(function(r){if(r.redirected||r.ok)window.location.reload();});'
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

    # Build query for sort/pagination (not q — we do q client-side)
    base_query: Dict[str, list[str]] = {}
    if sort_field:
        base_query["sort"] = [sort_field]
        base_query["order"] = [sort_dir]
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
    label = html.escape(field_name.replace("_id", "").replace("_", " ").title())
    edit_url = f"/ui/entities/{safe_entity}/{safe_record_id}/fields/{html.escape(field_name)}/edit"
    return (
        f'<div id="{fld_id}" class="py-3 border-b border-gray-50 last:border-0 group/field">'
        f'<dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">{label}</dt>'
        f'<dd class="flex items-center justify-between gap-2 min-h-6">'
        f'<span class="text-sm text-gray-900 break-words">{display}</span>'
        f'<button hx-get="{edit_url}" hx-target="#{fld_id}" hx-swap="outerHTML"'
        f' class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"'
        f' title="Edit {html.escape(field_name)}">'
        f'<svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">'
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
    label = html.escape(field_name.replace("_id", "").replace("_", " ").title())
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
    # Run detail
    detail_html = ""
    if run_id:
        try:
            run = get_workflow_run(run_id)
        except KeyError:
            run = None
        if run:
            trace = run.get("trace", [])
            trace_rows = []
            for t in trace:
                status_cls = (
                    "bg-green-100 text-green-800" if t.get("status") == "completed"
                    else "bg-red-100 text-red-800" if t.get("status") in {"failed", "circuit_open"}
                    else "bg-yellow-100 text-yellow-800"
                )
                attempts = t.get("attempts", [])
                attempts_html = f', {len(attempts)} attempt(s)' if len(attempts) > 1 else ""
                trace_rows.append(
                    f'<tr class="border-b border-gray-50">'
                    f'<td class="px-4 py-2 text-xs font-mono text-gray-500">{t.get("index", "")}</td>'
                    f'<td class="px-4 py-2 text-sm font-medium">{html.escape(str(t.get("step", "")))}</td>'
                    f'<td class="px-4 py-2"><span class="px-2 py-0.5 rounded-full text-xs font-semibold {status_cls}">{html.escape(str(t.get("status", "")))}</span></td>'
                    f'<td class="px-4 py-2 text-xs text-gray-500">{html.escape(str(t.get("timeout_spec", "")))}{attempts_html}</td>'
                    f'</tr>'
                )
            # Journal timeline
            journal = _get_journal(run_id)
            ev_color_map = {"step_completed": "bg-green-400", "step_failed": "bg-red-400", "saga_compensating": "bg-orange-400", "signal_received": "bg-purple-400"}
            journal_items = []
            for ev in journal:
                ev_color = ev_color_map.get(ev["event_type"], "bg-gray-400")
                journal_items.append(
                    f'<li class="ml-6 mb-3 relative">'
                    f'<span class="absolute flex items-center justify-center w-6 h-6 rounded-full -left-3 ring-2 ring-white {ev_color}"></span>'
                    f'<div class="pl-1">'
                    f'<p class="text-xs font-semibold text-gray-900">#{ev["seq"]} {html.escape(ev["event_type"].replace("_"," ").title())}</p>'
                    f'<p class="text-xs text-gray-500">Step: {html.escape(str(ev["step"]))} · {html.escape(ev["ts"][:19].replace("T"," "))} UTC</p>'
                    f'<p class="text-xs font-mono text-gray-300 truncate">{html.escape(ev["hash"][:16])}...</p>'
                    f'</div></li>'
                )
            journal_html = (
                f'<div class="px-4 py-3 border-t border-gray-100">'
                f'<h3 class="text-xs font-semibold text-gray-700 mb-2">Event Journal</h3>'
                f'<ol class="relative border-l border-gray-200 ml-3">'
                + ("".join(journal_items) if journal_items else '<li class="ml-6"><p class="text-xs text-gray-400">No journal events yet.</p></li>')
                + f'</ol></div>'
            )
            detail_html = (
                f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5">'
                f'<div class="px-4 py-3 border-b border-gray-100"><h2 class="text-sm font-semibold">Run: {html.escape(str(run_id))}'
                f' <span class="ml-2 text-xs text-gray-400">{html.escape(str(run.get("workflow","")))}</span></h2></div>'
                f'<table class="w-full text-sm"><thead class="bg-gray-50 text-xs font-semibold text-gray-500">'
                f'<tr><th class="px-4 py-2 text-left">#</th><th class="px-4 py-2 text-left">Step</th>'
                f'<th class="px-4 py-2 text-left">Status</th><th class="px-4 py-2 text-left">Notes</th></tr></thead>'
                f'<tbody>{" ".join(trace_rows)}</tbody></table>'
                + journal_html
                + f'</div>'
            )
    # Run list
    run_rows = []
    for r in sorted(runs, key=lambda x: str(x.get("id", "")), reverse=True)[:50]:
        rid = html.escape(str(r.get("id", "")))
        wf = html.escape(str(r.get("workflow", "")))
        st = html.escape(str(r.get("status", "")))
        sc = "bg-green-100 text-green-800" if st == "completed" else "bg-red-100 text-red-800" if st == "failed" else "bg-yellow-100 text-yellow-800"
        run_rows.append(
            f'<tr class="hover:bg-gray-50 border-b border-gray-50">'
            f'<td class="px-4 py-2 text-xs font-mono"><a href="/ui/debug/{rid}" class="text-apg-primary hover:underline">{rid}</a></td>'
            f'<td class="px-4 py-2 text-sm">{wf}</td>'
            f'<td class="px-4 py-2"><span class="px-2 py-0.5 rounded-full text-xs font-semibold {sc}">{st}</span></td>'
            f'<td class="px-4 py-2 text-xs text-gray-400">{len(r.get("trace", []))} steps</td>'
            f'</tr>'
        )
    # Circuit breakers section
    cb_rows = []
    for k, v in cb_status.items():
        st = v.get("state", "closed")
        sc = "bg-green-100 text-green-800" if st == "closed" else "bg-red-100 text-red-800" if st == "open" else "bg-yellow-100 text-yellow-800"
        cb_rows.append(
            f'<tr class="border-b border-gray-50">'
            f'<td class="px-4 py-2 text-xs font-mono">{html.escape(k)}</td>'
            f'<td class="px-4 py-2"><span class="px-2 py-0.5 rounded-full text-xs font-semibold {sc}">{st}</span></td>'
            f'<td class="px-4 py-2 text-xs tabular-nums">{v.get("failures", 0)}</td>'
            f'</tr>'
        )
    cb_section = (
        f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5">'
        f'<div class="px-4 py-3 border-b border-gray-100"><h2 class="text-sm font-semibold text-gray-900">Circuit Breakers</h2></div>'
        + (f'<table class="w-full text-sm"><thead class="bg-gray-50 text-xs font-semibold text-gray-500">'
           f'<tr><th class="px-4 py-2 text-left">Key</th><th class="px-4 py-2 text-left">State</th><th class="px-4 py-2 text-left">Failures</th></tr></thead>'
           f'<tbody>{" ".join(cb_rows)}</tbody></table>' if cb_rows else
           f'<p class="px-4 py-6 text-sm text-gray-400 text-center">No circuit breakers tripped.</p>')
        + f'</div>'
    )
    # Event subscriptions section
    sub_rows = [
        f'<tr class="border-b border-gray-50"><td class="px-4 py-2 text-xs font-mono">{html.escape(ev)}</td>'
        f'<td class="px-4 py-2 text-xs text-gray-600">{html.escape(", ".join(wfs))}</td></tr>'
        for ev, wfs in subs.items()
    ]
    sub_section = (
        f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5">'
        f'<div class="px-4 py-3 border-b border-gray-100"><h2 class="text-sm font-semibold text-gray-900">Event Subscriptions</h2></div>'
        + (f'<table class="w-full text-sm"><thead class="bg-gray-50 text-xs font-semibold text-gray-500">'
           f'<tr><th class="px-4 py-2 text-left">Event</th><th class="px-4 py-2 text-left">Subscribed Workflows</th></tr></thead>'
           f'<tbody>{" ".join(sub_rows)}</tbody></table>' if sub_rows else
           f'<p class="px-4 py-6 text-sm text-gray-400 text-center">No active subscriptions.</p>')
        + f'</div>'
    )
    body = (
        f'<nav class="flex items-center gap-2 text-sm mb-5 text-gray-500">'
        f'<a href="/ui" class="hover:text-apg-primary">Application</a><span>/</span>'
        f'<span class="font-semibold text-gray-900">Flow Debugger</span></nav>'
        + detail_html
        + f'<div class="bg-white rounded-xl border border-gray-200 shadow-sm mb-5">'
        + f'<div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">'
        + f'<h2 class="text-sm font-semibold text-gray-900">Workflow Runs</h2>'
        + f'<span class="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">{len(runs)} total</span></div>'
        + (f'<table class="w-full text-sm"><thead class="bg-gray-50 text-xs font-semibold text-gray-500">'
           f'<tr><th class="px-4 py-2 text-left">Run ID</th><th class="px-4 py-2 text-left">Workflow</th>'
           f'<th class="px-4 py-2 text-left">Status</th><th class="px-4 py-2 text-left">Steps</th></tr></thead>'
           f'<tbody>{" ".join(run_rows)}</tbody></table>' if run_rows else
           f'<p class="px-4 py-10 text-sm text-gray-400 text-center">No workflow runs yet.</p>')
        + f'</div>'
        + cb_section
        + sub_section
    )
    return 200, _html_page("Flow Debugger", body)


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
    description = html.escape(json.dumps(catalog[name], indent=2, sort_keys=True))
    result_html = _result_section(result, error)
    body = (
        '<nav><a href="/ui">Application</a> | <a href="/agents">Agent catalog</a></nav>'
        f"<h1>{html.escape(name)}</h1>"
        f"<pre>{description}</pre>"
        f'<form method="post" action="{action}">'
        '<label>Message <input name="message" type="text"></label><br>'
        '<label>Payload JSON<br><textarea name="payload_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Invoke</button>'
        '</form>'
        f"{result_html}"
    )
    return 200, _html_page(name, body)


def _ui_capability_console_html(name: str, result: Dict[str, Any] | None = None, error: str = "") -> tuple[int, str]:
    app = describe_application()
    capabilities = app.get("capability_descriptions", {})
    if name not in capabilities:
        return 404, _html_page("Unknown capability", f"<h1>Unknown capability</h1><p>{html.escape(name)}</p>")
    description = html.escape(json.dumps(capabilities[name], indent=2, sort_keys=True))
    safe_name = html.escape(name, quote=True)
    result_html = _result_section(result, error)
    body = (
        '<nav><a href="/ui">Application</a> | <a href="/capabilities">Capability catalog</a></nav>'
        f"<h1>{html.escape(name)}</h1>"
        f"<pre>{description}</pre>"
        f'<form method="post" action="/ui/capabilities/{safe_name}/rules/evaluate">'
        '<h2>Evaluate Rules</h2>'
        '<label>Context JSON<br><textarea name="context_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Evaluate</button>'
        '</form>'
        f'<form method="post" action="/ui/capabilities/{safe_name}/configuration/resolve">'
        '<h2>Resolve Configuration</h2>'
        '<label>Overrides JSON<br><textarea name="configuration_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Resolve</button>'
        '</form>'
        f'<form method="post" action="/ui/capabilities/{safe_name}/approval/plan">'
        '<h2>Plan Approval</h2>'
        '<label>Context JSON<br><textarea name="context_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Plan</button>'
        '</form>'
        f"{result_html}"
    )
    return 200, _html_page(name, body)


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
                pass
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
_APG_PG_URL: str | None = _os_env.environ.get("APG_PG_URL") or _os_env.environ.get("DATABASE_URL") or None


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
        pass


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
        pass
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


_load_record_store()


class ApplicationRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, response: Dict[str, Any]) -> None:
        body = _json_bytes(response)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html_payload: str, hx_trigger: dict | None = None) -> None:
        body = html_payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if hx_trigger:
            self.send_header("HX-Trigger", json.dumps(hx_trigger))
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, status: int, location: str) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorize_mutation(self) -> bool:
        if _authorized(self.headers):
            return True
        status, response = _auth_failure_payload()
        self._send_json(status, response)
        return False

    def _setup_tenant(self) -> None:
        tid = self.headers.get("X-APG-Tenant") or self.headers.get("X-Tenant-ID")
        _TENANT_LOCAL.tenant_id = tid or None

    def do_GET(self) -> None:
        self._setup_tenant()
        path, _, raw_query = self.path.partition("?")
        query = parse_qs(raw_query, keep_blank_values=True)
        if path in ("/", "/home"):
            body = _landing_page_html().encode("utf-8")
            status = 200
            content_type = "text/html; charset=utf-8"
        elif path == "/theme.css":
            body = theme_stylesheet().encode("utf-8")
            status = 200
            content_type = "text/css; charset=utf-8"
        elif path.startswith("/entities/") and path.endswith("/records.csv"):
            entity_name = path.split("/")[2]
            body = _csv_export_body(entity_name)
            status = 200
            content_type = "text/csv; charset=utf-8"
        elif path == "/ui" or path.startswith("/ui/"):
            status, html_payload = _ui_payload(path, query)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif _capability_screen(path) is not None:
            status, html_payload = _capability_screen_payload(path)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif _application_screen(path) is not None:
            status, html_payload = _application_screen_payload(path)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            status, payload = _route_payload(path, query)
            body = _json_bytes(payload)
            content_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self._setup_tenant()
        path = self.path.split("?", 1)[0]
        if not self._authorize_mutation():
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b"{}"
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type == "application/x-www-form-urlencoded":
                parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
                payload = {"record": {key: values[-1] if values else "" for key, values in parsed.items()}}
            else:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            if path.startswith("/ui/") and content_type == "application/x-www-form-urlencoded":
                status, response = _ui_post_payload(path, payload)
                if status in {302, 303}:
                    self._send_redirect(status, str(response["location"]))
                    return
                if "html" in response:
                    hx_trigger = response.get("hx_trigger")
                    self._send_html(status, str(response["html"]), hx_trigger=hx_trigger)
                    return
                self._send_html(status, _ui_error_payload(path, response))
                return
            else:
                status, response = _post_payload(path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            status, response = 400, {"error": "invalid_json", "message": str(error)}
        self._send_json(status, response)

    def do_PUT(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorize_mutation():
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            status, response = _put_payload(path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            status, response = 400, {"error": "invalid_json", "message": str(error)}
        self._send_json(status, response)

    def do_DELETE(self) -> None:
        path = self.path
        if not self._authorize_mutation():
            return
        status, response = _delete_record_payload(path)
        self._send_json(status, response)

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("APG_DEBUG") == "1":
            super().log_message(format, *args)


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
    server = HTTPServer((resolved_host, resolved_port), ApplicationRequestHandler)
    print(f"{MODULE_NAME} listening on http://{resolved_host}:{resolved_port}", flush=True)
    server.serve_forever()


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
