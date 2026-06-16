grammar apg;

options {
    language = Python3;
}

// ========================================
// COMPLETE APG v11 GRAMMAR
// Full implementation of all demonstrated features
// ========================================

program
    : module_declaration? (import_statement | include_statement | export_statement | entity)* EOF
    ;

// MODULE SYSTEM
module_declaration
    : 'module' module_name version_tag? '{' module_metadata? '}'
    ;

module_name
    : member_name ('.' member_name)*
    ;

module_metadata
    : module_property*
    ;

module_property
    : 'description' ':' STRING ';'
    | 'author' ':' STRING ';'
    | 'license' ':' STRING ';'
    | 'dependencies' ':' '[' dependency_list? ']' ';'
    | 'exports' ':' '[' export_list? ']' ';'
    | 'private' ':' '[' private_list? ']' ';'
    ;

dependency_list
    : dependency (',' dependency)*
    ;

dependency
    : module_name version_constraint?
    ;

version_constraint
    : '@' version_range
    ;

version_range
    : SEMVER
    | '>=' SEMVER
    | '<=' SEMVER
    | '~' SEMVER    // Compatible version
    | '^' SEMVER    // Minor version compatible
    | SEMVER '..' SEMVER  // Range
    ;

export_list
    : export_item (',' export_item)*
    ;

export_item
    : IDENTIFIER alias?
    | '*'  // Export all public items
    ;

private_list
    : IDENTIFIER (',' IDENTIFIER)*
    ;

alias
    : 'as' IDENTIFIER
    ;

// IMPORT AND INCLUDE SYSTEM
import_statement
    : 'import' module_path import_options? ('as' IDENTIFIER)? ';'
    | 'from' module_path 'import' import_list ';'
    ;

include_statement
    : 'include' file_path include_options? ';'
    | 'include' '<' IDENTIFIER '>' ';'  // Standard library include
    ;

export_statement
    : 'export' export_declaration ';'
    ;

export_declaration
    : entity
    | 'const' IDENTIFIER '=' value_expr
    | 'type' IDENTIFIER '=' type_annotation
    | 'default' entity
    ;

module_path
    : IDENTIFIER ('.' IDENTIFIER)*
    | STRING  // Relative or absolute module path
    ;

file_path
    : STRING
    | IDENTIFIER ('/' IDENTIFIER)*
    ;

import_options
    : '{' import_option (',' import_option)* '}'
    ;

import_option
    : 'lazy' ':' BOOLEAN
    | 'version' ':' version_range
    | 'optional' ':' BOOLEAN
    | 'cache' ':' BOOLEAN
    ;

include_options
    : '{' include_option (',' include_option)* '}'
    ;

include_option
    : 'once' ':' BOOLEAN
    | 'conditional' ':' expression
    | 'namespace' ':' IDENTIFIER
    ;

import_list
    : import_item (',' import_item)*
    | '*'
    ;

import_item
    : IDENTIFIER ('as' IDENTIFIER)?
    | '{' IDENTIFIER (',' IDENTIFIER)* '}' ('as' IDENTIFIER)?  // Destructuring
    ;

// UNIVERSAL ENTITY PATTERN with full support
entity
    : decorator* entity_type IDENTIFIER inheritance? version_tag? '{' entity_body '}' ';'?
    ;

decorator
    : '@' IDENTIFIER ('(' args? ')')?
    ;

entity_type
    : 'agent' | 'team' | 'agent_team' | 'robot' | 'sensor' | 'camera' | 'actuator' | 'drone'
    | 'chat' | 'llm' | 'db' | 'table' | 'biz' | 'flow' | 'workflow' | 'rule'
    | 'report' | 'form' | 'erp' | 'protocol' | 'chain' | 'master'
    | 'auto_system' | 'sense' | 'deploy' | 'process' | 'stream' | 'swarm'
    // Composable APG platform surface
    | 'app' | 'application' | 'capability' | 'capability_contract' | 'capability_pack'
    | 'composition' | 'contract' | 'rule_set' | 'policy' | 'guardrail'
    // ERP and enterprise application composition surface
    | 'erp_module' | 'erp_component' | 'ledger' | 'finance' | 'procurement'
    | 'inventory' | 'warehouse' | 'manufacturing' | 'sales' | 'crm'
    | 'hr' | 'payroll' | 'fixed_assets' | 'project_accounting'
    // First-class AI agent composition surface
    | 'agent_runtime' | 'agent_tool' | 'agent_memory' | 'agent_handoff'
    | 'prompt' | 'model' | 'tool' | 'memory_store' | 'handoff'
    // OSINT and Intelligence Extensions
    | 'source' | 'intel' | 'analyze' | 'graph' | 'pattern' | 'validate'
    | 'fuse' | 'verify' | 'detect' | 'profile' | 'map' | 'score'
    | 'share' | 'comply' | 'protect' | 'ethics' | 'opsec' | 'fairness'
    | 'standards' | 'geo' | 'track' | 'temporal' | 'context' | 'correlate'
    | 'hunt' | 'monitor'
    // Business Calculations and Reporting Extensions
    | 'calc' | 'formula' | 'metric' | 'kpi' | 'dashboard' | 'statement'
    // Digital Twin and Industrial Monitoring Extensions
    | 'twin' | 'simulate' | 'mirror' | 'virtual' | 'physical' | 'sync'
    | 'anomaly' | 'vision' | 'inspect' | 'quality' | 'defect' | 'classify'
    | 'production' | 'line' | 'machine' | 'process_monitor' | 'predictive'
    | 'maintenance' | 'failure' | 'threshold' | 'alert' | 'optimize'
    // Developer Ergonomics and System Extensions
    | 'test' | 'mock' | 'stub' | 'fixture' | 'scenario' | 'benchmark'
    | 'config' | 'env' | 'settings' | 'secrets' | 'vault' | 'registry'
    | 'logger' | 'tracer' | 'profiler' | 'debugger' | 'instrumenter'
    | 'notify' | 'alert_manager' | 'messenger' | 'publisher' | 'subscriber'
    | 'layout' | 'ui' | 'component' | 'widget' | 'screen' | 'view'
    | 'middleware' | 'interceptor' | 'filter' | 'transformer' | 'mapper'
    | 'cache' | 'session' | 'store' | 'repository' | 'gateway' | 'proxy'
    // Type system and data modeling
    | 'enum' | 'interface' | 'type_alias' | 'struct'
    // State machines and event sourcing
    | 'statemachine' | 'state_machine' | 'fsm'
    | 'event_schema' | 'event_store' | 'projection' | 'aggregate'
    // Database lifecycle
    | 'migration' | 'seed' | 'fixture_data'
    // Deployment and platform
    | 'deployment_strategy' | 'deployment_pattern'
    | 'marketplace' | 'ecommerce' | 'platform'
    // Reporting and analytics
    | 'pipeline' | 'etl' | 'dbt_model'
    // User-defined / decorator-backed entity kinds — any identifier is valid
    | IDENTIFIER
    ;

inheritance
    : 'extends' IDENTIFIER
    ;

version_tag
    : 'version' SEMVER
    ;

entity_body
    : entity_member*
    ;

entity_member
    : capability_contract_block
    | erp_component_block
    | agent_composition_block
    | rule_engine_block
    | screen_contract_block
    | ui_contract_block
    | theme_contract_block
    | stream_runtime_block
    | i18n_contract_block
    | config_item
    | behavior_item
    | annotation
    | method_def
    | nested_entity
    | class_def
    | exception_def
    | variable_declaration
    | database_schema  // DBML integration for db and table entities
    | form_layout           // Form layout sublanguage for form entities
    | state_transition      // State machine transition for statemachine entities
    | enum_variant_decl     // Enum variant for enum entities
    ;

// member_name: accept IDENTIFIER or any keyword that may also be used as a field name.
// APG field names like 'name', 'description', 'path', 'action' etc. conflict with
// implicit keyword tokens created by the grammar's many inline string literals.
// Listing them here as a "soft keyword" rule restores their use as field names.
member_name
    : IDENTIFIER
    // Common field names shadowed by contract-member keywords
    | 'name' | 'description' | 'title' | 'label' | 'code' | 'note' | 'author'
    | 'path' | 'route' | 'component' | 'permission' | 'nav_group'
    | 'action' | 'status' | 'state' | 'value' | 'source' | 'target' | 'via'
    | 'email' | 'phone' | 'address' | 'url' | 'format' | 'mode' | 'style'
    | 'required' | 'optional' | 'default' | 'enabled' | 'disabled'
    | 'key' | 'group' | 'tag' | 'category' | 'priority' | 'rank' | 'score'
    | 'date' | 'time' | 'start' | 'end' | 'count' | 'total' | 'level'
    | 'data' | 'payload' | 'body' | 'header' | 'content' | 'message' | 'text'
    | 'user' | 'steps' | 'stage' | 'type' | 'id' | 'version'
    // Database/engine keywords
    | 'engine' | 'postgresql' | 'mysql' | 'sqlite' | 'mongodb' | 'redis'
    | 'alembic' | 'schema' | 'table' | 'column' | 'index' | 'migration'
    // Additional missing values
    | 'query' | 'configure' | 'all' | 'check' | 'fields' | 'format'
    | 'sequential' | 'conditional' | 'parallel' | 'concurrent' | 'serial'
    | 'encode' | 'decode' | 'encrypt' | 'decrypt' | 'sign' | 'verify'
    | 'generate' | 'validate' | 'transform' | 'normalize' | 'sanitize'
    | 'connect' | 'disconnect' | 'authenticate' | 'authorize' | 'revoke'
    | 'allocate' | 'deallocate' | 'reserve' | 'release' | 'acquire'
    // Rule body keywords
    | 'when' | 'then' | 'effect' | 'exception' | 'applies_to'
    | 'effective_from' | 'effective_to' | 'approval' | 'priority'
    // Contract fields that also appear as config keys
    | 'rule_engine' | 'rules' | 'ui' | 'theme' | 'streaming'
    | 'i18n' | 'localization' | 'screens' | 'agents' | 'handoffs'
    | 'erp_modules' | 'components' | 'approvals' | 'master_data'
    | 'business_rules' | 'configuration' | 'provides' | 'requires'
    // More entity/capability reference names
    | 'identity_verification' | 'session_management' | 'role_assignment'
    | 'jwt_tokens' | 'audit_events' | 'compliance_trail' | 'change_history'
    | 'customer_records' | 'credit_profiles' | 'kyc_status'
    | 'sales_dashboard' | 'order_management' | 'customer_engagement'
    // Additional value keywords commonly used as identifiers
    | 'supported_languages' | 'default_language' | 'fallback_language'
    | 'profile' | 'portfolio' | 'dashboard' | 'workbench' | 'portal'
    | 'hub' | 'centre' | 'center' | 'desk' | 'console' | 'panel'
    | 'interface' | 'facade' | 'proxy' | 'adapter' | 'bridge' | 'bus'
    // Entity-type keywords usable as identifiers in non-entity contexts
    | 'workflow' | 'agent' | 'table' | 'capability' | 'capability_contract'
    | 'capability_pack' | 'composition' | 'contract' | 'rule_set' | 'policy'
    | 'guardrail' | 'app' | 'application' | 'form' | 'screen'
    | 'enum' | 'statemachine' | 'migration' | 'deployment' | 'marketplace'
    | 'platform' | 'pipeline' | 'db' | 'database' | 'flow' | 'rule'
    // Common APG service/system identifiers
    | 'auth' | 'audl' | 'ntfy' | 'wflo' | 'colb' | 'mten'
    // Common field/value names not yet in member_name
    | 'tags' | 'routes' | 'currency' | 'region' | 'tenant' | 'locale'
    | 'system' | 'notifications' | 'notification' | 'stream' | 'streams'
    // UI shell values  
    | 'python' | 'react' | 'mobile' | 'cli'
    // Stream processor values
    | 'bytewax' | 'bytewax_streams'
    // Memory/vector types
    | 'vector' | 'embedding' | 'redis' | 'postgres' | 'sqlite' | 'chroma' | 'pinecone'
    // Runtime/deployment values
    | 'container' | 'docker' | 'kubernetes' | 'serverless' | 'lambda' | 'local'
    // Agent runtime values (from agent_runtime_ref)
    | 'codex' | 'codex_cli' | 'claude' | 'claude_code' | 'opencode'
    | 'open_code' | 'pi' | 'openai' | 'ollama'
    // CSS / layout field names (from style/layout sub-language)
    | 'position' | 'display' | 'overflow' | 'cursor' | 'transition' | 'animation'
    | 'transform' | 'filter' | 'gradient' | 'shadow' | 'clip' | 'mask'
    // Common field names not yet covered
    | 'escalation' | 'dimensions' | 'ledger' | 'general_ledger'
    | 'en' | 'fr' | 'de' | 'es' | 'pt' | 'ja' | 'ko' | 'zh'
    | 'active' | 'inactive' | 'pending' | 'blocked' | 'suspended' | 'archived'
    | 'draft' | 'published' | 'reviewed' | 'approved' | 'rejected' | 'closed'
    // Additional common values used as identifiers
    | 'summary' | 'export' | 'import' | 'detail' | 'overview' | 'list' | 'search'
    | 'create' | 'update' | 'delete' | 'read' | 'write' | 'approve' | 'reject'
    | 'submit' | 'cancel' | 'complete' | 'process' | 'start' | 'stop' | 'pause'
    | 'resume' | 'reset' | 'refresh' | 'sync' | 'push' | 'pull' | 'merge'
    | 'connection' | 'socket' | 'port' | 'host' | 'scheme' | 'protocol'
    | 'username' | 'password' | 'token' | 'secret' | 'key' | 'salt'
    | 'algorithm' | 'cipher' | 'digest' | 'hash' | 'checksum' | 'signature'
    | 'metrics' | 'quota' | 'timeout' | 'retry' | 'fallback'
    | 'webhook' | 'endpoint' | 'service' | 'domain' | 'namespace'
    // Rule decision keywords (used as values in rule action fields)
    | 'require_review' | 'allow' | 'deny' | 'warn' | 'audit' | 'reject'
    // Common field names in tables/configs
    | 'metadata' | 'required' | 'department' | 'division' | 'unit'
    | 'quantity' | 'amount' | 'balance' | 'percentage' | 'rate' | 'price'
    | 'category' | 'subcategory' | 'product' | 'order' | 'invoice'
    | 'supplier' | 'vendor' | 'customer' | 'employee' | 'account'
    | 'project' | 'task' | 'issue' | 'ticket' | 'case' | 'record'
    | 'document' | 'report' | 'file' | 'image' | 'video' | 'audio'
    // Additional keywords from examples
    | 'forecast' | 'icon' | 'procurement' | 'payroll' | 'inventory'
    | 'manufacturing' | 'supply_chain' | 'general_ledger' | 'accounts_payable'
    | 'accounts_receivable' | 'fixed_assets' | 'project_accounting'
    | 'time_attendance' | 'benefits' | 'asset_management' | 'maintenance'
    | 'service_management' | 'reporting' | 'analytics' | 'budgeting'
    | 'tax' | 'compliance' | 'warehouse' | 'order_management'
    | 'materials_planning' | 'production_planning' | 'purchase_orders'
    | 'supplier_management' | 'priority' | 'parallel' | 'timeout' | 'retry'
    | 'condition' | 'filter' | 'icon' | 'variant' | 'size' | 'alignment'
    | 'share' | 'back' | 'forward' | 'reload' | 'grid' | 'config'
    | 'from' | 'to' | 'top' | 'bottom' | 'left' | 'right' | 'center'
    | 'x' | 'y' | 'z' | 'f' | 'n' | 'k' | 'r' | 's' | 't' | 'v' | 'w'
    | 'up' | 'down' | 'smooth' | 'auto' | 'behavior'
    | 'px' | 'em' | 'rem' | 'vh' | 'vw' | 'vmin' | 'vmax' | 'pt' | 'pc'
    | 'cm' | 'mm' | 'deg' | 'rad' | 'grad' | 'turn' | 'ms' | 'min' | 'h'
    | 'max' | 'min' | 'avg' | 'sum' | 'new' | 'return' | 'return_type'
    | 'quality' | 'immutable' | 'stable' | 'volatile'
    | 'stripe' | 'compact' | 'color' | 'color_scheme' | 'font' | 'border'
    | 'schedule' | 'download' | 'next' | 'finish' | 'save_progress'
    | 'location' | 'address' | 'country' | 'city' | 'state' | 'zip'
    | 'phone_number' | 'mobile' | 'fax' | 'website' | 'social'
    | 'notes' | 'comments' | 'remarks' | 'observation' | 'description'
    // Contract block names used as capability/agent field names
    // Note: do NOT include entity_type keywords here (agent, table, workflow, app, etc.)
    | 'rules' | 'ui' | 'streaming' | 'tools' | 'memory'
    | 'input' | 'output' | 'configuration' | 'provides' | 'requires'
    | 'capabilities' | 'agents' | 'handoffs' | 'runtimes'
    | 'erp_modules' | 'components' | 'screens' | 'i18n'
    | 'approvals' | 'master_data' | 'business_rules' | 'role' | 'model'
    // Deployment and runtime fields
    | 'deployments' | 'runtime' | 'deployment' | 'processor'
    | 'backend' | 'shell' | 'trigger' | 'handler' | 'condition'
    // Layout and UI fields
    | 'layout' | 'contains' | 'composes' | 'binds' | 'actions' | 'events'
    | 'relationships' | 'width' | 'height' | 'color' | 'font' | 'border'
    // Workflow body fields (NOT the workflow entity keyword itself)
    | 'assignments' | 'guards' | 'timers' | 'waits' | 'retry_policy'
    | 'compensation' | 'human_tasks' | 'stages' | 'process'
    | 'saga' | 'is_saga' | 'emit_events' | 'subscribe_events'
    // Auto-generated: all remaining parser keywords
    | 'actuator' | 'agent_handoff' | 'agent_memory' | 'agent_runtime' | 'agent_team' | 'agent_tool' | 'aggregate' | 'alert'
    | 'alert_manager' | 'analyze' | 'anomaly' | 'as' | 'auto_system' | 'benchmark' | 'biz' | 'cache'
    | 'calc' | 'camera' | 'chain' | 'chat' | 'classify' | 'comply' | 'const' | 'context'
    | 'correlate' | 'crm' | 'dbt_model' | 'debugger' | 'defect' | 'dependencies' | 'deploy' | 'deployment_pattern'
    | 'deployment_strategy' | 'detect' | 'drone' | 'ecommerce' | 'env' | 'erp' | 'erp_component' | 'erp_module'
    | 'ethics' | 'etl' | 'event_schema' | 'event_store' | 'exports' | 'extends' | 'failure' | 'fairness'
    | 'finance' | 'fixture' | 'fixture_data' | 'formula' | 'fsm' | 'fuse' | 'gateway' | 'geo'
    | 'graph' | 'handoff' | 'hr' | 'hunt' | 'include' | 'inspect' | 'instrumenter' | 'intel'
    | 'interceptor' | 'kpi' | 'lazy' | 'license' | 'line' | 'llm' | 'logger' | 'machine'
    | 'map' | 'mapper' | 'master' | 'memory_store' | 'messenger' | 'metric' | 'middleware' | 'mirror'
    | 'mock' | 'module' | 'monitor' | 'notify' | 'once' | 'opsec' | 'optimize' | 'pattern'
    | 'physical' | 'predictive' | 'private' | 'process_monitor' | 'production' | 'profiler' | 'projection' | 'prompt'
    | 'protect' | 'publisher' | 'registry' | 'repository' | 'robot' | 'sales' | 'scenario' | 'secrets'
    | 'seed' | 'sense' | 'sensor' | 'session' | 'settings' | 'simulate' | 'standards' | 'state_machine'
    | 'statement' | 'store' | 'struct' | 'stub' | 'subscriber' | 'swarm' | 'team' | 'temporal'
    | 'test' | 'threshold' | 'tool' | 'tracer' | 'track' | 'transformer' | 'twin' | 'type_alias'
    | 'vault' | 'view' | 'virtual' | 'vision' | 'widget'
    // Auto-generated: remaining parser keywords
    | 'Any' | 'DELETE' | 'Dict' | 'GET' | 'GHz' | 'GPa' | 'GW' | 'Hz'
    | 'List' | 'MHz' | 'MPa' | 'MV' | 'MW' | 'PATCH' | 'POST' | 'PUT'
    | 'Pa' | 'ab_testing' | 'absolute' | 'acceptance_test' | 'access_policy' | 'accessibility' | 'accessibility_test' | 'accordion'
    | 'adaptive' | 'after' | 'aggregation' | 'ai_assisted' | 'alerting' | 'allow_tenant_overrides' | 'alpha' | 'alphanumeric'
    | 'alt' | 'alternate' | 'alternate_reverse' | 'and' | 'announcement_system' | 'anomaly_score' | 'apg' | 'api'
    | 'api_endpoints' | 'api_gateway' | 'api_key' | 'api_prefix' | 'apis' | 'approval_required' | 'approvers' | 'aqua'
    | 'aria_describedby' | 'aria_label' | 'array' | 'asc' | 'assert' | 'async' | 'at' | 'atm'
    | 'attribute' | 'audit_logging' | 'auto_increment' | 'auto_resolution' | 'auto_spec' | 'autocomplete' | 'await' | 'aws_secrets'
    | 'azure_keyvault' | 'backdrop_filter' | 'background' | 'background_checks' | 'backup' | 'backwards' | 'badge' | 'bar'
    | 'baseline' | 'basic' | 'bearer' | 'before' | 'before_all' | 'before_each' | 'begin' | 'bigint'
    | 'bigserial' | 'binary' | 'bind' | 'binding' | 'bit' | 'bit_vector' | 'black' | 'blob'
    | 'block' | 'blue' | 'blur' | 'bold' | 'bolder' | 'bool' | 'boolean' | 'border_box'
    | 'both' | 'box_shadow' | 'breadcrumb' | 'break' | 'breakpoint' | 'brightness' | 'brin' | 'broadcast'
    | 'brown' | 'browser' | 'btree' | 'buckets' | 'business' | 'business_intelligence' | 'button' | 'by'
    | 'bytes' | 'calendar' | 'call_count' | 'call_graph' | 'called_on_null_input' | 'calls' | 'calls_through' | 'card'
    | 'cascade' | 'cascaded' | 'change' | 'char' | 'chart' | 'check_option' | 'checkbox' | 'checked'
    | 'chrome' | 'cidr' | 'circle' | 'circuit_breakers' | 'class' | 'click' | 'clip_path' | 'closest_corner'
    | 'closest_side' | 'cluster' | 'coarse' | 'cohort_analysis' | 'colors' | 'commands' | 'commission_structure' | 'communication'
    | 'compliance_frameworks' | 'composite' | 'compression' | 'compute' | 'computed' | 'conditions' | 'configuration_schema' | 'conic_gradient'
    | 'connections' | 'constraint' | 'constraints' | 'contain' | 'contained' | 'content_box' | 'content_moderation' | 'contents'
    | 'continue' | 'contrast' | 'conversion_tracking' | 'correlation' | 'cosine' | 'cosine_similarity' | 'cost' | 'cost_limit'
    | 'counter' | 'cover' | 'coverage' | 'critical' | 'cross_field' | 'csv' | 'cubic_bezier' | 'currency_conversion'
    | 'currentColor' | 'cursive' | 'custom' | 'cyan' | 'dark' | 'dashed' | 'data_access' | 'data_model'
    | 'data_warehouse' | 'datetime' | 'day' | 'days' | 'deadman' | 'debounce' | 'debug' | 'decimal'
    | 'decision' | 'deduplication' | 'deep' | 'def' | 'default_decision' | 'definer' | 'delivery_method' | 'dependency'
    | 'depends_on' | 'desc' | 'desktop' | 'deterministic' | 'deviation' | 'device' | 'discord' | 'dispatch'
    | 'dispute_resolution' | 'distance' | 'distance_function' | 'distributed_tracing' | 'distribution' | 'do' | 'dot_product' | 'dotted'
    | 'double' | 'dpcm' | 'dpi' | 'dppx' | 'dpr' | 'draggable' | 'drawing' | 'dynamic'
    | 'e2e_test' | 'ease' | 'ease_in' | 'ease_in_out' | 'ease_out' | 'edge' | 'ef_construction' | 'ef_search'
    | 'effective_dates' | 'elasticsearch' | 'elif' | 'ellipse' | 'else' | 'emergency' | 'emit' | 'enable_if'
    | 'encrypted' | 'encryption' | 'enrichment' | 'entities' | 'environment' | 'error' | 'errors' | 'escalation_policy'
    | 'escrow_enabled' | 'eu' | 'euclidean' | 'eventually' | 'evidence_required' | 'except' | 'execute' | 'execution_time'
    | 'expect' | 'expression' | 'extrabold' | 'faceted_search' | 'fake' | 'family' | 'fantasy' | 'farthest_corner'
    | 'farthest_side' | 'fatal' | 'fee_structure' | 'fieldset' | 'fifo' | 'fill_available' | 'filled' | 'filtering'
    | 'finally' | 'fine' | 'firefox' | 'first' | 'fit_content' | 'fixed' | 'fixtures' | 'flat'
    | 'flex' | 'float' | 'float4' | 'float8' | 'fluid' | 'focus_trap' | 'for' | 'for_each'
    | 'foreign' | 'forwards' | 'fraud_detection' | 'fraud_prevention' | 'frequency' | 'ftp' | 'ftps' | 'fuchsia'
    | 'function' | 'g' | 'gauge' | 'gcp_secret_manager' | 'geography' | 'geolocation' | 'geometry' | 'get'
    | 'gigabytes' | 'gin' | 'gist' | 'global' | 'governance' | 'gray' | 'grayscale' | 'green'
    | 'grey' | 'groove' | 'guard' | 'halfvec' | 'hamming' | 'handlers' | 'has' | 'hashicorp_vault'
    | 'having' | 'headers' | 'headless' | 'health' | 'heartbeat' | 'help' | 'help_text' | 'hidden'
    | 'high' | 'high_contrast' | 'histogram' | 'hnsw' | 'hour' | 'hours' | 'hover' | 'hsl'
    | 'hsla' | 'hstore' | 'http' | 'https' | 'hue_rotate' | 'ie' | 'if' | 'immediate'
    | 'implementation' | 'in' | 'in_app' | 'incident' | 'increment' | 'indexes' | 'indexing_strategy' | 'inet'
    | 'infinite' | 'info' | 'inherit' | 'initial' | 'inline' | 'inline_block' | 'inout' | 'inputs'
    | 'insert' | 'inset' | 'instead_of' | 'int' | 'integer' | 'integration_test' | 'interval' | 'into'
    | 'invert' | 'invoker' | 'is' | 'iso' | 'italic' | 'ivf_pq' | 'ivfflat' | 'javascript'
    | 'json' | 'jsonb' | 'justify' | 'kA' | 'kHz' | 'kPa' | 'kV' | 'kW'
    | 'key_rotation' | 'keyboard_navigation' | 'kg' | 'kilobytes' | 'km' | 'labels' | 'landscape' | 'lang'
    | 'language' | 'large' | 'larger' | 'last' | 'letter_spacing' | 'levels' | 'lfu' | 'lg'
    | 'light' | 'lighter' | 'lime' | 'limit' | 'line_height' | 'linear' | 'linear_gradient' | 'link'
    | 'load_balancing' | 'load_test' | 'localization_strategy' | 'logfmt' | 'longblob' | 'longtext' | 'loop' | 'low'
    | 'lru' | 'lsh' | 'm' | 'mA' | 'mV' | 'macaddr' | 'machine_learning' | 'magenta'
    | 'manhattan' | 'margin' | 'maroon' | 'masking' | 'match' | 'matches' | 'materialized' | 'matrix'
    | 'maxResults' | 'max_content' | 'max_length' | 'max_value' | 'max_width' | 'md' | 'measure' | 'measures'
    | 'medium' | 'mediumtext' | 'megabytes' | 'memory_usage' | 'messaging' | 'method' | 'mg' | 'microseconds'
    | 'microservices' | 'milliseconds' | 'minLength' | 'min_content' | 'min_length' | 'min_value' | 'min_width' | 'mobile_first'
    | 'mocks' | 'modal' | 'models' | 'money' | 'monitoring' | 'monospace' | 'month' | 'multi_party_splits'
    | 'must' | 'nanoseconds' | 'navbar' | 'navigate' | 'navy' | 'nearby' | 'negotiation_system' | 'networking'
    | 'nm' | 'no_action' | 'no_preference' | 'no_repeat' | 'none' | 'normal' | 'normalized' | 'not'
    | 'not_matches' | 'not_throws' | 'ns' | 'nulls' | 'number' | 'numeric' | 'oblique' | 'off'
    | 'offset' | 'olive' | 'on' | 'onblur' | 'onboarding_flow' | 'onchange' | 'onclick' | 'onfocus'
    | 'onhover' | 'onsubmit' | 'opacity' | 'operational' | 'operations' | 'or' | 'orange' | 'orchestration'
    | 'orientation' | 'others' | 'out' | 'outlier_detection' | 'outline' | 'outlined' | 'outputs' | 'outset'
    | 'ownership' | 'padding' | 'padding_box' | 'pager' | 'pagination' | 'params' | 'pass' | 'payment_methods'
    | 'payment_providers' | 'performance' | 'performance_metrics' | 'permissions' | 'persistence' | 'persistent' | 'personalization' | 'pink'
    | 'pk' | 'placeholder' | 'placement_strategy' | 'plain' | 'plpgsql' | 'point' | 'pointer' | 'polygon'
    | 'pool_size' | 'popover' | 'portrait' | 'predictive_analytics' | 'prefers_reduced_motion' | 'primary' | 'print' | 'procedure'
    | 'progress' | 'ps' | 'psi' | 'purple' | 'quantiles' | 'queries' | 'radial_gradient' | 'radio'
    | 'range' | 'rate_limiting' | 'rating_system' | 'ratio' | 'reactive' | 'readonly' | 'real' | 'real_time_analytics'
    | 'real_time_chat' | 'recipients' | 'recommendation_engine' | 'red' | 'redirect' | 'reduce' | 'ref' | 'references'
    | 'refund_policies' | 'regression_test' | 'relative' | 'repeat' | 'repeat_x' | 'repeat_y' | 'repeater' | 'requests'
    | 'require_if' | 'requires_theme' | 'resizable' | 'resolution' | 'resource_requirements' | 'responsibilities' | 'responsive' | 'restrict'
    | 'restricted' | 'rete' | 'retention' | 'retries' | 'return_value' | 'returns' | 'returns_null_on_null_input' | 'reverse'
    | 'review_system' | 'rgb' | 'rgba' | 'rich_text' | 'ridge' | 'roles' | 'rotate' | 'rotation'
    | 'round' | 'routing' | 'row' | 'row_level_security' | 'rows' | 'rpm' | 'rps' | 'runbook'
    | 'runner' | 'safari' | 'safe' | 'sampling' | 'sandbox' | 'sans_serif' | 'saturate' | 'scale'
    | 'scaling' | 'scheduling' | 'scientific' | 'screen_reader' | 'screen_reader_only' | 'screenshot' | 'scroll' | 'search_analytics'
    | 'search_discovery' | 'search_engine' | 'sec' | 'secondary' | 'seconds' | 'secrets_management' | 'section' | 'security'
    | 'security_barrier' | 'security_test' | 'segregation_of_duties' | 'select' | 'selected' | 'semibold' | 'seo' | 'sepia'
    | 'serif' | 'service_discovery' | 'service_mesh' | 'services' | 'set' | 'set_default' | 'set_null' | 'setof'
    | 'setup' | 'severity' | 'shadows' | 'shipping_zones' | 'should' | 'showMask' | 'show_if' | 'side_effect'
    | 'sidebar' | 'silver' | 'skew' | 'skill' | 'slack' | 'slider' | 'slot' | 'sm'
    | 'small' | 'smaller' | 'smallint' | 'smoke_test' | 'sms' | 'solid' | 'sortable' | 'space'
    | 'space_around' | 'space_between' | 'space_evenly' | 'spacing' | 'sparse' | 'sparsevec' | 'spec_set' | 'speech'
    | 'spgist' | 'spinner' | 'split' | 'spy' | 'sql' | 'sqlstate' | 'ssl' | 'stack'
    | 'stack_trace' | 'static' | 'stepper' | 'sticky' | 'storage' | 'str' | 'strategy' | 'stretch'
    | 'strict' | 'structured' | 'success' | 'supported_currencies' | 'supported_regions' | 'suppression' | 'switch' | 'symbol'
    | 'system_ui' | 'tabindex' | 'table_cell' | 'table_row' | 'tablegroup' | 'tables' | 'tablet' | 'tabs'
    | 'tax_calculation' | 'teal' | 'teams' | 'teardown' | 'technical' | 'template' | 'tenant_scoped' | 'text_shadow'
    | 'textarea' | 'thick' | 'thin' | 'thresholds' | 'throws' | 'timeline' | 'timer' | 'timeseries'
    | 'timestamp' | 'timezone' | 'tinyint' | 'to_be' | 'to_be_called' | 'to_be_called_times' | 'to_be_called_with' | 'to_be_falsy'
    | 'to_be_greater_than' | 'to_be_less_than' | 'to_be_null' | 'to_be_truthy' | 'to_be_undefined' | 'to_contain' | 'to_equal' | 'to_have_length'
    | 'to_have_property' | 'to_match' | 'tokens' | 'tooltip' | 'touch' | 'trace' | 'tracking' | 'transactions'
    | 'transformation' | 'translate' | 'transparent' | 'tree' | 'trend' | 'triggers' | 'truncate' | 'trust_safety'
    | 'try' | 'tsquery' | 'tsrange' | 'tstzrange' | 'tsvector' | 'ttl' | 'two_way' | 'twotone'
    | 'typeahead' | 'typography' | 'ui_monospace' | 'ui_sans_serif' | 'ui_serif' | 'unique' | 'unit_test' | 'unsafe'
    | 'unset' | 'urgent' | 'us' | 'user_types' | 'uuid' | 'valid_from' | 'valid_to' | 'validation'
    | 'values' | 'var' | 'varbinary' | 'varchar' | 'variables' | 'vector_dims' | 'vector_index' | 'vector_norm'
    | 'verification_required' | 'video_calls' | 'view_module' | 'visible' | 'visual_test' | 'voice' | 'void' | 'wait'
    | 'warning' | 'watch' | 'week' | 'weight' | 'where' | 'while' | 'white' | 'window'
    | 'with' | 'with_data' | 'with_no_data' | 'within' | 'wizard' | 'workflows' | 'wrap' | 'x_large'
    | 'x_small' | 'xl' | 'xml' | 'xpath' | 'xs' | 'xx_large' | 'xx_small' | 'year'
    | 'yellow' | 'yield' | 'z_index'
    ;

// CONFIGURATION with type annotations
config_item
    : member_name ':' type_annotation? '=' value_expr ';'
    | member_name ':' value_expr ';'             // Terse readable AI-agent and capability config
    | member_name ':' type_annotation ';'
    ;

// FIRST-CLASS CAPABILITY AND AGENT CONTRACTS
capability_contract_block
    : 'contract' ':' capability_contract ';'?
    | 'capability_contract' ':' capability_contract ';'?
    ;

capability_contract
    : '{' capability_contract_member* '}'
    ;

capability_contract_member
    : 'id' ':' contract_scalar contract_separator?
    | 'name' ':' contract_scalar contract_separator?
    | 'version' ':' (SEMVER | STRING | IDENTIFIER) contract_separator?
    | 'provides' ':' reference_list contract_separator?
    | 'requires' ':' reference_list contract_separator?
    | 'configuration' ':' contract_object contract_separator?
    | 'configuration_schema' ':' contract_object contract_separator?
    | 'rule_engine' ':' rule_engine_contract contract_separator?
    | 'rules' ':' rule_list contract_separator?
    | 'ui' ':' ui_contract contract_separator?
    | 'theme' ':' theme_contract contract_separator?
    | 'runtime' ':' runtime_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

erp_component_block
    : 'erp_modules' ':' erp_module_set contract_separator?
    | 'components' ':' erp_component_set contract_separator?
    | 'business_rules' ':' erp_rule_set contract_separator?
    | 'approvals' ':' approval_contract contract_separator?
    | 'master_data' ':' master_data_contract contract_separator?
    ;

erp_module_set
    : '[' (erp_domain (',' erp_domain)*)? ']'
    | '{' erp_component_binding* '}'
    ;

erp_component_set
    : '[' (erp_component_ref (',' erp_component_ref)*)? ']'
    | '{' erp_component_binding* '}'
    ;

erp_component_binding
    : erp_component_key ':' erp_domain? '{' erp_component_member* '}' contract_separator?
    ;

erp_component_key
    : IDENTIFIER
    | STRING
    ;

erp_component_ref
    : erp_domain
    | IDENTIFIER
    | STRING
    ;

erp_domain
    : 'finance' | 'general_ledger' | 'accounts_payable' | 'accounts_receivable'
    | 'procurement' | 'purchase_orders' | 'supplier_management'
    | 'inventory' | 'warehouse' | 'order_management' | 'sales' | 'crm'
    | 'manufacturing' | 'materials_planning' | 'production_planning'
    | 'hr' | 'payroll' | 'time_attendance' | 'benefits'
    | 'fixed_assets' | 'asset_management' | 'maintenance'
    | 'project_accounting' | 'budgeting' | 'tax' | 'compliance'
    | 'supply_chain' | 'service_management' | 'reporting'
    | IDENTIFIER
    | STRING
    ;

erp_component_member
    : 'capability' ':' capability_ref contract_separator?
    | 'configuration' ':' contract_object contract_separator?
    | 'data_model' ':' erp_data_contract contract_separator?
    | 'apis' ':' erp_api_contract contract_separator?
    | 'workflows' ':' erp_workflow_contract contract_separator?
    | 'rules' ':' erp_rule_set contract_separator?
    | 'approvals' ':' approval_contract contract_separator?
    | 'permissions' ':' permission_contract contract_separator?
    | 'audit' ':' audit_contract contract_separator?
    | 'effective_dates' ':' effective_date_contract contract_separator?
    | 'master_data' ':' master_data_contract contract_separator?
    | 'ui' ':' ui_contract contract_separator?
    | 'theme' ':' theme_contract contract_separator?
    | 'i18n' ':' i18n_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

erp_data_contract
    : '{' erp_data_member* '}'
    | reference_list
    ;

erp_data_member
    : 'entities' ':' reference_list contract_separator?
    | 'tables' ':' reference_list contract_separator?
    | 'dimensions' ':' reference_list contract_separator?
    | 'measures' ':' reference_list contract_separator?
    | 'master_data' ':' reference_list contract_separator?
    | 'retention' ':' contract_value contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

erp_api_contract
    : '{' erp_api_member* '}'
    | reference_list
    ;

erp_api_member
    : 'commands' ':' reference_list contract_separator?
    | 'queries' ':' reference_list contract_separator?
    | 'events' ':' reference_list contract_separator?
    | 'exports' ':' reference_list contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

erp_workflow_contract
    : '{' erp_workflow_member* '}'
    | reference_list
    ;

erp_workflow_member
    : IDENTIFIER ':' handoff_graph contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

erp_rule_set
    : rule_list
    | '{' erp_rule_group* '}'
    ;

erp_rule_group
    : IDENTIFIER ':' rule_list contract_separator?
    ;

approval_contract
    : '{' approval_member* '}'
    | reference_list
    ;

approval_member
    : 'levels' ':' contract_value contract_separator?
    | 'thresholds' ':' contract_object contract_separator?
    | 'approvers' ':' reference_list contract_separator?
    | 'segregation_of_duties' ':' BOOLEAN contract_separator?
    | 'escalation' ':' contract_value contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

permission_contract
    : '{' permission_member* '}'
    | reference_list
    ;

permission_member
    : 'roles' ':' reference_list contract_separator?
    | 'operations' ':' reference_list contract_separator?
    | 'tenant_scoped' ':' BOOLEAN contract_separator?
    | 'row_level_security' ':' BOOLEAN contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

audit_contract
    : '{' audit_member* '}'
    ;

audit_member
    : 'events' ':' reference_list contract_separator?
    | 'fields' ':' reference_list contract_separator?
    | 'retention' ':' contract_value contract_separator?
    | 'evidence_required' ':' BOOLEAN contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

effective_date_contract
    : '{' effective_date_member* '}'
    ;

effective_date_member
    : 'valid_from' ':' contract_value contract_separator?
    | 'valid_to' ':' contract_value contract_separator?
    | 'calendar' ':' contract_value contract_separator?
    | 'timezone' ':' contract_value contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

master_data_contract
    : '{' master_data_member* '}'
    | reference_list
    ;

master_data_member
    : 'entities' ':' reference_list contract_separator?
    | 'ownership' ':' contract_object contract_separator?
    | 'deduplication' ':' contract_value contract_separator?
    | 'governance' ':' rule_engine_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

agent_composition_block
    : 'agents' ':' agent_set contract_separator?
    | 'runtimes' ':' agent_runtime_set contract_separator?
    | 'tools' ':' agent_tool_set contract_separator?
    | 'handoffs' ':' handoff_graph contract_separator?
    | 'flow' ':' handoff_graph contract_separator?
    | 'memory' ':' agent_memory_contract contract_separator?
    ;

agent_set
    : reference_list
    | '{' agent_binding* '}'
    ;

agent_binding
    : IDENTIFIER ':' IDENTIFIER '{' agent_contract_member* '}' contract_separator?
    | IDENTIFIER contract_separator?
    ;

agent_contract_member
    : 'role' ':' contract_value contract_separator?
    | 'model' ':' model_chain contract_separator?
    | 'models' ':' model_chain contract_separator?
    | 'runtime' ':' agent_runtime_ref contract_separator?
    | 'runner' ':' agent_runtime_ref contract_separator?
    | 'system' ':' contract_value contract_separator?
    | 'capability' ':' capability_ref contract_separator?
    | 'capabilities' ':' reference_list contract_separator?
    | 'tools' ':' reference_list contract_separator?
    | 'memory' ':' agent_memory_contract contract_separator?
    | 'input' ':' io_contract contract_separator?
    | 'inputs' ':' io_contract contract_separator?
    | 'output' ':' io_contract contract_separator?
    | 'outputs' ':' io_contract contract_separator?
    | 'config' ':' contract_object contract_separator?
    | 'configuration' ':' contract_object contract_separator?
    | 'rules' ':' rule_list contract_separator?
    | 'ui' ':' ui_contract contract_separator?
    | 'theme' ':' theme_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

agent_runtime_set
    : '[' agent_runtime_ref (',' agent_runtime_ref)* ']'
    | '{' agent_runtime_contract_member* '}'
    ;

agent_runtime_contract_member
    : agent_runtime_ref ':' runtime_contract contract_separator?
    | IDENTIFIER ':' runtime_contract contract_separator?
    ;

agent_runtime_ref
    : 'local' | 'codex' | 'codex_cli' | 'claude' | 'claude_code'
    | 'opencode' | 'open_code' | 'pi' | 'openai' | 'ollama'
    | IDENTIFIER | STRING
    ;

agent_tool_set
    : reference_list
    | '{' agent_tool_contract_member* '}'
    ;

agent_tool_contract_member
    : IDENTIFIER ':' contract_object contract_separator?
    ;

agent_memory_contract
    : IDENTIFIER IDENTIFIER?
    | contract_object
    ;

handoff_graph
    : handoff_edge ((',' | ';') handoff_edge)* ';'?
    | '[' handoff_edge (',' handoff_edge)* ']'
    | contract_object
    ;

handoff_edge
    : IDENTIFIER '->' IDENTIFIER handoff_modifier*
    ;

handoff_modifier
    : '[' member_name ':' contract_value ']'
    ;

model_chain
    : model_ref ('->' model_ref)*
    | '[' model_ref (',' model_ref)* ']'
    ;

model_ref
    : IDENTIFIER ('.' IDENTIFIER)*
    | STRING
    ;

capability_ref
    : IDENTIFIER ('.' IDENTIFIER)*
    | STRING
    ;

io_contract
    : reference_list
    | contract_object
    | contract_scalar
    ;

rule_engine_block
    : 'rule_engine' ':' rule_engine_contract contract_separator?
    | 'rules' ':' rule_list contract_separator?
    ;

rule_engine_contract
    : '{' rule_engine_member* '}'
    ;

rule_engine_member
    : 'type' ':' rule_engine_type contract_separator?
    | 'default_decision' ':' rule_decision contract_separator?
    | 'rules' ':' rule_list contract_separator?
    | 'inputs' ':' reference_list contract_separator?
    | 'outputs' ':' reference_list contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

rule_engine_type
    : 'deterministic' | 'policy' | 'workflow' | 'rete' | 'expression' | 'ai_assisted'
    | IDENTIFIER | STRING
    ;

rule_list
    : '[' (rule_contract (',' rule_contract)*)? ']'
    | '{' rule_contract* '}'
    ;

rule_contract
    : '{' rule_contract_member* '}'
    | IDENTIFIER ':' contract_object contract_separator?
    ;

rule_contract_member
    : 'name' ':' contract_scalar contract_separator?
    | 'when' ':' contract_value contract_separator?
    | 'condition' ':' contract_value contract_separator?
    | 'then' ':' contract_value contract_separator?
    | 'action' ':' contract_value contract_separator?
    | 'effect' ':' contract_object contract_separator?
    | 'decision' ':' rule_decision contract_separator?
    | 'priority' ':' NUMBER contract_separator?
    | 'applies_to' ':' reference_list contract_separator?
    | 'effective_from' ':' contract_value contract_separator?
    | 'effective_to' ':' contract_value contract_separator?
    | 'exception' ':' contract_value contract_separator?
    | 'approval' ':' approval_contract contract_separator?
    | 'audit' ':' audit_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

rule_decision
    : 'allow' | 'deny' | 'require_review' | 'warn' | 'audit' | IDENTIFIER | STRING
    ;

ui_contract_block
    : 'ui' ':' ui_contract contract_separator?
    ;

ui_contract
    : '{' ui_contract_member* '}'
    ;

ui_contract_member
    : 'shell' ':' ui_shell contract_separator?
    | 'view_module' ':' contract_scalar contract_separator?
    | 'api_prefix' ':' contract_scalar contract_separator?
    | 'routes' ':' ui_route_list contract_separator?
    | 'screens' ':' screen_set contract_separator?
    | 'components' ':' contract_object contract_separator?
    | 'requires_theme' ':' BOOLEAN contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

screen_contract_block
    : 'screens' ':' screen_set contract_separator?
    | 'screen' ':' screen_contract contract_separator?
    ;

screen_set
    : reference_list
    | '{' screen_binding* '}'
    ;

screen_binding
    : screen_key ':' screen_contract contract_separator?
    | screen_key ':' 'screen' '{' screen_contract_member* '}' contract_separator?
    ;

screen_key
    : IDENTIFIER
    | STRING
    ;

screen_contract
    : '{' screen_contract_member* '}'
    ;

screen_contract_member
    : 'route' ':' contract_scalar contract_separator?
    | 'title' ':' contract_scalar contract_separator?
    | 'layout' ':' screen_layout contract_separator?
    | 'contains' ':' screen_element_list contract_separator?
    | 'composes' ':' screen_element_list contract_separator?
    | 'binds' ':' reference_list contract_separator?
    | 'actions' ':' reference_list contract_separator?
    | 'events' ':' screen_event_list contract_separator?
    | 'relationships' ':' screen_relationship_list contract_separator?
    | 'requires' ':' reference_list contract_separator?
    | 'permissions' ':' permission_contract contract_separator?
    | 'rules' ':' rule_list contract_separator?
    | 'ui' ':' ui_contract contract_separator?
    | 'theme' ':' theme_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

screen_layout
    : 'stack' | 'grid' | 'tabs' | 'split' | 'wizard' | 'dashboard' | 'form'
    | member_name | STRING
    ;

screen_element_list
    : '[' (screen_element_ref (',' screen_element_ref)*)? ']'
    ;

screen_element_ref
    : contract_scalar
    | '{' screen_element_member* '}'
    ;

screen_element_member
    : 'id' ':' contract_scalar contract_separator?
    | 'type' ':' contract_scalar contract_separator?
    | 'component' ':' contract_scalar contract_separator?
    | 'slot' ':' contract_scalar contract_separator?
    | 'binds' ':' reference_list contract_separator?
    | 'rules' ':' rule_list contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

screen_event_list
    : '[' (screen_event (',' screen_event)*)? ']'
    ;

screen_event
    : '{' screen_event_member* '}'
    ;

screen_event_member
    : 'on' ':' contract_scalar contract_separator?
    | 'do' ':' contract_value contract_separator?
    | 'target' ':' contract_scalar contract_separator?
    | 'when' ':' contract_value contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

screen_relationship_list
    : '[' (screen_relationship (',' screen_relationship)*)? ']'
    ;

screen_relationship
    : screen_relation_edge
    | '{' screen_relationship_member* '}'
    ;

screen_relation_edge
    : IDENTIFIER '->' IDENTIFIER screen_relation_modifier*
    ;

screen_relation_modifier
    : '[' IDENTIFIER ':' contract_value ']'
    ;

screen_relationship_member
    : 'from' ':' contract_scalar contract_separator?
    | 'to' ':' contract_scalar contract_separator?
    | 'via' ':' contract_scalar contract_separator?
    | 'type' ':' contract_scalar contract_separator?
    | 'when' ':' contract_value contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

ui_shell
    : 'python' | 'react' | 'mobile' | 'cli'
    | IDENTIFIER | STRING
    ;

ui_route_list
    : '[' (ui_route (',' ui_route)*)? ']'
    ;

ui_route
    : '{' ui_route_member* '}'
    ;

ui_route_member
    : 'name' ':' contract_scalar contract_separator?
    | 'path' ':' contract_scalar contract_separator?
    | 'component' ':' contract_scalar contract_separator?
    | 'permission' ':' contract_scalar contract_separator?
    | 'nav_group' ':' contract_scalar contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

theme_contract_block
    : 'theme' ':' theme_contract contract_separator?
    ;

theme_contract
    : '{' theme_contract_member* '}'
    ;

theme_contract_member
    : 'name' ':' contract_scalar contract_separator?
    | 'tokens' ':' theme_token_map contract_separator?
    | 'components' ':' contract_object contract_separator?
    | 'allow_tenant_overrides' ':' BOOLEAN contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

theme_token_map
    : '{' theme_token* '}'
    ;

theme_token
    : (member_name | STRING) ':' contract_value contract_separator?
    ;

runtime_contract
    : '{' runtime_contract_member* '}'
    | agent_runtime_ref
    ;

runtime_contract_member
    : 'backend' ':' runtime_backend contract_separator?
    | 'sandbox' ':' contract_value contract_separator?
    | 'approval_required' ':' BOOLEAN contract_separator?
    | 'cost_limit' ':' contract_value contract_separator?
    | 'streaming' ':' stream_runtime_contract contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

runtime_backend
    : 'python'
    | agent_runtime_ref
    | 'bytewax'
    ;

stream_runtime_block
    : 'streaming' ':' stream_runtime_contract contract_separator?
    ;

stream_runtime_contract
    : '{' stream_runtime_member* '}'
    ;

stream_runtime_member
    : 'processor' ':' stream_processor contract_separator?
    | 'input' ':' contract_value contract_separator?
    | 'output' ':' contract_value contract_separator?
    | 'state' ':' contract_value contract_separator?
    | 'window' ':' time_expr contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

stream_processor
    : 'bytewax' | 'bytewax_streams' | IDENTIFIER | STRING
    ;

i18n_contract_block
    : 'i18n' ':' i18n_contract contract_separator?
    | 'localization' ':' i18n_contract contract_separator?
    ;

i18n_contract
    : '{' i18n_contract_member* '}'
    ;

i18n_contract_member
    : 'supported_languages' ':' language_collection contract_separator?
    | 'default_language' ':' language_code contract_separator?
    | 'fallback_language' ':' language_code contract_separator?
    | member_name ':' contract_value contract_separator?
    ;

language_collection
    : '[' (language_code (',' language_code)*)? ']'
    ;

language_list
    : language_code (',' language_code)*
    ;

language_code
    : member_name
    | STRING
    ;

reference_list
    : '[' (contract_scalar (',' contract_scalar)*)? ']'
    ;

contract_object
    : '{' contract_member* '}'
    ;  // contract_member already has optional separator

contract_member
    : (member_name | STRING) ':' contract_value contract_separator?
    ;

contract_value
    : contract_object
    | contract_array
    | rule_engine_contract
    | ui_contract
    | theme_contract
    | stream_runtime_contract
    | model_chain
    | value_expr
    ;

contract_array
    : '[' (contract_value (',' contract_value)*)? ']'
    ;

contract_scalar
    : member_name ('.' member_name)*
    | STRING
    | NUMBER
    | BOOLEAN
    ;

contract_separator
    : ';' | ','
    ;

type_annotation
    : union_type
    ;
union_type
    : primary_type ('|' primary_type)*
    ;

primary_type
    : basic_type optional_suffix?
    ;

basic_type
    : 'str' | 'int' | 'float' | 'bool' | 'bytes' | 'datetime' | 'decimal'
    | 'Any' | 'None' | IDENTIFIER
    // Database-specific types for seamless integration
    | db_data_type
    | generic_type
    | list_type
    | dict_type
    ;

optional_suffix
    : '?'
    ;

generic_type
    : IDENTIFIER '[' type_annotation (',' type_annotation)* ']'
    ;

list_type
    : '[' type_annotation ']'
    | 'List' '[' type_annotation ']'
    ;

dict_type
    : '{' type_annotation ':' type_annotation '}'
    | 'Dict' '[' type_annotation ',' type_annotation ']'
    ;

// ENHANCED VALUE EXPRESSIONS
value_expr
    : simple_value
    | list_value
    | dict_value
    | cascade_value
    | fallback_chain           // ?? based fallback: gpt4 ?? claude3 ?? llama
    | physical_literal         // measurement literals: 80°C, 150psi
    | agent_memory_value
    | reference_value
    | lambda_expr
    | combination_expr
    | url_pattern
    | regex_pattern
    | time_expr
    | async_expr
    ;

simple_value
    : STRING | NUMBER | BOOLEAN | member_name
    | env_var
    | f_string
    ;

env_var
    : '$' IDENTIFIER
    | 'env' '(' STRING ')'
    ;

f_string
    : 'f' STRING
    ;

list_value
    : '[' (value_expr (',' value_expr)*)? ']'
    | simple_value (',' simple_value)+           // Compact list: a,b,c
    ;

dict_value
    : '{' (key_value_pair ((',' | ';') key_value_pair)* (','|';')?)? '}'
    ;

key_value_pair
    : (member_name | STRING) ':' value_expr
    ;

cascade_value
    : simple_value ('->' simple_value)+          // Fallback chain: gpt4->claude3->llama
    ;

fallback_chain
    : simple_value ('??' simple_value)+          // Preferred cascade: gpt4 ?? claude3 ?? llama
    ;

physical_literal
    : NUMBER PHYS_UNIT
    ;

agent_memory_value
    : member_name member_name?                     // vector support_memory
    ;

reference_value
    : member_name ('.' member_name)*         // Object reference: user.location
    | member_name '*'                        // Collection: cameras*
    | member_name '++'                       // All instances
    ;

combination_expr
    : simple_value ('+' simple_value)+           // Combination: speech + vision + gestures
    ;

url_pattern
    : URL
    | URL_PATTERN
    ;

regex_pattern
    : REGEX
    ;

time_expr
    : TIME_LITERAL
    | DURATION
    | CRON_EXPR
    ;

async_expr
    : 'await' expression
    ;

// BEHAVIORS with full annotation support
behavior_item
    : annotation
    | method_call ';'
    | flow_definition
    | when_clause
    | then_clause
    ;

annotation
    : '@' IDENTIFIER (':' annotation_body)?
    ;

annotation_body
    : simple_value
    | method_call
    | combination_expr
    | IDENTIFIER '{' annotation_member* '}'     // named block: @physics: finite_element { ... }
    | '{' annotation_member* '}'
    ;

annotation_member
    : IDENTIFIER ':' value_expr ';'
    | IDENTIFIER ';'
    | nested_annotation
    ;

nested_annotation
    : annotation
    ;

when_clause
    : 'when' ':' expression '->' statement_block
    | 'when' IDENTIFIER '{' statement* '}'
    ;

then_clause
    : 'then' ':' statement_block
    ;

// METHOD DEFINITIONS with full features
method_def
    : async_modifier? 'def' IDENTIFIER '(' param_list? ')' return_type? method_body
    ;

async_modifier
    : 'async'
    ;

param_list
    : parameter (',' parameter)*
    ;

parameter
    : IDENTIFIER ':' type_annotation? ('=' value_expr)?
    | '*' IDENTIFIER
    | '**' IDENTIFIER
    ;

return_type
    : '->' type_annotation
    ;

method_body
    : '{' statement* '}'
    | '=>' expression ';'
    ;

// STATEMENTS with full control flow
statement
    : simple_statement
    | compound_statement
    ;

simple_statement
    : assignment
    | method_call ';'
    | minion_command
    | return_statement
    | break_statement
    | continue_statement
    | pass_statement
    | assert_statement
    | yield_statement
    ;

compound_statement
    : if_statement
    | for_statement
    | while_statement
    | try_statement
    | with_statement
    | match_statement
    | async_statement
    ;

assignment
    : IDENTIFIER ':' type_annotation? '=' expression ';'
    | IDENTIFIER '=' expression ';'
    | IDENTIFIER '+=' expression ';'         // Increment
    | IDENTIFIER '<<' expression ';'         // Append
    | IDENTIFIER '|=' expression ';'         // Union
    | IDENTIFIER '&=' expression ';'         // Intersection
    ;

method_call
    : target=expression '.' method=IDENTIFIER '(' args? ')'
    | IDENTIFIER '(' args? ')'
    ;

args
    : argument (',' argument)*
    ;

argument
    : expression
    | IDENTIFIER '=' expression              // Named argument
    | '*' expression                         // Unpacking
    | '**' expression                        // Dict unpacking
    ;

return_statement
    : 'return' expression? ';'
    ;

break_statement
    : 'break' ';'
    ;

continue_statement
    : 'continue' ';'
    ;

pass_statement
    : 'pass' ';'
    ;

assert_statement
    : 'assert' expression (',' STRING)? ';'
    ;

yield_statement
    : 'yield' expression ';'
    | 'yield' 'from' expression ';'
    ;

// CONTROL FLOW STATEMENTS
if_statement
    : 'if' '(' expression ')' statement_block elif_clause* else_clause?
    ;

elif_clause
    : 'elif' '(' expression ')' statement_block
    ;

else_clause
    : 'else' statement_block
    ;

for_statement
    : 'for' IDENTIFIER 'in' expression statement_block else_clause?
    | 'async' 'for' IDENTIFIER 'in' expression statement_block else_clause?
    ;

while_statement
    : 'while' '(' expression ')' statement_block else_clause?
    ;

try_statement
    : 'try' statement_block except_clause+ else_clause? finally_clause?
    | 'try' statement_block finally_clause
    ;

except_clause
    : 'except' exception_spec? statement_block
    ;

exception_spec
    : type_annotation ('as' IDENTIFIER)?
    ;

finally_clause
    : 'finally' statement_block
    ;

with_statement
    : 'with' with_item (',' with_item)* statement_block
    | 'async' 'with' with_item (',' with_item)* statement_block
    ;

with_item
    : expression ('as' IDENTIFIER)?
    ;

match_statement
    : 'match' expression '{' case_clause+ '}'
    ;

case_clause
    : 'case' pattern guard? statement_block
    ;

pattern
    : or_pattern
    ;

or_pattern
    : primary_pattern ('|' primary_pattern)*
    ;

primary_pattern
    : literal_pattern
    | capture_pattern
    | wildcard_pattern
    | value_pattern
    | sequence_pattern
    | mapping_pattern
    | class_pattern
    | '(' pattern ')'
    ;

literal_pattern
    : STRING | NUMBER | BOOLEAN | 'None'
    ;

capture_pattern
    : IDENTIFIER
    ;

wildcard_pattern
    : '_'
    ;

value_pattern
    : IDENTIFIER ('.' IDENTIFIER)*
    ;

sequence_pattern
    : '[' pattern (',' pattern)* ']'
    | '(' pattern (',' pattern)* ')'
    ;

mapping_pattern
    : '{' mapping_pattern_pair (',' mapping_pattern_pair)* '}'
    ;

mapping_pattern_pair
    : (STRING | IDENTIFIER) ':' pattern
    | '**' IDENTIFIER
    ;

class_pattern
    : IDENTIFIER '(' pattern (',' pattern)* ')'
    ;

guard
    : 'if' expression
    ;

async_statement
    : 'async' compound_statement
    ;

statement_block
    : '{' statement* '}'
    | statement
    ;

// EXPRESSIONS with full operator support
expression
    : lambda_expr
    | conditional_expr
    ;

lambda_expr
    : 'lambda' param_list? ':' expression
    | param_list '=>' expression
    | IDENTIFIER '=>' expression             // Single parameter
    ;

conditional_expr
    : null_coalesce_expr ('if' null_coalesce_expr 'else' expression)?
    | null_coalesce_expr '?' null_coalesce_expr ':' expression    // C-style ternary
    ;

null_coalesce_expr
    : or_test ('??' or_test)*
    ;

or_test
    : and_test ('or' and_test)*
    | and_test ('||' and_test)*
    ;

and_test
    : not_test ('and' not_test)*
    | not_test ('&&' not_test)*
    ;

not_test
    : 'not' not_test
    | comparison
    ;

comparison
    : pipeline_expr (comp_op pipeline_expr)*
    ;

pipeline_expr
    : bitwise_or ('|>' bitwise_or)*
    ;

comp_op
    : '<' | '>' | '==' | '>=' | '<=' | '!=' | '<>'
    | 'in' | 'not' 'in' | 'is' | 'is' 'not'
    ;

bitwise_or
    : bitwise_xor ('|' bitwise_xor)*
    ;

bitwise_xor
    : bitwise_and ('^' bitwise_and)*
    ;

bitwise_and
    : shift_expr ('&' shift_expr)*
    ;

shift_expr
    : arith_expr (('<<' | '>>') arith_expr)*
    ;

arith_expr
    : term (('+' | '-') term)*
    ;

term
    : factor (('*' | '@' | '/' | '//' | '%') factor)*
    ;

factor
    : ('+' | '-' | '~') factor
    | power
    ;

power
    : atom_expr ('**' factor)?
    ;

atom_expr
    : atom trailer*
    ;

atom
    : '(' (yield_expr | testlist_comp)? ')'
    | '[' listmaker? ']'
    | '{' dictorsetmaker? '}'
    | IDENTIFIER | NUMBER | STRING | f_string
    | '...' | 'None' | 'True' | 'False'
    | env_var
    | await_expr
    ;

await_expr
    : 'await' atom_expr
    ;

trailer
    : '(' args? ')'
    | '[' subscriptlist ']'
    | '.' IDENTIFIER
    ;

subscriptlist
    : subscript (',' subscript)* ','?
    ;

subscript
    : expression
    | expression? ':' expression? sliceop?
    ;

sliceop
    : ':' expression?
    ;

listmaker
    : (expression | star_expr) (list_for | (',' (expression | star_expr))* ','?)
    ;

dictorsetmaker
    : ((expression ':' expression | '**' expression) (comp_for | (',' (expression ':' expression | '**' expression))* ','?) |
       (expression | star_expr) (comp_for | (',' (expression | star_expr))* ','?))
    ;

testlist_comp
    : (expression | star_expr) (comp_for | (',' (expression | star_expr))* ','?)
    ;

star_expr
    : '*' expression
    ;

comp_for
    : 'for' exprlist 'in' or_test ('if' expression)*
    | 'async' 'for' exprlist 'in' or_test ('if' expression)*
    ;

list_for
    : 'for' exprlist 'in' testlist ('if' expression)*
    ;

exprlist
    : (expression | star_expr) (',' (expression | star_expr))* ','?
    ;

testlist
    : expression (',' expression)* ','?
    ;

yield_expr
    : 'yield' yield_arg?
    ;

yield_arg
    : 'from' expression | testlist
    ;

// FLOWS with advanced workflow support
flow_definition
    : flow_step flow_connector*
    ;

flow_step
    : IDENTIFIER flow_modifiers?
    | IDENTIFIER '(' args? ')' flow_modifiers?
    | conditional_flow_step
    | parallel_flow_step
    ;

flow_connector
    : '->' flow_step
    | '|' flow_step                          // Parallel branch
    | '&' flow_step                          // Synchronized step
    ;

flow_modifiers
    : '[' flow_modifier (',' flow_modifier)* ']'
    ;

flow_modifier
    : 'timeout' ':' time_expr
    | 'retry' ':' NUMBER
    | 'condition' ':' expression
    | 'priority' ':' NUMBER
    | 'parallel' ':' NUMBER
    ;

conditional_flow_step
    : 'if' '(' expression ')' flow_step ('else' flow_step)?
    ;

parallel_flow_step
    : 'parallel' '{' flow_step (',' flow_step)* '}'
    ;

// MINION COMMANDS with full protocol support
minion_command
    : target=expression '.' command=minion_verb '(' args? ')' ';'
    | '@' scope=minion_scope '(' expression ')' command=minion_verb '(' args? ')' ';'
    | broadcast_command
    ;

broadcast_command
    : '@' 'all' '(' entity_type ')' minion_verb '(' args? ')' ';'
    | '@' 'broadcast' minion_verb '(' args? ')' ';'
    ;

minion_verb
    : 'do' | 'get' | 'set' | 'watch' | 'report' | 'help'
    | 'start' | 'stop' | 'pause' | 'resume' | 'reset'
    | 'configure' | 'status' | 'health' | 'metrics'
    ;

minion_scope
    : 'all' | 'nearby' | 'type' | 'group' | 'cluster' | 'region'
    ;

// NESTED ENTITIES with full composition
nested_entity
    : decorator* entity_type IDENTIFIER inheritance? '{' entity_body '}'
    ;

// CLASS DEFINITIONS
class_def
    : decorator* 'class' IDENTIFIER inheritance? '{' class_body '}'
    ;

class_body
    : class_member*
    ;

class_member
    : method_def
    | config_item
    | nested_entity
    | class_def
    ;

// EXCEPTION DEFINITIONS
exception_def
    : 'exception' IDENTIFIER inheritance? '{' exception_body '}'
    ;

exception_body
    : method_def*
    ;

// VARIABLE DECLARATIONS
variable_declaration
    : member_name ':' type_annotation ('=' value_expr)? ';'
    ;

// ========================================
// REUSABLE DOMAIN-SPECIFIC GRAMMAR FRAGMENTS
// These fragments keep specialized capability configuration terse while
// still accepting structured objects where richer contracts are needed.
// ========================================

// Configuration specifications
ab_testing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
access_policy_specification: IDENTIFIER | STRING | '{' policy_property* '}' ;
aggregation_specification: IDENTIFIER | STRING | '{' agg_property* '}' ;
alert_frequency: duration_value | NUMBER time_unit ;
alerting_specification: IDENTIFIER | STRING | '{' alert_property* '}' ;
algorithm_specification: IDENTIFIER | STRING | '{' algo_property* '}' ;
analytics_specification: IDENTIFIER | STRING | '{' analytics_property* '}' ;
announcement_config: IDENTIFIER | STRING | '{' config_property* '}' ;
api_call: IDENTIFIER '(' args? ')' ;
api_gateway_config: IDENTIFIER | STRING | '{' gateway_property* '}' ;
api_specification: IDENTIFIER | STRING | '{' api_property* '}' ;
auto_resolution_specification: IDENTIFIER | STRING | '{' resolution_property* '}' ;
autocomplete_config: IDENTIFIER | STRING | '{' config_property* '}' ;
background_check_config: IDENTIFIER | STRING | '{' check_property* '}' ;
backup_specification: IDENTIFIER | STRING | '{' backup_property* '}' ;
bi_configuration: IDENTIFIER | STRING | '{' bi_property* '}' ;
bucket_specification: IDENTIFIER | STRING | '{' bucket_property* '}' ;
chat_configuration: IDENTIFIER | STRING | '{' chat_property* '}' ;
circuit_breaker_config: IDENTIFIER | STRING | '{' breaker_property* '}' ;
cohort_analysis_config: IDENTIFIER | STRING | '{' cohort_property* '}' ;
condition_specification: IDENTIFIER | STRING | expression ;
constraint_specification: IDENTIFIER | STRING | '{' constraint_property* '}' ;
delivery_specification: IDENTIFIER | STRING | '{' delivery_property* '}' ;
document_verification_config: IDENTIFIER | STRING | '{' doc_property* '}' ;
escalation_specification: IDENTIFIER | STRING | '{' escalation_property* '}' ;
event_persistence_config: IDENTIFIER | STRING | '{' persistence_property* '}' ;
fraud_detection_config: IDENTIFIER | STRING | '{' fraud_property* '}' ;
funnel_analysis_config: IDENTIFIER | STRING | '{' funnel_property* '}' ;
group_reference: IDENTIFIER | STRING ;
identity_verification_config: IDENTIFIER | STRING | '{' identity_property* '}' ;
kyc_config: IDENTIFIER | STRING | '{' kyc_property* '}' ;
load_balancer_config: IDENTIFIER | STRING | '{' lb_property* '}' ;
localization_specification: IDENTIFIER | STRING | '{' localization_property* '}' ;
monitoring_specification: IDENTIFIER | STRING | '{' monitoring_property* '}' ;
personalization_specification: IDENTIFIER | STRING | '{' personalization_property* '}' ;
push_notification_config: IDENTIFIER | STRING | '{' push_property* '}' ;
rate_limit_specification: IDENTIFIER | STRING | '{' rate_limit_property* '}' ;
retry_policy_config: IDENTIFIER | STRING | '{' retry_property* '}' ;
retry_specification: IDENTIFIER | STRING | '{' retry_property* '}' ;
role_reference: IDENTIFIER | STRING ;
schedule_specification: IDENTIFIER | STRING | '{' schedule_property* '}' ;
schema_specification: IDENTIFIER | STRING | '{' schema_property* '}' ;
service_mesh_config: IDENTIFIER | STRING | '{' mesh_property* '}' ;
sms_config: IDENTIFIER | STRING | '{' sms_property* '}' ;
template_specification: IDENTIFIER | STRING | '{' template_property* '}' ;
tracking_specification: IDENTIFIER | STRING | '{' tracking_property* '}' ;
transformation_specification: IDENTIFIER | STRING | '{' transform_property* '}' ;
user_reference: IDENTIFIER | STRING ;
user_segmentation_config: IDENTIFIER | STRING | '{' segment_property* '}' ;

// Generic property types used in configurations above
config_property: IDENTIFIER ':' value_expr ;
policy_property: IDENTIFIER ':' value_expr ;
agg_property: IDENTIFIER ':' value_expr ;
algo_property: IDENTIFIER ':' value_expr ;
gateway_property: IDENTIFIER ':' value_expr ;
api_property: IDENTIFIER ':' value_expr ;
resolution_property: IDENTIFIER ':' value_expr ;
check_property: IDENTIFIER ':' value_expr ;
backup_property: IDENTIFIER ':' value_expr ;
bi_property: IDENTIFIER ':' value_expr ;
bucket_property: IDENTIFIER ':' value_expr ;
chat_property: IDENTIFIER ':' value_expr ;
breaker_property: IDENTIFIER ':' value_expr ;
cohort_property: IDENTIFIER ':' value_expr ;
constraint_property: IDENTIFIER ':' value_expr ;
delivery_property: IDENTIFIER ':' value_expr ;
doc_property: IDENTIFIER ':' value_expr ;
escalation_property: IDENTIFIER ':' value_expr ;
persistence_property: IDENTIFIER ':' value_expr ;
fraud_property: IDENTIFIER ':' value_expr ;
funnel_property: IDENTIFIER ':' value_expr ;
identity_property: IDENTIFIER ':' value_expr ;
kyc_property: IDENTIFIER ':' value_expr ;
lb_property: IDENTIFIER ':' value_expr ;
localization_property: IDENTIFIER ':' value_expr ;
monitoring_property: IDENTIFIER ':' value_expr ;
personalization_property: IDENTIFIER ':' value_expr ;
push_property: IDENTIFIER ':' value_expr ;
rate_limit_property: IDENTIFIER ':' value_expr ;
retry_property: IDENTIFIER ':' value_expr ;
schedule_property: IDENTIFIER ':' value_expr ;
template_property: IDENTIFIER ':' value_expr ;
tracking_property: IDENTIFIER ':' value_expr ;
transform_property: IDENTIFIER ':' value_expr ;
segment_property: IDENTIFIER ':' value_expr ;
mesh_property: IDENTIFIER ':' value_expr ;
sms_property: IDENTIFIER ':' value_expr ;

// More configuration stubs - second batch
commission_config: IDENTIFIER | STRING | '{' config_property* '}' ;
compliance_framework_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
composite_condition: condition_expression ('&&' | '||') condition_expression ;
compression_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
container_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
conversion_tracking_config: IDENTIFIER | STRING | '{' config_property* '}' ;
correlation_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
currency_conversion_config: IDENTIFIER | STRING | '{' config_property* '}' ;
currency_list: '[' STRING (',' STRING)* ']' ;
custom_condition: expression ;
custom_format_specification: IDENTIFIER | STRING ;
custom_unit: IDENTIFIER | STRING ;
dashboard_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
data_access_config: IDENTIFIER | STRING | '{' config_property* '}' ;
data_warehouse_config: IDENTIFIER | STRING | '{' config_property* '}' ;
database_reference: IDENTIFIER | STRING ;
deployment_environment: IDENTIFIER | STRING ;
dispute_config: IDENTIFIER | STRING | '{' config_property* '}' ;
duration_clause: duration_value | NUMBER time_unit ;
ecommerce_config: IDENTIFIER | STRING | '{' config_property* '}' ;
environment_variable: IDENTIFIER | STRING ;
error_handling_config: IDENTIFIER | STRING | '{' config_property* '}' ;
escrow_config: IDENTIFIER | STRING | '{' config_property* '}' ;
execution_environment: IDENTIFIER | STRING ;
experiment_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
failover_config: IDENTIFIER | STRING | '{' config_property* '}' ;
feature_flag_config: IDENTIFIER | STRING | '{' config_property* '}' ;
fulfillment_config: IDENTIFIER | STRING | '{' config_property* '}' ;
gdpr_config: IDENTIFIER | STRING | '{' config_property* '}' ;
health_check_config: IDENTIFIER | STRING | '{' config_property* '}' ;
identity_provider_config: IDENTIFIER | STRING | '{' config_property* '}' ;
infrastructure_requirement: IDENTIFIER | STRING ;
inventory_config: IDENTIFIER | STRING | '{' config_property* '}' ;
lambda_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
listing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
machine_learning_config: IDENTIFIER | STRING | '{' config_property* '}' ;
ml_model_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
model_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
notification_template: IDENTIFIER | STRING ;
orchestration_config: IDENTIFIER | STRING | '{' config_property* '}' ;
payment_config: IDENTIFIER | STRING | '{' config_property* '}' ;
platform_config: IDENTIFIER | STRING | '{' config_property* '}' ;
platform_name: IDENTIFIER | STRING ;
prediction_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
pricing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
quality_gate_config: IDENTIFIER | STRING | '{' config_property* '}' ;
region_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
resource_requirement: IDENTIFIER | STRING ;
retention_analysis_config: IDENTIFIER | STRING | '{' config_property* '}' ;
revenue_sharing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
rollback_config: IDENTIFIER | STRING | '{' config_property* '}' ;
scaling_config: IDENTIFIER | STRING | '{' config_property* '}' ;
security_policy: IDENTIFIER | STRING | '{' config_property* '}' ;
subscription_config: IDENTIFIER | STRING | '{' config_property* '}' ;
tax_calculation_config: IDENTIFIER | STRING | '{' config_property* '}' ;
time_range: duration_value | NUMBER time_unit ;
user_type_definition: IDENTIFIER | STRING | '{' config_property* '}' ;
webhook_config: IDENTIFIER | STRING | '{' config_property* '}' ;

// Final batch of undefined rule stubs
ecommerce_name: IDENTIFIER | STRING ;
encryption_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
endpoint_definition_list: '[' api_property (',' api_property)* ']' ;
enrichment_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
escalation_policy_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
event_name: IDENTIFIER | STRING ;
event_routing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
event_schema: IDENTIFIER | STRING | '{' config_property* '}' ;
export_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
facet_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
fee_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
filter_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
fraud_prevention_config: IDENTIFIER | STRING | '{' config_property* '}' ;
geolocation_config: IDENTIFIER | STRING | '{' config_property* '}' ;
handler_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
indexing_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
inventory_management: IDENTIFIER | STRING | '{' config_property* '}' ;
label_specification: IDENTIFIER | STRING ;
load_balancing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
localization_config: IDENTIFIER | STRING | '{' config_property* '}' ;
marketplace_name: IDENTIFIER | STRING ;
masking_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
messaging_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
metadata_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
metric_reference: IDENTIFIER | STRING ;
ml_analytics_config: IDENTIFIER | STRING | '{' config_property* '}' ;
moderation_config: IDENTIFIER | STRING | '{' config_property* '}' ;
monitoring_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
negotiation_config: IDENTIFIER | STRING | '{' config_property* '}' ;
networking_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
onboarding_definition: IDENTIFIER | STRING | '{' config_property* '}' ;
order_fulfillment: IDENTIFIER | STRING | '{' config_property* '}' ;
output_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
pattern_condition: expression ;
payment_method_list: '[' STRING (',' STRING)* ']' ;
payment_provider_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
percentage_clause: NUMBER '%' ;
permission_list: '[' STRING (',' STRING)* ']' ;
personalization_config: IDENTIFIER | STRING | '{' config_property* '}' ;
placement_strategy_config: IDENTIFIER | STRING | '{' config_property* '}' ;
predictive_analytics_config: IDENTIFIER | STRING | '{' config_property* '}' ;
quantile_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
query_expression: expression ;
rating_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
real_time_analytics_config: IDENTIFIER | STRING | '{' config_property* '}' ;
recommendation_config: IDENTIFIER | STRING | '{' config_property* '}' ;
refund_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
resource_requirements_config: IDENTIFIER | STRING | '{' config_property* '}' ;
responsibility_list: '[' STRING (',' STRING)* ']' ;
retention_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
review_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
rotation_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
runbook_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
sampling_specification: IDENTIFIER | STRING | '{' config_property* '}' ;
scaling_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
search_analytics_config: IDENTIFIER | STRING | '{' config_property* '}' ;
search_engine_type: IDENTIFIER | STRING ;
secrets_config: IDENTIFIER | STRING | '{' config_property* '}' ;
security_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
service_definition_list: '[' config_property (',' config_property)* ']' ;
service_dependency_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
service_discovery_config: IDENTIFIER | STRING | '{' config_property* '}' ;
service_name: IDENTIFIER | STRING ;
service_type: IDENTIFIER | STRING ;
shipping_zones_config: IDENTIFIER | STRING | '{' config_property* '}' ;
split_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
storage_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;
suppression_rules: IDENTIFIER | STRING | '{' config_property* '}' ;
threshold_value: NUMBER | percentage_value ;
tracing_config: IDENTIFIER | STRING | '{' config_property* '}' ;
trigger_list: '[' IDENTIFIER (',' IDENTIFIER)* ']' ;
user_type_name: IDENTIFIER | STRING ;
verification_config: IDENTIFIER | STRING | '{' config_property* '}' ;
verification_requirements: '[' STRING (',' STRING)* ']' ;
video_call_configuration: IDENTIFIER | STRING | '{' config_property* '}' ;

// Additional rule stubs to complete the grammar (duplicates removed)

// ========================================
// LEXER RULES - Complete implementation
// ========================================

// Identifiers
// BOOLEAN and NULL must precede IDENTIFIER for correct priority
BOOLEAN: 'True' | 'False' | 'true' | 'false';
NULL: 'null' | 'NULL' | 'nil';
IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]*;

// Numbers with full support
NUMBER
    : INTEGER
    | FLOAT_NUMBER
    | COMPLEX_NUMBER
    ;

INTEGER
    : DECIMAL_INTEGER
    | BIN_INTEGER
    | OCT_INTEGER
    | HEX_INTEGER
    ;

DECIMAL_INTEGER: [0-9] | [1-9] [0-9]*;
BIN_INTEGER: '0' [bB] [01]+;
OCT_INTEGER: '0' [oO] [0-7]+;
HEX_INTEGER: '0' [xX] [0-9a-fA-F]+;

FLOAT_NUMBER
    : [0-9]* '.' [0-9]+
    | [0-9]+ '.'
    | [0-9]+ [eE] [+-]? [0-9]+
    | [0-9]* '.' [0-9]+ [eE] [+-]? [0-9]+
    ;

COMPLEX_NUMBER: (FLOAT_NUMBER | DECIMAL_INTEGER) [jJ];

// Strings with full support
STRING
    : '"' STRING_CONTENT_DOUBLE* '"'
    | '\'' STRING_CONTENT_SINGLE* '\''
    | '"""' .*? '"""'
    | '\'\'\'' .*? '\'\'\''
    ;

fragment STRING_CONTENT_DOUBLE: ~["\\\r\n] | ESCAPE_SEQUENCE;
fragment STRING_CONTENT_SINGLE: ~['\\\r\n] | ESCAPE_SEQUENCE;
fragment ESCAPE_SEQUENCE: '\\' .;

// F-strings
F_STRING: [fF] STRING;

// Regular expressions
REGEX: '/' (~[/\\\r\n] | '\\' .)* '/';

// URLs and patterns
URL: ('http' | 'https' | 'ftp' | 'ftps') '://' ~[ \t\r\n]+;
URL_PATTERN: URL ('*' | '**' | '{' ~[}]* '}');

// Time expressions
TIME_LITERAL: [0-9]{1,2} ':' [0-9]{2} (':' [0-9]{2})?;
DURATION: [0-9]+ ('s' | 'sec' | 'min' | 'hour' | 'day' | 'week' | 'month' | 'year');
CRON_EXPR: '"' [0-9*,-/]+ (' ' [0-9*,-/]+){4,5} '"';

// Semantic versioning
SEMVER: [0-9]+ '.' [0-9]+ '.' [0-9]+ ('-' [a-zA-Z0-9-]+)? ('+' [a-zA-Z0-9-]+)?;

// Booleans and NULL

// Comments
COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
PYTHON_COMMENT: '#' ~[\r\n]* -> skip;

// Whitespace
WS: [ \t\r\n]+ -> skip;

// Null coalescing and fallback operator (must precede QUESTION token)
NULL_COALESCE: '??';

// Pipeline operator (must precede PIPE token)
PIPELINE_OP: '|>';

// Physical/measurement unit suffix (must precede other tokens that could overlap)
// Supports common SI and imperial units including degree symbol (U+00B0)
PHYS_UNIT
    : '°' [CFKRc]          // temperature: °C °F °K °R
    | 'μm' | 'nm'               // micro/nano length
    | 'kPa' | 'MPa' | 'GPa'    // pressure
    | 'kHz' | 'MHz' | 'GHz'    // frequency (must precede shorter forms)
    | 'mV' | 'kV' | 'MV'       // voltage
    | 'mA' | 'kA'              // current
    | 'kW' | 'MW' | 'GW'      // power
    | 'rpm' | 'rps'            // rotational
    | 'ms' | 'us' | 'ns' | 'ps' // time (sub-second)
    | 'psi' | 'bar' | 'atm'    // pressure (word forms)
    | 'Hz' | 'Pa'              // base SI units
    | 'km' | 'cm' | 'mm'       // length (must precede 'm')
    | 'kg' | 'mg'              // mass (must precede 'g')
    ;

// Special operators and symbols
ARROW: '->';
PIPE: '|';
AMP: '&';
AT: '@';
DOLLAR: '$';
STAR: '*';
DOUBLE_STAR: '**';
PLUS_EQ: '+=';
MINUS_EQ: '-=';
MULT_EQ: '*=';
DIV_EQ: '/=';
MOD_EQ: '%=';
AND_EQ: '&=';
OR_EQ: '|=';
XOR_EQ: '^=';
LSHIFT_EQ: '<<=';
RSHIFT_EQ: '>>=';
POWER_EQ: '**=';
FLOORDIV_EQ: '//=';
APPEND: '<<';
WALRUS: ':=';

// Operators
EQ: '==';
NE: '!=';
LE: '<=';
GE: '>=';
LT: '<';
GT: '>';
// LSHIFT removed - duplicate of APPEND
RSHIFT: '>>';
AND: '&&';
OR: '||';
NOT: '!';
TILDE: '~';
XOR: '^';
ASSIGN: '=';
PLUS: '+';
MINUS: '-';
// MULT removed - duplicate of STAR
DIV: '/';
FLOORDIV: '//';
MOD: '%';
// POWER removed - duplicate of DOUBLE_STAR
// MATRIX_MULT removed - duplicate of AT

// Delimiters
LPAREN: '(';
RPAREN: ')';
LBRACE: '{';
RBRACE: '}';
LBRACK: '[';
RBRACK: ']';
SEMI: ';';
COMMA: ',';
DOT: '.';
COLON: ':';
QUESTION: '?';
ELLIPSIS: '...';


// ========================================
// FORM LAYOUT SUBLANGUAGE
// ========================================

// Form Layout Definition
form_layout
    : 'layout' layout_type? '{' layout_definition '}'
    ;

layout_type
    : 'responsive' | 'fixed' | 'fluid' | 'adaptive' | 'mobile_first'
    ;

layout_definition
    : layout_element*
    ;

layout_element
    : container_element
    | field_element
    | component_element
    | layout_directive
    ;

container_element
    : container_type IDENTIFIER? layout_properties? '{' layout_element* '}'
    ;

container_type
    : 'row' | 'column' | 'grid' | 'flex' | 'stack' | 'wrap'
    | 'card' | 'panel' | 'section' | 'group' | 'fieldset'
    | 'tabs' | 'accordion' | 'modal' | 'sidebar' | 'navbar'
    ;

field_element
    : field_type IDENTIFIER field_properties?
    ;

field_type
    : 'input' | 'textarea' | 'select' | 'checkbox' | 'radio' | 'switch'
    | 'slider' | 'range' | 'date' | 'time' | 'datetime' | 'color'
    | 'file' | 'image' | 'signature' | 'drawing' | 'map' | 'rich_text'
    | 'number' | 'currency' | 'percentage' | 'phone' | 'email' | 'url'
    | 'password' | 'search' | 'autocomplete' | 'typeahead' | 'tags'
    ;

component_element
    : component_type IDENTIFIER component_properties?
    ;

component_type
    : 'button' | 'link' | 'icon' | 'image' | 'video' | 'audio'
    | 'chart' | 'graph' | 'table' | 'list' | 'tree' | 'calendar'
    | 'progress' | 'spinner' | 'badge' | 'tooltip' | 'popover'
    | 'breadcrumb' | 'pagination' | 'stepper' | 'timeline'
    ;

layout_directive
    : '@' directive_name directive_params?
    ;

directive_name
    : 'responsive' | 'conditional' | 'repeater' | 'template'
    | 'validate' | 'bind' | 'watch' | 'compute' | 'format'
    | 'security' | 'accessibility' | 'seo' | 'analytics'
    ;

directive_params
    : '(' directive_param (',' directive_param)* ')'
    ;

directive_param
    : IDENTIFIER ':' value_expr
    | value_expr
    ;

layout_properties
    : '[' layout_property (',' layout_property)* ']'
    ;

layout_property
    : style_property
    | behavior_property
    | data_property
    | accessibility_property
    ;

style_property
    : 'width' ':' size_value
    | 'height' ':' size_value
    | 'margin' ':' spacing_value
    | 'padding' ':' spacing_value
    | 'background' ':' color_value
    | 'border' ':' border_value
    | 'font' ':' font_value
    | 'color' ':' color_value
    | 'display' ':' display_value
    | 'position' ':' position_value
    | 'z_index' ':' NUMBER
    | 'opacity' ':' NUMBER
    | 'animation' ':' animation_value
    | 'transition' ':' transition_value
    | 'transform' ':' transform_value
    | 'filter' ':' filter_value
    | 'box_shadow' ':' shadow_value
    | 'text_shadow' ':' shadow_value
    | 'gradient' ':' gradient_value
    | 'backdrop_filter' ':' filter_value
    | 'clip_path' ':' clip_value
    | 'mask' ':' mask_value
    ;

field_properties
    : '[' field_property (',' field_property)* ']'
    ;

field_property
    : 'label' ':' STRING
    | 'placeholder' ':' STRING
    | 'help_text' ':' STRING
    | 'required' ':' BOOLEAN
    | 'disabled' ':' BOOLEAN
    | 'readonly' ':' BOOLEAN
    | 'hidden' ':' BOOLEAN
    | 'default' ':' value_expr
    | 'validation' ':' validation_rules
    | 'format' ':' format_specification
    | 'mask' ':' input_mask
    | 'autocomplete' ':' autocomplete_spec
    | 'dependency' ':' dependency_spec
    | 'conditional' ':' conditional_spec
    | 'accessibility' ':' accessibility_spec
    ;

component_properties
    : '[' component_property (',' component_property)* ']'
    ;

component_property
    : 'text' ':' STRING
    | 'icon' ':' icon_specification
    | 'action' ':' action_specification
    | 'data' ':' data_specification
    | 'state' ':' state_specification
    | 'theme' ':' theme_specification
    | 'variant' ':' variant_specification
    | 'size' ':' size_specification
    | 'alignment' ':' alignment_specification
    ;

// Layout Value Types
size_value
    : NUMBER unit?
    | 'auto' | 'inherit' | 'initial' | 'unset'
    | 'min_content' | 'max_content' | 'fit_content'
    | 'fill_available' | 'stretch'
    | percentage_value
    | viewport_value
    | calc_expression
    ;

spacing_value
    : size_value
    | size_value size_value  // horizontal vertical
    | size_value size_value size_value size_value  // top right bottom left
    ;

color_value
    : hex_color | rgb_color | hsl_color | named_color
    | 'transparent' | 'currentColor' | 'inherit'
    | css_variable
    ;

border_value
    : border_width border_style border_color
    | 'none' | 'inherit'
    ;

font_value
    : font_family font_size font_weight font_style
    | 'inherit' | 'initial' | 'unset'
    ;

display_value
    : 'block' | 'inline' | 'inline_block' | 'flex' | 'grid'
    | 'table' | 'table_row' | 'table_cell' | 'none' | 'contents'
    ;

position_value
    : 'static' | 'relative' | 'absolute' | 'fixed' | 'sticky'
    ;

animation_value
    : animation_name duration timing_function delay iteration_count direction fill_mode
    ;

transition_value
    : transition_property duration timing_function delay
    ;

// Responsive Design
responsive_breakpoints
    : '@media' media_query '{' layout_element* '}'
    ;

media_query
    : media_type media_features?
    | media_features
    ;

media_type
    : 'screen' | 'print' | 'speech' | 'all'
    ;

media_features
    : '(' media_feature (',' media_feature)* ')'
    ;

media_feature
    : 'min_width' ':' size_value
    | 'max_width' ':' size_value
    | 'orientation' ':' ('portrait' | 'landscape')
    | 'resolution' ':' resolution_value
    | 'hover' ':' ('hover' | 'none')
    | 'pointer' ':' ('fine' | 'coarse' | 'none')
    | 'color_scheme' ':' ('light' | 'dark')
    | 'prefers_reduced_motion' ':' ('reduce' | 'no_preference')
    ;

// Validation Rules
validation_rules
    : '{' validation_rule (',' validation_rule)* '}'
    | validation_rule
    ;

validation_rule
    : 'required' | 'email' | 'url' | 'numeric' | 'alpha' | 'alphanumeric'
    | 'min_length' ':' NUMBER
    | 'max_length' ':' NUMBER
    | 'min_value' ':' NUMBER
    | 'max_value' ':' NUMBER
    | 'pattern' ':' REGEX
    | 'custom' ':' IDENTIFIER
    | 'async' ':' IDENTIFIER
    | 'cross_field' ':' cross_field_validation
    ;

cross_field_validation
    : field_comparison
    | custom_validation_function
    ;

field_comparison
    : IDENTIFIER comparison_operator IDENTIFIER
    ;

comparison_operator
    : '==' | '!=' | '<' | '>' | '<=' | '>=' | 'matches' | 'not_matches'
    ;

// ========================================
// MISSING RULE DEFINITIONS (Form Layout)
// ========================================

// Missing form layout property types
behavior_property
    : 'onclick' ':' action_specification
    | 'onchange' ':' action_specification
    | 'onsubmit' ':' action_specification
    | 'onhover' ':' action_specification
    | 'onfocus' ':' action_specification
    | 'onblur' ':' action_specification
    | 'draggable' ':' BOOLEAN
    | 'resizable' ':' BOOLEAN
    | 'sortable' ':' BOOLEAN
    ;

data_property
    : 'binding' ':' data_binding_spec
    | 'source' ':' data_source_spec
    | 'model' ':' IDENTIFIER
    | 'computed' ':' computed_expression
    | 'watch' ':' watcher_spec
    | 'cache' ':' cache_spec
    ;

accessibility_property
    : 'aria_label' ':' STRING
    | 'aria_describedby' ':' STRING
    | 'role' ':' STRING
    | 'tabindex' ':' NUMBER
    | 'alt' ':' STRING
    | 'title' ':' STRING
    | 'lang' ':' STRING
    | 'screen_reader_only' ':' BOOLEAN
    ;

// Missing CSS value types
transform_value
    : transform_function+
    ;

transform_function
    : 'translate' '(' size_value (',' size_value)? ')'
    | 'scale' '(' NUMBER (',' NUMBER)? ')'
    | 'rotate' '(' angle_value ')'
    | 'skew' '(' angle_value (',' angle_value)? ')'
    | 'matrix' '(' NUMBER (',' NUMBER)* ')'
    ;

filter_value
    : filter_function+
    ;

filter_function
    : 'blur' '(' size_value ')'
    | 'brightness' '(' NUMBER ')'
    | 'contrast' '(' NUMBER ')'
    | 'grayscale' '(' NUMBER ')'
    | 'hue_rotate' '(' angle_value ')'
    | 'invert' '(' NUMBER ')'
    | 'opacity' '(' NUMBER ')'
    | 'saturate' '(' NUMBER ')'
    | 'sepia' '(' NUMBER ')'
    ;

shadow_value
    : size_value size_value size_value? size_value? color_value?
    | 'none' | 'inherit' | 'initial'
    ;

gradient_value
    : 'linear_gradient' '(' gradient_direction? gradient_stops ')'
    | 'radial_gradient' '(' gradient_shape? gradient_stops ')'
    | 'conic_gradient' '(' gradient_angle? gradient_stops ')'
    ;

gradient_direction
    : angle_value
    | 'to' ('top' | 'bottom' | 'left' | 'right' | 'top' 'left' | 'top' 'right' | 'bottom' 'left' | 'bottom' 'right')
    ;

gradient_stops
    : gradient_stop (',' gradient_stop)*
    ;

gradient_stop
    : color_value percentage_value?
    ;

clip_value
    : 'polygon' '(' clip_points ')'
    | 'circle' '(' size_value ('at' position_value)? ')'
    | 'ellipse' '(' size_value size_value ('at' position_value)? ')'
    | 'inset' '(' size_value size_value? size_value? size_value? ('round' size_value)? ')'
    ;

clip_points
    : clip_point (',' clip_point)*
    ;

clip_point
    : percentage_value percentage_value
    ;

mask_value
    : mask_source mask_position? mask_size? mask_repeat? mask_origin? mask_clip?
    | 'none'
    ;

// Missing field property specifications
format_specification
    : 'date' ':' date_format
    | 'number' ':' number_format
    | 'currency' ':' currency_format
    | 'custom' ':' custom_format
    ;

input_mask
    : mask_pattern mask_options?
    ;

mask_pattern
    : STRING  // e.g., "999-99-9999" for SSN
    ;

mask_options
    : '{' ('placeholder' ':' STRING | 'showMask' ':' BOOLEAN)* '}'
    ;

autocomplete_spec
    : autocomplete_source autocomplete_options?
    ;

autocomplete_source
    : 'static' ':' '[' STRING (',' STRING)* ']'
    | 'dynamic' ':' api_endpoint
    | 'function' ':' IDENTIFIER
    ;

autocomplete_options
    : '{' ('minLength' ':' NUMBER | 'maxResults' ':' NUMBER | 'debounce' ':' NUMBER)* '}'
    ;

dependency_spec
    : 'depends_on' ':' '[' IDENTIFIER (',' IDENTIFIER)* ']'
    | 'cascade' ':' cascade_rule
    ;

conditional_spec
    : 'show_if' ':' condition_expression
    | 'enable_if' ':' condition_expression
    | 'require_if' ':' condition_expression
    ;

accessibility_spec
    : '{' accessibility_rule (',' accessibility_rule)* '}'
    ;

accessibility_rule
    : 'aria_label' ':' STRING
    | 'screen_reader' ':' STRING
    | 'keyboard_navigation' ':' BOOLEAN
    | 'focus_trap' ':' BOOLEAN
    | 'high_contrast' ':' BOOLEAN
    ;

// Supporting value types
angle_value
    : NUMBER ('deg' | 'rad' | 'grad' | 'turn')
    ;

percentage_value
    : NUMBER '%'
    ;

viewport_value
    : NUMBER ('vw' | 'vh' | 'vmin' | 'vmax')
    ;

calc_expression
    : 'calc' '(' calc_operand (calc_operator calc_operand)* ')'
    ;

calc_operand
    : size_value | NUMBER | percentage_value
    ;

calc_operator
    : '+' | '-' | '*' | '/'
    ;

// Additional missing component specification rules
icon_specification
    : icon_name icon_style? icon_size?
    ;

icon_name
    : STRING | IDENTIFIER
    ;

icon_style
    : 'solid' | 'outline' | 'filled' | 'twotone'
    ;

icon_size
    : 'small' | 'medium' | 'large' | size_value
    ;

action_specification
    : function_call
    | event_handler
    | route_action
    | custom_action
    ;

function_call
    : IDENTIFIER '(' args? ')'
    ;

event_handler
    : 'emit' ':' STRING
    | 'dispatch' ':' STRING
    | 'trigger' ':' STRING
    ;

route_action
    : 'navigate' ':' STRING
    | 'redirect' ':' STRING
    | 'back' | 'forward' | 'reload'
    ;

custom_action
    : '{' action_property (',' action_property)* '}'
    ;

action_property
    : IDENTIFIER ':' value_expr
    ;

data_specification
    : data_binding | data_source | data_model
    ;

data_binding
    : 'bind' ':' IDENTIFIER
    | 'two_way' ':' IDENTIFIER
    ;

data_source
    : 'api' ':' api_endpoint
    | 'static' ':' value_expr
    | 'computed' ':' computed_expression
    ;

data_model
    : 'model' ':' IDENTIFIER
    | 'schema' ':' schema_definition
    ;

state_specification
    : state_property+
    ;

state_property
    : 'initial' ':' value_expr
    | 'computed' ':' computed_expression
    | 'persistent' ':' BOOLEAN
    | 'reactive' ':' BOOLEAN
    ;

theme_specification
    : 'theme' ':' STRING
    | 'variant' ':' STRING
    | 'custom' ':' theme_definition
    ;

theme_definition
    : '{' theme_property (',' theme_property)* '}'
    ;

theme_property
    : 'colors' ':' color_palette
    | 'typography' ':' typography_scale
    | 'spacing' ':' spacing_scale
    | 'shadows' ':' shadow_scale
    ;

variant_specification
    : 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info'
    | 'light' | 'dark' | 'outlined' | 'contained' | 'text'
    | STRING  // Custom variant
    ;

size_specification
    : 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'auto' | size_value
    ;

alignment_specification
    : 'left' | 'center' | 'right' | 'justify'
    | 'start' | 'end' | 'stretch' | 'baseline'
    | 'space_between' | 'space_around' | 'space_evenly'
    ;

// Missing CSS unit types
unit
    : 'px' | 'em' | 'rem' | 'vh' | 'vw' | 'vmin' | 'vmax'
    | '%' | 'pt' | 'pc' | 'in' | 'cm' | 'mm'
    | 'deg' | 'rad' | 'grad' | 'turn'
    | 's' | 'ms'
    ;

// Missing color value types
hex_color
    : '#' HEX_DIGIT HEX_DIGIT HEX_DIGIT
    | '#' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    | '#' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    ;

rgb_color
    : 'rgb' '(' NUMBER ',' NUMBER ',' NUMBER ')'
    | 'rgba' '(' NUMBER ',' NUMBER ',' NUMBER ',' NUMBER ')'
    ;

hsl_color
    : 'hsl' '(' NUMBER ',' percentage_value ',' percentage_value ')'
    | 'hsla' '(' NUMBER ',' percentage_value ',' percentage_value ',' NUMBER ')'
    ;

named_color
    : 'red' | 'green' | 'blue' | 'white' | 'black' | 'gray' | 'grey'
    | 'yellow' | 'orange' | 'purple' | 'pink' | 'brown' | 'cyan'
    | 'magenta' | 'lime' | 'maroon' | 'navy' | 'olive' | 'teal'
    | 'silver' | 'aqua' | 'fuchsia' | 'transparent' | 'currentColor'
    ;

css_variable
    : 'var' '(' '--' IDENTIFIER (',' value_expr)? ')'
    ;

// Missing border value components
border_width
    : size_value | 'thin' | 'medium' | 'thick'
    ;

border_style
    : 'solid' | 'dashed' | 'dotted' | 'double' | 'groove' | 'ridge'
    | 'inset' | 'outset' | 'none' | 'hidden'
    ;

border_color
    : color_value
    ;

// Missing font components
font_family
    : STRING | font_family_name
    ;

font_family_name
    : 'serif' | 'sans_serif' | 'monospace' | 'cursive' | 'fantasy'
    | 'system_ui' | 'ui_serif' | 'ui_sans_serif' | 'ui_monospace'
    ;

font_size
    : size_value | font_size_keyword
    ;

font_size_keyword
    : 'xx_small' | 'x_small' | 'small' | 'medium' | 'large' | 'x_large' | 'xx_large'
    | 'smaller' | 'larger'
    ;

font_weight
    : NUMBER | font_weight_keyword
    ;

font_weight_keyword
    : 'normal' | 'bold' | 'bolder' | 'lighter'
    | 'thin' | 'light' | 'medium' | 'semibold' | 'extrabold' | 'black'
    ;

font_style
    : 'normal' | 'italic' | 'oblique'
    ;

// Missing animation components
animation_name
    : IDENTIFIER | STRING
    ;

duration
    : NUMBER ('s' | 'ms')
    ;

timing_function
    : 'linear' | 'ease' | 'ease_in' | 'ease_out' | 'ease_in_out'
    | 'cubic_bezier' '(' NUMBER ',' NUMBER ',' NUMBER ',' NUMBER ')'
    | 'steps' '(' NUMBER (',' ('start' | 'end'))? ')'
    ;

delay
    : duration
    ;

iteration_count
    : NUMBER | 'infinite'
    ;

direction
    : 'normal' | 'reverse' | 'alternate' | 'alternate_reverse'
    ;

fill_mode
    : 'none' | 'forwards' | 'backwards' | 'both'
    ;

// Missing transition components
transition_property
    : 'all' | IDENTIFIER | STRING
    ;

// Additional missing rules for form layout and CSS
resolution_value
    : NUMBER 'dpi' | NUMBER 'dpcm' | NUMBER 'dppx'
    ;

custom_validation_function
    : IDENTIFIER '(' args? ')'
    ;

data_binding_spec
    : simple_binding | complex_binding
    ;

simple_binding
    : IDENTIFIER
    ;

complex_binding
    : '{' binding_property (',' binding_property)* '}'
    ;

binding_property
    : 'path' ':' STRING
    | 'transform' ':' IDENTIFIER
    | 'validate' ':' BOOLEAN
    | 'debounce' ':' NUMBER
    ;

data_source_spec
    : api_endpoint | static_data | computed_data
    ;

static_data
    : '[' value_expr (',' value_expr)* ']'
    | '{' object_property (',' object_property)* '}'
    ;

computed_data
    : 'computed' ':' computed_expression
    ;

computed_expression
    : expression | lambda_expression
    ;

lambda_expression
    : '(' parameter_list? ')' '=>' expression
    ;

watcher_spec
    : '{' watch_property (',' watch_property)* '}'
    ;

watch_property
    : 'path' ':' STRING
    | 'handler' ':' IDENTIFIER
    | 'deep' ':' BOOLEAN
    | 'immediate' ':' BOOLEAN
    ;

cache_spec
    : '{' cache_property (',' cache_property)* '}'
    ;

cache_property
    : 'ttl' ':' duration
    | 'size' ':' NUMBER
    | 'strategy' ':' ('lru' | 'fifo' | 'lfu')
    | 'key' ':' STRING
    ;

gradient_shape
    : 'circle' | 'ellipse'
    | 'closest_side' | 'closest_corner' | 'farthest_side' | 'farthest_corner'
    ;

gradient_angle
    : angle_value | 'from' angle_value
    ;

mask_source
    : 'url' '(' STRING ')'
    | 'linear_gradient' '(' gradient_stops ')'
    | 'radial_gradient' '(' gradient_stops ')'
    | 'none'
    ;

mask_position
    : 'top' | 'bottom' | 'left' | 'right' | 'center'
    | percentage_value percentage_value?
    | size_value size_value?
    ;

mask_size
    : 'auto' | 'contain' | 'cover'
    | size_value size_value?
    | percentage_value percentage_value?
    ;

mask_repeat
    : 'repeat' | 'repeat_x' | 'repeat_y' | 'no_repeat' | 'space' | 'round'
    ;

mask_origin
    : 'border_box' | 'padding_box' | 'content_box'
    ;

mask_clip
    : 'border_box' | 'padding_box' | 'content_box' | 'text'
    ;

// Additional format specifications
date_format
    : 'iso' | 'us' | 'eu' | 'custom' ':' STRING
    ;

number_format
    : 'integer' | 'decimal' ':' NUMBER | 'scientific' | 'percentage'
    ;

currency_format
    : 'symbol' ':' STRING | 'code' ':' STRING | 'locale' ':' STRING
    ;

custom_format
    : 'pattern' ':' STRING | 'function' ':' IDENTIFIER
    ;

// Color palette and typography
color_palette
    : '{' color_definition (',' color_definition)* '}'
    ;

color_definition
    : IDENTIFIER ':' color_value
    ;

typography_scale
    : '{' typography_definition (',' typography_definition)* '}'
    ;

typography_definition
    : IDENTIFIER ':' font_definition
    ;

font_definition
    : '{' font_property (',' font_property)* '}'
    ;

font_property
    : 'family' ':' font_family
    | 'size' ':' font_size
    | 'weight' ':' font_weight
    | 'style' ':' font_style
    | 'line_height' ':' NUMBER
    | 'letter_spacing' ':' size_value
    ;

spacing_scale
    : '{' spacing_definition (',' spacing_definition)* '}'
    ;

spacing_definition
    : IDENTIFIER ':' size_value
    ;

shadow_scale
    : '{' shadow_definition (',' shadow_definition)* '}'
    ;

shadow_definition
    : IDENTIFIER ':' shadow_value
    ;

// API endpoint specification
api_endpoint
    : url_string | endpoint_config
    ;

url_string
    : STRING
    ;

endpoint_config
    : '{' endpoint_property (',' endpoint_property)* '}'
    ;

endpoint_property
    : 'url' ':' STRING
    | 'method' ':' ('GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH')
    | 'headers' ':' '{' header_property (',' header_property)* '}'
    | 'params' ':' '{' param_property (',' param_property)* '}'
    | 'auth' ':' auth_config
    ;

header_property
    : STRING ':' STRING
    ;

param_property
    : STRING ':' value_expr
    ;

auth_config
    : 'bearer' ':' STRING
    | 'basic' ':' STRING
    | 'api_key' ':' STRING
    | 'none'
    ;

// Schema definition
schema_definition
    : '{' schema_property (',' schema_property)* '}'
    ;

schema_property
    : IDENTIFIER ':' type_annotation
    ;

// Cascade rule
cascade_rule
    : '{' cascade_step (',' cascade_step)* '}'
    ;

cascade_step
    : 'when' ':' condition_expression 'then' ':' action_specification
    ;

condition_expression
    : expression
    ;

object_property
    : IDENTIFIER ':' value_expr
    | STRING ':' value_expr
    ;

// Missing HEX_DIGIT token
fragment HEX_DIGIT
    : [0-9a-fA-F]
    ;

// Additional missing rules for lambda expressions and parameters
parameter_list
    : parameter (',' parameter)*
    ;

// Missing duration value (different from duration)
duration_value
    : NUMBER time_unit
    ;

time_unit
    : 's' | 'ms' | 'min' | 'h' | 'hour' | 'hours' | 'day' | 'days'
    ;

// ========================================
// DEVELOPER ERGONOMICS EXTENSIONS
// ========================================

// Testing Framework Integration
test_definition
    : test_type IDENTIFIER test_configuration? '{' test_body '}'
    ;

test_type
    : 'unit_test' | 'integration_test' | 'e2e_test' | 'load_test'
    | 'security_test' | 'accessibility_test' | 'visual_test'
    | 'smoke_test' | 'regression_test' | 'acceptance_test'
    ;

test_configuration
    : '[' test_config_item (',' test_config_item)* ']'
    ;

test_config_item
    : 'timeout' ':' duration_value
    | 'retries' ':' NUMBER
    | 'parallel' ':' BOOLEAN
    | 'tags' ':' '[' STRING (',' STRING)* ']'
    | 'setup' ':' IDENTIFIER
    | 'teardown' ':' IDENTIFIER
    | 'fixtures' ':' '[' IDENTIFIER (',' IDENTIFIER)* ']'
    | 'mocks' ':' mock_configuration
    | 'environment' ':' STRING
    | 'browser' ':' browser_specification
    | 'device' ':' device_specification
    ;

// Missing browser and device specifications
browser_specification
    : 'chrome' | 'firefox' | 'safari' | 'edge' | 'ie'
    | browser_config
    ;

browser_config
    : '{' browser_property (',' browser_property)* '}'
    ;

browser_property
    : 'name' ':' STRING
    | 'version' ':' STRING
    | 'headless' ':' BOOLEAN
    | 'mobile' ':' BOOLEAN
    | 'width' ':' NUMBER
    | 'height' ':' NUMBER
    ;

device_specification
    : 'desktop' | 'mobile' | 'tablet'
    | device_config
    ;

device_config
    : '{' device_property (',' device_property)* '}'
    ;

device_property
    : 'name' ':' STRING
    | 'width' ':' NUMBER
    | 'height' ':' NUMBER
    | 'dpr' ':' NUMBER
    | 'mobile' ':' BOOLEAN
    | 'touch' ':' BOOLEAN
    ;

test_body
    : test_step*
    ;

test_step
    : test_assertion
    | test_action
    | test_setup
    | test_verification
    ;

// Missing test step types
test_action
    : 'click' '(' selector_expr ')'
    | 'type' '(' selector_expr ',' STRING ')'
    | 'wait' '(' duration ')'
    | 'navigate' '(' STRING ')'
    | 'screenshot' '(' STRING? ')'
    | 'scroll' '(' scroll_options ')'
    | 'hover' '(' selector_expr ')'
    | 'select' '(' selector_expr ',' option_expr ')'
    | custom_action
    ;

test_setup
    : 'setup' ':' setup_action
    | 'before_each' ':' setup_action
    | 'before_all' ':' setup_action
    ;

setup_action
    : function_call | code_block
    ;

test_verification
    : 'verify' verification_condition
    | 'check' verification_condition
    ;

verification_condition
    : element_verification | state_verification | data_verification
    ;

element_verification
    : selector_expr property_matcher
    ;

state_verification
    : 'state' IDENTIFIER matcher_expression
    ;

data_verification
    : 'data' IDENTIFIER matcher_expression
    ;

test_assertion
    : 'assert' assertion_expression ';'
    | 'expect' expectation_expression ';'
    | 'verify' verification_expression ';'
    ;

assertion_expression
    : expression comparison_operator expression
    | expression matcher_expression
    | 'throws' '(' exception_type? ')'
    | 'not_throws'
    ;

// Missing expectation and verification expressions
expectation_expression
    : expression matcher_expression
    | async_expectation
    ;

async_expectation
    : 'eventually' '(' assertion_expression ')'
    | 'within' '(' duration ')' assertion_expression
    ;

verification_expression
    : assertion_expression
    | element_state_verification
    ;

element_state_verification
    : selector_expr 'is' element_state
    | selector_expr 'has' element_property
    ;

element_state
    : 'visible' | 'hidden' | 'enabled' | 'disabled' | 'selected' | 'checked'
    ;

element_property
    : 'text' '(' STRING ')'
    | 'value' '(' value_expr ')'
    | 'attribute' '(' STRING ',' value_expr ')'
    | 'class' '(' STRING ')'
    ;

// Missing exception type
exception_type
    : IDENTIFIER | STRING
    ;

// Additional missing test-related rules
selector_expr
    : css_selector | xpath_selector | id_selector | class_selector
    ;

css_selector
    : STRING  // CSS selector string like "#myId", ".myClass", "div > p"
    ;

xpath_selector
    : 'xpath' '(' STRING ')'
    ;

id_selector
    : '#' IDENTIFIER
    ;

class_selector
    : '.' IDENTIFIER
    ;

scroll_options
    : '{' scroll_property (',' scroll_property)* '}'
    | scroll_direction
    ;

scroll_property
    : member_name ':' contract_value
    ;

scroll_direction
    : 'up' | 'down' | 'left' | 'right'
    ;

option_expr
    : STRING | NUMBER | IDENTIFIER
    ;

code_block
    : '{' statement* '}'
    ;

property_matcher
    : 'should' matcher_expression
    | 'must' matcher_expression
    ;

custom_matcher
    : IDENTIFIER '(' args? ')'
    ;

// Mock and spy options
spy_options
    : '{' spy_property (',' spy_property)* '}'
    ;

spy_property
    : 'calls' ':' BOOLEAN
    | 'return_value' ':' value_expr
    | 'side_effect' ':' IDENTIFIER
    ;

stub_options
    : '{' stub_property (',' stub_property)* '}'
    ;

stub_property
    : 'return_value' ':' value_expr
    | 'implementation' ':' IDENTIFIER
    | 'calls_through' ':' BOOLEAN
    ;

fake_options
    : '{' fake_property (',' fake_property)* '}'
    ;

fake_property
    : 'data' ':' value_expr
    | 'behavior' ':' IDENTIFIER
    ;

// Debug and profiling expressions
trace_expression
    : 'trace' '(' STRING? ')'
    | 'debug' '(' STRING? ')'
    ;

profile_expression
    : 'profile' '(' STRING? ')'
    | 'time' '(' STRING? ')'
    ;

benchmark_expression
    : 'benchmark' '(' STRING? ')'
    | 'measure' '(' STRING? ')'
    ;

// Configuration specifications
vault_specification
    : vault_provider vault_config?
    ;

vault_provider
    : 'hashicorp_vault' | 'aws_secrets' | 'azure_keyvault' | 'gcp_secret_manager'
    ;

vault_config
    : '{' vault_property (',' vault_property)* '}'
    ;

vault_property
    : 'url' ':' STRING
    | 'token' ':' STRING
    | 'path' ':' STRING
    | 'version' ':' STRING
    ;

database_specification
    : database_type database_config?
    ;

database_type
    : 'postgresql' | 'mysql' | 'sqlite' | 'mongodb' | 'redis' | 'elasticsearch'
    ;

database_config
    : '{' database_property (',' database_property)* '}'
    ;

database_property
    : 'url' ':' STRING
    | 'host' ':' STRING
    | 'port' ':' NUMBER
    | 'database' ':' STRING
    | 'username' ':' STRING
    | 'password' ':' STRING
    | 'pool_size' ':' NUMBER
    | 'ssl' ':' BOOLEAN
    ;

// ========================================
// DBML INTEGRATION - Database Schema Definition Language
// Comprehensive table, column, and relationship modeling
// ========================================

// Main database schema construct
database_schema
    : 'schema' IDENTIFIER '{' schema_element* '}'
    ;

// ENUM VARIANT DECLARATIONS
// Used inside enum entity bodies.
enum_variant_decl
    : IDENTIFIER ('=' (NUMBER | STRING))? enum_variant_doc? contract_separator?
    ;

enum_variant_doc
    : '[' 'label' ':' STRING (',' 'description' ':' STRING)? ']'
    ;

// STATE MACHINE TRANSITIONS
// Used inside statemachine/fsm entity bodies.
state_transition
    : IDENTIFIER '->' IDENTIFIER ('[' state_transition_props ']')? contract_separator?
    ;

state_transition_props
    : state_transition_prop (',' state_transition_prop)*
    ;

state_transition_prop
    : 'on' ':' contract_scalar
    | 'guard' ':' contract_value
    | 'action' ':' contract_value
    | 'priority' ':' NUMBER
    | 'timeout' ':' time_expr
    | IDENTIFIER ':' contract_value
    ;

schema_element
    : table_definition
    | enum_definition
    | table_reference
    | table_group
    | note_definition
    | trigger_definition
    | procedure_definition
    | function_definition
    | view_definition
    ;

// Table definition with DBML capabilities
table_definition
    : 'table' IDENTIFIER table_alias? table_note? '{' table_body '}'
    ;

table_alias
    : 'as' IDENTIFIER
    ;

table_note
    : '[' 'note' ':' STRING ']'
    ;

table_body
    : (column_definition | table_index | table_constraint | vector_index_definition | trigger_definition)*
    ;

// Column definition with comprehensive typing and constraints
column_definition
    : IDENTIFIER column_type column_nullable? column_constraints?
    ;

column_type
    : db_data_type
    | db_data_type '(' NUMBER ')'  // varchar(255)
    | db_data_type '(' NUMBER ',' NUMBER ')'  // decimal(10,2)
    ;

db_data_type
    : 'int' | 'integer' | 'bigint' | 'smallint' | 'tinyint'
    | 'varchar' | 'char' | 'text' | 'longtext' | 'mediumtext'
    | 'decimal' | 'numeric' | 'float' | 'double' | 'real'
    | 'boolean' | 'bool' | 'bit'
    | 'date' | 'time' | 'datetime' | 'timestamp' | 'year'
    | 'json' | 'jsonb' | 'xml'
    | 'binary' | 'varbinary' | 'blob' | 'longblob'
    | 'uuid' | 'serial' | 'bigserial'
    | 'point' | 'polygon' | 'geometry' | 'geography'
    | 'inet' | 'cidr' | 'macaddr'
    | 'array' | 'hstore'
    // Vector and ML/AI data types
    | 'vector' | 'embedding' | 'float4' | 'float8'
    | 'halfvec' | 'sparsevec' | 'bit_vector'
    // Time series and specialized types
    | 'timeseries' | 'tsrange' | 'tstzrange'
    | 'interval' | 'money' | 'tsvector' | 'tsquery'
    ;

column_nullable
    : 'null' | 'not' 'null'
    ;

column_constraints
    : '[' column_constraint (',' column_constraint)* ']'
    | '[' vector_column_constraint (',' vector_column_constraint)* ']'
    ;

column_constraint
    : 'pk' | 'primary' 'key'  // Primary key
    | 'unique'
    | 'not' 'null'
    | 'increment' | 'auto_increment'
    | 'default' ':' (NUMBER | STRING | BOOLEAN | 'now()' | 'uuid()' | NULL | NONE)
    | 'note' ':' STRING
    | 'ref' reference_spec  // Foreign key reference
    ;

// Table relationships and references
reference_spec
    : reference_type table_column_ref
    ;

reference_type
    : '>' | '<' | '-' | '<>'  // one-to-many, many-to-one, one-to-one, many-to-many
    ;

table_column_ref
    : IDENTIFIER '.' IDENTIFIER  // table.column
    ;

// Standalone table references (for complex relationships)
table_reference
    : 'ref' IDENTIFIER? ':' table_column_ref reference_type table_column_ref reference_options?
    ;

reference_options
    : '[' reference_option (',' reference_option)* ']'
    ;

reference_option
    : 'delete' ':' reference_action
    | 'update' ':' reference_action
    | 'note' ':' STRING
    ;

reference_action
    : 'cascade' | 'restrict' | 'set_null' | 'set_default' | 'no_action'
    ;

// Table indexes
table_index
    : 'indexes' '{' index_definition* '}'
    ;

index_definition
    : index_columns index_options?
    ;

index_columns
    : '(' IDENTIFIER (',' IDENTIFIER)* ')'  // (col1, col2)
    | IDENTIFIER  // single column
    ;

index_options
    : '[' index_option (',' index_option)* ']'
    ;

index_option
    : 'unique'
    | 'type' ':' index_type
    | 'note' ':' STRING
    | 'name' ':' STRING
    ;

index_type
    : 'btree' | 'hash' | 'gin' | 'gist' | 'spgist' | 'brin'
    ;

// Table constraints (check constraints, etc.)
table_constraint
    : 'constraint' IDENTIFIER constraint_definition
    | constraint_definition
    ;

constraint_definition
    : 'check' '(' expression ')'
    | 'unique' '(' IDENTIFIER (',' IDENTIFIER)* ')'
    | 'foreign' 'key' '(' IDENTIFIER (',' IDENTIFIER)* ')' 'references' table_column_ref
    ;

// Enums for database values
enum_definition
    : 'enum' IDENTIFIER '{' enum_values '}'
    ;

enum_values
    : enum_value (',' enum_value)*
    ;

enum_value
    : IDENTIFIER enum_note?
    ;

enum_note
    : '[' 'note' ':' STRING ']'
    ;

// Table groups for organization
table_group
    : 'tablegroup' IDENTIFIER '{' IDENTIFIER (',' IDENTIFIER)* '}'
    ;

// Standalone notes
note_definition
    : 'note' '{' STRING '}'
    | 'note' IDENTIFIER '{' STRING '}'
    ;

// ========================================
// EXTENDED DBML - TRIGGERS AND PROCEDURES
// Advanced database programming capabilities
// ========================================

// Trigger definitions for automated database logic
trigger_definition
    : 'trigger' IDENTIFIER trigger_spec '{' trigger_body '}'
    ;

trigger_spec
    : trigger_timing trigger_event 'on' IDENTIFIER trigger_condition?
    ;

trigger_timing
    : 'before' | 'after' | 'instead_of'
    ;

trigger_event
    : 'insert' | 'update' | 'delete' | 'truncate'
    | 'insert' 'or' 'update'
    | 'update' 'or' 'delete'
    | 'insert' 'or' 'update' 'or' 'delete'
    ;

trigger_condition
    : 'when' '(' expression ')'
    | 'for_each' trigger_level
    ;

trigger_level
    : 'row' | 'statement'
    ;

trigger_body
    : trigger_statement*
    ;

trigger_statement
    : 'begin' db_statement_block 'end'
    | 'execute' 'procedure' IDENTIFIER '(' db_parameter_list? ')'
    | 'execute' sql_statement
    | apg_statement  // Allows full APG language constructs
    ;

// Stored procedure definitions
procedure_definition
    : 'procedure' IDENTIFIER '(' procedure_parameters? ')' procedure_options? '{' procedure_body '}'
    ;

procedure_parameters
    : procedure_parameter (',' procedure_parameter)*
    ;

procedure_parameter
    : parameter_mode? IDENTIFIER column_type default_value?
    ;

parameter_mode
    : 'in' | 'out' | 'inout'
    ;

default_value
    : 'default' (NUMBER | STRING | BOOLEAN | NULL | NONE)
    ;

procedure_options
    : '[' procedure_option (',' procedure_option)* ']'
    ;

procedure_option
    : 'language' ':' procedure_language
    | 'security' ':' ('definer' | 'invoker')
    | 'cost' ':' NUMBER
    | 'rows' ':' NUMBER
    | 'immutable' | 'stable' | 'volatile'
    | 'strict' | 'returns_null_on_null_input'
    | 'parallel' ':' ('safe' | 'unsafe' | 'restricted')
    ;

procedure_language
    : 'sql' | 'plpgsql' | 'python' | 'javascript' | 'apg'
    ;

procedure_body
    : procedure_statement*
    ;

procedure_statement
    : variable_declaration
    | sql_statement
    | control_flow_statement
    | exception_handling
    | db_return_statement
    | apg_statement  // Full APG language integration
    ;

// Database function definitions (similar to procedures but with return types)
function_definition
    : 'function' IDENTIFIER '(' procedure_parameters? ')' 'returns' db_return_type function_options? '{' procedure_body '}'
    ;

db_return_type
    : column_type
    | 'table' '(' table_column_list ')'
    | 'setof' column_type
    | 'void'
    ;

table_column_list
    : table_column_def (',' table_column_def)*
    ;

table_column_def
    : IDENTIFIER column_type
    ;

function_options
    : '[' function_option (',' function_option)* ']'
    ;

function_option
    : procedure_option  // Inherits all procedure options
    | 'returns_null_on_null_input'
    | 'called_on_null_input'
    ;

// Database view definitions
view_definition
    : 'view' IDENTIFIER view_options? 'as' '{' sql_query '}'
    ;

view_options
    : '[' view_option (',' view_option)* ']'
    ;

view_option
    : 'materialized'
    | 'security_barrier'
    | 'check_option' ':' ('local' | 'cascaded')
    | 'with_data' | 'with_no_data'
    ;

// SQL statement support within database objects
sql_statement
    : select_statement
    | insert_statement
    | update_statement
    | delete_statement
    | execute_statement
    | sql_expression ';'
    ;

sql_query
    : select_statement
    ;

select_statement
    : 'select' select_list from_clause? where_clause? group_by_clause? having_clause? order_by_clause? limit_clause?
    ;

select_list
    : '*'
    | select_item (',' select_item)*
    ;

select_item
    : expression alias?
    | IDENTIFIER '.' '*'
    ;

from_clause
    : 'from' table_reference (',' table_reference)*
    ;

where_clause
    : 'where' expression
    ;

group_by_clause
    : 'group' 'by' expression (',' expression)*
    ;

having_clause
    : 'having' expression
    ;

order_by_clause
    : 'order' 'by' order_item (',' order_item)*
    ;

order_item
    : expression ('asc' | 'desc')? ('nulls' ('first' | 'last'))?
    ;

limit_clause
    : 'limit' NUMBER ('offset' NUMBER)?
    ;

insert_statement
    : 'insert' 'into' IDENTIFIER '(' column_list? ')' ('values' '(' value_list ')' | select_statement)
    ;

update_statement
    : 'update' IDENTIFIER 'set' assignment_list where_clause?
    ;

delete_statement
    : 'delete' 'from' IDENTIFIER where_clause?
    ;

execute_statement
    : 'execute' IDENTIFIER '(' db_parameter_list? ')'
    ;

column_list
    : IDENTIFIER (',' IDENTIFIER)*
    ;

value_list
    : expression (',' expression)*
    ;

assignment_list
    : db_assignment (',' db_assignment)*
    ;

db_assignment
    : IDENTIFIER '=' expression
    ;

// Control flow within procedures/functions
control_flow_statement
    : if_statement
    | loop_statement
    | while_statement
    | for_statement
    | case_statement
    ;

loop_statement
    : 'loop' db_statement_block 'end' 'loop'
    ;

case_statement
    : 'case' expression? db_when_clause+ db_else_clause? 'end' 'case'
    ;

db_when_clause
    : 'when' expression 'then' db_statement_block
    ;

db_else_clause
    : 'else' db_statement_block
    ;

// Exception handling in procedures
exception_handling
    : 'exception' 'when' exception_condition 'then' db_statement_block
    ;

exception_condition
    : IDENTIFIER
    | 'others'
    | 'sqlstate' STRING
    ;

db_return_statement
    : 'return' expression? ';'
    ;

// Vector and AI/ML specific enhancements
vector_index_definition
    : 'vector_index' IDENTIFIER 'on' IDENTIFIER '(' IDENTIFIER ')' vector_index_options?
    ;

vector_index_options
    : '[' vector_index_option (',' vector_index_option)* ']'
    ;

vector_index_option
    : 'method' ':' vector_index_method
    | 'distance' ':' distance_function
    | 'dimensions' ':' NUMBER
    | 'ef_construction' ':' NUMBER
    | 'ef_search' ':' NUMBER
    | IDENTIFIER ':' NUMBER
    ;

vector_index_method
    : 'ivfflat' | 'hnsw' | 'ivf_pq' | 'flat' | 'lsh'
    ;

distance_function
    : 'cosine' | 'euclidean' | 'manhattan' | 'dot_product' | 'hamming'
    ;

// AI/ML specific table constraints and checks
vector_constraint
    : 'check' '(' vector_constraint_expression ')'
    ;

vector_constraint_expression
    : 'vector_dims' '(' IDENTIFIER ')' '=' NUMBER
    | 'vector_norm' '(' IDENTIFIER ')' ('<' | '<=' | '>' | '>=') NUMBER
    | 'cosine_similarity' '(' IDENTIFIER ',' IDENTIFIER ')' ('<' | '<=' | '>' | '>=') NUMBER
    ;

// Enhanced column constraints for vectors
vector_column_constraint
    : column_constraint
    | 'dimensions' ':' NUMBER
    | 'normalized'
    | 'sparse'
    | 'distance_function' ':' distance_function
    ;

// Missing rule definitions for database extensions
apg_statement
    : statement
    ;

sql_expression
    : expression
    ;

db_parameter_list
    : expression (',' expression)*
    ;

db_statement_block
    : '{' statement* '}'
    ;

matcher_expression
    : 'to_be' '(' value_expr ')'
    | 'to_equal' '(' value_expr ')'
    | 'to_contain' '(' value_expr ')'
    | 'to_match' '(' REGEX ')'
    | 'to_be_null' | 'to_be_undefined' | 'to_be_truthy' | 'to_be_falsy'
    | 'to_be_greater_than' '(' NUMBER ')'
    | 'to_be_less_than' '(' NUMBER ')'
    | 'to_have_length' '(' NUMBER ')'
    | 'to_have_property' '(' STRING ')'
    | 'to_be_called' | 'to_be_called_with' '(' args? ')'
    | 'to_be_called_times' '(' NUMBER ')'
    | custom_matcher
    ;

// Mock and Stub Definitions
mock_configuration
    : '{' mock_item (',' mock_item)* '}'
    ;

mock_item
    : IDENTIFIER ':' mock_specification
    ;

mock_specification
    : 'mock' '(' mock_options? ')'
    | 'spy' '(' spy_options? ')'
    | 'stub' '(' stub_options? ')'
    | 'fake' '(' fake_options? ')'
    ;

mock_options
    : mock_option (',' mock_option)*
    ;

mock_option
    : 'return_value' ':' value_expr
    | 'side_effect' ':' IDENTIFIER
    | 'call_count' ':' NUMBER
    | 'implementation' ':' lambda_expr
    | 'auto_spec' ':' BOOLEAN
    | 'spec_set' ':' type_annotation
    ;

// Debugging and Profiling
debug_statement
    : 'debug' debug_expression ';'
    | 'breakpoint' breakpoint_condition? ';'
    | 'trace' trace_expression ';'
    | 'profile' profile_expression ';'
    | 'benchmark' benchmark_expression ';'
    ;

debug_expression
    : expression
    | '{' debug_info (',' debug_info)* '}'
    ;

debug_info
    : 'variables' | 'stack_trace' | 'memory_usage' | 'execution_time'
    | 'call_graph' | 'coverage' | 'performance_metrics'
    ;

breakpoint_condition
    : 'when' '(' expression ')'
    | 'after' '(' NUMBER ')'
    | 'if' '(' expression ')'
    ;

// Configuration Management
config_definition
    : config_scope? config_source* config_validation?
    ;

config_scope
    : 'global' | 'local' | 'environment' | 'user' | 'session'
    ;

config_source
    : 'file' ':' STRING
    | 'env' ':' STRING
    | 'vault' ':' vault_specification
    | 'database' ':' database_specification
    | 'api' ':' api_specification
    | 'default' ':' value_expr
    ;

config_validation
    : 'schema' ':' schema_specification
    | 'constraints' ':' constraint_specification
    | 'transformation' ':' transformation_specification
    ;

// Secret and Security Management
secret_definition
    : 'encrypted' ':' BOOLEAN
    | 'key_rotation' ':' duration_value
    | 'access_policy' ':' access_policy_specification
    | 'audit_logging' ':' BOOLEAN
    | 'backup' ':' backup_specification
    ;

// ========================================
// NOTIFICATION AND ALERTING SYSTEM
// ========================================

// Notification Definitions
notification_definition
    : notification_type IDENTIFIER notification_configuration
    ;

notification_type
    : 'email' | 'sms' | 'push' | 'slack' | 'discord' | 'teams'
    | 'webhook' | 'in_app' | 'desktop' | 'mobile' | 'voice'
    | 'pager' | 'incident' | 'broadcast' | 'emergency'
    ;

notification_configuration
    : '{' notification_property* '}'
    ;

notification_property
    : 'recipients' ':' recipient_list
    | 'template' ':' template_specification
    | 'priority' ':' priority_level
    | 'delivery_method' ':' delivery_specification
    | 'retry_policy' ':' retry_specification
    | 'rate_limiting' ':' rate_limit_specification
    | 'scheduling' ':' schedule_specification
    | 'conditions' ':' condition_specification
    | 'escalation' ':' escalation_specification
    | 'tracking' ':' tracking_specification
    | 'personalization' ':' personalization_specification
    | 'localization' ':' localization_specification
    | 'analytics' ':' analytics_specification
    ;

recipient_list
    : '[' recipient (',' recipient)* ']'
    | recipient_group
    | dynamic_recipient_list
    ;

recipient
    : STRING  // Email, phone, username
    | user_reference
    | role_reference
    | group_reference
    ;

recipient_group
    : IDENTIFIER  // Predefined group
    | 'role' '(' STRING ')'
    | 'department' '(' STRING ')'
    | 'location' '(' STRING ')'
    | 'skill' '(' STRING ')'
    ;

dynamic_recipient_list
    : 'query' '(' query_expression ')'
    | 'function' '(' function_call ')'
    | 'api' '(' api_call ')'
    ;

priority_level
    : 'low' | 'normal' | 'high' | 'urgent' | 'critical' | 'emergency'
    | NUMBER  // 1-10 scale
    ;

// Alert Management
alert_definition
    : alert_type IDENTIFIER alert_configuration
    ;

alert_type
    : 'threshold' | 'anomaly' | 'trend' | 'pattern' | 'composite'
    | 'heartbeat' | 'deadman' | 'change' | 'forecast'
    | 'security' | 'performance' | 'business' | 'operational'
    ;

alert_configuration
    : '{' alert_property* '}'
    ;

alert_property
    : 'condition' ':' alert_condition
    | 'severity' ':' severity_level
    | 'frequency' ':' alert_frequency
    | 'suppression' ':' suppression_rules
    | 'enrichment' ':' enrichment_specification
    | 'correlation' ':' correlation_specification
    | 'auto_resolution' ':' auto_resolution_specification
    | 'escalation_policy' ':' escalation_policy_specification
    | 'runbook' ':' runbook_specification
    | 'tags' ':' '[' STRING (',' STRING)* ']'
    | 'metadata' ':' metadata_specification
    ;

alert_condition
    : threshold_condition
    | anomaly_condition
    | pattern_condition
    | composite_condition
    | custom_condition
    ;

threshold_condition
    : metric_reference comparison_operator threshold_value duration_clause?
    ;

anomaly_condition
    : 'anomaly_score' comparison_operator NUMBER
    | 'deviation' comparison_operator 'baseline' percentage_clause?
    | 'outlier_detection' algorithm_specification
    ;

severity_level
    : 'info' | 'warning' | 'error' | 'critical' | 'fatal'
    | NUMBER  // 1-5 scale
    ;

// Logging and Measurement
logger_definition
    : logger_type IDENTIFIER logger_configuration
    ;

logger_type
    : 'structured' | 'json' | 'plain' | 'binary' | 'metric'
    | 'trace' | 'audit' | 'security' | 'performance' | 'business'
    ;

logger_configuration
    : '{' logger_property* '}'
    ;

logger_property
    : 'level' ':' log_level
    | 'format' ':' log_format
    | 'output' ':' output_specification
    | 'rotation' ':' rotation_specification
    | 'retention' ':' retention_specification
    | 'sampling' ':' sampling_specification
    | 'filtering' ':' filter_specification
    | 'enrichment' ':' enrichment_specification
    | 'masking' ':' masking_specification
    | 'compression' ':' compression_specification
    | 'encryption' ':' encryption_specification
    | 'backup' ':' backup_specification
    ;

log_level
    : 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'fatal'
    | 'off' | 'all'
    ;

log_format
    : 'json' | 'logfmt' | 'plain' | 'csv' | 'xml'
    | custom_format_specification
    ;

// Metrics and Measurement
metric_definition
    : metric_type IDENTIFIER metric_configuration
    ;

metric_type
    : 'counter' | 'gauge' | 'histogram' | 'summary' | 'timer'
    | 'rate' | 'ratio' | 'percentage' | 'distribution'
    | 'custom' | 'business' | 'technical' | 'operational'
    ;

metric_configuration
    : '{' metric_property* '}'
    ;

metric_property
    : 'description' ':' STRING
    | 'unit' ':' unit_specification
    | 'labels' ':' label_specification
    | 'buckets' ':' bucket_specification  // For histograms
    | 'quantiles' ':' quantile_specification  // For summaries
    | 'aggregation' ':' aggregation_specification
    | 'sampling' ':' sampling_specification
    | 'retention' ':' retention_specification
    | 'export' ':' export_specification
    | 'alerting' ':' alerting_specification
    ;

unit_specification
    : 'seconds' | 'milliseconds' | 'microseconds' | 'nanoseconds'
    | 'bytes' | 'kilobytes' | 'megabytes' | 'gigabytes'
    | 'requests' | 'errors' | 'connections' | 'operations'
    | 'percentage' | 'ratio' | 'count' | 'rate'
    | custom_unit
    ;

// ========================================
// DIGITAL TWIN AND INDUSTRIAL MONITORING EXTENSIONS
// ========================================

// ========================================
// CALCULATION AND REPORTING EXTENSIONS
// ========================================

// Statistical and analytical functions
// STATS_FUNCTIONS removed - causing 9 token overlap conflicts and not used in grammar rules

// Industrial protocols
// INDUSTRIAL_PROTOCOLS removed - causing 11 token overlap conflicts and not used in grammar rules

// Duplicate removed - SIMULATION_TYPES defined earlier

// Duplicate removed - MATH_FUNCTIONS defined earlier

// Additional statistical functions
// STAT_FUNCTIONS removed - causing 13 token overlap conflicts and not used in grammar rules

// ========================================
// MARKETPLACE AND ECOMMERCE EXTENSIONS
// ========================================

// Marketplace entity types
marketplace_entity
    : 'marketplace' marketplace_name '{' marketplace_config '}'
    | 'ecommerce' ecommerce_name '{' ecommerce_config '}'
    | 'platform' platform_name '{' platform_config '}'
    ;

marketplace_config
    : marketplace_component (';' marketplace_component)*
    ;

marketplace_component
    : user_types_definition
    | transaction_engine
    | trust_safety_system
    | search_discovery_engine
    | communication_system
    | inventory_management
    | order_fulfillment
    | analytics_intelligence
    | microservices_architecture
    | internationalization_config
    ;

// User types and multi-tenancy
user_types_definition
    : 'user_types' ':' '{' user_type_list '}'
    ;

user_type_list
    : user_type (',' user_type)*
    ;

user_type
    : user_type_name '{' user_type_config '}'
    ;

user_type_config
    : user_type_property (';' user_type_property)*
    ;

user_type_property
    : 'permissions' ':' '[' permission_list ']'
    | 'data_access' ':' data_access_config
    | 'verification_required' ':' verification_requirements
    | 'onboarding_flow' ':' onboarding_definition
    | 'dashboard' ':' dashboard_configuration
    | 'payment_methods' ':' '[' payment_method_list ']'
    | 'commission_structure' ':' commission_config
    ;

// Transaction and payment engine
transaction_engine
    : 'transactions' ':' '{' transaction_config '}'
    ;

transaction_config
    : transaction_property (';' transaction_property)*
    ;

transaction_property
    : 'escrow_enabled' ':' BOOLEAN
    | 'payment_providers' ':' '[' payment_provider_list ']'
    | 'supported_currencies' ':' '[' currency_list ']'
    | 'fee_structure' ':' fee_configuration
    | 'dispute_resolution' ':' dispute_config
    | 'refund_policies' ':' refund_configuration
    | 'multi_party_splits' ':' split_configuration
    | 'fraud_detection' ':' fraud_detection_config
    ;

// Trust and safety system
trust_safety_system
    : 'trust_safety' ':' '{' trust_safety_config '}'
    ;

trust_safety_config
    : trust_safety_property (';' trust_safety_property)*
    ;

trust_safety_property
    : 'identity_verification' ':' verification_config
    | 'rating_system' ':' rating_configuration
    | 'review_system' ':' review_configuration
    | 'content_moderation' ':' moderation_config
    | 'fraud_prevention' ':' fraud_prevention_config
    | 'compliance_frameworks' ':' '[' compliance_framework_list ']'
    | 'background_checks' ':' background_check_config
    ;

// Search and discovery engine
search_discovery_engine
    : 'search_discovery' ':' '{' search_config '}'
    ;

search_config
    : search_property (';' search_property)*
    ;

search_property
    : 'search_engine' ':' search_engine_type
    | 'indexing_strategy' ':' indexing_configuration
    | 'recommendation_engine' ':' recommendation_config
    | 'personalization' ':' personalization_config
    | 'geolocation' ':' geolocation_config
    | 'faceted_search' ':' facet_configuration
    | 'autocomplete' ':' autocomplete_config
    | 'search_analytics' ':' search_analytics_config
    ;

// Communication system
communication_system
    : 'communication' ':' '{' communication_config '}'
    ;

communication_config
    : communication_property (';' communication_property)*
    ;

communication_property
    : 'messaging' ':' messaging_configuration
    | 'notifications' ':' notification_configuration
    | 'real_time_chat' ':' chat_configuration
    | 'video_calls' ':' video_call_configuration
    | 'negotiation_system' ':' negotiation_config
    | 'announcement_system' ':' announcement_config
    ;

// Microservices architecture definition
microservices_architecture
    : 'microservices' ':' '{' microservices_config '}'
    ;

microservices_config
    : microservices_property (';' microservices_property)*
    ;

microservices_property
    : 'services' ':' '[' service_definition_list ']'
    | 'api_gateway' ':' api_gateway_config
    | 'service_mesh' ':' service_mesh_config
    | 'service_discovery' ':' service_discovery_config
    | 'load_balancing' ':' load_balancing_config
    | 'circuit_breakers' ':' circuit_breaker_config
    | 'distributed_tracing' ':' tracing_config
    ;

// Service definition
service_definition
    : 'service' service_name '{' service_config '}'
    ;

service_config
    : service_property (';' service_property)*
    ;

service_property
    : 'type' ':' service_type
    | 'responsibilities' ':' '[' responsibility_list ']'
    | 'api_endpoints' ':' '[' endpoint_definition_list ']'
    | 'database' ':' database_reference
    | 'dependencies' ':' '[' service_dependency_list ']'
    | 'scaling' ':' scaling_configuration
    | 'deployment' ':' deployment_configuration
    | 'monitoring' ':' monitoring_configuration
    | 'security' ':' security_configuration
    ;

// Service placement and deployment
deployment_configuration
    : 'deployment' ':' '{' deployment_config '}'
    ;

deployment_config
    : deployment_property (';' deployment_property)*
    ;

deployment_property
    : 'environment' ':' deployment_environment
    | 'container' ':' container_configuration
    | 'orchestration' ':' orchestration_config
    | 'placement_strategy' ':' placement_strategy_config
    | 'resource_requirements' ':' resource_requirements_config
    | 'networking' ':' networking_configuration
    | 'storage' ':' storage_configuration
    | 'secrets_management' ':' secrets_config
    ;

// Internationalization and localization
internationalization_config
    : 'i18n' ':' '{' i18n_config '}'
    ;

i18n_config
    : i18n_property (';' i18n_property)*
    ;

i18n_property
    : 'supported_languages' ':' '[' language_list ']'
    | 'supported_currencies' ':' '[' currency_list ']'
    | 'supported_regions' ':' '[' region_list ']'
    | 'localization_strategy' ':' localization_config
    | 'currency_conversion' ':' currency_conversion_config
    | 'tax_calculation' ':' tax_calculation_config
    | 'shipping_zones' ':' shipping_zones_config
    ;

// Advanced analytics and business intelligence
analytics_intelligence
    : 'analytics' ':' '{' analytics_config '}'
    ;

analytics_config
    : analytics_property (';' analytics_property)*
    ;

analytics_property
    : 'data_warehouse' ':' data_warehouse_config
    | 'real_time_analytics' ':' real_time_analytics_config
    | 'business_intelligence' ':' bi_configuration
    | 'machine_learning' ':' ml_analytics_config
    | 'ab_testing' ':' ab_testing_config
    | 'conversion_tracking' ':' conversion_tracking_config
    | 'cohort_analysis' ':' cohort_analysis_config
    | 'predictive_analytics' ':' predictive_analytics_config
    ;

// Event-driven marketplace architecture
marketplace_events
    : 'events' ':' '{' event_definitions '}'
    ;

event_definitions
    : event_definition (';' event_definition)*
    ;

event_definition
    : event_name ':' '{' event_config '}'
    ;

event_config
    : event_property (';' event_property)*
    ;

event_property
    : 'schema' ':' event_schema
    | 'triggers' ':' '[' trigger_list ']'
    | 'handlers' ':' '[' handler_list ']'
    | 'routing' ':' event_routing_config
    | 'persistence' ':' event_persistence_config
    | 'retry_policy' ':' retry_policy_config
    ;

// MARKETPLACE_KEYWORDS removed - causing 35 token overlap conflicts and not used in grammar rules

// TRUST_SAFETY_KEYWORDS removed - causing 17 token overlap conflicts and not used in grammar rules

// SEARCH_KEYWORDS removed - causing 18 token overlap conflicts and not used in grammar rules

// Duplicate removed - COMMUNICATION_KEYWORDS defined earlier

// ANALYTICS_KEYWORDS removed - causing 24 token overlap conflicts and not used in grammar rules
