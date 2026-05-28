grammar apg;

options {
	language = Python3;
}

// ========================================
// MAIN PROGRAM STRUCTURE
// ========================================

apg
	: import_statement* global_declaration* EOF
	;

global_declaration
	: appconfig
	| i18n_config
	| template_definition
	| version_tag
	| conditional_block
	| plugin_definition
	| event_definition
	| event_handler
	| custom_component
	| table_definition
	| enum_definition
	| relationship
	| index_definition
	| role_definition
	| trigger_definition
	| script_definition
	| workflow_definition
	| form_definition
	| masterdetail_form
	| wizard_definition
	| form_storage
	| chart_definition
	| ai_config
	| function_definition
	| class_definition
	| report_definition
	| notification_definition
	| scheduled_task
	| background_job
	| api_config
	| webhook_definition
	| exception_definition
	| test_definition
	| migration_definition
	| variable_declaration
	// NEW AGENT-ORIENTED DECLARATIONS
	| llm_provider_definition
	| llm_model_definition
	| agent_definition
	| agent_factory_definition
	| orchestrator_definition
	| sequence_definition
	| parallel_definition
	| reactive_definition
	| swarm_definition
	| message_protocol_definition
	| camera_input_definition
	| audio_input_definition
	| iot_hub_definition
	| sensor_definition
	| deployment_target_definition
	| visualization_dashboard_definition
	| monitoring_system_definition
	| reactive_system_definition
	| event_sourcing_definition
	| stream_definition
	;

// ========================================
// IMPORT STATEMENTS & MODULES
// ========================================

import_statement
	: 'import' module_path ('as' IDENTIFIER)? ';'
	| 'from' module_path 'import' import_list ';'
	;

module_path
	: IDENTIFIER ('.' IDENTIFIER)*
	;

import_list
	: import_item (',' import_item)*
	| '*'
	;

import_item
	: IDENTIFIER ('as' IDENTIFIER)?
	;

// ========================================
// TYPE SYSTEM (Python-like)
// ========================================

type_annotation
	: union_type
	| generic_type
	| optional_type
	| basic_type
	| custom_type
	;

union_type
	: type_annotation ('|' type_annotation)+
	;

generic_type
	: IDENTIFIER '[' type_annotation (',' type_annotation)* ']'
	;

optional_type
	: type_annotation '?'
	;

basic_type
	: 'str' | 'int' | 'float' | 'bool' | 'bytes' | 'datetime' | 'decimal'
	| 'list' | 'dict' | 'set' | 'tuple' | 'frozenset'
	| 'Any' | 'None' | 'object'
	// Database specific types
	| 'smallint' | 'integer' | 'bigint' | 'serial' | 'bigserial'
	| 'decimal' '(' INT ',' INT ')'
	| 'numeric' '(' INT ',' INT ')'
	| 'real' | 'double' 'precision' | 'money'
	| 'char' '(' INT ')' | 'varchar' '(' INT ')' | 'text'
	| 'bytea' | 'json' | 'jsonb' | 'xml'
	| 'timestamp' ('with' 'time' 'zone' | 'without' 'time' 'zone')?
	| 'date' | 'time' ('with' 'time' 'zone' | 'without' 'time' 'zone')?
	| 'interval' | 'boolean' | 'uuid'
	| 'cidr' | 'inet' | 'macaddr'
	| 'bit' '(' INT ')' | 'bit' 'varying' '(' INT ')'
	| 'tsvector' | 'tsquery'
	| 'int4range' | 'int8range' | 'numrange' | 'tsrange' | 'tstzrange' | 'daterange'
	| encrypted_type | vector_type | graph_type | document_type
	;

custom_type
	: IDENTIFIER
	;

encrypted_type
	: 'encrypted' '(' STRING_LITERAL ')'
	;

vector_type
	: 'vector' '(' INT ')'
	;

graph_type
	: 'graph'
	;

document_type
	: 'document'
	;

// ========================================
// EXPRESSIONS (Enhanced Python-like)
// ========================================

expression
	: lambda_expr
	| conditional_expr
	;

lambda_expr
	: 'lambda' parameter_list? ':' expression
	;

conditional_expr
	: or_test ('if' or_test 'else' expression)?
	;

or_test
	: and_test ('or' and_test)*
	;

and_test
	: not_test ('and' not_test)*
	;

not_test
	: 'not' not_test
	| comparison
	;

comparison
	: bitwise_or (comp_op bitwise_or)*
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
	| '`' testlist1 '`'
	| IDENTIFIER | NUMBER | string_literal+
	| '...' | 'None' | 'True' | 'False'
	;

trailer
	: '(' arglist? ')'
	| '[' subscriptlist ']'
	| '.' IDENTIFIER
	;

subscriptlist
	: subscript (',' subscript)* ','?
	;

subscript
	: test
	| test? ':' test? sliceop?
	;

sliceop
	: ':' test?
	;

// String literals with f-string support
string_literal
	: STRING_LITERAL
	| F_STRING
	| RAW_STRING
	;

// Comprehensions
listmaker
	: (test | star_expr) (list_for | (',' (test | star_expr))* ','?)
	;

dictorsetmaker
	: ((test ':' test | '**' expr) (comp_for | (',' (test ':' test | '**' expr))* ','?) |
	   (test | star_expr) (comp_for | (',' (test | star_expr))* ','?))
	;

list_for
	: 'for' exprlist 'in' testlist ('if' test)*
	;

comp_for
	: 'for' exprlist 'in' or_test ('if' test)*
	;

testlist_comp
	: (test | star_expr) (comp_for | (',' (test | star_expr))* ','?)
	;

yield_expr
	: 'yield' (yield_arg)?
	;

yield_arg
	: 'from' test | testlist
	;

// ========================================
// FUNCTION & CLASS DEFINITIONS
// ========================================

function_definition
	: decorator* function_def
	;

function_def
	: 'def' IDENTIFIER '(' parameter_list? ')' ('->' type_annotation)? '{' suite '}'
	| 'async' 'def' IDENTIFIER '(' parameter_list? ')' ('->' type_annotation)? '{' suite '}'
	;

parameter_list
	: parameter (',' parameter)* (',' variadic_parameter)?
	| variadic_parameter
	;

parameter
	: IDENTIFIER (':' type_annotation)? ('=' expression)?
	;

variadic_parameter
	: '*' IDENTIFIER (':' type_annotation)?
	| '**' IDENTIFIER (':' type_annotation)?
	;

class_definition
	: decorator* 'class' IDENTIFIER ('(' inheritance_list ')')? '{' class_body '}'
	;

inheritance_list
	: type_annotation (',' type_annotation)*
	;

class_body
	: class_member*
	;

class_member
	: function_definition
	| variable_declaration
	| property_definition
	| class_definition
	;

property_definition
	: '@property' function_def
	| '@' IDENTIFIER '.setter' function_def
	| '@' IDENTIFIER '.deleter' function_def
	;

decorator
	: '@' dotted_name ('(' arglist? ')')?
	;

dotted_name
	: IDENTIFIER ('.' IDENTIFIER)*
	;

// ========================================
// CONTROL FLOW STATEMENTS
// ========================================

statement
	: simple_stmt
	| compound_stmt
	;

simple_stmt
	: small_stmt (';' small_stmt)* ';'?
	;

small_stmt
	: expr_stmt
	| del_stmt
	| pass_stmt
	| flow_stmt
	| import_stmt
	| global_stmt
	| nonlocal_stmt
	| assert_stmt
	;

expr_stmt
	: testlist_star_expr (annassign | augassign (yield_expr | testlist) |
						('=' (yield_expr | testlist_star_expr))*)
	;

annassign
	: ':' type_annotation ('=' (yield_expr | testlist_star_expr))?
	;

augassign
	: ('+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | '|=' | '^=' |
	   '<<=' | '>>=' | '**=' | '//=')
	;

del_stmt
	: 'del' exprlist
	;

pass_stmt
	: 'pass'
	;

flow_stmt
	: break_stmt | continue_stmt | return_stmt | raise_stmt | yield_stmt
	;

break_stmt
	: 'break'
	;

continue_stmt
	: 'continue'
	;

return_stmt
	: 'return' testlist?
	;

raise_stmt
	: 'raise' (test ('from' test)?)?
	;

yield_stmt
	: yield_expr
	;

global_stmt
	: 'global' IDENTIFIER (',' IDENTIFIER)*
	;

nonlocal_stmt
	: 'nonlocal' IDENTIFIER (',' IDENTIFIER)*
	;

assert_stmt
	: 'assert' test (',' test)?
	;

compound_stmt
	: if_stmt | while_stmt | for_stmt | try_stmt | with_stmt | funcdef | classdef
	| decorated | async_stmt | match_stmt
	;

if_stmt
	: 'if' test '{' suite '}' ('elif' test '{' suite '}')* ('else' '{' suite '}')?
	;

while_stmt
	: 'while' test '{' suite '}' ('else' '{' suite '}')?
	;

for_stmt
	: 'for' exprlist 'in' testlist '{' suite '}' ('else' '{' suite '}')?
	;

try_stmt
	: ('try' '{' suite '}' ((except_clause '{' suite '}')+ ('else' '{' suite '}')? ('finally' '{' suite '}')? |
	   'finally' '{' suite '}'))
	;

with_stmt
	: 'with' with_item (',' with_item)* '{' suite '}'
	;

with_item
	: test ('as' expr)?
	;

except_clause
	: 'except' (test ('as' IDENTIFIER)?)?
	;

// Async statements
async_stmt
	: 'async' (funcdef | with_stmt | for_stmt)
	;

await_expr
	: 'await' power
	;

// Pattern matching (Python 3.10+)
match_stmt
	: 'match' subject_expr '{' case_block+ '}'
	;

subject_expr
	: star_named_expression (',' star_named_expressions)?
	;

case_block
	: 'case' patterns guard? '{' suite '}'
	;

guard
	: 'if' named_expression
	;

patterns
	: open_sequence_pattern
	| pattern
	;

pattern
	: as_pattern
	| or_pattern
	;

as_pattern
	: or_pattern 'as' pattern_capture_target
	;

or_pattern
	: closed_pattern ('|' closed_pattern)*
	;

closed_pattern
	: literal_pattern
	| capture_pattern
	| wildcard_pattern
	| value_pattern
	| group_pattern
	| sequence_pattern
	| mapping_pattern
	| class_pattern
	;

// ========================================
// TABLE DEFINITIONS (Enhanced)
// ========================================

table_definition
	: decorator* 'table' IDENTIFIER ('(' inheritance_list ')')? version_tag? '{' table_body '}'
	;

table_body
	: table_member*
	;

table_member
	: column_definition
	| method_definition
	| constraint_definition
	| index_definition
	| trigger_definition
	;

column_definition
	: IDENTIFIER ':' type_annotation column_attributes? ('=' default_value)? ';'
	;

column_attributes
	: '[' column_attribute (',' column_attribute)* ']'
	;

column_attribute
	: 'pk' | 'primary_key'
	| 'unique'
	| 'not_null' | 'required'
	| 'auto_increment'
	| 'index'
	| 'ref' ':' relationship_spec
	| validation_attribute
	| 'label' ':' STRING_LITERAL
	| 'help' ':' STRING_LITERAL
	| 'choices' ':' '[' choice_list ']'
	;

relationship_spec
	: relationship_type IDENTIFIER '.' IDENTIFIER
	;

relationship_type
	: '>' | '-' | '<' | '<>'  // one-to-many, one-to-one, many-to-one, many-to-many
	;

validation_attribute
	: 'min' ':' expression
	| 'max' ':' expression
	| 'pattern' ':' STRING_LITERAL
	| 'regex' ':' STRING_LITERAL
	| 'custom' ':' IDENTIFIER
	;

choice_list
	: choice (',' choice)*
	;

choice
	: '(' expression ',' STRING_LITERAL ')'
	;

constraint_definition
	: 'constraint' IDENTIFIER constraint_type ';'
	;

constraint_type
	: 'check' '(' expression ')'
	| 'unique' '(' identifier_list ')'
	| 'foreign_key' '(' identifier_list ')' 'references' IDENTIFIER '(' identifier_list ')'
	;

// ========================================
// FORM DEFINITIONS (Enhanced)
// ========================================

form_definition
	: decorator* 'form' IDENTIFIER ('extends' IDENTIFIER)? 'for' IDENTIFIER? version_tag? '{' form_body '}'
	;

form_body
	: form_member*
	;

form_member
	: form_field_definition
	| method_definition
	| form_layout
	| form_button_definition
	| form_navigator
	| form_meta_class
	;

form_field_definition
	: IDENTIFIER ':' field_type field_attributes? ('=' default_value)? ';'
	;

field_type
	: type_annotation
	| widget_type
	;

widget_type
	: 'CharField' | 'TextField' | 'EmailField' | 'URLField' | 'IntegerField'
	| 'FloatField' | 'BooleanField' | 'DateField' | 'TimeField' | 'DateTimeField'
	| 'FileField' | 'ImageField' | 'ChoiceField' | 'MultipleChoiceField'
	| 'HiddenField' | 'PasswordField'
	| custom_widget
	;

custom_widget
	: IDENTIFIER '(' widget_params? ')'
	;

widget_params
	: widget_param (',' widget_param)*
	;

widget_param
	: IDENTIFIER '=' expression
	;

field_attributes
	: '[' field_attribute (',' field_attribute)* ']'
	;

field_attribute
	: 'required'
	| 'optional'
	| 'widget' ':' widget_type
	| 'label' ':' STRING_LITERAL
	| 'help_text' ':' STRING_LITERAL
	| 'initial' ':' expression
	| validation_attribute
	;

form_layout
	: 'layout' '{' layout_element+ '}'
	;

layout_element
	: 'row' '(' field_reference (',' field_reference)* ')'
	| 'fieldset' '(' STRING_LITERAL ',' field_reference (',' field_reference)* ')'
	| 'tab' '(' STRING_LITERAL ',' layout_element+ ')'
	;

field_reference
	: IDENTIFIER
	;

form_button_definition
	: 'button' IDENTIFIER '{' button_attributes '}'
	;

button_attributes
	: button_attribute+
	;

button_attribute
	: 'label' ':' STRING_LITERAL ';'
	| 'action' ':' button_action ';'
	| 'style' ':' button_style ';'
	| 'onclick' ':' expression ';'
	;

button_action
	: 'submit' | 'reset' | 'button' | 'custom'
	;

button_style
	: 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'info' | 'light' | 'dark'
	;

form_navigator
	: 'navigator' IDENTIFIER '{' navigator_button* '}'
	;

navigator_button
	: 'first' | 'previous' | 'next' | 'last' | 'insert' | 'delete' | 'edit' | 'save' | 'cancel'
	;

form_meta_class
	: 'class' 'Meta' '{' meta_attribute+ '}'
	;

meta_attribute
	: 'model' '=' IDENTIFIER ';'
	| 'fields' '=' '[' identifier_list ']' ';'
	| 'exclude' '=' '[' identifier_list ']' ';'
	| 'widgets' '=' '{' widget_mapping '}' ';'
	;

widget_mapping
	: widget_map_item (',' widget_map_item)*
	;

widget_map_item
	: IDENTIFIER ':' widget_type
	;

// ========================================
// WORKFLOW DEFINITIONS (Enhanced)
// ========================================

workflow_definition
	: decorator* 'workflow' IDENTIFIER '{' workflow_body '}'
	;

workflow_body
	: workflow_member*
	;

workflow_member
	: workflow_metadata
	| workflow_step
	| method_definition
	| workflow_handler
	;

workflow_metadata
	: 'initiator' ':' IDENTIFIER ';'
	| 'description' ':' STRING_LITERAL ';'
	| 'deadline' ':' expression ';'
	| 'schedule' ':' cron_expression ';'
	;

workflow_step
	: 'step' STRING_LITERAL '{' step_body '}'
	;

step_body
	: step_member*
	;

step_member
	: step_metadata
	| statement
	;

step_metadata
	: 'form' ':' IDENTIFIER ';'
	| 'assign_to' ':' assignment_target ';'
	| 'deadline' ':' expression ';'
	| 'condition' ':' expression ';'
	;

assignment_target
	: IDENTIFIER
	| 'role' '(' IDENTIFIER ')'
	;

workflow_handler
	: 'on_enter' '{' suite '}'
	| 'on_exit' '{' suite '}'
	| 'on_error' '{' suite '}'
	;

// ========================================
// API & INTEGRATION
// ========================================

api_config
	: 'api' IDENTIFIER '{' api_member+ '}'
	;

api_member
	: 'endpoint' ':' STRING_LITERAL ';'
	| 'method' ':' http_method ';'
	| 'headers' ':' '{' header_list '}' ';'
	| 'params' ':' '{' param_list '}' ';'
	| 'body' ':' expression ';'
	| 'auth' ':' auth_config ';'
	| 'rate_limit' ':' rate_limit_config ';'
	| 'cache' ':' cache_config ';'
	;

http_method
	: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS'
	;

header_list
	: header_item (',' header_item)*
	;

header_item
	: STRING_LITERAL ':' STRING_LITERAL
	;

param_list
	: param_item (',' param_item)*
	;

param_item
	: IDENTIFIER ':' expression
	;

auth_config
	: 'oauth' '{' oauth_details '}'
	| 'api_key' ':' STRING_LITERAL
	| 'bearer' ':' STRING_LITERAL
	;

oauth_details
	: oauth_detail+
	;

oauth_detail
	: 'client_id' ':' STRING_LITERAL ';'
	| 'client_secret' ':' STRING_LITERAL ';'
	| 'auth_url' ':' STRING_LITERAL ';'
	| 'token_url' ':' STRING_LITERAL ';'
	| 'scope' ':' STRING_LITERAL ';'
	;

rate_limit_config
	: '{' 'requests' ':' INT ',' 'per' ':' time_unit '}'
	;

time_unit
	: 'second' | 'minute' | 'hour' | 'day'
	;

cache_config
	: '{' 'duration' ':' INT time_unit '}'
	;

// ========================================
// ERROR HANDLING & TESTING
// ========================================

exception_definition
	: 'exception' IDENTIFIER ('(' IDENTIFIER ')')? '{' exception_body? '}'
	;

exception_body
	: method_definition*
	;

test_definition
	: 'test' IDENTIFIER '{' test_body '}'
	;

test_body
	: test_member*
	;

test_member
	: statement
	| assert_statement
	;

assert_statement
	: 'assert' expression (',' STRING_LITERAL)? ';'
	;

// ========================================
// ENHANCED FEATURES
// ========================================

// Variable declarations
variable_declaration
	: IDENTIFIER ':' type_annotation ('=' expression)? ';'
	;

// Migration definitions
migration_definition
	: 'migration' STRING_LITERAL '{' migration_operation+ '}'
	;

migration_operation
	: 'create_table' '(' IDENTIFIER ')' ';'
	| 'drop_table' '(' IDENTIFIER ')' ';'
	| 'add_column' '(' IDENTIFIER ',' IDENTIFIER ',' type_annotation ')' ';'
	| 'drop_column' '(' IDENTIFIER ',' IDENTIFIER ')' ';'
	| 'alter_column' '(' IDENTIFIER ',' IDENTIFIER ',' type_annotation ')' ';'
	| 'create_index' '(' IDENTIFIER ',' '[' identifier_list ']' ')' ';'
	| 'drop_index' '(' IDENTIFIER ')' ';'
	;

// Enhanced AI configuration
ai_config
	: 'ai_config' '{' ai_service+ '}'
	;

ai_service
	: 'chatbot' '{' chatbot_config+ '}'
	| 'content_generator' '{' content_gen_config+ '}'
	| 'data_analyzer' '{' data_analyzer_config+ '}'
	;

chatbot_config
	: 'model' ':' STRING_LITERAL ';'
	| 'api_key' ':' STRING_LITERAL ';'
	| 'endpoint' ':' STRING_LITERAL ';'
	| 'prompt' ':' prompt_definition ';'
	;

content_gen_config
	: 'model' ':' STRING_LITERAL ';'
	| 'api_key' ':' STRING_LITERAL ';'
	| 'endpoint' ':' STRING_LITERAL ';'
	| 'content_types' ':' '[' string_list ']' ';'
	;

data_analyzer_config
	: 'model' ':' STRING_LITERAL ';'
	| 'api_key' ':' STRING_LITERAL ';'
	| 'endpoint' ':' STRING_LITERAL ';'
	| 'data_types' ':' '[' string_list ']' ';'
	;

prompt_definition
	: '{' 'text' ':' STRING_LITERAL ';' '}'
	;

string_list
	: STRING_LITERAL (',' STRING_LITERAL)*
	;

// ========================================
// UTILITY RULES
// ========================================

suite
	: statement+
	;

testlist
	: test (',' test)* ','?
	;

test
	: or_test ('if' or_test 'else' test)?
	| lambdef
	;

testlist_star_expr
	: (test | star_expr) (',' (test | star_expr))* ','?
	;

star_expr
	: '*' expr
	;

exprlist
	: (expr | star_expr) (',' (expr | star_expr))* ','?
	;

expr
	: bitwise_or
	;

identifier_list
	: IDENTIFIER (',' IDENTIFIER)*
	;

arglist
	: argument (',' argument)* ','?
	;

argument
	: test ('=' test)?
	| '**' test
	| '*' test
	;

default_value
	: expression
	;

cron_expression
	: STRING_LITERAL
	;

version_tag
	: 'version' SEMVER
	;

note
	: 'note' ':' STRING_LITERAL
	;

// Legacy support for existing constructs
appconfig
	: 'appconfig' '{' app_type '}'
	;

app_type
	: 'flask_app_builder' '{' flask_config* '}'
	| 'kivy_desktop' '{' kivy_config* '}'
	| 'kivy_mobile' '{' kivy_config* '}'
	;

flask_config
	: 'base_url' ':' STRING_LITERAL ';'
	| 'database_uri' ':' STRING_LITERAL ';'
	| 'secret_key' ':' STRING_LITERAL ';'
	| 'debug' ':' BOOLEAN ';'
	;

kivy_config
	: 'window_size' ':' STRING_LITERAL ';'
	| 'orientation' ':' STRING_LITERAL ';'
	| 'icon' ':' STRING_LITERAL ';'
	;

i18n_config
	: 'i18n' '{' language_definition+ '}'
	;

language_definition
	: IDENTIFIER '{' translation+ '}'
	;

translation
	: STRING_LITERAL ':' STRING_LITERAL ';'
	;

template_definition
	: 'template' IDENTIFIER '{' template_content '}'
	;

template_content
	: global_declaration*
	;

conditional_block
	: '#if' condition_expression '{' global_declaration+ '}' ('#else' '{' global_declaration+ '}')? '#endif'
	;

condition_expression
	: expression
	;

plugin_definition
	: 'plugin' IDENTIFIER '{' plugin_config+ '}'
	;

plugin_config
	: 'version' ':' SEMVER ';'
	| 'source' ':' STRING_LITERAL ';'
	| 'config' ':' '{' plugin_param+ '}' ';'
	;

plugin_param
	: IDENTIFIER ':' expression ';'
	;

event_definition
	: 'event' IDENTIFIER '{' event_property* '}'
	;

event_property
	: IDENTIFIER ':' type_annotation ';'
	;

event_handler
	: 'on' IDENTIFIER '{' suite '}'
	;

custom_component
	: 'component' IDENTIFIER '{' component_property* '}'
	;

component_property
	: IDENTIFIER ':' expression ';'
	;

enum_definition
	: 'enum' IDENTIFIER '{' enum_value+ '}'
	;

enum_value
	: IDENTIFIER ('=' expression)? ';'
	;

relationship
	: 'ref' IDENTIFIER '.' IDENTIFIER relationship_type IDENTIFIER '.' IDENTIFIER ';'
	;

index_definition
	: 'index' IDENTIFIER 'on' IDENTIFIER '(' identifier_list ')' index_attributes? ';'
	;

index_attributes
	: '[' index_attribute (',' index_attribute)* ']'
	;

index_attribute
	: 'unique'
	| 'partial' '(' expression ')'
	;

role_definition
	: 'role' IDENTIFIER ('inherits' IDENTIFIER)? '{' permission+ '}'
	;

permission
	: 'permission' action 'on' IDENTIFIER ('where' expression)? ';'
	;

action
	: 'read' | 'write' | 'delete' | 'execute' | 'create' | 'update'
	;

trigger_definition
	: 'trigger' IDENTIFIER 'on' IDENTIFIER trigger_type '{' suite '}'
	;

trigger_type
	: 'before' | 'after' | 'instead_of'
	;

script_definition
	: 'script' IDENTIFIER script_lang '{' suite '}'
	;

script_lang
	: 'python' | 'bash' | 'zsh' | 'javascript' | 'sql'
	;

masterdetail_form
	: 'master_detail_form' IDENTIFIER 'master' IDENTIFIER 'details' '{' detail_component+ '}'
	;

detail_component
	: 'detail' IDENTIFIER ('exclude' '(' identifier_list ')')? ';'
	;

wizard_definition
	: 'wizard' IDENTIFIER ('store_in' '{' table_mapping (',' table_mapping)* '}')? '{' wizard_step+ '}'
	;

table_mapping
	: IDENTIFIER 'fields' '(' identifier_list ')'
	;

wizard_step
	: 'step' IDENTIFIER '{' wizard_statement+ '}'
	;

wizard_statement
	: statement
	| 'form' '(' form_field_list ')'
	;

form_field_list
	: form_field_ref (',' form_field_ref)*
	;

form_field_ref
	: IDENTIFIER ('as' widget_type)? ('hint' ':' STRING_LITERAL)? ('help' ':' STRING_LITERAL)?
	;

form_storage
	: 'form_storage' IDENTIFIER 'form' IDENTIFIER 'save_for_later' ';'
	;

chart_definition
	: 'chart' IDENTIFIER 'type' chart_type 'for' IDENTIFIER 'fields' '(' identifier_list ')' ('title' STRING_LITERAL)? ';'
	;

chart_type
	: 'bar' | 'line' | 'pie' | 'scatter' | 'area' | 'heatmap' | 'treemap'
	| 'gantt' | 'bubble' | 'candlestick' | 'radar' | 'polar_area' | 'funnel' | 'waterfall'
	;

report_definition
	: 'report' IDENTIFIER '{' report_config+ '}'
	;

report_config
	: 'template' ':' STRING_LITERAL ';'
	| 'datasource' ':' IDENTIFIER ';'
	| 'fields' ':' '(' identifier_list ')' ';'
	| 'filter' ':' expression ';'
	| 'sort' ':' STRING_LITERAL ';'
	| 'group' ':' STRING_LITERAL ';'
	| 'schedule' ':' schedule_config ';'
	;

schedule_config
	: '{' 'frequency' ':' cron_expression ',' 'time' ':' TIME_LITERAL '}'
	;

notification_definition
	: 'notification' IDENTIFIER '{' notification_config+ '}'
	;

notification_config
	: 'type' ':' notification_type ';'
	| 'trigger' ':' trigger_type ';'
	| 'recipient' ':' STRING_LITERAL ';'
	| 'message' ':' STRING_LITERAL ';'
	| 'template' ':' STRING_LITERAL ';'
	;

notification_type
	: 'email' | 'sms' | 'in_app' | 'push'
	;

scheduled_task
	: 'scheduled_task' IDENTIFIER '{' task_config+ '}'
	;

task_config
	: 'cron' ':' cron_expression ';'
	| 'action' ':' task_action ';'
	| 'condition' ':' expression ';'
	;

task_action
	: 'run_script' '(' IDENTIFIER ')'
	| 'send_notification' '(' IDENTIFIER ')'
	| 'update_record' '(' update_assignment (',' update_assignment)* ')'
	;

update_assignment
	: IDENTIFIER '.' IDENTIFIER '=' expression
	;

background_job
	: 'background_job' IDENTIFIER '{' job_config+ '}'
	;

job_config
	: 'interval' ':' STRING_LITERAL ';'
	| 'action' ':' job_action ';'
	| 'condition' ':' expression ';'
	;

job_action
	: 'run_script' '(' IDENTIFIER ')'
	| 'send_notification' '(' IDENTIFIER ')'
	| 'update_record' '(' update_assignment (',' update_assignment)* ')'
	;

webhook_definition
	: 'webhook' IDENTIFIER '{' webhook_config+ '}'
	;

webhook_config
	: 'url' ':' STRING_LITERAL ';'
	| 'method' ':' http_method ';'
	| 'headers' ':' '{' header_list '}' ';'
	| 'payload' ':' expression ';'
	| 'retry' ':' INT ';'
	;

method_definition
	: function_definition
	;

// ========================================
// LEXER RULES
// ========================================

// Identifiers and Names
IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]*;

// Numbers
NUMBER: INTEGER | FLOAT_NUMBER | IMAG_NUMBER;
INTEGER: DECIMAL_INTEGER | OCT_INTEGER | HEX_INTEGER | BIN_INTEGER;
DECIMAL_INTEGER: NON_ZERO_DIGIT DIGIT* | '0'+;
OCT_INTEGER: '0' ('o' | 'O') OCT_DIGIT+;
HEX_INTEGER: '0' ('x' | 'X') HEX_DIGIT+;
BIN_INTEGER: '0' ('b' | 'B') BIN_DIGIT+;
NON_ZERO_DIGIT: [1-9];
DIGIT: [0-9];
OCT_DIGIT: [0-7];
HEX_DIGIT: [0-9a-fA-F];
BIN_DIGIT: [01];

FLOAT_NUMBER: POINT_FLOAT | EXPONENT_FLOAT;
POINT_FLOAT: INT_PART? FRACTION | INT_PART '.';
EXPONENT_FLOAT: (INT_PART | POINT_FLOAT) EXPONENT;
INT_PART: DIGIT+;
FRACTION: '.' DIGIT+;
EXPONENT: ('e' | 'E') ('+' | '-')? DIGIT+;

IMAG_NUMBER: (FLOAT_NUMBER | INT_PART) ('j' | 'J');

// String Literals
STRING_LITERAL: 
	SHORT_STRING | LONG_STRING;

SHORT_STRING: 
	'\'' SHORT_STRING_ITEM_SINGLE* '\'' |
	'"' SHORT_STRING_ITEM_DOUBLE* '"';

LONG_STRING:
	'\'\'\'' LONG_STRING_ITEM* '\'\'\'' |
	'"""' LONG_STRING_ITEM* '"""';

SHORT_STRING_ITEM_SINGLE: SHORT_STRING_CHAR_SINGLE | STRING_ESCAPE_SEQ;
SHORT_STRING_ITEM_DOUBLE: SHORT_STRING_CHAR_DOUBLE | STRING_ESCAPE_SEQ;
LONG_STRING_ITEM: LONG_STRING_CHAR | STRING_ESCAPE_SEQ;

SHORT_STRING_CHAR_SINGLE: ~['\\] ;
SHORT_STRING_CHAR_DOUBLE: ~["\\] ;
LONG_STRING_CHAR: ~'\\';

STRING_ESCAPE_SEQ: '\\' .;

// F-Strings (simplified)
F_STRING: 
	'f' '\'' F_STRING_ITEM* '\'' |
	'f' '"' F_STRING_ITEM* '"' |
	'F' '\'' F_STRING_ITEM* '\'' |
	'F' '"' F_STRING_ITEM* '"';

F_STRING_ITEM: ~['{\\] | STRING_ESCAPE_SEQ | '{' .*? '}';

// Raw Strings
RAW_STRING:
	'r' STRING_LITERAL |
	'R' STRING_LITERAL;

// Boolean
BOOLEAN: 'True' | 'False';

// Special Literals
NONE: 'None';

// Operators
WALRUS_OP: ':=';
POWER_OP: '**';
FLOOR_DIV: '//';
MATRIX_MULT: '@';

// Time and Date
TIME_LITERAL: DIGIT DIGIT ':' DIGIT DIGIT (':' DIGIT DIGIT)?;
DATE_LITERAL: DIGIT DIGIT DIGIT DIGIT '-' DIGIT DIGIT '-' DIGIT DIGIT;

// Semantic Versioning
SEMVER: DIGIT+ '.' DIGIT+ '.' DIGIT+ ('-' [a-zA-Z0-9-]+)?;

// Comments and Whitespace
COMMENT: '#' ~[\r\n]* -> skip;
LINE_COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;

NEWLINE: ('\r'? '\n' | '\r' | '\f');
WS: [ \t]+ -> skip;
EXPLICIT_LINE_JOINING: '\\' '\r'? '\n';

// Skip these
SKIP_: (SPACES | COMMENT | LINE_COMMENT | BLOCK_COMMENT | EXPLICIT_LINE_JOINING) -> skip;
SPACES: [ \t]+;

// Patterns and Regular Expressions  
PATTERN: '/' (~[/\\\r\n] | '\\' .)* '/';

// URLs and Emails
URL: ('http' | 'https') '://' (~[ \t\r\n])+;
EMAIL: [a-zA-Z0-9_.+-]+ '@' [a-zA-Z0-9-]+ '.' [a-zA-Z0-9-.]+;

// File extensions
FILENAME: [a-zA-Z0-9_]+ '.' [a-zA-Z0-9_]+;

// Integer for legacy compatibility  
INT: DIGIT+;

// ========================================
// AGENT-ORIENTED PROGRAMMING EXTENSIONS
// ========================================

// LLM Provider & Model Definitions
llm_provider_definition
	: 'llm_provider' IDENTIFIER '{' llm_provider_config+ '}'
	;

llm_provider_config
	: 'api_key' ':' expression ';'
	| 'base_url' ':' STRING_LITERAL ';'
	| 'models' ':' '[' string_list ']' ';'
	| 'rate_limits' ':' '{' rate_limit_item+ '}' ';'
	;

rate_limit_item
	: STRING_LITERAL ':' expression
	;

llm_model_definition
	: 'llm_model' IDENTIFIER '{' llm_model_config+ '}'
	;

llm_model_config
	: 'provider' ':' IDENTIFIER ';'
	| 'model_name' ':' STRING_LITERAL ';'
	| 'temperature' ':' expression ';'
	| 'max_tokens' ':' expression ';'
	| 'system_prompt' ':' STRING_LITERAL ';'
	| 'functions' ':' '[' function_list ']' ';'
	| 'context_window' ':' expression ';'
	| 'memory_type' ':' STRING_LITERAL ';'
	;

function_list
	: IDENTIFIER (',' IDENTIFIER)*
	;

// Agent Definitions
agent_definition
	: decorator* 'agent' IDENTIFIER ('extends' IDENTIFIER)? '{' agent_body '}'
	;

agent_body
	: agent_member*
	;

agent_member
	: agent_config
	| method_definition
	| lifecycle_hook
	;

agent_config
	: 'llm' ':' IDENTIFIER ';'
	| 'name' ':' STRING_LITERAL ';'
	| 'description' ':' STRING_LITERAL ';'
	| 'capabilities' ':' '[' string_list ']' ';'
	| 'tools' ':' '[' tool_list ']' ';'
	| 'memory' ':' memory_config ';'
	| 'state' ':' IDENTIFIER ';'
	| 'autonomy_level' ':' STRING_LITERAL ';'
	| 'decision_threshold' ':' expression ';'
	| 'escalation_policy' ':' escalation_config ';'
	;

tool_list
	: tool_reference (',' tool_reference)*
	;

tool_reference
	: IDENTIFIER '(' arglist? ')'
	;

memory_config
	: '{' memory_setting+ '}'
	;

memory_setting
	: STRING_LITERAL ':' expression
	;

escalation_config
	: '{' escalation_setting+ '}'
	;

escalation_setting
	: STRING_LITERAL ':' expression
	;

lifecycle_hook
	: 'on_initialize' '{' suite '}'
	| 'on_activate' '{' suite '}'
	| 'on_deactivate' '{' suite '}'
	| 'on_error' '(' parameter ')' '{' suite '}'
	;

// Agent Factory
agent_factory_definition
	: 'agent_factory' IDENTIFIER '{' agent_factory_body '}'
	;

agent_factory_body
	: agent_factory_member*
	;

agent_factory_member
	: 'template' ':' IDENTIFIER ';'
	| method_definition
	;

// Orchestration Constructs
orchestrator_definition
	: 'orchestrator' IDENTIFIER '{' orchestrator_body '}'
	;

orchestrator_body
	: orchestrator_member*
	;

orchestrator_member
	: orchestrator_config
	| sequence_definition
	| parallel_definition
	| reactive_definition
	| swarm_definition
	| method_definition
	;

orchestrator_config
	: 'agents' ':' '{' agent_mapping+ '}' ';'
	| 'workflows' ':' '{' workflow_mapping+ '}' ';'
	| 'message_bus' ':' IDENTIFIER '(' arglist? ')' ';'
	;

agent_mapping
	: STRING_LITERAL ':' IDENTIFIER
	;

workflow_mapping
	: STRING_LITERAL ':' IDENTIFIER
	;

// Sequence Definition
sequence_definition
	: 'sequence' IDENTIFIER '{' sequence_step+ '}'
	;

sequence_step
	: 'step' STRING_LITERAL '{' step_config+ '}'
	;

step_config
	: 'agent' ':' IDENTIFIER ';'
	| 'input' ':' input_spec ';'
	| 'output' ':' type_annotation ';'
	| 'timeout' ':' STRING_LITERAL ';'
	| 'retry_policy' ':' retry_policy_spec ';'
	| 'depends_on' ':' '[' string_list ']' ';'
	| 'parallel_instances' ':' expression ';'
	| 'merge_strategy' ':' STRING_LITERAL ';'
	;

input_spec
	: type_annotation
	| type_annotation 'from' STRING_LITERAL
	;

retry_policy_spec
	: IDENTIFIER '(' retry_params? ')'
	;

retry_params
	: retry_param (',' retry_param)*
	;

retry_param
	: IDENTIFIER '=' expression
	;

// Parallel Definition  
parallel_definition
	: 'parallel' IDENTIFIER '{' parallel_body '}'
	;

parallel_body
	: parallel_member*
	;

parallel_member
	: 'coordinator' ':' IDENTIFIER ';'
	| parallel_branch
	| method_definition
	;

parallel_branch
	: 'branch' STRING_LITERAL '{' branch_config+ '}'
	;

branch_config
	: 'agent' ':' IDENTIFIER ';'
	| 'input' ':' input_spec ';'
	| 'output' ':' output_spec ';'
	;

output_spec
	: type_annotation
	| type_annotation 'stream'
	;

// Reactive Definition
reactive_definition
	: 'reactive' IDENTIFIER '{' reactive_body '}'
	;

reactive_body
	: reactive_member*
	;

reactive_member
	: 'trigger' ':' IDENTIFIER ';'
	| reactive_when
	;

reactive_when
	: 'when' IDENTIFIER '{' reactive_action+ '}'
	;

reactive_action
	: 'agent' ':' IDENTIFIER ';'
	| 'action' ':' IDENTIFIER ';'
	| 'response_time' ':' STRING_LITERAL ';'
	| 'escalate_if' ':' expression ';'
	| 'agents' ':' '[' agent_list ']' ';'
	| 'coordination' ':' STRING_LITERAL ';'
	;

agent_list
	: IDENTIFIER (',' IDENTIFIER)*
	;

// Swarm Definition
swarm_definition
	: 'swarm' IDENTIFIER '{' swarm_body '}'
	;

swarm_body
	: swarm_member*
	;

swarm_member
	: swarm_config
	| method_definition
	;

swarm_config
	: 'population_size' ':' expression ';'
	| 'generations' ':' expression ';'
	| 'diversity_threshold' ':' expression ';'
	| 'agents' ':' '[' agent_type_list ']' ';'
	;

agent_type_list
	: type_annotation (',' type_annotation)*
	;

// Message Protocol
message_protocol_definition
	: 'message_protocol' IDENTIFIER '{' protocol_body '}'
	;

protocol_body
	: protocol_member*
	;

protocol_member
	: protocol_config
	| message_type_definition
	;

protocol_config
	: 'transport' ':' STRING_LITERAL ';'
	| 'serialization' ':' STRING_LITERAL ';'
	| 'encryption' ':' BOOLEAN ';'
	;

message_type_definition
	: 'message' IDENTIFIER '{' message_field+ '}'
	;

message_field
	: IDENTIFIER ':' type_annotation ';'
	;

// Camera Input
camera_input_definition
	: 'camera_input' IDENTIFIER ('extends' IDENTIFIER)? '{' camera_body '}'
	;

camera_body
	: camera_member*
	;

camera_member
	: camera_config
	| vision_pipeline
	| camera_event_handler
	| method_definition
	;

camera_config
	: 'device_id' ':' STRING_LITERAL ';'
	| 'resolution' ':' tuple_literal ';'
	| 'fps' ':' expression ';'
	| 'format' ':' STRING_LITERAL ';'
	| 'vision_models' ':' '[' model_list ']' ';'
	;

tuple_literal
	: '(' expression (',' expression)* ')'
	;

model_list
	: model_reference (',' model_reference)*
	;

model_reference
	: IDENTIFIER '(' STRING_LITERAL ')'
	;

vision_pipeline
	: 'pipeline' IDENTIFIER '{' pipeline_stage+ '}'
	;

pipeline_stage
	: 'stage' STRING_LITERAL '{' stage_config+ '}'
	;

stage_config
	: 'operations' ':' '[' string_list ']' ';'
	| 'model' ':' IDENTIFIER ';'
	| 'confidence_threshold' ':' expression ';'
	| 'nms_threshold' ':' expression ';'
	| 'output' ':' type_annotation ';'
	| 'prompt_templates' ':' '[' string_list ']' ';'
	| 'agent' ':' IDENTIFIER ';'
	| 'context' ':' STRING_LITERAL ';'
	;

camera_event_handler
	: 'on' IDENTIFIER '(' parameter ')' '{' suite '}'
	;

// Audio Input
audio_input_definition
	: 'audio_input' IDENTIFIER ('extends' IDENTIFIER)? '{' audio_body '}'
	;

audio_body
	: audio_member*
	;

audio_member
	: audio_config
	| audio_pipeline
	| audio_event_handler
	| background_task_definition
	| method_definition
	;

audio_config
	: 'device_name' ':' STRING_LITERAL ';'
	| 'sample_rate' ':' expression ';'
	| 'channels' ':' expression ';'
	| 'bit_depth' ':' expression ';'
	| 'audio_models' ':' '[' model_list ']' ';'
	;

audio_pipeline
	: 'pipeline' IDENTIFIER '{' pipeline_stage+ '}'
	;

audio_event_handler
	: 'on' IDENTIFIER '(' parameter ')' '{' suite '}'
	;

background_task_definition
	: 'background_task' IDENTIFIER '{' task_body '}'
	;

task_body
	: task_member*
	;

task_member
	: 'interval' ':' STRING_LITERAL ';'
	| method_definition
	;

// IoT Hub and Sensors
iot_hub_definition
	: 'iot_hub' IDENTIFIER '{' iot_hub_body '}'
	;

iot_hub_body
	: iot_hub_member*
	;

iot_hub_member
	: iot_hub_config
	| sensor_definition
	| fusion_engine_definition
	| method_definition
	;

iot_hub_config
	: 'protocol' ':' STRING_LITERAL ';'
	| 'broker_url' ':' STRING_LITERAL ';'
	| 'device_registry' ':' IDENTIFIER '(' arglist? ')' ';'
	;

sensor_definition
	: 'sensor' IDENTIFIER '{' sensor_body '}'
	;

sensor_body
	: sensor_member*
	;

sensor_member
	: sensor_config
	| sensor_event_handler
	| method_definition
	;

sensor_config
	: 'device_type' ':' STRING_LITERAL ';'
	| 'location' ':' STRING_LITERAL ';'
	| 'update_interval' ':' STRING_LITERAL ';'
	| 'sensitivity' ':' expression ';'
	| 'access_codes' ':' '[' string_list ']' ';'
	;

sensor_event_handler
	: 'on' IDENTIFIER '(' parameter_list? ')' '{' suite '}'
	;

fusion_engine_definition
	: 'fusion_engine' IDENTIFIER '{' fusion_body '}'
	;

fusion_body
	: fusion_member*
	;

fusion_member
	: fusion_config
	| method_definition
	;

fusion_config
	: 'sensors' ':' '[' sensor_list ']' ';'
	| 'fusion_algorithms' ':' '[' algorithm_list ']' ';'
	;

sensor_list
	: type_annotation (',' type_annotation)*
	;

algorithm_list
	: algorithm_reference (',' algorithm_reference)*
	;

algorithm_reference
	: IDENTIFIER '(' arglist? ')'
	;

// Deployment Targets
deployment_target_definition
	: 'deployment_target' IDENTIFIER '{' deployment_body '}'
	;

deployment_body
	: deployment_member*
	;

deployment_member
	: deployment_config
	| cluster_definition
	| deployment_manifest
	| service_mesh_definition
	;

deployment_config
	: 'provider' ':' STRING_LITERAL ';'
	;

cluster_definition
	: 'cluster' IDENTIFIER '{' cluster_body '}'
	;

cluster_body
	: cluster_member*
	;

cluster_member
	: cluster_config
	;

cluster_config
	: 'name' ':' STRING_LITERAL ';'
	| 'nodes' ':' expression ';'
	| 'node_type' ':' STRING_LITERAL ';'
	| 'gpu_nodes' ':' expression ';'
	| 'gpu_type' ':' STRING_LITERAL ';'
	| 'autoscaling' ':' autoscaling_config ';'
	| 'resource_profiles' ':' '{' resource_profile_mapping+ '}' ';'
	;

autoscaling_config
	: '{' autoscaling_setting+ '}'
	;

autoscaling_setting
	: STRING_LITERAL ':' expression
	;

resource_profile_mapping
	: STRING_LITERAL ':' resource_profile_spec
	;

resource_profile_spec
	: IDENTIFIER '(' resource_params ')'
	;

resource_params
	: resource_param (',' resource_param)*
	;

resource_param
	: IDENTIFIER '=' STRING_LITERAL
	;

deployment_manifest
	: 'deployment' IDENTIFIER '{' manifest_body '}'
	;

manifest_body
	: manifest_member*
	;

manifest_member
	: manifest_config
	| method_definition
	;

manifest_config
	: 'agent_type' ':' STRING_LITERAL ';'
	| 'replicas' ':' expression ';'
	| 'strategy' ':' STRING_LITERAL ';'
	;

service_mesh_definition
	: 'service_mesh' IDENTIFIER '{' mesh_body '}'
	;

mesh_body
	: mesh_member*
	;

mesh_member
	: mesh_config
	;

mesh_config
	: 'traffic_management' ':' traffic_policy ';'
	| 'security' ':' security_policy ';'
	| 'observability' ':' observability_config ';'
	;

traffic_policy
	: '{' traffic_setting+ '}'
	;

traffic_setting
	: STRING_LITERAL ':' expression
	;

security_policy
	: '{' security_setting+ '}'
	;

security_setting
	: STRING_LITERAL ':' expression
	;

observability_config
	: '{' observability_setting+ '}'
	;

observability_setting
	: STRING_LITERAL ':' STRING_LITERAL
	;

// Visualization Dashboard
visualization_dashboard_definition
	: 'visualization_dashboard' IDENTIFIER '{' dashboard_body '}'
	;

dashboard_body
	: dashboard_member*
	;

dashboard_member
	: dashboard_config
	| scene_definition
	| control_panel_definition
	| ar_interface_definition
	;

dashboard_config
	: 'framework' ':' STRING_LITERAL ';'
	| 'backend' ':' STRING_LITERAL ';'
	| 'realtime_transport' ':' STRING_LITERAL ';'
	;

scene_definition
	: 'scene' IDENTIFIER '{' scene_body '}'
	;

scene_body
	: scene_member*
	;

scene_member
	: scene_config
	| mesh_definition
	| connection_lines_definition
	| environment_objects_definition
	;

scene_config
	: 'camera' ':' camera_spec ';'
	| 'lighting' ':' '[' light_list ']' ';'
	;

camera_spec
	: '{' camera_setting+ '}'
	;

camera_setting
	: STRING_LITERAL ':' expression
	;

light_list
	: light_reference (',' light_reference)*
	;

light_reference
	: IDENTIFIER '(' light_params? ')'
	;

light_params
	: light_param (',' light_param)*
	;

light_param
	: IDENTIFIER '=' expression
	;

mesh_definition
	: 'agent_mesh' IDENTIFIER '{' mesh_config_body '}'
	;

mesh_config_body
	: mesh_config_member*
	;

mesh_config_member
	: mesh_property
	| method_definition
	;

mesh_property
	: 'geometry' ':' STRING_LITERAL ';'
	| 'material' ':' material_spec ';'
	;

material_spec
	: '{' material_setting+ '}'
	;

material_setting
	: STRING_LITERAL ':' expression
	;

connection_lines_definition
	: 'connection_lines' IDENTIFIER '{' method_definition+ '}'
	;

environment_objects_definition
	: 'environment_objects' IDENTIFIER '{' environment_body '}'
	;

environment_body
	: environment_member*
	;

environment_member
	: environment_config
	| method_definition
	;

environment_config
	: 'camera_objects' ':' '[' type_annotation ']' ';'
	| 'sensor_objects' ':' '[' type_annotation ']' ';'
	| 'data_streams' ':' '[' type_annotation ']' ';'
	;

control_panel_definition
	: 'control_panel' IDENTIFIER '{' control_panel_body '}'
	;

control_panel_body
	: control_panel_member*
	;

control_panel_member
	: control_panel_config
	| component_definition
	;

control_panel_config
	: 'layout' ':' STRING_LITERAL ';'
	;

component_definition
	: 'component' IDENTIFIER '{' component_body '}'
	;

component_body
	: component_member*
	;

component_member
	: component_config
	| method_definition
	;

component_config
	: 'charts' ':' '[' chart_list ']' ';'
	| 'log_sources' ':' '[' string_list ']' ';'
	| 'filters' ':' filter_spec ';'
	;

chart_list
	: chart_reference (',' chart_reference)*
	;

chart_reference
	: IDENTIFIER '(' chart_params ')'
	;

chart_params
	: chart_param (',' chart_param)*
	;

chart_param
	: IDENTIFIER '=' expression
	;

filter_spec
	: '{' filter_setting+ '}'
	;

filter_setting
	: STRING_LITERAL ':' expression
	;

ar_interface_definition
	: 'ar_interface' IDENTIFIER '{' ar_body '}'
	;

ar_body
	: ar_member*
	;

ar_member
	: ar_config
	| gesture_recognizer_definition
	| method_definition
	;

ar_config
	: 'platform' ':' STRING_LITERAL ';'
	;

gesture_recognizer_definition
	: 'gesture_recognizer' IDENTIFIER '{' gesture_body '}'
	;

gesture_body
	: gesture_member*
	;

gesture_member
	: gesture_config
	| gesture_event_handler
	;

gesture_config
	: 'gestures' ':' '{' gesture_mapping+ '}' ';'
	;

gesture_mapping
	: STRING_LITERAL ':' gesture_reference
	;

gesture_reference
	: IDENTIFIER '(' arglist? ')'
	;

gesture_event_handler
	: 'on' IDENTIFIER '(' parameter_list ')' '{' suite '}'
	;

// Monitoring System
monitoring_system_definition
	: 'monitoring_system' IDENTIFIER '{' monitoring_body '}'
	;

monitoring_body
	: monitoring_member*
	;

monitoring_member
	: monitoring_config
	| health_analyzer_definition
	;

monitoring_config
	: 'metrics_collector' ':' IDENTIFIER '(' arglist? ')' ';'
	| 'trace_collector' ':' IDENTIFIER '(' arglist? ')' ';'
	| 'log_aggregator' ':' IDENTIFIER '(' arglist? ')' ';'
	;

health_analyzer_definition
	: 'health_analyzer' IDENTIFIER '{' health_analyzer_body '}'
	;

health_analyzer_body
	: health_analyzer_member*
	;

health_analyzer_member
	: health_analyzer_config
	| method_definition
	;

health_analyzer_config
	: 'anomaly_detector' ':' IDENTIFIER '(' arglist? ')' ';'
	| 'forecasting_model' ':' IDENTIFIER '(' arglist? ')' ';'
	;

// Reactive System
reactive_system_definition
	: 'reactive_system' IDENTIFIER '{' reactive_system_body '}'
	;

reactive_system_body
	: reactive_system_member*
	;

reactive_system_member
	: stream_definition
	| event_processor_definition
	| reactive_agent_definition
	| coordination_system_definition
	| sensor_network_definition
	;

// Stream Definition
stream_definition
	: 'stream' IDENTIFIER '<' type_annotation '>' '{' stream_body '}'
	;

stream_body
	: stream_member*
	;

stream_member
	: stream_config
	| method_definition
	;

stream_config
	: 'buffer_size' ':' expression ';'
	| 'backpressure_strategy' ':' STRING_LITERAL ';'
	;

event_processor_definition
	: 'event_processor' IDENTIFIER '{' event_processor_body '}'
	;

event_processor_body
	: event_processor_member*
	;

event_processor_member
	: pattern_definition
	| window_definition
	| analytics_engine_definition
	;

pattern_definition
	: 'pattern' IDENTIFIER '{' pattern_body '}'
	;

pattern_body
	: pattern_member*
	;

pattern_member
	: pattern_config
	| method_definition
	;

pattern_config
	: 'within' ':' expression ';'
	;

window_definition
	: 'window' IDENTIFIER '<' type_annotation '>' '{' window_body '}'
	;

window_body
	: window_member*
	;

window_member
	: window_config
	| method_definition
	;

window_config
	: 'size' ':' expression ';'
	| 'slide' ':' expression ';'
	;

analytics_engine_definition
	: 'analytics_engine' IDENTIFIER '{' method_definition+ '}'
	;

reactive_agent_definition
	: 'reactive_agent' IDENTIFIER ('extends' IDENTIFIER)? '{' reactive_agent_body '}'
	;

reactive_agent_body
	: reactive_agent_member*
	;

reactive_agent_member
	: reactive_agent_config
	| method_definition
	;

reactive_agent_config
	: 'video_input' ':' type_annotation ';'
	| 'command_input' ':' type_annotation ';'
	| 'detection_output' ':' type_annotation ';'
	| 'alert_output' ':' type_annotation ';'
	;

coordination_system_definition
	: 'coordination_system' IDENTIFIER '{' coordination_body '}'
	;

coordination_body
	: coordination_member*
	;

coordination_member
	: coordination_config
	| method_definition
	;

coordination_config
	: 'event_bus' ':' type_annotation ';'
	| 'agent_registry' ':' type_annotation ';'
	;

sensor_network_definition
	: 'sensor_network' IDENTIFIER '{' sensor_network_body '}'
	;

sensor_network_body
	: sensor_network_member*
	;

sensor_network_member
	: sensor_network_config
	| method_definition
	;

sensor_network_config
	: 'temperature_stream' ':' type_annotation ';'
	| 'motion_stream' ':' type_annotation ';'
	| 'audio_stream' ':' type_annotation ';'
	;

// Event Sourcing
event_sourcing_definition
	: 'event_sourcing' IDENTIFIER '{' event_sourcing_body '}'
	;

event_sourcing_body
	: event_sourcing_member*
	;

event_sourcing_member
	: event_type_definition
	| event_store_definition
	| projection_definition
	;

event_type_definition
	: 'event' IDENTIFIER '{' event_type_body '}'
	;

event_type_body
	: event_field+
	;

event_field
	: IDENTIFIER ':' type_annotation ';'
	;

event_store_definition
	: 'event_store' IDENTIFIER '{' event_store_body '}'
	;

event_store_body
	: event_store_member*
	;

event_store_member
	: event_store_config
	| method_definition
	;

event_store_config
	: 'storage' ':' IDENTIFIER '(' arglist? ')' ';'
	;

projection_definition
	: 'projection' IDENTIFIER '{' method_definition+ '}'
	;