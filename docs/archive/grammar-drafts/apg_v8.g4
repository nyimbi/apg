grammar apg;

options {
 language = Python3;
}

// Parser Rules

apg
    : import_statement* appconfig? i18n_config? (template_definition | version_tag | conditional_block | plugin_definition | event_definition | event_handler | custom_component | table | enumdefinition | relationship | index | role | trigger | script | workflow | form | masterdetailform | wizard | formstorage | chart | aiconfig | customfunction | report | notification | scheduledtask | backgroundjob | apiconfig | webhook)* EOF
    ;



// Import Statements
import_statement
    : 'import' FILENAME (, FILENAME)* note?
    ;

appconfig
    : 'appconfig' '{' apptype '}' note?
    ;

apptype
    : 'flaskappbuilder' '{' flaskconfig* '}' note?
    | 'kivydesktop' '{' kivyconfig* '}' note?
    | 'kivymobile' '{' kivyconfig* '}' note?
    ;

flaskconfig
    : 'baseurl:' URL note?
    | 'databaseuri:' URL note?
    | 'secretkey:' SIMPLETEXT note?
    | 'debug:' BOOLEAN note?
    ;

kivyconfig
    : 'windowsize:' SIMPLETEXT note?
    | 'orientation:' SIMPLETEXT note?
    | 'icon:' URL note?
    ;

aiconfig
    : 'aiconfig' '{' aiservice+ '}' note?
    ;

aiservice
    : 'chatbot' '{' chatbotconfig* '}' note?
    | 'contentgenerator' '{' contentgeneratorconfig* '}' note?
    | 'dataanalyzer' '{' dataanalyzerconfig* '}' note?
    ;

chatbotconfig
    : 'model:' SIMPLETEXT note?
    | 'apikey:' SIMPLETEXT note?
    | 'endpoint:' URL note?
    | 'intents:' SIMPLETEXT note?
    | 'prompt:' promptdefinition note?
    ;

contentgeneratorconfig
    : 'model:' SIMPLETEXT note?
    | 'apikey:' SIMPLETEXT note?
    | 'endpoint:' URL note?
    | 'contenttypes:' SIMPLETEXT note?
    | 'prompt:' promptdefinition note?
    ;

dataanalyzerconfig
    : 'model:' SIMPLETEXT note?
    | 'apikey:' SIMPLETEXT note?
    | 'endpoint:' URL note?
    | 'datatypes:' SIMPLETEXT note?
    | 'prompt:' promptdefinition note?
    ;

promptdefinition
    : 'prompt' PROMPTNAME '{' prompttext '}' note?
    ;

prompttext
    : 'text:' MULTILINETEXT note?
    ;

// Template Definitions
template_definition
    : 'template' TEMPLATENAME '{' template_content '}' note?
    ;

template_content
    : form | workflow | report
    ;

// Version Control
version_tag
    : 'version' SEMVER note?
    ;

// Enhanced Data Validation
validation
    : 'min:' INT note?
    | 'max:' INT note?
    | 'pattern:' PATTERN note?
    | 'required' note?
    | 'crossfield:' SIMPLETEXT note?
    | 'regex:' PATTERN note?
    | 'conditional:' SIMPLETEXT note?
    | 'custom:' CUSTOMVALIDATIONNAME note?
    ;

// Internationalization (i18n)
i18n_config
    : 'i18n' '{' language_definition+ '}' note?
    ;

language_definition
    : LANG_CODE '{' translation+ '}' note?
    ;

translation
    : KEY ':' SIMPLETEXT note?
    ;

// Conditional Compilation
conditional_block
    : '#if' condition_expression '{' apg_element+ '}' ('#else' '{' apg_element+ '}')? '#endif' note?
    ;

condition_expression
    : IDENTIFIER
    | BOOLEAN
    | '(' condition_expression ')'
    | condition_expression ('&&' | '||') condition_expression
    ;

apg_element
    : table | form | workflow | enumdefinition | relationship | index | role | trigger | script | masterdetailform | wizard | formstorage | chart | aiconfig | customfunction | report | notification | scheduledtask | backgroundjob | apiconfig | webhook | event_definition | event_handler | custom_component
    ;

// Custom UI Components
custom_component
    : 'component' COMPONENTNAME '{' component_property* '}' note?
    ;

component_property
    : PROPERTYNAME ':' (SIMPLETEXT | INT | BOOLEAN) note?
    ;

// API Integration Capabilities
apiconfig
    : 'apiconfig' APIIDENTIFIER '{' apidetails '}' note?
    ;

apidetails
    : 'endpoint:' URL note?
    | 'method:' httpmethod note?
    | 'headers:' '{' header (',' header)* '}' note?
    | 'params:' '{' param (',' param)* '}' note?
    | 'body:' MULTILINETEXT note?
    | 'auth:' auth_config note?
    | 'ratelimit:' rate_limit_config note?
    | 'cache:' cache_config note?
    ;

auth_config
    : 'oauth' '{' oauth_details '}' note?
    | 'apikey' SIMPLETEXT note?
    ;

oauth_details
    : 'client_id:' SIMPLETEXT note?
    | 'client_secret:' SIMPLETEXT note?
    | 'auth_url:' URL note?
    | 'token_url:' URL note?
    ;

rate_limit_config
    : 'requests:' INT note?
    | 'per:' ('second' | 'minute' | 'hour' | 'day') note?
    ;

cache_config
    : 'duration:' INT ('seconds' | 'minutes' | 'hours') note?
    ;

// Event-Driven Architecture
event_definition
    : 'event' EVENTNAME '{' event_property* '}' note?
    ;

event_property
    : PROPERTYNAME ':' datatype note?
    ;

event_handler
    : 'on' EVENTNAME '{' statement+ '}' note?
    ;

// Plugin System
plugin_definition
    : 'plugin' PLUGINNAME '{' plugin_config* '}' note?
    ;

plugin_config
    : 'version:' SEMVER note?
    | 'source:' URL note?
    | 'config:' '{' plugin_param+ '}' note?
    ;

plugin_param
    : PARAMNAME ':' (SIMPLETEXT | INT | BOOLEAN) note?
    ;

// Existing Rules
table
    : 'table' TABLENAME version_tag? '{' column+ '}' note?
    ;

column
    : FIELDNAME datatype columnattributes? note?
    ;

datatype
    : 'smallint' note?
    | 'integer' note?
    | 'bigint' note?
    | 'decimal' '(' INT ',' INT ')' note?
    | 'numeric' '(' INT ',' INT ')' note?
    | 'real' note?
    | 'double precision' note?
    | 'serial' note?
    | 'bigserial' note?
    | 'money' note?
    | 'char' '(' INT ')' note?
    | 'varchar' '(' INT ')' note?
    | 'text' note?
    | 'bytea' note?
    | 'timestamp' ('with time zone' | 'without time zone')? note?
    | 'date' note?
    | 'time' ('with time zone' | 'without time zone')? note?
    | 'interval' note?
    | 'boolean' note?
    | 'enum' ENUMNAME note?
    | 'uuid' note?
    | 'xml' note?
    | 'json' note?
    | 'jsonb' note?
    | 'cidr' note?
    | 'inet' note?
    | 'macaddr' note?
    | 'bit' '(' INT ')' note?
    | 'bit varying' '(' INT ')' note?
    | 'tsvector' note?
    | 'tsquery' note?
    | 'uuid' note?
    | 'int4range' note?
    | 'int8range' note?
    | 'numrange' note?
    | 'tsrange' note?
    | 'tstzrange' note?
    | 'daterange' note?
    | encryptedtype note?
    | vectortype note?
    | graphtype note?
    | documenttype note?
    ;

encryptedtype
    : 'encrypted' '(' SIMPLETEXT ')' note?
    ;

vectortype
    : 'vector' '(' INT ')' note?
    ;

graphtype
    : 'graph' note?
    ;

documenttype
    : 'document' note?
    ;

columnattributes
    : '[' columnattribute (',' columnattribute)* ']' note?
    ;

columnattribute
    : 'pk' note?
    | 'increment' note?
    | 'unique' note?
    | 'not null' note?
    | 'default:' expression note?
    | 'ref:' relationshiptype TABLENAME '.' FIELDNAME note?
    | 'validation:' validation note?
    | 'label:' LABEL note?
    | 'hint:' SIMPLETEXT note?
    | 'help:' MULTILINETEXT note?
    ;

relationship
    : 'ref' TABLENAME '.' FIELDNAME relationshiptype TABLENAME '.' FIELDNAME note?
    ;

relationshiptype
    : '>'   // one-to-many note?
    | '-'   // one-to-one note?
    | '<'   // many-to-one note?
    | '<>'  // many-to-many note?
    ;

index
    : 'table' TABLENAME '{' indexes '}' note?
    ;

indexes
    : 'indexes' '{' indexdefinition+ '}' note?
    ;

indexdefinition
    : '(' FIELDNAME (',' FIELDNAME)* ')' indexattributes? note?
    ;

indexattributes
    : '[' indexattribute (',' indexattribute)* ']' note?
    ;

indexattribute
    : 'unique' note?
    ;

enumdefinition
    : 'enum' ENUMNAME '{' enumvalue+ '}' note?
    ;

enumvalue
    : ENUMVAL note?
    ;

role
    : 'role' ROLENAME ('inherits' ROLENAME)? '{' permission+ '}' note?
    ;

permission
    : 'permission' action 'on' TABLENAME ('where' condition)? note?
    ;

action
    : 'read' note?
    | 'write' note?
    | 'delete' note?
    | 'execute' note?
    ;

condition
    : expression note?
    ;

trigger
    : 'trigger' TRIGGERNAME 'on' TABLENAME triggertype '{' triggerbody '}' note?
    ;

triggertype
    : 'before' note?
    | 'after' note?
    | 'instead of' note?
    ;

triggerbody
    : statement+ note?
    ;

script
    : 'script' SCRIPTNAME scriptlang '{' scriptbody '}' note?
    ;

scriptlang
    : 'zsh' note?
    | 'bash' note?
    | 'csh' note?
    | 'python' note?
    ;

scriptbody
    : statement+ note?
    ;

workflow
    : 'workflow' WORKFLOWNAME '{' workflowmetadata workflowstep+ '}' note?
    ;

workflowmetadata
    : 'initiator:' USERNAME note?
    | 'description:' MULTILINETEXT note?
    | 'deadline:' SIMPLETEXT note?
    | 'cron:' CRONEXPRESSION note?
    ;

workflowstep
    : 'step' STEPIDENTIFIER '{' stepmetadata workflowstatement+ '}' note?
    ;

stepmetadata
    : 'form:' FORMNAME note?
    | 'assignto:' (USERNAME | 'role' ROLENAME) note?
    | 'responsible:' USERNAME note?
    | 'accountable:' USERNAME note?
    | 'consulted:' USERNAME note?
    | 'informed:' USERNAME note?
    | 'deadline:' SIMPLETEXT note?
    | 'escalateto:' (USERNAME | 'role' ROLENAME) note?
    | 'condition:' SIMPLETEXT note?
    | 'eventtrigger:' SIMPLETEXT note?
    ;

workflowstatement
    : 'showform' '(' FORMNAME ')' note?
    | 'assigntask' '(' (USERNAME | 'role' ROLENAME) ')' note?
    | 'setdeadline' '(' SIMPLETEXT ')' note?
    | 'sendnotification' '(' notificationconfig ')' note?
    | 'updaterecord' '(' updateassignment (',' updateassignment)* ')' note?
    | 'executescript' '(' SCRIPTNAME ')' note?
    | 'if' '(' conditionexpression ')' '{' workflowstatement+ '}' ('else' '{' workflowstatement+ '}')? note?
    | 'while' '(' conditionexpression ')' '{' workflowstatement+ '}' note?
    | 'for' '(' assignment ';' conditionexpression ';' assignment ')' '{' workflowstatement+ '}' note?
    | 'checkdata' '(' SIMPLETEXT ')' note?
    | 'subworkflow' '(' WORKFLOWNAME ')' note?
    | 'parallel' '{' workflowstatement+ '}' note?
    | 'customwidget' '(' WIDGETNAME ')' note?
    | 'apicall' '(' apicallconfig ')' note?
    | 'escalationpath' '{' escalationlevel+ '}' note?
    ;

updateassignment
    : TABLENAME '.' FIELDNAME '=' expression note?
    ;

escalationlevel
    : 'level' INT '{' escalationtarget escalationcondition? '}' note?
    ;

escalationtarget
    : 'notify' (USERNAME | 'role' ROLENAME) note?
    ;

escalationcondition
    : 'condition:' conditionexpression note?
    ;

form
    : 'form' FORMNAME 'for' TABLENAME version_tag? '{' formlayout? formcomponent+ formbutton* formnavigator? '}' note?
    ;

formlayout
    : 'layout' '{' layoutelement+ '}' note?
    ;

layoutelement
    : 'row' '(' rowelement (',' rowelement)* ')' note?
    ;

rowelement
    : FIELDNAME ('as' componenttype)? ('hint:' SIMPLETEXT)? ('help:' MULTILINETEXT)? note?
    ;

formcomponent
    : FIELDNAME ('as' (componenttype | COMPONENTNAME)) ('hint:' SIMPLETEXT)? ('help:' MULTILINETEXT)? ('upload:' uploadconfig)? ('customwidget:' WIDGETNAME)? note?
    ;

uploadconfig
    : '{' 'storage:' SIMPLETEXT ',' 'allowedtypes:' SIMPLETEXT (',' 'maxsize:' INT)? '}' note?
    ;

componenttype
    : 'text' note?
    | 'textarea' note?
    | 'select' note?
    | 'radio' note?
    | 'checkbox' note?
    | 'password' note?
    | 'hidden' note?
    | 'file' note?
    | 'date' note?
    | 'time' note?
    | 'datetime' note?
    | 'email' note?
    | 'url' note?
    | 'number' note?
    | 'range' note?
    | 'color' note?
    | 'tel' note?
    | 'search' note?
    | 'speech' note?
    | 'autofill' note?
    | 'image' note?
    | 'video' note?
    | 'audio' note?
    | 'location' note?
    | 'heatmap' note?
    | 'treemap' note?
    | 'gantt' note?
    | 'bubble' note?
    | 'candlestick' note?
    | 'radar' note?
    | 'polararea' note?
    | 'funnel' note?
    | 'waterfall' note?
    ;

formbutton
    : 'button' BUTTONNAME 'label' SIMPLETEXT ('action' actiontype)? ('style' buttonstyle)? note?
    ;

actiontype
    : 'submit' note?
    | 'reset' note?
    | 'button' note?
    ;

buttonstyle
    : 'primary' note?
    | 'secondary' note?
    | 'success' note?
    | 'danger' note?
    | 'warning' note?
    | 'info' note?
    | 'light' note?
    | 'dark' note?
    | 'link' note?
    ;

formnavigator
    : 'navigator' NAVIGATORNAME (('{' navigatorbutton* '}') | 'exclude' '(' navigatorbutton (',' navigatorbutton)* ')')? note?
    ;

navigatorbutton
    : 'first' note?
    | 'previous' note?
    | 'next' note?
    | 'last' note?
    | 'insert' note?
    | 'delete' note?
    | 'edit' note?
    | 'post' note?
    | 'cancel' note?
    ;

masterdetailform
    : 'masterdetailform' FORMNAME 'master' TABLENAME version_tag? 'details' '{' detailcomponent+ '}' note?
    ;

detailcomponent
    : 'detail' TABLENAME ('exclude' '(' excludefields ')')? note?
    ;

excludefields
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizard
    : 'wizard' WIZARDNAME version_tag? ('store' 'in' '{' tablemapping (',' tablemapping)* '}')? '{' wizardstep+ '}' note?
    ;

tablemapping
    : TABLENAME 'fields' '(' fieldlist ')' note?
    ;

fieldlist
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizardstep
    : 'step' STEPIDENTIFIER '{' wizardstatement+ '}' note?
    ;

wizardstatement
    : statement note?
    | 'form' '(' formfieldlist ')' note?
    ;

formfieldlist
    : formfield (',' formfield)* note?
    ;

formfield
    : FIELDNAME ('as' (componenttype | COMPONENTNAME)) ('hint:' SIMPLETEXT)? ('help:' MULTILINETEXT)? note?
    ;

formstorage
    : 'formstorage' STORAGEIDENTIFIER 'form' FORMNAME 'save' 'for' 'later' note?
    ;

chart
    : 'chart' CHARTNAME 'type' charttype 'for' TABLENAME 'fields' '(' fieldlist ')' ('title' SIMPLETEXT)? note?
    ;

charttype
    : 'bar' note?
    | 'line' note?
    | 'pie' note?
    | 'scatter' note?
    | 'area' note?
    | 'heatmap' note?
    | 'treemap' note?
    | 'gantt' note?
    | 'bubble' note?
    | 'candlestick' note?
    | 'radar' note?
    | 'polararea' note?
    | 'funnel' note?
    | 'waterfall' note?
    ;

customfunction
    : 'function' FUNCTIONNAME '(' parameterlist ')' 'returns' datatype '{' functionbody '}' note?
    ;

parameterlist
    : parameter (',' parameter)* note?
    ;

parameter
    : PARAMNAME datatype note?
    ;

functionbody
    : statement+ note?
    ;

report
    : 'report' REPORTNAME '{' reportconfig* '}' note?
    ;

reportconfig
    : 'template:' TEMPLATE note?
    | 'datasource:' IDENTIFIER note?
    | 'fields:' '(' fieldlist ')' note?
    | 'filter:' MULTILINETEXT note?
    | 'sort:' SIMPLETEXT note?
    | 'group:' SIMPLETEXT note?
    | 'schedule:' scheduleconfig note?
    | 'transform:' datatransform note?
    | 'aggregate:' dataaggregate note?
    ;

datatransform
    : '{' 'field:' FIELDNAME ',' 'expression:' SIMPLETEXT '}' note?
    ;

dataaggregate
    : '{' 'field:' FIELDNAME ',' 'function:' aggregatefunction '}' note?
    ;

aggregatefunction
    : 'sum' note?
    | 'avg' note?
    | 'count' note?
    | 'min' note?
    | 'max' note?
    ;

scheduleconfig
    : '{' 'frequency:' CRONEXPRESSION ',' 'time:' TIME note?
    ;

notification
    : 'notification' NOTIFICATIONNAME '{' notificationconfig+ '}' note?
    ;

notificationconfig
    : 'type:' notificationtype note?
    | 'trigger:' triggertype note?
    | 'recipient:' EMAIL note?
    | 'message:' MULTILINETEXT note?
    | 'template:' TEMPLATE note?
    | 'data:' MULTILINETEXT note?
    ;

notificationtype
    : 'email' note?
    | 'sms' note?
    | 'in-app' note?
    ;

scheduledtask
    : 'scheduledtask' TASKNAME '{' taskconfig* '}' note?
    ;

taskconfig
    : 'cron:' CRONEXPRESSION note?
    | 'action:' taskaction note?
    | 'condition:' SIMPLETEXT note?
    ;

taskaction
    : 'runscript' '(' SCRIPTNAME ')' note?
    | 'sendnotification' '(' notificationconfig ')' note?
    | 'updaterecord' '(' updateassignment (',' updateassignment)* ')' note?
    ;

backgroundjob
    : 'backgroundjob' JOBNAME '{' jobconfig* '}' note?
    ;

jobconfig
    : 'interval:' SIMPLETEXT note?
    | 'action:' jobaction note?
    | 'condition:' SIMPLETEXT note?
    ;

jobaction
    : 'runscript' '(' SCRIPTNAME ')' note?
    | 'sendnotification' '(' notificationconfig ')' note?
    | 'updaterecord' '(' updateassignment (',' updateassignment)* ')' note?
    ;

// Statements and Expressions

statement
    : expression ';' note?
    ;

expression
    : IDENTIFIER note?
    | TABLENAME '.' FIELDNAME note?
    | INT note?
    | SIMPLETEXT note?
    ;

assignment
    : FIELDNAME '=' expression note?
    ;

conditionexpression
    : conditionterm (logicalop conditionterm)* note?
    ;

conditionterm
    : conditionfactor (comparisonop conditionfactor)* note?
    ;

conditionfactor
    : IDENTIFIER note?
    | TABLENAME '.' FIELDNAME note?
    | INT note?
    | SIMPLETEXT note?
    | BOOLEAN note?
    | '(' conditionexpression ')' note?
    ;

logicalop
    : '&&' note?
    | '||' note?
    ;

comparisonop
    : '==' note?
    | '!=' note?
    | '<' note?
    | '<=' note?
    | '>' note?
    | '>=' note?
    ;

// Lexer Rules

TABLENAME
    : [A-Z_][A-Z0-9_]*
    ;

FIELDNAME
    : [a-z_][a-z0-9_]*
    ;

USERNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

ROLENAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

PROMPTNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

FORMNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

SCRIPTNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

WORKFLOWNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

STEPIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

BUTTONNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

NAVIGATORNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

STORAGEIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

CHARTNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

FUNCTIONNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

PARAMNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

REPORTNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

NOTIFICATIONNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

TASKNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

JOBNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

WIDGETNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

ENUMNAME
    : [A-Z][a-zA-Z0-9_]*
    ;

ENUMVAL
    : [A-Z_][A-Z0-9_]*
    ;

BOOLEAN
    : 'true' | 'false'
    ;

INT
    : [0-9]+
    ;

SIMPLETEXT
    : '\'' ( ~['\\] | '\\' . )* '\''
    ;

MULTILINETEXT
    : '"""' ( ~["\\] | '\\' . )* '"""'
    ;

URL
    : ('http' | 'https') '://' (LETTER | DIGIT | '.' | '-' | '_')+ ('/' (LETTER | DIGIT | '.' | '-' | '_' | '%' | ':' | '@' | '&' | '=' | '+' | '$' | ',' | '?' | '#' | '!' | '(' | ')' | '*' | '~')*)?
    ;

PATTERN
    : '/' ('\\' . | ~('\\' | '/'))* '/'
    ;

HEADERNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

WEBHOOKNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

APIIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

LABEL
    : '"' ( ~["\\] | '\\' . )* '"'
    ;

CRONEXPRESSION
    : [a-zA-Z0-9_/*,-]+
    ;

TIME
    : [0-9]{2} ':' [0-9]{2}
    ;

EMAIL
    : [a-zA-Z0-9_.+-]+ '@' [a-zA-Z0-9-]+ '.' [a-zA-Z0-9-.]+
    ;

TEMPLATE
    : '<template>' ( ~[<\\] | '\\' . )* '</template>'
    ;

FILENAME
    : [a-zA-Z0-9_]+ '.' [a-zA-Z0-9_]+
    ;

SEMVER
    : DIGIT+ '.' DIGIT+ '.' DIGIT+ ('-' [a-zA-Z0-9-]+)?
    ;

PROPERTYNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

EVENTNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

PLUGINNAME
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

WS
    : [ \t\r\n]+ -> skip
    ;

COMMENT
    : '//' ~[\r\n]* -> skip
    ;

// Fragments

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [a-zA-Z]
    ;

note
    : 'note' ':' SIMPLETEXT
    ;

