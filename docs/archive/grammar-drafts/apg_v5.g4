grammar apg;

options {
 language = Python3;
}

// Parser Rules

apg
    : appconfig? (table | enumdefinition | relationship | index | role | trigger | script | workflow | form | masterdetailform | wizard | formstorage | chart | aiconfig | customfunction | report | notification | scheduledtask | backgroundjob | apiconfig | webhook)* EOF
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

table
    : 'table' TABLENAME '{' column+ '}' note?
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

validation
    : 'min:' INT note?
    | 'max:' INT note?
    | 'pattern:' PATTERN note?
    | 'required' note?
    | 'crossfield:' SIMPLETEXT note?
    | 'regex:' PATTERN note?
    | 'conditional:' SIMPLETEXT note?
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
    : 'form' FORMNAME 'for' TABLENAME '{' formlayout? formcomponent+ formbutton* formnavigator? '}' note?
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
    : FIELDNAME ('as' componenttype)? ('hint:' SIMPLETEXT)? ('help:' MULTILINETEXT)? ('upload:' uploadconfig)? ('customwidget:' WIDGETNAME)? note?
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
    : 'masterdetailform' FORMNAME 'master' TABLENAME 'details' '{' detailcomponent+ '}' note?
    ;

detailcomponent
    : 'detail' TABLENAME ('exclude' '(' excludefields ')')? note?
    ;

excludefields
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizard
    : 'wizard' WIZARDNAME ('store' 'in' '{' tablemapping (',' tablemapping)* '}')? '{' wizardstep+ '}' note?
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
    : FIELDNAME ('as' componenttype)? ('hint:' SIMPLETEXT)? ('help:' MULTILINETEXT)? note?
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

apiconfig
    : 'apiconfig' APIIDENTIFIER '{' apidetails '}' note?
    ;

apidetails
    : 'endpoint:' URL note?
    | 'method:' httpmethod note?
    | 'headers:' '{' header (',' header)* '}' note?
    | 'params:' '{' param (',' param)* '}' note?
    | 'body:' MULTILINETEXT note?
    ;

httpmethod
    : 'get' note?
    | 'post' note?
    | 'put' note?
    | 'delete' note?
    ;

header
    : HEADERNAME ':' SIMPLETEXT note?
    ;

param
    : PARAMNAME ':' SIMPLETEXT note?
    ;

webhook
    : 'webhook' WEBHOOKNAME '{' webhookdetails '}' note?
    ;

webhookdetails
    : 'url:' URL note?
    | 'event:' SIMPLETEXT note?
    | 'workflow:' WORKFLOWNAME note?
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
    : [A-Z_][A-Z0-9_]* note?
    ;

FIELDNAME
    : [a-z_][a-z0-9_]* note?
    ;

USERNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

ROLENAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

PROMPTNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

FORMNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

SCRIPTNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

WORKFLOWNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

STEPIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

BUTTONNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

NAVIGATORNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

STORAGEIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

CHARTNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

FUNCTIONNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

PARAMNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

REPORTNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

NOTIFICATIONNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

TASKNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

JOBNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

WIDGETNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

ENUMNAME
    : [A-Z][a-zA-Z0-9_]* note?
    ;

ENUMVAL
    : [A-Z_][A-Z0-9_]* note?
    ;

BOOLEAN
    : 'true' | 'false' note?
    ;

INT
    : [0-9]+ note?
    ;

SIMPLETEXT
    : '\'' ( ~['\\] | '\\' . )* '\'' note?
    ;

MULTILINETEXT
    : '"""' ( ~["\\] | '\\' . )* '"""' note?
    ;

URL
    : ('http' | 'https') '://' (LETTER | DIGIT | '.' | '-' | '_')+ ('/' (LETTER | DIGIT | '.' | '-' | '_' | '%' | ':' | '@' | '&' | '=' | '+' | '$' | ',' | '?' | '#' | '!' | '(' | ')' | '*' | '~')*)? note?
    ;

PATTERN
    : '/' ('\\' . | ~('\\' | '/'))* '/' note?
    ;

HEADERNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

WEBHOOKNAME
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

APIIDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

LABEL
    : '"' ( ~["\\] | '\\' . )* '"' note?
    ;

CRONEXPRESSION
    : [a-zA-Z0-9_/*,-]+ note?
    ;

TIME
    : [0-9]{2} ':' [0-9]{2} note?
    ;

EMAIL
    : [a-zA-Z0-9_.+-]+ '@' [a-zA-Z0-9-]+ '.' [a-zA-Z0-9-.]+ note?
    ;

TEMPLATE
    : '<template>' ( ~[<\\] | '\\' . )* '</template>' note?
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

