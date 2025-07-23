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
    : 'baseurl:' STRING note?
    | 'databaseuri:' STRING note?
    | 'secretkey:' STRING note?
    | 'debug:' BOOLEAN note?
    ;

kivyconfig
    : 'windowsize:' STRING note?
    | 'orientation:' STRING note?
    | 'icon:' STRING note?
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
    : 'model:' STRING note?
    | 'apikey:' STRING note?
    | 'endpoint:' STRING note?
    | 'intents:' STRING note?
    | 'prompt:' promptdefinition note?
    ;

contentgeneratorconfig
    : 'model:' STRING note?
    | 'apikey:' STRING note?
    | 'endpoint:' STRING note?
    | 'contenttypes:' STRING note?
    | 'prompt:' promptdefinition note?
    ;

dataanalyzerconfig
    : 'model:' STRING note?
    | 'apikey:' STRING note?
    | 'endpoint:' STRING note?
    | 'datatypes:' STRING note?
    | 'prompt:' promptdefinition note?
    ;

promptdefinition
    : 'prompt' PROMPTNAME '{' prompttext '}' note?
    ;

prompttext
    : 'text:' STRING note?
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
    : 'encrypted' '(' STRING ')' note?
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
    | 'hint:' STRING note?
    | 'help:' STRING note?
    ;

validation
    : 'min:' INT note?
    | 'max:' INT note?
    | 'pattern:' STRING note?
    | 'required' note?
    | 'crossfield:' STRING note?
    | 'regex:' STRING note?
    | 'conditional:' STRING note?
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
    | 'description:' STRING note?
    | 'deadline:' STRING note?
    | 'cron:' STRING note?
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
    | 'deadline:' STRING note?
    | 'escalateto:' (USERNAME | 'role' ROLENAME) note?
    | 'condition:' STRING note?
    | 'eventtrigger:' STRING note?
    ;

workflowstatement
    : 'showform' '(' FORMNAME ')' note?
    | 'assigntask' '(' (USERNAME | 'role' ROLENAME) ')' note?
    | 'setdeadline' '(' STRING ')' note?
    | 'sendnotification' '(' notificationconfig ')' note?
    | 'updaterecord' '(' updateassignment (',' updateassignment)* ')' note?
    | 'executescript' '(' SCRIPTNAME ')' note?
    | 'if' '(' conditionexpression ')' '{' workflowstatement+ '}' ('else' '{' workflowstatement+ '}')? note?
    | 'while' '(' conditionexpression ')' '{' workflowstatement+ '}' note?
    | 'for' '(' assignment ';' conditionexpression ';' assignment ')' '{' workflowstatement+ '}' note?
    | 'checkdata' '(' STRING ')' note?
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
    : FIELDNAME ('as' componenttype)? ('hint:' STRING)? ('help:' STRING)? note?
    ;

formcomponent
    : FIELDNAME ('as' componenttype)? ('hint:' STRING)? ('help:' STRING)? ('upload:' uploadconfig)? ('customwidget:' WIDGETNAME)? note?
    ;

uploadconfig
    : '{' 'storage:' STRING ',' 'allowedtypes:' STRING (',' 'maxsize:' INT)? '}' note?
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
    : 'button' BUTTONNAME 'label' STRING ('action' actiontype)? ('style' buttonstyle)? note?
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
    : FIELDNAME ('as' componenttype)? ('hint:' STRING)? ('help:' STRING)? note?
    ;

formstorage
    : 'formstorage' STORAGEIDENTIFIER 'form' FORMNAME 'save' 'for' 'later' note?
    ;

chart
    : 'chart' CHARTNAME 'type' charttype 'for' TABLENAME 'fields' '(' fieldlist ')' ('title' STRING)? note?
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
    : 'template:' STRING note?
    | 'datasource:' IDENTIFIER note?
    | 'fields:' '(' fieldlist ')' note?
    | 'filter:' STRING note?
    | 'sort:' STRING note?
    | 'group:' STRING note?
    | 'schedule:' scheduleconfig note?
    | 'transform:' datatransform note?
    | 'aggregate:' dataaggregate note?
    ;

datatransform
    : '{' 'field:' FIELDNAME ',' 'expression:' STRING '}' note?
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
    : '{' 'frequency:' STRING ',' 'time:' STRING '}' note?
    ;

notification
    : 'notification' NOTIFICATIONNAME '{' notificationconfig+ '}' note?
    ;

notificationconfig
    : 'type:' notificationtype note?
    | 'trigger:' triggertype note?
    | 'recipient:' STRING note?
    | 'message:' STRING note?
    | 'data:' STRING note?
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
    : 'cron:' STRING note?
    | 'action:' taskaction note?
    | 'condition:' STRING note?
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
    : 'interval:' STRING note?
    | 'action:' jobaction note?
    | 'condition:' STRING note?
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
    : 'endpoint:' STRING note?
    | 'method:' httpmethod note?
    | 'headers:' '{' header (',' header)* '}' note?
    | 'params:' '{' param (',' param)* '}' note?
    | 'body:' STRING note?
    ;

httpmethod
    : 'get' note?
    | 'post' note?
    | 'put' note?
    | 'delete' note?
    ;

header
    : HEADERNAME ':' STRING note?
    ;

param
    : PARAMNAME ':' STRING note?
    ;

webhook
    : 'webhook' WEBHOOKNAME '{' webhookdetails '}' note?
    ;

webhookdetails
    : 'url:' STRING note?
    | 'event:' STRING note?
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
    | STRING note?
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
    | STRING note?
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

STRING
    : '\'' ('\\' . | ~('\\' | '\''))* '\'' note?
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
    : 'note' ':' STRING
    ;

