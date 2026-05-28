grammar apg;

options {
 language = Python3;
}

// Parser Rules

apg
    : appConfig? (table | enumDefinition | relationship | index | role | trigger | script | workflow | form | masterDetailForm | wizard | formStorage | chart | aiConfig | customFunction | report | notification)* EOF
    ;

appConfig
    : 'AppConfig' '{' appType '}' note?
    ;

appType
    : 'FlaskAppBuilder' '{' flaskConfig* '}' note?
    | 'KivyDesktop' '{' kivyConfig* '}' note?
    | 'KivyMobile' '{' kivyConfig* '}' note?
    ;

flaskConfig
    : 'baseUrl:' STRING note?
    | 'databaseUri:' STRING note?
    | 'secretKey:' STRING note?
    | 'debug:' BOOLEAN note?
    ;

kivyConfig
    : 'windowSize:' STRING note?
    | 'orientation:' STRING note?
    | 'icon:' STRING note?
    ;

aiConfig
    : 'AIConfig' '{' aiService+ '}' note?
    ;

aiService
    : 'Chatbot' '{' chatbotConfig* '}' note?
    | 'ContentGenerator' '{' contentGeneratorConfig* '}' note?
    | 'DataAnalyzer' '{' dataAnalyzerConfig* '}' note?
    ;

chatbotConfig
    : 'model:' STRING note?
    | 'apiKey:' STRING note?
    | 'endpoint:' STRING note?
    | 'intents:' STRING note?
    | 'prompt:' promptDefinition note?
    ;

contentGeneratorConfig
    : 'model:' STRING note?
    | 'apiKey:' STRING note?
    | 'endpoint:' STRING note?
    | 'contentTypes:' STRING note?
    | 'prompt:' promptDefinition note?
    ;

dataAnalyzerConfig
    : 'model:' STRING note?
    | 'apiKey:' STRING note?
    | 'endpoint:' STRING note?
    | 'dataTypes:' STRING note?
    | 'prompt:' promptDefinition note?
    ;

promptDefinition
    : 'Prompt' PROMPTNAME '{' promptText '}' note?
    ;

promptText
    : 'text:' STRING note?
    ;

table
    : 'Table' TABLENAME '{' column+ '}' note?
    ;

column
    : FIELDNAME dataType columnAttributes? note?
    ;

dataType
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
    | encryptedType note?
    | vectorType note?
    | graphType note?
    | documentType note?
    ;

encryptedType
    : 'encrypted' '(' STRING ')' note?
    ;

vectorType
    : 'vector' '(' INT ')' note?
    ;

graphType
    : 'graph' note?
    ;

documentType
    : 'document' note?
    ;

columnAttributes
    : '[' columnAttribute (',' columnAttribute)* ']' note?
    ;

columnAttribute
    : 'pk' note?
    | 'increment' note?
    | 'unique' note?
    | 'not null' note?
    | 'default:' expression note?
    | 'ref:' relationshipType TABLENAME '.' FIELDNAME note?
    | 'validation:' validation note?
    | 'hint:' STRING note?
    | 'help:' STRING note?
    ;

validation
    : 'min:' INT note?
    | 'max:' INT note?
    | 'pattern:' STRING note?
    | 'required' note?
    ;

relationship
    : 'Ref' TABLENAME '.' FIELDNAME relationshipType TABLENAME '.' FIELDNAME note?
    ;

relationshipType
    : '>'   // one-to-many note?
    | '-'   // one-to-one note?
    | '<'   // many-to-one note?
    | '<>'  // many-to-many note?
    ;

index
    : 'Table' TABLENAME '{' indexes '}' note?
    ;

indexes
    : 'indexes' '{' indexDefinition+ '}' note?
    ;

indexDefinition
    : '(' FIELDNAME (',' FIELDNAME)* ')' indexAttributes? note?
    ;

indexAttributes
    : '[' indexAttribute (',' indexAttribute)* ']' note?
    ;

indexAttribute
    : 'unique' note?
    ;

enumDefinition
    : 'Enum' ENUMNAME '{' enumValue+ '}' note?
    ;

enumValue
    : ENUMVAL note?
    ;

role
    : 'Role' ROLENAME ('inherits' ROLENAME)? '{' permission+ '}' note?
    ;

permission
    : 'Permission' action 'on' TABLENAME ('where' condition)? note?
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
    : 'Trigger' TRIGGERNAME 'on' TABLENAME triggerEvent '{' triggerBody '}' note?
    ;

triggerEvent
    : 'before' note?
    | 'after' note?
    | 'instead of' note?
    ;

triggerBody
    : statement+ note?
    ;

script
    : 'Script' SCRIPTNAME scriptLang '{' scriptBody '}' note?
    ;

scriptLang
    : 'zsh' note?
    | 'bash' note?
    | 'csh' note?
    | 'python' note?
    ;

scriptBody
    : statement+ note?
    ;

workflow
    : 'Workflow' WORKFLOWNAME '{' workflowMetadata workflowStep+ '}' note?
    ;

workflowMetadata
    : 'initiator:' USERNAME note?
    | 'description:' STRING note?
    | 'deadline:' STRING note?
    | 'cron:' STRING note?
    ;

workflowStep
    : 'Step' STEPIDENTIFIER '{' stepMetadata workflowStatement+ '}' note?
    ;

stepMetadata
    : 'form:' FORMNAME note?
    | 'assignTo:' (USERNAME | 'role' ROLENAME) note?
    | 'responsible:' USERNAME note?
    | 'accountable:' USERNAME note?
    | 'consulted:' USERNAME note?
    | 'informed:' USERNAME note?
    | 'deadline:' STRING note?
    | 'escalateTo:' (USERNAME | 'role' ROLENAME) note?
    | 'condition:' STRING note?
    ;

workflowStatement
    : 'showForm' '(' FORMNAME ')' note?
    | 'assignTask' '(' (USERNAME | 'role' ROLENAME) ')' note?
    | 'setDeadline' '(' STRING ')' note?
    | 'sendNotification' '(' notificationConfig ')' note?
    | 'updateRecord' '(' TABLENAME '.' FIELDNAME '=' expression ')' note?
    | 'executeScript' '(' SCRIPTNAME ')' note?
    | 'if' '(' condition ')' '{' workflowStatement+ '}' ('else' '{' workflowStatement+ '}')? note?
    | 'checkData' '(' STRING ')' note?
    ;

form
    : 'Form' FORMNAME 'for' TABLENAME '{' formLayout? formComponent+ formButton* formNavigator? '}' note?
    ;

formLayout
    : 'Layout' '{' layoutElement+ '}' note?
    ;

layoutElement
    : 'row' '(' rowElement (',' rowElement)* ')' note?
    ;

rowElement
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)? note?
    ;

formComponent
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)? ('upload:' uploadConfig)? note?
    ;

uploadConfig
    : '{' 'storage:' STRING ',' 'allowedTypes:' STRING (',' 'maxSize:' INT)? '}' note?
    ;

componentType
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
    ;

formButton
    : 'Button' BUTTONNAME 'label' STRING ('action' actionType)? ('style' buttonStyle)? note?
    ;

actionType
    : 'submit' note?
    | 'reset' note?
    | 'button' note?
    ;

buttonStyle
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

formNavigator
    : 'Navigator' NAVIGATORNAME (('{' navigatorButton* '}') | 'exclude' '(' navigatorButton (',' navigatorButton)* ')')? note?
    ;

navigatorButton
    : 'First' note?
    | 'Previous' note?
    | 'Next' note?
    | 'Last' note?
    | 'Insert' note?
    | 'Delete' note?
    | 'Edit' note?
    | 'Post' note?
    | 'Cancel' note?
    ;

masterDetailForm
    : 'MasterDetailForm' FORMNAME 'master' TABLENAME 'details' '{' detailComponent+ '}' note?
    ;

detailComponent
    : 'Detail' TABLENAME ('exclude' '(' excludeFields ')')? note?
    ;

excludeFields
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizard
    : 'Wizard' WIZARDNAME ('store' 'in' '{' tableMapping (',' tableMapping)* '}')? '{' wizardStep+ '}' note?
    ;

tableMapping
    : TABLENAME 'fields' '(' fieldList ')' note?
    ;

fieldList
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizardStep
    : 'Step' STEPIDENTIFIER '{' wizardStatement+ '}' note?
    ;

wizardStatement
    : statement note?
    | 'form' '(' formFieldList ')' note?
    ;

formFieldList
    : formField (',' formField)* note?
    ;

formField
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)? note?
    ;

formStorage
    : 'FormStorage' STORAGEIDENTIFIER 'form' FORMNAME 'save' 'for' 'later' note?
    ;

chart
    : 'Chart' CHARTNAME 'type' chartType 'for' TABLENAME 'fields' '(' fieldList ')' ('title' STRING)? note?
    ;

chartType
    : 'bar' note?
    | 'line' note?
    | 'pie' note?
    | 'scatter' note?
    | 'area' note?
    ;

customFunction
    : 'Function' FUNCTIONNAME '(' parameterList ')' 'returns' dataType '{' functionBody '}' note?
    ;

parameterList
    : parameter (',' parameter)* note?
    ;

parameter
    : PARAMNAME dataType note?
    ;

functionBody
    : statement+ note?
    ;

report
    : 'Report' REPORTNAME '{' reportConfig* '}' note?
    ;

reportConfig
    : 'template:' STRING note?
    | 'dataSource:' IDENTIFIER note?
    | 'fields:' '(' fieldList ')' note?
    | 'filter:' STRING note?
    | 'sort:' STRING note?
    | 'group:' STRING note?
    | 'schedule:' scheduleConfig note?
    ;

scheduleConfig
    : '{' 'frequency:' STRING ',' 'time:' STRING '}' note?
    ;

notification
    : 'Notification' NOTIFICATIONNAME '{' notificationConfig+ '}' note?
    ;

notificationConfig
    : 'type:' notificationType note?
    | 'trigger:' triggerEvent note?
    | 'recipient:' STRING note?
    | 'message:' STRING note?
    ;

notificationType
    : 'email' note?
    | 'sms' note?
    | 'in-app' note?
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

TRIGGERNAME
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

