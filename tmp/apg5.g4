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
    : 'Prompt' IDENTIFIER '{' promptText '}' note?
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
    : IDENTIFIER note?
    ;

role
    : 'Role' IDENTIFIER ('inherits' IDENTIFIER)? '{' permission+ '}' note?
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
    : 'Trigger' IDENTIFIER 'on' TABLENAME triggerEvent '{' triggerBody '}' note?
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
    : 'Script' IDENTIFIER scriptLang '{' scriptBody '}' note?
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
    : 'Workflow' IDENTIFIER '{' workflowStep+ '}' note?
    ;

workflowStep
    : 'Step' IDENTIFIER 'do' '{' workflowStatement+ '}' note?
    ;

workflowStatement
    : statement note?
    | 'log' '(' STRING ')' note?
    ;

form
    : 'Form' IDENTIFIER 'for' TABLENAME '{' formLayout? formComponent+ formButton* formNavigator? '}' note?
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
    : 'Button' IDENTIFIER 'label' STRING ('action' actionType)? ('style' buttonStyle)? note?
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
    : 'Navigator' IDENTIFIER (('{' navigatorButton* '}') | 'exclude' '(' navigatorButton (',' navigatorButton)* ')')? note?
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
    : 'MasterDetailForm' IDENTIFIER 'master' TABLENAME 'details' '{' detailComponent+ '}' note?
    ;

detailComponent
    : 'Detail' TABLENAME ('exclude' '(' excludeFields ')')? note?
    ;

excludeFields
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizard
    : 'Wizard' IDENTIFIER ('store' 'in' '{' tableMapping (',' tableMapping)* '}')? '{' wizardStep+ '}' note?
    ;

tableMapping
    : TABLENAME 'fields' '(' fieldList ')' note?
    ;

fieldList
    : FIELDNAME (',' FIELDNAME)* note?
    ;

wizardStep
    : 'Step' IDENTIFIER '{' wizardStatement+ '}' note?
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
    : 'FormStorage' IDENTIFIER 'form' IDENTIFIER 'save' 'for' 'later' note?
    ;

chart
    : 'Chart' IDENTIFIER 'type' chartType 'for' TABLENAME 'fields' '(' fieldList ')' ('title' STRING)? note?
    ;

chartType
    : 'bar' note?
    | 'line' note?
    | 'pie' note?
    | 'scatter' note?
    | 'area' note?
    ;

customFunction
    : 'Function' IDENTIFIER '(' parameterList ')' 'returns' dataType '{' functionBody '}' note?
    ;

parameterList
    : parameter (',' parameter)* note?
    ;

parameter
    : IDENTIFIER dataType note?
    ;

functionBody
    : statement+ note?
    ;

report
    : 'Report' IDENTIFIER '{' reportConfig* '}' note?
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
    : 'Notification' IDENTIFIER '{' notificationConfig+ '}' note?
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
    | IDENTIFIER '.' IDENTIFIER note?
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

IDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]* note?
    ;

ENUMNAME
    : [A-Z][a-zA-Z0-9_]* note?
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

