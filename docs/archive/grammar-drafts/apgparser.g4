parser grammar apgparser;

options {
tokenVocab=APGLexer;
language = Python3;
}

// Top-level rule
apg
    : importStatement*
    | appConfig?
    | i18nConfig?
      (templateDefinition 
      | versionTag 
      | conditionalBlock 
      | pluginDefinition 
      | eventDefinition 
      | eventHandler 
      | customComponent 
      | table 
      | enumDefinition 
      | relationship 
      | index 
      | role 
      | trigger 
      | script 
      | workflow 
      | form 
      | masterDetailForm 
      | wizard 
      | formStorage 
      | chart 
      | aiConfig 
      | customFunction 
      | report 
      | notification 
      | scheduledTask 
      | backgroundJob 
      | apiConfig 
      | webhook)* 
    EOF
    ;

// Import Statements
importStatement
    : IMPORT fileName note?
    ;

fileName
    : IDENTIFIER ('.' IDENTIFIER)?
    ;

// Template Definitions
templateDefinition
    : TEMPLATE IDENTIFIER LCURLY templateContent RCURLY note?
    ;

templateContent
    : form | workflow | report
    ;

// Version Control
versionTag
    : VERSION SEMVER note?
    ;

// Conditional Compilation
conditionalBlock
    : HASH IF conditionExpression LCURLY apgElement+ RCURLY
      (HASH ELSE LCURLY apgElement+ RCURLY)?
      HASH ENDIF note?
    ;

conditionExpression
    : IDENTIFIER
    | BOOLEAN
    | LPAREN conditionExpression RPAREN
    | conditionExpression (AND | OR) conditionExpression
    ;

apgElement
    : table | form | workflow | enumDefinition | relationship | index | role 
    | trigger | script | masterDetailForm | wizard | formStorage | chart 
    | aiConfig | customFunction | report | notification | scheduledTask 
    | backgroundJob | apiConfig | webhook | eventDefinition | eventHandler 
    | customComponent
    ;

// Workflows
workflow
    : WORKFLOW IDENTIFIER versionTag? (ON IDENTIFIER)? LCURLY 
      workflowMetadata 
      workflowVariables?
      preExecution?
      workflowStep+ 
      postExecution?
      RCURLY note?
    ;

workflowMetadata
    : (initiator | description | deadline | cronExpression | inputs | outputs)*
    ;

initiator
    : INITIATOR COLON IDENTIFIER note?
    ;

description
    : DESCRIPTION COLON MULTILINE_TEXT note?
    ;

deadline
    : DEADLINE COLON SIMPLE_TEXT note?
    ;

cronExpression
    : CRON COLON CRON_EXPRESSION note?
    ;

inputs
    : INPUTS LCURLY inputDefinition+ RCURLY
    ;

outputs
    : OUTPUTS LCURLY outputDefinition+ RCURLY
    ;

inputDefinition
    : IDENTIFIER COLON dataType (EQUALS expression)?
    ;

outputDefinition
    : IDENTIFIER COLON dataType
    ;

workflowVariables
    : VARIABLES LCURLY variable+ RCURLY
    ;

variable
    : IDENTIFIER COLON dataType (EQUALS expression)?
    ;

preExecution
    : PRE_EXECUTION COLON IDENTIFIER
    ;

postExecution
    : POST_EXECUTION COLON IDENTIFIER
    ;

workflowStep
    : STEP IDENTIFIER dependsOn? LCURLY stepMetadata workflowStatement+ RCURLY note?
    | PARALLEL LCURLY workflowStep+ RCURLY note?
    | SEQUENCE LCURLY workflowStep+ RCURLY note?
    ;

dependsOn
    : DEPENDS_ON COLON IDENTIFIER (COMMA IDENTIFIER)*
    ;

stepMetadata
    : (form | assignTo | responsible | accountable | consulted | informed 
    | deadline | escalateTo | condition | eventTrigger | onError | retries 
    | retryDelay | timeout)*
    ;

form
    : FORM COLON IDENTIFIER note?
    ;

assignTo
    : ASSIGN_TO COLON (IDENTIFIER | ROLE IDENTIFIER) note?
    ;

responsible
    : RESPONSIBLE COLON IDENTIFIER note?
    ;

accountable
    : ACCOUNTABLE COLON IDENTIFIER note?
    ;

consulted
    : CONSULTED COLON IDENTIFIER note?
    ;

informed
    : INFORMED COLON IDENTIFIER note?
    ;

escalateTo
    : ESCALATE_TO COLON (IDENTIFIER | ROLE IDENTIFIER) note?
    ;

condition
    : CONDITION COLON SIMPLE_TEXT note?
    ;

eventTrigger
    : EVENT_TRIGGER COLON SIMPLE_TEXT note?
    ;

onError
    : ON_ERROR COLON errorHandler
    ;

errorHandler
    : CONTINUE | ABORT | RETRY | IDENTIFIER
    ;

retries
    : RETRIES COLON INT note?
    ;

retryDelay
    : RETRY_DELAY COLON INT (SECONDS | MINUTES | HOURS) note?
    ;

timeout
    : TIMEOUT COLON INT (SECONDS | MINUTES | HOURS) note?
    ;

workflowStatement
    : showForm
    | assignTask
    | setDeadline
    | sendNotification
    | updateRecord
    | executeScript
    | ifStatement
    | whileLoop
    | forLoop
    | checkData
    | subWorkflow
    | parallel
    | customWidget
    | apiCall
    | escalationPath
    | foreachLoop
    | humanTask
    ;

showForm
    : SHOW_FORM LPAREN IDENTIFIER RPAREN note?
    ;

assignTask
    : ASSIGN_TASK LPAREN (IDENTIFIER | ROLE IDENTIFIER) RPAREN note?
    ;

setDeadline
    : SET_DEADLINE LPAREN SIMPLE_TEXT RPAREN note?
    ;

sendNotification
    : SEND_NOTIFICATION LPAREN notificationConfig RPAREN note?
    ;

updateRecord
    : UPDATE_RECORD LPAREN updateAssignment (COMMA updateAssignment)* RPAREN note?
    ;

executeScript
    : EXECUTE_SCRIPT LPAREN IDENTIFIER RPAREN note?
    ;

ifStatement
    : IF LPAREN conditionExpression RPAREN LCURLY workflowStatement+ RCURLY 
      (ELSE_IF LPAREN conditionExpression RPAREN LCURLY workflowStatement+ RCURLY)*
      (ELSE LCURLY workflowStatement+ RCURLY)?
    ;

whileLoop
    : WHILE LPAREN conditionExpression RPAREN LCURLY workflowStatement+ RCURLY
    ;

forLoop
    : FOR LPAREN assignment SEMICOLON conditionExpression SEMICOLON assignment RPAREN 
      LCURLY workflowStatement+ RCURLY
    ;

checkData
    : CHECK_DATA LPAREN SIMPLE_TEXT RPAREN note?
    ;

subWorkflow
    : SUB_WORKFLOW LPAREN IDENTIFIER RPAREN note?
    ;

parallel
    : PARALLEL LCURLY workflowStatement+ RCURLY note?
    ;

customWidget
    : CUSTOM_WIDGET LPAREN IDENTIFIER RPAREN note?
    ;

apiCall
    : API_CALL LPAREN apiCallConfig RPAREN note?
    ;

escalationPath
    : ESCALATION_PATH LCURLY escalationLevel+ RCURLY note?
    ;

foreachLoop
    : FOREACH LPAREN IDENTIFIER IN expression RPAREN LCURLY workflowStatement+ RCURLY
    ;

humanTask
    : HUMAN_TASK LPAREN (IDENTIFIER | ROLE IDENTIFIER) RPAREN LCURLY 
      taskDescription 
      taskAction* 
      RCURLY
    ;

taskDescription
    : DESCRIPTION COLON MULTILINE_TEXT
    ;

taskAction
    : APPROVE | REJECT | INPUT IDENTIFIER
    ;

updateAssignment
    : IDENTIFIER DOT IDENTIFIER EQUALS expression note?
    ;

escalationLevel
    : LEVEL INT LCURLY escalationTarget escalationCondition? RCURLY note?
    ;

escalationTarget
    : NOTIFY (IDENTIFIER | ROLE IDENTIFIER) note?
    ;

escalationCondition
    : CONDITION COLON conditionExpression note?
    ;

// Other rules (table, form, etc.) would go here...

// Utility rules
note
    : NOTE COLON SIMPLE_TEXT
    ;

dataType
    : SMALLINT | INTEGER | BIGINT | DECIMAL LPAREN INT COMMA INT RPAREN
    | NUMERIC LPAREN INT COMMA INT RPAREN | REAL | DOUBLE PRECISION
    | SERIAL | BIGSERIAL | MONEY | CHAR LPAREN INT RPAREN
    | VARCHAR LPAREN INT RPAREN | TEXT | BYTEA
    | TIMESTAMP (WITH TIME ZONE | WITHOUT TIME ZONE)?
    | DATE | TIME (WITH TIME ZONE | WITHOUT TIME ZONE)?
    | INTERVAL | BOOLEAN | ENUM IDENTIFIER | UUID | XML | JSON | JSONB
    | CIDR | INET | MACADDR | BIT LPAREN INT RPAREN
    | BIT VARYING LPAREN INT RPAREN | TSVECTOR | TSQUERY | UUID
    | INT4RANGE | INT8RANGE | NUMRANGE | TSRANGE | TSTZRANGE | DATERANGE
    | encryptedType | vectorType | graphType | documentType
    ;

encryptedType
    : ENCRYPTED LPAREN SIMPLE_TEXT RPAREN note?
    ;

vectorType
    : VECTOR LPAREN INT RPAREN note?
    ;

graphType
    : GRAPH note?
    ;

documentType
    : DOCUMENT note?
    ;

expression
    : IDENTIFIER
    | IDENTIFIER DOT IDENTIFIER
    | INT
    | SIMPLE_TEXT
    ;

assignment
    : IDENTIFIER EQUALS expression
    ;

apiCallConfig
    : // Define API call configuration here
    ;

notificationConfig
    : // Define notification configuration here
    ;
