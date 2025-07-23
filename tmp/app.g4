grammar app;

// Parser Rules

app
    : (table | enumDefinition | relationship | index | role | trigger | script | workflow | masterDetailForm | wizard | formStorage)* EOF
    ;

table
    : 'Table' IDENTIFIER '{' column+ '}'
    ;

column
    : IDENTIFIER dataType columnAttributes?
    ;

dataType
    : 'int'
    | 'varchar'
    | 'text'
    | 'timestamp'
    | 'decimal'
    | 'boolean'
    | IDENTIFIER // for enum types
    ;

columnAttributes
    : '[' columnAttribute (',' columnAttribute)* ']'
    ;

columnAttribute
    : 'pk'
    | 'increment'
    | 'unique'
    | 'not null'
    | 'default:' expression
    | 'ref:' relationshipType  IDENTIFIER '.' IDENTIFIER
    ;

relationship
    : 'Ref' IDENTIFIER '.' IDENTIFIER relationshipType IDENTIFIER '.' IDENTIFIER
    ;

relationshipType
    : '>'   // one-to-many
    | '-'   // one-to-one
    | '<'   // many-to-one
    | '<>'  // many-to-many
    ;

index
    : 'Table' IDENTIFIER '{' indexes '}'
    ;

indexes
    : 'indexes' '{' indexDefinition+ '}'
    ;

indexDefinition
    : '(' IDENTIFIER (',' IDENTIFIER)* ')' indexAttributes?
    ;

indexAttributes
    : '[' indexAttribute (',' indexAttribute)* ']'
    ;

indexAttribute
    : 'unique'
    ;

enumDefinition
    : 'Enum' IDENTIFIER '{' enumValue+ '}'
    ;

enumValue
    : IDENTIFIER
    ;

role
    : 'Role' IDENTIFIER '{' permission+ '}'
    ;

permission
    : 'Permission' action 'on' IDENTIFIER
    ;

action
    : 'read'
    | 'write'
    | 'delete'
    ;

trigger
    : 'Trigger' IDENTIFIER 'on' IDENTIFIER triggerEvent '{' triggerBody '}'
    ;

triggerEvent
    : 'before'
    | 'after'
    | 'instead of'
    ;

triggerBody
    : statement+
    ;

script
    : 'Script' IDENTIFIER scriptLang '{' scriptBody '}'
    ;

scriptLang
    : 'zsh'
    | 'bash'
    | 'csh'
    | 'python'
    ;

scriptBody
    : statement+
    ;

workflow
    : 'Workflow' IDENTIFIER '{' workflowStep+ '}'
    ;

workflowStep
    : 'Step' IDENTIFIER 'do' '{' workflowStatement+ '}'
    ;

workflowStatement
    : statement
    | 'log' '(' STRING ')'
    ;

masterDetailForm
    : 'MasterDetailForm' IDENTIFIER 'master' IDENTIFIER 'details' '{' detailComponent+ '}'
    ;

detailComponent
    : 'Detail' IDENTIFIER ('exclude' '(' excludeFields ')')?
    ;

excludeFields
    : IDENTIFIER (',' IDENTIFIER)*
    ;

wizard
    : 'Wizard' IDENTIFIER 'store' 'in' '{' IDENTIFIER (',' IDENTIFIER)* '}' '{' wizardStep+ '}'
    ;

wizardStep
    : 'Step' IDENTIFIER '{' wizardStatement+ '}'
    ;

wizardStatement
    : statement
    | 'form' '(' formDefinition ')'
    ;

formDefinition
    : IDENTIFIER '(' fieldDefinition (',' fieldDefinition)* ')'
    ;

fieldDefinition
    : IDENTIFIER dataType columnAttributes?
    ;

formStorage
    : 'FormStorage' IDENTIFIER 'form' IDENTIFIER 'save' 'for' 'later'
    ;

// Statements and Expressions

statement
    : expression ';'
    ;

expression
    : IDENTIFIER
    | IDENTIFIER '.' IDENTIFIER
    | INT
    | STRING
    ;

// Lexer Rules

IDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

INT
    : [0-9]+
    ;

STRING
    : '\'' ('\\' . | ~('\\' | '\''))* '\''
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

