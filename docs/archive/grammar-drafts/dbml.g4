grammar DBML;

// Parser Rules

dbml
    : (table | enumDefinition | relationship | index | trigger | procedure | rowLevelSecurity)* EOF
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
    | 'ref:' '>' IDENTIFIER '.' IDENTIFIER
    ;

relationship
    : 'Ref' IDENTIFIER '.' IDENTIFIER '>' IDENTIFIER '.' IDENTIFIER
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

// New Constructs

trigger
    : 'Trigger' IDENTIFIER 'on' IDENTIFIER triggerEvent '{' triggerBody '}'
    ;

triggerEvent
    : 'before' | 'after' | 'instead of'
    ;

triggerBody
    : statement+
    ;

procedure
    : 'Procedure' IDENTIFIER '(' parameterList? ')' '{' procedureBody '}'
    ;

parameterList
    : parameter (',' parameter)*
    ;

parameter
    : IDENTIFIER dataType
    ;

procedureBody
    : statement+
    ;

rowLevelSecurity
    : 'RLS' IDENTIFIER 'on' IDENTIFIER '{' policy+ '}'
    ;

policy
    : 'Policy' IDENTIFIER '(' 'for' rlsEvent 'using' '(' condition ')' 'with check' '(' condition ')' ')'
    ;

rlsEvent
    : 'all' | 'select' | 'insert' | 'update' | 'delete'
    ;

condition
    : expression
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

