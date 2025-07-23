grammar apg;

options {
    language = Python3;
}

// ========================================
// ULTRA-TERSE APG v10 GRAMMAR
// Unified syntax for: agents, robots, sensors, databases, workflows, ERP
// Design principle: One pattern fits all domains
// ========================================

program
    : entity* EOF
    ;

// UNIVERSAL ENTITY PATTERN: type name { config; behavior; }
entity
    : entity_type IDENTIFIER '{' entity_body '}' ';'?
    ;

entity_type
    : 'agent' | 'robot' | 'sensor' | 'camera' | 'actuator' | 'drone' 
    | 'chat' | 'llm' | 'db' | 'table' | 'biz' | 'flow' | 'rule' 
    | 'report' | 'form' | 'erp' | 'protocol' | 'chain' | 'master'
    | 'auto_system' | 'sense' | 'deploy'
    // OSINT and Intelligence Extensions
    | 'source' | 'intel' | 'analyze' | 'graph' | 'pattern' | 'validate'
    | 'fuse' | 'verify' | 'detect' | 'profile' | 'map' | 'score'
    | 'share' | 'comply' | 'protect' | 'ethics' | 'opsec' | 'fairness'
    | 'standards' | 'geo' | 'track' | 'temporal' | 'context' | 'correlate'
    | 'hunt' | 'monitor'
    ;

entity_body
    : entity_member*
    ;

entity_member
    : config_item
    | behavior_item
    | annotation
    | method_def
    | nested_entity
    ;

// CONFIGURATION - key: value pairs
config_item
    : IDENTIFIER ':' value_expr ';'
    ;

value_expr
    : simple_value
    | list_value  
    | cascade_value
    | reference_value
    | lambda_expr
    ;

simple_value
    : STRING | NUMBER | BOOLEAN | IDENTIFIER
    | '$' IDENTIFIER                          // Environment variable
    ;

list_value
    : '[' (value_expr (',' value_expr)*)? ']'
    | value_expr (',' value_expr)+           // Compact list: a,b,c
    ;

cascade_value
    : value_expr ('->' value_expr)+          // Fallback chain: gpt4->claude3->llama
    ;

reference_value
    : IDENTIFIER ('.' IDENTIFIER)*           // Object reference: user.location
    | IDENTIFIER '*'                         // Collection: cameras*
    ;

// BEHAVIORS - @annotation or method calls
behavior_item
    : annotation
    | method_call ';'
    | flow_definition
    ;

annotation
    : '@' IDENTIFIER (':' annotation_body)?
    ;

annotation_body
    : simple_value
    | method_call
    | '{' entity_member* '}'
    ;

// METHOD DEFINITIONS - ultra compact
method_def
    : IDENTIFIER '(' param_list? ')' ('{' statement* '}' | '=>' expression) ';'?
    ;

param_list
    : IDENTIFIER (',' IDENTIFIER)*
    ;

// STATEMENTS - terse action syntax
statement
    : assignment
    | method_call ';'
    | control_flow
    | minion_command
    ;

assignment
    : IDENTIFIER '=' expression ';'
    | IDENTIFIER '+=' expression ';'         // Increment
    | IDENTIFIER '<<' expression ';'         // Append
    ;

method_call
    : target=expression '.' method=IDENTIFIER '(' args? ')'
    | IDENTIFIER '(' args? ')'
    ;

args
    : expression (',' expression)*
    ;

// CONTROL FLOW - minimal syntax
control_flow
    : 'if' '(' expression ')' statement ('else' statement)?
    | 'for' IDENTIFIER 'in' expression statement  
    | 'while' '(' expression ')' statement
    | 'when' ':' expression '->' statement    // Event-driven
    | 'then' ':' statement                    // Rule consequence
    ;

// EXPRESSIONS - supporting all operations
expression
    : primary
    | expression '.' IDENTIFIER               // Property access
    | expression '(' args? ')'               // Method call
    | expression op=('*'|'/'|'%') expression // Arithmetic
    | expression op=('+'|'-') expression     
    | expression op=('=='|'!='|'<'|'>'|'<='|'>=') expression // Comparison
    | expression op=('&&'|'||') expression   // Logic
    | expression '->' expression             // Pipeline
    | expression '|' expression              // Union/OR
    | expression '&' expression              // Intersection/AND
    ;

primary
    : STRING | NUMBER | BOOLEAN
    | IDENTIFIER
    | '$' IDENTIFIER                         // Environment variable
    | '(' expression ')'
    | '[' (expression (',' expression)*)? ']' // List literal
    | '{' (key_value (',' key_value)*)? '}'  // Dict literal
    ;

key_value
    : (IDENTIFIER | STRING) ':' expression
    ;

// FLOWS - workflow definitions
flow_definition
    : flow_step ('->' flow_step)*
    ;

flow_step
    : IDENTIFIER
    | IDENTIFIER '(' args? ')'
    ;

// MINION COMMANDS - universal interface
minion_command
    : target=expression '.' command=minion_verb '(' args? ')' ';'
    | '@' scope=minion_scope '(' IDENTIFIER ')' command=minion_verb '(' args? ')' ';'
    ;

minion_verb
    : 'do' | 'get' | 'set' | 'watch' | 'report' | 'help'
    ;

minion_scope
    : 'all' | 'nearby' | 'type' | 'group'
    ;

// NESTED ENTITIES - for composition
nested_entity
    : entity_type IDENTIFIER '{' entity_body '}'
    ;

// LAMBDA EXPRESSIONS - for inline behavior
lambda_expr
    : '(' param_list? ')' '=>' expression
    | IDENTIFIER '=>' expression             // Single parameter
    ;

// ========================================
// LEXER RULES - Ultra minimal
// ========================================

// Identifiers
IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]*;

// Numbers
NUMBER: [0-9]+ ('.' [0-9]+)?;

// Strings - simplified
STRING: 
    '"' (~["\r\n] | '""')* '"'
    | "'" (~['\r\n] | "''")* "'"
    ;

// Booleans
BOOLEAN: 'true' | 'false' | 'yes' | 'no' | 'on' | 'off';

// Comments
COMMENT: '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .*? '*/' -> skip;

// Whitespace
WS: [ \t\r\n]+ -> skip;

// Special characters for terse syntax
ARROW: '->';
PIPE: '|';
AMP: '&';
AT: '@';
DOLLAR: '$';
STAR: '*';
PLUS_EQ: '+=';
APPEND: '<<';

// Operators
EQ: '==';
NE: '!=';
LE: '<=';
GE: '>=';
LT: '<';
GT: '>';
AND: '&&';
OR: '||';
ASSIGN: '=';
PLUS: '+';
MINUS: '-';
MULT: '*';
DIV: '/';
MOD: '%';

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