lexer grammar apglexer;

// Keywords
IMPORT : 'import';
TEMPLATE : 'template';
VERSION : 'version';
WORKFLOW : 'workflow';
STEP : 'step';
PARALLEL : 'parallel';
SEQUENCE : 'sequence';
VARIABLES : 'variables';
PRE_EXECUTION : 'pre_execution';
POST_EXECUTION : 'post_execution';
INITIATOR : 'initiator';
DESCRIPTION : 'description';
DEADLINE : 'deadline';
CRON : 'cron';
INPUTS : 'inputs';
OUTPUTS : 'outputs';
DEPENDS_ON : 'depends_on';
FORM : 'form';
ASSIGN_TO : 'assignto';
RESPONSIBLE : 'responsible';
ACCOUNTABLE : 'accountable';
CONSULTED : 'consulted';
INFORMED : 'informed';
ESCALATE_TO : 'escalateto';
CONDITION : 'condition';
EVENT_TRIGGER : 'eventtrigger';
ON_ERROR : 'onerror';
RETRIES : 'retries';
RETRY_DELAY : 'retrydelay';
TIMEOUT : 'timeout';
SHOW_FORM : 'showform';
ASSIGN_TASK : 'assigntask';
SET_DEADLINE : 'setdeadline';
SEND_NOTIFICATION : 'sendnotification';
UPDATE_RECORD : 'updaterecord';
EXECUTE_SCRIPT : 'executescript';
CHECK_DATA : 'checkdata';
SUB_WORKFLOW : 'subworkflow';
CUSTOM_WIDGET : 'customwidget';
API_CALL : 'apicall';
ESCALATION_PATH : 'escalationpath';
HUMAN_TASK : 'human_task';
LEVEL : 'level';
NOTIFY : 'notify';
ROLE : 'role';
NOTE : 'note';
ON : 'on';

// Control flow
IF : 'if';
ELSE : 'else';
ELSE_IF : 'elseif';
WHILE : 'while';
FOR : 'for';
FOREACH : 'foreach';
IN : 'in';

// Data types
SMALLINT : 'smallint';
INTEGER : 'integer';
BIGINT : 'bigint';
DECIMAL : 'decimal';
NUMERIC : 'numeric';
REAL : 'real';
DOUBLE : 'double precision';
SERIAL : 'serial';
BIGSERIAL : 'bigserial';
MONEY : 'money';
CHAR : 'char';
VARCHAR : 'varchar';
TEXT : 'text';
BYTEA : 'bytea';
TIMESTAMP : 'timestamp';
DATE : 'date';
TIME : 'time';
INTERVAL : 'interval';
BOOLEAN : 'boolean';
ENUM : 'enum';
UUID : 'uuid';
XML : 'xml';
JSON : 'json';
JSONB : 'jsonb';
CIDR : 'cidr';
INET : 'inet';
MACADDR : 'macaddr';
BIT : 'bit';
TSVECTOR : 'tsvector';
TSQUERY : 'tsquery';
INT4RANGE : 'int4range';
INT8RANGE : 'int8range';
NUMRANGE : 'numrange';
TSRANGE : 'tsrange';
TSTZRANGE : 'tstzrange';
DATERANGE : 'daterange';
ENCRYPTED : 'encrypted';
VECTOR : 'vector';
GRAPH : 'graph';
DOCUMENT : 'document';

// Time zone specifiers
WITH : 'with';
WITHOUT : 'without';
TIME_ZONE : 'time zone';

// Bit varying
VARYING : 'varying';

// Time units
SECONDS : 'seconds';
MINUTES : 'minutes';
HOURS : 'hours';

// Error handling
CONTINUE : 'continue';
ABORT : 'abort';
RETRY : 'retry';

// Human task actions
APPROVE : 'approve';
REJECT : 'reject';
INPUT : 'input';

// Symbols
LCURLY : '{';
RCURLY : '}';
LBRACK : '[';
RBRACK : ']';
LPAREN : '(';
RPAREN : ')';
COMMA : ',';
COLON : ':';
SEMICOLON : ';';
DOT : '.';
EQUALS : '=';
HASH : '#';

// Operators
AND : '&&';
OR : '||';
NOT : '!';
EQ : '==';
NEQ : '!=';
LT : '<';
LTE : '<=';
GT : '>';
GTE : '>=';

// Literals
IDENTIFIER : [a-zA-Z_][a-zA-Z0-9_]*;
INT : [0-9]+;
FLOAT : [0-9]+ '.' [0-9]+ ([eE] [+-]? [0-9]+)?;
BOOL : 'true' | 'false';
SIMPLE_TEXT : '\'' ( ~['\\] | '\\' . )* '\'';
MULTILINE_TEXT : '"""' .*? '"""';
SEMVER : INT '.' INT '.' INT ('-' [a-zA-Z0-9-]+)?;
CRON_EXPRESSION : [a-zA-Z0-9_/*,-]+;

// URLs
URL : ('http' | 'https') '://' (LETTER | DIGIT | '.' | '-' | '_')+ ('/' (LETTER | DIGIT | '.' | '-' | '_' | '%' | ':' | '@' | '&' | '=' | '+' | '$' | ',' | '?' | '#' | '!' | '(' | ')' | '*' | '~')*)?;

// Email addresses
EMAIL : [a-zA-Z0-9._%+-]+ '@' [a-zA-Z0-9.-]+ '.' [a-zA-Z]{2,6};

// Whitespace and comments
WS : [ \t\r\n]+ -> skip;
SINGLE_LINE_COMMENT : '//' ~[\r\n]* -> skip;
MULTI_LINE_COMMENT : '/*' .*? '*/' -> skip;

// Fragments
fragment LETTER : [a-zA-Z];
fragment DIGIT : [0-9];

// Error handling
ERROR_CHAR : . ;
