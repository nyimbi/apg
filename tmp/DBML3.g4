grammar DBML3;

// Lexer rules
COMMENT    : '#' ~[\r\n]* -> skip;
SPACE      : [ \t]+ -> skip;
NL         : [\r\n]+ -> skip;
WORD       : [a-zA-Z0-9_]+;
ID         : [a-zA-Z_][a-zA-Z0-9_]+;
STRING     : '\'' (~'\'' | '\'\'' | '\\\'')* '\'';
COMMA      : ',';

// Parser rules
dbml        : statement* EOF;
statement   : (object| ref |enum | config | directive);
object      : (table | mixin);
table       : 'Table' ID '(' mixin+')'   (':' STRING)? '{' column+ '}';
mixin       : 'Mixin' ID '{' column+ '}';
column      : ID type ('?'?) '['property+ ']' ;
type        : 'int' | 'long' | 'float' | 'double' | 'bool' | 'string' | 'date' | 'time' | 'datetime' | 'text' | 'serial' | 'json' | 'blob' | 'file' | 'point' | 'image';
property    : ('id' | 'pk') | 'default' |'required'|'note'  | ('ref' ID ('.' ID)?);
ref         : 'Ref' ID ('.' ID)? ('<' | '>'|'-'| '*' ) ID ('.' ID)?;
enum        : 'Enum' ID '{' (ID (',' ID)*)? '}';
config      : 'Config' '{' (WORD '=' STRING (',' WORD '=' STRING)*)? '}';
directive   : '@' WORD (':' STRING)?;

// Helper rules
DIGIT      : [0-9];
LOWER      : [a-z];
UPPER      : [A-Z];
FLOAT      : [0-9]+ '.' [0-9]* ('E'('+'|'-')? ('0'..'9')+)?;
INT        : DIGIT+;
BOOLEAN    : ('true'|'false');

