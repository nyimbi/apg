grammar DBML1;

/* Lexer rules */

INTEGER: ('0'..'9')+;

FLOAT: ('0'..'9')+ '.' ('0'..'9')* ('E'('+'|'-')? ('0'..'9')+)?;

BOOLEAN: ('true'|'false');

IDENTIFIER: ('a'..'z'|'A'..'Z'|'_'|'@'|'#'|'$'|'0'..'9')+;

STRING: ('"' (~["\\\r\n] | '\\' ["\\/bfnrt] | UNICODE))* '"';

WS: [ \t\r\n]+ -> skip;

/* Parser rules */

dbml: statement+ EOF;

statement: object | relation | enum | config | directive;

object: ('table' | 'ref') IDENTIFIER (STRING)? '{' (property ';')* '}';

relation: 'ref' IDENTIFIER IDENTIFIER '{' (property ';')* '}';

enum: 'enum' IDENTIFIER '{' (enumValue ','?)+ '}' (STRING)?;

config: ('sequence' | 'default') INTEGER;

directive: '#' IDENTIFIER;

property: ('id' | 'pk') | 'default' | 'note' | ('type' IDENTIFIER) | ('ref' IDENTIFIER ('.' IDENTIFIER)?);

enumValue: IDENTIFIER (STRING)?;
