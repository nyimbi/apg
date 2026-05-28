grammar DBML2;

// Lexer rules
COMMENT    : '#' ~[\r\n]* -> skip;
SPACE      : [ \t]+ -> skip;
NL         : [\r\n]+ -> skip;
WORD       : [a-zA-Z0-9_]+;
STRING     : '\'' (~'\'' | '\'\'' | '\\\'')* '\'';
COMMA      : ',';

// Parser rules
dbml       : (table | ref)*;
// table      : 'Table' WORD (':' STRING)? '{' column+ '}';
table      : 'Table' (WORD | STRING) (':' STRING)? '{' column+ '}';
ref        : 'Ref' (WORD | STRING) (('<' | '>'|'-')  WORD ('?'?)?)*;
column     : WORD type ('?'?)? (':' STRING)?;
type       : 'int' | 'long' | 'float' | 'double' | 'bool' | 'string' | 'date' | 'time' | 'datetime' | 'text';

/* ref        : 'Ref' WORD '{' (foreign_key | on_delete)* '}'; */
foreign_key: 'ForeignKey' (WORD '.')? WORD '.' WORD (':' WORD)?;
on_delete  : 'OnDelete' ('cascade' | 'restrict' | 'nullify' | 'set_default' | 'set_null');

// Helper rules
DIGIT      : [0-9];
LOWER      : [a-z];
UPPER      : [A-Z];

