lexer grammar AppGenLexer;


options { caseInsensitive = true; }

fragment A
   : ('a' | 'A')
   ;


fragment B
   : ('b' | 'B')
   ;


fragment C
   : ('c' | 'C')
   ;


fragment D
   : ('d' | 'D')
   ;


fragment E
   : ('e' | 'E')
   ;


fragment F
   : ('f' | 'F')
   ;


fragment G
   : ('g' | 'G')
   ;


fragment H
   : ('h' | 'H')
   ;


fragment I
   : ('i' | 'I')
   ;


fragment J
   : ('j' | 'J')
   ;


fragment K
   : ('k' | 'K')
   ;


fragment L
   : ('l' | 'L')
   ;


fragment M
   : ('m' | 'M')
   ;


fragment N
   : ('n' | 'N')
   ;


fragment O
   : ('o' | 'O')
   ;


fragment P
   : ('p' | 'P')
   ;


fragment Q
   : ('q' | 'Q')
   ;


fragment R
   : ('r' | 'R')
   ;


fragment S
   : ('s' | 'S')
   ;


fragment T
   : ('t' | 'T')
   ;


fragment U
   : ('u' | 'U')
   ;


fragment V
   : ('v' | 'V')
   ;


fragment W
   : ('w' | 'W')
   ;


fragment X
   : ('x' | 'X')
   ;


fragment Y
   : ('y' | 'Y')
   ;


fragment Z
   : ('z' | 'Z')
   ;

// Lexer rules
fragment LETTER
    : [a-zA-Z]
    ;

fragment DIGIT
    : [0-9]
    ;


// Lexer tokens
COMMA           : ',';
COLON           : ':';
SEMI_COLON      : ';';
EQUALS          : '=';
L_ANGLE_BRACKET : '<';
R_ANGLE_BRACKET : '>';
L_PARENTHESES   : '(';
R_PARENTHESES   : ')';
L_BRACKET       : '[';
R_BRACKET       : ']';
HASH_SYMBOL     : '#';
L_BRACE         : '{';
R_BRACE         : '}';
ASTERISK        : '*';
DOT             : '.';

// Whitespace and comments
EOL                 : [\r\n\u2028\u2029]+;

LINE_COMMENT        : '//' ~[\r\n]*     -> channel(HIDDEN);
C_STYLE_COMMENT     : '/*' .*? '*/'     -> channel(HIDDEN);
PASCAL_STYLE_COMMENT: '#' ~[(\r)? \n]*  -> skip;

WS          : [ \t\u000C]+      -> skip;
NL          : '\r'? '\n'        -> skip;
SPACE       : [ \t]+ -> skip;
WORD        : [a-zA-Z0-9_]+;
IDENT       : [a-zA-Z_][a-zA-Z0-9_-]+;




// no leading zeros
//Can use _ and , to separate digits in a integer 1,000 or 1_00 are valid numbers
INT
   : '0' | [1-9] ([0-9]|','|'_')*
   ;

NUMBER
   : '-'? INT ('.' DIGIT+)? EXP?
   | '-'? INT EXP
   | '-'? INT
   ;

FLOAT
    :   ('0'..'9')+ '.' ('0'..'9')* EXP? FloatTypeSuffix?
    |   '.' ('0'..'9')+ EXP? FloatTypeSuffix?
    |   ('0'..'9')+ EXP FloatTypeSuffix?
    |   ('0'..'9')+ FloatTypeSuffix
    ;

fragment
EXP : ('e'|'E') ('+'|'-')? DIGIT+ ;


fragment
IntegerTypeSuffix : ('l'|'L'|'I') ;

fragment
FloatTypeSuffix : ('f'|'F'|'d'|'D'|'B') ;

HEX 	: '0' ('x'|'X') HexDigit+ IntegerTypeSuffix? ;

DECIMAL	: INT IntegerTypeSuffix? ;

STRING
    :  ('"' ( EscapeSequence | ~('\\'|'"') )* '"')
    |  ('\'' ( EscapeSequence | ~('\\'|'\'') )* '\'')
    { // Semantic Action
        setText( normalizeString( getText() ) );
    }
    ;

TIME_INTERVAL
    : (('0'..'9')+ 'd') (('0'..'9')+ 'h')?(('0'..'9')+ 'm')?(('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'h') (('0'..'9')+ 'm')?(('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'm') (('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 's') (('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'ms'?)
    ;

fragment
HexDigit : ('0'..'9'|'a'..'f'|'A'..'F') ;

fragment
EscapeSequence
    :   '\\' ('b'|'B'|'t'|'n'|'f'|'r'|'\''|'\\'|'.'|'o'|
              'x'|'a'|'e'|'c'|'d'|'D'|'s'|'S'|'w'|'W'|'p'|'A'|
              'G'|'Z'|'z'|'Q'|'E'|'*'|'['|']'|'('|')'|'$'|'^'|
              '{'|'}'|'?'|'+'|'-'|'&'|'|')
    |   UnicodeEscape
    |   OctalEscape
    ;

fragment
OctalEscape
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

fragment
UnicodeEscape
    :   '\\' 'u' HexDigit HexDigit HexDigit HexDigit
    ;

BOOL
	: ('true'|'false')
	| ('T' | 'F')
	| ('True'|'False')
    ;


NULL
	: 'null'
	| 'nil'
	| 'naught'
    ;


DATE_TIME_LITERAL: Bound FullDate 'T' FullTime Bound;

fragment Bound: '"' | '\'';
fragment FullDate: Year '-' Month '-' Day;
fragment Year: DIGIT DIGIT DIGIT DIGIT;
fragment Month: [0][0-9]|[1][0-2];
fragment Day: [0-2][0-9]|[0-3][01];

fragment FullTime
    : PartialTime TimeOffset;

fragment TimeOffset
    : 'Z' | TimeNumOffset;

fragment TimeNumOffset
    : '-' [01][0-2] (':' (HalfHour))?
    | '+' [01][0-5] (':' (HalfHour | [4][5]))?
    ;
fragment HalfHour: [0][0] | [3][0];

fragment PartialTime
    : [0-2][0-3] ':' Sixty ':' Sixty ('.' [0-9]*)?;

fragment Sixty: [0-5] DIGIT;


VersionLiteral
  : [0-9]+ '.' [0-9]+ ('.' [0-9]+)? ;


//keywords

REF: R E F;
TABLE: T A B L E;
MIXIN: M I X I N;
TABLEGROUP: T A B L E G R O U P;
REPORT: R E P O R T;
PROJECT: P R O J E C T;
VIEW: V I E W;
CONFIG: C O N F I G;
GENERATE: G E N E R A T E;
DEPLOYMENT: D E P L O Y M E N T;
LANGUAGES: L A N G U A G E S;
THEME: T H E M E;
ENUM: E N U M;
CHART: C H A R T;
MIN: M I N;
MAX: M A X;
IMPORT: I M P O R T;
INCLUDE: I N C L U D E;
DEFAULT: D E F A U L T;
NOW: N O W;
BLOB: B L O B;
FILE: F I L E;
PK: P K;
REQUIRED: R E Q U I R E D;
UNIQUE: U N I Q U E;
NULLABLE: N U L L A B L E;
