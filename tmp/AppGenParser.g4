grammar AppGenParser;
// Parser rules
// This is a combination of DBML and ERDiagramLanguage optimized for Flask-Appbuilder
// A file can include other files using the Include 'filename'
// Copyright (c) Nyimbi Odero 2023
//o ptions { tokenVoca b=AppGenLexer; }
//o ptions { caseInsensitiv ße = true; }

appgen
    : projectBlock? importDeclaration? (statement)* EOF
    ;

importDeclaration
	: IMPORT '{'? fileNames '}'?
	| '#' INCLUDE '{'? fileNames	'}'?
	;
fileNames
	: impFileName ((',')? impFileName)*
	;
impFileName
	: ident '.ags'
	;

projectBlock
    : PROJECT projectName '{' project_property_list '}'
    ;
projectName : ident;

project_property_list
    : (project_property (',' project_property)*)?
    ;

project_property
    : 'Name' '=' string
    | 'Version' '=' string
    | 'Description' '=' string
    | 'Author' '=' string
    | 'Created' '=' string
    | 'Updated' '=' string
    | 'DB_URI' '=' string
    | deployment
    | language
    | report_specification
    | theme
    | appGenOptions
    ;

appGenOptions
	: GENERATE '=' '{' appGenOption (',' appGenOption)* '}'
	;

appGenOption
	: 'ios'
	| 'web'
	| 'desktop'
	| 'android'
	| 'sql'    //Generate just the sql for the tables
	;

deployment
    : DEPLOYMENT '=' '{' deployment_opt+ '}'
    ;
deployment_opt
    : deploy_optName '=' string
    ;
deploy_optName: string;

language
    : LANGUAGES '=' '{' (string (',' string)*)? '}'
    ;
theme
    : THEME '=' string
    ;

statement
    : (object
    | ref
    | enum
    | config)
    ;
object
    : (table
    | mixin
    | table_group)
    ;
table_group
    : TABLEGROUP tablegroupName '{' tableName (',' tableName)* '}'
    ;
tablegroupName: ident;

table
    : TABLE tableName ('(' mixin_list ')')?  '{' note? column_list  view? '}'
    ;
tableName : ident;

mixin
    : MIXIN mixinName '{' note? column_list view? '}'
    ;
mixinName : ident;
mixin_list: mixinName (',' mixinName)*;

column_list
    : column (',' column)*
    ;

column
    : columnName type ('[' propertyList ']')?
    ;

columnName: ident;

propertyList
    : property (',' property)*
    ;

type
    : 'int' | 'long' | 'float' | 'double'
    | 'bool' | 'string' | 'date' | 'time'
    | 'datetime' | 'text' | 'serial' | 'json'
    | 'blob' | 'point' | 'interval' | 'array'
    | 'file' | 'document' | 'image' | 'audio' | 'video' | 'url'
    | 'calc' | 'code' | 'activity'
    | enumName
    | varchar
    ;

varchar
    : 'varchar' '(' INT ')';

property
    : PK
    | DEFAULT
    | REQUIRED
    | UNIQUE
    | NULLABLE
    | (REF ('<' |'>' | '-' | '<>')? ident ('.' ident)?)
    | MIN '=' INT
    | MAX '=' INT
    | display
    | note
    ;

note
    : 'note' ':' STRING
    ;

display
    : ('display' | 'show') '=' ident
    | ('widget' | 'control') '=' ident
    | 'sequence' '=' INT
    | 'tab' '=' INT
    | 'hint' '=' string
    | 'hide' '=' BOOL
    ;

ref
    : (oneToOne
    | oneToMany
    | manyToOne
    | manyToMany)
    ;

oneToOne
    : REF ident ('.' ident)?  '-'  ident ('.' ident)?
    ;

oneToMany
    : REF ident ('.' ident)? '<'  ident ('.' ident)?
    ;

manyToOne
    : REF ident ('.' ident)?  '>' ident ('.' ident)?
    ;

manyToMany
    : REF ident ('.' ident)?  '<>'  ident ('.' ident)?
    ;

// Types supporter by
enum
    : ENUM enumName '{' enumItems? '}'
    ;
enumName: ident;

enumItems
   : enumItem (',' enumItem)* ','?
   ;
enumItem
	: ident '=' INT
	| ident
	;

config
    : CONFIG '{' (ident '=' string (',' ident '=' string)*)? '}'
    ;

view
    : VIEW viewName '{' view_opt_list '}'
    ;
viewName:  ident;

view_opt_list
    : view_opt (','? view_opt)*
    ;

view_opt
    : v_OptName '=' string
    ;
v_OptName: ident;


report_specification
    : REPORT reportName '{' report_property+ '}'
    ;
reportName: ident;

report_property
    : 'Title' '=' string
    | 'Type' '=' string
    | 'Data' '=' string
    | 'Query' '=' string
    | 'Filters' '=' string
    | 'Options' '=' string
    | 'Height' '=' INT
    | 'Width' '=' INT
    | ident '=' string
    ;

chart_specification
    : CHART ident '{' chart_property+ '}'
    ;

chart_property
    : 'Title' '=' string
    | 'Type' '=' string
    | 'Data' '=' string
    | 'Filters' '=' string
    | 'Options' '=' string
    | 'Height' '=' INT
    | 'Width' '=' INT
    | 'X' '=' INT
    | 'Y' '=' INT
    ;

comment
    : LINE_COMMENT
    | C_STYLE_COMMENT
    | PASCAL_STYLE_COMMENT
    ;

ident
    : IDENT
    ;

string
    : STRING
    ;

//////////////////////////////////////////////////
//////////////////// LEXER //////////////////////

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
WORD        : LETTER (LETTER | DIGIT | '_')+;
IDENT       : (LETTER | '_')(LETTER|DIGIT|'_'|'-')+;




// no leading zeros
//Can use _ and , to separate digits in a integer 1,000 or 1_00 are valid numbers
INT
   : '0' | [1-9] (DIGIT|','|'_')*
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
EXP : E ('+'|'-')? DIGIT+ ;


fragment
IntegerTypeSuffix : (L|I) ;

fragment
FloatTypeSuffix : (F | D | B) ;

HEX 	: '0' X HexDigit+ IntegerTypeSuffix? ;

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
HexDigit : ('0'..'9'|'A'..'F') ;

fragment
EscapeSequence
    :   '\\' (B|'t'|'n'|'f'|'r'|'\''|'\\'|'.'|'o'|
              'x'|A|E|'c'|'D'|S|W|'p'|
              'G'|Z|'Q'|'*'|'['|']'|'('|')'|'$'|'^'|
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

bool
	: (TRUE| FALSE)
	| ('T' | 'F')
    ;


null
	: 'null'
	| 'nil'
	| 'naught'
    ;


DATE_TIME_LITERAL: Bound FullDate T FullTime Bound;

fragment Bound: '"' | '\'';
fragment FullDate: Year '-' Month '-' Day;
fragment Year: DIGIT DIGIT DIGIT DIGIT;
fragment Month: [0][0-9]|[1][0-2];
fragment Day: [0-2][0-9]|[0-3][01];

fragment FullTime
    : PartialTime TimeOffset;

fragment TimeOffset
    : Z | TimeNumOffset;

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

REF: 'ref';
TABLE: 'table';
MIXIN: 'mixin';
TABLEGROUP: 'tablegroup';
REPORT: 'report';
PROJECT: 'project';
VIEW: 'view';
CONFIG: 'config';
GENERATE: 'generate';
DEPLOYMENT: 'deployment';
LANGUAGES: 'languages';
THEME: 'theme';
ENUM: 'enum';
CHART: 'chart';
MIN: 'min';
MAX: 'max';
IMPORT: 'import';
INCLUDE: 'include';
DEFAULT: 'default';
NOW: 'now';
BLOB: 'blob';
FILE: 'file';
PK: 'pk';
REQUIRED: 'required';
UNIQUE: 'uniq';
NULLABLE: 'nullable';
DISPLAY: 'display';
NOTE: 'note';
TRUE: 'true';
FALSE: 'false';
NULL: ('null' | 'nil' | 'naught');
