grammar AppGenq;
// Parser rules
// This is a combination of DBML and ERDiagramLanguage optimized for Flask-Appbuilder
// A file can include other files using the Include 'filename'
// Copyright (c) Nyimbi Odero 2023

appgen
    : projectBlock?
    | importDeclaration?
    | (statement)* EOF
    ;

importDeclaration
	: 'import' '{'? fileNames '}'?
	| '#include' '{'? fileNames	'}'?
	;
fileNames
	: impFileName ((',')? impFileName)*
	;
impFileName
	: ident '.ags'
	;

projectBlock
    : 'project' projectName '{' project_property_list '}'
    ;
projectName : ident;

project_property_list
    : (project_property (',' project_property)*)?
    ;

project_property
    : 'Name' '=' STRING
    | 'Version' '=' STRING
    | 'Description' '=' STRING
    | 'Author' '=' STRING
    | 'Created' '=' STRING
    | 'Updated' '=' STRING
    | 'DB_URI' '=' STRING
    | deployment
    | language
    | report_specification
    | theme
    | appGenOptions
    ;

appGenOptions
	: 'Generate' '=' '{' appGenOption (',' appGenOption)* '}'
	;

appGenOption
	: 'ios'
	| 'web'
	| 'desktop'
	| 'android'
	| 'sql'    //Generate just the sql for the tables
	;

deployment
    : 'Deployment' '=' '{' deployment_opt+ '}'
    ;
deployment_opt
    : deploy_optName '=' STRING
    ;
deploy_optName: STRING;

language
    : 'Languages' '=' '{' (STRING (',' STRING)*)? '}'
    ;
theme
    : 'Theme' '=' STRING
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
    : 'table_group' tablegroupName '{' tableName (',' tableName)* '}'
    ;
tablegroupName: ident;

table
    : 'table' tableName ('(' mixin_list ')')?  '{' note? column_list  view? '}'
    ;
tableName : ident;

mixin
    : 'mixin' mixinName '{' note? column_list view? '}'
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
    | 'blob' | 'file' | 'point' | 'image'| 'interval'
    | enumName
    | varchar
    ;

varchar
    : 'varchar' '(' INT ')';

property
    : 'pk'
    | 'default'
    | 'required'
    | 'unique'
    | 'nullable'
    | ('ref' ('<' |'>' | '-' | '<>')? ident ('.' ident)?)
    | 'min' '=' INT
    | 'max' '=' INT
    | display
    | note
    ;

note
    : 'note' ':' STRING (',')?
    ;

display
    : ('display' | 'show') '=' (ident | STRING)
    | ('widget' | 'control') '=' ident
    | 'sequence' '=' INT
    | 'tab' '=' INT
    | 'hint' '=' STRING
    | 'hide' '=' BOOL
    ;

ref
    : (oneToOne
    | oneToMany
    | manyToOne
    | manyToMany)
    ;

oneToOne
    : 'Ref' tableName? (('.')? columnName)  '-'  tableName? (('.')? columnName)
    ;

oneToMany
    : 'Ref' ident ('.' ident)? '<'  ident (DOT ident)?
    ;

manyToOne
    : 'Ref' ident ('.' ident)?  '>' ident ('.' ident)?
    ;

manyToMany
    : 'Ref' ident ('.' ident)?  '<>'  ident ('.' ident)?
    ;

// Types supporter by
enum
    : 'Enum' enumName '{' enumItems? '}'
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
    : 'Config' '{' (WORD '=' STRING (',' WORD '=' STRING)*)? '}'
    ;

view
    : 'view' viewName '{' view_opt_list '}'
    ;
viewName:  ident;

view_opt_list
    : view_opt (','? view_opt)*
    ;

view_opt
    : v_OptName '=' STRING
    ;
v_OptName: ident;


report_specification
    : 'Report' reportName '{' report_property+ '}'
    ;
reportName: ident;

report_property
    : 'Title' '=' STRING
    | 'Type' '=' STRING
    | 'Data' '=' STRING
    | 'Query' '=' STRING
    | 'Filters' '=' STRING
    | 'Options' '=' STRING
    | 'Height' '=' INT
    | 'Width' '=' INT
    | 'X' '=' INT
    | 'Y' '=' INT
    ;

chart_specification
    : 'Chart' ident '{' chart_property+ '}'
    ;

chart_property
    : 'Title' '=' STRING
    | 'Type' '=' STRING
    | 'Data' '=' STRING
    | 'Filters' '=' STRING
    | 'Options' '=' STRING
    | 'Height' '=' INT
    | 'Width' '=' INT
    | 'X' '=' INT
    | 'Y' '=' INT
    ;

ident: IDENTIFIER;
string: STRING;
// Lexer rules

// Whitespace and comments
WS:                 [ \t\u000C]+ -> skip;
NL: 				'\r'? '\n' -> skip;
LINE_COMMENT:       '//' ~[\r\n]*    -> channel(HIDDEN);
C_COMMENT   :           '/*' .*? '*/'    -> channel(HIDDEN);
P_COMMENT     : '#' ~[(\r)? \n]* -> skip;
//COMMENT
//    : LINE_COMMENT
//    | C_COMMENT
//    | P_COMMENT  -> channel(HIDDEN);

SPACE       : [ \t]+ -> skip;
WORD        : [a-zA-Z0-9_]+;
IDENTIFIER  : [a-zA-Z_][a-zA-Z0-9_-]+;
COMMA       : ',';
DOT         : '.';

EOL
   : [\r\n\u2028\u2029]+
   ;

NUMBER
   : '-'? INT ('.' DIGIT+)? EXP?
   | '-'? INT EXP
   | '-'? INT
   ;

DIGIT
	: [0-9]
	;

// no leading zeros
INT
   : '0' | [1-9] ([0-9]|','|'_')* //Can use _ and , to separate digits in a integer 1,000 or 1_00 are valid numbers
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
FloatTypeSuffix : ('f'|'F'|'d'|'D'|'B') ;

HEX 	: '0' ('x'|'X') HexDigit+ IntegerTypeSuffix? ;

DECIMAL	: INT IntegerTypeSuffix? ;

fragment
IntegerTypeSuffix : ('l'|'L'|'I') ;

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

