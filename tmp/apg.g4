grammar apg;

options {
 language = Python3;
}

// Lexer TOKENS
ADD             : '+';
AND             : '&&';
COMMA           : ',';
COLON           : ':';
DIV             : '/';
DOT             : '.';
EQ              : '=';
GT              : '>';
GTE             : '>=';
L_CURLY         : '{';
L_PAR           : '(';
L_SQUARE        : '[';
LT              : '<';
LTE             : '<=';
M2M             : ('<>' | MUL);
MUL             : '*';
NEQ             : '!=';
OR              : '||';
R_CURLY         : '}';
R_PAR           : ')';
R_SQUARE        : ']';
SEMI_COLON      : ';';
SUB             : '-';

// Whitespace and Comments
C_LINE_COMMENT      : '//' ~[\r\n]*     -> channel(HIDDEN);
C_STYLE_COMMENT     : '/*' .*? '*/'     -> channel(HIDDEN);
NL                  : [\r\n\u2028\u2029]        -> skip;
WS                  : [ \t\r\n\u000C\u00A0]+    -> skip;

fragment Letter
    : [_A-Za-z]
    ;

fragment SpecialChars
    :  [!$&'*+;=?^_`|~]
    ;

fragment Digit
    : [0-9]
    ;

fragment Count
    : [1-9]
    ;

fragment LetterOrDigit
    : Letter
    | Digit
    ;

fragment DirectorySeparators
    :  [\\/] // Both Unix-like and Windows directory separators
    ;

fragment OctalEscape
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

fragment UnicodeEscape
    :   '\\' 'u' HexDigit HexDigit HexDigit HexDigit
    ;

fragment EscapeSequence
    :   '\\' ('b'|'B'|'t'|'n'|'f'|'r'|'\''|'\\'|'.'|'o'|
              'x'|'a'|'e'|'c'|'d'|'D'|'s'|'S'|'w'|'W'|'p'|'A'|
              'G'|'Z'|'z'|'Q'|'E'|'*'|'['|']'|'('|')'|'$'|'^'|
              '{'|'}'|'?'|'+'|'-'|'&'|'|')
    |   UnicodeEscape
    |   OctalEscape
    ;

String
    :  ('"' ( EscapeSequence | ~('\\'|'"') )* '"')
    |  ('\'' ( EscapeSequence | ~('\\'|'\'') )* '\'')
    { // Semantic Action
        setText( normalizeString( getText() ) );
    }
    ;

// INT has no leading zeros
//Can use _ to separate digits in a integer 1000 or 1_000_000 are valid numbers
Int
   : '0' | Count+ (Digit|'_')*
   ;

fragment Exp
    : ('e'|'E') ('+'|'-')? Digit+
    ;

fragment FloatTypeSuffix
    : ('f'|'F'|'d'|'D'|'B')
    ;

fragment HexDigit
    : (Digit |'a'..'f'|'A'..'F')
    ;

fragment IntegerTypeSuffix
    : ('l'|'L'|'I')
    ;

Number
    :   Digit+ '.' Digit* (Exp)? (FloatTypeSuffix)?
    |   '.' Digit+ (Exp)? (FloatTypeSuffix)?
    |   Digit+ Exp (FloatTypeSuffix)?
    |   Digit+ IntegerTypeSuffix
    |   ('0x'|'0X') (HexDigit)+ (IntegerTypeSuffix)?
    { // Semantic Action
        setText( normalizeNumber( getText() ) );
    }
    ;

fragment True
    : ('T' | 'True' | 'true')
    ;

fragment False
    : ('F' | 'False' | 'false')
    ;

Bool
	: (True|False)
    ;

HEX 	: '0' ('x'|'X') HexDigit+ IntegerTypeSuffix? ;

DECIMAL	: Int IntegerTypeSuffix? ;

// Now to define time and dates, complex
TIME_INTERVAL
    : (('0'..'9')+ 'd') (('0'..'9')+ 'h')?(('0'..'9')+ 'm')?(('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'h') (('0'..'9')+ 'm')?(('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'm') (('0'..'9')+ 's')?(('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 's') (('0'..'9')+ 'ms'?)?
    | (('0'..'9')+ 'ms'?)
    ;

DATE_TIME_LITERAL: Bound FullDate 'T' FullTime Bound;

fragment Bound: '"' | '\'';
fragment FullDate: Year '-' Month '-' Day;
fragment Year: Digit Digit (Digit Digit)?;  // EITHER 2 or 4 digits
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

fragment Sixty: [0-5] Digit;

VersionLiteral
  : Digit+ '.' Digit+ ('.' Digit+)? ;

//KEYWORDS
ABSTRACT    : 'abstract';
ASYNC       : 'async';
AWAIT       : 'await';
BOOL        : 'bool';
BREAK       : 'break';
CASE        : 'case';
CATCH       : 'catch';
CHAR        : 'char';
CLASS       : 'class';
CONST       : 'const';
CONTINUE    : 'continue';
DEFAULT     : 'default';
DOUBLE      : 'double';
DYNAMIC     : 'dynamic';
ELSE        : 'else';
ENUM        : 'enum';
EXTENDS     : 'extends';
FALSE       : 'false';
FINAL       : 'final';
FINALLY     : 'finally';
FLOAT       : 'float';
FOR         : 'for';
GLOBAL      : 'global';
IF          : 'if';
IMPLEMENTS  : 'implements';
INTERFACE   : 'interface';
NEW         : 'new';
NOT         : 'not';
PACKAGE     : 'package';
PRIVATE     : 'private';
//PUBLIC      : 'public';
//REGISTER    : 'register';
RESTRICT    : 'restrict';
RETURN      : 'return';
STATIC      : 'static';
SUPER       : 'super';
SWITCH      : 'switch';
THIS        : 'this';
THROW       : 'throw';
TRUE        : 'true';
TRY         : 'try';
VOID        : 'void';
WHILE       : 'while';
XOR         : 'xor';

//AppGen Specific Keywords
ANDROID : 'android';
BLOB    : 'blob';
CACHE   : 'cache';
CHART   : 'chart';
CHECK   : 'check';
CONFIG  : 'config';
COUCHDB : 'couchdb';
DBFUNC  : 'dbfunc';
DBVIEW  : 'dbview';
DECR    : ('decr' | 'decrement' | '--');
DEFAULT_EXPR: 'default-expression' ;
DEPLOYMENT: 'deployment';
DESKTOP : 'desktop';
DIALECT : 'dialect';
DISPLAY : 'display';
FILE    : 'file';
GENERATE: 'generate';
IMPORT  : '#import' ;
INCLUDE : '#include' ;
INCR    : ('incr' | 'increment' |  '++');
IOS     : 'ios';
LANGUAGES: 'languages';
MAX     : 'max';
MIN     : 'min';
MIXIN   : 'mixin';
MODE    : 'mode';
MONGODB : 'mongodb';
MSSQL   : 'mssql';
MYSQL   : 'mysql';
NOT_NULL: 'not null';
//not_null: NOT_NULL;
NOTE    : 'note';
NOW     : 'now';
NULL    : ('null' | 'nil' | 'naught');
NULLABLE: 'nullable';
ORACLE  : 'oracle';
OS      : 'os';
PGSQL   : 'pgsql';
PK_LONG : 'primary_key';
PK_SHORT: 'pk';
PROJECT : 'project';
REF     : 'ref';
REPORT  : 'report';
REQUIRED: 'required';
RULE    : 'rule';
SQL     : 'sql';
SQLITE  : 'sqlite';
TABLE   : 'table';
TBLGROUP: 'tablegroup';
THEME   : 'theme';
TODAY   : 'today';
TOMORROW: 'tomorrow';
TSERIES : ('tseries' | 'time_series' | 'timeseries');   //so that we can handle
UNIQ    : ('uniq' | '!' | 'unique') ;
VARCHAR : 'varchar';
VERSION : 'version';
VIEW    : 'view';
WEB     : 'web';
YESTERDAY: 'yesterday';


apg
    : importDeclaration? projectBlock?  (statement)+ EOF
    ;

unique: UNIQ;
db
    : PGSQL
    | MYSQL
    | SQLITE
    | ORACLE
    | MSSQL
    | MONGODB
    | COUCHDB
    ;

Ident
    : Letter+ (LetterOrDigit)*
    ;

// Covenience Parser Rules
int: Int;
string
    : String
    | String String
    ;
ident: Ident;
name_attr: ident ;

int_list
    :  int (COMMA int)* COMMA?
    ;

ident_list
    :  ident (COMMA ident)* COMMA?
    ;

string_list
    :  string (',' string)* COMMA?
    ;

option
    : ident  EQ (string | L_CURLY string_list R_CURLY | ident| L_CURLY ident_list R_CURLY| int |L_CURLY int_list R_CURLY)
    ;

option_list
    :  option (COMMA option)* COMMA?
    ;

importDeclaration
	: IMPORT import_file_list
	| INCLUDE import_file_list
	;

import_file_list
    : (ident_list  | string_list)+
    ;

projectBlock
    : PROJECT projectName  L_CURLY project_property_list  R_CURLY
    ;
projectName: name_attr;

project_property_list
    : project_property (COMMA project_property)* COMMA?
    ;

project_property
    : 'project_name' EQ string
    | (VERSION EQ (VersionLiteral | string))?
    | language
    | theme?
    | cloudCfg?
    | authCfg?
    | thirdPartyCfg?
    | perfCfg?
    | versionCfg?
    | pluginCfg?
    | genOptions?
    | option_list?
    ;

cloudCfg
    : 'cloud_cfg' EQ L_CURLY cloud_option_list R_CURLY
    ;
cloud_option_list
    : 'csp' EQ ('AWS' | 'Azure' | 'Linode')
    | option_list
    ;
authCfg
    : 'auth' EQ L_CURLY option_list R_CURLY
    ;
thirdPartyCfg
    : 'third_party_cfg' EQ L_CURLY option_list R_CURLY
    ;
perfCfg
    : 'perf' EQ L_CURLY option_list R_CURLY
    ;
versionCfg
    : 'version_mgmt' EQ L_CURLY option_list R_CURLY
    ;
pluginCfg
    : 'plugins' EQ L_CURLY option_list R_CURLY
    ;
genOptions
    : 'generate' EQ L_SQUARE (appGenTarget)+ R_SQUARE
    ;

appGenTarget
    : IOS
	| WEB
	| DESKTOP
	| ANDROID
	| SQL (DIALECT (db | string))?   //Generate just the sql for the tables
	| string
	;

language
    : LANGUAGES EQ lang_list
    ;
lang_list
    : L_SQUARE string_list R_SQUARE
    ;

theme
    : THEME EQ string
    ;

statement
    : (object | business_rule)+
    ;

object
    : database
    | mixin             // Mixins are global
    | enum_ext          // Enumerations are global
    | table
    | dbview          // Database function, trigges, procedures
    | dbfunc          // Database function, trigges, procedures
    | index_ext       // Externally defined indices
    | ref_ext         // References defined outside the table definition
    ;

database
    : 'database' dbname L_CURLY database_options+ R_CURLY
    ;
dbname: name_attr;

database_options
    : 'dburl' EQ string
    | 'backup-frequency' EQ TIME_INTERVAL
    | 'contains-tables'  EQ string_list
    | 'excludes-tables' EQ string_list
    | option
    ;

mixin
    : MIXIN mixin_name L_CURLY column_list R_CURLY
    ;

mixin_name: name_attr;

table
    : TABLE table_name ('(' mixin_list ')')? L_CURLY (column_list | index_int | note_option)+ R_CURLY
    ;

mixin_list
    : mixin_name (COMMA mixin_name)*
    ;

table_name: name_attr;

column_list
    :   column (COMMA column)* COMMA?
    ;

column
    : column_name COLON data_type (column_option_list)?
    ;

column_name: name_attr;

data_type
    : 'bit'|'bitlist'
    | 'boolean' | 'bool'
    | 'tinyint'| 'smallint'| 'int'| 'integer' | 'bigint'
    | 'double' | 'decimal' | 'float'
    | 'money'| 'smallmoney'
    | 'char'| 'nchar'| varchar | 'nvarchar' | 'string'
    | 'text'| 'mediumtext' | 'longtext' | 'xml' | 'document'
    | 'blob'| 'mediumblob'| 'longblob'
    | 'binary'|'varbinary'
    | 'json'| 'jsonb'
    | 'interval'| 'time'| 'timestamp'| 'timestamptz'| 'timestampltz'| 'datetime'| 'date'
    | 'geography'| 'geometry'| 'point'
    | 'hllsketch'| 'hstore'| 'pseudo_type'|'super'
    | 'serial'| 'smallserial'| 'bigserial'| 'uniqueidentifier'
    | 'rowversion'| 'variant'| 'inet'|
    | 'file' | 'image' | 'object' | 'uuid'
    | enum_name
    | 'array' int_list  'of' data_type
    ;

varchar
    : VARCHAR L_PAR int R_PAR
    ;

column_option_list
    :  '[' column_option (COMMA column_option)* (COMMA)? ']'
    ;

column_option
    : primary_key
    | DEFAULT EQ column_default
    | CACHE ( '(' int ')' )? // Cache this column (lifetime defaults to 300 seconds)
    | DEFAULT_EXPR EQ '(' string ')'
    | INCR | DECR
    | unique
    | NULLABLE
    | NOT_NULL
    | ref_internal   //internal reference
    | enum_internal
    | MIN EQ int
    | MAX EQ int
    | check
    | display_method
    | note_option
    ;
primary_key: PK_SHORT | PK_SHORT;

column_default
    : Number
    | Bool
    | NOW | TODAY | TOMORROW
    | 'CURRENT_DATE' | 'CURRENT_USER'
    | NULL
    | string
    ;

ref_internal //defined in table options
    : REF (ref_name)? COLON ref_type (dbname DOT)? table_name DOT column_name
    ;
ref_ext //Free standing Reference Foreign_Key
    : REF (ref_name)? COLON  table_name DOT (column_name) ref_type table_name DOT (column_name)
    ;

ref_name: name_attr;

ref_type
    : oneToOne
    | oneToMany
    | manyToOne
    | manyToMany
    ;

oneToOne: SUB;
oneToMany: LT;
manyToOne: GT;
manyToMany: M2M;

enum_name: name_attr;

enum_internal  //Defined in table options, anonymous enum
    : ENUM EQ L_CURLY enum_list R_CURLY
    ;

enum_ext  //self standing enums enum thisenum={}
    : ENUM enum_name EQ L_CURLY enum_list R_CURLY
    ;

enum_list
    : enum_item (COMMA enum_item)* COMMA?
    ;

enum_item
    : (enum_idx EQ)? enum_value ('[' note_option ']')?
    ;

enum_idx: int;
enum_value : string ;

check: CHECK L_PAR check_expr R_PAR;
check_expr: string;

display_method  //TODO: Enhance the display method for a field
    : DISPLAY EQ L_CURLY option_list R_CURLY
    ;

note_option
    : NOTE COLON note_value
    ;

note_value
    : string
    ;



dbview
    : DBVIEW view_name L_CURLY ((table_name | view_name) DOT column_name) EQ ((table_name | view_name) DOT column_name) R_CURLY
    ;

view_name: name_attr;


index_ext
    : 'index' (index_name)? 'on' table_name L_SQUARE column_names R_SQUARE ('of type' ('btree'|'gin'|'gist'|'hash'))?
    ;

index_int //For specification inside a table
    : 'indexes'  EQ '{' index_item_list '}'
    ;

index_item_list
    :  index_item (',' index_item)*
    ;

index_item
    :( index_name EQ)? '[' column_names ']' ('of type' ('btree'|'gin'|'gist'|'hash'))?
    ;

column_names
    :  column_name (COMMA column_name)* (COMMA)?
    ;
index_name: name_attr;

dbfunc
    : DBFUNC func_name L_CURLY func_body R_CURLY
    ;

func_name: name_attr;
func_body: string;

/////CRONTAB Schedule //////
schedule: field COLON field COLON field COLON field COLON field;
field: ( int | sched_range | sched_list );
sched_range: int SUB int;
sched_list: int_list;
SP: WS;
//step: field '/' DIGITS;

business_rule: RULE rule_name EQ L_CURLY businessRule R_CURLY;
businessRule: condition actionExpr;

condition
    : ifExpr    COLON                    #IfExpression
    | 'at'      L_PAR schedule R_PAR     #AtTimeExpression      //do at a particluar time
    | 'every'   L_PAR schedule R_PAR     #EveryTimeExpression   // at intevals
    | 'on'      L_PAR event_desc R_PAR   #OnEventExpression     // On occurrence of an event
    ;

ifExpr: IF  L_PAR expr R_PAR ; // ( ELSE '(' actionExpr ')' )?
rule_name: name_attr;
event_desc: string;

actionExpr  : 'do' action_value;
action_value: action_verb ( '(' action_object ')' )?;
action_verb: ident;
action_object: string_list;

//destination: string;
//server_loc: string;

expr
    : ident                            # identExpression
    | literal                          # literalExpr
    | functionCall                     # functionCallExpr
    | L_PAR expr R_PAR                 # nestedExpr
    | expr op=(MUL | DIV) expr         # binaryMultiplicationDiv
    | expr op=(ADD | SUB) expr         # binaryAdditionSubtraction
    | expr booleanOp expr              # booleanCombination
    | expr comparisonOp expr           # binaryComparison
    | SUB expr                         # unaryMinus
    | 'sin'   L_PAR expr R_PAR         # trigonometricSin
    | 'cos'   L_PAR expr R_PAR         # trigonometricCos
    | 'tan'   L_PAR expr R_PAR         # trigonometricTan
    | 'asin'  L_PAR expr R_PAR         # inverseTrigonometricSin
    | 'acos'  L_PAR expr R_PAR         # inverseTrigonometricCos
    | 'atan'  L_PAR expr R_PAR         # inverseTrigonometricTan
    | 'sinh'  L_PAR expr R_PAR         # hyperbolicSine
    | 'cosh'  L_PAR expr R_PAR         # hyperbolicCosine
    | 'tanh'  L_PAR expr R_PAR         # hyperbolicTangent
    | 'asinh' L_PAR expr R_PAR         # inverseHyperbolicSine
    | 'acosh' L_PAR expr R_PAR         # inverseHyperbolicCosine
    | 'atanh' L_PAR expr R_PAR         # inverseHyperbolicTangent
    | 'avg'   L_PAR expr_list R_PAR    # statisticalAverage
    | 'min'   L_PAR expr_list R_PAR    # statisticalMinimum
    | 'max'   L_PAR expr_list R_PAR    # statisticalMaximum
    | 'sum'   L_PAR expr_list R_PAR    # statisticalSum
  ;

expr_list
    : expr (COMMA expr)* COMMA?
    ;

literal: int | FLOAT | string | Bool ;
booleanOp: AND | OR | XOR;
comparisonOp: EQ | NEQ | LT | LTE | GT | GTE;

functionCall: 'exec' function_name ('(' param_list? ')')?;  // params are in square brackets
function_name: name_attr;
param_list:  (string_list | ident_list)+ ;
