grammar appgen;

//TODO: Extend the list of supported data types
//TODO: Tidy up indexes and views on tables
//TODO: Add Event and Notification to Tables/View/Databases
//TODO: Add jhipster style deployment features
//TODO: Add a business rules language
//TODO: Add a singer.io tap interface

appgen
    : (importDeclaration | projectBlock)*  (statement)+ EOF
    ;

importDeclaration
	: '#import'  fileNames
	| '#include'  fileNames
	;

fileNames
	: L_CURLY importFileName ((',')? importFileName)* R_CURLY  //with braces or not
	| importFileName ((',')? importFileName)*
	| L_CURLY R_CURLY  //empty list
	;

importFileName
	:  ident  //filename
	;

projectBlock
    : 'project' projectName  L_CURLY project_property_list R_CURLY
    ;

projectName : name_attr;

project_property_list
    :  (project_property (',' project_property)*)?
    ;

project_property
    : 'project_name' '=' string
    | 'version' EQ (string | VersionLiteral)
    | dev_db_uri EQ string
    | prod_db_uri EQ string
    | option
    | config
    | deployment
    | language
    | report_spec
    | theme
    | appGenOptions
    ;

appGenOptions
	: 'generate' EQ L_CURLY appGenOption_list  R_CURLY
	;

appGenOption_list
    : appGenOption (COMMA appGenOption)* COMMA?
    ;

appGenOption
    : option
    | 'mode' EQ '[' appGenModeOption+ ']'
    ;

appGenModeOption
	: 'ios'
	| 'web'
	| 'desktop'
	| 'android'
	| 'sql' ('dialect' (db | string))?   //Generate just the sql for the tables
	| string
	;

//appGenBackOption
//    : 'monolithic'   //default
//    | 'microsservice_app'
//    | 'microservice_gw'
//    ;

deployment
    : 'deployment' EQ deployment_option_list
    ;

deployment_option_list: option_list;

language
    : 'languages' EQ  lang_list
    ;
lang_list
    : string_list
    ;

theme
    : 'theme' '=' string
    ;

report_spec
    : 'report' report_name L_CURLY report_property_list R_CURLY
    ;
report_name: name_attr;

report_property_list
    : report_property (COMMA report_property)* COMMA?
    ;
report_property
    : option
    ;

chart_specification
    : 'chart' chart_name L_CURLY chart_property_list R_CURLY
    ;
chart_name: name_attr;

chart_property_list
    : chart_property (COMMA chart_property)* COMMA?
    ;
chart_property
    : option
    ;


config
    : 'config' EQ L_CURLY config_options_list R_CURLY
    ;
config_options_list
    : config_option ( COMMA config_option)* COMMA?
    ;
config_option
    : option
    ;

statement
    : object
    | ext_ref           //References defined outside the table definition
    | enum_out              // Enumeration
    | index_ext        //Index definition
    | dbfunc            // Database function
    | businessRuleDecl
//    | func              // External function
//    | rule              // Business Rule like drools
//    | view              // Flask-appbuilder view
//    | snippet           // Reusable code snippet
//    | vardecl           //variable declaration
    ;

dbfunc // A container for triggers and functions. Will be passed through directly
    : 'dbfunc' func_name string
    ;

func_name
    : name_attr
    ;

object
    : database
    | schema
    | tableDecl
    | mixin
    | dbview
    ;

database
    : name_attr
    ;
schema
    : ('public' | ident)
    ;

mixin
    : 'mixin' mixin_name L_CURLY column_list R_CURLY
    ;
mixin_name : name_attr;

column_list
    :  column (COMMA column)* COMMA?
    ;

column
    : column_name column_type ('[' column_option_list ']')?
    ;

column_name: name_attr;  // (ident | string)

column_reference:
     (table_reference DOT)? column_name
    ;
table_reference
    : (schema DOT)? table_name
    ;

column_option_list
    :  column_option (COMMA column_option)* COMMA?
    ;

column_option
    : primary_key
    | 'default' EQ column_default
    | 'cache' ( '(' INT ')' )? // Cache this column (lifetime defaults to 300 seconds)
    | 'default-expression' EQ '(' string ')'
    | INCR | DECR
    | unique
    | 'nullable'
    | 'not null'
    | ref_internal   //internal reference
    | enum_internal
    | 'min' EQ INT
    | 'max' EQ INT
    | 'check' check_expr
    | display_method
    | note_option
//    | option
    ;

check_expr: '(' string ')';  //TODO refine the check expressions

column_type
    : 'bit'| 'boolean'
    | 'tinyint'| 'smallint'| 'int'| 'integer' | 'bigint'
    | 'double' | 'decimal' | 'float'
    | 'money'| 'smallmoney'
    | 'char'| 'nchar'| varchar | 'nvarchar'
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
    | 'array' int_list  'of' column_type
    ;



column_default
    : NUMBER
    | string
    | BOOL
    | NOW | TODAY | TOMORROW
    | 'CURRENT_DATE' | 'CURRENT_USER'
    | NULL
    ;

enum_name: name_attr;

enum_internal  //Defined in table options, anonymous enum
    : 'enum' EQ L_CURLY enum_list R_CURLY
    ;

enum_out  //self standing enums
    : 'enum' enum_name L_CURLY enum_list R_CURLY
    ;

enum_list
    : enum_item (COMMA enum_item)* COMMA?
    ;

enum_item
    : (enum_idx EQ)? enum_value ('[' note_option ']')?
    ;

enum_idx: int;
enum_value : string ;

primary_key
    : 'pk'
    | 'primary_key'
    ;

display_method
    : 'display' EQ option_list
    ;


note_option
    : 'note' EQ note_value
    ;

note_value
    : string
    ;

varchar
    : VARCHAR L_PAR INT R_PAR
    ;

tableDecl
    : 'table' table_name ('(' mixin_list ')')? '{' column_list (index_int)? (view_s_spec)?  '}'
    ;

mixin_list
    : mixin_name? (COMMA mixin_name)*
    ;

table_name: name_attr;

dbview //TODO develop a method of creating views
    : 'dbview' EQ '(' db_join  ')'
    ;
db_join
    : column_reference EQ column_reference
    ;
ref_internal //defined in table options
    : 'ref' ref_name? COLON ref_type (schema DOT)? table_name DOT column_name
    ;

ext_ref //Free standing Reference Foreign_Key
    : 'ref' ref_name? COLON (schema DOT)? table_name DOT (column_name)+ ref_type table_name DOT (column_name)+
    ;
ref_name: name_attr;
ref_type
    : oneToOne
    | oneToMany
    | manyToOne
    | manyToMany
    ;

oneToOne: MINUS;
oneToMany: LT;
manyToOne: GT;
manyToMany: M2M;

index_ext
    : 'index' (index_name)? 'on' table_name '[' column_names ']' ('of type' ('btree'|'gin'|'gist'|'hash'))?
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
    :  column_name (',' column_name)* (',')?
    ;

index_name: name_attr;
view_s_spec
    : 'views' EQ '{' view_spec_list  '}'
    ;

view_spec_list
    : view_spec (',' view_spec)
    ;

view_spec
    : view_type COLON '{' view_spec_options '}'
    ;

view_type
    : ('add' | 'show' | 'list' | 'edit' | 'all')
    ;

view_spec_options
    : ('ex' | 'exclude') EQ '[' column_names ']'
    | ('in' | 'include') EQ '[' column_names ']'
    ;


businessRuleDecl: 'rule' rule_name ':' businessRule;
businessRule: ifExpr 'then' actionExpr;
ifExpr: ('if' | 'on') '(' expr ')' ( 'else' '(' expr ')' )?;
rule_name: name_attr;

actionExpr: 'action' (python_code | 'email' '(' string ')' | sms| notify | search | flag | execute_query| upload | download);
python_code: 'python' '(' string ')';
sms: 'sms' '(' destination ',' string ')';
notify: 'NOTIFY' '(' string ')'; // using postgresql notifications?
search: 'search' '(' string ')';
flag: 'flag' '(' string ')';
upload: 'upload' '(' server_loc ',' string ')';
download: 'download' '(' server_loc ',' string ')';
execute_query: 'execute_query' '(' string ')';

destination: string;
server_loc: string;

expr
    : ident                             # identExpression
    | literal                           # literalExpr
    | functionCall                      # functionCallExpr
    | '(' expr ')'                      # nestedExpr
    | expr op=(ASTERISK | DIV) expr     # binaryMultiplicationDiv
    | expr op=(ADD | MINUS) expr        # binaryAdditionSubtraction
    | expr booleanOp expr               # booleanCombination
    | expr comparisonOp expr            # binaryComparison
    | MINUS expr                        # unaryMinus
    | 'sin' expr                        # trigonometricSin
    | 'cos' expr                        # trigonometricCos
    | 'tan' expr                        # trigonometricTan
    | 'asin' expr                       # inverseTrigonometricSin
    | 'acos' expr                       # inverseTrigonometricCos
    | 'atan' expr                       # inverseTrigonometricTan
    | 'sinh' expr                       # hyperbolicSine
    | 'cosh' expr                       # hyperbolicCosine
    | 'tanh' expr                       # hyperbolicTangent
    | 'asinh' expr                      # inverseHyperbolicSine
    | 'acosh' expr                      # inverseHyperbolicCosine
    | 'atanh' expr                      # inverseHyperbolicTangent
    | 'avg' '(' expr_list ')'           # statisticalAverage
    | 'min' '(' expr_list ')'           # statisticalMinimum
    | 'max' '(' expr_list ')'           # statisticalMaximum
    | 'sum' '(' expr_list ')'           # statisticalSum
  ;

expr_list
    : expr (COMMA expr)* COMMA?
    ;

literal: INT | FLOAT | STRING | TRUE | FALSE;
booleanOp: AND | OR;
comparisonOp: EQ | NEQ | LT | LTE | GT | GTE;
arithmeticOp: ADD | MINUS | ASTERISK | DIV;

functionCall: 'exec' function_name  param_list?;  // params are in square brackets
function_name: name_attr;
param_list: string_list;

//binaryOp
//    : expr arithmeticOp expr
//    | expr booleanOp expr
//    | expr comparisonOp expr;



// Covenience Parser Rules

option_list
    : '[' option (COMMA option)* COMMA? ']'
    ;
option
    : (ident | string) EQ (string | string_list | ident| ident_list | int_list | int)
    ;

ident_list
    : '[' ident (',' ident)* ',' ']'
    | '[' ']'
    ;

string_list
    : '[' string (',' string)* COMMA? ']'
    | '[' ']'
    ;

int_list
    : '[' INT (COMMA INT)* COMMA? ']'
    | '[' ']'
    ;

int: INT;
string: STRING;

ident: IDENT;

name_attr
     : ident
     ;

unique: ('uniq' | '!' | 'unique') ;

db
    : 'pgsql'
    | 'mysql'
    | 'sqlite'
    | 'oracle'
    | 'mssql'
    ;

// LEXER Part
// Lexer tokens
COMMA           : ',';
COLON           : ':';
SEMI_COLON      : ';';
EQ              : '=';
L_PAR           : '(';
R_PAR           : ')';
L_SQUARE         : '[';
R_SQUARE         : ']';
//HASH_SYMBOL     : '#';
L_CURLY         : '{';
R_CURLY         : '}';
ASTERISK        : '*';
DOT             : '.';
MINUS           : '-';
GT              : '>';
LT              : '<';
M2M             : ('<>' | '*');
ADD: '+';
//SUB: '-';
//MUL: '*';
DIV: '/';
NEQ: '!=';
LTE: '<=';
GTE: '>=';
AND: '&&';
OR: '||';
// Whitespace and comments
//WS                  : [ \t]+      -> skip;
WS                  :  [ \t\r\n\u000C\u00A0]+ -> skip;
NL                  : [\r\n\u2028\u2029]        -> skip;
C_LINE_COMMENT      : '//' ~[\r\n]*     -> channel(HIDDEN);
C_STYLE_COMMENT     : '/*' .*? '*/'     -> channel(HIDDEN);
//P_STYLE_COMMENT     : '#' ~[(\r)? \n]*  -> channel(HIDDEN);
fragment LetterOrDigit
    : Letter
    | Digit
    ;

fragment Letter
    : [A-Za-z_]
    | ~[\u0000-\u00FF\uD800-\uDBFF]
    | [\uD800-\uDBFF] [\uDC00-\uDFFF]
    | [\u00E9]
    ;

fragment Digit
    : [0-9]
    ;

IDENT
    : Letter+ (LetterOrDigit)*
    ;
fragment SpecialChars
    :  [!$&'*+;=?^_`|~]
    ;
fragment DirectorySeparators
    :  [\\/] // Both Unix-like and Windows directory separators
    ;
// Now to define a string
fragment EscapeSequence
    :   '\\' ('b'|'B'|'t'|'n'|'f'|'r'|'\''|'\\'|'.'|'o'|
              'x'|'a'|'e'|'c'|'d'|'D'|'s'|'S'|'w'|'W'|'p'|'A'|
              'G'|'Z'|'z'|'Q'|'E'|'*'|'['|']'|'('|')'|'$'|'^'|
              '{'|'}'|'?'|'+'|'-'|'&'|'|')
    |   UnicodeEscape
    |   OctalEscape
    ;

fragment OctalEscape
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

fragment UnicodeEscape
    :   '\\' 'u' HexDigit HexDigit HexDigit HexDigit
    ;
STRING
    :  ('"' ( EscapeSequence | ~('\\'|'"') )* '"')
    |  ('\'' ( EscapeSequence | ~('\\'|'\'') )* '\'')
    { // Semantic Action
        setText( normalizeString( getText() ) );
    }
    ;

// Now define numbers

// INT has no leading zeros
//Can use _ and , to separate digits in a integer 1,000 or 1_000_000 are valid numbers
INT
   : '0' | [1-9] (Digit|'_')*
   ;

NUMBER
   : '-'? INT (DOT Digit+)? EXP?
   | '-'? INT EXP
   | '-'? INT
   ;

FLOAT
    :   Digit+ DOT Digit* EXP? FloatTypeSuffix?
    |   '.' Digit+ EXP? FloatTypeSuffix?
    |   Digit+ EXP FloatTypeSuffix?
    |   Digit+ FloatTypeSuffix
    ;

BOOL
	: ('true'|'false')
	| ('T' | 'F')
	| ('True'|'False')
    ;

fragment
EXP : ('e'|'E') ('+'|'-')? Digit+ ;

fragment
FloatTypeSuffix : ('f'|'F'|'d'|'D'|'B') ;

fragment HexDigit
    : ('0'..'9'|'a'..'f'|'A'..'F')
    ;

fragment IntegerTypeSuffix
    : ('l'|'L'|'I')
    ;

HEX 	: '0' ('x'|'X') HexDigit+ IntegerTypeSuffix? ;

DECIMAL	: INT IntegerTypeSuffix? ;

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
  : [0-9]+ '.' [0-9]+ ('.' [0-9]+)? ;

//REQUIRED: 'required';

//Parameters
dev_db_uri: 'dev_db_uri';
prod_db_uri: 'prod_db_uri';


INCR : ('increment' | 'incr' | '++');
DECR : ('decrement' | 'decr' | '--');
NOT_NULL: 'not' 'null';
NULL: ('null' | 'nil' | 'naught');
DISPLAY: 'display';


DIALECT: 'dialect';

CACHE: 'cache';
TimeSeries: ('tseries' | 'time_series' | 'timeseries');   //so that we can handle
VARCHAR: 'varchar';

// Defaults
NOW: 'now';
TODAY : 'today';
YESTERDAY: 'yesterday';
TOMORROW: 'tomorrow';
