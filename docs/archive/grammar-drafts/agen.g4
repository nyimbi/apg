grammar agen;

/* Lexer rules */

fragment LETTER
    : [a-zA-Z]
    ;

fragment DIGIT
    : [0-9]
    ;

//fragment SYMBOL
//    : [^]+-=<>{},()[];#'"`/
//    ;

COMMENT
    : '#' ~[\r\n]* -> skip
    ;

SINGLE_QUOTED_STRING
    : '\'' ~('\'')* '\''
    ;

DOUBLE_QUOTED_STRING
    : '"' ~('"')* '"'
    ;


// Lexer tokens
SPACES          : [ \t\n\r]+ -> skip;
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
L_BRACE         : '{';
R_BRACE         : '}';
ASTERISK        : '*';
HASH_SYMBOL     : '#';

ID
    : LETTER (LETTER | DIGIT | '_')*
    ;

INT
    : DIGIT+
    ;

/* Parser rules */

dbml: entity*;

entity: mixin | table;

mixin
    : 'mixin' mixin_name L_BRACKET mixin_column* R_BRACKET mixin_options?
    ;

mixin_name
    : ID
    ;

mixin_column
    : column
    ;

mixin_options
    : mixin_option*
    ;

mixin_option
    : 'note' EQUALS STRING
    ;

table
    : 'table' table_name table_options? L_BRACKET table_body R_BRACKET
    ;

table_name
    : ID
    ;

table_options
    : table_option*
    ;

table_option
    : 'ref' L_ANGLE_BRACKET ref_column R_ANGLE_BRACKET
    | 'note' EQUALS STRING
    ;

table_body
    : column*
    view_section?
    ;

column
    : column_name column_type? L_BRACKET column_option* R_BRACKET
    ;

column_name
    : ID
    ;

column_type
    : INT
    | 'bool'
    | 'string'
    | 'date'
    | 'time'
    | 'datetime'
    | 'text'
    | 'serial'
    | 'json'
    | 'blob'
    | 'file'
    | 'point'
    | 'image'
    | 'interval'
    | enum_name
    | varchar
    ;

enum_name
    : ID
    ;

varchar
    : 'varchar' L_PARENTHESES INT R_PARENTHESES
    ;

column_option
    : 'pk'
    | 'default' EQUALS column_default
    | 'increment'
    | 'unique'
    | 'nullable'
    | 'ref' L_ANGLE_BRACKET ref_column R_ANGLE_BRACKET
    | 'enum' L_BRACKET enum_value (COMMA enum_value)* R_BRACKET
    | 'default-expression' L_PARENTHESES SINGLE_QUOTED_STRING R_PARENTHESES
    | 'min' EQUALS INT
    | 'max' EQUALS INT
    | display_option
    | note_option
    ;

column_default
    : INT
    | STRING
    | 'true'
    | 'false'
    ;

enum_value
    : STRING
    ;

ref_column
    : ID (DOT ID)?
    ;

view_section
    : 'View' view_name L_BRACKET view_body R_BRACKET
    ;

view_name
    : ID
    ;

view_body
    : view_line*
    ;

view_line
    : '(' STRING ',' SINGLE_QUOTED_STRING ')' SEMI_COLON
    ;

display_option
    : 'display' EQUALS STRING
    ;

note_option
    : 'note' EQUALS STRING
    ;

/* Special characters */

fragment DOT
    : '.'
    ;

/* Special rules */

STRING
    : (SINGLE_QUOTED_STRING | DOUBLE_QUOTED_STRING | BACKTICK_QUOTED_STRING)+
    ;