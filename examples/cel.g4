grammar cel;


parse
    :  expression EOF
    ;

expression
    :  (unary_expr (binary_op unary_expr)* | conditional_expr)
    ;

unary_expr
    :  ('+' | '-') unary_expr
    |   negate unary_expr
    |   primary
    ;

negate
    :  '!' unary_expr
    ;

primary
    :  IDENTIFIER
    |   STRING_LITERAL
    |   NUMBER_LITERAL
    |   '(' expression ')'
    ;

binary_op
    :  '=='
    |   '!='
    |   '<'
    |   '<='
    |   '>'
    |   '>='
    |   '+'
    |   '-'
    |   '*'
    |   '/'
    |   '%'
    ;

conditional_expr
    :  logical_or_expr ('?' expression ':' expression)?
    ;

logical_or_expr
    :  logical_and_expr ('||' logical_and_expr)*
    ;

logical_and_expr
    :  unary_expr ('&&' unary_expr)*
    ;

STRING_LITERAL
    :  '"' (ESC | ~[\r\n"\\])* '"'
    ;

fragment ESC
    :  '\\' ('n' | 'r' | 't' | 'b' | 'f' | '"' | '\\')
    ;

NUMBER_LITERAL
    :  DIGIT+ (('.' DIGIT+)? EXPONENT? | EXPONENT)
    ;

fragment DIGIT
    :  [0-9]
    ;

EXPONENT
    :  ('e' | 'E') ('+' | '-')? DIGIT+
    ;

IDENTIFIER
    :  ([a-zA-Z] | '_') ([a-zA-Z0-9] | '_')*
    ;

WS
    :  (' ' | '\t' | '\r' | '\n')+ -> skip
    ;