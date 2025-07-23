grammar app;

options {
 language = Python3;
}

// Parser Rules

app
    : (table | enumDefinition | relationship | index | role | trigger | script | workflow | form | masterDetailForm | wizard | formStorage | chart)* EOF
    ;

table
    : 'Table' TABLENAME '{' column+ '}'
    ;

column
    : FIELDNAME dataType columnAttributes?
    ;

dataType
    : 'int'
    | 'varchar'
    | 'text'
    | 'timestamp'
    | 'decimal'
    | 'boolean'
    | IDENTIFIER // for enum types
    ;

columnAttributes
    : '[' columnAttribute (',' columnAttribute)* ']'
    ;

columnAttribute
    : 'pk'
    | 'increment'
    | 'unique'
    | 'not null'
    | 'default:' expression
    | 'ref:' relationshipType TABLENAME '.' FIELDNAME
    | 'validation:' validation
    | 'hint:' STRING
    | 'help:' STRING
    ;

validation
    : 'min:' INT
    | 'max:' INT
    | 'pattern:' STRING
    | 'required'
    ;

relationship
    : 'Ref' TABLENAME '.' FIELDNAME relationshipType TABLENAME '.' FIELDNAME
    ;

relationshipType
    : '>'   // one-to-many
    | '-'   // one-to-one
    | '<'   // many-to-one
    | '<>'  // many-to-many
    ;

index
    : 'Table' TABLENAME '{' indexes '}'
    ;

indexes
    : 'indexes' '{' indexDefinition+ '}'
    ;

indexDefinition
    : '(' FIELDNAME (',' FIELDNAME)* ')' indexAttributes?
    ;

indexAttributes
    : '[' indexAttribute (',' indexAttribute)* ']'
    ;

indexAttribute
    : 'unique'
    ;

enumDefinition
    : 'Enum' IDENTIFIER '{' enumValue+ '}'
    ;

enumValue
    : IDENTIFIER
    ;

role
    : 'Role' IDENTIFIER '{' permission+ '}'
    ;

permission
    : 'Permission' action 'on' TABLENAME
    ;

action
    : 'read'
    | 'write'
    | 'delete'
    ;

trigger
    : 'Trigger' IDENTIFIER 'on' TABLENAME triggerEvent '{' triggerBody '}'
    ;

triggerEvent
    : 'before'
    | 'after'
    | 'instead of'
    ;

triggerBody
    : statement+
    ;

script
    : 'Script' IDENTIFIER scriptLang '{' scriptBody '}'
    ;

scriptLang
    : 'zsh'
    | 'bash'
    | 'csh'
    | 'python'
    ;

scriptBody
    : statement+
    ;

workflow
    : 'Workflow' IDENTIFIER '{' workflowStep+ '}'
    ;

workflowStep
    : 'Step' IDENTIFIER 'do' '{' workflowStatement+ '}'
    ;

workflowStatement
    : statement
    | 'log' '(' STRING ')'
    ;

form
    : 'Form' IDENTIFIER 'for' TABLENAME '{' formLayout? formComponent+ formButton* formNavigator? '}'
    ;

formLayout
    : 'Layout' '{' layoutElement+ '}'
    ;

layoutElement
    : 'row' '(' rowElement (',' rowElement)* ')'
    ;

rowElement
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)?
    ;

formComponent
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)?
    ;

componentType
    : 'text'
    | 'textarea'
    | 'select'
    | 'radio'
    | 'checkbox'
    | 'password'
    | 'hidden'
    | 'file'
    | 'date'
    | 'time'
    | 'datetime'
    | 'email'
    | 'url'
    | 'number'
    | 'range'
    | 'color'
    | 'tel'
    | 'search'
    ;

formButton
    : 'Button' IDENTIFIER 'label' STRING ('action' actionType)? ('style' buttonStyle)?
    ;

actionType
    : 'submit'
    | 'reset'
    | 'button'
    ;

buttonStyle
    : 'primary'
    | 'secondary'
    | 'success'
    | 'danger'
    | 'warning'
    | 'info'
    | 'light'
    | 'dark'
    | 'link'
    ;

formNavigator
    : 'Navigator' IDENTIFIER (('{' navigatorButton* '}') | 'exclude' '(' navigatorButton (',' navigatorButton)* ')')?
    ;

navigatorButton
    : 'First'
    | 'Previous'
    | 'Next'
    | 'Last'
    | 'Insert'
    | 'Delete'
    | 'Edit'
    | 'Post'
    | 'Cancel'
    ;

masterDetailForm
    : 'MasterDetailForm' IDENTIFIER 'master' TABLENAME 'details' '{' detailComponent+ '}'
    ;

detailComponent
    : 'Detail' TABLENAME ('exclude' '(' excludeFields ')')?
    ;

excludeFields
    : FIELDNAME (',' FIELDNAME)*
    ;

wizard
    : 'Wizard' IDENTIFIER ('store' 'in' '{' tableMapping (',' tableMapping)* '}')? '{' wizardStep+ '}'
    ;

tableMapping
    : TABLENAME 'fields' '(' fieldList ')'
    ;

fieldList
    : FIELDNAME (',' FIELDNAME)*
    ;

wizardStep
    : 'Step' IDENTIFIER '{' wizardStatement+ '}'
    ;

wizardStatement
    : statement
    | 'form' '(' formFieldList ')'
    ;

formFieldList
    : formField (',' formField)*
    ;

formField
    : FIELDNAME ('as' componentType)? ('hint:' STRING)? ('help:' STRING)?
    ;

formStorage
    : 'FormStorage' IDENTIFIER 'form' IDENTIFIER 'save' 'for' 'later'
    ;

chart
    : 'Chart' IDENTIFIER 'type' chartType 'for' TABLENAME 'fields' '(' fieldList ')' ('title' STRING)?
    ;

chartType
    : 'bar'
    | 'line'
    | 'pie'
    | 'scatter'
    | 'area'
    ;

// Statements and Expressions

statement
    : expression ';'
    ;

expression
    : IDENTIFIER
    | IDENTIFIER '.' IDENTIFIER
    | INT
    | STRING
    ;

// Lexer Rules

TABLENAME
    : [A-Z_][A-Z0-9_]*
    ;

FIELDNAME
    : [a-z_][a-z0-9_]*
    ;

IDENTIFIER
    : [a-zA-Z_][a-zA-Z0-9_]*
    ;

INT
    : [0-9]+
    ;

STRING
    : '\'' ('\\' . | ~('\\' | '\''))* '\''
    ;

WS
    : [ \t\r\n]+ -> skip
    ;

COMMENT
    : '//' ~[\r\n]* -> skip
    ;

// Fragments

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [a-zA-Z]
    ;

