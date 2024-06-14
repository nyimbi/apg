from antlr4 import *
from apgLexer import apgLexer
from apgParser import apgParser
from apgSQL import SQLVisitor

# assuming apg_code is a string containing APG code
input_stream = InputStream(apg_code)
lexer = apgLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = apgParser(token_stream)

parse_tree = parser.apg() # get the root of the parse tree
sql_visitor = SQLVisitor()
sql_output = sql_visitor.visit(parse_tree) # generate SQL
print(sql_output)


#   #### Listener
input_stream = FileStream('example.apg')
lexer = apgLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = apgParser(token_stream)

tree = parser.apg() # get the root of the parse tree
listener = SQLGenerator()
walker = ParseTreeWalker()
walker.walk(listener, tree)

for statement in listener.get_statements():
    print(statement)

#### model gen
from antlr4 import *
from apgLexer import apgLexer
from apgParser import apgParser

input_stream = FileStream('example.apg')
lexer = apgLexer(input_stream)
token_stream = CommonTokenStream(lexer)
parser = apgParser(token_stream)

tree = parser.apg() # get the root of the parse tree

# Then, you can create a visitor instance and traverse the parse tree using tree.accept(visitor),
# where visitor is an instance of the ModelGenerator class. This will generate the appropriate
# model classes for your Flask-AppBuilder application.

visitor = ModelGenerator()
model_classes = []
for child in tree.children:
    model_class = child.accept(visitor)
    model_classes.append(model_class)