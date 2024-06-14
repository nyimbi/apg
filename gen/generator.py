from antlr4 import FileStream, CommonTokenStream
from DBML4Lexer import DBML4Lexer
from DBML4Parser import DBML4Parser
from SQLGenerator import SQLGenerator
from FABModelGenerator import FlaskAppBuilderModelGenerator
from FABViewGenerator import FlaskAppBuilderViewGenerator

lexer = DBML4Lexer(FileStream('example.dbml'))
tokens = CommonTokenStream(lexer)
parser = DBML4Parser(tokens)
tree = parser.dbml()

sql_generator = SQLGenerator()
sql = sql_generator.generate_sql(tree)
print(sql)

fab_model_generator = FlaskAppBuilderModelGenerator()
fab_view_generator = FlaskAppBuilderViewGenerator()
models = fab_model_generator.generate_flask_appbuilder_models(tree)
views = fab_view_generator.generate_flask_appbuilder_views(tree)
print(models)
print(views)