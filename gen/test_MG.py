from antlr4 import *
from AppGenLexer import AppGenLexer
from AppGenParser import AppGenParser
from FMG import FABModelGenerator

input_code = """
class User:
    name: str
    email: str
    posts: List[Ref(Post)]
    followers: List[FK:User]
    following: List[FK:User]

class Post:
    title: str
    content: str
    author: Ref(User)
    comments: List[Ref(Comment)]

class Comment:
    content: str
    author: Ref(User)
    post: Ref(Post)
"""

# Create a lexer and parser for the input code
lexer = AppGenLexer(InputStream(input_code))
tokens = CommonTokenStream(lexer)
parser = AppGenParser(tokens)

# Create a visitor and visit the parsed code
generator = FABModelGenerator()
tree = parser.model()
generator.visit(tree)

# Generate the Flask-AppBuilder model.py file
output_code = generator.generate_models()
print(output_code)