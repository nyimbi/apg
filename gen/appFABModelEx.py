import appgen_parser  # import the generated parser
from FABModelsVisitor import FABModelsVisitor

# Parse the input file and create the CST
with open("input_file.ag", "r") as file:
    input_data = file.read()
parser = appgen_parser.Parser()
cst = parser.parse(input_data)

# Generate FAB models from the CST
visitor = FABModelsVisitor()
visitor.visit(cst)
models = visitor.models

# Write the models to the models.py file
with open("models.py", "w") as file:
    file.write("from flask_appbuilder import Model\n")
    file.write("\n")
    for model in models:
        table_name = model[0]
        columns = model[1]
        schema = model[2]
        file.write(schema)
        file.write(f"class {table_name}(Model):\n")
        file.write(f"    id = Column(Integer, primary_key=True)\n")
        for column in columns:
            column_name, column_type, primary_key, unique, nullable, default, foreign_key = column
            if foreign_key:
                file.write(f"    {column_name}_id = Column(Integer, ForeignKey('{foreign_key}'), index=True)\n")
                file.write(f"    {column_name} = relationship('{foreign_key.split('.')[0]}')\n")
            else:
                file.write(f"    {column_name} = Column({column_type}")
                if primary_key:
                    file.write(", primary_key=True")
                if unique:
                    file.write(", unique=True")
                if not nullable:
                    file.write(", nullable=False")
                if default is not None:
                    file.write(f", default={default}")
                file.write(")\n")
        file.write("\n")