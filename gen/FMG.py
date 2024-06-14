from typing import List

class FABModelGenerator(modelVisitor):
    def __init__(self):
        self.models = {}
        self.current_model = None

    def visitClassDef(self, ctx):
        model_name = ctx.getChild(1).getText()
        self.models[model_name] = []
        self.current_model = model_name

    def visitMemberVarDecl(self, ctx):
        column_name = ctx.getChild(1).getText()
        column_type = self.visitType(ctx.getChild(0))
        if column_type.startswith('FK'):
            # The column is a foreign key, so we need to extract the referenced model
            ref_model = column_type.split(':')[1].strip()
            ref_column = 'id'
            # Add the foreign key column to the current model
            self.models[self.current_model].append((column_name, column_type, ref_model, ref_column))
            # Add a corresponding relation to the referenced model
            self.models[ref_model].append(('backref_' + self.current_model.lower(),
                                            f'{self.current_model}', None, None))
        else:
            # The column is a regular column, add it to the current model
            self.models[self.current_model].append((column_name, column_type, None, None))

    def visitType(self, ctx) -> str:
        if ctx.getChildCount() > 1:
            if ctx.getChild(1).getText() == '?':
                return "String(160)"
        type_str = ctx.getText().lower()
        if type_str == "ref":
            next_ctx = ctx.parentCtx.getChild(ctx.getChildIndex() + 1)
            next_text = next_ctx.getText().lower()
            if next_text == "manytomany":
                # The column is a many-to-many relationship, we need to create an association table
                table_names = sorted([ctx.parentCtx.parentCtx.getChild(i).getText()
                                      for i in range(ctx.parentCtx.parentCtx.getChildCount())
                                      if ctx.parentCtx.parentCtx.getChild(i).getRuleIndex() == modelParser.RULE_className])
                association_table_name = "_".join(table_names)
                # Add two foreign key columns for the tables involved and a primary key column for the association table
                column1 = f"{table_names[0].lower()}_id"
                column2 = f"{table_names[1].lower()}_id"
                self.models[association_table_name] = [(column1, f'ForeignKey("{table_names[0]}.id")', table_names[0], 'id'),
                                                       (column2, f'ForeignKey("{table_names[1]}.id")', table_names[1], 'id'),
                                                       ('id', 'Integer', None, None),
                                                       (f'{table_names[0].lower()}', f'relationship("{table_names[0]}", back_populates="{association_table_name.lower()}")', None, None),
                                                       (f'{table_names[1].lower()}', f'relationship("{table_names[1]}", back_populates="{association_table_name.lower()}")', None, None)]
                return None
            else:
                # The column is a regular foreign key
                ref_model = next(ctx.getChildren()).getText()
                return f'FK::{ref_model}'

        elif type_str == 'file':
            return 'string'

        elif type_str == 'image':
            return 'image'

        else:
            return type_str.capitalize()

    def generate_models(self) -> str:
        output = ""
        for model_name in self.models:
            output += f'class {model_name}(Model):\n'

            # Generate the columns of the model
            for column in self.models[model_name]:
                if column[0].startswith('backref_'):
                    continue
                output += f'    {column[0]} = Column({column[1]}'
                if column[2] is not None:
                    output += f', ForeignKey("{column[2]}.{column[3]}")'
                output += ')\n'

            # Generate the relations of the model (for regular columns only)
            for column in self.models[model_name]:
                if column[0].startswith('backref_') or column[1].startswith('FK'):
                    continue
                if column[2] is not None:
                    output += f'    {column[0]}_{column[2].lower()} = relationship("{column[2]}", back_populates="{model_name.lower()}")\n'

            output += '\n'
        return output