from collections import defaultdict

class D2DiagramVisitor:
    def __init__(self):
        self.tables = defaultdict(list)
        self.relations = []

    # Visitor methods for each rule in the grammar
    def visit_appgen(self, node):
        for child in node.children:
            self.visit(child)

    def visit_table(self, node):
        table_name = node.table_name.text
        schema = ""

        columns = []
        for column_node in node.column_list.children:
            column_name = column_node.column_name.text
            column_type = column_node.column_type.text.lower()

            options = [opt.text.lower() for opt in column_node.column_option_list.children]

            if "primary key" in options:
                primary_key = True
            else:
                primary_key = False

            if "foreign key" in options:
                ref_table, ref_column = options[options.index("references") + 1:]
                foreign_key = f"{ref_table}.{ref_column}"
            else:
                foreign_key = None

            columns.append((column_name, column_type, primary_key, foreign_key))

        for rel_node in node.relation_list.children:
            rel_type = rel_node.relation_type.text.lower()
            other_table = rel_node.table_name.text
            other_column = rel_node.column_name.text

            if rel_type == "onetomany":
                # Add a foreign key to this table
                columns.append((f"{other_table.lower()}_id", "Integer", False, f"{other_table.lower()}.id"))
                self.tables[other_table].append((table_name, "N"))
                self.tables[table_name].append((other_table, "1"))
            elif rel_type == "manytomany":
                association_table_name = f"{table_name.lower()}_{other_table.lower()}"
                columns.append((f"{other_table.lower()}_ids", "JSON", False, None))
                self.tables[association_table_name].append((table_name, "N"))
                self.tables[table_name].append((association_table_name, "N"))
                self.tables[association_table_name].append((other_table, "N"))
                self.tables[other_table].append((association_table_name, "N"))
                self.relations.append((table_name, association_table_name, "N"))
                self.relations.append((association_table_name, other_table, "N"))

        self.tables[table_name] = columns

    def visit_enum(self, node):
        pass

    def visit_mixin(self, node):
        pass

    # Helper methods for generating D2 script
    def generate_d2script(self):
        d2script = ""

        for table in self.tables:
            d2script += f"node(\"{table}\")\n"
            for column in self.tables[table]:
                column_name, column_type, primary_key, foreign_key = column
                if primary_key:
                    d2script += f"\t.primary_key(\"{column_name}\")\n"
                if foreign_key:
                    foreign_table, foreign_column = foreign_key.split(".")
                    d2script += f"\t.foreign_key(\"{column_name}\", \"{foreign_table}\", \"{foreign_column}\")\n"

        for relation in self.relations:
            d2script += f"relation(\"{relation[0]}\", \"{relation[1]}\")\n"

        return d2script