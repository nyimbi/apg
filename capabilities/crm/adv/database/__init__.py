"""Database schema for APG Advanced CRM Analytics."""

class DatabaseManager:
    def __init__(self): self._tables = {}
    def create_table(self, name): self._tables.setdefault(name, [])
    def insert(self, table, record): self._tables.setdefault(table, []).append(record)
    def query(self, table, **filters):
        return [r for r in self._tables.get(table, []) if all(r.get(k)==v for k,v in filters.items())]
    def delete(self, table, **filters):
        rows = self._tables.get(table, [])
        orig = len(rows)
        self._tables[table] = [r for r in rows if not all(r.get(k)==v for k,v in filters.items())]
        return orig - len(self._tables[table])
