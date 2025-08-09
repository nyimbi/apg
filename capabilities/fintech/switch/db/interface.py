import asyncpg
from iso8583.message import ISO8583Message

class DatabaseInterface:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.connection_string)

    async def store_transaction(self, message: ISO8583Message):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO transactions (mti, pan, amount, datetime)
                VALUES ($1, $2, $3, $4)
            """, message.mti, message.fields[2], message.fields[4], message.fields[7])

    async def get_transaction(self, reference_number: str):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow("""
                SELECT * FROM transactions WHERE reference_number = $1
            """, reference_number)