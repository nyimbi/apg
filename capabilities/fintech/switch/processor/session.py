import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional
from iso8583.message import ISO8583Message


class SessionState:
    def __init__(self):
        self.transaction_count = 0
        self.last_activity = None
        self.authenticated = False
        self.terminal_id = None
        self.merchant_id = None
        self.batch_number = None
        self.sequence_number = 0
        self.reversal_queue = []

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SessionState()
        self.created_at = datetime.now()
        self.expires_at = None
        self.lock = asyncio.Lock()

    async def increment_sequence(self) -> int:
        async with self.lock:
            self.state.sequence_number += 1
            return self.state.sequence_number

    def update_activity(self):
        self.state.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int) -> bool:
        if not self.state.last_activity:
            return False
        return (datetime.now() - self.state.last_activity).seconds > timeout_seconds

class SessionManager:
    def __init__(self, timeout_seconds: int = 3600):
        self.sessions = {}
        self.timeout_seconds = timeout_seconds
        self.lock = asyncio.Lock()

    async def create_session(self, terminal_id: str, merchant_id: str) -> Session:
        session_id = self._generate_session_id()
        session = Session(session_id)
        session.state.terminal_id = terminal_id
        session.state.merchant_id = merchant_id
        session.state.authenticated = True
        session.update_activity()

        async with self.lock:
            self.sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> Session:
        async with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            if session.is_expired(self.timeout_seconds):
                await self.remove_session(session_id)
                raise ValueError(f"Session {session_id} has expired")
            session.update_activity()
            return session

    async def remove_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    async def cleanup_expired(self):
        async with self.lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(self.timeout_seconds)
            ]
            for session_id in expired:
                await self.remove_session(session_id)

    def _generate_session_id(self) -> str:
        return str(uuid.uuid4())

class SessionContext:
    def __init__(self, session: Session):
        self.session = session
        self.transaction_start = None
        self.transaction_data = {}
        self.reversal_data = None

    async def start_transaction(self, message: ISO8583Message):
        self.transaction_start = datetime.now()
        self.transaction_data = {
            'mti': message.get_mti(),
            'stan': message.get_field(11),
            'amount': message.get_field(4),
            'timestamp': self.transaction_start
        }
        await self.session.increment_sequence()
        self.session.state.transaction_count += 1

    def store_reversal_data(self):
        self.reversal_data = self.transaction_data.copy()
        self.session.state.reversal_queue.append(self.reversal_data)

    def clear_reversal_data(self):
        self.reversal_data = None
        if self.session.state.reversal_queue:
            self.session.state.reversal_queue.pop()

class SessionMiddleware:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def process(self, message: ISO8583Message) -> Session:
        terminal_id = message.get_field(41)
        merchant_id = message.get_field(42)

        try:
            session = await self.session_manager.get_session(terminal_id)
        except ValueError:
            session = await self.session_manager.create_session(terminal_id, merchant_id)

        return SessionContext(session)

    async def cleanup(self):
        await self.session_manager.cleanup_expired()
