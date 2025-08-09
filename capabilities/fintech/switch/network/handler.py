import asyncio
from iso8583.message import ISO8583Message
from processor.transaction import TransactionProcessor
from logging.monitor import LoggingMonitor

class NetworkHandler:
    def __init__(self, host: str, port: int, processor: TransactionProcessor, monitor: LoggingMonitor,
                 ssl_context: ssl.SSLContext = None, timeout: int = 30,
                 max_connections: int = 1000, keep_alive: bool = True):
        self.host = host
        self.port = port
        self.processor = processor
        self.monitor = monitor
        self.ssl_context = ssl_context
        self.timeout = timeout
        self.max_connections = max_connections
        self.keep_alive = keep_alive
        self.connections = set()
        self._shutdown = False

    async def start_server(self):
        """Start the network server with configured SSL/TLS if enabled"""
        server = await asyncio.start_server(
            self.handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            limit=64*1024,  # 64KB buffer limit
            backlog=2048
        )

        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                self._shutdown = True
                # Clean shutdown - wait for connections to complete
                if self.connections:
                    await asyncio.gather(*self.connections, return_exceptions=True)

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle individual client connections with proper error handling and timeouts"""
        peer = writer.get_extra_info('peername')
        connection = asyncio.current_task()
        self.connections.add(connection)

        try:
            while not self._shutdown:
                # Set timeout for reading message length
                try:
                    length_bytes = await asyncio.wait_for(
                        reader.readexactly(2),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    self.monitor.log_error(f"Connection timeout from {peer}")
                    break

                if not length_bytes:  # Connection closed
                    break

                message_length = int.from_bytes(length_bytes, 'big')

                # Validate message length
                if message_length > 8192:  # 8KB max message size
                    raise ValueError(f"Message too large: {message_length} bytes")

                # Read complete message with timeout
                try:
                    data = await asyncio.wait_for(
                        reader.readexactly(message_length),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    self.monitor.log_error(f"Message read timeout from {peer}")
                    break

                # Parse and process message
                message = ISO8583Message()
                message.parse(data)

                start_time = asyncio.get_event_loop().time()

                try:
                    # Process transaction with timeout
                    response = await asyncio.wait_for(
                        self.processor.process_transaction(message),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    self.monitor.log_error(f"Transaction processing timeout for {message.get_mti()}")
                    continue

                # Build and send response
                response_data = response.build()
                response_length = len(response_data).to_bytes(2, 'big')

                try:
                    writer.write(response_length + response_data)
                    await writer.drain()
                except ConnectionError as e:
                    self.monitor.log_error(f"Connection error while sending response: {str(e)}")
                    break

                # Log successful transaction
                duration = asyncio.get_event_loop().time() - start_time
                self.monitor.log_transaction(message, duration)

                if not self.keep_alive:
                    break

        except Exception as e:
            self.monitor.log_error(f"Connection error from {peer}: {str(e)}")

        finally:
            writer.close()
            await writer.wait_closed()
            self.connections.remove(connection)
