import asyncio
from config.manager import ConfigManager
from network.handler import NetworkHandler
from iso8583.message import ISO8583Message
from processor.transaction import TransactionProcessor
from routing.engine import RoutingEngine
from db.interface import DatabaseInterface
from logging.monitor import LoggingMonitor

async def main():
    # Initialize components
    config_manager = ConfigManager("config.yaml")
    config_manager.load_config()

    # Set up database
    db = DatabaseInterface(config_manager.get_database_config()['connection_string'])
    await db.connect()

    # Initialize components
    routing_engine = RoutingEngine(config_manager.get_routing_table())
    transaction_processor = TransactionProcessor(db, routing_engine)
    logging_monitor = LoggingMonitor()

    # Set up network handler
    network_config = config_manager.get_network_config()
    network_handler = NetworkHandler(
        network_config['host'],
        network_config['port'],
        transaction_processor,
        logging_monitor
    )

    # Start server
    print(f"Starting payment switch on {network_config['host']}:{network_config['port']}")
    await network_handler.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down payment switch...")
