import logging
from prometheus_client import Counter, Histogram

class LoggingMonitor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PaymentSwitch")

        self.transaction_counter = Counter('transactions_total', 'Total number of transactions', ['mti'])
        self.transaction_duration = Histogram('transaction_duration_seconds', 'Transaction processing duration')

    def log_transaction(self, message, duration):
        self.logger.info(f"Processed transaction: MTI={message.mti}, PAN={message.fields[2][:6]}XXXXXX")
        self.transaction_counter.labels(mti=message.mti).inc()
        self.transaction_duration.observe(duration)

    def log_error(self, error_message):
        self.logger.error(f"Error: {error_message}")