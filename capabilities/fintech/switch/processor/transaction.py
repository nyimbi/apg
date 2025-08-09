import asyncio
from datetime import datetime
from typing import Dict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from iso8583.message import ISO8583Message
from db.interface import DatabaseInterface
from routing.engine import RoutingEngine


class HSM:
    """Hardware Security Module interface"""
    def __init__(self, key_source='file'):
        self.key_source = key_source
        self.zmk = self._load_zmk()
        self.tmk = self._load_tmk()
        self.working_keys: Dict[str, bytes] = {}
        self.pin_block_formats = {
            'ISO0': self._generate_iso0_pin_block,
            'ISO1': self._generate_iso1_pin_block,
            'ISO3': self._generate_iso3_pin_block
        }

    def _load_zmk(self) -> bytes:
        if self.key_source == 'file':
            try:
                with open('keys/zmk.key', 'rb') as f:
                    return f.read()
            except:
                return b'1234567890123456'  # Fallback
        elif self.key_source == 'api':
            # Implement API call to get ZMK
            return b'1234567890123456'  # Placeholder
        return b'1234567890123456'

    def _load_tmk(self) -> bytes:
        if self.key_source == 'file':
            try:
                with open('keys/tmk.key', 'rb') as f:
                    return f.read()
            except:
                return b'ABCDEFGHIJKLMNOP'  # Fallback
        elif self.key_source == 'api':
            # Implement API call to get TMK
            return b'ABCDEFGHIJKLMNOP'  # Placeholder
        return b'ABCDEFGHIJKLMNOP'

    def generate_key_pair(self) -> tuple:
        """Generate a new key pair for key exchange"""
        import os
        new_key = os.urandom(16)
        encrypted_key = self.encrypt_under_zmk(new_key)
        return new_key, encrypted_key

    def encrypt_under_zmk(self, key_data: bytes) -> bytes:
        """Encrypt a key under the Zone Master Key"""
        cipher = Cipher(algorithms.TripleDES(self.zmk), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(key_data) + encryptor.finalize()

    def decrypt_zpk_double_variant(self, encrypted_zpk: bytes) -> bytes:
        """Decrypt ZPK using double-variant key"""
        variant_key = self._apply_variant(self.zmk, b'\x00\x00\x00\x00\x00\x00\x00\xFF')
        cipher = Cipher(algorithms.TripleDES(variant_key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_zpk) + decryptor.finalize()

    def _apply_variant(self, key: bytes, variant: bytes) -> bytes:
        """Apply variant to key"""
        return bytes(a ^ b for a, b in zip(key, variant))

    def generate_pin_block(self, pin: str, pan: str, format: str = 'ISO0') -> bytes:
        """Generate PIN block in specified format"""
        if format not in self.pin_block_formats:
            raise ValueError(f"Unsupported PIN block format: {format}")

        return self.pin_block_formats[format](pin, pan)

    def _generate_iso0_pin_block(self, pin: str, pan: str) -> bytes:
        """Generate ISO-0 PIN block format"""
        # Format PIN field
        pin_length = len(pin)
        if not (4 <= pin_length <= 12):
            raise ValueError("PIN must be between 4 and 12 digits")

        pin_field = f"0{pin_length}{pin}{'F' * (14 - pin_length)}"

        # Format PAN field
        pan_field = f"0000{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"

        # XOR the fields
        pin_block = bytes.fromhex(pin_field)
        pan_block = bytes.fromhex(pan_field)
        return bytes(a ^ b for a, b in zip(pin_block, pan_block))

    def _generate_iso1_pin_block(self, pin: str, pan: str) -> bytes:
        """Generate ISO-1 PIN block format"""
        # Format PIN field with random fill
        pin_length = len(pin)
        if not (4 <= pin_length <= 12):
            raise ValueError("PIN must be between 4 and 12 digits")

        import random
        random_fill = ''.join(str(random.randint(0,9)) for _ in range(14 - pin_length))
        pin_field = f"1{pin_length}{pin}{random_fill}"

        # Format PAN field
        pan_field = f"0000{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"

        # XOR the fields
        pin_block = bytes.fromhex(pin_field)
        pan_block = bytes.fromhex(pan_field)
        return bytes(a ^ b for a, b in zip(pin_block, pan_block))

    def _generate_iso3_pin_block(self, pin: str, pan: str) -> bytes:
        """Generate ISO-3 PIN block format"""
        # Format PIN field
        pin_length = len(pin)
        if not (4 <= pin_length <= 12):
            raise ValueError("PIN must be between 4 and 12 digits")

        pin_field = f"3{pin_length}{pin}{'F' * (14 - pin_length)}"

        # Format PAN field differently for ISO-3
        pan_field = f"{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"

        # XOR the fields
        pin_block = bytes.fromhex(pin_field)
        pan_block = bytes.fromhex(pan_field)
        return bytes(a ^ b for a, b in zip(pin_block, pan_block))

    def decrypt_pin_block(self, pin_block: bytes, terminal_id: str, pan: str, format: str = 'ISO0') -> str:
        """Decrypt PIN block using terminal's working key"""
        if terminal_id not in self.working_keys:
            raise ValueError("No working key found for terminal")

        working_key = self.working_keys[terminal_id]
        cipher = Cipher(algorithms.TripleDES(working_key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(pin_block) + decryptor.finalize()

        # Generate PAN block based on format
        if format == 'ISO0':
            pan_field = f"0000{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"
        elif format == 'ISO3':
            pan_field = f"{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"
        else:
            pan_field = f"0000{'0' * (16 - len(pan[-12:]))}{pan[-12:]}"

        pan_block = bytes.fromhex(pan_field)

        # XOR with PAN block to recover PIN field
        pin_field = bytes(a ^ b for a, b in zip(decrypted, pan_block))
        return self._extract_pin(pin_field.hex(), format)

    def _extract_pin(self, pin_field: str, format: str) -> str:
        """Extract PIN from PIN block format"""
        pin_length = int(pin_field[1])
        if format == 'ISO1':
            return pin_field[2:2+pin_length]
        return pin_field[2:2+pin_length]

    def _format_pin(self, pin_block: bytes) -> str:
        """Extract PIN from PIN block format"""
        return pin_block.hex()[:4]  # Simplified example

ISO8583_RESPONSE_CODES = {
    # Approved Transactions
    "00": "Approved or completed successfully",
    "08": "Honor with identification",
    "10": "Approved, partial amount",
    "11": "Approved, VIP customer",
    "85": "No reason to decline transaction",
    # Referral Required
    "01": "Refer to card issuer",
    "02": "Refer to card issuer, special condition",
    "03": "Invalid merchant or service provider",
    "37": "Card acceptor call acquirer security",
    "38": "PIN tries exceeded - Call issuer",
    # Card Security
    "04": "Pick up card (no fraud)",
    "07": "Pick up card, special condition (fraud account)",
    "34": "Suspected fraud - retain card",
    "36": "Restricted card - retain",
    "41": "Lost card - pick up",
    "43": "Stolen card - pick up",
    "62": "Restricted card - decline",
    # Processing Issues
    "06": "System/Processing error",
    "09": "Request in progress - please wait",
    "12": "Invalid transaction type",
    "13": "Invalid amount specified",
    "14": "Invalid card number/PAN",
    "15": "No such issuing institution",
    "19": "Re-enter transaction",
    "21": "No action taken (unable to back out prior transaction)",
    "25": "Unable to locate record",
    "28": "File temporarily unavailable",
    "30": "Format error/invalid message format",
    "31": "Bank not supported by switch/Unable to route transaction",
    "40": "Requested function not supported",
    "68": "Response received too late",
    "90": "Cut-off in progress - retry",
    "91": "Issuer/switch inoperative",
    "92": "Routing error/unable to route transaction",
    "96": "System malfunction/error",
    # Account Issues
    "39": "No credit account",
    "42": "No universal account",
    "51": "Insufficient funds",
    "52": "No checking account",
    "53": "No savings account",
    "54": "Expired card",
    "56": "No card record found",
    "57": "Transaction not permitted for cardholder",
    "58": "Transaction not allowed at terminal",
    "61": "Exceeds withdrawal amount limit",
    "65": "Exceeds withdrawal frequency limit",
    "98": "Exceeds cash limit",
    # Security/Authentication
    "55": "Incorrect PIN entered",
    "63": "Security violation",
    "66": "Contact acquirer security department",
    "75": "Allowable PIN tries exceeded",
    "86": "Cannot verify PIN",
    "89": "Invalid terminal/Bad terminal",
    "93": "Transaction violates law",
    "99": "PIN Block encryption error",
    # Capture Response
    "64": "Original amount was incorrect",
    "67": "Hard capture - retain card at ATM",
    # Reconciliation
    "94": "Duplicate transaction detected",
    "95": "Reconciliation error/balancing error",
    # Administrative
    "77": "Intervene, bank approval required for transaction",
    "78": "Intervene, bank approval required for partial amount",
    "97": "Reserved for national use/assignment",
}

RESPONSE_CATEGORIES = {
    "SUCCESS": ["00", "08", "10", "11", "85"],
    "REFER_TO_ISSUER": ["01", "02", "03", "37", "38"],
    "CARD_ISSUES": ["04", "07", "34", "36", "41", "43", "62", "67"],
    "SECURITY_ISSUES": ["34", "36", "37", "63", "66", "93", "99"],
    "PIN_ISSUES": ["38", "55", "75", "86"],
    "ROUTING_ISSUES": ["15", "31", "92"],
    "SYSTEM_ISSUES": ["06", "90", "91", "96"],
    "ACCOUNT_ISSUES": ["39", "42", "51", "52", "53", "54", "56"],
    "FORMAT_ISSUES": ["30", "40"],
    "TRANSACTION_ISSUES": ["12", "13", "14", "19", "57", "58", "61", "65", "98"],
    "PROCESSING_ISSUES": ["09", "21", "25", "28", "68"],
    "RECONCILIATION": ["64", "94", "95"],
    "ADMINISTRATIVE": ["77", "78", "97"],
}

class TransactionProcessor:
    """
    Handles processing of ISO8583 financial transactions including message validation,
    key exchange, PIN verification, MAC validation, and routing.

    Args:
        db (DatabaseInterface): Database interface for transaction storage and lookup
        routing (RoutingEngine): Routing engine for transaction routing

    Attributes:
        db (DatabaseInterface): Database interface instance
        routing (RoutingEngine): Routing engine instance
        hsm (HSM): Hardware Security Module interface
        response_codes (dict): Standard ISO8583 response codes
        max_retries (int): Maximum number of processing retries
        retry_delay (int): Delay between retries in seconds
    """

    def __init__(self, db: DatabaseInterface, routing: RoutingEngine):
        self.db = db
        self.routing = routing
        self.hsm = HSM()
        self.response_codes = ISO8583_RESPONSE_CODES
        self.max_retries = 3
        self.retry_delay = 30  # seconds

    async def process_transaction(self, message: ISO8583Message) -> ISO8583Message:
        """
        Main transaction processing method. Handles validation, key exchange,
        PIN verification and routes to appropriate processing logic.

        Args:
            message (ISO8583Message): The incoming ISO8583 message to process

        Returns:
            ISO8583Message: The response message
        """
        try:
            # Validate MAC if present
            if not self._verify_mac(message):
                return self._create_error_response(message, "Invalid MAC", "91")

            # Handle key exchange if necessary
            if self._is_key_exchange(message):
                return await self._handle_key_exchange(message)

            # Handle PIN verification if present
            if self._has_pin_block(message):
                if not await self._verify_pin(message):
                    return self._create_declined_response(message, "55")

            # Regular transaction processing
            return await self._process_regular_transaction(message)

        except Exception as e:
            return self._create_error_response(message, str(e), "96")

    def _is_key_exchange(self, message: ISO8583Message) -> bool:
        """
        Check if message is a key exchange request.

        Args:
            message (ISO8583Message): Message to check

        Returns:
            bool: True if message is a key exchange request
        """
        return message.get_mti() == "0800" and message.get_field(70) == "001"

    def _has_pin_block(self, message: ISO8583Message) -> bool:
        """
        Check if message contains a PIN block.

        Args:
            message (ISO8583Message): Message to check

        Returns:
            bool: True if message contains PIN block field
        """
        return 52 in message.fields

    async def _handle_key_exchange(self, message: ISO8583Message) -> ISO8583Message:
        """
        Process a key exchange request message.

        Args:
            message (ISO8583Message): The key exchange request message

        Returns:
            ISO8583Message: Key exchange response message
        """
        try:
            terminal_id = message.get_field(41)
            working_key, encrypted_key = self.hsm.generate_key_pair()
            self.hsm.working_keys[terminal_id] = working_key

            response = ISO8583Message()
            response.set_mti("0810")
            response.set_field(39, "00")
            response.set_field(41, terminal_id)
            response.set_field(53, encrypted_key.hex())
            self._add_mac(response)

            return response

        except Exception as e:
            return self._create_error_response(message, f"Key exchange failed: {str(e)}", "96")

    async def _verify_pin(self, message: ISO8583Message) -> bool:
        """
        Verify PIN block in message.

        Args:
            message (ISO8583Message): Message containing PIN block

        Returns:
            bool: True if PIN verification successful
        """
        try:
            pin_block = bytes.fromhex(message.get_field(52))
            terminal_id = message.get_field(41)
            pan = message.get_field(2)

            decrypted_pin = self.hsm.decrypt_pin_block(pin_block, terminal_id, pan)
            return len(decrypted_pin) == 4

        except Exception:
            return False

    def _verify_mac(self, message: ISO8583Message) -> bool:
        """
        Verify Message Authentication Code.

        Args:
            message (ISO8583Message): Message to verify MAC for

        Returns:
            bool: True if MAC is valid or not present
        """
        if 64 not in message.fields:
            return True

        try:
            received_mac = message.get_field(64)
            calculated_mac = self._calculate_mac(message)
            return received_mac == calculated_mac

        except Exception:
            return False

    def _calculate_mac(self, message: ISO8583Message) -> str:
        """
        Calculate MAC for message.

        Args:
            message (ISO8583Message): Message to calculate MAC for

        Returns:
            str: Calculated MAC value
        """
        message_data = message.build()
        return "0" * 16  # Dummy MAC - replace with actual MAC calculation

    def _add_mac(self, message: ISO8583Message) -> None:
        """
        Add MAC to message.

        Args:
            message (ISO8583Message): Message to add MAC to
        """
        mac = self._calculate_mac(message)
        message.set_field(64, mac)

    async def _process_regular_transaction(self, message: ISO8583Message) -> ISO8583Message:
        """
        Process regular financial transaction messages.

        Args:
            message (ISO8583Message): Transaction message to process

        Returns:
            ISO8583Message: Response message
        """
        self._validate_message(message)
        response = ISO8583Message()
        response.set_mti(self._get_response_mti(message.get_mti()))
        self._copy_original_fields(message, response)

        # Get transaction destination
        destination = None
        if self._requires_routing(message):
            destination = self.routing.get_destination(message)
            # Forward transaction here if needed
            pass

        # Process based on type
        if message.get_mti().startswith("01"):
            await self._process_financial(message, response)
        elif message.get_mti().startswith("04"):
            await self._process_reversal(message, response)

        self._add_mac(response)
        await self.db.store_transaction(message)

        analysis = get_response_analysis(response.get_field(39))
        if analysis["logging_required"]:
            # Add logging here
            pass

        return response

    def _validate_message(self, message: ISO8583Message):
        """
        Validate required message fields.

        Args:
            message (ISO8583Message): Message to validate

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = {
            "0100": [2, 3, 4, 7, 11, 41],
            "0200": [2, 3, 4, 7, 11, 41],
            "0400": [2, 3, 4, 7, 11, 41, 90]
        }

        mti = message.get_mti()
        if mti not in required_fields:
            raise ValueError(f"Unsupported MTI: {mti}")

        for field in required_fields[mti]:
            if field not in message.fields:
                raise ValueError(f"Missing required field: {field}")

    def _get_response_mti(self, request_mti: str) -> str:
        """
        Get response MTI for request MTI.

        Args:
            request_mti (str): Request message MTI

        Returns:
            str: Response message MTI
        """
        return str(int(request_mti) + 10)

    def _copy_original_fields(self, request: ISO8583Message, response: ISO8583Message):
        """
        Copy required fields from request to response.

        Args:
            request (ISO8583Message): Original request message
            response (ISO8583Message): Response message to copy fields to
        """
        for field in [2, 3, 4, 7, 11, 41]:
            if field in request.fields:
                response.set_field(field, request.fields[field])

    async def _process_financial(self, request: ISO8583Message, response: ISO8583Message):
        """
        Process financial transaction message.

        Args:
            request (ISO8583Message): Financial transaction request
            response (ISO8583Message): Response message to populate
        """
        response.set_field(39, "00")
        response.set_field(38, "123456")

    async def _process_reversal(self, request: ISO8583Message, response: ISO8583Message):
        """
        Process reversal transaction message.

        Args:
            request (ISO8583Message): Reversal request message
            response (ISO8583Message): Response message to populate
        """
        original_txn = await self.db.get_transaction(request.fields[11])
        response.set_field(39, "00" if original_txn else "12")

    def _requires_routing(self, message: ISO8583Message) -> bool:
        """
        Check if message requires routing to another institution.

        Args:
            message (ISO8583Message): Message to check routing for

        Returns:
            bool: True if message needs to be routed
        """
        return True if message.fields[2][:6] in self.routing.routing_table else False

    def _create_error_response(self, request: ISO8583Message, error: str, resp_code: str) -> ISO8583Message:
        """
        Create error response message.

        Args:
            request (ISO8583Message): Original request message
            error (str): Error description
            resp_code (str): Response code to use

        Returns:
            ISO8583Message: Error response message
        """
        response = ISO8583Message()
        response.set_mti(self._get_response_mti(request.get_mti()))
        self._copy_original_fields(request, response)
        response.set_field(39, resp_code)
        return response

    def _create_declined_response(self, request: ISO8583Message, resp_code: str) -> ISO8583Message:
        """
        Create declined transaction response.

        Args:
            request (ISO8583Message): Original request message
            resp_code (str): Response code to use

        Returns:
            ISO8583Message: Declined response message
        """
        response = ISO8583Message()
        response.set_mti(self._get_response_mti(request.get_mti()))
        self._copy_original_fields(request, response)
        response.set_field(39, resp_code)
        return response
