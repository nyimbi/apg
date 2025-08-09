from typing import Dict, Any, Tuple
import struct
from datetime import datetime
from field import ISO8583Field, FieldType, LengthType

class ISO8583Message:
    """
    Represents an ISO8583 message with methods for parsing, building, and manipulating
    message data according to the ISO8583 financial transaction message specification.
    """

    # MTI Versions
    VERSION_0 = '0' # ISO 8583:1987
    VERSION_1 = '1' # ISO 8583:1993
    VERSION_2 = '2' # ISO 8583:2003

    # MTI Message Classes
    CLASS_AUTH = '1'      # Authorization
    CLASS_FINANCIAL = '2' # Financial
    CLASS_FILE = '3'      # File actions
    CLASS_REVERSAL = '4'  # Reversal
    CLASS_RECON = '5'     # Reconciliation
    CLASS_ADMIN = '6'     # Administrative
    CLASS_FEE = '7'       # Fee collection

    # MTI Function Types
    FUNC_REQUEST = '00'
    FUNC_RESPONSE = '10'
    FUNC_ADVICE = '20'
    FUNC_ADVICE_RESPONSE = '30'

    # Define standard ISO8583 fields specified in ISO 8583:1987
    # Field format: {field_number: ISO8583Field(type, max_length, length_type)}
    FIELDS = {
        # Basic Data Elements
        2: ISO8583Field(FieldType.NUMERIC, 19, LengthType.LLVAR),   # Primary Account Number (PAN)
        3: ISO8583Field(FieldType.NUMERIC, 6),                      # Processing Code (positions 1-2: txn type, 3-4: acct type 1, 5-6: acct type 2)
        4: ISO8583Field(FieldType.NUMERIC, 12),                     # Amount, Transaction (in minor currency units)

        # Date/Time Elements
        7: ISO8583Field(FieldType.NUMERIC, 10),                     # Transmission Date & Time (MMDDhhmmss)
        11: ISO8583Field(FieldType.NUMERIC, 6),                     # System Trace Audit Number (STAN)
        12: ISO8583Field(FieldType.NUMERIC, 6),                     # Time, Local Transaction (hhmmss)
        13: ISO8583Field(FieldType.NUMERIC, 4),                     # Date, Local Transaction (MMDD)
        14: ISO8583Field(FieldType.NUMERIC, 4),                     # Date, Expiration (YYMM)
        15: ISO8583Field(FieldType.NUMERIC, 4),                     # Date, Settlement (MMDD)

        # Merchant Details
        18: ISO8583Field(FieldType.NUMERIC, 4),                     # Merchant Type/Category Code (MCC)
        22: ISO8583Field(FieldType.NUMERIC, 3),                     # Point of Service Entry Mode (POS entry mode)
        23: ISO8583Field(FieldType.NUMERIC, 3),                     # Card Sequence Number
        25: ISO8583Field(FieldType.NUMERIC, 2),                     # Point of Service Condition Code

        # Financial Institution Details
        32: ISO8583Field(FieldType.NUMERIC, 11, LengthType.LLVAR),  # Acquiring Institution Identification Code
        33: ISO8583Field(FieldType.NUMERIC, 11, LengthType.LLVAR),  # Forwarding Institution Identification Code

        # Card Data
        35: ISO8583Field(FieldType.TRACK2, 37, LengthType.LLVAR),   # Track 2 Data
        36: ISO8583Field(FieldType.TRACK3, 104, LengthType.LLLVAR), # Track 3 Data
        37: ISO8583Field(FieldType.ALPHANUMERIC, 12),               # Retrieval Reference Number
        38: ISO8583Field(FieldType.ALPHANUMERIC, 6),                # Authorization Identification Response
        39: ISO8583Field(FieldType.ALPHANUMERIC, 2),                # Response Code (00=approved)

        # Terminal Data
        41: ISO8583Field(FieldType.ALPHANUMERIC_SPECIAL, 8),        # Card Acceptor Terminal Identification
        42: ISO8583Field(FieldType.ALPHANUMERIC_SPECIAL, 15),       # Card Acceptor Identification Code
        43: ISO8583Field(FieldType.ALPHANUMERIC_SPECIAL, 40),       # Card Acceptor Name/Location (1-23: name, 24-36: city, 37-38: state, 39-40: country)

        # Additional Data Elements
        48: ISO8583Field(FieldType.ALPHANUMERIC_SPECIAL, 999, LengthType.LLLVAR), # Additional Data - Private
        49: ISO8583Field(FieldType.NUMERIC, 3),                     # Currency Code, Transaction (ISO 4217)
        50: ISO8583Field(FieldType.NUMERIC, 3),                     # Currency Code, Settlement
        51: ISO8583Field(FieldType.NUMERIC, 3),                     # Currency Code, Cardholder Billing

        # Security Elements
        52: ISO8583Field(FieldType.BINARY, 16),                     # Personal Identification Number (PIN) Data
        53: ISO8583Field(FieldType.NUMERIC, 16),                    # Security Related Control Information
        54: ISO8583Field(FieldType.ALPHANUMERIC_SPECIAL, 120, LengthType.LLLVAR), # Additional Amounts
        55: ISO8583Field(FieldType.BINARY, 999, LengthType.LLLVAR), # ICC Data - EMV Having Multiple Tags
        64: ISO8583Field(FieldType.BINARY, 16),                     # Message Authentication Code (MAC)

        # Settlement Data
        90: ISO8583Field(FieldType.NUMERIC, 42),                    # Original Data Elements
        95: ISO8583Field(FieldType.ALPHANUMERIC, 42),              # Replacement Amounts
        96: ISO8583Field(FieldType.BINARY, 16),                    # Message Security Code
        128: ISO8583Field(FieldType.BINARY, 16)                    # Message Authentication Code (MAC)
    }

    def __init__(self):
        """Initialize an empty ISO8583 message."""
        self.mti: str = ""
        self.primary_bitmap: int = 0
        self.secondary_bitmap: int = 0
        self.fields: Dict[int, Any] = {}

    def set_mti(self, mti: str) -> None:
        """Set the Message Type Indicator."""
        if not isinstance(mti, str) or len(mti) != 4 or not mti.isdigit():
            raise ValueError("MTI must be a 4-digit string")
        self.mti = mti

    def get_mti(self) -> str:
        """Get the Message Type Indicator."""
        return self.mti

    def set_field(self, field_num: int, value: Any) -> None:
        """Set a field value with validation."""
        if field_num not in self.FIELDS:
            raise ValueError(f"Invalid field number: {field_num}")

        field_spec = self.FIELDS[field_num]
        str_value = str(value)

        # Validate field type
        if field_spec.field_type == FieldType.NUMERIC and not str_value.isdigit():
            raise ValueError(f"Field {field_num} requires numeric value")
        elif field_spec.field_type == FieldType.BINARY and not isinstance(value, bytes):
            raise ValueError(f"Field {field_num} requires binary data")

        # Validate length
        if field_spec.length_type == LengthType.FIXED and len(str_value) != field_spec.max_length:
            raise ValueError(f"Field {field_num} requires length {field_spec.max_length}")
        elif len(str_value) > field_spec.max_length:
            raise ValueError(f"Field {field_num} exceeds maximum length {field_spec.max_length}")

        self.fields[field_num] = value
        self._update_bitmap()

    def get_field(self, field_num: int) -> Any:
        """Get a field value."""
        if field_num not in self.fields:
            raise ValueError(f"Field {field_num} not present")
        return self.fields[field_num]

    def _update_bitmap(self) -> None:
        """Update bitmap based on present fields."""
        self.primary_bitmap = 0
        self.secondary_bitmap = 0
        for field_num in self.fields.keys():
            if 1 <= field_num <= 64:
                self.primary_bitmap |= (1 << (64 - field_num))
            elif 65 <= field_num <= 128:
                if self.secondary_bitmap == 0:
                    self.primary_bitmap |= (1 << 63)
                self.secondary_bitmap |= (1 << (128 - field_num))

    def _calculate_bitmaps(self) -> Tuple[int, int]:
        """
        Calculate primary and secondary bitmaps based on present fields.

        Returns:
            Tuple[int, int]: Primary and secondary bitmap values
        """
        primary = 0
        secondary = 0

        for field_num in self.fields.keys():
            if 1 <= field_num <= 64:
                primary |= (1 << (64 - field_num))
            elif 65 <= field_num <= 128:
                if secondary == 0:  # First secondary field
                    primary |= (1 << 63)  # Set bit 1
                secondary |= (1 << (128 - field_num))

        return primary, secondary

    def _parse_length_indicator(self, data: bytes, offset: int, length_type: LengthType) -> Tuple[int, int]:
        """
        Parse variable length indicator.

        Args:
            data: Raw message data
            offset: Current position in data
            length_type: Type of length field

        Returns:
            Tuple[int, int]: Field length and new offset
        """
        if length_type == LengthType.LLVAR:
            length = int(data[offset:offset+2].decode())
            return length, offset + 2
        elif length_type == LengthType.LLLVAR:
            length = int(data[offset:offset+3].decode())
            return length, offset + 3
        return 0, offset

    def _build_length_indicator(self, length: int, length_type: LengthType) -> bytes:
        """
        Build variable length indicator.

        Args:
            length: Field length to encode
            length_type: Type of length field

        Returns:
            bytes: Encoded length indicator
        """
        if length_type == LengthType.LLVAR:
            return str(length).zfill(2).encode()
        elif length_type == LengthType.LLLVAR:
            return str(length).zfill(3).encode()
        return b''

    def parse(self, data: bytes) -> None:
        """
        Parse an ISO8583 message from raw bytes.

        Args:
            data: Raw ISO8583 message data

        Raises:
            ValueError: If parsing fails
        """
        try:
            offset = 0

            # Parse MTI
            self.mti = data[offset:offset+4].decode()
            offset += 4

            # Parse Primary Bitmap
            self.primary_bitmap = int.from_bytes(data[offset:offset+8], 'big')
            offset += 8

            # Check for Secondary Bitmap
            has_secondary = bool(self.primary_bitmap & (1 << 63))
            if has_secondary:
                self.secondary_bitmap = int.from_bytes(data[offset:offset+8], 'big')
                offset += 8

            # Parse Fields
            for field_num in range(2, 129):
                if field_num <= 64:
                    if not (self.primary_bitmap & (1 << (64 - field_num))):
                        continue
                else:
                    if not has_secondary or not (self.secondary_bitmap & (1 << (128 - field_num))):
                        continue

                field_spec = self.FIELDS.get(field_num)
                if not field_spec:
                    continue

                # Handle variable length fields
                if field_spec.length_type != LengthType.FIXED:
                    field_length, offset = self._parse_length_indicator(
                        data, offset, field_spec.length_type
                    )
                else:
                    field_length = field_spec.max_length

                # Parse field data
                field_data = data[offset:offset+field_length]

                # Convert based on field type
                if field_spec.field_type == FieldType.NUMERIC:
                    self.fields[field_num] = int(field_data.decode())
                elif field_spec.field_type == FieldType.BINARY:
                    self.fields[field_num] = field_data
                else:
                    self.fields[field_num] = field_data.decode().strip()

                offset += field_length

        except Exception as e:
            raise ValueError(f"Failed to parse ISO8583 message: {str(e)}")

    def build(self) -> bytes:
        """
        Build an ISO8583 message from current state.

        Returns:
            bytes: Raw ISO8583 message

        Raises:
            ValueError: If building fails
        """
        try:
            data = bytearray()

            # Add MTI
            data.extend(self.mti.encode())

            # Calculate bitmaps
            self.primary_bitmap, self.secondary_bitmap = self._calculate_bitmaps()

            # Add Primary Bitmap
            data.extend(self.primary_bitmap.to_bytes(8, 'big'))

            # Add Secondary Bitmap if needed
            if self.primary_bitmap & (1 << 63):
                data.extend(self.secondary_bitmap.to_bytes(8, 'big'))

            # Add Fields
            for field_num in sorted(self.fields.keys()):
                field_spec = self.FIELDS.get(field_num)
                if not field_spec:
                    continue

                field_data = self.fields[field_num]

                # Convert to bytes based on field type
                if field_spec.field_type == FieldType.NUMERIC:
                    encoded_data = str(field_data).zfill(field_spec.max_length).encode()
                elif field_spec.field_type == FieldType.BINARY:
                    encoded_data = field_data
                else:
                    encoded_data = str(field_data).ljust(field_spec.max_length).encode()

                # Add length indicator for variable length fields
                if field_spec.length_type != LengthType.FIXED:
                    data.extend(self._build_length_indicator(
                        len(encoded_data), field_spec.length_type
                    ))

                data.extend(encoded_data)

            return bytes(data)

        except Exception as e:
            raise ValueError(f"Failed to build ISO8583 message: {str(e)}")

    def __str__(self) -> str:
        """Return a string representation of the message."""
        result = [f"MTI: {self.mti}"]
        result.append(f"Primary Bitmap: {bin(self.primary_bitmap)[2:].zfill(64)}")
        if self.primary_bitmap & (1 << 63):
            result.append(f"Secondary Bitmap: {bin(self.secondary_bitmap)[2:].zfill(64)}")
        for field_num in sorted(self.fields.keys()):
            result.append(f"Field {field_num}: {self.fields[field_num]}")
        return "\n".join(result)

    def clear(self) -> None:
        """Clear all message data."""
        self.mti = ""
        self.primary_bitmap = 0
        self.secondary_bitmap = 0
        self.fields.clear()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ISO8583Message':
        """Create message from bytes data."""
        message = cls()
        message.parse(data)
        return message



# Example usage
def create_authorization_request():
    msg = ISO8583Message()

    # Construct MTI: 0100
    # Version 0 (1987) + Class 1 (Authorization) + Function 00 (Request)
    mti = (ISO8583Message.VERSION_0 +
           ISO8583Message.CLASS_AUTH +
           ISO8583Message.FUNC_REQUEST)
    msg.set_mti(mti)

    # Set fields
    msg.set_field(2, "4111111111111111")  # PAN
    msg.set_field(3, "000000")            # Processing Code
    msg.set_field(4, "000000012500")      # Amount
    msg.set_field(7, "0701123456")        # Transaction Date/Time
    msg.set_field(11, "123456")           # STAN
    msg.set_field(41, "TEST_TERM")        # Terminal ID
    msg.set_field(42, "ID123456789")      # Merchant ID

    return msg

def create_authorization_response(request: ISO8583Message):
    response = ISO8583Message()

    # Create response MTI by adding 10 to request MTI
    response_mti = request.mti[:2] + ISO8583Message.FUNC_RESPONSE
    response.set_mti(response_mti)

    # Copy original fields
    for field in [2, 3, 4, 7, 11, 41, 42]:
        if field in request.fields:
            response.set_field(field, request.fields[field])

    # Add response-specific fields
    response.set_field(39, "00")          # Response Code (Approved)
    response.set_field(38, "A12345")      # Approval Code

    return response

def create_sample_message() -> ISO8583Message:
    """
    Create a sample ISO8583 message for testing purposes.
    This creates a basic authorization request message.

    Returns:
        ISO8583Message: A sample authorization request message.
    """
    message = ISO8583Message()
    message.set_mti("0100")  # Authorization request
    message.set_field(2, 1234567890123456)  # PAN
    message.set_field(3, 000000)  # Processing Code
    message.set_field(4, 100)  # Amount
    message.set_field(7, int(datetime.now().strftime("%m%d%H%M%S")))  # Transmission Date & Time
    message.set_field(11, 123456)  # System Trace Audit Number
    message.set_field(41, "TEST1234")  # Card Acceptor Terminal ID
    message.set_field(42, "TEST MERCHANT ")  # Card Acceptor ID Code
    message.set_field(49, 840)  # Currency Code (USD)
    return message

if __name__ == "__main__":
    # Example usage
    sample_message = create_sample_message()
    print("Sample Message:")
    print(sample_message)

    # Build the message
    built_data = sample_message.build()
    print("\nBuilt Message (hex):")
    print(built_data.hex())

    # Parse the built message
    parsed_message = ISO8583Message()
    parsed_message.parse(built_data)
    print("\nParsed Message:")
    print(parsed_message)

    assert str(sample_message) == str(parsed_message), "Build and parse operations should result in identical messages"
    print("\nBuild and parse operations successful!")
