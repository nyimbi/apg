from enum import Enum
from typing import Any

class FieldType(Enum):
    NUMERIC = 'n'        # Numeric values
    ALPHA = 'a'         # Alphabetic characters
    SPECIAL = 's'       # Special characters
    ALPHANUMERIC = 'an' # Alpha-numeric
    BINARY = 'b'        # Binary data
    TRACK2 = 'z'        # Track 2 data
    ALPHANUMERIC_SPECIAL = 'ans' # Alpha-numeric-special

class LengthType(Enum):
    FIXED = 'fixed'
    LLVAR = 'll'    # 2-digit length indicator
    LLLVAR = 'lll'  # 3-digit length indicator

class ISO8583Field:
    def __init__(self, field_type: FieldType, max_length: int, length_type: LengthType = LengthType.FIXED):
        self.field_type = field_type
        self.max_length = max_length
        self.length_type = length_type

    def validate(self, value: Any) -> bool:
        """Validate field value against its type and length constraints."""
        str_value = str(value)

        # Check length
        if self.length_type == LengthType.FIXED and len(str_value) != self.max_length:
            return False
        if len(str_value) > self.max_length:
            return False

        # Check type
        if self.field_type == FieldType.NUMERIC:
            return str_value.isdigit()
        elif self.field_type == FieldType.ALPHA:
            return str_value.isalpha()
        elif self.field_type == FieldType.BINARY:
            return isinstance(value, bytes)

        return True
