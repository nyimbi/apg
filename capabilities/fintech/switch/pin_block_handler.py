"""
PIN Block Handler for ISO8583 Messages

This module handles PIN block formatting, encryption, and validation for ISO8583 messages.
It supports multiple PIN block formats and secure key management.
"""

import logging
from typing import Dict, Optional, Union, Tuple
from random import randint
from dataclasses import dataclass
from enum import Enum
from Crypto.Cipher import DES3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pin_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PINBlockFormat(Enum):
    """Supported PIN block formats"""
    ISO0 = "ISO0"
    ISO1 = "ISO1"
    ISO3 = "ISO3"
    ANSI = "ANSI"

class PINError(Exception):
    """Base exception for PIN-related errors"""
    pass

class PINFormatError(PINError):
    """Invalid PIN format"""
    pass

class PINBlockError(PINError):
    """PIN block formatting/encryption error"""
    pass

@dataclass
class PINBlock:
    """Represents a formatted PIN block"""
    format: PINBlockFormat
    value: bytes
    encrypted: bool = False

    def __str__(self):
        return f"PINBlock(format={self.format.value}, encrypted={self.encrypted})"

class PINValidator:
    """Validates PIN and related data"""

    @staticmethod
    def validate_pin(pin: str) -> bool:
        """Validate PIN format and length"""
        if not pin.isdigit():
            raise PINFormatError("PIN must contain only digits")
        if not (4 <= len(pin) <= 12):
            raise PINFormatError("PIN length must be between 4 and 12 digits")
        return True

    @staticmethod
    def validate_pan(pan: str) -> bool:
        """Validate PAN format"""
        if not pan.isdigit():
            raise PINFormatError("PAN must contain only digits")
        if not (12 <= len(pan) <= 19):
            raise PINFormatError("PAN length must be between 12 and 19 digits")
        return True

class PINBlockFormatter:
    """Handles PIN block formatting in various formats"""

    @staticmethod
    def _pad_pin(pin: str) -> str:
        """Pad PIN with random digits"""
        pad_length = 16 - len(pin)
        return f"{len(pin):02d}{pin}{''.join(str(randint(0, 9)) for _ in range(pad_length))}"

    @staticmethod
    def _extract_pan_digits(pan: str) -> str:
        """Extract PAN digits for PIN block"""
        return f"0000{pan[-13:-1]}"

    @classmethod
    def format_pin_block(cls, pin: str, pan: str, format: PINBlockFormat = PINBlockFormat.ISO0) -> PINBlock:
        """Format PIN block in specified format"""
        try:
            PINValidator.validate_pin(pin)
            PINValidator.validate_pan(pan)

            if format == PINBlockFormat.ISO0:
                pin_block_1 = int(cls._pad_pin(pin), 16)
                pan_block = int(cls._extract_pan_digits(pan), 16)
                final_block = pin_block_1 ^ pan_block
                block_value = bytes.fromhex(f"{final_block:016x}")

            elif format == PINBlockFormat.ISO1:
                pin_length = len(pin)
                block_str = f"1{pin_length:01x}{pin}{'F' * (14 - pin_length)}"
                block_value = bytes.fromhex(block_str)

            elif format == PINBlockFormat.ISO3:
                pin_block_1 = int(f"3{len(pin):01x}{pin}{'F' * (14 - len(pin))}", 16)
                pan_block = int(cls._extract_pan_digits(pan), 16)
                final_block = pin_block_1 ^ pan_block
                block_value = bytes.fromhex(f"{final_block:016x}")

            else:
                raise PINBlockError(f"Unsupported PIN block format: {format}")

            return PINBlock(format=format, value=block_value)

        except Exception as e:
            raise PINBlockError(f"Error formatting PIN block: {str(e)}")

class PINBlockCrypto:
    """Handles PIN block encryption/decryption"""

    def __init__(self, zpk: bytes):
        """Initialize with Zone PIN Key"""
        self.zpk = zpk
        self._cipher = DES3.new(zpk, DES3.MODE_ECB)

    def encrypt_pin_block(self, pin_block: PINBlock) -> PINBlock:
        """Encrypt a PIN block"""
        try:
            encrypted_value = self._cipher.encrypt(pin_block.value)
            return PINBlock(
                format=pin_block.format,
                value=encrypted_value,
                encrypted=True
            )
        except Exception as e:
            raise PINBlockError(f"Error encrypting PIN block: {str(e)}")

    def decrypt_pin_block(self, pin_block: PINBlock) -> PINBlock:
        """Decrypt a PIN block"""
        try:
            decrypted_value = self._cipher.decrypt(pin_block.value)
            return PINBlock(
                format=pin_block.format,
                value=decrypted_value,
                encrypted=False
            )
        except Exception as e:
            raise PINBlockError(f"Error decrypting PIN block: {str(e)}")

class ISO8583PINHandler:
    """Handles PIN operations for ISO8583 messages"""

    def __init__(self, zpk: bytes):
        self.formatter = PINBlockFormatter()
        self.crypto = PINBlockCrypto(zpk)

    def prepare_pin_data(
        self,
        pin: str,
        pan: str,
        format: PINBlockFormat = PINBlockFormat.ISO0
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Prepare PIN data for ISO8583 message

        Returns:
            Tuple containing encrypted PIN block and required field updates
        """
        try:
            # Format and encrypt PIN block
            pin_block = self.formatter.format_pin_block(pin, pan, format)
            encrypted_block = self.crypto.encrypt_pin_block(pin_block)

            # Prepare field updates
            field_updates = {
                '52': encrypted_block.value.hex().upper(),
                '22': '901',  # Example: PIN entry capability
                '25': '00',   # Example: PIN capture capability
            }

            return encrypted_block.value, field_updates

        except Exception as e:
            logger.error(f"Error preparing PIN data: {str(e)}")
            raise

    def verify_pin_block(self, encrypted_block: bytes, pan: str) -> bool:
        """Verify PIN block format and encryption"""
        try:
            # Create PIN block object
            pin_block = PINBlock(
                format=PINBlockFormat.ISO0,
                value=encrypted_block,
                encrypted=True
            )

            # Decrypt and validate
            decrypted = self.crypto.decrypt_pin_block(pin_block)

            # Basic format validation
            if len(decrypted.value) != 8:
                return False

            # Format-specific validation
            first_digit = decrypted.value[0] >> 4
            if first_digit not in [0, 1, 3]:
                return False

            return True

        except Exception as e:
            logger.error(f"PIN block verification failed: {str(e)}")
            return False

def create_pin_handler(zpk_hex: str) -> ISO8583PINHandler:
    """Factory function to create PIN handler"""
    try:
        zpk = bytes.fromhex(zpk_hex)
        return ISO8583PINHandler(zpk)
    except Exception as e:
        raise PINError(f"Error creating PIN handler: {str(e)}")

# Usage example:
if __name__ == "__main__":
    # Example key (replace with actual ZPK)
    TEST_ZPK = "0123456789ABCDEF0123456789ABCDEF"

    try:
        # Create PIN handler
        pin_handler = create_pin_handler(TEST_ZPK)

        # Example usage
        pin = "1234"
        pan = "4111111111111111"

        # Prepare PIN data for message
        encrypted_block, field_updates = pin_handler.prepare_pin_data(pin, pan)

        print("Encrypted PIN block:", encrypted_block.hex().upper())
        print("Field updates:", field_updates)

        # Verify PIN block
        is_valid = pin_handler.verify_pin_block(encrypted_block, pan)
        print("PIN block valid:", is_valid)

    except PINError as e:
        print(f"PIN processing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



"""
from pin_block_handler import create_pin_handler, PINError

def process_transaction_with_pin(
    field_data: dict,
    pin: str,
    zpk_hex: str,
    field_specs: dict
) -> dict:
    """Process transaction with PIN data"""
    try:
        # Create PIN handler
        pin_handler = create_pin_handler(zpk_hex)

        # Get PAN from field data
        pan = field_data.get('2')
        if not pan:
            raise ValueError("PAN (Field 2) required for PIN processing")

        # Prepare PIN data
        _, field_updates = pin_handler.prepare_pin_data(pin, pan)

        # Update field data with PIN-related fields
        field_data.update(field_updates)

        return field_data

    except PINError as e:
        logger.error(f"PIN processing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Transaction processing error: {e}")
        raise

# Example usage:
def send_transaction(field_data: dict, pin: Optional[str] = None):
    try:
        if pin:
            zpk_hex = "YOUR_ZPK_HERE"  # Get from secure configuration
            field_data = process_transaction_with_pin(
                field_data,
                pin,
                zpk_hex,
                field_specs
            )

        # Continue with regular transaction processing...

    except Exception as e:
        logger.error(f"Transaction failed: {e}")
        raise
"""
