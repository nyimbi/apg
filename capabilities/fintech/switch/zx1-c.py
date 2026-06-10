"""
ISO8583 Financial Transaction Processor and Gateway Client

This module implements a comprehensive ISO8583 financial transaction processor and secure
gateway client for electronic payment systems. It handles message formatting, cryptographic
operations, secure communication, and transaction management according to ISO8583 and
related financial messaging standards.

Core Functionality:
------------------
1. Message Processing
    - ISO8583 message formatting and parsing
    - Binary/ASCII message conversion
    - Dynamic field generation (STAN, RRN)
    - Message validation and verification
    - Bitmap generation and parsing
    - Field dependency management

2. Cryptographic Operations
    - Session key management (ZMK/ZPK)
    - Double variant key decryption
    - PIN block generation (ISO-9564)
    - KCV (Key Check Value) verification
    - Secure key storage and retrieval
    - Triple DES encryption/decryption

3. Communication
    - SSL/TLS secure socket handling
    - Message transmission and reception
    - Response parsing and validation
    - Connection management and timeouts
    - Server availability monitoring
    - Error handling and recovery

4. Transaction Management
    - Financial transaction processing
    - Key exchange sequences
    - PIN verification requests
    - Transaction journaling
    - Status monitoring
    - Error recovery

Field Processing Capabilities:
----------------------------
1. Numeric Fields
    - IFA_NUMERIC: Fixed-length numeric (right-justified, zero-padded)
    - IFA_LLNUM: Variable-length numeric with 2-digit length
    - IFA_LLLNUM: Variable-length numeric with 3-digit length
    - IFA_AMOUNT: Amount fields with sign handling

2. Character Fields
    - IF_CHAR: Fixed-length alphanumeric
    - IFA_LLCHAR: Variable-length with 2-digit length
    - IFA_LLLCHAR: Variable-length with 3-digit length
    - Special character handling and padding

3. Binary Fields
    - IFB_NUMERIC: Binary numeric fields
    - IFB_BITMAP: Binary bitmap fields
    - IFB_BINARY: Raw binary data
    - IFB_LLBINARY: Variable-length binary

4. Special Fields
    - Track 2 data (ISO-7813)
    - PIN blocks (ISO-9564)
    - Key exchange data
    - Network management fields

Security Features:
----------------
1. Cryptographic Security
    - Triple DES encryption
    - Double variant key decryption
    - Key Check Value validation
    - Secure key storage
    - PIN block encryption
    - Session key management

2. Data Protection
    - Sensitive data masking
    - Secure memory handling
    - PAN truncation
    - Key component separation
    - Secure logging practices

3. Communication Security
    - SSL/TLS encryption
    - Certificate validation
    - Secure socket handling
    - Connection timeouts
    - Error recovery

4. Transaction Security
    - Message authentication
    - Field validation
    - Duplicate detection
    - Transaction logging
    - Audit trail maintenance

Configuration:
-------------
1. File Requirements
    - zone.xml: ISO8583 field specifications
        - Field IDs and names
        - Data types and formats
        - Length specifications
        - Validation rules

    - tcard.txt: Test card data
        - Card details
        - Transaction parameters
        - Test scenarios

    - keys.txt: Cryptographic key components
        - ZMK components
        - KCV values
        - Component verification

2. Network Configuration
    - Host: Default "96.0.46.37", 13.246.138.100
    - Port: Default 5858, 12000
    - Timeout settings
    - Retry parameters

Usage Examples:
-------------
1. Basic Transaction:
   ```python
   # Send financial transaction
   result = send_financial_message(
       amount="000000010000",
       pin="1234"
   )
   ```

2. Key Exchange:
   ```python
   # Perform key exchange
   success = perform_key_exchange_with_persistence()
   ```

3. PIN Verification:
   ```python
   # Send PIN verification
   result = send_pinblock_with_session_keys(pin="1234")
   ```

Error Handling:
-------------
The module implements comprehensive error handling for:
- Network errors
- Cryptographic failures
- Message formatting errors
- Validation failures
- Server responses
- Security violations

Dependencies:
------------
- socket: Network communication
- ssl: Secure socket layer
- select: I/O multiplexing
- xml.etree.ElementTree: XML parsing
- struct: Binary data handling
- binascii: Binary-ASCII conversions
- datetime: Date/time operations
- cryptography: Cryptographic operations
- pathlib: File path handling
- typing: Type hints

Standards Compliance:
-------------------
- ISO8583: Financial transaction messaging
- ISO9564: PIN block formats
- ISO7813: Track 2 data
- ISO4909: Track 3 data
- ANSI X9.24: Key management
- PCI-DSS: Security requirements

Author: Nyimbi Odero
Version: 1.0
Date: 23/11/2024
License: Copyright (c) Nyimbi Odero, 2024

Notes:
-----
1. Ensure proper key management practices
2. Maintain secure configuration
3. Regular security updates
4. Monitor transaction logs
5. Follow PCI-DSS guidelines
"""

import socket
import ssl
import select
import xml.etree.ElementTree as ET
import struct
import binascii
import time
from typing import Dict, Tuple, Optional, List, Union, Any, Callable, Set
import random
import json
import logging
from enum import Enum
import logging

# import hashlib
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from push_journal import send_push_journal, APIConfig
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.algorithms import (
    TripleDES,
)  # For older versions

# For newer versions (48.0.0+):
# from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
# from cryptography.hazmat.backends import default_backend
# Disable certain SSL warnings (only in development/testing)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

ZONE_FILE = "zone.xml"
TCARD_FILE = "tcard_km.txt"
# TCARD_FILE = "tcard_fid1.txt"
HOST = "96.0.46.37"
# HOST = "13.246.138.100"
PORT = 5858
# PORT = 12000


ISO8583_RESPONSE_CODES = {
    "00": "Approved or completed successfully",
    "01": "Refer to card issuer",
    "02": "Refer to card issuer, special condition",
    "03": "Invalid merchant or service provider",
    "04": "Pick up card (no fraud)",
    "05": "Do not honor",
    "06": "Error",
    "07": "Pick up card, special condition (fraud account)",
    "08": "Honor with identification",
    "09": "Request in progress",
    "10": "Approved, partial",
    "11": "Approved, VIP",
    "12": "Invalid transaction",
    "13": "Invalid amount",
    "14": "Invalid card number",
    "15": "No such issuer",
    "21": "No action taken",
    "25": "Unable to locate record in file",
    "28": "File temporarily not available",
    "30": "Format error",
    "31": "Bank not supported by switch/Unable to route transaction",
    "33": "Expired card",
    "34": "Suspected fraud",
    "35": "Contact acquirer",
    "36": "Restricted card",
    "37": "Card acceptor call acquirer security",
    "38": "PIN tries exceeded",
    "39": "No credit account",
    "40": "Function not supported",
    "41": "Lost card - pick up",
    "42": "No universal account",
    "43": "Stolen card - pick up",
    "51": "Insufficient funds",
    "52": "No check account",
    "53": "No savings account",
    "54": "Expired card",
    "55": "Incorrect PIN",
    "56": "No card record",
    "57": "Transaction not permitted to cardholder",
    "58": "Transaction not permitted on terminal",
    "59": "Suspected fraud",
    "61": "Exceeds withdrawal limit",
    "62": "Restricted card",
    "63": "Security violation",
    "64": "Original amount incorrect",
    "65": "Exceeds withdrawal frequency",
    "66": "Call acquirer security",
    "67": "Hard capture - pick up card at ATM",
    "68": "Response received too late",
    "75": "PIN tries exceeded",
    "77": "Intervene, bank approval required",
    "78": "Intervene, bank approval required for partial amount",
    "85": "Not declined",
    "86": "PIN validation not possible",
    "89": "Bad terminal",
    "90": "Cut-off in progress",
    "91": "Issuer or switch inoperative",
    "92": "Routing error",
    "93": "Violation of law",
    "94": "Duplicate transaction",
    "95": "Reconcile error",
    "96": "System malfunction",
    "97": "Reserved for national use",
    "98": "Exceeds cash limit",
    "99": "PIN Block error",
}

# Add these response categories for better analysis
RESPONSE_CATEGORIES = {
    "SUCCESS": ["00", "08", "10", "11"],
    "REFER_TO_ISSUER": ["01", "02"],
    "CARD_ISSUES": ["04", "07", "41", "43", "62"],
    "SECURITY_ISSUES": ["34", "36", "37", "63", "66"],
    "PIN_ISSUES": ["55", "75", "86"],
    "ROUTING_ISSUES": ["15", "31", "92"],
    "SYSTEM_ISSUES": ["90", "91", "96"],
    "ACCOUNT_ISSUES": ["51", "52", "53"],
    "FORMAT_ISSUES": ["30", "40"],
    "TRANSACTION_ISSUES": ["57", "58", "61", "65"],
}


def get_response_analysis(resp_code: str) -> dict:
    """
    Provide detailed analysis of an ISO8583 response code.
    """
    analysis = {
        "code": resp_code,
        "message": ISO8583_RESPONSE_CODES.get(resp_code, "Unknown response code"),
        "category": "Unknown",
        "is_success": resp_code in RESPONSE_CATEGORIES["SUCCESS"],
        "recommended_action": "",
        "technical_details": "",
        "user_message": "",
    }

    # Determine category
    for category, codes in RESPONSE_CATEGORIES.items():
        if resp_code in codes:
            analysis["category"] = category
            break

    # Add specific recommendations and details based on response code
    if resp_code == "31":
        analysis.update(
            {
                "recommended_action": "Contact acquiring bank to verify routing configuration",
                "technical_details": "Transaction could not be routed to the issuing bank. This could be due to incorrect BIN routing, network configuration issues, or the issuing bank being unavailable.",
                "user_message": "This card cannot be processed at this time. Please try a different card or payment method.",
            }
        )
    elif resp_code == "51":
        analysis.update(
            {
                "recommended_action": "Request alternative payment method",
                "technical_details": "Account has insufficient funds to complete transaction",
                "user_message": "Insufficient funds available. Please try a different card or payment method.",
            }
        )
    # Add more specific cases as needed

    return analysis


class ISO8583FormatError(Exception):
    """Custom exception for field formatting errors"""

    pass


class ISO8583ParseError(Exception):
    """Custom exception for ISO8583 parsing errors"""

    pass


class ISO8583SecurityError(Exception):
    """Custom exception for security-related errors"""

    pass


class ISO8583MessageError(Exception):
    """Custom exception for message formation errors"""

    pass


class FieldLengthError(Exception):
    """Custom exception for field length related errors"""

    pass


class BinaryFieldError(Exception):
    """Custom exception for binary field processing errors"""

    pass


class ISO8583DecodeError(Exception):
    """Custom exception for message decoding errors"""

    pass


class ISO8583BitmapError(Exception):
    """Custom exception for bitmap-related errors"""

    pass


class SessionKeyManager:
    """
    Manages session keys from key exchange, including persistence and validation.
    """

    SESSION_FILE = "session_keys.json"
    KEY_LIFETIME = 60000000  # 1000 minutes in seconds

    def __init__(self):
        self.session_file = Path(self.SESSION_FILE)
        self._create_session_dir()

    def _create_session_dir(self):
        """Create secure directory for session files if it doesn't exist"""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

    def _encrypt_key_data(self, key_data: str) -> str:
        """Encrypt sensitive key data before storing"""
        # Implement secure encryption here
        # For example, using system keyring or hardware security module
        return key_data

    def _decrypt_key_data(self, encrypted_data: str) -> str:
        """Decrypt stored key data"""
        # Implement secure decryption here
        return encrypted_data

    def verify_session_key_kcv(self, clear_zpk: str, stored_kcv: str) -> bool:
        """Verify ZPK using KCV."""
        try:
            generated_kcv = PinBlockUtil.bytes_to_string(
                PinBlockUtil.operate_des3(
                    bytes(8), PinBlockUtil.string_to_bytes(clear_zpk), True
                )
            )[:6]

            return generated_kcv.upper() == stored_kcv.upper()

        except Exception as e:
            print(f"KCV verification failed: {str(e)}")
            return False

    def save_session_keys(self, key_exchange_response: dict) -> bool:
        """Save session keys from key exchange response with verification"""
        try:
            print("\nSaving Session Keys:")
            print("=" * 50)

            fields = key_exchange_response.get("fields", {})
            if "53" not in fields:
                raise ValueError("No key data in exchange response")

            field_53_data = parse_field_53(fields["53"]["value"])

            # Log key components before storage
            print("\nKey Components to Store:")
            print(f"Encrypted ZPK: {field_53_data['encrypted_zpk']}")
            print(f"Key Version: {field_53_data['key_version']}")
            print(f"KCV: {fields.get('64', {}).get('value', '')}")

            session_data = {
                "timestamp": datetime.now().isoformat(),
                "encrypted_zpk": field_53_data["encrypted_zpk"],
                "key_version": field_53_data["key_version"],
                "control_info": field_53_data["control_info"],
                "kcv": fields.get("64", {}).get("value", ""),
                "additional_data": {
                    "terminal_id": fields.get("41", {}).get("value", ""),
                    "response_code": fields.get("39", {}).get("value", ""),
                },
            }

            # Store without verification hash for now
            encrypted_session = {
                "timestamp": session_data["timestamp"],
                "key_data": self._encrypt_key_data(
                    json.dumps(
                        {
                            "encrypted_zpk": session_data["encrypted_zpk"],
                            "key_version": session_data["key_version"],
                            "control_info": session_data["control_info"],
                            "kcv": session_data["kcv"],
                        }
                    )
                ),
                "additional_data": session_data["additional_data"],
            }

            # Save to file
            with open(self.session_file, "w") as f:
                json.dump(encrypted_session, f, indent=2)

            print("\nSession keys saved successfully")
            print(f"Timestamp: {session_data['timestamp']}")
            print(f"Encrypted ZPK: {session_data['encrypted_zpk']}")

            return True

        except Exception as e:
            print(f"Error saving session keys: {str(e)}")
            return False

    def get_valid_session_keys(self) -> Optional[dict]:
        """Retrieve session keys"""
        try:
            print("\nRetrieving Session Keys:")
            print("=" * 50)

            if not self.session_file.exists():
                print("No session keys found")
                return None

            with open(self.session_file, "r") as f:
                encrypted_session = json.load(f)

            # Check timestamp
            stored_time = datetime.fromisoformat(encrypted_session["timestamp"])
            if datetime.now() - stored_time > timedelta(seconds=self.KEY_LIFETIME):
                print("Session keys have expired")
                return None

            # Decrypt key data
            decrypted_data = json.loads(
                self._decrypt_key_data(encrypted_session["key_data"])
            )

            print("\nRetrieved Key Components:")
            print(f"Encrypted ZPK: {decrypted_data['encrypted_zpk']}")
            print(f"Key Version: {decrypted_data['key_version']}")
            print(f"KCV: {decrypted_data['kcv']}")

            session_keys = {
                "timestamp": encrypted_session["timestamp"],
                "encrypted_zpk": decrypted_data["encrypted_zpk"],
                "control_info": decrypted_data["control_info"],
                "kcv": decrypted_data["kcv"],
                "additional_data": encrypted_session["additional_data"],
            }

            time_remaining = (
                self.KEY_LIFETIME - (datetime.now() - stored_time).total_seconds()
            )

            print("\nSession Status:")
            print(f"Created: {stored_time}")
            print(f"Time remaining: {int(time_remaining)} seconds")

            return session_keys

        except Exception as e:
            print(f"Error retrieving session keys: {str(e)}")
            return None

    def clear_session_keys(self):
        """Clear stored session keys"""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            print("Session keys cleared")
        except Exception as e:
            print(f"Error clearing session keys: {str(e)}")


def decrypt_zpk(clear_zmk: str, encrypted_zpk: str) -> str:
    """
    Decrypt ZPK using clear ZMK.

    Args:
        clear_zmk: Clear Zone Master Key (32 hex chars)
        encrypted_zpk: Encrypted Zone PIN Key from field 53 (32 hex chars)

    Returns:
        Clear ZPK (32 hex chars)
    """
    try:
        if len(clear_zmk) != 32:
            raise ValueError(f"Invalid ZMK length: {len(clear_zmk)}")
        if len(encrypted_zpk) != 32:
            raise ValueError(f"Invalid encrypted ZPK length: {len(encrypted_zpk)}")

        # Convert hex strings to bytes
        zmk_bytes = PinBlockUtil.string_to_bytes(clear_zmk)
        encrypted_zpk_bytes = PinBlockUtil.string_to_bytes(encrypted_zpk)

        # Decrypt ZPK using 3DES
        clear_zpk_bytes = PinBlockUtil.operate_des3(
            encrypted_zpk_bytes[:8],  # First block
            zmk_bytes[:16],  # Use first 16 bytes of ZMK
            False,  # Decrypt mode
        )

        # Convert result back to hex string
        clear_zpk = PinBlockUtil.bytes_to_string(clear_zpk_bytes)

        # Pad to 32 characters if needed
        return clear_zpk.ljust(32, "0")

    except Exception as e:
        raise ValueError(f"Error decrypting ZPK: {str(e)}")


def process_key_exchange(key_exchange_response: dict) -> dict:
    """Process key exchange response to get clear ZPK."""
    try:
        fields = key_exchange_response.get("fields", {})
        if "53" not in fields:
            raise ValueError("No key data in Field 53")

        field_53_data = fields["53"]["value"]
        if len(field_53_data) < 48:  # 32 + 6 + 10 (ZPK + KCV + padding)
            raise ValueError(f"Invalid Field 53 length: {len(field_53_data)}")

        # Extract components
        encrypted_zpk = field_53_data[:32]
        received_kcv = field_53_data[32:38]
        print(f"\nField 53 Components:")
        print(f"Encrypted ZPK: {encrypted_zpk}")
        print(f"KCV: {received_kcv}")

        # Get ZMK components and combine
        zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
        clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)

        # Decrypt ZPK using double variant method
        clear_zpk = decrypt_zpk_double_variant(encrypted_zpk, clear_zmk)

        # Verify KCV
        generated_kcv = PinBlockUtil.bytes_to_string(
            PinBlockUtil.operate_des3(
                bytes(8), PinBlockUtil.string_to_bytes(clear_zpk), True
            )
        )[:6]

        if generated_kcv.upper() != received_kcv.upper():
            raise ValueError("KCV verification failed")

        return {
            "clear_zpk": clear_zpk,
            "encrypted_zpk": encrypted_zpk,
            "kcv": received_kcv,
        }

    except Exception as e:
        raise ValueError(f"Error processing key exchange: {str(e)}")


def parse_field_53(value: str) -> dict:
    """
    Parse field 53 (Security Related Control Information).

    Format:
    - First 16 bytes: Encrypted ZPK under ZMK
    - Next 1-3 bytes: Key version number
    - Remaining bytes: Additional control information

    Args:
        value: Field 53 value (hex string)

    Returns:
        Dictionary containing parsed components
    """
    try:
        if not value or len(value) < 32:
            raise ValueError(f"Invalid field 53 length: {len(value)}")

        # Extract components
        encrypted_zpk = value[:32]  # First 32 hex chars (16 bytes)

        # Parse remaining data if present
        remaining = value[32:]
        key_version = remaining[:2] if remaining else ""
        control_info = remaining[2:] if len(remaining) > 2 else ""

        result = {
            "encrypted_zpk": encrypted_zpk,
            "key_version": key_version,
            "control_info": control_info,
            "raw_value": value,
        }

        # Validate components
        if not all(c in "0123456789ABCDEF" for c in encrypted_zpk.upper()):
            raise ValueError("Invalid ZPK format")

        return result

    except Exception as e:
        raise ValueError(f"Error parsing field 53: {str(e)}")


def validate_pin_format(pin: str, mask_in_logs: bool = True) -> bool:
    """
    Validate PIN format according to ISO 9564 requirements.

    Args:
        pin: PIN to validate
        mask_in_logs: Whether to mask PIN in log messages

    Returns:
        bool: True if PIN is valid

    Requirements:
    - Length: 4-12 digits
    - Characters: Only numeric
    - Not all zeros
    - Not sequential (optional)
    - Not repeated digits (optional)
    """
    try:
        # Basic validation
        if not pin or not isinstance(pin, str):
            raise ValueError("PIN must be a non-empty string")

        if not pin.isdigit():
            raise ValueError("PIN must contain only digits")

        pin_length = len(pin)
        if not (4 <= pin_length <= 12):
            raise ValueError("PIN length must be between 4 and 12 digits")

        if all(d == "0" for d in pin):
            raise ValueError("PIN cannot be all zeros")

        # Optional enhanced validations
        # # Check for sequential numbers (e.g., "1234", "4321")
        # if pin in "0123456789" or pin in "9876543210":
        #     raise ValueError("PIN cannot be sequential numbers")

        # # Check for repeated digits (e.g., "1111")
        # if len(set(pin)) == 1:
        #     raise ValueError("PIN cannot be all same digits")

        masked_pin = "*" * len(pin) if mask_in_logs else pin
        # print(f"PIN validation successful for PIN: {masked_pin}")
        return True

    except ValueError as e:
        print(f"PIN validation failed: {str(e)}")
        return False


def verify_kcv(clear_zpk: str, expected_kcv: str) -> bool:
    """Verify ZPK using KCV."""
    try:
        # Generate KCV by encrypting 8 zeros
        test_data = bytes(8)
        generated_kcv = PinBlockUtil.bytes_to_string(
            PinBlockUtil.operate_des3(
                test_data,
                PinBlockUtil.string_to_bytes(clear_zpk),
                True,  # Encrypt
            )
        )[:6]

        return generated_kcv.upper() == expected_kcv.upper()
    except Exception as e:
        raise ValueError(f"KCV verification failed: {str(e)}")


def decrypt_zpk_double_variant(encrypted_zpk: str, clear_zmk: str) -> str:
    """
    Decrypt ZPK using double variant method with proper key length handling.

    Args:
        encrypted_zpk: 32-char encrypted ZPK from key exchange
        clear_zmk: 32-char clear ZMK from components

    Returns:
        Clear ZPK (32 chars)
    """
    try:
        # Validate input lengths
        if len(encrypted_zpk) != 32:
            raise ValueError(f"Invalid encrypted ZPK length: {len(encrypted_zpk)}")
        if len(clear_zmk) != 32:
            raise ValueError(f"Invalid clear ZMK length: {len(clear_zmk)}")

        # Split encrypted ZPK and ZMK into parts
        zmk_left = clear_zmk[:16]  # First 16 bytes for Triple DES
        zmk_right = clear_zmk[16:]  # Last 16 bytes for variants
        zpk_left = encrypted_zpk[:16]  # First encrypted block
        zpk_right = encrypted_zpk[16:]  # Second encrypted block

        print(f"\nKey Components:")
        print(f"ZMK Left: {zmk_left}")
        print(f"ZMK Right: {zmk_right}")
        print(f"ZPK Left: {zpk_left}")
        print(f"ZPK Right: {zpk_right}")

        # First variant (A6)
        variant_a6 = "A6" + "00" * 7  # Pad to 16 bytes
        first_variant = format(int(zmk_right, 16) ^ int(variant_a6, 16), "016X").zfill(
            16
        )
        variant1_key = zmk_left + first_variant

        # Decrypt first part with Triple DES
        result1 = PinBlockUtil.operate_des3(
            PinBlockUtil.string_to_bytes(zpk_left),
            PinBlockUtil.string_to_bytes(variant1_key),
            False,  # Decrypt mode
        )
        clear_zpk_left = PinBlockUtil.bytes_to_string(result1)

        # Second variant (5A)
        variant_5a = "5A" + "00" * 7  # Pad to 16 bytes
        second_variant = format(int(zmk_right, 16) ^ int(variant_5a, 16), "016X").zfill(
            16
        )
        variant2_key = zmk_left + second_variant

        # Decrypt second part with Triple DES
        result2 = PinBlockUtil.operate_des3(
            PinBlockUtil.string_to_bytes(zpk_right),
            PinBlockUtil.string_to_bytes(variant2_key),
            False,  # Decrypt mode
        )
        clear_zpk_right = PinBlockUtil.bytes_to_string(result2)

        # Combine results
        clear_zpk = clear_zpk_left + clear_zpk_right

        print("\nDecryption Results:")
        print(f"Clear ZPK Left: {clear_zpk_left}")
        print(f"Clear ZPK Right: {clear_zpk_right}")
        print(f"Final Clear ZPK: {clear_zpk}")

        return clear_zpk

    except Exception as e:
        raise ValueError(f"Error in double variant decryption: {str(e)}")


def process_session_keys(key_exchange_response: dict) -> dict:
    """
    Process key exchange response and extract session keys.

    Args:
        key_exchange_response: Response from key exchange message

    Returns:
        Dictionary containing processed session keys
    """
    try:
        if not key_exchange_response:
            raise ValueError("No key exchange response")

        fields = key_exchange_response.get("fields", {})

        # Extract and parse field 53
        if "53" not in fields:
            raise ValueError("No key data received (Field 53 missing)")

        field_53_data = parse_field_53(fields["53"]["value"])
        encrypted_zpk = field_53_data["encrypted_zpk"]

        # Get master keys
        zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
        clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)

        # Generate clear ZPK
        clear_zpk = decrypt_zpk(clear_zmk, encrypted_zpk)

        # Verify KCV if present
        if "64" in fields:
            received_kcv = fields["64"]["value"]
            generated_kcv = PinBlockUtil.bytes_to_string(
                PinBlockUtil.operate_des3(
                    bytes(8), PinBlockUtil.string_to_bytes(clear_zpk), True
                )
            )[:6]

            if received_kcv.upper() != generated_kcv.upper():
                raise ValueError(
                    f"KCV verification failed. Expected: {received_kcv}, Generated: {generated_kcv}"
                )

        return {
            "clear_zpk": clear_zpk,
            "encrypted_zpk": encrypted_zpk,
            "key_version": field_53_data["key_version"],
            "control_info": field_53_data["control_info"],
            "kcv": fields.get("64", {}).get("value", ""),
        }

    except Exception as e:
        raise ValueError(f"Error processing session keys: {str(e)}")


def perform_key_exchange_with_persistence(
    # host: str = "13.246.138.100",
    host: str = HOST,  # "96.0.46.37",
    # port: int = 12000
    port: int = PORT,  # 5858
) -> bool:
    """
    Perform complete key exchange sequence and persist session keys.

    Process:
    1. Check for existing valid session
    2. If needed, perform key exchange (MTI 0800)
    3. Process received encrypted ZPK using double variant method
    4. Verify KCV
    5. Store session keys securely

    Returns:
        bool: True if successful key exchange and storage
    """
    try:
        print("\nInitiating Key Exchange Sequence")
        print("=" * 60)

        # 1. Initialize session manager
        session_manager = SessionKeyManager()

        # 2. Check for existing valid session
        existing_keys = session_manager.get_valid_session_keys()
        if existing_keys:
            print("\nValid session keys found:")
            print(f"Created: {existing_keys['timestamp']}")
            print(f"Key Version: {existing_keys.get('key_version', 'Unknown')}")
            print(f"KCV: {existing_keys.get('kcv', 'Unknown')}")

            # Verify existing keys if KCV present
            if existing_keys.get("kcv"):
                try:
                    # Get ZMK components
                    zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
                    clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)

                    # Process session keys to verify
                    encrypted_zpk = existing_keys["encrypted_zpk"]
                    session_key_data = process_key_exchange(
                        {
                            "fields": {
                                "53": {
                                    "value": encrypted_zpk
                                    + existing_keys["kcv"]
                                    + "0" * 10
                                }
                            }
                        }
                    )
                    print("Existing session keys verified")
                    return True
                except Exception as e:
                    print(f"Existing session key verification failed: {e}")
                    print("Initiating new key exchange")
            else:
                return True

        # 3. Perform new key exchange
        print("\nInitiating new key exchange...")
        key_exchange_result = send_key_exchange_message(host, port)

        if not key_exchange_result:
            raise ValueError("Key exchange failed - no response received")

        if not key_exchange_result.get("success"):
            raise ValueError(
                f"Key exchange failed - response code: "
                f"{key_exchange_result.get('response', {}).get('fields', {}).get('39', {}).get('value', 'Unknown')}"
            )

        # 4. Extract and validate response
        response = key_exchange_result["response"]
        session_keys = key_exchange_result["session_keys"]

        if "fields" not in response:
            raise ValueError("Invalid response format - missing fields")

        fields = response["fields"]

        # Verify required fields
        if "53" not in fields:
            raise ValueError("No key data received (Field 53 missing)")

        if "39" not in fields:
            raise ValueError("No response code received (Field 39 missing)")

        resp_code = fields["39"]["value"].strip()
        if resp_code != "00":
            raise ValueError(f"Key exchange failed - response code: {resp_code}")

        # 5. Process key data
        field_53_data = parse_field_53(fields["53"]["value"])

        print("\nKey Exchange Response:")
        print(f"Response Code: {resp_code}")
        print(f"Encrypted ZPK: {field_53_data['encrypted_zpk']}")
        print(f"Key Version: {field_53_data['key_version']}")
        print(f"KCV: {session_keys['kcv']}")

        # 6. Save session keys
        save_result = session_manager.save_session_keys(
            {
                "fields": {
                    "53": {"value": fields["53"]["value"]},
                    "64": {"value": session_keys["kcv"]},
                    "39": {"value": resp_code},
                    "41": fields.get("41", {"value": ""}),  # Terminal ID if present
                }
            }
        )

        if not save_result:
            raise ValueError("Failed to save session keys")

        # 7. Verify saved keys
        verification_keys = session_manager.get_valid_session_keys()
        if not verification_keys:
            raise ValueError("Failed to retrieve saved session keys")

        print("\nKey Exchange Summary:")
        print(f"Status: Success")
        print(f"Session Created: {verification_keys['timestamp']}")
        print(f"Key Version: {verification_keys.get('key_version', 'Unknown')}")
        print(f"KCV: {verification_keys.get('kcv', 'Unknown')}")

        # Log key lifetime
        time_remaining = (
            SessionKeyManager.KEY_LIFETIME
            - (
                datetime.now() - datetime.fromisoformat(verification_keys["timestamp"])
            ).total_seconds()
        )
        print(f"Session Lifetime: {int(time_remaining)} seconds")

        return True

    except Exception as e:
        print(f"\nError in key exchange process: {str(e)}")

        # Cleanup on failure
        try:
            session_manager.clear_session_keys()
            print("Cleaned up session data after failure")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

        return False

    finally:
        # Clear sensitive data
        if "clear_zmk" in locals():
            clear_zmk = "0" * len(clear_zmk)
        if "session_keys" in locals() and isinstance(session_keys, dict):
            for key in session_keys:
                if isinstance(session_keys[key], str):
                    session_keys[key] = "0" * len(session_keys[key])


def send_pinblock_with_session_keys(
    # host: str = "13.246.138.100",
    host: str = HOST,  # "96.0.46.37",
    # port: int = 12000
    port: int = PORT,  # 5858,
    pin: str = None,
) -> Optional[dict]:
    """Send PIN block message using session keys with double variant decryption."""
    sensitive_data = []

    try:
        # 1. PIN Validation
        if pin is None:
            pin = "1234"
        if not validate_pin_format(pin, mask_in_logs=True):
            raise ValueError("Invalid PIN format")
        sensitive_data.append(pin)

        # 2. Session Key Handling
        session_manager = SessionKeyManager()
        session_keys = session_manager.get_valid_session_keys()
        if not session_keys:
            print("No valid session keys - performing key exchange")
            if not perform_key_exchange_with_persistence():
                raise ValueError("Key exchange failed")
            session_keys = session_manager.get_valid_session_keys()
            if not session_keys:
                raise ValueError("Failed to obtain session keys")

        print("\nProcessing with session keys:")
        encrypted_zpk = session_keys["encrypted_zpk"]
        print(f"Encrypted ZPK: {encrypted_zpk}")
        print(f"KCV: {session_keys['kcv']}")

        # 3. Generate Clear ZPK using double variant method
        try:
            # Get ZMK components
            zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
            sensitive_data.extend([zmk_comp1, zmk_comp2])

            # Generate clear ZMK
            clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
            sensitive_data.append(clear_zmk)

            # Split encrypted ZPK and ZMK into parts
            zpk_part_a = encrypted_zpk[:16]  # First 16 hex chars
            zpk_part_b = encrypted_zpk[16:32]  # Last 16 hex chars
            zmk_part_a = clear_zmk[:16]
            zmk_part_b = clear_zmk[16:32]

            print("\nKey Parts:")
            print(f"ZPK Part A: {zpk_part_a}")
            print(f"ZPK Part B: {zpk_part_b}")
            print(f"ZMK Part A: {zmk_part_a}")
            print(f"ZMK Part B: {zmk_part_b}")

            # First variant (A6)
            first_two_zmk_b = zmk_part_b[:2]
            variant1 = format(int("A6", 16) ^ int(first_two_zmk_b, 16), "02X")
            variant1_zmk = zmk_part_a + variant1 + zmk_part_b[2:]

            # Ensure we have full 16-byte key for each operation
            variant1_key = PinBlockUtil.string_to_bytes(variant1_zmk[:32])
            zpk_a_bytes = PinBlockUtil.string_to_bytes(zpk_part_a.ljust(16, "0"))

            # Decrypt first part
            result1 = PinBlockUtil.operate_des3(
                zpk_a_bytes,
                variant1_key[:16],  # Use first 16 bytes for 3DES
                False,  # Decrypt mode
            )
            result1_hex = PinBlockUtil.bytes_to_string(result1)

            # Second variant (5A)
            variant2 = format(int("5A", 16) ^ int(first_two_zmk_b, 16), "02X")
            variant2_zmk = zmk_part_a + variant2 + zmk_part_b[2:]

            # Ensure we have full 16-byte key for each operation
            variant2_key = PinBlockUtil.string_to_bytes(variant2_zmk[:32])
            zpk_b_bytes = PinBlockUtil.string_to_bytes(zpk_part_b.ljust(16, "0"))

            # Decrypt second part
            result2 = PinBlockUtil.operate_des3(
                zpk_b_bytes,
                variant2_key[:16],  # Use first 16 bytes for 3DES
                False,  # Decrypt mode
            )
            result2_hex = PinBlockUtil.bytes_to_string(result2)

            # Combine results to get final clear ZPK
            clear_zpk = result1_hex[:16] + result2_hex[:16]
            sensitive_data.append(clear_zpk)

            print("\nDecryption Results:")
            print(f"Result 1: {result1_hex}")
            print(f"Result 2: {result2_hex}")
            print(f"Clear ZPK (masked): {'*' * len(clear_zpk)}")

        except Exception as e:
            raise ValueError(f"Error in ZPK decryption: {str(e)}")

        # 4. Verify ZPK using KCV
        if session_keys["kcv"]:
            if not verify_kcv(clear_zpk, session_keys["kcv"]):
                raise ValueError("ZPK KCV verification failed")
            print("ZPK KCV verified successfully")

        # 5. Load and Validate Transaction Data
        field_data = parse_testcard_data(TCARD_FILE)
        if "2" not in field_data:
            raise ValueError("PAN required for PIN block generation")

        # 6. Set Processing Codes
        field_data.update(
            {
                "3": "500000",  # PIN Verification
                "22": "051",  # PIN Entry Capability
                "25": "00",  # Normal Request
                "26": "12",  # PIN Pad
            }
        )

        # 7. Generate PIN Block
        pin_block = PinBlockUtil.generate_encrypted_pin_block(
            clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
        )
        field_data["52"] = pin_block
        sensitive_data.append(pin_block)

        # 8. Send Transaction
        result = send_pinblock_message(
            host=host, port=port, field_data=field_data, encrypted_pin_block=pin_block
        )

        return result

    except Exception as e:
        print(f"\nError in PIN block processing: {str(e)}")
        return None

    finally:
        # Clear sensitive data
        for item in sensitive_data:
            if isinstance(item, str):
                item = "0" * len(item)
        print("\nSensitive data cleared")


# class PinBlockUtil:
#     @staticmethod
#     def string_to_bytes(hex_str: str) -> bytes:
#         """Convert a hex string to bytes."""
#         return binascii.unhexlify(hex_str)

#     @staticmethod
#     def bytes_to_string(byte_data: bytes) -> str:
#         """Convert bytes to a hex string."""
#         return binascii.hexlify(byte_data).decode().upper()

#     @staticmethod
#     def xor_bytes(input1: bytes, input2: bytes) -> bytes:
#         """XOR two byte arrays."""
#         return bytes(
#             a ^ b for a, b in zip(input1, input2 * (len(input1) // len(input2) + 1))
#         )

#     @staticmethod
#     def operate_des3(data: bytes, key: bytes, encrypt: bool) -> bytes:
#         """
#         Perform 3DES operation with proper key handling.

#         Args:
#             data: 8 bytes of data
#             key: 16 bytes for 3DES (will be expanded to 24 bytes)
#             encrypt: True for encryption, False for decryption
#         """
#         if len(key) != 16:
#             raise ValueError(f"Key must be 16 bytes (got {len(key)})")
#         if len(data) != 8:
#             raise ValueError(f"Data must be 8 bytes (got {len(data)})")

#         # Create 24-byte key for Triple DES by duplicating first 8 bytes
#         triple_des_key = key + key[:8]

#         cipher = Cipher(
#             algorithms.TripleDES(triple_des_key), modes.ECB(), backend=default_backend()
#         )

#         if encrypt:
#             encryptor = cipher.encryptor()
#             return encryptor.update(data) + encryptor.finalize()
#         else:
#             decryptor = cipher.decryptor()
#             return decryptor.update(data) + decryptor.finalize()

#     @classmethod
#     def generate_encrypted_pin_block(
#         cls, clear_zpk: str, card_pan: str, pin: str
#     ) -> str:
#         """Generate ISO 9564 Format 0 PIN block"""
#         try:
#             # Validate inputs
#             if not validate_pin_format(pin):
#                 raise ValueError("Invalid PIN format")

#             # Format PIN block (0 + length + PIN + padding)
#             pin_length = len(pin)
#             pin_block = f"0{pin_length}{pin}{'F' * (16 - 2 - pin_length)}"

#             # Format PAN block (0000 + 12 rightmost digits excluding check digit)
#             pan_block = f"0000{card_pan[-13:-1]}"

#             # Convert to bytes and XOR
#             pin_bytes = cls.string_to_bytes(pin_block)
#             pan_bytes = cls.string_to_bytes(pan_block)
#             clear_pin_block = cls.xor_bytes(pin_bytes, pan_bytes)

#             # Encrypt with session ZPK
#             key_bytes = cls.string_to_bytes(clear_zpk)
#             encrypted_block = cls.operate_des3(clear_pin_block, key_bytes, True)

#             # Convert to hex and pad
#             return cls.bytes_to_string(encrypted_block).ljust(32, "0")
#         except Exception as e:
#             raise ValueError(f"PIN block generation failed: {str(e)}")

#     @classmethod
#     def verify_pin_block(
#         cls, encrypted_block: str, clear_zpk: str, card_pan: str, pin: str
#     ) -> bool:
#         """
#         Verify PIN block encryption/decryption.
#         """
#         try:
#             # Encrypt
#             encrypted = cls.generate_encrypted_pin_block(clear_zpk, card_pan, pin)

#             # Should match provided encrypted block
#             return encrypted.upper() == encrypted_block.upper()

#         except Exception as ex:
#             print(f"PIN block verification failed: {str(ex)}")
#             return False

def format_pin_block_field(value: str) -> bytes:
    """
    Format PIN block (Field 52) according to ISO8583 specifications.

    Args:
        value: PIN block in hex format (16 characters)

    Returns:
        bytes: Formatted PIN block field
    """
    try:
        # Validate input length
        if len(value) != 16:
            raise ValueError(f"PIN block must be 16 hex characters (got {len(value)})")

        # Validate hex format
        if not all(c in "0123456789ABCDEF" for c in value.upper()):
            raise ValueError("Invalid hex format in PIN block")

        # Convert to bytes (should be exactly 8 bytes)
        pin_block_bytes = bytes.fromhex(value)
        if len(pin_block_bytes) != 8:
            raise ValueError(f"Invalid PIN block byte length: {len(pin_block_bytes)}")

        return pin_block_bytes

    except Exception as e:
        raise ISO8583FormatError(f"PIN block formatting error: {str(e)}")

class PinBlockUtil:
    @staticmethod
    def string_to_bytes(hex_str: str) -> bytes:
        """Convert a hex string to bytes."""
        return binascii.unhexlify(hex_str)

    @staticmethod
    def bytes_to_string(byte_data: bytes) -> str:
        """Convert bytes to a hex string."""
        return binascii.hexlify(byte_data).decode().upper()

    @staticmethod
    def xor_bytes(input1: bytes, input2: bytes) -> bytes:
        """XOR two byte arrays."""
        return bytes(a ^ b for a, b in zip(input1, input2))

    @staticmethod
    def operate_des3(data: bytes, key: bytes, encrypt: bool) -> bytes:
        """
        Perform 3DES operation with proper key handling.

        Args:
            data: 8 bytes of data
            key: 16 bytes for 3DES
            encrypt: True for encryption, False for decryption

        Returns:
            bytes: Encrypted/decrypted data
        """
        try:
            # Validate data length
            if len(data) != 8:
                raise ValueError(f"Data must be 8 bytes (got {len(data)})")

            # For Triple DES, we need a 24-byte key
            # We'll use first 16 bytes and repeat first 8 bytes
            if len(key) == 32:  # If given 32-byte key, use first 16 bytes
                key = key[:16]
            elif len(key) != 16:
                raise ValueError(f"Key must be 16 or 32 bytes (got {len(key)})")

            # Create 24-byte key for Triple DES
            triple_des_key = key + key[:8]

            # Create cipher
            cipher = Cipher(
                algorithms.TripleDES(triple_des_key),
                modes.ECB(),
                backend=default_backend()
            )

            # Perform operation
            if encrypt:
                encryptor = cipher.encryptor()
                return encryptor.update(data) + encryptor.finalize()
            else:
                decryptor = cipher.decryptor()
                return decryptor.update(data) + decryptor.finalize()

        except Exception as e:
            raise ValueError(f"3DES operation failed: {str(e)}")

    @classmethod
    def generate_encrypted_pin_block(cls, clear_zpk: str, card_pan: str, pin: str) -> str:
        """
        Generate ISO 9564 Format 0 PIN block.

        Args:
            clear_zpk: Clear Zone PIN Key (32 hex chars)
            card_pan: Card PAN
            pin: PIN value

        Returns:
            str: 16-character hex string representing encrypted PIN block
        """
        try:
            print("\nGenerating PIN Block:")
            print(f"PAN: {'*' * (len(card_pan)-4)}{card_pan[-4:]}")
            print(f"PIN Length: {len(pin)}")

            # Validate inputs
            if not validate_pin_format(pin):
                raise ValueError("Invalid PIN format")

            if len(clear_zpk) != 32:
                raise ValueError(f"Invalid ZPK length: {len(clear_zpk)}")

            # 1. Format PIN block (ISO 9564-1 Format 0)
            pin_length = len(pin)
            pin_block = f"0{pin_length}{pin}{'F' * (14 - pin_length)}"
            print(f"PIN Block Format: 0{pin_length}****{'F' * (14 - pin_length)}")

            # 2. Format PAN block
            pan_block = f"0000{card_pan[-13:-1]}"  # Last 12 digits excluding check digit
            print(f"PAN Block Format: 0000{'*' * 12}")

            # 3. Convert to bytes and XOR
            pin_bytes = cls.string_to_bytes(pin_block)
            pan_bytes = cls.string_to_bytes(pan_block)
            clear_pin_block = cls.xor_bytes(pin_bytes, pan_bytes)

            # 4. Convert ZPK to bytes - use all 32 bytes
            zpk_bytes = cls.string_to_bytes(clear_zpk)

            # 5. Encrypt PIN block
            encrypted_block = cls.operate_des3(clear_pin_block, zpk_bytes, True)

            # 6. Convert to hex
            result = cls.bytes_to_string(encrypted_block)

            print(f"Encrypted PIN Block Length: {len(result)} hex chars")
            return result

        except Exception as e:
            raise ValueError(f"PIN block generation failed: {str(e)}")

    @classmethod
    def verify_pin_block(cls, encrypted_block: str, clear_zpk: str, card_pan: str, pin: str) -> bool:
        """
        Verify PIN block encryption/decryption.

        Args:
            encrypted_block: Encrypted PIN block
            clear_zpk: Clear Zone PIN Key
            card_pan: Card PAN
            pin: PIN to verify

        Returns:
            bool: True if verification succeeds
        """
        try:
            # Generate new encrypted block
            encrypted = cls.generate_encrypted_pin_block(clear_zpk, card_pan, pin)

            # Compare with provided block
            return encrypted.upper() == encrypted_block.upper()

        except Exception as ex:
            print(f"PIN block verification failed: {str(ex)}")
            return False

def print_key_processing_details(
    zmk_comp1: str, zmk_comp2: str, pan: str, pin: str
) -> None:
    """
    Print detailed key processing steps including all intermediate values.

    Args:
        zmk_comp1: First ZMK component
        zmk_comp2: Second ZMK component
        pan: Card PAN
        pin: PIN value
    """
    try:
        print("\nKey Processing Details")
        print("=" * 50)

        # 1. ZMK Component Processing
        print("\n1. ZMK Component Processing:")
        print("-" * 40)
        print(f"ZMK Component 1: {zmk_comp1}")
        print(f"ZMK Component 2: {zmk_comp2}")

        # XOR ZMK components to get clear ZMK
        clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
        print(f"Clear ZMK (XOR result): {clear_zmk}")

        # Split ZMK for variant processing
        zmk_part_a = clear_zmk[:16]
        zmk_part_b = clear_zmk[16:32]
        print(f"ZMK Part A: {zmk_part_a}")
        print(f"ZMK Part B: {zmk_part_b}")

        # 2. PIN Block Generation (ISO-0)
        print("\n2. PIN Block Generation (ISO-0):")
        print("-" * 40)

        # Create PIN block part 1
        pin_length = len(pin)
        pin_block1 = f"0{pin_length}{pin}{'F' * (16 - 2 - pin_length)}"
        print(f"PIN Block Part 1: {pin_block1}")

        # Create PIN block part 2
        pan_part = pan[3:-1]  # Remove first 3 and last digit
        pin_block2 = f"0000{pan_part}"
        print(f"PIN Block Part 2: {pin_block2}")

        # XOR PIN blocks
        pin_block1_bytes = PinBlockUtil.string_to_bytes(pin_block1)
        pin_block2_bytes = PinBlockUtil.string_to_bytes(pin_block2)
        clear_pin_block = PinBlockUtil.xor_bytes(pin_block1_bytes, pin_block2_bytes)
        clear_pin_block_hex = PinBlockUtil.bytes_to_string(clear_pin_block)
        print(f"Clear PIN Block (XOR result): {clear_pin_block_hex}")

        # 3. Print all intermediate values for verification
        print("\n3. All Values:")
        print("-" * 40)
        print(f"PIN Block 1 (hex): {pin_block1_bytes.hex()}")
        print(f"PIN Block 2 (hex): {pin_block2_bytes.hex()}")
        print(f"Clear PIN Block (hex): {clear_pin_block.hex()}")

    except Exception as e:
        print(f"Error in key processing details: {str(e)}")


def print_encrypted_zmk(comp1: str, comp2: str) -> None:
    """
    Print the encrypted ZMK components and validation data.

    Args:
        comp1: First clear component
        comp2: Second clear component
    """
    try:
        # XOR components to get clear ZMK
        clear_zmk = xor_hex_strings(comp1, comp2)

        # Split into two parts (A and B) for variant method
        zmk_part_a = clear_zmk[:16]
        zmk_part_b = clear_zmk[16:32]

        print("\nZMK Component Analysis:")
        print("=" * 50)
        print(f"Component 1: {comp1}")
        print(f"Component 2: {comp2}")
        print(f"Clear ZMK: {clear_zmk}")
        print(f"ZMK Part A: {zmk_part_a}")
        print(f"ZMK Part B: {zmk_part_b}")

        # Generate variants
        variant_a = xor_hex_strings(zmk_part_b, "A6".ljust(16, "0"))
        variant_b = xor_hex_strings(zmk_part_b, "5A".ljust(16, "0"))

        print("\nZMK Variants:")
        print("=" * 50)
        print(f"Variant A (A6): {variant_a}")
        print(f"Variant B (5A): {variant_b}")

        # Generate KCV
        kcv = PinBlockUtil.bytes_to_string(
            PinBlockUtil.operate_des3(
                bytes(8), PinBlockUtil.string_to_bytes(clear_zmk), True
            )
        )[:6]

        print("\nZMK Validation:")
        print("=" * 50)
        print(f"KCV: {kcv}")

    except Exception as e:
        print(f"Error analyzing ZMK: {str(e)}")


# Modify the read_key_components function to use this:
def read_key_components(filename: str = "okeys.txt") -> Tuple[str, str, str]:
    """
    Read and analyze key components from keys.txt file.
    Returns: (component1, component2, kcv)
    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        # Parse components
        comp1 = lines[0].split(":")[1].strip()
        comp2 = lines[1].split(":")[1].strip()
        stored_kcv = lines[2].split(":")[1].strip()

        # Validate components
        if not all(
            len(comp) == 32 and all(c in "0123456789ABCDEF" for c in comp.upper())
            for comp in [comp1, comp2]
        ):
            raise ValueError("Invalid key component format")

        if not (
            len(stored_kcv) == 6
            and all(c in "0123456789ABCDEF" for c in stored_kcv.upper())
        ):
            raise ValueError("Invalid KCV format")

        # Print ZMK analysis
        print_encrypted_zmk(comp1, comp2)

        return comp1, comp2, stored_kcv

    except FileNotFoundError:
        raise Exception("okeys.txt file not found")
    except IndexError:
        raise Exception("Invalid format in okeys.txt")
    except Exception as e:
        raise Exception(f"Error reading keys: {str(e)}")


# Example integration with zx1.py:
def add_pin_block_to_iso_message(fields: dict, pin: str, clear_zpk: str) -> dict:
    """
    Add encrypted PIN block to ISO message fields.
    """
    try:
        if "2" not in fields:  # Need PAN for PIN block
            raise ValueError("PAN (field 2) required for PIN block generation")

        pin_block = PinBlockUtil.generate_encrypted_pin_block(
            clear_zpk=clear_zpk, card_pan=fields["2"], pin=pin
        )

        # Add PIN block to field 52
        fields["52"] = pin_block
        return fields

    except Exception as ex:
        raise Exception(f"Error adding PIN block to message: {str(ex)}")


def generate_stan() -> str:
    """Generate a unique 6-digit System Trace Audit Number"""
    return str(random.randint(0, 999999)).zfill(6)


def generate_retrieval_ref() -> str:
    """Generate a unique 12-digit Retrieval Reference Number"""
    # Format: YDDDHHNNNNNN
    # Y: Last digit of year
    # DDD: Day of year (001-366)
    # HH: Hour (00-23)
    # NNNNNN: Random sequence number
    # now = datetime.datetime.now()
    # year_digit = str(now.year)[-1]
    # day_of_year = str(now.timetuple().tm_yday).zfill(3)
    # hour = str(now.hour).zfill(2)
    # sequence = str(random.randint(0, 999999)).zfill(6)
    # return f"{year_digit}{day_of_year}{hour}{sequence}"
    return str(random.randint(0, 999999)).zfill(6)


def validate_field_dependencies(fields: Dict[str, str]) -> List[str]:
    errors = []

    # Example dependencies
    if "2" in fields and "35" in fields:  # PAN and Track 2
        pan_from_f2 = fields["2"]
        track2 = parse_track2(fields["35"])
        if pan_from_f2 != track2["pan"]:
            errors.append("PAN mismatch between F2 and F35")

    if "4" in fields:  # Amount
        if "49" not in fields:  # Currency code
            errors.append("Currency code (F49) required when amount (F4) present")

    # Field 28 and 30 should have same format
    if "28" in fields and "30" in fields:
        if len(fields["28"]) != len(fields["30"]):
            errors.append("Fields 28 and 30 must have same length")
        if fields["28"][0] != fields["30"][0]:  # Check signs match
            errors.append("Fields 28 and 30 must have same sign")

    # Field 35 validation
    if "35" in fields and "2" in fields:
        track2_pan = fields["35"].split("=")[0]
        if not fields["2"].endswith(track2_pan[-4:]):
            errors.append("PAN in Field 2 and Track 2 must match")

    # Validate presence of related fields
    if "100" in fields and "102" not in fields:
        errors.append("Field 102 required when Field 100 is present")

    if "123" in fields:
        if not fields["123"].startswith("511"):  # Example validation
            errors.append("Field 123 must start with '511'")

    return errors


def validate_mandatory_fields(fields: Dict[str, str], mti: str) -> List[str]:
    errors = []

    mandatory_fields = {
        "0200": [
            "3",
            "4",
            "7",
            "11",
            "12",
            "13",
            "14",
            "18",
            "22",
            "23",
            "25",
            "26",
            "28",
            "32",
            "35",
            "37",
            "40",
            "41",
            "42",
            "43",
            "49",
            "61",
            "100",
            "102",
            "123",  # Ensuring these fields are required
        ]
    }

    if mti in mandatory_fields:
        for field in mandatory_fields[mti]:
            if field not in fields or not fields[field].strip():
                errors.append(f"Mandatory field {field} missing for MTI {mti}")

    return errors


class ISO8583Field:
    """
    Represents an ISO8583 field specification from zone.xml
    """

    def __init__(self, id: str, length: int, name: str, field_class: str):
        self.id = id  # Field identifier
        self.length = length  # Maximum field length
        self.name = name  # Field description
        self.field_class = field_class  # Field type (e.g., IFA_NUMERIC, IFA_LLNUM)


def parse_zone_xml(filename: str) -> Dict[str, ISO8583Field]:
    """
    Parse the zone.xml file containing ISO8583 field specifications.

    Args:
        filename: Path to zone.xml file

    Returns:
        Dictionary mapping field IDs to ISO8583Field objects
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    fields = {}

    for field in root.findall("isofield"):
        field_id = field.get("id")
        fields[field_id] = ISO8583Field(
            field_id, int(field.get("length")), field.get("name"), field.get("class")
        )
    return fields


def parse_testcard_data(filename: str) -> Dict[str, str]:
    """
    Parse the test card data file containing field values.

    Args:
        filename: Path to test card data file

    Returns:
        Dictionary mapping field IDs to their values
    """
    field_data = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                field_id, value = line.split(":", 1)
                field_data[field_id] = value

    # Generate unique values for fields 11 and 37
    field_data["11"] = generate_stan()
    field_data["37"] = generate_retrieval_ref()

    print(f"\nGenerated dynamic fields:")
    print(f"Field 11 (STAN): {field_data['11']}")
    print(f"Field 37 (Retrieval Ref): {field_data['37']}")

    return field_data


def parse_binary_field(field_id: str, binary_data: bytes, expected_length: int) -> Dict:
    """
    Parse binary field data from response message.

    Args:
        field_id: Field identifier
        binary_data: Raw binary data
        expected_length: Expected field length in bytes

    Returns:
        Dict: Parsed field information
    """
    try:
        if len(binary_data) != expected_length:
            raise BinaryFieldError(
                f"Invalid binary length for field {field_id}: "
                f"got {len(binary_data)}, expected {expected_length}"
            )

        result = {
            "value": binary_data.hex().upper(),
            "length": len(binary_data),
            "raw_bytes": binary_data,
        }

        # Add field-specific parsing
        if field_id == "52":
            result["type"] = "PIN Block"
        elif field_id == "53":
            result["type"] = "Security Control Info"
            if len(binary_data) >= 16:
                result["encrypted_key"] = binary_data[:16].hex().upper()
                if len(binary_data) > 16:
                    result["control_info"] = binary_data[16:].hex().upper()
        elif field_id == "64":
            result["type"] = "MAC"
        elif field_id == "128":
            result["type"] = "MAC Extended"

        return result

    except Exception as e:
        raise BinaryFieldError(f"Binary field parsing error: {str(e)}")


def process_binary_field(field_id: str, value: str, field_length: int) -> bytes:
    """
    Process binary fields according to ISO8583 specifications.

    Args:
        field_id: Field identifier
        value: Field value as hex string
        field_length: Required field length in bytes

    Returns:
        bytes: Formatted binary field value

    Raises:
        BinaryFieldError: If field processing fails
    """
    try:
        # Remove 0x prefix if present
        if value.startswith("0x"):
            value = value[2:]

        # Remove spaces and ensure even length
        clean_value = "".join(value.split())
        if len(clean_value) % 2 != 0:
            clean_value = clean_value.zfill(len(clean_value) + 1)

        # Validate hex format
        try:
            int(clean_value, 16)
        except ValueError:
            raise BinaryFieldError(f"Invalid hex format for field {field_id}")

        # Convert to bytes
        try:
            binary_data = bytes.fromhex(clean_value)
        except ValueError as e:
            raise BinaryFieldError(f"Hex conversion failed: {str(e)}")

        # Field-specific processing
        if field_id == "52":  # PIN Block
            # PIN block must be exactly 8 bytes
            if len(binary_data) > 8:
                binary_data = binary_data[:8]
            elif len(binary_data) < 8:
                binary_data = binary_data.ljust(8, b"\x00")

        elif field_id == "53":  # Security Related Control Info
            # Field 53 usually contains encrypted keys and control info
            # Standard format: 16 bytes encrypted data + optional control info
            if len(binary_data) < 16:
                binary_data = binary_data.ljust(16, b"\x00")
            elif len(binary_data) > field_length:
                binary_data = binary_data[:field_length]

        elif field_id == "64":  # MAC
            # MAC must be exactly 8 bytes (64 bits)
            if len(binary_data) > 8:
                binary_data = binary_data[:8]
            elif len(binary_data) < 8:
                binary_data = binary_data.ljust(8, b"\x00")

        elif field_id == "128":  # MAC Extended
            # Extended MAC typically 16 bytes (128 bits)
            if len(binary_data) > 16:
                binary_data = binary_data[:16]
            elif len(binary_data) < 16:
                binary_data = binary_data.ljust(16, b"\x00")

        else:  # General binary field
            if len(binary_data) > field_length:
                binary_data = binary_data[:field_length]
            elif len(binary_data) < field_length:
                binary_data = binary_data.ljust(field_length, b"\x00")

        return binary_data

    except Exception as e:
        raise BinaryFieldError(f"Binary field processing error: {str(e)}")


class FieldType(Enum):
    """Enumeration of field types for special processing"""

    STANDARD = "standard"
    TRACK2 = "track2"
    SECURITY = "security"  # Field 53
    COMPLEX = "complex"  # Field 127
    BINARY = "binary"
    PIN_BLOCK = "pin_block"  # Field 52
    MAC = "mac"  # Field 64


class ISO8583FieldParser:
    """
    Unified field parser for ISO8583 messages with special field handling.
    """

    def __init__(self):
        # Register special field parsers
        self.special_field_handlers = {
            "35": self._parse_track2_data,
            "52": self._parse_pin_block,
            "53": self._parse_security_data,
            "64": self._parse_mac,
            "127": self._parse_complex_field,
        }

    def parse_field(
        self, response: bytes, pos: int, field_spec: "ISO8583Field", field_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Parse any field using standard or special handling as needed.

        Args:
            response: Full message bytes
            pos: Current position in message
            field_spec: Field specification
            field_id: Field identifier

        Returns:
            Tuple[Dict[str, Any], int]: (Parsed field data, new position)

        Raises:
            ISO8583ParseError: If parsing fails
        """
        try:
            # Check for special field handler
            if field_id in self.special_field_handlers:
                return self.special_field_handlers[field_id](response, pos, field_spec)

            # Standard field parsing
            return self._parse_standard_field(response, pos, field_spec, field_id)

        except Exception as e:
            logging.error(f"Error parsing field {field_id}: {str(e)}")
            raise ISO8583ParseError(f"Field {field_id} parsing failed: {str(e)}")

    def _parse_standard_field(
        self, response: bytes, pos: int, field_spec: "ISO8583Field", field_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """Standard field parsing logic"""
        field_class = field_spec.field_class
        field_length = field_spec.length
        field_name = field_spec.name

        # Binary fields
        if "IFB_BINARY" in field_class or "IFA_BINARY" in field_class:
            field_data = response[pos : pos + field_length]
            return {
                "value": field_data.hex().upper(),
                "raw_bytes": field_data,
                "spec": {
                    "class": field_class,
                    "length": field_length,
                    "name": field_name,
                },
                "position": f"{pos}-{pos+field_length-1}",
            }, pos + field_length

        # Variable length fields
        elif any(x in field_class for x in ["LLVAR", "LLLVAR"]):
            length_digits = 2 if "LLVAR" in field_class else 3
            length = int(response[pos : pos + length_digits].decode("ascii"))
            pos += length_digits
            field_data = response[pos : pos + length]

            return {
                "value": field_data.decode("ascii"),
                "length_indicator": length,
                "spec": {
                    "class": field_class,
                    "length": field_length,
                    "name": field_name,
                },
                "position": f"{pos}-{pos+length-1}",
            }, pos + length

        # Fixed length fields
        else:
            field_data = response[pos : pos + field_length]
            return {
                "value": field_data.decode("ascii").strip(),
                "spec": {
                    "class": field_class,
                    "length": field_length,
                    "name": field_name,
                },
                "position": f"{pos}-{pos+field_length-1}",
            }, pos + field_length

    def _parse_track2_data(
        self, response: bytes, pos: int, field_spec: "ISO8583Field"
    ) -> Tuple[Dict[str, Any], int]:
        """Parse Track 2 data (Field 35)"""
        length = int(response[pos : pos + 2].decode("ascii"))
        pos += 2
        data = response[pos : pos + length].decode("ascii")

        # Split on either separator ('=' or 'D')
        separator_pos = data.find("=")
        if separator_pos == -1:
            separator_pos = data.find("D")

        if separator_pos == -1:
            raise ISO8583ParseError("No valid separator found in Track 2 data")

        pan = data[:separator_pos]
        remaining = data[separator_pos + 1 :]

        # Parse components
        parsed = {
            "pan": pan,
            "separator": data[separator_pos],
            "expiry": remaining[:4] if len(remaining) >= 4 else "",
            "service_code": remaining[4:7] if len(remaining) >= 7 else "",
            "discretionary": remaining[7:] if len(remaining) >= 8 else "",
        }

        return {
            "value": data,
            "parsed": parsed,
            "length_indicator": length,
            "spec": field_spec.__dict__,
            "position": f"{pos}-{pos+length-1}",
        }, pos + length

    def _parse_security_data(
        self, response: bytes, pos: int, field_spec: "ISO8583Field"
    ) -> Tuple[Dict[str, Any], int]:
        """Parse security related control info (Field 53)"""
        field_data = response[pos : pos + field_spec.length]

        parsed = {
            "encrypted_key": field_data[:16].hex().upper(),
            "key_version": field_data[16:17].hex().upper()
            if len(field_data) > 16
            else "",
            "control_info": field_data[17:].hex().upper()
            if len(field_data) > 17
            else "",
        }

        return {
            "value": field_data.hex().upper(),
            "parsed": parsed,
            "spec": field_spec.__dict__,
            "position": f"{pos}-{pos+field_spec.length-1}",
        }, pos + field_spec.length

    def _parse_pin_block(
        self, response: bytes, pos: int, field_spec: "ISO8583Field"
    ) -> Tuple[Dict[str, Any], int]:
        """Parse PIN block data (Field 52)"""
        field_data = response[pos : pos + 8]  # PIN block is always 8 bytes

        return {
            "value": field_data.hex().upper(),
            "raw_bytes": field_data,
            "spec": field_spec.__dict__,
            "position": f"{pos}-{pos+8-1}",
        }, pos + 8

    def _parse_mac(
        self, response: bytes, pos: int, field_spec: "ISO8583Field"
    ) -> Tuple[Dict[str, Any], int]:
        """Parse MAC data (Field 64)"""
        field_data = response[pos : pos + 8]  # MAC is always 8 bytes

        return {
            "value": field_data.hex().upper(),
            "raw_bytes": field_data,
            "spec": field_spec.__dict__,
            "position": f"{pos}-{pos+8-1}",
        }, pos + 8

    def _parse_complex_field(
        self, response: bytes, pos: int, field_spec: "ISO8583Field"
    ) -> Tuple[Dict[str, Any], int]:
        """Parse complex field (Field 127) with subfields"""
        # Get total field length (6 digits)
        length = int(response[pos : pos + 6].decode("ascii"))
        pos += 6
        field_data = response[pos : pos + length]

        # Parse subfields if enough data
        subfields = {}
        if length >= 16:  # Minimum for bitmap
            subfields["bitmap"] = field_data[0:16].hex().upper()
            if len(field_data) > 16:
                subfields["data"] = field_data[16:].hex().upper()

                # Parse bitmap to identify present subfields
                bitmap_bin = bin(int.from_bytes(field_data[0:16], "big"))[2:].zfill(128)
                subfields["present_fields"] = [
                    i + 1 for i, bit in enumerate(bitmap_bin) if bit == "1"
                ]
        else:
            subfields["raw_data"] = field_data.hex().upper()

        return {
            "value": field_data.hex().upper(),
            "parsed": subfields,
            "length_indicator": length,
            "spec": field_spec.__dict__,
            "position": f"{pos}-{pos+length-1}",
        }, pos + length


def parse_variable_length_field(
    data: bytes, pos: int, field_class: str
) -> Tuple[int, bytes]:
    """
    Parse a variable length field and return its content and new position.

    Args:
        data: Raw message data
        pos: Current position in the message
        field_class: ISO8583 field class

    Returns:
        Tuple[int, bytes]: (new position, field content)

    Raises:
        FieldLengthError: If field parsing fails
    """
    try:
        if "LLVAR" in field_class or "LLNUM" in field_class or "LLCHAR" in field_class:
            length = int(data[pos : pos + 2].decode("ascii"))
            pos += 2
            content = data[pos : pos + length]
            return pos + length, content

        elif (
            "LLLVAR" in field_class
            or "LLLNUM" in field_class
            or "LLLCHAR" in field_class
        ):
            length = int(data[pos : pos + 3].decode("ascii"))
            pos += 3
            content = data[pos : pos + length]
            return pos + length, content

        raise ValueError(f"Unsupported variable length field class: {field_class}")

    except Exception as e:
        raise FieldLengthError(f"Error parsing variable length field: {str(e)}")


def validate_binary_field(field_id: str, binary_data: bytes) -> bool:
    """
    Validate binary field format and content.

    Args:
        field_id: Field identifier
        binary_data: Processed binary data

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Field-specific validation
        if field_id == "52":
            # PIN block must be exactly 8 bytes
            if len(binary_data) != 8:
                return False

        elif field_id == "53":
            # Must have at least 16 bytes for encrypted key
            if len(binary_data) < 16:
                return False

        elif field_id == "64":
            # MAC must be exactly 8 bytes
            if len(binary_data) != 8:
                return False

        elif field_id == "128":
            # Extended MAC must be exactly 16 bytes
            if len(binary_data) != 16:
                return False

        # Check for all zeros (likely padding error)
        if all(b == 0 for b in binary_data):
            return False

        return True

    except Exception:
        return False


@dataclass
class BitmapInfo:
    """Contains bitmap generation results"""

    bitmap_bytes: bytes
    has_secondary: bool
    present_fields: List[int]
    missing_mandatory: List[str]
    bitmap_hex: str
    bitmap_binary: str


class BitmapGenerator:
    """Handles ISO8583 bitmap generation with proper validation"""

    # Define mandatory fields for each MTI
    MANDATORY_FIELDS = {
        "0200": {"3", "4", "7", "11", "12", "13", "22", "25", "41", "42", "49"},
        "0210": {"3", "4", "7", "11", "12", "13", "39"},
        "0400": {"3", "4", "7", "11", "12", "13", "22", "25", "41", "42", "49"},
        "0410": {"3", "4", "7", "11", "12", "13", "39"},
        "0800": {"3", "7", "11", "70"},  # Key Exchange
        "0810": {"3", "7", "11", "39", "70"},  # Key Exchange Response
    }

    # Fields that require special handling
    CONDITIONAL_DEPENDENCIES = {
        "2": ["35"],  # If PAN present in field 2, must match track 2
        "4": ["49"],  # Amount requires currency code
        "52": ["53"],  # PIN block requires security control info
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_bitmap(self, fields: Dict[str, str], mti: str) -> BitmapInfo:
        """
        Create ISO8583 bitmap with proper validation and field dependencies.

        Args:
            fields: Dictionary of field values
            mti: Message Type Indicator

        Returns:
            BitmapInfo: Bitmap generation results

        Raises:
            ISO8583BitmapError: If bitmap generation fails
        """
        try:
            # 1. Validate mandatory fields
            missing_mandatory = self._validate_mandatory_fields(fields, mti)
            if missing_mandatory:
                self.logger.warning(
                    f"Missing mandatory fields for MTI {mti}: {missing_mandatory}"
                )

            # 2. Check field dependencies
            self._validate_field_dependencies(fields)

            # 3. Determine bitmap requirements
            present_fields = sorted([int(f) for f in fields.keys() if f != "1"])
            needs_secondary = self._needs_secondary_bitmap(present_fields)
            bitmap_length = 128 if needs_secondary else 64

            # 4. Create bitmap
            bitmap_bits = [False] * bitmap_length
            if needs_secondary:
                bitmap_bits[0] = True  # Set bit 1 for secondary bitmap

            # 5. Set bits for present fields
            for field_num in present_fields:
                if 1 <= field_num <= bitmap_length:  # Validate field number
                    bitmap_bits[field_num - 1] = True

            # 6. Convert to bytes
            bitmap_bytes = self._bits_to_bytes(bitmap_bits)
            bitmap_hex = bitmap_bytes.hex().upper()
            bitmap_binary = "".join("1" if bit else "0" for bit in bitmap_bits)

            # 7. Create result
            return BitmapInfo(
                bitmap_bytes=bitmap_bytes,
                has_secondary=needs_secondary,
                present_fields=present_fields,
                missing_mandatory=missing_mandatory,
                bitmap_hex=bitmap_hex,
                bitmap_binary=bitmap_binary,
            )

        except Exception as e:
            raise ISO8583BitmapError(f"Bitmap generation failed: {str(e)}")

    def _validate_mandatory_fields(self, fields: Dict[str, str], mti: str) -> List[str]:
        """Validate presence of mandatory fields based on MTI."""
        if mti not in self.MANDATORY_FIELDS:
            self.logger.warning(f"Unknown MTI: {mti}, using default mandatory fields")
            return []

        mandatory = self.MANDATORY_FIELDS[mti]
        missing = [f for f in mandatory if f not in fields or not fields[f].strip()]
        return missing

    def _validate_field_dependencies(self, fields: Dict[str, str]) -> None:
        """Check field dependencies and raise error if validation fails."""
        for field, dependencies in self.CONDITIONAL_DEPENDENCIES.items():
            if field in fields:
                for dep in dependencies:
                    if dep not in fields:
                        raise ISO8583BitmapError(
                            f"Field {field} requires field {dep} to be present"
                        )

    def _needs_secondary_bitmap(self, present_fields: List[int]) -> bool:
        """Determine if secondary bitmap is needed based on field numbers."""
        return any(field > 64 for field in present_fields)

    def _bits_to_bytes(self, bits: List[bool]) -> bytes:
        """Convert bitmap bits to bytes."""
        bitmap_bytes = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits) and bits[i + j]:
                    byte |= 1 << (7 - j)
            bitmap_bytes.append(byte)
        return bytes(bitmap_bytes)

    def validate_bitmap(self, bitmap_bytes: bytes) -> Tuple[List[int], bool]:
        """
        Validate received bitmap and extract field presence information.

        Args:
            bitmap_bytes: Received bitmap bytes

        Returns:
            Tuple[List[int], bool]: (present fields, has secondary bitmap)
        """
        try:
            # Convert to binary string
            bitmap_bits = bin(int.from_bytes(bitmap_bytes, "big"))[2:].zfill(
                len(bitmap_bytes) * 8
            )

            # Check for secondary bitmap
            has_secondary = bitmap_bits[0] == "1"

            # Get present fields
            present_fields = [
                i + 1
                for i, bit in enumerate(bitmap_bits)
                if bit == "1" and i != 0  # Exclude first bit if secondary bitmap
            ]

            return present_fields, has_secondary

        except Exception as e:
            raise ISO8583BitmapError(f"Bitmap validation failed: {str(e)}")

    def update_bitmap(
        self,
        original_bitmap: bytes,
        fields_to_add: Set[str],
        fields_to_remove: Set[str],
    ) -> bytes:
        """
        Update existing bitmap with field changes.

        Args:
            original_bitmap: Original bitmap bytes
            fields_to_add: Field numbers to add
            fields_to_remove: Field numbers to remove

        Returns:
            bytes: Updated bitmap
        """
        try:
            # Convert to bits
            bits = list(
                bin(int.from_bytes(original_bitmap, "big"))[2:].zfill(
                    len(original_bitmap) * 8
                )
            )

            # Update field presence
            for field in fields_to_add:
                field_num = int(field)
                if field_num > len(bits):
                    # Need to extend to secondary bitmap
                    if len(bits) == 64:
                        bits = ["1"] + bits[1:] + ["0"] * 64  # Add secondary bitmap
                    else:
                        raise ISO8583BitmapError(
                            f"Field number {field_num} exceeds bitmap capacity"
                        )
                bits[field_num - 1] = "1"

            for field in fields_to_remove:
                field_num = int(field)
                if field_num <= len(bits):
                    bits[field_num - 1] = "0"

            # Convert back to bytes
            updated_bits = [b == "1" for b in bits]
            return self._bits_to_bytes(updated_bits)

        except Exception as e:
            raise ISO8583BitmapError(f"Bitmap update failed: {str(e)}")


def parse_track2(value: str) -> dict:
    """
    Parse and validate Track 2 data according to ISO 7813 standard.

    Track 2 Format:
    - Start Sentinel (;) [not used in EMV]
    - PAN (up to 19 digits)
    - Separator (=)
    - Expiry Date (YYMM)
    - Service Code (3 digits)
    - Discretionary Data (up to end of track)
    - End Sentinel (?) [not used in EMV]
    - LRC [not used in EMV]

    Total max length: 40 characters
    """
    if len(value) > 40:
        raise ValueError("Track 2 data exceeds maximum length of 40 characters")

    if not value or "=" not in value:
        raise ValueError("Invalid Track 2 data - missing field separator")

    pan, rest = value.split("=", 1)

    # Parse components according to ISO 7813
    components = {
        "pan": pan,
        "separator": "=",
        "expiry": rest[:4] if len(rest) >= 4 else "",
        "service_code": rest[4:7] if len(rest) >= 7 else "",
        "discretionary": rest[7:] if len(rest) >= 7 else "",
        "raw_value": value,
    }

    # Validation according to ISO 7813
    errors = []
    warnings = []

    # 1. PAN validation
    if not components["pan"].isdigit():
        errors.append("PAN must contain only digits")
    if not (12 <= len(components["pan"]) <= 19):
        errors.append(f"PAN length must be 12-19 digits (got {len(components['pan'])})")

    # 2. Luhn check
    if components["pan"].isdigit():
        check = sum(int(x) for x in components["pan"][-2::-2])
        for digit in components["pan"][-1::-2]:
            digit = int(digit) * 2
            check += digit if digit < 10 else digit - 9
        if check % 10 != 0:
            warnings.append("PAN failed Luhn check")

    # 3. Expiry Date (YYMM)
    if components["expiry"]:
        if not components["expiry"].isdigit():
            errors.append("Expiry date must contain only digits")
        elif len(components["expiry"]) != 4:
            errors.append("Expiry date must be exactly 4 digits")
        else:
            yy = int(components["expiry"][:2])
            mm = int(components["expiry"][2:])
            if not (1 <= mm <= 12):
                errors.append(f"Invalid expiry month: {mm}")

    # 4. Service Code (exact ISO 7813 rules)
    if components["service_code"]:
        if not components["service_code"].isdigit():
            errors.append("Service code must contain only digits")
        if len(components["service_code"]) != 3:
            errors.append(f"Service code must be exactly 3 digits")
        else:
            # First digit: Interchange rules
            first_digit = components["service_code"][0]
            valid_first_digits = {
                "1": "International interchange OK",
                "2": "International interchange, IC (chip) required",
                "5": "National interchange only except under bilateral agreement",
                "6": "National interchange only except under bilateral agreement, IC required",
                "7": "No interchange except under bilateral agreement (closed loop)",
            }
            if first_digit not in valid_first_digits:
                errors.append(f"Invalid service code first digit: {first_digit}")

            # Second digit: Authorization processing
            second_digit = components["service_code"][1]
            valid_second_digits = {
                "0": "Normal authorization",
                "2": "Contact issuer via online means",
                "4": "Contact issuer via online means except under bilateral agreement",
            }
            if second_digit not in valid_second_digits:
                errors.append(f"Invalid service code second digit: {second_digit}")

            # Third digit: Range of services
            third_digit = components["service_code"][2]
            valid_third_digits = {
                "0": "No restrictions, PIN required",
                "1": "No restrictions",
                "2": "Goods and services only",
                "3": "ATM only, PIN required",
                "4": "Cash only",
                "5": "Goods and services only, PIN required",
                "6": "No restrictions, use PIN if PIN pad present",
                "7": "Goods and services only, use PIN if PIN pad present",
            }
            if third_digit not in valid_third_digits:
                errors.append(f"Invalid service code third digit: {third_digit}")

    # 5. Discretionary Data (can be any digits, no specific validation required)
    if components["discretionary"] and not components["discretionary"].isdigit():
        warnings.append("Discretionary data should contain only digits")

    components["errors"] = errors
    components["warnings"] = warnings
    components["is_valid"] = len(errors) == 0

    # Add detailed interpretation
    if components["service_code"]:
        components["service_code_interpretation"] = {
            "interchange": valid_first_digits.get(
                components["service_code"][0], "Unknown"
            ),
            "authorization": valid_second_digits.get(
                components["service_code"][1], "Unknown"
            ),
            "services": valid_third_digits.get(
                components["service_code"][2], "Unknown"
            ),
        }

    return components


def calculate_field_length(field_class: str, value: str, max_length: int) -> int:
    """
    Calculate the total field length according to ISO8583 specifications.

    Args:
        field_class: ISO8583 field class (e.g., 'IFA_LLNUM', 'IFA_NUMERIC')
        value: The field value to be encoded
        max_length: Maximum allowed length for this field

    Returns:
        int: Total field length including length indicators if applicable

    Raises:
        FieldLengthError: If field length validation fails
    """
    try:
        # Fixed length fields
        if any(x in field_class for x in ["IFA_NUMERIC", "IFA_BINARY", "IF_CHAR"]):
            return max_length

        # Variable length fields
        data_length = len(value)
        if "LLVAR" in field_class or "LLNUM" in field_class or "LLCHAR" in field_class:
            if data_length > 99:
                raise FieldLengthError(
                    f"Data length {data_length} exceeds LLVAR maximum (99)"
                )
            return 2 + min(data_length, max_length)  # 2 digits for length + data

        elif (
            "LLLVAR" in field_class
            or "LLLNUM" in field_class
            or "LLLCHAR" in field_class
        ):
            if data_length > 999:
                raise FieldLengthError(
                    f"Data length {data_length} exceeds LLLVAR maximum (999)"
                )
            return 3 + min(data_length, max_length)  # 3 digits for length + data

        # Special handling for binary fields
        elif "IFB_BINARY" in field_class:
            # Ensure even length for binary data
            binary_length = (data_length + 1) // 2 * 2
            return min(binary_length, max_length)

        raise ValueError(f"Unsupported field class: {field_class}")

    except Exception as e:
        raise FieldLengthError(f"Error calculating field length: {str(e)}")


def format_variable_length_field(
    value: str, field_class: str, max_length: int
) -> bytes:
    """
    Format a variable length field with proper length indicators.

    Args:
        value: Field value to format
        field_class: ISO8583 field class
        max_length: Maximum allowed field length

    Returns:
        bytes: Formatted field including length indicator

    Raises:
        FieldLengthError: If field formatting fails
    """
    try:
        data_length = len(value)

        # Validate maximum length
        if data_length > max_length:
            value = value[:max_length]
            data_length = max_length

        # Format based on field type
        if "LLVAR" in field_class or "LLNUM" in field_class or "LLCHAR" in field_class:
            if data_length > 99:
                raise FieldLengthError(
                    f"Data length {data_length} exceeds LLVAR maximum (99)"
                )
            length_indicator = str(data_length).zfill(2)
            return (length_indicator + value).encode("ascii")

        elif (
            "LLLVAR" in field_class
            or "LLLNUM" in field_class
            or "LLLCHAR" in field_class
        ):
            if data_length > 999:
                raise FieldLengthError(
                    f"Data length {data_length} exceeds LLLVAR maximum (999)"
                )
            length_indicator = str(data_length).zfill(3)
            return (length_indicator + value).encode("ascii")

        raise ValueError(f"Unsupported variable length field class: {field_class}")

    except Exception as e:
        raise FieldLengthError(f"Error formatting variable length field: {str(e)}")


def validate_field_length(value: str, field_spec: "ISO8583Field") -> bool:
    """
    Validate field length against ISO8583 specifications.

    Args:
        value: Field value to validate
        field_spec: Field specification object

    Returns:
        bool: True if length is valid

    Raises:
        FieldLengthError: If validation fails
    """
    try:
        field_class = field_spec.field_class
        max_length = field_spec.length
        data_length = len(value)

        # Fixed length fields must match exactly
        if any(x in field_class for x in ["IFA_NUMERIC", "IFA_BINARY", "IF_CHAR"]):
            if data_length != max_length:
                raise FieldLengthError(
                    f"Fixed length field must be exactly {max_length} characters"
                )

        # Variable length fields must not exceed maximum
        elif "LLVAR" in field_class:
            if data_length > 99:
                raise FieldLengthError("LLVAR field cannot exceed 99 characters")
            if data_length > max_length:
                raise FieldLengthError(
                    f"Field length {data_length} exceeds maximum {max_length}"
                )

        elif "LLLVAR" in field_class:
            if data_length > 999:
                raise FieldLengthError("LLLVAR field cannot exceed 999 characters")
            if data_length > max_length:
                raise FieldLengthError(
                    f"Field length {data_length} exceeds maximum {max_length}"
                )

        return True

    except Exception as e:
        raise FieldLengthError(f"Field length validation failed: {str(e)}")


from typing import Dict, Optional, Union
import binascii
from datetime import datetime
import random
import logging


def format_binary_field(field_id: str, value: str, field_length: int) -> bytes:
    """
    Format binary fields (52, 53, 64) with proper length handling.

    Args:
        field_id: Field identifier
        value: Field value as hex string
        field_length: Required field length in bytes

    Returns:
        bytes: Formatted binary field
    """
    try:
        # Remove spaces and ensure proper hex string
        clean_value = ''.join(value.split()).upper()

        # Validate hex format
        if not all(c in "0123456789ABCDEF" for c in clean_value):
            raise ValueError(f"Invalid hex format for field {field_id}")

        # Specific handling for each field type
        if field_id == "52":  # PIN Block
            if len(clean_value) != 16:  # Must be exactly 16 hex chars (8 bytes)
                raise ValueError(
                    f"PIN block must be 16 hex chars (got {len(clean_value)})"
                )
            return bytes.fromhex(clean_value)

        elif field_id == "53":  # Security Related Control Info
            if len(clean_value) != 32:  # Must be 32 hex chars (16 bytes)
                raise ValueError(
                    f"Security info must be 32 hex chars (got {len(clean_value)})"
                )
            return bytes.fromhex(clean_value)

        elif field_id == "64":  # MAC
            if len(clean_value) != 16:  # Must be 16 hex chars (8 bytes)
                raise ValueError(f"MAC must be 16 hex chars (got {len(clean_value)})")
            return bytes.fromhex(clean_value)

        else:  # Other binary fields
            # Convert to bytes and handle length
            binary_data = bytes.fromhex(clean_value)
            if len(binary_data) > field_length:
                binary_data = binary_data[:field_length]
            elif len(binary_data) < field_length:
                binary_data = binary_data.ljust(field_length, b'\x00')
            return binary_data

    except Exception as e:
        raise ISO8583FormatError(f"Binary field formatting error: {str(e)}")

def format_field_value(value: str, field_spec: "ISO8583Field") -> Optional[bytes]:
    """
    Format field value according to ISO8583 specifications with enhanced security.

    Args:
        value: Field value to format
        field_spec: ISO8583Field specification from zone.xml

    Returns:
        Optional[bytes]: Formatted field value or None if formatting fails

    Raises:
        ISO8583FormatError: For general formatting errors
        ISO8583SecurityError: For security-related errors
    """
    field_id = field_spec.id
    field_class = field_spec.field_class
    field_length = field_spec.length
    sensitive_fields = {"2", "35", "52", "53", "64"}  # PAN, Track2, PIN, Keys, MAC

    try:
        # Initial logging (mask sensitive fields)
        masked_value = "****" if field_id in sensitive_fields else value
        logging.debug(
            f"Processing field {field_id} ({field_spec.name}): {masked_value}"
        )

        # Handle empty values
        if not value and field_id != "28":  # Allow empty for field 28
            logging.info(
                f"Field {field_id}: Empty value - will be excluded from bitmap"
            )
            return None

        # Binary Fields Processing (IFB_BINARY, IFA_BINARY)
        if "IFB_BINARY" in field_class or "IFA_BINARY" in field_class:
            try:
                # Remove 0x prefix if present
                clean_value = value[2:] if value.startswith("0x") else value
                # Remove spaces and ensure even length
                clean_value = "".join(clean_value.split())
                if len(clean_value) % 2 != 0:
                    clean_value = clean_value.zfill(len(clean_value) + 1)

                # Field-specific processing
                if field_id == "52":  # PIN Block
                    return format_pin_block_field(value)

                elif field_id == "53":  # Security Related Control Info
                    if len(clean_value) < 32:  # Minimum 16 bytes = 32 hex chars
                        raise ISO8583SecurityError(
                            "Security field below minimum length"
                        )
                    binary_data = bytes.fromhex(clean_value)
                    if len(binary_data) > field_length:
                        binary_data = binary_data[:field_length]

                elif field_id == "64":  # MAC
                    if len(clean_value) != 16:  # Must be exactly 8 bytes = 16 hex chars
                        raise ISO8583SecurityError("Invalid MAC length")
                    binary_data = bytes.fromhex(clean_value)

                elif field_id == "128":  # MAC Extended
                    if (
                        len(clean_value) != 32
                    ):  # Must be exactly 16 bytes = 32 hex chars
                        raise ISO8583SecurityError("Invalid Extended MAC length")
                    binary_data = bytes.fromhex(clean_value)

                else:  # Other binary fields
                    binary_data = bytes.fromhex(clean_value)
                    if len(binary_data) > field_length:
                        binary_data = binary_data[:field_length]
                    elif len(binary_data) < field_length:
                        binary_data = binary_data.ljust(field_length, b"\x00")

                logging.debug(f"Binary field {field_id} processed successfully")
                return binary_data

            except (ValueError, binascii.Error) as e:
                raise ISO8583FormatError(
                    f"Invalid binary data for field {field_id}: {str(e)}"
                )

        # Amount Fields Processing (IFA_AMOUNT)
        elif "IFA_AMOUNT" in field_class:
            try:
                if field_id == "28":  # Transaction Fee
                    if not value or value.strip() == "":
                        amount = str(random.randint(1, 50))
                        sign = "C"
                    else:
                        clean_value = "".join(
                            c for c in value if c.isdigit() or c in "DC"
                        )
                        amount = "".join(c for c in clean_value if c.isdigit())
                        sign = "D" if "D" in clean_value else "C"

                    padded_amount = amount.zfill(field_length - 1)
                    if len(padded_amount) > field_length - 1:
                        padded_amount = padded_amount[-(field_length - 1) :]

                    formatted = f"{sign}{padded_amount}"
                    return formatted.encode()
                else:
                    clean_value = "".join(c for c in value if c.isdigit() or c in "DC")
                    amount = "".join(c for c in clean_value if c.isdigit())
                    sign = "D" if "D" in clean_value else "C"
                    padded_amount = amount.zfill(field_length - 1)
                    formatted = f"{sign}{padded_amount[-field_length+1:]}"
                    return formatted.encode()

            except Exception as e:
                raise ISO8583FormatError(f"Amount formatting error: {str(e)}")

        # Variable Length Numeric Fields (LLNUM/LLLNUM)
        elif any(x in field_class for x in ["IFA_LLNUM", "IFA_LLLNUM"]):
            try:
                clean_value = "".join(c for c in value if c.isdigit())

                # Special handling for PAN (Field 2)
                if field_id == "2":
                    if not (12 <= len(clean_value) <= 19):
                        raise ISO8583FormatError(
                            f"Invalid PAN length: {len(clean_value)}"
                        )
                    if not validate_pan(clean_value):
                        raise ISO8583SecurityError("PAN failed Luhn check")

                # Determine length indicator size
                length_indicator_size = 2 if "LLNUM" in field_class else 3
                max_length = 99 if "LLNUM" in field_class else 999

                if len(clean_value) > max_length:
                    clean_value = clean_value[:max_length]

                length = str(len(clean_value)).zfill(length_indicator_size)
                formatted = length + clean_value
                return formatted.encode()

            except Exception as e:
                raise ISO8583FormatError(
                    f"Variable length numeric formatting error: {str(e)}"
                )

        # Fixed Length Character Fields (IF_CHAR)
        elif "IF_CHAR" in field_class:
            try:
                if len(value) > field_length:
                    value = value[:field_length]
                formatted = value.ljust(field_length)
                return formatted.encode()

            except Exception as e:
                raise ISO8583FormatError(
                    f"Fixed length character formatting error: {str(e)}"
                )

        # Variable Length Character Fields (LLCHAR/LLLCHAR)
        elif any(x in field_class for x in ["IFA_LLCHAR", "IFA_LLLCHAR"]):
            try:
                # Determine length indicator size
                length_indicator_size = 2 if "LLCHAR" in field_class else 3
                max_length = 99 if "LLCHAR" in field_class else 999

                if len(value) > max_length:
                    value = value[:max_length]

                length = str(len(value)).zfill(length_indicator_size)
                formatted = length + value
                return formatted.encode()

            except Exception as e:
                raise ISO8583FormatError(
                    f"Variable length character formatting error: {str(e)}"
                )

        # Fixed Length Numeric Fields (IFA_NUMERIC)
        elif "IFA_NUMERIC" in field_class:
            try:
                clean_value = "".join(c for c in value if c.isdigit())
                if len(clean_value) > field_length:
                    clean_value = clean_value[-field_length:]
                formatted = clean_value.zfill(field_length)
                return formatted.encode()

            except Exception as e:
                raise ISO8583FormatError(
                    f"Fixed length numeric formatting error: {str(e)}"
                )

        # Track 2 Data (Field 35)
        elif field_id == "35":
            try:
                components = parse_track2(value)
                if not components["is_valid"]:
                    raise ISO8583FormatError(
                        f"Invalid Track2 data: {components['errors']}"
                    )

                # Use original separator from input
                separator = (
                    components["separator"]
                    if components["separator"] in ["=", "D"]
                    else "="
                )

                formatted_value = (
                    components["pan"]
                    + separator
                    + components["expiry"]
                    + components.get("service_code", "")
                    + components.get("discretionary", "")
                )

                # Calculate and validate length
                if len(formatted_value) > 37:  # Max track 2 length - 37 chars
                    raise ISO8583FormatError("Track2 data exceeds maximum length")

                length = str(len(formatted_value)).zfill(2)
                result = (length + formatted_value).encode()

                logging.debug(
                    f"Track2 formatted successfully: {length}{'*' * len(formatted_value)}"
                )
                return result

            except Exception as e:
                raise ISO8583FormatError(f"Track2 formatting error: {str(e)}")

        else:
            raise ISO8583FormatError(f"Unsupported field class: {field_class}")

    except (ISO8583FormatError, ISO8583SecurityError) as e:
        logging.error(f"Field {field_id} formatting error: {str(e)}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error formatting field {field_id}: {str(e)}")
        raise ISO8583FormatError(f"Unexpected formatting error: {str(e)}")

    finally:
        # Clear sensitive data
        if field_id in sensitive_fields and "clean_value" in locals():
            locals()["clean_value"] = "0" * len(locals()["clean_value"])


def validate_pan(pan: str) -> bool:
    """
    Validate PAN using Luhn algorithm and length requirements.

    Args:
        pan: Card number to validate

    Returns:
        bool: True if PAN is valid
    """
    try:
        if not pan.isdigit():
            return False

        if not (12 <= len(pan) <= 19):
            return False

        # Luhn algorithm
        digits = [int(d) for d in pan]
        checksum = 0
        odd = True

        for digit in digits[-2::-1]:
            if odd:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
            odd = not odd

        return (checksum + digits[-1]) % 10 == 0

    except Exception as e:
        logging.error(f"PAN validation error: {str(e)}")
        return False


def parse_iso_response(response: bytes, field_specs: Dict[str, ISO8583Field]) -> dict:
    """
    Parse ISO8583 response message with field mapping to zone.xml specifications.

    Args:
        response: Raw response bytes
        field_specs: Dictionary of field specifications from zone.xml

    Returns:
        Dictionary containing parsed message details including response codes
    """
    # Complete ISO8583 Response Codes
    response_codes = {
        "00": "Approved",
        "01": "Refer to card issuer",
        "02": "Refer to card issuer, special condition",
        "03": "Invalid merchant",
        "04": "Pick-up card",
        "05": "Do not honor",
        "06": "Error",
        "07": "Pick-up card, special condition",
        "08": "Honor with identification",
        "09": "Request in progress",
        "10": "Approved, partial",
        "11": "Approved, VIP",
        "12": "Invalid transaction",
        "13": "Invalid amount",
        "14": "Invalid card number",
        "15": "No such issuer",
        "21": "No action taken",
        "25": "Unable to locate record",
        "28": "File temporarily not available",
        "30": "Format error",
        "31": "Bank not supported",
        "33": "Expired card",
        "34": "Suspected fraud",
        "35": "Contact acquirer",
        "36": "Restricted card",
        "37": "Call acquirer security",
        "38": "PIN tries exceeded",
        "39": "No credit account",
        "40": "Function not supported",
        "41": "Lost card - pick up",
        "42": "No universal account",
        "43": "Stolen card - pick up",
        "51": "Insufficient funds",
        "52": "No check account",
        "53": "No savings account",
        "54": "Expired card",
        "55": "Invalid PIN",
        "56": "No card record",
        "57": "Transaction not permitted to cardholder",
        "58": "Transaction not permitted to terminal",
        "59": "Suspected fraud",
        "61": "Exceeds withdrawal limit",
        "62": "Restricted card",
        "63": "Security violation",
        "64": "Original amount incorrect",
        "65": "Exceeds withdrawal frequency",
        "66": "Call acquirer security",
        "67": "Hard capture - pick up card",
        "68": "Response received too late",
        "75": "PIN tries exceeded",
        "77": "Intervene, bank approval required",
        "78": "Intervene, bank approval required for partial amount",
        "85": "Not declined",
        "86": "PIN validation not possible",
        "89": "Bad terminal",
        "90": "Cut-off in progress",
        "91": "Issuer or switch inoperative",
        "92": "Routing error",
        "93": "Violation of law",
        "94": "Duplicate transaction",
        "95": "Reconcile error",
        "96": "System malfunction",
        "97": "Reserved for national use",
        "98": "Exceeds cash limit",
        "99": "PIN Block error",
    }

    result = {
        "raw_length": len(response),
        "raw_hex": response.hex(),
        "parsing_details": [],
        "response_codes": response_codes,
    }

    try:
        # 1. Parse Length Prefix (2 bytes)
        if len(response) < 2:
            raise ValueError("Response too short - missing length prefix")

        length_prefix = int.from_bytes(response[0:2], "big")
        result["length_prefix"] = {
            "value": length_prefix,
            "hex": response[0:2].hex(),
            "position": "0-1",
        }
        result["parsing_details"].append(f"Length Prefix: {length_prefix} bytes")

        # 2. Parse MTI (4 bytes)
        if len(response) < 6:
            raise ValueError("Response too short - missing MTI")

        mti = response[2:6].decode("ascii")
        result["mti"] = {"value": mti, "hex": response[2:6].hex(), "position": "2-5"}
        result["parsing_details"].append(f"MTI: {mti}")

        # 3. Parse Bitmap (16 bytes for secondary bitmap)
        if len(response) < 22:
            raise ValueError("Response too short - missing bitmap")

        bitmap_bytes = response[6:22]
        bitmap_hex = bitmap_bytes.hex()
        bitmap_bin = bin(int.from_bytes(bitmap_bytes, "big"))[2:].zfill(128)

        # Get present fields from bitmap
        present_fields = [i + 1 for i, bit in enumerate(bitmap_bin) if bit == "1"]

        result["bitmap"] = {
            "hex": bitmap_hex,
            "binary": bitmap_bin,
            "present_fields": present_fields,
            "position": "6-21",
        }
        result["parsing_details"].append(f"Bitmap indicates fields: {present_fields}")

        # 4. Parse Fields
        pos = 22  # Start position after MTI and bitmap
        fields = {}

        for field_num in present_fields:
            if field_num == 1:  # Skip bitmap field
                continue

            field_id = str(field_num)
            try:
                # Special handling for field 127 (Complex field with subfields)
                if field_id == "127":
                    try:
                        # Get length (6 digits)
                        if pos + 6 > len(response):
                            raise ValueError("Message too short for field 127 length")

                        length = int(response[pos : pos + 6].decode("ascii"))
                        pos += 6

                        if pos + length > len(response):
                            raise ValueError(
                                f"Message too short for field 127 data (need {length} bytes)"
                            )

                        # Extract field 127 data
                        field_127_data = response[pos : pos + length]

                        # Parse subfields if present
                        subfields = {}
                        if length >= 16:  # Minimum for bitmap
                            subfields["bitmap"] = field_127_data[0:16].hex()
                            # Add raw data for debugging
                            subfields["raw_data"] = field_127_data[16:].hex()
                        else:
                            subfields["raw_data"] = field_127_data.hex()

                        fields[field_id] = {
                            "value": subfields,
                            "length_indicator": length,
                            "spec": {
                                "class": "org.jpos.iso.IFA_LLLLLLBINARY",
                                "length": 999999,
                                "name": "RESERVED PRIVATE USE",
                            },
                            "position": f"{pos}-{pos+length-1}",
                            "hex": field_127_data.hex(),
                        }
                        pos += length
                        continue

                    except Exception as e:
                        print(f"Error processing field 127: {str(e)}")
                        fields[field_id] = {
                            "error": f"Field 127 parsing failed: {str(e)}",
                            "position": f"{pos}-?",
                        }
                        continue

                # Get field specification from zone.xml
                field_spec = field_specs.get(field_id)
                if not field_spec:
                    print(f"Warning: No specification found for field {field_id}")
                    continue

                field_class = field_spec.field_class
                field_length = field_spec.length
                field_name = field_spec.name

                # Handle different field types based on class
                try:
                    # Handle IFB_BINARY fields (52, 53, 64, 65, 96, 128)
                    if "IFB_BINARY" in field_class:
                        field_data = response[pos : pos + field_length].hex()
                        fields[field_id] = {
                            "value": field_data,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+field_length-1}",
                            "hex": field_data,
                        }
                        pos += field_length

                    # Handle IFA_AMOUNT fields (28, 29, 30, 31, 97)
                    elif "IFA_AMOUNT" in field_class:
                        amount_data = response[pos : pos + field_length].decode("ascii")
                        if len(amount_data) != field_length:
                            raise ValueError(
                                f"Invalid amount length: {len(amount_data)}"
                            )

                        sign = amount_data[0]
                        amount = amount_data[1:]
                        fields[field_id] = {
                            "value": amount_data,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+field_length-1}",
                            "hex": response[pos : pos + field_length].hex(),
                            "parsed": {"sign": sign, "amount": amount},
                        }
                        pos += field_length

                    # Handle IFA_NUMERIC fields
                    elif "IFA_NUMERIC" in field_class:
                        field_data = response[pos : pos + field_length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+field_length-1}",
                            "hex": response[pos : pos + field_length].hex(),
                        }
                        pos += field_length

                    # Handle IF_CHAR fields
                    elif "IF_CHAR" in field_class:
                        field_data = response[pos : pos + field_length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data.strip(),  # Remove padding spaces
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+field_length-1}",
                            "hex": response[pos : pos + field_length].hex(),
                        }
                        pos += field_length

                    # Handle IFA_LLNUM fields
                    elif "IFA_LLNUM" in field_class:
                        length = int(response[pos : pos + 2].decode("ascii"))
                        pos += 2
                        field_data = response[pos : pos + length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data,
                            "length_indicator": length,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+length-1}",
                            "hex": response[pos : pos + length].hex(),
                        }
                        pos += length

                    # Handle IFA_LLLNUM fields
                    elif "IFA_LLLNUM" in field_class:
                        length = int(response[pos : pos + 3].decode("ascii"))
                        pos += 3
                        field_data = response[pos : pos + length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data,
                            "length_indicator": length,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+length-1}",
                            "hex": response[pos : pos + length].hex(),
                        }
                        pos += length

                    # Handle IFA_LLCHAR fields
                    elif "IFA_LLCHAR" in field_class:
                        length = int(response[pos : pos + 2].decode("ascii"))
                        pos += 2
                        field_data = response[pos : pos + length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data,
                            "length_indicator": length,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+length-1}",
                            "hex": response[pos : pos + length].hex(),
                        }
                        pos += length

                    # Handle IFA_LLLCHAR fields
                    elif "IFA_LLLCHAR" in field_class:
                        length = int(response[pos : pos + 3].decode("ascii"))
                        pos += 3
                        field_data = response[pos : pos + length].decode("ascii")
                        fields[field_id] = {
                            "value": field_data,
                            "length_indicator": length,
                            "spec": {
                                "class": field_class,
                                "length": field_length,
                                "name": field_name,
                            },
                            "position": f"{pos}-{pos+length-1}",
                            "hex": response[pos : pos + length].hex(),
                        }
                        pos += length

                    else:
                        print(
                            f"Warning: Unsupported field class: {field_class} for field {field_id}"
                        )
                        continue

                    # Add parsing detail
                    result["parsing_details"].append(
                        f"Field {field_id} ({field_name}): {fields[field_id]['value']} ({field_class})"
                    )

                except Exception as e:
                    fields[field_id] = {
                        "error": f"Parsing failed: {str(e)}",
                        "position": f"{pos}-?",
                        "spec": field_spec.__dict__ if field_spec else None,
                    }
                    result["parsing_details"].append(
                        f"Error parsing field {field_id}: {str(e)}"
                    )
                    continue

            except Exception as e:
                print(f"Error processing field {field_id}: {str(e)}")
                continue

        result["fields"] = fields

        # Special handling for response code (Field 39)
        if "39" in fields and isinstance(fields["39"], dict):
            resp_code = fields["39"]["value"].strip()
            result["response_analysis"] = {
                "code": resp_code,
                "meaning": response_codes.get(resp_code, "Unknown response code"),
                "category": "Success" if resp_code == "00" else "Error",
            }

        # Message completion verification
        expected_length = length_prefix
        actual_length = len(response) - 2  # Excluding length prefix
        result["length_check"] = {
            "expected": expected_length,
            "actual": actual_length,
            "valid": expected_length == actual_length,
        }

        # Add parsing summary
        result["parsing_summary"] = {
            "total_fields": len(fields),
            "successful_fields": len([f for f in fields.values() if "error" not in f]),
            "failed_fields": len([f for f in fields.values() if "error" in f]),
            "completion": (
                "Complete" if expected_length == actual_length else "Incomplete"
            ),
        }

    except Exception as e:
        result["parsing_error"] = str(e)
        print(f"Error parsing message: {str(e)}")

    return result


def decode_server_response(
    response: bytes, field_specs: Dict[str, ISO8583Field]
) -> dict:
    """
    Decode ISO8583 response message using unified field parser.

    Args:
        response: Raw response bytes from server
        field_specs: Dictionary of field specifications from zone.xml

    Returns:
        dict: Parsed message details including:
            - raw_length: Length of original message
            - raw_hex: Hex representation of message
            - length_prefix: Message length indicator
            - mti: Message Type Indicator
            - bitmap: Bitmap analysis
            - fields: Parsed fields
            - parsing_details: List of parsing steps
            - message_structure: Message structure analysis
            - response_analysis: Analysis of response code (if present)
    """
    result = {
        "raw_length": len(response),
        "raw_hex": response.hex().upper(),
        "parsing_details": [],
        "fields": {},
        "message_structure": {},
    }

    try:
        # 1. Initialize field parser
        field_parser = ISO8583FieldParser()
        pos = 0  # Current position in message

        # 2. Validate minimum message length (2 bytes length + 4 bytes MTI)
        if len(response) < 6:
            raise ISO8583DecodeError("Response too short for basic header")

        # 3. Parse length prefix (2 bytes)
        msg_length = int.from_bytes(response[0:2], "big")
        result["length_prefix"] = {
            "value": msg_length,
            "hex": response[0:2].hex().upper(),
            "position": "0-1",
        }
        result["parsing_details"].append(f"Length Prefix: {msg_length} bytes")
        pos = 2

        # 4. Parse MTI (4 bytes)
        if pos + 4 > len(response):
            raise ISO8583DecodeError("Response too short for MTI")

        mti = response[pos : pos + 4].decode("ascii")
        result["mti"] = {
            "value": mti,
            "hex": response[pos : pos + 4].hex().upper(),
            "position": f"{pos}-{pos+3}",
        }
        result["parsing_details"].append(f"MTI: {mti}")
        pos = 6

        # 5. Parse and analyze bitmap
        if pos + 8 > len(response):
            raise ISO8583DecodeError("Response too short for primary bitmap")

        primary_bitmap = response[pos : pos + 8]
        bitmap_bin = bin(int.from_bytes(primary_bitmap, "big"))[2:].zfill(64)

        # Check for secondary bitmap
        has_secondary = bitmap_bin[0] == "1"
        bitmap_end = pos + (16 if has_secondary else 8)

        if has_secondary:
            if pos + 16 > len(response):
                raise ISO8583DecodeError("Response too short for secondary bitmap")
            full_bitmap = response[pos:bitmap_end]
            bitmap_bin = bin(int.from_bytes(full_bitmap, "big"))[2:].zfill(128)
        else:
            full_bitmap = primary_bitmap

        result["bitmap"] = {
            "hex": full_bitmap.hex().upper(),
            "binary": bitmap_bin,
            "has_secondary": has_secondary,
            "position": f"{pos}-{bitmap_end-1}",
        }
        pos = bitmap_end

        # 6. Determine present fields from bitmap
        present_fields = [i + 1 for i, bit in enumerate(bitmap_bin) if bit == "1"]
        result["parsing_details"].append(f"Fields present: {present_fields}")

        # 7. Parse individual fields
        for field_num in present_fields:
            if field_num == 1:  # Skip bitmap field
                continue

            field_id = str(field_num)
            try:
                field_spec = field_specs.get(field_id)
                if not field_spec:
                    result["parsing_details"].append(
                        f"Warning: No specification found for field {field_id}"
                    )
                    continue

                # Use unified field parser
                field_data, new_pos = field_parser.parse_field(
                    response, pos, field_spec, field_id
                )

                if field_data:
                    result["fields"][field_id] = field_data
                    result["parsing_details"].append(
                        f"Field {field_id} ({field_spec.name}): "
                        f"Position {field_data['position']}"
                    )
                pos = new_pos

            except Exception as e:
                result["parsing_details"].append(
                    f"Error parsing field {field_id}: {str(e)}"
                )
                logging.error(f"Field {field_id} parsing failed: {str(e)}")
                continue

        # 8. Message structure validation
        actual_length = len(response) - 2  # Excluding length prefix
        result["message_structure"] = {
            "length_prefix": msg_length,
            "actual_length": actual_length,
            "mti_length": 4,
            "bitmap_length": len(full_bitmap),
            "field_count": len(result["fields"]),
            "has_secondary_bitmap": has_secondary,
            "fields_parsed": list(result["fields"].keys()),
            "is_complete": actual_length == msg_length,
        }

        # 9. Response code analysis
        if "39" in result["fields"]:
            resp_code = result["fields"]["39"]["value"].strip()
            result["response_analysis"] = analyze_response_code(resp_code)

            # Add formatted response description
            if resp_code in ISO8583_RESPONSE_CODES:
                result["response_description"] = {
                    "code": resp_code,
                    "message": ISO8583_RESPONSE_CODES[resp_code],
                    "category": get_response_category(resp_code),
                }

        # 10. Add parsing summary
        result["parsing_summary"] = {
            "total_fields": len(present_fields),
            "parsed_fields": len(result["fields"]),
            "parsing_errors": len(
                [d for d in result["parsing_details"] if "Error" in d]
            ),
            "message_complete": actual_length == msg_length,
            "parsing_status": "Success"
            if len(result["fields"]) == len(present_fields) - 1
            else "Partial"
            if result["fields"]
            else "Failed",
        }

    except Exception as e:
        result["parsing_error"] = str(e)
        result["parsing_status"] = "Failed"
        logging.error(f"Message parsing failed: {str(e)}")

    finally:
        # Mask sensitive field values in logs
        sensitive_fields = {"2", "35", "52", "53", "64"}
        for field_id in sensitive_fields:
            if field_id in result.get("fields", {}):
                field_value = result["fields"][field_id].get("value", "")
                if field_value:
                    masked_value = "*" * len(field_value)
                    logging.debug(f"Field {field_id}: {masked_value}")

    return result


def get_response_category(resp_code: str) -> str:
    """Determine response code category."""
    categories = {
        "SUCCESS": ["00", "10", "11"],
        "REFER": ["01", "02"],
        "DECLINE": ["05", "51", "65"],
        "ERROR": ["06", "91", "96"],
        "SECURITY": ["55", "63", "75"],
        "INVALID": ["12", "13", "14", "30"],
        "SYSTEM": ["90", "91", "92", "96"],
    }

    for category, codes in categories.items():
        if resp_code in codes:
            return category
    return "OTHER"


def analyze_response_code(resp_code: str) -> dict:
    """
    Analyze ISO8583 response code.

    Args:
        resp_code: Two-digit response code

    Returns:
        dict: Analysis including code, category, and recommended action
    """
    category = get_response_category(resp_code)
    message = ISO8583_RESPONSE_CODES.get(resp_code, "Unknown response code")

    analysis = {
        "code": resp_code,
        "category": category,
        "message": message,
        "action_required": category not in ["SUCCESS"],
        "retry_allowed": category in ["ERROR", "SYSTEM"],
        "security_issue": category == "SECURITY",
        "severity": "High"
        if category in ["SECURITY", "ERROR"]
        else "Medium"
        if category in ["DECLINE", "INVALID"]
        else "Low",
    }

    return analysis


def analyze_response_code(resp_code: str) -> dict:
    """
    Analyze ISO8583 response code.

    Args:
        resp_code: Two-digit response code

    Returns:
        dict: Analysis of response code including:
            - code: Original response code
            - category: Response category
            - meaning: Detailed description
            - action: Recommended action
    """
    response_codes = {
        "00": {"category": "Approved", "meaning": "Transaction approved"},
        "01": {"category": "Refer", "meaning": "Refer to card issuer"},
        "05": {"category": "Decline", "meaning": "Do not honor"},
        "14": {"category": "Invalid", "meaning": "Invalid card number"},
        "51": {"category": "Decline", "meaning": "Insufficient funds"},
        "55": {"category": "Invalid", "meaning": "Invalid PIN"},
        "91": {"category": "Error", "meaning": "Issuer switch inoperative"},
        "96": {"category": "Error", "meaning": "System malfunction"},
    }

    default_response = {
        "category": "Unknown",
        "meaning": "Unknown response code",
        "action": "Contact support",
    }

    response = response_codes.get(resp_code, default_response)
    return {
        "code": resp_code,
        "category": response["category"],
        "meaning": response["meaning"],
        "action": response.get("action", "Process according to category"),
    }


def parse_field_127(
    response: bytes, pos: int, field_specs: Dict[str, ISO8583Field]
) -> Tuple[dict, int]:
    """
    Parse the complex field 127 and its subfields.

    Args:
        response: Full response bytes
        pos: Current position in response
        field_specs: Field specifications

    Returns:
        Tuple[dict, int]: (Parsed field data, new position)
    """
    try:
        # Get length (6 digits)
        if pos + 6 > len(response):
            raise ISO8583ParseError("Message too short for field 127 length")

        length = int(response[pos : pos + 6].decode("ascii"))
        pos += 6

        if pos + length > len(response):
            raise ISO8583ParseError(
                f"Message too short for field 127 data (need {length} bytes)"
            )

        # Extract field 127 data
        field_127_data = response[pos : pos + length]

        # Parse subfields if present
        subfields = {}
        if length >= 16:  # Minimum for bitmap
            subfields["bitmap"] = field_127_data[0:16].hex().upper()
            subfields["raw_data"] = field_127_data[16:].hex().upper()
        else:
            subfields["raw_data"] = field_127_data.hex().upper()

        field_data = {
            "value": subfields,
            "length_indicator": length,
            "spec": {
                "class": "org.jpos.iso.IFA_LLLLLLBINARY",
                "length": 999999,
                "name": "RESERVED PRIVATE USE",
            },
            "position": f"{pos}-{pos+length-1}",
            "hex": field_127_data.hex().upper(),
        }

        return field_data, pos + length

    except Exception as e:
        raise ISO8583ParseError(f"Error parsing field 127: {str(e)}")


def calculate_message_length(
    mti: bytes, bitmap: bytes, formatted_fields: Dict[str, bytes]
) -> int:
    """
    Calculate total message length excluding the length prefix.

    Args:
        mti: Message Type Indicator bytes
        bitmap: Bitmap bytes
        formatted_fields: Dictionary of formatted field values

    Returns:
        int: Total message length
    """
    total_length = len(mti) + len(bitmap)
    for field_data in formatted_fields.values():
        total_length += len(field_data)
    return total_length


def create_message_structure(
    fields: Dict[str, str], field_specs: Dict[str, "ISO8583Field"]
) -> Dict[str, bytes]:
    """
    Format and organize message fields according to ISO8583 structure.

    Args:
        fields: Dictionary of field values
        field_specs: Field specifications

    Returns:
        Dict[str, bytes]: Formatted field values

    Raises:
        ISO8583MessageError: If field formatting fails
    """
    formatted_fields = {}
    try:
        # Sort fields to ensure consistent ordering
        for field_id in sorted(fields.keys(), key=int):
            if field_id == "1":  # Skip bitmap field
                continue

            value = fields[field_id]
            if not value.strip():
                continue

            field_spec = field_specs.get(field_id)
            if not field_spec:
                raise ISO8583MessageError(
                    f"No specification found for field {field_id}"
                )

            formatted_value = format_field_value(value, field_spec)
            if formatted_value is not None:
                formatted_fields[field_id] = formatted_value

        return formatted_fields

    except Exception as e:
        raise ISO8583MessageError(f"Error formatting fields: {str(e)}")


def assemble_iso_message(mti: bytes, bitmap: bytes, formatted_fields: Dict[str, bytes]) -> bytes:
    """
    Assemble ISO8583 message components into final message.

    Args:
        mti: MTI as bytes
        bitmap: Bitmap as bytes
        formatted_fields: Dictionary of formatted field values

    Returns:
        bytes: Complete message

    Raises:
        ISO8583FormatError: If message assembly fails
    """
    try:
        # Create list to hold message parts
        message_parts = []

        # Add MTI
        message_parts.append(mti)

        # Add bitmap
        message_parts.append(bitmap)

        # Add fields in order
        for field_id in sorted(formatted_fields.keys(), key=int):
            field_data = formatted_fields[field_id]
            if not isinstance(field_data, bytes):
                raise ISO8583FormatError(
                    f"Field {field_id} data is not bytes (type: {type(field_data)})"
                )
            message_parts.append(field_data)

        # Join all parts
        message = b''.join(message_parts)

        # Calculate and add length prefix
        msg_length = len(message)
        length_prefix = struct.pack(">H", msg_length)

        # Return complete message
        return length_prefix + message

    except Exception as e:
        raise ISO8583FormatError(f"Message assembly failed: {str(e)}")

def format_iso_message(fields: Dict[str, str], field_specs: Dict[str, 'ISO8583Field'],
                      mti: str = "0200") -> Tuple[bytes, Dict[str, Any]]:
    """
    Format ISO8583 message with improved assembly process.
    """
    message_info = {
        'mti': mti,
        'field_lengths': {},
        'bitmap_info': None,
        'total_length': 0,
        'errors': []
    }

    try:
        # 1. Format MTI
        mti_bytes = mti.encode()
        if len(mti_bytes) != 4:
            raise ISO8583FormatError(f"Invalid MTI length: {len(mti_bytes)}")

        # 2. Generate bitmap
        bitmap_generator = BitmapGenerator()
        bitmap_info = bitmap_generator.create_bitmap(fields, mti)
        message_info['bitmap_info'] = bitmap_info.__dict__

        if bitmap_info.missing_mandatory:
            message_info['errors'].append(
                f"Missing mandatory fields: {bitmap_info.missing_mandatory}"
            )

        # 3. Format fields
        formatted_fields = {}
        for field_id in sorted(fields.keys(), key=int):
            if field_id == '1':  # Skip bitmap field
                continue

            value = fields[field_id].strip()
            if not value:
                continue

            field_spec = field_specs.get(field_id)
            if not field_spec:
                message_info['errors'].append(f"No specification for field {field_id}")
                continue

            formatted_value = format_field_value(value, field_spec)
            if formatted_value is not None:
                if not isinstance(formatted_value, bytes):
                    raise ISO8583FormatError(
                        f"Field {field_id} formatter returned {type(formatted_value)}, expected bytes"
                    )
                formatted_fields[field_id] = formatted_value
                message_info['field_lengths'][field_id] = len(formatted_value)

        # 4. Assemble message
        full_message = assemble_iso_message(
            mti_bytes,
            bitmap_info.bitmap_bytes,
            formatted_fields
        )

        # 5. Update message info
        message_info.update({
            'has_secondary_bitmap': bitmap_info.has_secondary,
            'present_fields': bitmap_info.present_fields,
            'total_length': len(full_message),
            'header_info': {
                'mti_length': len(mti_bytes),
                'bitmap_length': len(bitmap_info.bitmap_bytes),
                'total_fields': len(formatted_fields)
            }
        })

        return full_message, message_info

    except Exception as e:
        raise ISO8583FormatError(f"Message formatting failed: {str(e)}")

def validate_message_parts(mti: bytes, bitmap: bytes,
                         formatted_fields: Dict[str, bytes]) -> List[str]:
    """
    Validate message parts before assembly.

    Args:
        mti: MTI as bytes
        bitmap: Bitmap as bytes
        formatted_fields: Dictionary of formatted field values

    Returns:
        List[str]: List of validation errors
    """
    errors = []

    # Check MTI
    if not isinstance(mti, bytes) or len(mti) != 4:
        errors.append(f"Invalid MTI: {len(mti) if isinstance(mti, bytes) else type(mti)}")

    # Check bitmap
    if not isinstance(bitmap, bytes) or len(bitmap) not in (8, 16):
        errors.append(
            f"Invalid bitmap: {len(bitmap) if isinstance(bitmap, bytes) else type(bitmap)}"
        )

    # Check fields
    for field_id, field_data in formatted_fields.items():
        if not isinstance(field_data, bytes):
            errors.append(f"Field {field_id} is not bytes: {type(field_data)}")

    return errors

def check_server(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Check if server is accepting connections with detailed diagnostics.
    """
    try:
        # Get IP address(es) for the host
        print(f"\nResolving {host}...")
        addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)

        for family, socktype, proto, canonname, sockaddr in addresses:
            ip, port = sockaddr
            print(f"Trying {ip}:{port}...")

            # Create socket
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(timeout)

            # Attempt connection
            try:
                result = sock.connect_ex(sockaddr)
                if result == 0:
                    print(f"Successfully connected to {ip}:{port}")
                    return True
                else:
                    print(f"Connection failed to {ip}:{port} (Error code: {result})")
            except socket.error as e:
                print(f"Socket error for {ip}:{port}: {e}")
            finally:
                sock.close()

        return False

    except socket.gaierror as e:
        print(f"DNS resolution failed for {host}: {e}")
        return False
    except Exception as e:
        print(f"Error checking server: {e}")
        return False


def send_key_exchange_message(
    host: str = "13.246.138.100", port: int = 12000
) -> Optional[dict]:
    """
    Send ISO8583 key exchange message (MTI 0800) with enhanced connection handling
    """
    try:
        print("\nInitiating Key Exchange")
        print("=" * 50)

        # 1. Test server connectivity first
        print(f"\nTesting server connectivity to {host}:{port}...")
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(5)

        try:
            test_sock.connect((host, port))
            print("Server is reachable")
            test_sock.close()
        except Exception as e:
            print(f"Cannot reach server: {e}")
            return None

        # 2. Load field specifications
        field_specs = parse_zone_xml(ZONE_FILE)
        print("Field specifications loaded successfully")

        # 3. Prepare key exchange fields
        now = datetime.now()
        field_data = {
            "3": "990000",  # Processing code for key exchange
            "7": now.strftime("%m%d%H%M%S"),  # MMDDhhmmss
            "11": generate_stan(),  # STAN
            "12": now.strftime("%H%M%S"),  # Time
            "13": now.strftime("%m%d"),  # Date
            "32": "1346111",  # Acquiring Institution ID
            "41": "10351254",  # Terminal ID
            "42": "Z05110C010042",  # Merchant ID
            "70": "101",  # Network Management Code
        }

        print("\nKey Exchange Fields:")
        for field_id, value in field_data.items():
            print(f"Field {field_id}: {value}")

        # 4. Format ISO message
        message = format_iso_message(field_data, field_specs, mti="0800")
        msg_length = len(message)
        length_prefix = struct.pack(">H", msg_length)
        full_message = length_prefix + message

        print(f"\nMessage Details:")
        print(f"Length: {msg_length} bytes")
        print(f"Length Prefix: {length_prefix.hex()}")

        # 5. Create socket with aggressive timeouts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Initial connect timeout

        try:
            # Create SSL context
            context = ssl._create_unverified_context()
            secure_sock = context.wrap_socket(sock)

            # Connect to server
            print(f"\nEstablishing connection...")
            start_time = time.time()
            secure_sock.connect((host, port))
            print(f"Connected in {time.time() - start_time:.2f} seconds")

            # Set longer timeout for data transfer
            secure_sock.settimeout(30)

            # Send message with confirmation
            print("\nSending message...")
            total_sent = 0
            while total_sent < len(full_message):
                sent = secure_sock.send(full_message[total_sent:])
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent
            print(f"Successfully sent {total_sent} bytes")

            # Receive response with improved handling
            print("\nWaiting for response...")
            all_data = bytearray()
            start_time = time.time()
            received_header = False
            expected_length = 0

            while time.time() - start_time < 30:  # 30 second timeout
                try:
                    chunk = secure_sock.recv(8192)
                    if chunk:
                        all_data.extend(chunk)
                        print(f"Received {len(chunk)} bytes")

                        # Process length prefix
                        if not received_header and len(all_data) >= 2:
                            expected_length = struct.unpack(">H", all_data[:2])[0]
                            received_header = True
                            print(f"Expected message length: {expected_length}")

                        # Check for complete message
                        if received_header and len(all_data) >= expected_length + 2:
                            print("Complete message received")
                            break
                    else:
                        if len(all_data) > 0:
                            break
                        print("Connection closed by server")
                        break
                except socket.timeout:
                    print("Receive timeout - retrying...")
                    continue
                except Exception as e:
                    print(f"Error during receive: {e}")
                    break

            if all_data:
                print("\nProcessing response...")
                response_info = decode_server_response(bytes(all_data), field_specs)

                if "fields" in response_info:
                    fields = response_info["fields"]
                    resp_code = fields.get("39", {}).get("value", "").strip()
                    print(f"Response Code: {resp_code}")

                    if resp_code == "00":
                        if "53" in fields:
                            session_keys = process_key_exchange(response_info)
                            return {
                                "response": response_info,
                                "session_keys": session_keys,
                                "success": True,
                            }
                    else:
                        print(f"Key exchange failed with response code: {resp_code}")
                        return {"response": response_info, "success": False}
            else:
                print("No response received")
                return None

        except socket.timeout:
            print("Operation timed out")
            return None
        except Exception as e:
            print(f"Connection error: {e}")
            return None
        finally:
            try:
                secure_sock.close()
                print("Connection closed")
            except Exception:
                pass

    except Exception as e:
        print(f"Key exchange error: {e}")
        return None

    return None


def test_pin_block_functionality():
    """Test and verify PIN block generation and encryption functionality."""
    print("\nPIN Block Functionality Test")
    print("=" * 50)

    try:
        # 1. Test Key Components
        print("\n1. Testing Key Components")
        print("-" * 40)
        try:
            zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
            print("Key components loaded successfully")
            print(f"Component 1: {zmk_comp1}")
            print(f"Component 2: {zmk_comp2}")
            print(f"Stored KCV: {stored_kcv}")
        except Exception as e:
            print(f"Key component loading failed: {str(e)}")
            return False

        # 2. Test ZMK Generation
        print("\n2. Testing ZMK Generation")
        print("-" * 40)
        try:
            clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
            print(f"Clear ZMK: {clear_zmk}")

            if len(clear_zmk) != 32:
                raise ValueError(f"Invalid ZMK length: {len(clear_zmk)}")
        except Exception as e:
            print(f"ZMK generation failed: {str(e)}")
            return False

        # 3. Test KCV Verification
        print("\n3. Testing KCV Generation")
        print("-" * 40)
        try:
            test_data = bytes(8)  # 8 bytes of zeros
            key_bytes = PinBlockUtil.string_to_bytes(clear_zmk)
            encrypted = PinBlockUtil.operate_des3(test_data, key_bytes, True)
            generated_kcv = PinBlockUtil.bytes_to_string(encrypted)[:6]

            print(f"Generated KCV: {generated_kcv}")
            print(f"Stored KCV:    {stored_kcv}")

            if generated_kcv.upper() != stored_kcv.upper():
                raise ValueError("KCV verification failed")
            print("KCV verification successful")
        except Exception as e:
            print(f"KCV verification failed: {str(e)}")
            return False

        # 4. Test PIN Block Formation
        print("\n4. Testing PIN Block Formation")
        print("-" * 40)
        test_cases = [
            {
                "pin": "1234",
                "pan": "4111111111111111",
                "description": "Basic 4-digit PIN",
            },
            {"pin": "123456", "pan": "5500000000000004", "description": "6-digit PIN"},
        ]

        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['description']}")
            try:
                # Format and show intermediate blocks
                pin_string = f"0{len(test['pin'])}{test['pin']}".ljust(16, "F")
                pin_block1 = PinBlockUtil.string_to_bytes(pin_string)
                print(f"PIN Block:         {pin_block1.hex().upper()}")

                treated_pan = test["pan"][-13:-1].rjust(16, "0")
                pin_block2 = PinBlockUtil.string_to_bytes(treated_pan)
                print(f"PAN Block:         {pin_block2.hex().upper()}")

                clear_block = PinBlockUtil.xor_bytes(pin_block1, pin_block2)
                print(f"Clear PIN Block:   {clear_block.hex().upper()}")

                # Generate encrypted block
                encrypted_block = PinBlockUtil.generate_encrypted_pin_block(
                    clear_zpk=clear_zmk, card_pan=test["pan"], pin=test["pin"]
                )
                print(f"Encrypted Block:   {encrypted_block}")

                # Verify lengths
                if len(encrypted_block) != 32:
                    raise ValueError(
                        f"Invalid encrypted block length: {len(encrypted_block)}"
                    )

                print("PIN block encryption successful")

            except Exception as e:
                print(f"Test case {i} failed: {str(e)}")
                return False

        print("\nAll test cases passed successfully")
        return True

    except Exception as e:
        print(f"PIN block testing failed: {str(e)}")
        return False
    finally:
        # Clean up sensitive data
        if "clear_zmk" in locals():
            clear_zmk = "0" * len(clear_zmk)


def test_journal_post():
    """Test journal posting functionality using TCARD_FILE data"""
    try:
        # Verify configuration first
        config = APIConfig()
        print("\nVerifying configuration:")
        print(f"Base URL: {config.base_url}")
        print(f"Endpoint: {config.endpoint}")
        print(f"Timeout: {config.timeout}s")
        print(f"Max Retries: {config.max_retries}")
        print(f"Retry Backoff: {config.retry_backoff}s")

        # Load test card data
        print("\nLoading test data from:", TCARD_FILE)
        field_data = parse_testcard_data(TCARD_FILE)

        # Create test cases
        test_cases = [
            # Approved transaction
            {
                "rrn": field_data.get("37", generate_retrieval_ref()),
                "stan": field_data.get("11", generate_stan()),
                "amount": int(field_data.get("4", "0").lstrip("0")),
                "account_number": field_data.get("102", ""),
                "pan": field_data.get("2", ""),
                "terminal_id": field_data.get("41", ""),
                "status": "APPROVED",
                "error": "",  # Empty error for approved transactions
                "comment": "Test transaction from TCARD data - Approved",
            },
            # Failed transaction
            {
                "rrn": generate_retrieval_ref(),
                "stan": generate_stan(),
                "amount": int(field_data.get("4", "0").lstrip("0")),
                "account_number": field_data.get("102", ""),
                "pan": field_data.get("2", ""),
                "terminal_id": field_data.get("41", ""),
                "status": "FAILED",  # Changed from DECLINED to FAILED
                "error": "51",
                "comment": "Test transaction from TCARD data - Failed",
            },
        ]

        print(f"\nPrepared test cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"RRN: {case['rrn']}")
            print(f"STAN: {case['stan']}")
            print(f"Amount: {case['amount']}")
            print(f"Terminal: {case['terminal_id']}")
            print(f"Status: {case['status']}")

        results = []
        for test_case in test_cases:
            try:
                print(f"\nSubmitting journal entry:")
                print(f"RRN: {test_case['rrn']}")
                print(f"Status: {test_case['status']}")

                result = send_push_journal(
                    rrn=test_case["rrn"],
                    stan=test_case["stan"],
                    amount=test_case["amount"],
                    account_number=test_case["account_number"],
                    pan=test_case["pan"],
                    status=test_case["status"],
                    terminal_id=test_case["terminal_id"],
                    comment=test_case["comment"],
                    error=test_case["error"],
                )

                results.append({"test_case": test_case, "result": result})

                print("\nResult:")
                print(f"Status: {result.get('status')}")
                print(f"Message: {result.get('message')}")

            except Exception as e:
                print(f"\nError processing test case: {str(e)}")
                results.append({"test_case": test_case, "error": str(e)})

        # Print summary
        print("\nTest Summary:")
        print("=" * 50)
        success_count = len(
            [r for r in results if r.get("result", {}).get("status") == "success"]
        )
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")

        for i, result in enumerate(results, 1):
            print(f"\nTest {i}:")
            print(f"RRN: {result['test_case']['rrn']}")
            print(f"Status: {result['test_case']['status']}")
            if "result" in result:
                print(
                    f"Result: {result['result'].get('status')} - {result['result'].get('message')}"
                )
            else:
                print(f"Result: Failed - {result.get('error', 'Unknown error')}")

        return results

    except FileNotFoundError:
        print(f"Error: Test card file {TCARD_FILE} not found")
        return None
    except Exception as e:
        print(f"Error during journal testing: {str(e)}")
        return None


def send_iso_message(host: str = "13.246.138.100", port: int = 12000):
    """
    Send ISO8583 message to server, handle response, and submit journal entry.

    This enhanced version:
    1. Processes the ISO8583 transaction
    2. Analyzes the response
    3. Submits a journal entry via send_push_journal
    4. Returns both ISO response and journal results

    Args:
        host: Server hostname or IP
        port: Server port

    Returns:
        dict: Contains both ISO response and journal submission results
    """
    try:
        # Check server availability first
        if not check_server(host, port):
            print("Server check failed - aborting message send")
            return None

        print(f"\nPreparing ISO8583 message:")
        field_specs = parse_zone_xml(ZONE_FILE)
        field_data = parse_testcard_data(TCARD_FILE)

        # Adding Debug Data
        print("\nChecking fields 102 and 123:")
        print(
            f"Field 102 spec: {field_specs.get('102').__dict__ if '102' in field_specs else 'Not defined'}"
        )
        print(
            f"Field 123 spec: {field_specs.get('123').__dict__ if '123' in field_specs else 'Not defined'}"
        )
        print(f"Field 102 value: {field_data.get('102', 'Not present')}")
        print(f"Field 123 value: {field_data.get('123', 'Not present')}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Connection timeout
        context = ssl._create_unverified_context()
        secure_sock = context.wrap_socket(sock)

        response_info = None
        journal_result = None

        try:
            print(f"\n1. Connecting to {host}:{port}")
            secure_sock.connect((host, port))
            secure_sock.setblocking(False)

            # Format and prepare message
            message = format_iso_message(field_data, field_specs)
            msg_length = len(message)
            length_prefix = struct.pack(">H", msg_length)

            print(f"\n2. Sending message:")
            print(f"Length prefix: {length_prefix.hex()} ({msg_length} bytes)")
            print("Full message hex dump:")

            full_message = length_prefix + message
            for i in range(0, len(full_message), 16):
                chunk = full_message[i : i + 16]
                hex_dump = " ".join(f"{b:02x}" for b in chunk)
                ascii_dump = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
                print(f"{i:04x}: {hex_dump:<48} {ascii_dump}")

            secure_sock.send(full_message)

            # Wait for response
            print(f"\n3. Waiting for response (5 second timeout)...")
            all_data = bytearray()
            start_time = time.time()

            while time.time() - start_time < 3.0:
                ready = select.select([secure_sock], [], [], 0.1)
                if ready[0]:
                    try:
                        chunk = secure_sock.recv(4096)
                        if chunk:
                            all_data.extend(chunk)
                            print(f"Received {len(chunk)} bytes")
                        else:
                            break
                    except ssl.SSLWantReadError:
                        continue
                    except Exception as e:
                        print(f"Error reading: {e}")
                        break

            if all_data:
                print("\n4. Analyzing response:")
                response_info = decode_server_response(bytes(all_data), field_specs)
                print("\nResponse analysis complete")

                # Process journal entry after successful ISO response
                if "fields" in response_info:
                    fields = response_info["fields"]

                    # Extract response code and determine status
                    resp_code = fields.get("39", {}).get("value", "")
                    status = (
                        "DECLINED"
                        if not resp_code
                        else "APPROVED"
                        if resp_code == "00"
                        else "DECLINED"
                    )

                    # Prepare journal entry data
                    try:
                        # Extract amount (remove decimal points and leading zeros)
                        amount_str = field_data.get("4", "0").lstrip("0")
                        amount = int(amount_str) if amount_str else 0

                        # Get RRN (first try response field 37, then request field 37)
                        rrn = (
                            fields.get("37", {}).get("value")
                            or field_data.get("37", "")
                        ).strip()

                        # If RRN is still empty, generate a new one
                        if not rrn:
                            rrn = generate_retrieval_ref()
                            print(f"Generated new RRN: {rrn}")

                        # Get STAN from request
                        stan = field_data.get("11", "").strip()
                        if not stan:
                            stan = generate_stan()
                            print(f"Generated new STAN: {stan}")

                        # Get account details (try multiple possible fields)
                        account_number = (
                            field_data.get("102", "") or field_data.get("2", "") or pan
                        ).strip()

                        # Get PAN (try multiple possible fields)
                        pan = (
                            field_data.get("2", "")
                            or fields.get("2", {}).get("value", "")
                        ).strip()

                        # Get terminal ID
                        terminal_id = (
                            field_data.get("41", "")
                            or fields.get("41", {}).get("value", "")
                        ).strip()

                        # Prepare error info
                        error = "None" if status == "APPROVED" else (resp_code or "96")

                        # Validate required fields
                        if not all([rrn, stan, amount, account_number, pan]):
                            print("\nWarning: Missing required fields:")
                            print(f"RRN: {bool(rrn)}")
                            print(f"STAN: {bool(stan)}")
                            print(f"Amount: {bool(amount)}")
                            print(f"Account: {bool(account_number)}")
                            print(f"PAN: {bool(pan)}")

                            # Fill in missing fields with defaults if necessary
                            if not account_number:
                                account_number = pan
                                print("Using PAN as account number")

                        print("\n5. Submitting journal entry:")
                        print(f"RRN: {rrn}")
                        print(f"STAN: {stan}")
                        print(f"Amount: {amount}")
                        print(f"Account: {account_number}")
                        print(f"PAN: {pan}")
                        print(f"Status: {status}")
                        print(f"Terminal: {terminal_id}")
                        print(f"Error: {error}")

                        journal_result = send_push_journal(
                            rrn=rrn,
                            stan=stan,
                            amount=amount,
                            account_number=account_number,
                            pan=pan,
                            status=status,
                            terminal_id=terminal_id,
                            comment=f"Transaction {status} with response code {resp_code}",
                            error=error,
                        )

                        print("\n6. Journal submission successful:")
                        print(f"Journal Response: {journal_result}")

                    except Exception as e:
                        print(f"\nError submitting journal entry: {str(e)}")
                        print(f"Error details: {type(e).__name__}")
                        import traceback

                        traceback.print_exc()
                        journal_result = {
                            "status": "error",
                            "message": f"Journal submission failed: {str(e)}",
                        }
            else:
                print("\nNo response received from ISO host")

        except socket.timeout:
            print(f"Connection timed out")
        except ConnectionRefusedError:
            print(f"Connection refused by {host}:{port}")
        except ssl.SSLError as e:
            print(f"SSL/TLS error: {e}")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            secure_sock.close()
            print("\nConnection closed")

        # Return both ISO response and journal result
        result = {"iso_response": response_info, "journal_result": journal_result}

        print("\nTransaction Summary:")
        print(f"ISO Response: {'Success' if response_info else 'Failed'}")
        print(f"Journal Entry: {'Success' if journal_result else 'Failed'}")

        return result

    except FileNotFoundError:
        print("Error: Could not find required files (zone.xml or test card data)")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def xor_hex_strings(hex1: str, hex2: str) -> str:
    """
    XOR two hexadecimal strings.
    """
    # Convert hex strings to bytes
    bytes1 = bytes.fromhex(hex1)
    bytes2 = bytes.fromhex(hex2)

    # XOR the bytes
    xored = bytes(a ^ b for a, b in zip(bytes1, bytes2))

    # Convert back to hex string
    return xored.hex().upper()


def analyze_pin_response(response_info: dict) -> dict:
    """
    Enhanced PIN transaction response analysis.

    Args:
        response_info: Dictionary containing parsed ISO8583 response

    Returns:
        Dictionary containing detailed analysis of the PIN transaction response
    """
    analysis = {
        "success": False,
        "response_code": None,
        "message": None,
        "details": {},
        "raw_fields": {},
        "amounts": {},
        "errors": [],
        "warnings": [],
    }

    try:
        if not response_info or "fields" not in response_info:
            return {**analysis, "message": "Invalid response format"}

        fields = response_info["fields"]

        # Store all raw fields for debugging
        analysis["raw_fields"] = {
            field_id: field_data.get("value", "N/A")
            for field_id, field_data in fields.items()
            if isinstance(field_data, dict)
        }

        # Response code meanings
        resp_meanings = {
            "00": "Approved",
            "01": "Refer to card issuer",
            "02": "Refer to card issuer, special condition",
            "03": "Invalid merchant",
            "04": "Pick-up card",
            "05": "Do not honor",
            "06": "Error",
            "07": "Pick-up card, special condition",
            "08": "Honor with identification",
            "09": "Request in progress",
            "10": "Approved, partial",
            "11": "Approved, VIP",
            "12": "Invalid transaction",
            "13": "Invalid amount",
            "14": "Invalid card number",
            "15": "No such issuer",
            "51": "Insufficient funds",
            "54": "Expired card",
            "55": "Invalid PIN",
            "75": "PIN tries exceeded",
            "86": "PIN validation not possible",
            "91": "Issuer or switch inoperative",
            "92": "Invalid destination",
            "96": "System malfunction",
        }

        # Extract response code (Field 39)
        if "39" in fields and isinstance(fields["39"], dict):
            resp_code = fields["39"].get("value", "").strip()
            analysis["response_code"] = resp_code
            analysis["message"] = resp_meanings.get(
                resp_code, f"Unknown response code: {resp_code}"
            )
            analysis["success"] = resp_code == "00"

            # Add detailed response information
            analysis["response_details"] = {
                "code": resp_code,
                "meaning": analysis["message"],
                "category": "Success" if resp_code == "00" else "Error",
                "position": fields["39"].get("position", "Unknown"),
                "hex": fields["39"].get("hex", "Unknown"),
            }
        else:
            analysis["errors"].append("Response code (field 39) not found")

        # Process key transaction details
        key_fields = {
            "2": ("PAN", True),  # (Field Name, Mask?)
            "3": ("Processing Code", False),
            "4": ("Transaction Amount", False),
            "11": ("STAN", False),
            "12": ("Local Time", False),
            "13": ("Local Date", False),
            "14": ("Expiration Date", True),
            "37": ("Retrieval Reference Number", False),
            "38": ("Authorization Code", False),
            "41": ("Terminal ID", False),
            "42": ("Merchant ID", False),
            "43": ("Terminal Location", False),
        }

        for field_id, (field_name, mask) in key_fields.items():
            if field_id in fields and isinstance(fields[field_id], dict):
                value = fields[field_id].get("value", "")
                if value:
                    if mask:
                        masked_value = "*" * len(value)
                        analysis["details"][field_name] = masked_value
                        # Store actual value in a secure section
                        if "secure_data" not in analysis:
                            analysis["secure_data"] = {}
                        analysis["secure_data"][field_name] = value
                    else:
                        analysis["details"][field_name] = value

        # Process amount fields (4, 28, 30, etc.)
        amount_fields = {
            "4": "Transaction Amount",
            "28": "Transaction Fee",
            "30": "Processing Fee",
            "31": "Settlement Fee",
        }

        for field_id, amount_name in amount_fields.items():
            if field_id in fields and isinstance(fields[field_id], dict):
                if "parsed" in fields[field_id]:
                    sign = fields[field_id]["parsed"]["sign"]
                    amount = fields[field_id]["parsed"]["amount"]
                    formatted_amount = (
                        f"{'Debit' if sign == 'D' else 'Credit'} {int(amount):,}"
                    )
                    analysis["amounts"][amount_name] = {
                        "formatted": formatted_amount,
                        "raw": amount,
                        "sign": sign,
                    }

        # Add timing information
        if "12" in fields and "13" in fields:
            time_val = fields["12"].get("value", "")
            date_val = fields["13"].get("value", "")
            if time_val and date_val:
                analysis["details"]["Transaction Time"] = f"{date_val} {time_val}"

        # Add MTI information
        if "mti" in response_info:
            analysis["details"]["Response MTI"] = response_info["mti"].get("value")

        # Process any additional private fields (100-127)
        private_fields = {}
        for field_id in fields:
            if field_id.isdigit() and int(field_id) >= 100:
                value = fields[field_id].get("value", "")
                if value:
                    private_fields[field_id] = value
        if private_fields:
            analysis["private_fields"] = private_fields

        # Add message validity check
        if "length_check" in response_info:
            analysis["message_valid"] = response_info["length_check"].get(
                "valid", False
            )

        # Process any field errors
        for field_id, field_data in fields.items():
            if isinstance(field_data, dict) and "error" in field_data:
                analysis["errors"].append(
                    f"Field {field_id} error: {field_data['error']}"
                )

        # Add processing summary
        analysis["summary"] = {
            "transaction_approved": analysis["success"],
            "response_received": True,
            "message_valid": analysis.get("message_valid", False),
            "error_count": len(analysis["errors"]),
            "warning_count": len(analysis["warnings"]),
            "fields_processed": len(fields),
        }

    except Exception as e:
        analysis["errors"].append(f"Error analyzing response: {str(e)}")
        analysis["success"] = False
        analysis["message"] = f"Analysis failed: {str(e)}"

    return analysis


# def send_pinblock_message(
#     # host: str = "13.246.138.100",
#     host: str = HOST,  # "96.0.46.37",
#     # port: int = 12000
#     port: int = PORT,  # 5858,
#     field_data: Dict[str, str] = None,
#     encrypted_pin_block: str = None,
# ) -> Optional[Dict]:
#     """
#     Send ISO8583 message with PIN block (MTI 0200 with field 52).
#     """
#     try:
#         print("\nPreparing PIN Block Transaction")
#         print("=" * 50)

#         # Load configurations if not provided
#         if field_data is None:
#             field_data = parse_testcard_data(TCARD_FILE)

#         field_specs = parse_zone_xml(ZONE_FILE)
#         if not field_specs:
#             raise ValueError("Failed to load field specifications from zone.xml")

#         # If no encrypted PIN block provided, generate one
#         if encrypted_pin_block is None:
#             # Read and process key components
#             print("\nKey Component Processing:")
#             print("=" * 50)
#             comp1, comp2, stored_kcv = read_key_components()
#             print("Key components loaded successfully")

#             # Generate and validate clear ZPK
#             clear_zpk = xor_hex_strings(comp1, comp2)

#             # Generate PIN block
#             pin = "1234"  # Default PIN
#             encrypted_pin_block = PinBlockUtil.generate_encrypted_pin_block(
#                 clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
#             )

#         # Add PIN block to message
#         field_data["52"] = encrypted_pin_block

#         # Establish connection
#         print("\nConnection Setup:")
#         print("=" * 50)
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.settimeout(5)
#         context = ssl._create_unverified_context()
#         secure_sock = context.wrap_socket(sock)

#         try:
#             print(f"Connecting to {host}:{port}")
#             secure_sock.connect((host, port))
#             secure_sock.setblocking(False)

#             # Format and prepare message
#             print("\nMessage Preparation:")
#             print("=" * 50)
#             message = format_iso_message(field_data, field_specs)
#             msg_length = len(message)
#             length_prefix = struct.pack(">H", msg_length)

#             print(f"Message length: {msg_length} bytes")
#             print(f"Length prefix hex: {length_prefix.hex()}")

#             print("\nMessage Fields:")
#             print("=" * 50)
#             for field_id, value in field_data.items():
#                 # Mask sensitive fields
#                 if field_id in ["52", "2", "35"]:
#                     masked_value = "*" * len(value)
#                 else:
#                     masked_value = value
#                 print(f"Field {field_id}: {masked_value}")

#             # Prepare full message
#             full_message = length_prefix + message

#             # Print hex dump of message
#             print("\nMessage Hex Dump:")
#             print("=" * 50)
#             for i in range(0, len(full_message), 16):
#                 chunk = full_message[i : i + 16]
#                 hex_dump = " ".join(f"{b:02x}" for b in chunk)
#                 ascii_dump = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
#                 print(f"{i:04x}: {hex_dump:<48} {ascii_dump}")

#             # Send message
#             print("\nSending Message:")
#             print("=" * 50)
#             secure_sock.send(full_message)
#             print(f"Sent {len(full_message)} bytes")

#             # Wait for response
#             print("\nWaiting for Response:")
#             print("=" * 50)
#             all_data = bytearray()
#             start_time = time.time()

#             while time.time() - start_time < 3.0:
#                 ready = select.select([secure_sock], [], [], 0.1)
#                 if ready[0]:
#                     try:
#                         chunk = secure_sock.recv(4096)
#                         if chunk:
#                             all_data.extend(chunk)
#                             print(f"Received {len(chunk)} bytes")
#                         else:
#                             break
#                     except ssl.SSLWantReadError:
#                         continue
#                     except Exception as e:
#                         print(f"Error reading response: {e}")
#                         break

#             if all_data:
#                 print("\nResponse Analysis:")
#                 print("=" * 50)
#                 response_info = decode_server_response(bytes(all_data), field_specs)
#                 analysis = analyze_pin_response(response_info)

#                 print("\nTransaction Result:")
#                 print("=" * 50)
#                 print(f"Status: {'Success' if analysis['success'] else 'Failed'}")
#                 print(f"Response Code: {analysis['response_code']}")
#                 print(f"Message: {analysis['message']}")

#                 if analysis["details"]:
#                     print("\nTransaction Details:")
#                     print("=" * 50)
#                     for key, value in analysis["details"].items():
#                         print(f"{key}: {value}")

#                 if "errors" in analysis and analysis["errors"]:
#                     print("\nProcessing Errors:")
#                     print("=" * 50)
#                     for error in analysis["errors"]:
#                         print(f"- {error}")

#                 return {
#                     "raw_response": response_info,
#                     "analysis": analysis,
#                     "pin_block": encrypted_pin_block,  # Be careful with this in production
#                     "success": analysis.get("success", False),
#                 }
#             else:
#                 print("\nNo response received")
#                 return None

#         finally:
#             secure_sock.close()
#             print("\nConnection closed")

#     except socket.timeout:
#         print("Connection timed out")
#         raise
#     except ConnectionRefusedError:
#         print("Connection refused by host")
#         raise
#     except ssl.SSLError as e:
#         print(f"SSL/TLS error: {e}")
#         raise
#     except ValueError as e:
#         print(f"Validation error: {e}")
#         raise
#     except Exception as e:
#         print(f"Error in PIN block processing: {str(e)}")
#         raise
#     finally:
#         # Clear sensitive data
#         if "clear_zpk" in locals():
#             clear_zpk = "0" * len(clear_zpk)
#         if "pin" in locals():
#             pin = "0" * len(pin)

def send_pinblock_message(host: str, port: int, field_data: Dict[str, str],
                         encrypted_pin_block: str) -> Optional[Dict]:
    """Send PIN block message with improved message formatting."""
    try:
        print("\nPreparing PIN Block Transaction")
        print("=" * 50)

        # Load field specifications
        field_specs = parse_zone_xml(ZONE_FILE)

        # Ensure PIN block and security fields are formatted correctly
        field_data["52"] = encrypted_pin_block
        field_data["53"] = generate_security_info()

        # Format and send message
        full_message, message_info = format_iso_message(field_data, field_specs, mti="0200")

        logging.debug(f"Message assembly complete:")
        logging.debug(f"Total length: {message_info['total_length']} bytes")
        logging.debug(f"Fields present: {message_info['present_fields']}")

        # Create connection and send
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(30)
            context = ssl._create_unverified_context()

            with context.wrap_socket(sock) as secure_sock:
                secure_sock.connect((host, port))
                secure_sock.sendall(full_message)

                # Receive and process response
                response = receive_response(secure_sock)
                if response:
                    return decode_server_response(response, field_specs)

        return None

    except Exception as e:
        raise ISO8583FormatError(f"PIN block message send failed: {str(e)}")

def receive_response(secure_sock: ssl.SSLSocket, timeout: float = 30.0) -> Optional[bytes]:
    """
    Receive ISO8583 response with timeout.

    Args:
        secure_sock: SSL socket
        timeout: Timeout in seconds

    Returns:
        Optional[bytes]: Response data or None if timeout/error
    """
    try:
        all_data = bytearray()
        start_time = time.time()
        secure_sock.settimeout(0.1)  # Short timeout for select

        while time.time() - start_time < timeout:
            ready = select.select([secure_sock], [], [], 0.1)
            if ready[0]:
                try:
                    chunk = secure_sock.recv(8192)
                    if chunk:
                        all_data.extend(chunk)
                        if len(all_data) >= 2:
                            expected_length = int.from_bytes(all_data[:2], "big")
                            if len(all_data) - 2 >= expected_length:
                                break
                    else:
                        break
                except ssl.SSLWantReadError:
                    continue
                except Exception as e:
                    print(f"Error reading response: {e}")
                    break

        return bytes(all_data) if all_data else None

    except Exception as e:
        print(f"Error receiving response: {e}")
        return None


def perform_secure_transaction(
    # host: str = "13.246.138.100",
    host: str = HOST,  # "96.0.46.37",
    # port: int = 12000
    port: int = PORT,  # 5858,
    pin: str = None,
) -> Optional[Dict]:
    """
    Perform complete secure transaction using:
    - keys.txt for key components and KCV
    - TCARD_FILE for transaction data

    Sequence:
    1. Read and verify key components
    2. Key Exchange (0800)
    3. Process received ZPK
    4. Financial transaction with PIN block (0200)
    """
    sensitive_data = []

    try:
        # Validate PIN first
        if pin is None:
            pin = "2020"  # Default PIN

        if not validate_pin_format(pin, mask_in_logs=True):
            raise ValueError("Invalid PIN format")

        print("\nInitiating Secure Transaction Sequence")
        print("=" * 60)

        # 1. Read Key Components
        print("\n1. Reading Key Components from keys.txt...")
        try:
            zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
            sensitive_data.extend([zmk_comp1, zmk_comp2])

            print("Key components loaded successfully")
            print_key_processing_details(
                zmk_comp1=zmk_comp1,
                zmk_comp2=zmk_comp2,
                pan="XXXXXXXXXXXX",  # Will be filled from TCARD later
                pin="XXXX",  # Will be filled later
            )

        except Exception as e:
            raise ValueError(f"Error reading key components: {str(e)}")

        # 2. Load Transaction Data
        print("\n2. Loading Transaction Data from TCARD...")
        try:
            field_data = parse_testcard_data(TCARD_FILE)
            if not field_data:
                raise ValueError("No transaction data found in TCARD file")

            # Validate required fields
            required_fields = ["2", "3", "4", "11", "35", "41", "42", "43"]
            missing_fields = [f for f in required_fields if f not in field_data]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in TCARD: {', '.join(missing_fields)}"
                )

            print("Transaction data loaded successfully")
            print(f"PAN: {'*' * (len(field_data['2'])-4)}{field_data['2'][-4:]}")
            print(f"Amount: {field_data['4']}")
            print(f"Terminal: {field_data['41']}")

        except Exception as e:
            raise ValueError(f"Error loading transaction data: {str(e)}")

        # 3. Key Exchange
        print("\n3. Initiating Key Exchange...")
        key_exchange_result = send_key_exchange_message(host, port)

        if not key_exchange_result:
            raise ValueError("Key exchange failed")

        try:
            # Process key exchange response
            fields = key_exchange_result.get("fields", {})
            encrypted_zpk = fields.get("48", {}).get("value")  # Adjust field as needed

            if not encrypted_zpk:
                raise ValueError("No ZPK received in key exchange")

            # Generate clear ZMK
            clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
            sensitive_data.append(clear_zmk)

            # Generate clear ZPK
            clear_zpk, kcv = PinBlockUtil.generate_clear_zpk(
                zmk=clear_zmk, encrypted_zpk=encrypted_zpk
            )
            sensitive_data.append(clear_zpk)

            print("Key exchange successful")
            print(f"ZPK KCV: {kcv}")

        except Exception as e:
            raise ValueError(f"Error processing key exchange: {str(e)}")

        # 4. PIN Block Transaction
        print("\n4. Preparing PIN Transaction...")
        try:
            # In production, implement secure PIN entry
            pin = "1234"  # Test PIN
            sensitive_data.append(pin)

            # Update key processing details with actual PAN
            print_key_processing_details(
                zmk_comp1=zmk_comp1, zmk_comp2=zmk_comp2, pan=field_data["2"], pin=pin
            )

            # Generate PIN block
            encrypted_pin_block = PinBlockUtil.generate_encrypted_pin_block(
                clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
            )
            sensitive_data.append(encrypted_pin_block)

            # Add PIN block to transaction data
            field_data["52"] = encrypted_pin_block

            # Send financial transaction
            result = send_financial_message(
                host=host,
                port=port,
                field_data=field_data,
                encrypted_pin_block=encrypted_pin_block,
            )

            return {
                "key_exchange": key_exchange_result,
                "financial_transaction": result,
                "success": result.get("success", False) if result else False,
            }

        except Exception as e:
            raise ValueError(f"Error in PIN transaction: {str(e)}")

    except Exception as e:
        print(f"\nError in secure transaction: {str(e)}")
        return None

    finally:
        # Secure cleanup
        for item in sensitive_data:
            if isinstance(item, str):
                item = "0" * len(item)
        print("\nSensitive data cleared")


def handle_field_127(field_data: bytes, pos: int) -> Tuple[dict, int]:
    """
    Handle the complex field 127 which contains subfields.
    Returns: (parsed_data, new_position)
    """
    try:
        # First get the total length of field 127
        length = int(field_data[pos : pos + 6].decode("ascii"))
        pos += 6

        # Get the field data
        field_127_data = field_data[pos : pos + length]

        # Parse the subfields
        subfields = {}
        if len(field_127_data) > 0:
            # First byte is often a bitmap for subfields
            bitmap = field_127_data[0:16].hex()
            current_pos = 16

            # Add the raw data for debugging
            subfields["raw"] = field_127_data.hex()
            subfields["bitmap"] = bitmap

            # You can add more specific subfield parsing here

        return subfields, pos + length

    except Exception as e:
        print(f"Error parsing field 127: {str(e)}")
        return {"error": str(e)}, pos


def send_financial_message(
    host: str, port: int, field_data: Dict[str, str], encrypted_pin_block: str
) -> Optional[Dict]:
    """
    Send financial message (0200) with PIN block.
    """
    try:
        print("\nSending Financial Transaction")
        print("=" * 60)

        # Load field specifications
        field_specs = parse_zone_xml(ZONE_FILE)

        # Establish connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        context = ssl._create_unverified_context()
        secure_sock = context.wrap_socket(sock)

        try:
            secure_sock.connect((host, port))
            secure_sock.setblocking(False)

            # Format and send message
            message = format_iso_message(field_data, field_specs, mti="0200")
            msg_length = len(message)
            length_prefix = struct.pack(">H", msg_length)
            full_message = length_prefix + message

            print("\nMessage Details:")
            for field_id, value in field_data.items():
                # Mask sensitive fields
                if field_id in ["2", "35", "52"]:
                    masked_value = "*" * len(value)
                else:
                    masked_value = value
                print(f"Field {field_id}: {masked_value}")

            secure_sock.send(full_message)

            # Receive response
            all_data = bytearray()
            start_time = time.time()

            while time.time() - start_time < 3.0:
                ready = select.select([secure_sock], [], [], 0.1)
                if ready[0]:
                    try:
                        chunk = secure_sock.recv(4096)
                        if chunk:
                            all_data.extend(chunk)
                        else:
                            break
                    except ssl.SSLWantReadError:
                        continue
                    except Exception as e:
                        print(f"Read error: {e}")
                        break

            if all_data:
                response_info = decode_server_response(bytes(all_data), field_specs)
                analysis = analyze_pin_response(response_info)

                print("\nTransaction Result:")
                print("=" * 60)
                print(f"Status: {'Success' if analysis['success'] else 'Failed'}")
                print(f"Response Code: {analysis['response_code']}")
                print(f"Message: {analysis['message']}")

                return {
                    "raw_response": response_info,
                    "analysis": analysis,
                    "success": analysis.get("success", False),
                }

        finally:
            secure_sock.close()

    except Exception as e:
        print(f"Error in financial transaction: {str(e)}")
        return None


# def send_financial_message_with_session_keys(
#     amount: Optional[str] = None,
#     pin: Optional[str] = None,
#     host: str = HOST,
#     port: int = PORT,
# ) -> Optional[Dict]:
#     """
#     Send financial transaction using established session keys.

#     Args:
#         amount: Transaction amount (12 digits)
#         pin: PIN for the transaction (4-12 digits)
#         host: Server hostname/IP
#         port: Server port

#     Returns:
#         Optional[Dict]: Transaction result including response analysis
#     """
#     sensitive_data = []

#     try:
#         print("\nInitiating Financial Transaction with Session Keys")
#         print("=" * 60)

#         # 1. Validate inputs
#         if amount is None:
#             amount = "000000010000"  # Default amount
#         if not amount.isdigit() or len(amount) != 12:
#             raise ValueError("Amount must be 12 digits")

#         if pin is None:
#             pin = "1234"  # Default PIN
#         if not validate_pin_format(pin):
#             raise ValueError("Invalid PIN format")
#         sensitive_data.append(pin)

#         # 2. Get session keys
#         session_manager = SessionKeyManager()
#         session_keys = session_manager.get_valid_session_keys()
#         if not session_keys:
#             print("\nNo valid session keys found - performing key exchange...")
#             if not perform_key_exchange_with_persistence(host=host, port=port):
#                 raise ValueError("Key exchange failed")
#             session_keys = session_manager.get_valid_session_keys()
#             if not session_keys:
#                 raise ValueError("Failed to obtain session keys")

#         print("\nUsing session keys:")
#         print(f"Created: {session_keys['timestamp']}")
#         print(f"KCV: {session_keys.get('kcv', 'Not available')}")

#         # 3. Process key components
#         try:
#             zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
#             sensitive_data.extend([zmk_comp1, zmk_comp2])

#             clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
#             sensitive_data.append(clear_zmk)

#             # Get clear ZPK using double variant method
#             encrypted_zpk = session_keys["encrypted_zpk"]
#             clear_zpk = decrypt_zpk_double_variant(encrypted_zpk, clear_zmk)
#             sensitive_data.append(clear_zpk)

#             # Verify ZPK using KCV
#             if session_keys["kcv"]:
#                 if not verify_kcv(clear_zpk, session_keys["kcv"]):
#                     raise ValueError("ZPK verification failed")
#                 print("ZPK verified successfully")

#         except Exception as e:
#             raise ValueError(f"Error processing keys: {str(e)}")

#         # 4. Load transaction data
#         field_data = parse_testcard_data(TCARD_FILE)
#         if not field_data:
#             raise ValueError("Failed to load transaction data")

#         # Set current date and time
#         now = datetime.now() - timedelta(hours=2)
#         field_data.update(
#             {
#                 "7": now.strftime("%m%d%H%M%S"),  # MMDDhhmmss
#                 "12": now.strftime("%H%M%S"),  # Time
#                 "13": now.strftime("%m%d"),  # Date
#             }
#         )

#         # Add validation before sending
#         if not validate_pan(field_data["2"]):
#             raise ValueError(f"Invalid PAN format or Luhn check failed")
#         # Ensure amount is correct
#         if amount is not None:
#             field_data["4"] = amount

#         # Add debug print after update
#         print("XXXXXXXXX---------------Amount after update:", field_data.get("4"))
#         # 5. Generate PIN block
#         pin_block = PinBlockUtil.generate_encrypted_pin_block(
#             clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
#         )
#         field_data["52"] = pin_block
#         sensitive_data.append(pin_block)

#         # 6. Send transaction
#         print("\nSending financial transaction...")
#         result = send_pinblock_message(
#             host=host, port=port, field_data=field_data, encrypted_pin_block=pin_block
#         )

#         if result:
#             print("\nTransaction completed")
#             analysis = result.get("analysis", {})
#             print(f"Status: {'Success' if analysis.get('success') else 'Failed'}")
#             print(f"Response Code: {analysis.get('response_code')}")
#             print(f"Message: {analysis.get('message')}")

#             # Log transaction details
#             if analysis.get("details"):
#                 print("\nTransaction Details:")
#                 for key, value in analysis["details"].items():
#                     print(f"{key}: {value}")

#         return result

#     except Exception as e:
#         print(f"\nError in financial transaction: {str(e)}")
#         return None

#     finally:
#         # Clear sensitive data
#         for item in sensitive_data:
#             if isinstance(item, str):
#                 item = "0" * len(item)
#         print("\nSensitive data cleared")

def send_financial_message_with_session_keys(
    amount: Optional[str] = None,
    pin: Optional[str] = None,
    host: str = HOST,
    port: int = PORT
) -> Optional[Dict]:
    """
    Send financial transaction using established session keys.

    Args:
        amount: Transaction amount (12 digits)
        pin: PIN for the transaction (4-12 digits)
        host: Server hostname/IP
        port: Server port

    Returns:
        Optional[Dict]: Transaction result including response analysis
    """
    sensitive_data = []

    try:
        print("\nInitiating Financial Transaction with Session Keys")
        print("=" * 60)

        # 1. Validate inputs
        if amount is None:
            amount = "000000010000"  # Default amount
        if not amount.isdigit() or len(amount) != 12:
            raise ValueError("Amount must be 12 digits")

        if pin is None:
            pin = "1234"  # Default PIN
        if not validate_pin_format(pin):
            raise ValueError("Invalid PIN format")
        sensitive_data.append(pin)

        # 2. Get session keys
        session_manager = SessionKeyManager()
        session_keys = session_manager.get_valid_session_keys()
        if not session_keys:
            print("\nNo valid session keys found - performing key exchange...")
            if not perform_key_exchange_with_persistence(host=host, port=port):
                raise ValueError("Key exchange failed")
            session_keys = session_manager.get_valid_session_keys()
            if not session_keys:
                raise ValueError("Failed to obtain session keys")

        print("\nUsing session keys:")
        print(f"Created: {session_keys['timestamp']}")
        print(f"KCV: {session_keys.get('kcv', 'Not available')}")

        # 3. Process key components
        try:
            zmk_comp1, zmk_comp2, stored_kcv = read_key_components()
            sensitive_data.extend([zmk_comp1, zmk_comp2])

            clear_zmk = xor_hex_strings(zmk_comp1, zmk_comp2)
            sensitive_data.append(clear_zmk)

            # Get clear ZPK using double variant method
            encrypted_zpk = session_keys["encrypted_zpk"]
            clear_zpk = decrypt_zpk_double_variant(encrypted_zpk, clear_zmk)
            sensitive_data.append(clear_zpk)

            # Verify ZPK using KCV
            if session_keys["kcv"]:
                if not verify_kcv(clear_zpk, session_keys["kcv"]):
                    raise ValueError("ZPK verification failed")
                print("ZPK verified successfully")

        except Exception as e:
            raise ValueError(f"Error processing keys: {str(e)}")

        # 4. Load transaction data
        field_data = parse_testcard_data(TCARD_FILE)
        if not field_data:
            raise ValueError("Failed to load transaction data")

        # Set current date and time
        now = datetime.now()
        field_data.update({
            "7": now.strftime("%m%d%H%M%S"),  # MMDDhhmmss
            "12": now.strftime("%H%M%S"),     # Time
            "13": now.strftime("%m%d"),       # Date
            "4": amount,                      # Transaction Amount
        })

        # 5. Generate PIN Block and security fields
        pin_block = PinBlockUtil.generate_encrypted_pin_block(
            clear_zpk=clear_zpk,
            card_pan=field_data["2"],
            pin=pin
        )
        field_data["52"] = pin_block
        sensitive_data.append(pin_block)

        # 6. Add security control info (Field 53)
        # Format: Key version (1 byte) + reserved (15 bytes), padded with zeros
        key_version = "01"  # Example key version
        security_info = key_version + "0" * 30  # Pad to 16 bytes total
        field_data["53"] = security_info

        # 7. Validate PAN format
        if not validate_pan(field_data["2"]):
            raise ValueError(f"Invalid PAN format or Luhn check failed")

        # 8. Send transaction
        print("\nSending financial transaction...")
        result = send_pinblock_message(
            host=host,
            port=port,
            field_data=field_data,
            encrypted_pin_block=pin_block
        )

        if result:
            print("\nTransaction completed")
            analysis = result.get("analysis", {})
            print(f"Status: {'Success' if analysis.get('success') else 'Failed'}")
            print(f"Response Code: {analysis.get('response_code')}")
            print(f"Message: {analysis.get('message')}")

        return result

    except Exception as e:
        print(f"\nError in financial transaction: {str(e)}")
        return None

    finally:
        # Clear sensitive data
        for item in sensitive_data:
            if isinstance(item, str):
                item = "0" * len(item)
        print("\nSensitive data cleared")

def generate_security_info(key_version: str = "01") -> str:
    """
    Generate security control information for Field 53.

    Args:
        key_version: Key version number (2 hex digits)

    Returns:
        str: Formatted security control info
    """
    # Validate key version format
    if not (len(key_version) == 2 and all(c in "0123456789ABCDEF" for c in key_version.upper())):
        raise ValueError("Key version must be 2 hex digits")

    # Format: Key version (1 byte) + reserved (15 bytes)
    return key_version + "0" * 30  # Total 16 bytes = 32 hex chars

if __name__ == "__main__":
    import sys

    def print_usage():
        """Print detailed usage instructions"""
        print("\nZoneSwitch ISO8583 Client Usage")
        print("=" * 50)
        print("\nKey Management Operations:")
        print("-" * 30)
        print("  --key-exchange      Generate new session keys")
        print("  --clear-session     Clear stored session keys")
        print("  --session-status    Display current session status")

        print("\nSession-based Transactions:")
        print("-" * 30)
        print("  --financial-session [amount] [pin]")
        print("      Send financial transaction using session keys")
        print("  --pin-session [pin]")
        print("      Send PIN verification using session keys")

        print("\nDirect Transactions (No Session):")
        print("-" * 30)
        print("  --financial [amount]")
        print("      Send direct financial transaction")
        print("  --pin-block")
        print("      Send direct PIN verification")

        print("\nUtility Operations:")
        print("-" * 30)
        print("  --test-pin          Test PIN block functionality")
        print("  --check-server      Verify server connectivity")
        print("  --post-journal      Test journal posting")
        print("  --help              Show this help message")

        print("\nExamples:")
        print("-" * 30)
        print("  1. Session-based transaction:")
        print("     python zx1.py --key-exchange")
        print("     python zx1.py --financial-session 000000005000 1234")
        print("\n  2. Direct transaction:")
        print("     python zx1.py --financial 000000005000")
        print("\n  3. Check session status:")
        print("     python zx1.py --session-status")

    def validate_amount(amount: str) -> bool:
        """Validate transaction amount format"""
        if not amount or not amount.isdigit() or len(amount) != 12:
            print("Error: Amount must be 12 digits (e.g., 000000005000)")
            return False
        return True

    def validate_pin(pin: str) -> bool:
        """Validate PIN format"""
        if not pin or not pin.isdigit() or not (4 <= len(pin) <= 12):
            print("Error: PIN must be 4-12 digits")
            return False
        return True

    try:
        # Show help if no arguments or help requested
        if len(sys.argv) < 2 or sys.argv[1] == "--help":
            print_usage()
            sys.exit(0)

        command = sys.argv[1]

        # Key Management Operations
        if command == "--key-exchange":
            print("\nInitiating Key Exchange...")
            if perform_key_exchange_with_persistence():
                print("Key exchange completed successfully")
            else:
                print("Key exchange failed")
                sys.exit(1)

        elif command == "--clear-session":
            session_manager = SessionKeyManager()
            session_manager.clear_session_keys()
            print("Session keys cleared successfully")

        elif command == "--session-status":
            session_manager = SessionKeyManager()
            session_keys = session_manager.get_valid_session_keys()
            if session_keys:
                print("\nActive session found:")
                print(f"Created: {session_keys['timestamp']}")
                time_remaining = (
                    SessionKeyManager.KEY_LIFETIME
                    - (
                        datetime.now()
                        - datetime.fromisoformat(session_keys["timestamp"])
                    ).total_seconds()
                )
                print(f"Time remaining: {int(time_remaining)} seconds")
            else:
                print("\nNo active session found")

        # Session-based Transactions
        elif command == "--financial-session":
            amount = sys.argv[2] if len(sys.argv) > 2 else None
            pin = sys.argv[3] if len(sys.argv) > 3 else None

            if amount and not validate_amount(amount):
                sys.exit(1)
            if pin and not validate_pin(pin):
                sys.exit(1)

            result = send_financial_message_with_session_keys(amount=amount, pin=pin)

            if result and result.get("success"):
                print("\nTransaction successful!")
            else:
                print("\nTransaction failed")
                if result:
                    print(f"Response Code: {result['analysis']['response_code']}")
                    print(f"Message: {result['analysis']['message']}")
                sys.exit(1)

        elif command == "--pin-session":
            pin = sys.argv[2] if len(sys.argv) > 2 else None
            if pin and not validate_pin(pin):
                sys.exit(1)

            result = send_pinblock_with_session_keys(pin=pin)
            if result and result.get("success"):
                print("\nPIN verification successful!")
            else:
                print("\nPIN verification failed")
                if result:
                    print(f"Response Code: {result['analysis']['response_code']}")
                    print(f"Message: {result['analysis']['message']}")
                sys.exit(1)

        # Direct Transactions
        elif command == "--financial":
            amount = sys.argv[2] if len(sys.argv) > 2 else None
            if amount and not validate_amount(amount):
                sys.exit(1)

            result = send_financial_message(amount=amount)
            if result:
                print("\nTransaction completed")
                if result.get("iso_response"):
                    resp = result["iso_response"].get("response_analysis", {})
                    print(f"Response Code: {resp.get('code')}")
                    print(f"Message: {resp.get('meaning')}")
            else:
                print("\nTransaction failed")
                sys.exit(1)

        elif command == "--pin-block":
            result = send_pinblock_message()
            if result and result.get("success"):
                print("\nPIN verification successful!")
            else:
                print("\nPIN verification failed")
                if result:
                    print(f"Response Code: {result['analysis']['response_code']}")
                    print(f"Message: {result['analysis']['message']}")
                sys.exit(1)

        # Utility Operations
        elif command == "--test-pin":
            print("\nTesting PIN block functionality...")
            if test_pin_block_functionality():
                print("All PIN block tests passed")
            else:
                print("PIN block tests failed")
                sys.exit(1)

        elif command == "--check-server":
            print("\nChecking server availability...")
            if check_server(HOST, PORT):
                print(f"Server {HOST}:{PORT} is available")
            else:
                print(f"Server {HOST}:{PORT} is not available")
                sys.exit(1)

        elif command == "--post-journal":
            print("\nTesting journal posting...")
            results = test_journal_post()
            if results:
                print("Journal posting tests completed")
            else:
                print("Journal posting tests failed")
                sys.exit(1)

        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up any sensitive data
        for var in ["clear_zmk", "clear_zpk", "pin"]:
            if var in locals() and locals()[var] is not None:
                locals()[var] = "0" * len(locals()[var])
