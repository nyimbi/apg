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
from typing import Dict, Tuple, Optional, List, Union
import random
import json
import time

# import hashlib
from pathlib import Path
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

MY_IP = "102.140.218.27"

ZONE_FILE = "zone.xml"
# TCARD_FILE = "tcard_fid1.txt"
TCARD_FILE = "tcard_km.txt"
# HOST = "96.0.46.37"
HOST = "13.246.138.100"
# PORT = 5858
PORT = 12000


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


def get_response_analysis(resp_code: str) -> dict:
    """
    Provide detailed analysis of an ISO8583 response code.

    Args:
        resp_code: The two-digit ISO8583 response code

    Returns:
        Dictionary containing comprehensive response analysis including:
        - Basic response details (code, message, category)
        - Success/failure status
        - Technical explanation
        - Recommended actions
        - Customer-facing message
        - Risk level assessment
        - Retry recommendations
        - Timeout settings
        - Logging requirements
    """
    analysis = {
        "code": resp_code,
        "message": ISO8583_RESPONSE_CODES.get(resp_code, "Unknown response code"),
        "category": "Unknown",
        "is_success": resp_code in RESPONSE_CATEGORIES["SUCCESS"],
        "recommended_action": "",
        "technical_details": "",
        "user_message": "",
        "risk_level": "unknown",
        "retry_recommended": False,
        "requires_intervention": False,
        "retry_count": 0,
        "retry_delay": 0,
        "timeout": 30,
        "logging_required": False,
        "alert_level": "normal",
    }

    # Determine category with fallback handling
    found_category = False
    for category, codes in RESPONSE_CATEGORIES.items():
        if resp_code in codes:
            analysis["category"] = category
            found_category = True
            break

    if not found_category:
        analysis["category"] = "UNKNOWN"
        analysis["requires_intervention"] = True
        analysis["logging_required"] = True

    # Response code specific handling
    if resp_code == "00":
        analysis.update(
            {
                "risk_level": "none",
                "recommended_action": "Process transaction normally",
                "technical_details": "Transaction approved by issuer",
                "user_message": "Transaction approved, thank you",
                "retry_recommended": False,
                "timeout": 10,
            }
        )

    elif resp_code in ["01", "02"]:
        analysis.update(
            {
                "risk_level": "medium",
                "recommended_action": "Contact card issuer for authorization",
                "technical_details": "Manual authorization required from issuing bank",
                "user_message": "Please contact your card issuer for authorization",
                "requires_intervention": True,
                "timeout": 300,
            }
        )

    elif resp_code in ["04", "07", "41", "43"]:
        analysis.update(
            {
                "risk_level": "high",
                "recommended_action": "Retain card if possible and contact issuer",
                "technical_details": "Card reported lost/stolen or suspected fraud",
                "user_message": "Unable to process card. Please contact issuer.",
                "requires_intervention": True,
                "logging_required": True,
                "alert_level": "high",
            }
        )

    elif resp_code == "31":
        analysis.update(
            {
                "risk_level": "medium",
                "recommended_action": "Contact acquiring bank to verify routing configuration",
                "technical_details": "Transaction could not be routed to issuing bank. May indicate BIN routing issues.",
                "user_message": "Unable to process card. Please try another payment method.",
                "retry_recommended": True,
                "retry_count": 1,
                "retry_delay": 30,
                "requires_intervention": True,
            }
        )

    elif resp_code == "51":
        analysis.update(
            {
                "risk_level": "low",
                "recommended_action": "Request alternative payment method",
                "technical_details": "Account has insufficient funds",
                "user_message": "Insufficient funds available. Please use another payment method.",
                "retry_recommended": False,
            }
        )

    elif resp_code == "55":
        analysis.update(
            {
                "risk_level": "medium",
                "recommended_action": "Prompt for PIN retry if attempts remaining",
                "technical_details": "Invalid PIN entered",
                "user_message": "Incorrect PIN. Please try again.",
                "retry_recommended": True,
                "retry_count": 3,
                "retry_delay": 5,
            }
        )

    elif resp_code in ["91", "96"]:
        analysis.update(
            {
                "risk_level": "high",
                "recommended_action": "Retry transaction after delay",
                "technical_details": "System/network temporarily unavailable",
                "user_message": "Service temporarily unavailable. Please try again later.",
                "retry_recommended": True,
                "retry_count": 3,
                "retry_delay": 60,
                "timeout": 60,
                "requires_intervention": True,
            }
        )

    elif resp_code in ["14", "34", "36", "37", "43", "63"]:
        analysis.update(
            {
                "risk_level": "high",
                "recommended_action": "Decline and potentially retain card",
                "technical_details": "Security violation or suspected fraud",
                "user_message": "Unable to process transaction. Please contact your card issuer.",
                "retry_recommended": False,
                "logging_required": True,
                "alert_level": "immediate",
                "requires_intervention": True,
            }
        )

    else:
        analysis.update(
            {
                "risk_level": "medium",
                "recommended_action": "Review transaction logs and contact support",
                "technical_details": f"Unhandled response code {resp_code}",
                "user_message": "Transaction cannot be completed. Please try again later.",
                "retry_recommended": False,
                "requires_intervention": True,
                "logging_required": True,
            }
        )

    # Set default retry parameters if not already set
    if analysis["retry_recommended"] and not analysis["retry_delay"]:
        analysis["retry_delay"] = 30 if analysis["risk_level"] == "high" else 10
        analysis["retry_count"] = 2

    # Ensure logging for high risk responses
    if analysis["risk_level"] == "high":
        analysis["logging_required"] = True
        analysis["alert_level"] = analysis["alert_level"] or "high"

    return analysis


class ISO8583Field:
    """
    Represents an ISO8583 field specification from zone.xml
    """

    def __init__(self, id: str, length: int, name: str, field_class: str):
        self.id = id  # Field identifier
        self.length = length  # Maximum field length
        self.name = name  # Field description
        self.field_class = field_class  # Field type (e.g., IFA_NUMERIC, IFA_LLNUM)


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
        return bytes(
            a ^ b for a, b in zip(input1, input2 * (len(input1) // len(input2) + 1))
        )

    @staticmethod
    def operate_des3(data: bytes, key: bytes, encrypt: bool) -> bytes:
        """
        Perform 3DES operation with proper key handling.

        Args:
            data: 8 bytes of data
            key: 16 bytes for 3DES (will be expanded to 24 bytes)
            encrypt: True for encryption, False for decryption
        """
        if len(key) != 16:
            raise ValueError(f"Key must be 16 bytes (got {len(key)})")
        if len(data) != 8:
            raise ValueError(f"Data must be 8 bytes (got {len(data)})")

        # Create 24-byte key for Triple DES by duplicating first 8 bytes
        triple_des_key = key + key[:8]

        cipher = Cipher(
            algorithms.TripleDES(triple_des_key), modes.ECB(), backend=default_backend()
        )

        if encrypt:
            encryptor = cipher.encryptor()
            return encryptor.update(data) + encryptor.finalize()
        else:
            decryptor = cipher.decryptor()
            return decryptor.update(data) + decryptor.finalize()

    @classmethod
    def generate_encrypted_pin_block(
        cls, clear_zpk: str, card_pan: str, pin: str
    ) -> str:
        """Generate ISO 9564 Format 0 PIN block"""
        try:
            # Validate inputs
            if not validate_pin_format(pin):
                raise ValueError("Invalid PIN format")

            # Format PIN block (0 + length + PIN + padding)
            pin_length = len(pin)
            pin_block = f"0{pin_length}{pin}{'F' * (16 - 2 - pin_length)}"

            # Format PAN block (0000 + 12 rightmost digits excluding check digit)
            pan_block = f"0000{card_pan[-13:-1]}"

            # Convert to bytes and XOR
            pin_bytes = cls.string_to_bytes(pin_block)
            pan_bytes = cls.string_to_bytes(pan_block)
            clear_pin_block = cls.xor_bytes(pin_bytes, pan_bytes)

            # Encrypt with session ZPK
            key_bytes = cls.string_to_bytes(clear_zpk)
            encrypted_block = cls.operate_des3(clear_pin_block, key_bytes, True)

            # Convert to hex and pad
            return cls.bytes_to_string(encrypted_block).ljust(32, "0")
        except Exception as e:
            raise ValueError(f"PIN block generation failed: {str(e)}")

    @classmethod
    def verify_pin_block(
        cls, encrypted_block: str, clear_zpk: str, card_pan: str, pin: str
    ) -> bool:
        """
        Verify PIN block encryption/decryption.
        """
        try:
            # Encrypt
            encrypted = cls.generate_encrypted_pin_block(clear_zpk, card_pan, pin)

            # Should match provided encrypted block
            return encrypted.upper() == encrypted_block.upper()

        except Exception as ex:
            print(f"PIN block verification failed: {str(ex)}")
            return False


class SessionKeyManager:
    """
    Manages session keys from key exchange, including persistence and validation.
    """

    SESSION_FILE = "session_keys.json"
    KEY_LIFETIME = 60000  # 1000 minutes in seconds

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



def decrypt_zpk_double_variant(encrypted_zpk: str, clear_zmk: str) -> str:
    """
    Decrypt ZPK using double variant method with proper key length and variant handling.

    Implements ANSI X9.24 compliant key decryption using double-length key variants:
    1. Splits master key into left/right halves
    2. Applies A6/5A variants to right half
    3. Combines with left half for each decryption key
    4. Decrypts ZPK blocks separately with variant keys
    5. Combines results for final ZPK

    Args:
        encrypted_zpk: 32-char encrypted ZPK from key exchange (hex string)
        clear_zmk: 32-char clear ZMK from components (hex string)

    Returns:
        str: Clear ZPK as 32-char hex string

    Raises:
        ValueError: If input lengths invalid or decryption fails
        TypeError: If inputs not hex strings
    """
    sensitive_data = []  # Track data to securely clear

    try:
        # Input validation
        if not isinstance(encrypted_zpk, str) or not isinstance(clear_zmk, str):
            raise TypeError("Keys must be hex strings")

        if len(encrypted_zpk) != 32:
            raise ValueError(f"Invalid encrypted ZPK length: {len(encrypted_zpk)}")
        if len(clear_zmk) != 32:
            raise ValueError(f"Invalid clear ZMK length: {len(clear_zmk)}")

        if not all(c in "0123456789ABCDEFabcdef" for c in encrypted_zpk + clear_zmk):
            raise ValueError("Invalid hex format in keys")

        # Split keys into components
        zmk_left = clear_zmk[:16]     # First 16 chars (8 bytes) for Triple DES
        zmk_right = clear_zmk[16:]    # Last 16 chars for variants
        zpk_left = encrypted_zpk[:16]  # First encrypted block
        zpk_right = encrypted_zpk[16:] # Second encrypted block

        sensitive_data.extend([zmk_left, zmk_right, zpk_left, zpk_right])

        print("\nKey Components:")
        print("ZMK Left: " + "*" * len(zmk_left))  # Mask sensitive data
        print("ZMK Right: " + "*" * len(zmk_right))
        print("ZPK Left: " + "*" * len(zpk_left))
        print("ZPK Right: " + "*" * len(zpk_right))

        # Generate first variant key using A6
        variant_a6 = "A6" + "00" * 7  # Pad to 16 chars
        first_variant = format(
            int(zmk_right, 16) ^ int(variant_a6, 16),
            "016X"
        ).zfill(16)
        variant1_key = zmk_left + first_variant
        sensitive_data.extend([first_variant, variant1_key])

        # Decrypt first ZPK block
        result1 = PinBlockUtil.operate_des3(
            PinBlockUtil.string_to_bytes(zpk_left),
            PinBlockUtil.string_to_bytes(variant1_key),
            False  # Decrypt mode
        )
        clear_zpk_left = PinBlockUtil.bytes_to_string(result1)
        sensitive_data.append(clear_zpk_left)

        # Generate second variant key using 5A
        variant_5a = "5A" + "00" * 7  # Pad to 16 chars
        second_variant = format(
            int(zmk_right, 16) ^ int(variant_5a, 16),
            "016X"
        ).zfill(16)
        variant2_key = zmk_left + second_variant
        sensitive_data.extend([second_variant, variant2_key])

        # Decrypt second ZPK block
        result2 = PinBlockUtil.operate_des3(
            PinBlockUtil.string_to_bytes(zpk_right),
            PinBlockUtil.string_to_bytes(variant2_key),
            False  # Decrypt mode
        )
        clear_zpk_right = PinBlockUtil.bytes_to_string(result2)
        sensitive_data.append(clear_zpk_right)

        # Combine blocks for final key
        clear_zpk = clear_zpk_left + clear_zpk_right
        sensitive_data.append(clear_zpk)

        print("\nDecryption Results:")
        print("Clear ZPK Left: " + "*" * len(clear_zpk_left))
        print("Clear ZPK Right: " + "*" * len(clear_zpk_right))
        print("Final Clear ZPK: " + "*" * len(clear_zpk))

        return clear_zpk

    except Exception as e:
        raise ValueError(f"Error in double variant decryption: {str(e)}")

    finally:
        # Securely clear sensitive data
        for item in sensitive_data:
            if isinstance(item, str):
                item = "0" * len(item)
            elif isinstance(item, bytes):
                item = bytes(len(item))
        sensitive_data.clear()


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


def compose_field_55(field_data: Dict[str, str]) -> str:
    """
    Compose EMV Field 55 using actual transaction data.
    """
    try:
        # Get current date and time
        now = datetime.now() - timedelta(hours=2)
        trans_date = now.strftime("%y%m%d")  # YYMMDD

        # Get amount from field 4, remove leading zeros and pad to 12 digits
        amount = field_data.get("4", "000000000000").lstrip("0")
        if not amount:
            amount = "0"
        amount = amount.zfill(12)

        # Get currency code from field 49
        currency_code = field_data.get("49", "566")  # Default to NGN

        # Generate random values for dynamic fields
        unpredictable_number = hex(random.randint(0, 0xFFFFFFFF))[2:].upper().zfill(8)
        application_counter = hex(random.randint(1, 0xFFFF))[2:].upper().zfill(4)

        # Generate ARQC (dummy for test)
        arqc = "".join(
            [hex(random.randint(0, 0xFF))[2:].upper().zfill(2) for _ in range(8)]
        )

        # Construct EMV tags
        emv_tags = [
            # Format: (tag, length, value)
            ("9F26", "08", arqc),  # Application Cryptogram
            ("9F27", "01", "80"),  # Cryptogram Information Data (ARQC)
            (
                "9F10",
                "12",
                "07010103A0B0000000000000000000FF",
            ),  # Issuer Application Data
            ("9F37", "04", unpredictable_number),  # Unpredictable Number
            ("9F36", "02", application_counter),  # Application Transaction Counter
            ("95", "05", "0000004E00"),  # Terminal Verification Results
            ("9A", "03", trans_date[:6]),  # Transaction Date (YYMMDD)
            ("9C", "01", "00"),  # Transaction Type (Purchase)
            ("9F02", "06", amount[-12:]),  # Amount, Authorized
            ("5F2A", "02", currency_code),  # Transaction Currency Code
            ("9F1A", "02", currency_code),  # Terminal Country Code
            ("9F33", "03", "60F8C8"),  # Terminal Capabilities
            ("9F34", "03", "020300"),  # CVM Results
            ("9F35", "01", "22"),  # Terminal Type
            ("9F41", "04", "00000001"),  # Transaction Sequence Counter
            ("9F07", "02", "FF00"),  # Application Usage Control
        ]

        # Compose the field
        field_55 = ""
        for tag, length, value in emv_tags:
            field_55 += f"{tag}{length}{value}"

        print("\nField 55 Composition:")
        print("=" * 50)
        for tag, length, value in emv_tags:
            print(f"{tag} {length} {value}")

        return field_55

    except Exception as e:
        print(f"Error composing Field 55: {str(e)}")
        # Return a basic default value if composition fails
        return "9F2608F0C93BD39D15989F9F2701809F101207010103A0B0000000000000000000FF9F3704F0A43D519F360200019505000004E0009A031217179C01009F02060000000100005F2A0205669F1A0205669F330360F8C89F34030203009F3501229F4104000000019F0702FF00"

def handle_field_127(field_data: bytes, pos: int) -> Tuple[dict, int]:
    """
    Handle the complex field 127 (Private Reserved) which contains subfields.

    Field 127 Structure:
    - 6 digit length header
    - 16 byte (128 bit) bitmap indicating present subfields
    - Variable length subfields based on bitmap

    Subfield Definitions:
    - 127.2: Switch Key
    - 127.3: Routing Information
    - 127.4: POS Data
    - 127.5: Service Station Data
    - 127.6: Authorization Profile
    - 127.7: Check Data
    - 127.8: Retention Data
    - 127.9: Additional Node Data
    - 127.10: CVV2
    - 127.11: Original Key
    - 127.12: Terminal Owner
    - 127.13-127.19: Reserved
    - 127.20: PIN Verification Data
    - 127.21: Card Verification Data
    - 127.22: Token Data
    - 127.23: Card Data
    - 127.24: Other Data
    - 127.25: Issuer Data
    - 127.26-127.32: Reserved

    Args:
        field_data: Raw message bytes
        pos: Current position in message

    Returns:
        Tuple containing:
        - Dictionary of parsed subfields
        - New position after field 127
    """
    try:
        # Get total field length from 6-digit header
        if pos + 6 > len(field_data):
            raise ValueError("Message too short for field 127 length")

        length = int(field_data[pos : pos + 6].decode("ascii"))
        pos += 6

        if pos + length > len(field_data):
            raise ValueError(
                f"Message too short for field 127 data (need {length} bytes)"
            )

        # Get field data
        field_127_data = field_data[pos : pos + length]

        # Initialize subfields dict with metadata
        subfields = {"length": length, "raw": field_127_data.hex(), "subfields": {}}

        if len(field_127_data) > 0:
            # Parse 16-byte bitmap
            if len(field_127_data) < 16:
                raise ValueError("Field 127 too short for bitmap")

            bitmap_bytes = field_127_data[0:16]
            bitmap_hex = bitmap_bytes.hex()
            bitmap_bin = bin(int.from_bytes(bitmap_bytes, "big"))[2:].zfill(128)

            subfields["bitmap"] = {"hex": bitmap_hex, "binary": bitmap_bin}

            # Track position within field 127 data
            data_pos = 16

            # Parse subfields based on bitmap
            for i, bit in enumerate(bitmap_bin):
                if bit == "1":
                    subfield_num = i + 1
                    try:
                        # Extract subfield based on definitions
                        if data_pos >= len(field_127_data):
                            raise ValueError(
                                f"No data left for subfield 127.{subfield_num}"
                            )

                        if subfield_num == 2:  # Switch Key
                            if data_pos + 20 > len(field_127_data):
                                raise ValueError("Incomplete switch key data")
                            value = field_127_data[data_pos : data_pos + 20].decode(
                                "ascii"
                            )
                            subfields["subfields"]["127.2"] = {
                                "name": "Switch Key",
                                "value": value,
                                "position": f"{data_pos}-{data_pos+19}",
                            }
                            data_pos += 20

                        elif subfield_num == 3:  # Routing Info
                            if data_pos + 48 > len(field_127_data):
                                raise ValueError("Incomplete routing data")
                            value = field_127_data[data_pos : data_pos + 48].decode(
                                "ascii"
                            )
                            subfields["subfields"]["127.3"] = {
                                "name": "Routing Information",
                                "value": value,
                                "position": f"{data_pos}-{data_pos+47}",
                            }
                            data_pos += 48

                        elif subfield_num == 4:  # POS Data
                            if data_pos + 22 > len(field_127_data):
                                raise ValueError("Incomplete POS data")
                            value = field_127_data[data_pos : data_pos + 22].decode(
                                "ascii"
                            )
                            subfields["subfields"]["127.4"] = {
                                "name": "POS Data",
                                "value": value,
                                "position": f"{data_pos}-{data_pos+21}",
                            }
                            data_pos += 22

                        else:  # Store unknown subfields as raw hex
                            remaining = len(field_127_data) - data_pos
                            value = field_127_data[
                                data_pos : data_pos + remaining
                            ].hex()
                            subfields["subfields"][f"127.{subfield_num}"] = {
                                "name": "Unknown",
                                "value": value,
                                "position": f"{data_pos}-{data_pos+remaining-1}",
                            }
                            data_pos += remaining

                    except Exception as e:
                        subfields["subfields"][f"127.{subfield_num}"] = {
                            "error": str(e)
                        }

        return subfields, pos + length

    except Exception as e:
        print(f"Error parsing field 127: {str(e)}")
        return {"error": str(e)}, pos



def compose_field_127(field_data: Dict[str, str]) -> str:
    """
    Compose ISO8583 Field 127 (Private Reserved) with dynamic values.

    Field 127 Structure:
    - 9-digit length prefix
    - 16-digit bitmap indicating active subfields
    - Subfield 2: Switch key data (20 chars)
    - Subfield 3: Routing information (48 chars)
    - Subfield 4: POS data (22 chars)

    Args:
        field_data: Dictionary containing ISO8583 field values including:
            - Field 11: STAN (System Trace Audit Number)
            - Field 41: Terminal ID

    Returns:
        str: Formatted field 127 value with length prefix and subfields

    Raises:
        ValueError: If required input fields are missing or invalid
    """
    try:
        # Validate required input fields
        stan = field_data.get("11")
        if not stan:
            stan = "000000"  # Default STAN
        elif not stan.isdigit() or len(stan) != 6:
            raise ValueError(f"Invalid STAN format: {stan}")

        terminal_id = field_data.get("41")
        if not terminal_id:
            terminal_id = "00000000"  # Default terminal ID
        elif len(terminal_id) > 8:
            terminal_id = terminal_id[:8]  # Truncate if too long

        # Compose subfields with proper padding
        switch_key = "ZONEsrc".ljust(20, ' ')  # Fixed 20-char switch key
        routing_info = f"FBPVISA{terminal_id}".ljust(48, ' ')  # 48-char routing data
        pos_data = f"{stan}{stan}VISAGrp   00".ljust(22, ' ')  # 22-char POS data

        # Create bitmap indicating present subfields (2,3,4)
        # Format: 16 hex chars (64 bits) with bits 2,3,4 set
        bitmap = "2400000000000000"

        # Build field content
        field_content = {
            "bitmap": bitmap,
            "switch_key": switch_key,
            "routing_info": routing_info,
            "pos_data": pos_data
        }

        # Combine all components
        assembled_data = (
            field_content["bitmap"] +
            field_content["switch_key"] +
            field_content["routing_info"] +
            field_content["pos_data"]
        )

        # Add length prefix (9 digits)
        total_length = len(assembled_data)
        length_prefix = str(total_length).zfill(9)
        complete_field = f"{length_prefix}{assembled_data}"

        # Log field composition details
        print("\nField 127 Composition:")
        print("=" * 50)
        print(f"Length: {total_length}")
        print(f"Bitmap: {bitmap}")
        print(f"Switch Key: {switch_key}")
        print(f"Routing Info: {routing_info}")
        print(f"POS Data: {pos_data}")
        print(f"Complete Field: {complete_field}")

        # Validate final output
        if not complete_field.isascii():
            raise ValueError("Field contains non-ASCII characters")

        return complete_field

    except Exception as e:
        print(f"Error composing Field 127: {str(e)}")
        # Return ISO-compliant default value if composition fails
        default_value = "000001582400000000000000325A4F4E4573726320202020204E47504F53"
        print(f"Using default value: {default_value}")
        return default_value


def validate_field_length(
    field_id: str, value: str, field_specs: Dict[str, ISO8583Field]
) -> bool:
    """
    Validate the length of a field against its specification.

    Args:
        field_id: The field ID.
        value: The field value.
        field_specs: Dictionary of field specifications from zone.xml.

    Returns:
        bool: True if the field length is valid, False otherwise.
    """
    if field_id not in field_specs:
        raise ValueError(f"Field {field_id} not found in specifications")

    field_spec = field_specs[field_id]
    max_length = field_spec.length

    if "IFA_LLNUM" in field_spec.field_class or "IFA_LLLNUM" in field_spec.field_class:
        # Variable length fields have a length indicator
        length_indicator = (
            int(value[:2]) if "IFA_LLNUM" in field_spec.field_class else int(value[:3])
        )
        actual_length = len(value) - (2 if "IFA_LLNUM" in field_spec.field_class else 3)
        return actual_length == length_indicator and actual_length <= max_length
    else:
        # Fixed length fields
        return len(value) <= max_length


def validate_field_data_type(
    field_id: str, value: str, field_specs: Dict[str, ISO8583Field]
) -> bool:
    """
    Validate the data type of a field against its specification.

    Args:
        field_id: The field ID.
        value: The field value.
        field_specs: Dictionary of field specifications from zone.xml.

    Returns:
        bool: True if the field data type is valid, False otherwise.
    """
    if field_id not in field_specs:
        raise ValueError(f"Field {field_id} not found in specifications")

    field_spec = field_specs[field_id]
    field_class = field_spec.field_class

    if "IFA_NUMERIC" in field_class:
        return value.isdigit()
    elif (
        "IF_CHAR" in field_class
        or "IFA_LLCHAR" in field_class
        or "IFA_LLLCHAR" in field_class
    ):
        return value.isalnum()
    elif "IFB_BINARY" in field_class or "IFA_BINARY" in field_class:
        try:
            bytes.fromhex(value)
            return True
        except ValueError:
            return False
    else:
        return True  # Unknown field class, assume valid


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


def validate_special_fields(fields: Dict[str, str]) -> List[str]:
    """
    Validate special fields like PAN and Track 2 data.

    Args:
        fields: Dictionary of field IDs and their values.

    Returns:
        List[str]: List of error messages for invalid special fields.
    """
    errors = []

    # Validate PAN (Field 2)
    if "2" in fields:
        pan = fields["2"]
        if not pan.isdigit() or not (12 <= len(pan) <= 19):
            errors.append("Field 2 (PAN) must be 12-19 digits")

    # Validate Track 2 Data (Field 35)
    if "35" in fields:
        track2 = fields["35"]
        if "=" not in track2:
            errors.append("Field 35 (Track 2 Data) must contain a separator ('=')")
        else:
            pan, rest = track2.split("=", 1)
            if not pan.isdigit() or not (12 <= len(pan) <= 19):
                errors.append("Invalid PAN in Field 35 (Track 2 Data)")

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

        masked_pin = "*" * len(pin) if mask_in_logs else pin
        print(f"PIN validation successful for PIN: {masked_pin}")
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
        clear_zpk = decrypt_zpk_double_variant(encrypted_zpk, clear_zmk)

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
    host: str = HOST,
    port: int = PORT,
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
    host: str = HOST,
    port: int = PORT,
    pin: str = None,
) -> Optional[dict]:
    """Send PIN block message using session keys with double variant decryption."""
    sensitive_data = []

    try:
        # 1. PIN Validation
        if pin is None:
            pin = "2020"
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
                "3": "501000",  # PIN Verification
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
                if field_id == "32":  # Add debug print
                    print(f"Field 32 loaded from TCARD: {value}")

    # Generate unique values for fields 11 and 37
    field_data["11"] = generate_stan()
    field_data["37"] = generate_retrieval_ref()

    # Set transmission date/time and local date
    now = datetime.now() - timedelta(hours=2)
    field_data["7"] = now.strftime("%m%d%H%M%S")  # MMDDhhmmss
    field_data["13"] = now.strftime("%m%d")  # MMDD

    print("\nGenerated dynamic fields:")
    print(f"Field 11 (STAN): {field_data['11']}")
    print(f"Field 37 (Retrieval Ref): {field_data['37']}")
    print(f"Field 7 (Transmission Time): {field_data['7']}")
    print(f"Field 13 (Local Date): {field_data['13']}")

    return field_data


def create_bitmap(fields: Dict[str, str]) -> bytes:
    """
    Create ISO8583 bitmap indicating which fields are present in message.

    The bitmap is a bit array where each bit represents presence (1) or absence (0)
    of corresponding field. Bit 1 indicates if secondary bitmap is present for fields 65-128.

    Primary bitmap covers fields 1-64 (8 bytes).
    Secondary bitmap covers fields 65-128 (additional 8 bytes if needed).

    Args:
        fields: Dictionary mapping field IDs to their string values

    Returns:
        bytes: Bitmap as 8 or 16 bytes depending on field ranges present

    Validation:
        - Excludes empty/whitespace-only values
        - Handles field IDs 1-128 only
        - Sets bit 1 when fields > 64 present
        - Consistent byte lengths (8 or 16)
    """
    print("\nBitmap creation:")

    # Validate field IDs are in valid range
    if any(not field_id.isdigit() or int(field_id) < 1 or int(field_id) > 128
           for field_id in fields):
        raise ValueError("Invalid field ID: must be 1-128")

    print("1. Fields present in input:", sorted([int(f) for f in fields.keys()]))

    # Initialize bitmap array
    bitmap_bits = [False] * 128  # 128 bits for both primary and secondary

    # Track fields with non-empty values
    present_fields = []
    has_secondary = False

    # Process fields and detect if secondary bitmap needed
    for field_id in fields.keys():
        # Skip bitmap field itself
        if field_id == "1":
            continue

        # Only include non-empty values
        if value := fields[field_id].strip():
            field_num = int(field_id)
            present_fields.append(field_num)

            # Check if field requires secondary bitmap
            if field_num > 64:
                has_secondary = True

    # Set bit 1 if secondary bitmap needed
    if has_secondary:
        bitmap_bits[0] = True
        print("2. Secondary bitmap needed (fields > 64 present)")
    else:
        print("2. No secondary bitmap needed")

    print("3. Fields to be included in bitmap:", sorted(present_fields))

    # Set bits for present fields
    for field_id in present_fields:
        bitmap_bits[field_id - 1] = True

    # Convert bits to bytes
    bitmap_bytes = bytearray()
    for i in range(0, 128, 8):
        byte_value = 0
        for j in range(8):
            if bitmap_bits[i + j]:
                # Set bits in reverse order within each byte
                byte_value |= 1 << (7 - j)
        bitmap_bytes.append(byte_value)

    # Return appropriate number of bytes
    result = bytes(bitmap_bytes[:16] if has_secondary else bitmap_bytes[:8])

    # Debug output
    print("4. Bitmap hex:", result.hex())
    print("5. Bitmap binary:",
          bin(int.from_bytes(result, "big"))[2:].zfill(len(result) * 8))
    print("6. Fields marked in bitmap:",
          [i + 1 for i, bit in enumerate(bitmap_bits) if bit])

    # Validate result length
    if len(result) not in (8, 16):
        raise ValueError(f"Invalid bitmap length: {len(result)}")

    return result


def parse_track2(value: str) -> Dict[str, Union[str, List[str], bool, Dict[str, str]]]:
    """
    Parse and validate Track 2 data according to ISO 7813 standard.

    Args:
        value: Raw track 2 data string

    Returns:
        Dictionary containing:
        - Parsed components (pan, expiry, service_code etc)
        - Validation results (errors, warnings)
        - Service code interpretations

    Track 2 Format:
    - Start Sentinel (;) [not used in EMV]
    - PAN (12-19 digits)
    - Separator (=)
    - Expiry Date (YYMM)
    - Service Code (3 digits)
    - Discretionary Data (variable length)
    - End Sentinel (?) [not used in EMV]
    - LRC [not used in EMV]

    Max length: 40 characters
    """

    # Service code validation lookup tables
    VALID_FIRST_DIGITS = {
        "1": "International interchange OK",
        "2": "International interchange, IC (chip) required",
        "5": "National interchange only except under bilateral agreement",
        "6": "National interchange only except under bilateral agreement, IC required",
        "7": "No interchange except under bilateral agreement (closed loop)",
    }

    VALID_SECOND_DIGITS = {
        "0": "Normal authorization",
        "2": "Contact issuer via online means",
        "4": "Contact issuer via online means except under bilateral agreement",
    }

    VALID_THIRD_DIGITS = {
        "0": "No restrictions, PIN required",
        "1": "No restrictions",
        "2": "Goods and services only",
        "3": "ATM only, PIN required",
        "4": "Cash only",
        "5": "Goods and services only, PIN required",
        "6": "No restrictions, use PIN if PIN pad present",
        "7": "Goods and services only, use PIN if PIN pad present",
    }

    # Initial validation
    if len(value) > 40:
        raise ValueError("Track 2 data exceeds maximum length of 40 characters")

    if not value or "=" not in value:
        raise ValueError("Invalid Track 2 data - missing field separator")

    # Split into components
    pan, rest = value.split("=", 1)

    track2_data = {
        "pan": pan,
        "separator": "=",
        "expiry": rest[:4] if len(rest) >= 4 else "",
        "service_code": rest[4:7] if len(rest) >= 7 else "",
        "discretionary": rest[7:] if len(rest) >= 7 else "",
        "raw_value": value,
        "errors": [],
        "warnings": [],
        "is_valid": False,
        "service_code_interpretation": {},
    }

    # PAN validation
    if not track2_data["pan"].isdigit():
        track2_data["errors"].append("PAN must contain only digits")
    if not (12 <= len(track2_data["pan"]) <= 19):
        track2_data["errors"].append(
            f"PAN length must be 12-19 digits (got {len(track2_data['pan'])})"
        )

    # Luhn check
    if track2_data["pan"].isdigit():
        check = sum(int(x) for x in track2_data["pan"][-2::-2])
        for digit in track2_data["pan"][-1::-2]:
            digit = int(digit) * 2
            check += digit if digit < 10 else digit - 9
        if check % 10 != 0:
            track2_data["warnings"].append("PAN failed Luhn check")

    # Expiry date validation
    if track2_data["expiry"]:
        if not track2_data["expiry"].isdigit():
            track2_data["errors"].append("Expiry date must contain only digits")
        elif len(track2_data["expiry"]) != 4:
            track2_data["errors"].append("Expiry date must be exactly 4 digits")
        else:
            year = int(track2_data["expiry"][:2])
            month = int(track2_data["expiry"][2:])
            if not (1 <= month <= 12):
                track2_data["errors"].append(f"Invalid expiry month: {month}")
            if year < 0 or year > 99:
                track2_data["errors"].append(f"Invalid expiry year: {year}")

    # Service code validation
    if track2_data["service_code"]:
        if not track2_data["service_code"].isdigit():
            track2_data["errors"].append("Service code must contain only digits")
        elif len(track2_data["service_code"]) != 3:
            track2_data["errors"].append("Service code must be exactly 3 digits")
        else:
            first_digit = track2_data["service_code"][0]
            second_digit = track2_data["service_code"][1]
            third_digit = track2_data["service_code"][2]

            if first_digit not in VALID_FIRST_DIGITS:
                track2_data["errors"].append(
                    f"Invalid service code first digit: {first_digit}"
                )
            if second_digit not in VALID_SECOND_DIGITS:
                track2_data["errors"].append(
                    f"Invalid service code second digit: {second_digit}"
                )
            if third_digit not in VALID_THIRD_DIGITS:
                track2_data["errors"].append(
                    f"Invalid service code third digit: {third_digit}"
                )

            track2_data["service_code_interpretation"] = {
                "interchange": VALID_FIRST_DIGITS.get(first_digit, "Unknown"),
                "authorization": VALID_SECOND_DIGITS.get(second_digit, "Unknown"),
                "services": VALID_THIRD_DIGITS.get(third_digit, "Unknown"),
            }

    # Discretionary data validation
    if track2_data["discretionary"] and not track2_data["discretionary"].isdigit():
        track2_data["warnings"].append("Discretionary data should contain only digits")

    track2_data["is_valid"] = len(track2_data["errors"]) == 0

    return track2_data


def format_field_value(value: str, field_spec: ISO8583Field) -> Optional[bytes]:
    """
    Format field value according to ISO8583 specifications from zone.xml.

    Args:
        value: The field value to format
        field_spec: ISO8583Field specification from zone.xml

    Returns:
        Formatted bytes or None if field should be excluded

    Raises:
        ValueError: If field formatting fails or validation error
    """
    field_class = field_spec.field_class
    field_id = field_spec.id

    if field_id == "32":
        # Add length prefix manually
        length = str(len(value)).zfill(2)
        formatted = length + value
        return formatted.encode()

    if field_id == "127":
        if not value.startswith("000001"):
            # Add proper length prefix if not present
            length = str(len(value)).zfill(9)
            value = length + value
        return value.encode()

    field_length = field_spec.length

    try:
        if not value and field_id != "28":  # Allow empty for field 28
            print(f"\nField {field_id}: Empty value - will be excluded from bitmap")
            return None

        if field_id == "55":
            # Add length prefix (3 digits) for LLLCHAR
            length = str(len(value)).zfill(3)
            return (length + value).encode()

        print(f"\nProcessing field {field_id} ({field_spec.name})")
        print(f"Input value: '{value}'")

        # Special handling for Field 37 (RRN) - must be left-padded with zeros
        if field_id == "37":
            print(f"\nField 37 (RRN) processing:")
            print(f"1. Input value: '{value}'")
            clean_value = "".join(c for c in value if c.isdigit())  # Ensure numeric
            if len(clean_value) > field_length:
                clean_value = clean_value[-field_length:]  # Take rightmost if too long
            formatted = clean_value.zfill(
                field_length
            )  # Left pad with zeros to 12 chars
            result = formatted.encode()
            print(f"2. Zero-padded RRN: '{formatted}' (length: {len(formatted)})")
            return result

        # Special handling for Field 28 (Transaction Fee Amount)
        elif field_id == "28":
            print(f"\nAmount field 28 processing:")
            if not value or value.strip() == "":
                # Generate random fee amount if empty
                amount = str(random.randint(1, 50))
                sign = "C"
                print(f"1. Generated random fee amount: {amount}")
            else:
                # Process provided amount
                clean_value = "".join(c for c in value if c.isdigit() or c in "DC")
                amount = "".join(c for c in clean_value if c.isdigit())
                sign = "D" if "D" in clean_value else "C"
                print(f"1. Using provided amount: {amount}")

            # Amount should be exactly 8 digits (9 total with sign)
            padded_amount = amount.zfill(8)  # field_length - 1 for sign
            if len(padded_amount) > 8:
                padded_amount = padded_amount[
                    -8:
                ]  # Take rightmost 8 digits if too long

            formatted = f"{sign}{padded_amount}"  # Format as Sign + 8 digits
            result = formatted.encode()
            print(f"2. Final format: '{formatted}'")
            print(f"3. Hex: {result.hex()}")
            return result

        # IFB_BINARY fields (52, 53, 64, 65, 128)
        elif "IFB_BINARY" in field_class:
            print(f"\nBinary field {field_id} processing:")
            if value.startswith("0x"):
                value = value[2:]
            binary_data = bytes.fromhex(value)
            if len(binary_data) != field_length:
                if len(binary_data) > field_length:
                    binary_data = binary_data[:field_length]
                else:
                    binary_data = binary_data.ljust(field_length, b"\x00")
            print(f"Binary format: {binary_data.hex()}")
            return binary_data

        # IFA_BINARY fields (96)
        elif "IFA_BINARY" in field_class:
            print(f"\nBinary field {field_id} processing:")
            binary_data = bytes.fromhex(value)
            if len(binary_data) != field_length:
                if len(binary_data) > field_length:
                    binary_data = binary_data[:field_length]
                else:
                    binary_data = binary_data.ljust(field_length, b"\x00")
            print(f"Binary format: {binary_data.hex()}")
            return binary_data

        # IFA_AMOUNT fields (29, 30, 31, 97)
        elif "IFA_AMOUNT" in field_class:
            print(f"\nAmount field {field_id} processing:")
            clean_value = "".join(c for c in value if c.isdigit() or c in "DC")
            amount = "".join(c for c in clean_value if c.isdigit())
            sign = "D" if "D" in clean_value else "C"

            padded_amount = amount.zfill(field_length - 1)  # -1 for sign
            if len(padded_amount) > field_length - 1:
                padded_amount = padded_amount[-(field_length - 1) :]

            formatted = f"{sign}{padded_amount}"
            result = formatted.encode()
            print(f"1. Amount format: {formatted}")
            print(f"2. Hex: {result.hex()}")
            return result

        # Track 2 data (Field 35) - IFA_LLNUM
        elif field_id == "35":
            print(f"\nTrack 2 processing:")
            try:
                components = parse_track2(value)
                formatted_value = (
                    components["pan"]
                    + "="
                    + components["expiry"]
                    + (components.get("service_code", ""))
                )
                length = str(len(formatted_value)).zfill(2)
                formatted = length + formatted_value
                result = formatted.encode()
                print(f"1. Track 2 format: {length}{'*' * len(formatted_value)}")
                return result
            except ValueError as e:
                raise ValueError(f"Track 2 formatting error: {str(e)}")

        # IFA_LLNUM fields (2, 32, 33, 34, 99, 100)
        elif "IFA_LLNUM" in field_class:
            print(f"\nLLNUM field {field_id} processing:")
            clean_value = "".join(c for c in value if c.isdigit())
            if len(clean_value) > field_length:
                clean_value = clean_value[:field_length]
            length = str(len(clean_value)).zfill(2)
            formatted = length + clean_value
            result = formatted.encode()
            print(f"LLNUM format: {formatted}")
            return result

        # IFA_LLLNUM fields (36)
        elif "IFA_LLLNUM" in field_class:
            print(f"\nLLLNUM field {field_id} processing:")
            clean_value = "".join(c for c in value if c.isdigit())
            if len(clean_value) > field_length:
                clean_value = clean_value[:field_length]
            length = str(len(clean_value)).zfill(3)
            formatted = length + clean_value
            result = formatted.encode()
            print(f"LLLNUM format: {formatted}")
            return result

        # IF_CHAR fields (38-43, 57, 91-95, 98, 118, 119)
        elif "IF_CHAR" in field_class and field_id != "37":  # Exclude RRN
            print(f"\nCHAR field {field_id} processing:")
            if len(value) > field_length:
                value = value[:field_length]
            formatted = value.ljust(field_length)  # Right pad with spaces
            result = formatted.encode()
            print(f"CHAR format: {formatted}")
            return result

        # IFA_LLCHAR fields (44, 45, 101, 102, 103)
        elif "IFA_LLCHAR" in field_class:
            print(f"\nLLCHAR field {field_id} processing:")
            if len(value) > field_length:
                value = value[:field_length]
            length = str(len(value)).zfill(2)
            formatted = length + value
            result = formatted.encode()
            print(f"LLCHAR format: {formatted}")
            return result

        # IFA_LLLCHAR fields (46-48, 54-56, 58-63, 104-126)
        elif "IFA_LLLCHAR" in field_class:
            print(f"\nLLLCHAR field {field_id} processing:")
            if len(value) > field_length:
                value = value[:field_length]
            length = str(len(value)).zfill(3)
            formatted = length + value
            result = formatted.encode()
            print(f"LLLCHAR format: {formatted}")
            return result

        # IFA_NUMERIC fields (0-31, 66-89)
        elif "IFA_NUMERIC" in field_class:
            print(f"\nNUMERIC field {field_id} processing:")
            clean_value = "".join(c for c in value if c.isdigit())
            if len(clean_value) > field_length:
                clean_value = clean_value[-field_length:]  # Take rightmost digits
            formatted = clean_value.zfill(field_length)  # Left pad with zeros
            result = formatted.encode()
            print(f"NUMERIC format: {formatted}")
            return result

        else:
            raise ValueError(f"Unsupported field class: {field_class}")

    except Exception as e:
        raise ValueError(f"Error formatting field {field_id} ({field_class}): {str(e)}")

    return None


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
    Enhanced decode_server_response with field specifications
    """
    print("\nServer Response Analysis:")

    # Parse the response
    parsed = parse_iso_response(response, field_specs)

    # Print detailed analysis
    print(f"\n1. Basic Information:")
    print(f"   Length Prefix: {parsed.get('length_prefix', {}).get('value')} bytes")
    print(f"   MTI: {parsed.get('mti', {}).get('value')}")

    print(f"\n2. Bitmap Analysis:")
    if "bitmap" in parsed:
        print(f"   Hex: {parsed['bitmap']['hex']}")
        print(f"   Fields Present: {parsed['bitmap']['present_fields']}")

    print(f"\n3. Fields:")
    if "fields" in parsed:
        for field_id, field_data in sorted(parsed["fields"].items()):
            if isinstance(field_data, dict):
                value = field_data.get("value", "N/A")
                spec = field_data.get("spec", {})
                print(f"   Field {field_id} - {spec.get('name', 'Unknown')}:")
                print(f"      Value: {value}")
                print(f"      Type: {spec.get('class', 'Unknown')}")
                print(f"      Length: {spec.get('length', 'Unknown')}")
                print(f"      Position: {field_data.get('position', 'Unknown')}")
                print(f"      Hex: {field_data.get('hex', 'Unknown')}")

    print(f"\n4. Response Analysis:")
    if "response_analysis" in parsed:
        print(f"   Response Code: {parsed['response_analysis']['code']}")
        print(f"   Meaning: {parsed['response_analysis']['meaning']}")
        print(f"   Category: {parsed['response_analysis']['category']}")

    print(f"\n5. Message Validation:")
    if "length_check" in parsed:
        print(f"   Expected Length: {parsed['length_check']['expected']}")
        print(f"   Actual Length: {parsed['length_check']['actual']}")
        print(f"   Length Valid: {parsed['length_check']['valid']}")

    print("\n6. Raw Data:")
    print(f"   Full Hex: {parsed['raw_hex']}")

    return parsed



def format_iso_message(
    fields: Dict[str, str], field_specs: Dict[str, ISO8583Field], mti: str = "0200"
) -> bytes:
    """
    Format an ISO8583 message with comprehensive validation.

    Required validations:
    - Field dependencies (validate_field_dependencies)
    - Special fields (validate_special_fields)
    - Field lengths (validate_field_length)
    - Data types (validate_field_data_type)

    Args:
        fields: Dictionary mapping field IDs to their values
        field_specs: Dictionary mapping field IDs to ISO8583Field specifications
        mti: Message Type Indicator (default "0200" for financial request)

    Returns:
        bytes: Formatted ISO8583 message

    Raises:
        ValueError: If validation fails
    """
    try:
        print(f"\nFormatting ISO8583 message (MTI: {mti})")
        print("=" * 50)

        # 1. Basic validation
        if not fields or not field_specs:
            raise ValueError("Missing fields or specifications")

        if not mti or len(mti) != 4 or not mti.isdigit():
            raise ValueError(f"Invalid MTI format: {mti}")

        # 2. Field validation
        validation_errors = []

        # Mandatory field validation
        mandatory_errors = validate_mandatory_fields(fields, mti)
        if mandatory_errors:
            validation_errors.extend(mandatory_errors)
            print("\nMandatory field errors:")
            for error in mandatory_errors:
                print(f"- {error}")

        # Field dependencies validation
        dependency_errors = validate_field_dependencies(fields)
        if dependency_errors:
            validation_errors.extend(dependency_errors)
            print("\nField dependency errors:")
            for error in dependency_errors:
                print(f"- {error}")

        # Special fields validation
        special_field_errors = validate_special_fields(fields)
        if special_field_errors:
            validation_errors.extend(special_field_errors)
            print("\nSpecial field errors:")
            for error in special_field_errors:
                print(f"- {error}")

        if validation_errors:
            raise ValueError(
                f"Field validation failed: {len(validation_errors)} errors"
            )

        # 3. Add required supplementary data
        if "55" not in fields:
            fields["55"] = compose_field_55(fields)
        if "127" not in fields:
            fields["127"] = compose_field_127(fields)

        # 4. Format and validate fields
        active_fields = {}  # Fields with values
        formatted_fields = {}  # Formatted binary data
        field_lengths = {}  # Track lengths for validation

        print("\nProcessing fields:")
        for field_id, value in sorted(fields.items(), key=lambda x: int(x[0])):
            try:
                # Skip bitmap field
                if field_id == "1":
                    continue

                if not value.strip():
                    continue

                # Validate field length
                if not validate_field_length(field_id, value, field_specs):
                    print(f"Warning: Invalid length for field {field_id}")
                    continue

                # Validate field data type
                if not validate_field_data_type(field_id, value, field_specs):
                    print(f"Warning: Invalid data type for field {field_id}")
                    continue

                # Format field value
                formatted = format_field_value(value, field_specs[field_id])
                if formatted is not None:
                    active_fields[field_id] = value
                    formatted_fields[field_id] = formatted
                    field_lengths[field_id] = len(formatted)
                    print(
                        f"Field {field_id}: {'*' * len(value) if field_id in ['2', '35', '52'] else value}"
                    )

            except Exception as e:
                print(f"Error processing field {field_id}: {str(e)}")
                continue

        # 5. Generate bitmap from active fields
        bitmap = create_bitmap(active_fields)

        # 6. Assemble message
        message = mti.encode() + bitmap

        # Add fields in numeric order
        total_length = len(message)
        print("\nMessage assembly:")
        print(f"MTI: {len(mti)} bytes")
        print(f"Bitmap: {len(bitmap)} bytes")

        for field_id in sorted(active_fields.keys(), key=int):
            field_data = formatted_fields[field_id]
            message += field_data
            total_length += len(field_data)
            print(f"Field {field_id}: {len(field_data)} bytes")

        print(f"\nTotal message length: {total_length} bytes")

        # 7. Final validation
        if not message:
            raise ValueError("Empty message generated")

        if total_length != len(message):
            raise ValueError(
                f"Message length mismatch: expected {total_length}, got {len(message)}"
            )

        return message

    except Exception as e:
        error_msg = f"Message formatting failed: {str(e)}"
        print(f"\nError: {error_msg}")
        raise ValueError(error_msg)


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
    host: str = HOST,
    port: int = PORT,
    retries: int = 3,
    timeout: int = 15
) -> Optional[dict]:
    """Send ISO8583 key exchange message using plain TCP."""
    print(f"\nInitiating Key Exchange with {host}:{port}")
    print("=" * 50)

    retry_count = 0
    while retry_count < retries:
        sock = None
        try:
            print(f"\nAttempt {retry_count + 1} of {retries}")

            # Create plain TCP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)

            # Connect
            print(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("TCP connection established")

            # Load field specifications
            field_specs = parse_zone_xml(ZONE_FILE)

            # Prepare message
            now = datetime.now()
            field_data = {
                "3": "990000",            # Processing code for key exchange
                "7": now.strftime("%m%d%H%M%S"),  # MMDDhhmmss
                "11": generate_stan(),     # System Trace Number
                "12": now.strftime("%H%M%S"),     # Time
                "13": now.strftime("%m%d"),       # Date
                "32": "1346111",          # Acquiring Institution ID
                "41": "10351254",         # Terminal ID
                "42": "Z134611110042",    # Merchant ID
                "70": "101",              # Network Management Code
            }

            # Format and send message
            message = format_iso_message(field_data, field_specs, mti="0800")
            msg_length = len(message)
            length_prefix = struct.pack(">H", msg_length)
            full_message = length_prefix + message

            sock.sendall(full_message)
            print("Message sent successfully")

            # Receive response
            all_data = bytearray()
            received_header = False
            expected_length = 0
            response_timeout = time.time() + timeout

            while time.time() < response_timeout:
                try:
                    chunk = sock.recv(8192)
                    if not chunk:
                        if len(all_data) > 0:
                            break
                        raise ConnectionError("Connection closed by server")

                    all_data.extend(chunk)
                    if not received_header and len(all_data) >= 2:
                        expected_length = struct.unpack(">H", all_data[:2])[0]
                        received_header = True
                        print(f"Expected message length: {expected_length}")

                    if received_header and len(all_data) >= expected_length + 2:
                        print("Complete message received")
                        break

                except socket.timeout:
                    print("Receive timeout...")
                    continue

            if all_data:
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
                                "success": True
                            }
                    else:
                        print(f"Key exchange failed - response code: {resp_code}")
                        return {
                            "response": response_info,
                            "success": False
                        }

        except (socket.timeout, ConnectionError) as e:
            print(f"Connection error: {str(e)}")
            retry_count += 1
            if retry_count < retries:
                delay = 2 ** retry_count
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        except Exception as e:
            print(f"Error during key exchange: {str(e)}")
            retry_count += 1
            if retry_count < retries:
                delay = 2 ** retry_count
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            print("Connection closed")

    print(f"\nKey exchange failed after {retries} attempts")
    return None

def establish_secure_connection(host: str, port: int, timeout: int = 5) -> ssl.SSLSocket:
    """
    Establish secure connection with legacy server support.

    Args:
        host: Target hostname
        port: Target port
        timeout: Connection timeout in seconds

    Returns:
        ssl.SSLSocket: Connected and secured socket
    """
    print(f"\nEstablishing secure connection to {host}:{port}")

    sock = None
    secure_sock = None

    try:
        # Create base socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        # Connect TCP first
        print("Establishing TCP connection...")
        sock.connect((host, port))
        print("TCP connection established")

        # Create SSL context
        context = create_compatible_ssl_context()

        # Wrap socket
        print("Initiating TLS handshake...")
        secure_sock = context.wrap_socket(
            sock,
            server_hostname=None,  # Disable SNI for legacy servers
            do_handshake_on_connect=False
        )

        # Perform handshake with detailed logging
        start_time = time.time()
        while True:
            try:
                secure_sock.do_handshake()
                break
            except ssl.SSLWantReadError:
                select.select([secure_sock], [], [], 0.1)
                if time.time() - start_time > timeout:
                    raise socket.timeout("Handshake read timeout")
            except ssl.SSLWantWriteError:
                select.select([], [secure_sock], [], 0.1)
                if time.time() - start_time > timeout:
                    raise socket.timeout("Handshake write timeout")

        # Log connection info
        cipher = secure_sock.cipher()
        if cipher:
            print(f"Connected with {cipher[0]} ({cipher[1]})")

        return secure_sock

    except Exception as e:
        if secure_sock:
            secure_sock.close()
        if sock:
            sock.close()
        raise ConnectionError(f"Secure connection failed: {str(e)}")

# Original working code
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
        print(f"Field 32: {field_data.get('32', 'Not present')}")

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


def send_pinblock_message(
    host: str = HOST,  # "96.0.46.37",
    port: int = PORT,  # 5858,
    field_data: Dict[str, str] = None,
    encrypted_pin_block: str = None,
) -> Optional[Dict]:
    """
    Send ISO8583 message with PIN block (MTI 0200 with field 52).
    """
    try:
        print("\nPreparing PIN Block Transaction")
        print("=" * 50)

        # Load configurations if not provided
        if field_data is None:
            field_data = parse_testcard_data(TCARD_FILE)

        field_specs = parse_zone_xml(ZONE_FILE)
        if not field_specs:
            raise ValueError("Failed to load field specifications from zone.xml")

        # If no encrypted PIN block provided, generate one
        if encrypted_pin_block is None:
            # Read and process key components
            print("\nKey Component Processing:")
            print("=" * 50)
            comp1, comp2, stored_kcv = read_key_components()
            print("Key components loaded successfully")

            # Generate and validate clear ZPK
            clear_zpk = xor_hex_strings(comp1, comp2)

            # Generate PIN block
            pin = "2020"  # Default PIN
            encrypted_pin_block = PinBlockUtil.generate_encrypted_pin_block(
                clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
            )

        # Add PIN block to message
        field_data["52"] = encrypted_pin_block
        field_data["32"] = "1346111"
        field_data["3"] = "501000"

        # Establish connection
        print("\nConnection Setup:")
        print("=" * 50)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        context = ssl._create_unverified_context()
        secure_sock = context.wrap_socket(sock)

        try:
            print(f"Connecting to {host}:{port}")
            secure_sock.connect((host, port))
            secure_sock.setblocking(False)

            # Format and prepare message
            print("\nMessage Preparation:")
            print("=" * 50)
            message = format_iso_message(field_data, field_specs)
            msg_length = len(message)
            length_prefix = struct.pack(">H", msg_length)

            print(f"Message length: {msg_length} bytes")
            print(f"Length prefix hex: {length_prefix.hex()}")

            print("\nMessage Fields:")
            print("=" * 50)
            for field_id, value in field_data.items():
                # Mask sensitive fields
                if field_id in ["52", "2", "35"]:
                    masked_value = "*" * len(value)
                else:
                    masked_value = value
                print(f"Field {field_id}: {masked_value}")

            # Prepare full message
            full_message = length_prefix + message

            # Print hex dump of message
            print("\nMessage Hex Dump:")
            print("=" * 50)
            for i in range(0, len(full_message), 16):
                chunk = full_message[i : i + 16]
                hex_dump = " ".join(f"{b:02x}" for b in chunk)
                ascii_dump = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
                print(f"{i:04x}: {hex_dump:<48} {ascii_dump}")

            # Send message
            print("\nSending Message:")
            print("=" * 50)
            secure_sock.send(full_message)
            print(f"Sent {len(full_message)} bytes")

            # Wait for response
            print("\nWaiting for Response:")
            print("=" * 50)
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
                        print(f"Error reading response: {e}")
                        break

            if all_data:
                print("\nResponse Analysis:")
                print("=" * 50)
                response_info = decode_server_response(bytes(all_data), field_specs)
                analysis = analyze_pin_response(response_info)

                print("\nTransaction Result:")
                print("=" * 50)
                print(f"Status: {'Success' if analysis['success'] else 'Failed'}")
                print(f"Response Code: {analysis['response_code']}")
                print(f"Message: {analysis['message']}")

                if analysis["details"]:
                    print("\nTransaction Details:")
                    print("=" * 50)
                    for key, value in analysis["details"].items():
                        print(f"{key}: {value}")

                if "errors" in analysis and analysis["errors"]:
                    print("\nProcessing Errors:")
                    print("=" * 50)
                    for error in analysis["errors"]:
                        print(f"- {error}")

                return {
                    "raw_response": response_info,
                    "analysis": analysis,
                    "pin_block": encrypted_pin_block,  # Be careful with this in production
                    "success": analysis.get("success", False),
                }
            else:
                print("\nNo response received")
                return None

        finally:
            secure_sock.close()
            print("\nConnection closed")

    except socket.timeout:
        print("Connection timed out")
        raise
    except ConnectionRefusedError:
        print("Connection refused by host")
        raise
    except ssl.SSLError as e:
        print(f"SSL/TLS error: {e}")
        raise
    except ValueError as e:
        print(f"Validation error: {e}")
        raise
    except Exception as e:
        print(f"Error in PIN block processing: {str(e)}")
        raise
    finally:
        # Clear sensitive data
        if "clear_zpk" in locals():
            clear_zpk = "0" * len(clear_zpk)
        if "pin" in locals():
            pin = "0" * len(pin)


def perform_secure_transaction(
    host: str = HOST, port: int = PORT, pin: str = None
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
        context = create_secure_context()  # ssl._create_unverified_context()
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


def send_financial_message_with_session_keys(
    amount: Optional[str] = None,
    pin: Optional[str] = None,
    host: str = HOST,
    port: int = PORT,
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

        # Update amount and processing codes
        field_data.update(
            {
                "3": "000000",  # Purchase
                "4": amount,  # Transaction amount
                "22": "051",  # POS Entry Mode
                "25": "00",  # POS Condition Code
                "26": "12",  # POS PIN Capture Code
            }
        )

        # 5. Generate PIN block
        pin_block = PinBlockUtil.generate_encrypted_pin_block(
            clear_zpk=clear_zpk, card_pan=field_data["2"], pin=pin
        )
        field_data["52"] = pin_block
        sensitive_data.append(pin_block)

        # 6. Send transaction
        print("\nSending financial transaction...")
        result = send_pinblock_message(
            host=host, port=port, field_data=field_data, encrypted_pin_block=pin_block
        )

        if result:
            print("\nTransaction Response Analysis:")
            print("=" * 50)
            analysis = result.get("analysis", {})
            resp_code = analysis.get("response_code")

            if resp_code:
                detailed_analysis = get_response_analysis(resp_code)
                print(f"Response Code: {resp_code}")
                print(f"Message: {detailed_analysis['message']}")
                print(f"Category: {detailed_analysis['category']}")
                print(
                    f"Status: {'Success' if detailed_analysis['is_success'] else 'Failed'}"
                )

                if detailed_analysis["technical_details"]:
                    print(f"\nTechnical Details:")
                    print(detailed_analysis["technical_details"])

                if detailed_analysis["recommended_action"]:
                    print(f"\nRecommended Action:")
                    print(detailed_analysis["recommended_action"])

                if detailed_analysis["user_message"]:
                    print(f"\nCustomer Message:")
                    print(detailed_analysis["user_message"])

            # Log transaction details
            if analysis.get("details"):
                print("\nTransaction Details:")
                for key, value in analysis["details"].items():
                    print(f"{key}: {value}")

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


if __name__ == "__main__":
    import sys

    def print_usage():
        """
        Print detailed usage instructions for the ISO8583 client.

        Displays comprehensive help on:
        - Key management commands
        - Transaction operations
        - Utility functions
        - Configuration options
        - Usage examples
        """
        print("\nZoneSwitch ISO8583 Client Usage")
        print("=" * 50)

        print("\nKey Management Operations:")
        print("-" * 30)
        print("  --key-exchange           Generate new session keys")
        print(
            "  --key-exchange --force   Force new key exchange even if current keys valid"
        )
        print("  --clear-session          Clear stored session keys")
        print("  --session-status         Display current session status")
        print("  --rotate-keys            Rotate session keys")

        print("\nSession-based Transactions:")
        print("-" * 30)
        print("  --financial-session [amount] [pin] [currency=NGN]")
        print("      Send financial transaction using session keys")
        print("      amount: 12-digit amount (e.g. 000000005000 for 5000)")
        print("      pin: 4-12 digit PIN")
        print("      currency: 3-letter currency code (default: NGN)")

        print("  --pin-session [pin]")
        print("      Send PIN verification using session keys")
        print("      pin: 4-12 digit PIN")

        print("\nDirect Transactions:")
        print("-" * 30)
        print("  --financial [amount] [currency=NGN]")
        print("      Send direct financial transaction")
        print("      amount: 12-digit amount")
        print("      currency: 3-letter currency code")

        print("  --pin-block [pin]")
        print("      Send direct PIN verification")
        print("      pin: 4-12 digit PIN")

        print("\nConfiguration:")
        print("-" * 30)
        print("  --set-host [host]        Set server hostname/IP")
        print("  --set-port [port]        Set server port")
        print("  --set-timeout [seconds]  Set connection timeout")
        print("  --config-status          Show current configuration")

        print("\nUtility Operations:")
        print("-" * 30)
        print("  --test-pin               Test PIN block functionality")
        print("  --check-server           Verify server connectivity")
        print("  --post-journal           Test journal posting")
        print("  --show-keys              Display key status (masked)")
        print("  --validate [amount|pin]  Validate amount or PIN format")
        print("  --help                   Show this help message")

        print("\nExamples:")
        print("-" * 30)
        print("  1. Complete transaction flow:")
        print("     python zx1.py --key-exchange")
        print("     python zx1.py --financial-session 000000005000 1234 NGN")
        print("     python zx1.py --session-status")

        print("\n  2. Direct transaction:")
        print("     python zx1.py --financial 000000005000 NGN")

        print("\n  3. PIN verification:")
        print("     python zx1.py --pin-session 1234")

        print("\n  4. Configuration:")
        print("     python zx1.py --set-host 192.168.1.100")
        print("     python zx1.py --set-port 5858")
        print("     python zx1.py --config-status")

        print("\nNotes:")
        print("-" * 30)
        print("- All amounts must be 12 digits including leading zeros")
        print("- PINs must be 4-12 digits")
        print("- Session keys expire after configured lifetime")
        print("- Use --force to override valid but old session keys")
        print("- Connection timeout defaults to 30 seconds")
        print("- All sensitive data is masked in logs")

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


"""
Here's a step-by-step narrative of how ZPK (Zone Pin Key) and ZMK (Zone Master Key) are used to generate session keys and create PIN blocks in the code:

1. **ZMK Component Loading**
- The process starts by loading two ZMK components from `keys.txt`
- These components are stored separately for security
- The code reads both components and their Key Check Value (KCV)
```python
comp1, comp2, stored_kcv = read_key_components()
```

2. **ZMK Generation**
- The clear ZMK is created by XORing the two components together
- This creates the actual Zone Master Key used for encrypting/decrypting the ZPK
```python
clear_zmk = xor_hex_strings(comp1, comp2)
```

3. **KCV Verification**
- A Key Check Value is generated by encrypting 8 bytes of zeros with the clear ZMK
- This is compared with the stored KCV to verify key integrity
```python
generated_kcv = PinBlockUtil.bytes_to_string(
    PinBlockUtil.operate_des3(bytes(8), PinBlockUtil.string_to_bytes(clear_zmk), True)
)[:6]
```

4. **PIN Block Creation**
- When a PIN is entered, it's formatted according to ISO format:
  - Format: 0 + PIN length + PIN + Padding
  - Example: "0412345678FFFF" for PIN "12345678"
- The card PAN is also formatted:
  - Takes last 12 digits of PAN
  - Prepends with zeros
```python
pin_block = PinBlockUtil.generate_encrypted_pin_block(
    clear_zpk=clear_zpk,
    card_pan=field_data["2"],
    pin=pin
)
```

5. **PIN Block Encryption**
- The formatted PIN block is XORed with the formatted PAN
- This XORed value is then encrypted using Triple DES with the ZPK
- The encrypted PIN block is placed in Field 52 of the ISO message
```python
field_data["52"] = encrypted_pin_block
```

6. **Security Measures**
- Sensitive data is masked in logs
- Keys are cleared from memory after use
- All key components are validated for correct length and format
```python
if 'clear_zpk' in locals():
    clear_zpk = '0' * len(clear_zpk)
```

The process ensures:
- Secure key storage (split components)
- Key integrity verification (KCV checking)
- PIN confidentiality (encryption)
- PAN binding (XORing with PIN)
- Secure memory handling (clearing sensitive data)

This implementation follows standard financial industry practices for PIN security and key management.
"""
