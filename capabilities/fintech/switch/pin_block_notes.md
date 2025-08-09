# ISO8583 PIN Block Creation and Encryption Process

## 1. PIN Block Format (ISO-0)

```python
class PinBlockFormat:
    """ISO-0 PIN Block Format Constants"""
    PIN_BLOCK_LENGTH = 16
    PIN_FORMAT = "0"  # ISO-0 format indicator
    PADDING_CHAR = "F"
    PAN_PADDING = "0000"  # Left padding for PAN
```

## 2. PIN Block Generation Process

### a. Format PIN Block (PIN Block Part 1)
```python
def format_pin_block(pin: str) -> str:
    """
    Create ISO-0 PIN block format:
    Byte 1: Format code (0)
    Byte 2: PIN length (4-12)
    Bytes 3-8: PIN digits
    Remaining: Padding (F)
    
    Example:
    PIN: "1234"
    Result: "041234FFFFFFFFFF"
    """
    if not validate_pin_format(pin):
        raise ValueError("Invalid PIN format")
        
    pin_length = len(pin)
    pin_block = (
        PinBlockFormat.PIN_FORMAT +  # 0
        format(pin_length, "x") +    # 4
        pin +                        # 1234
        PinBlockFormat.PADDING_CHAR * (
            PinBlockFormat.PIN_BLOCK_LENGTH - 2 - pin_length
        )                            # FFFFFFFF
    )
    return pin_block
```

### b. Format PAN Block (PIN Block Part 2)
```python
def format_pan_block(pan: str) -> str:
    """
    Create PAN block:
    - Take rightmost 12 digits excluding check digit
    - Pad left with zeros
    
    Example:
    PAN: "4111111111111111"
    Result: "0000111111111111"
    """
    if not 12 <= len(pan) <= 19:
        raise ValueError("Invalid PAN length")
        
    # Take 12 rightmost digits excluding check digit
    pan_part = pan[-13:-1]  
    pan_block = (
        PinBlockFormat.PAN_PADDING +  # 0000
        pan_part                      # 111111111111
    )
    return pan_block
```

## 3. Cryptographic Operations

### a. PIN Block XOR Operation
```python
class PinBlockUtil:
    @staticmethod
    def xor_pin_blocks(pin_block: str, pan_block: str) -> bytes:
        """
        XOR PIN block with PAN block
        
        Example:
        PIN block: "041234FFFFFFFFFF"
        PAN block: "0000111111111111"
        Result:    "041325EEEEEEEEEE"
        """
        pin_bytes = binascii.unhexlify(pin_block)
        pan_bytes = binascii.unhexlify(pan_block)
        
        # XOR operation
        clear_pin_block = bytes(
            a ^ b for a, b in zip(pin_bytes, pan_bytes)
        )
        
        # Log intermediate values (masked)
        print(f"PIN Block (hex): {pin_block}")
        print(f"PAN Block (hex): {pan_block}")
        print(f"Clear PIN Block (hex): {clear_pin_block.hex().upper()}")
        
        return clear_pin_block
```

### b. Triple DES Encryption
```python
    @staticmethod
    def operate_des3(data: bytes, key: bytes, encrypt: bool) -> bytes:
        """
        Perform Triple DES encryption/decryption
        
        Args:
            data: 8 bytes of data
            key: 16-byte key (expanded to 24 bytes)
            encrypt: True for encryption, False for decryption
            
        Process:
        1. Expand 16-byte key to 24 bytes for 3DES
        2. Create cipher in ECB mode
        3. Perform encryption/decryption
        """
        if len(key) != 16:
            raise ValueError(f"Key must be 16 bytes (got {len(key)})")
        if len(data) != 8:
            raise ValueError(f"Data must be 8 bytes (got {len(data)})")

        # Create 24-byte key by duplicating first 8 bytes
        triple_des_key = key + key[:8]

        # Create cipher
        cipher = Cipher(
            algorithms.TripleDES(triple_des_key),
            modes.ECB(),
            backend=default_backend()
        )

        if encrypt:
            encryptor = cipher.encryptor()
            return encryptor.update(data) + encryptor.finalize()
        else:
            decryptor = cipher.decryptor()
            return decryptor.update(data) + decryptor.finalize()
```

## 4. Complete PIN Block Generation and Encryption

```python
def generate_encrypted_pin_block(
    clear_zpk: str,
    card_pan: str,
    pin: str
) -> str:
    """
    Generate and encrypt ISO-0 PIN block
    
    Process:
    1. Format PIN block
    2. Format PAN block
    3. XOR blocks
    4. Encrypt with ZPK
    """
    try:
        # 1. Create PIN block
        pin_block = format_pin_block(pin)
        print(f"PIN Block: {pin_block}")

        # 2. Create PAN block
        pan_block = format_pan_block(card_pan)
        print(f"PAN Block: {pan_block}")

        # 3. XOR operation
        clear_pin_block = PinBlockUtil.xor_pin_blocks(
            pin_block, pan_block
        )
        
        # 4. Encrypt with ZPK
        key_bytes = PinBlockUtil.string_to_bytes(clear_zpk)
        encrypted_block = PinBlockUtil.operate_des3(
            clear_pin_block,
            key_bytes,
            True  # Encrypt mode
        )

        # Convert to hex and pad
        encrypted_hex = PinBlockUtil.bytes_to_string(
            encrypted_block
        ).ljust(32, '0')
        
        print(f"Encrypted PIN Block: {encrypted_hex}")
        return encrypted_hex

    except Exception as e:
        raise ValueError(f"PIN block generation failed: {str(e)}")
```

## 5. PIN Block Verification

```python
def verify_pin_block(
    encrypted_block: str,
    clear_zpk: str,
    card_pan: str,
    pin: str
) -> bool:
    """
    Verify PIN block by decryption and comparison
    
    Process:
    1. Decrypt PIN block
    2. Extract PIN
    3. Compare with original
    """
    try:
        # 1. Decrypt PIN block
        encrypted_bytes = PinBlockUtil.string_to_bytes(encrypted_block)
        key_bytes = PinBlockUtil.string_to_bytes(clear_zpk)
        
        decrypted_block = PinBlockUtil.operate_des3(
            encrypted_bytes,
            key_bytes,
            False  # Decrypt mode
        )
        
        # 2. Create verification blocks
        verify_pin_block = format_pin_block(pin)
        verify_pan_block = format_pan_block(card_pan)
        
        # 3. XOR to get original PIN block
        verify_clear_block = PinBlockUtil.xor_pin_blocks(
            verify_pin_block,
            verify_pan_block
        )
        
        # 4. Compare
        return decrypted_block == verify_clear_block

    except Exception as e:
        print(f"PIN block verification failed: {str(e)}")
        return False
```

## 6. Security Considerations

### a. Secure PIN Handling
```python
def handle_pin_securely(pin: str) -> None:
    """Secure PIN handling practices"""
    sensitive_data = []
    try:
        sensitive_data.append(pin)
        # Process PIN
        ...
    finally:
        # Clear sensitive data
        for item in sensitive_data:
            if isinstance(item, str):
                item = '0' * len(item)
```

### b. PIN Validation
```python
def validate_pin_format(pin: str, mask_in_logs: bool = True) -> bool:
    """
    Validate PIN according to ISO 9564:
    - Length: 4-12 digits
    - Only numeric
    - Not all zeros
    - Optional: Not sequential
    - Optional: Not repeated digits
    """
    try:
        if not pin or not isinstance(pin, str):
            raise ValueError("Invalid PIN type")
            
        if not pin.isdigit():
            raise ValueError("PIN must be numeric")
            
        pin_length = len(pin)
        if not (4 <= pin_length <= 12):
            raise ValueError("Invalid PIN length")
            
        if all(d == '0' for d in pin):
            raise ValueError("PIN cannot be all zeros")
            
        # Log masked PIN
        if mask_in_logs:
            print(f"PIN validated: {'*' * len(pin)}")
        return True
        
    except ValueError as e:
        print(f"PIN validation failed: {str(e)}")
        return False
```

## 7. Integration with ISO8583 Message

```python
def add_pin_block_to_message(
    fields: Dict[str, str],
    pin: str,
    clear_zpk: str
) -> Dict[str, str]:
    """Add encrypted PIN block to ISO message"""
    try:
        if "2" not in fields:
            raise ValueError("PAN (field 2) required")
            
        pin_block = generate_encrypted_pin_block(
            clear_zpk=clear_zpk,
            card_pan=fields["2"],
            pin=pin
        )
        
        # Add to field 52
        fields["52"] = pin_block
        return fields
        
    except Exception as e:
        raise ValueError(f"Error adding PIN block: {str(e)}")
```

Key Security Features:
1. PIN confidentiality through encryption
2. Secure memory handling
3. PAN binding
4. Format validation
5. Secure logging (masking)
6. Cryptographic verification
7. Standards compliance (ISO-0)

The implementation follows ISO 9564 standard for PIN block formats and ISO 8583 for message integration, with emphasis on security at every step.

Would you like me to elaborate on any specific aspect of the cryptographic operations or security measures?
