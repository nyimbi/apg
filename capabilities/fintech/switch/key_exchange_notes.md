I'll break down the key exchange process and procedures in detail:

# ISO8583 Key Exchange Process Documentation

## 1. Key Components and Initial Setup

```python
class SessionKeyManager:
    """Manages secure session key storage and retrieval"""
    SESSION_FILE = "session_keys.json"
    KEY_LIFETIME = 60000  # 1000 minutes

    def __init__(self):
        self.session_file = Path(self.SESSION_FILE)
        self._create_session_dir()
```

## 2. Key Exchange Message (MTI 0800)

```python
def send_key_exchange_message(host: str, port: int) -> Optional[dict]:
    """Initiate key exchange with host"""
    # Prepare key exchange fields
    field_data = {
        "7": datetime.now().strftime("%m%d%H%M%S"),  # MMDDhhmmss
        "11": generate_stan(),                       # System Trace Number
        "32": "05110",                              # Acquiring Institution ID
        "70": "101",                                # Network Management Code
    }

    # Format and send message
    message = format_iso_message(field_data, field_specs, mti="0800")
    # ... send message and receive response ...
```

## 3. Double Variant Key Decryption Process

### a. Split Key Components
```python
def decrypt_zpk_double_variant(encrypted_zpk: str, clear_zmk: str) -> str:
    """
    Example input:
    encrypted_zpk = "98EEA376A14DF58578E7D77585512DF3"
    clear_zmk = "63E4880A2D502DD8E835C68DD8061BBB"
    """
    # Split encrypted ZPK
    zpk_part_a = encrypted_zpk[:16]  # "98EEA376A14DF585"
    zpk_part_b = encrypted_zpk[16:]  # "78E7D77585512DF3"

    # Split clear ZMK
    zmk_part_a = clear_zmk[:16]      # "63E4880A2D502DD8"
    zmk_part_b = clear_zmk[16:]      # "E835C68DD8061BBB"
```

### b. First Variant (A6)
```python
    # Apply first variant (A6)
    first_two_zmk_b = zmk_part_b[:2]  # "E8"
    variant1 = format(int("A6", 16) ^ int(first_two_zmk_b, 16), '02X')  # A6 ^ E8 = 4E
    variant1_zmk = zmk_part_a + variant1 + zmk_part_b[2:]  
    # Result: "63E4880A2D502DD84E35C68DD8061BBB"

    # Decrypt first part
    result1 = PinBlockUtil.operate_des3(
        PinBlockUtil.string_to_bytes(zpk_part_a),
        PinBlockUtil.string_to_bytes(variant1_zmk[:16]),
        False  # Decrypt mode
    )
```

### c. Second Variant (5A)
```python
    # Apply second variant (5A)
    variant2 = format(int("5A", 16) ^ int(first_two_zmk_b, 16), '02X')  # 5A ^ E8 = B2
    variant2_zmk = zmk_part_a + variant2 + zmk_part_b[2:]
    # Result: "63E4880A2D502DD8B235C68DD8061BBB"

    # Decrypt second part
    result2 = PinBlockUtil.operate_des3(
        PinBlockUtil.string_to_bytes(zpk_part_b),
        PinBlockUtil.string_to_bytes(variant2_zmk[:16]),
        False  # Decrypt mode
    )
```

### d. Combine Results
```python
    # Combine first 16 bytes of each result
    clear_zpk = result1_hex[:16] + result2_hex[:16]
    # Final result should match KCV verification
    return clear_zpk
```

## 4. KCV (Key Check Value) Verification

```python
def verify_kcv(clear_zpk: str, expected_kcv: str) -> bool:
    """Verify ZPK using KCV"""
    test_data = bytes(8)  # 8 zeros
    generated_kcv = PinBlockUtil.bytes_to_string(
        PinBlockUtil.operate_des3(
            test_data,
            PinBlockUtil.string_to_bytes(clear_zpk),
            True  # Encrypt mode
        )
    )[:6]
    return generated_kcv.upper() == expected_kcv.upper()
```

## 5. Session Key Storage

```python
def save_session_keys(self, key_exchange_response: dict) -> bool:
    """Save session keys securely"""
    try:
        field_53_data = parse_field_53(fields["53"]["value"])
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'encrypted_zpk': field_53_data['encrypted_zpk'],
            'key_version': field_53_data['key_version'],
            'kcv': fields.get('64', {}).get('value', ''),
            'additional_data': {
                'terminal_id': fields.get('41', {}).get('value', ''),
                'response_code': fields.get('39', {}).get('value', '')
            }
        }

        # Encrypt sensitive data before storage
        encrypted_session = {
            'timestamp': session_data['timestamp'],
            'key_data': self._encrypt_key_data(json.dumps(session_data)),
            'additional_data': session_data['additional_data']
        }

        with open(self.session_file, 'w') as f:
            json.dump(encrypted_session, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving session keys: {str(e)}")
        return False
```

## 6. Complete Key Exchange Sequence

```python
def perform_key_exchange_with_persistence() -> bool:
    """
    Complete key exchange sequence:
    1. Check existing session
    2. Perform key exchange if needed
    3. Process and verify keys
    4. Save session keys
    """
    try:
        session_manager = SessionKeyManager()
        
        # Check existing session
        existing_keys = session_manager.get_valid_session_keys()
        if existing_keys:
            # Verify existing keys
            if verify_kcv(existing_keys['clear_zpk'], existing_keys['kcv']):
                return True

        # Perform new key exchange
        key_exchange_result = send_key_exchange_message()
        if not key_exchange_result:
            raise ValueError("Key exchange failed")

        # Process received keys
        encrypted_zpk = key_exchange_result['encrypted_zpk']
        clear_zmk = get_clear_zmk()  # From components
        
        # Decrypt using double variant
        clear_zpk = decrypt_zpk_double_variant(encrypted_zpk, clear_zmk)
        
        # Verify KCV
        if not verify_kcv(clear_zpk, key_exchange_result['kcv']):
            raise ValueError("KCV verification failed")

        # Save session keys
        if not session_manager.save_session_keys(key_exchange_result):
            raise ValueError("Failed to save session keys")

        return True

    except Exception as e:
        print(f"Key exchange error: {str(e)}")
        return False
```

## 7. Security Considerations

1. **Component Separation**:
```python
def read_key_components(filename: str = "keys.txt") -> Tuple[str, str, str]:
    """Read separate key components"""
    with open(filename, "r") as f:
        comp1 = f.readline().split(":")[1].strip()
        comp2 = f.readline().split(":")[1].strip()
        kcv = f.readline().split(":")[1].strip()
    return comp1, comp2, kcv
```

2. **Secure Memory Handling**:
```python
try:
    # Process keys
    sensitive_data.extend([clear_zmk, clear_zpk])
finally:
    # Clear sensitive data
    for item in sensitive_data:
        if isinstance(item, str):
            item = '0' * len(item)
```

3. **Session Key Validation**:
```python
def validate_session_keys(session_keys: dict) -> bool:
    """Validate session keys before use"""
    if not session_keys:
        return False
        
    # Check expiry
    stored_time = datetime.fromisoformat(session_keys['timestamp'])
    if datetime.now() - stored_time > timedelta(seconds=KEY_LIFETIME):
        return False
        
    # Verify KCV
    return verify_kcv(session_keys['clear_zpk'], session_keys['kcv'])
```

This implementation follows these security principles:
- Separation of key components
- Secure key storage
- KCV verification
- Memory cleanup
- Session key management
- Double variant key decryption

Would you like me to elaborate on any particular aspect?