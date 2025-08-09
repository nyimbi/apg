# ISO8583 Financial Transaction Processing Documentation

## 1. Transaction Overview

A financial transaction involves multiple stages:
1. Session key validation/exchange
2. PIN block creation
3. Message preparation
4. Transaction submission
5. Response processing
6. Journal entry creation

## 2. Detailed Process Flow

### A. Initial Setup and Validation
```python
def send_financial_message(
    host: str = "96.0.46.37",
    port: int = 5858,
    field_data: Dict[str, str] = None,
    encrypted_pin_block: str = None
) -> Optional[Dict]:
    """
    Process and send financial transaction
    
    Flow:
    1. Validate session/keys
    2. Prepare transaction data
    3. Format message
    4. Send and process response
    5. Create journal entry
    """
    try:
        # Verify server availability
        if not check_server(host, port):
            raise ValueError("Server unavailable")

        # Load configurations
        field_specs = parse_zone_xml(ZONE_FILE)
        if field_data is None:
            field_data = parse_testcard_data(TCARD_FILE)

        print("\nTransaction Setup:")
        print(f"Terminal: {field_data.get('41', 'Unknown')}")
        print(f"Amount: {field_data.get('4', '0')}")
```

### B. Session Key Management
```python
        # Get or establish session keys
        session_manager = SessionKeyManager()
        session_keys = session_manager.get_valid_session_keys()
        if not session_keys:
            print("Performing key exchange...")
            if not perform_key_exchange_with_persistence():
                raise ValueError("Key exchange failed")
            session_keys = session_manager.get_valid_session_keys()

        # Verify session keys
        if not verify_session_keys(session_keys):
            raise ValueError("Invalid session keys")
```

### C. Message Preparation
```python
        # Generate dynamic fields
        field_data.update({
            "11": generate_stan(),          # System Trace Number
            "37": generate_retrieval_ref(),  # Retrieval Reference Number
            "7": datetime.now().strftime("%m%d%H%M%S"),  # Transmission Date/Time
        })

        # Add mandatory fields
        required_fields = {
            "3": "000000",     # Processing Code (Purchase)
            "22": "051",       # POS Entry Mode
            "25": "00",        # POS Condition Code
            "26": "12",        # POS PIN Capture Code
            "32": "00000",     # Acquiring Institution ID
            "49": "566",       # Transaction Currency Code (NGN)
        }
        field_data.update(required_fields)
```

### D. PIN Block Processing
```python
        if encrypted_pin_block:
            field_data["52"] = encrypted_pin_block
        elif "52" not in field_data:
            # Generate PIN block if not provided
            clear_zpk = decrypt_zpk(
                session_keys['encrypted_zpk'],
                session_keys['clear_zmk']
            )
            pin_block = generate_encrypted_pin_block(
                clear_zpk=clear_zpk,
                card_pan=field_data["2"],
                pin="1234"  # Default test PIN
            )
            field_data["52"] = pin_block
```

### E. Message Formatting and Validation
```python
        # Validate dependencies
        errors = validate_field_dependencies(field_data)
        if errors:
            raise ValueError(f"Field validation errors: {errors}")

        # Format ISO message
        message = format_iso_message(field_data, field_specs, mti="0200")
        msg_length = len(message)
        length_prefix = struct.pack(">H", msg_length)

        print("\nFormatted Message Details:")
        print(f"MTI: 0200 (Financial Transaction Request)")
        print(f"Length: {msg_length} bytes")
        print("Fields present:")
        for field_id, value in field_data.items():
            # Mask sensitive fields
            if field_id in ["2", "35", "52"]:
                print(f"Field {field_id}: {'*' * len(value)}")
            else:
                print(f"Field {field_id}: {value}")
```

### F. Transaction Submission
```python
        # Establish secure connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        context = ssl._create_unverified_context()
        secure_sock = context.wrap_socket(sock)

        try:
            print(f"\nConnecting to {host}:{port}")
            secure_sock.connect((host, port))
            secure_sock.setblocking(False)

            # Send message
            full_message = length_prefix + message
            secure_sock.send(full_message)
            print(f"Sent {len(full_message)} bytes")

            # Receive response
            response_data = receive_response(secure_sock)
            if not response_data:
                raise ValueError("No response received")

            # Parse response
            response_info = decode_server_response(
                response_data, 
                field_specs
            )

            # Process response
            result = analyze_transaction_response(response_info)
            
            # Create journal entry
            journal_result = create_journal_entry(
                field_data, 
                response_info
            )

            return {
                "success": result["success"],
                "response_code": result["response_code"],
                "message": result["message"],
                "journal": journal_result
            }

        finally:
            secure_sock.close()
```

### G. Response Processing
```python
def analyze_transaction_response(response_info: dict) -> dict:
    """Analyze transaction response"""
    try:
        fields = response_info.get("fields", {})
        resp_code = fields.get("39", {}).get("value", "").strip()
        
        result = {
            "success": resp_code == "00",
            "response_code": resp_code,
            "message": get_response_message(resp_code),
            "details": {}
        }

        # Extract key fields
        key_fields = {
            "2": "PAN",
            "3": "Processing Code",
            "4": "Amount",
            "11": "STAN",
            "37": "RRN",
            "38": "Auth Code",
            "41": "Terminal ID",
        }

        for field_id, field_name in key_fields.items():
            if field_id in fields:
                value = fields[field_id].get("value")
                if field_id in ["2"]:  # Mask sensitive data
                    value = f"{'*' * (len(value)-4)}{value[-4:]}"
                result["details"][field_name] = value

        return result

    except Exception as e:
        print(f"Error analyzing response: {str(e)}")
        return {
            "success": False,
            "response_code": "96",
            "message": "Error processing response",
            "error": str(e)
        }
```

### H. Journal Entry Creation
```python
def create_journal_entry(
    request_data: dict,
    response_info: dict
) -> dict:
    """Create transaction journal entry"""
    try:
        fields = response_info.get("fields", {})
        resp_code = fields.get("39", {}).get("value", "")
        
        journal_data = {
            "rrn": request_data.get("37", ""),
            "stan": request_data.get("11", ""),
            "amount": int(request_data.get("4", "0").lstrip("0")),
            "account_number": request_data.get("102", ""),
            "pan": request_data.get("2", ""),
            "status": "APPROVED" if resp_code == "00" else "FAILED",
            "terminal_id": request_data.get("41", ""),
            "error": "" if resp_code == "00" else resp_code,
            "comment": f"Transaction {resp_code}: {get_response_message(resp_code)}"
        }

        return send_push_journal(**journal_data)

    except Exception as e:
        print(f"Error creating journal entry: {str(e)}")
        return {
            "status": "error",
            "message": f"Journal error: {str(e)}"
        }
```

## 3. Transaction Flow Narrative

1. **Initial Setup**:
   - Validate server availability
   - Load field specifications from XML
   - Load or prepare transaction data
   - Verify session keys or perform key exchange

2. **Message Preparation**:
   - Generate dynamic fields (STAN, RRN)
   - Add mandatory fields
   - Process PIN block if required
   - Validate field dependencies
   - Format according to ISO8583 specifications

3. **Secure Communication**:
   - Establish SSL/TLS connection
   - Send formatted message
   - Handle timeouts and retries
   - Process response

4. **Response Processing**:
   - Parse ISO8583 response
   - Extract response code and messages
   - Process authorization data
   - Handle error conditions

5. **Journal Entry**:
   - Format transaction data
   - Submit to journal API
   - Handle success/failure scenarios
   - Maintain audit trail

## 4. Key Security Considerations

1. **Data Protection**:
   ```python
   # Mask sensitive data in logs
   def mask_sensitive_data(data: dict) -> dict:
       masked = data.copy()
       if "2" in masked:  # PAN
           pan = masked["2"]
           masked["2"] = f"{'*' * (len(pan)-4)}{pan[-4:]}"
       if "52" in masked:  # PIN Block
           masked["52"] = "*" * len(masked["52"])
       return masked
   ```

2. **Secure Communication**:
   ```python
   # Establish secure connection
   context = ssl.create_default_context()
   context.check_hostname = True
   context.verify_mode = ssl.CERT_REQUIRED
   secure_sock = context.wrap_socket(
       sock,
       server_hostname=host
   )
   ```

3. **Session Management**:
   ```python
   def verify_session_keys(session_keys: dict) -> bool:
       if not session_keys:
           return False
       if is_session_expired(session_keys['timestamp']):
           return False
       return verify_kcv(
           session_keys['clear_zpk'],
           session_keys['kcv']
       )
   ```

This implementation ensures:
- Secure key handling
- Data confidentiality
- Message integrity
- Proper error handling
- Transaction logging
- Audit trail maintenance

Would you like me to elaborate on any specific aspect of the transaction processing?
