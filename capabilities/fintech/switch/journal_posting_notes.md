Journal Posting Requirements:

1. Transaction Status:
   - Use "APPROVED" for successful transactions
   - Use "FAILED" for unsuccessful transactions (not "DECLINED")

2. Error Field Rules:
   - Must be empty string ("") for APPROVED transactions
   - Should contain error code for FAILED transactions

3. Required Fields Format:
   - Rrn: 12-digit zero-padded string
   - Stan: System Trace Audit Number
   - AcquirerBank: "107"
   - Amount: Integer value
   - AccountNumber: Account number (can be empty)
   - Pan: Full PAN number (masked in logs)
   - TransactionStatus: "APPROVED" or "FAILED"
   - CurrencyCode: "566"
   - TransactionDate: DD/MM/YYYY format
   - TransactionTime: HH:MM format
   - TerminalId: Terminal identifier

4. API Endpoint:
   URL: https://stagingnode.zonenetwork.com/pushjournal/api/push-journal
   Method: POST
   Headers: 
   - accept: application/json
   - content-type: application/json
   - x-api-key: [API_KEY]

Example Successful Payloads:

1. Approved Transaction:
{
    "Rrn": "000000938922",
    "Stan": "991545",
    "AcquirerBank": "107",
    "Amount": 3000,
    "AccountNumber": "",
    "Pan": "5559405048128222",
    "TransactionStatus": "APPROVED",
    "CurrencyCode": "566",
    "Comment": "Test transaction - Approved",
    "TransactionDate": "29/11/2024",
    "TransactionTime": "15:27",
    "Error": "",
    "TerminalId": "10351254"
}

2. Failed Transaction:
{
    "Rrn": "000000818505",
    "Stan": "613645",
    "AcquirerBank": "107",
    "Amount": 3000,
    "AccountNumber": "",
    "Pan": "5559405048128222",
    "TransactionStatus": "FAILED",
    "CurrencyCode": "566",
    "Comment": "Test transaction - Failed",
    "TransactionDate": "29/11/2024",
    "TransactionTime": "15:27",
    "Error": "51",
    "TerminalId": "10351254"
}

Response Format:
{
    "status": true,
    "message": "done"
}
"""