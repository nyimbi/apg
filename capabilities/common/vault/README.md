# APG Vault (PCI DSS Tokenization) (`vault`)

**Version**: 1.0.0 | **Domain**: common

## Overview

PCI DSS format-preserving tokenization for cardholder PANs. Tokens preserve BIN and last 4 digits; Luhn-valid. OPA-gated detokenization.

## Usage

```python
from apg_common_vault import *
```

## Governance Rules

- tenant_context_required
- operation_type_required  
- audit_logged
- access_controlled

## License

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke
