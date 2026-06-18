# Africa Mobile Money Connectors

Production-ready connectors for Africa's major mobile money providers.

| Connector | Provider | Markets | Key Operations |
|-----------|----------|---------|----------------|
| `AirtelConnector` | Airtel Money | KE, UG, TZ, RW, ZM | C2B, B2C, balance, status |
| `MShwariConnector` | M-Shwari (CBA/Safaricom) | KE | lock savings, loan apply/repay |
| `MTNConnector` | MTN MoMo | NG, GH, UG, CM, CI, ZM | send, request, balance, status |
| `OrangeConnector` | Orange Money | CI, SN, CM, ML, BF | send, request, balance |
| `WaveConnector` | Wave | SN, CI, ML, BF, GN | send, request, balance |

All extend `BaseConnector`, use circuit breakers, and register with `CONNECTORS_MANIFEST`.
