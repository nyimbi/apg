# Enums & Validation

## Enums

### Named enums

```apg
enum OrderStatus {
    Draft;
    Confirmed;
    Shipped;
    Delivered;
    Cancelled;
}

table Order {
    status: OrderStatus = Draft;
}
```

The generated API rejects any `status` value not in the enum with `422 Unprocessable Entity`.

### Inline string enums

```apg
table Ticket {
    priority: str = "normal" [values=["low", "normal", "high", "critical"]];
}
```

### Referencing enums across files

```apg
// shared.apg
enum Currency { KES; USD; EUR; GBP; }

// orders.apg
table Invoice {
    currency: Currency = KES;
}
```

## Field validation rules

Validation rules are declared in square brackets after the field type:

```apg
table User {
    username: str [min_length=3, max_length=32];
    email:    str [email];
    age:      int [min=0, max=150];
    website:  str? [pattern="^https?://"];
    score:    float [min=0.0, max=100.0];
}
```

### Available validators

| Validator | Applies to | Description |
|-----------|-----------|-------------|
| `email` | `str` | Validates RFC 5321 email format |
| `min_length=N` | `str` | Minimum character count |
| `max_length=N` | `str` | Maximum character count |
| `min=N` | `int`, `float`, `decimal` | Minimum numeric value |
| `max=N` | `int`, `float`, `decimal` | Maximum numeric value |
| `pattern="re"` | `str` | Python regex that the value must match |
| `optional` | any | Explicit optionality marker (same as `?` suffix) |
| `unique` | any | Adds a uniqueness check at the app layer |
| `values=[...]` | `str` | Restrict to a fixed list of allowed strings |

### Combining validators

```apg
table Product {
    sku:       str [min_length=4, max_length=20, pattern="^[A-Z0-9-]+$"];
    price:     decimal [min=0.01, max=999999.99];
    email:     str [email, max_length=255];
    category:  str [values=["electronics", "clothing", "food"]];
}
```

## Validation error response

When validation fails the generated app returns:

```json
{
  "error": "validation_error",
  "field": "email",
  "message": "Invalid email format",
  "status": 422
}
```

## Enum as status-machine seed

Enums integrate naturally with state-machine patterns:

```apg
enum JobState { Queued; Running; Done; Failed; }

table Job {
    state:      JobState = Queued;
    started_at: datetime?;
    ended_at:   datetime?;
    result:     str | None;
}
```

The REST API enforces allowed transitions when transition guards are declared in a `flow` entity (see the Language Reference).
