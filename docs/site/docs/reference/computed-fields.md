# Computed Fields

Computed fields are derived from other fields in the same entity. They are re-evaluated on every read and cannot be set directly through the API.

## Syntax

Assign an expression to a field using `=`:

```apg
table Order {
    subtotal:   decimal = 0.0;
    tax_rate:   float = 0.16;
    tax_amount: decimal = subtotal * tax_rate;
    total:      decimal = subtotal + tax_amount;
}
```

The compiler detects that `tax_amount` and `total` depend on other fields and marks them as computed.

## Expression support

Computed field expressions support:

- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`
- Null coalescing: `?? default_value`
- Conditional: `x if condition else y`
- Field references: any field name in the same entity
- Literal values: numbers, strings, booleans

## Examples

### Price with discount

```apg
table Product {
    list_price:     decimal;
    discount_pct:   float = 0.0;
    sale_price:     decimal = list_price * (1 - discount_pct);
}
```

### Full name

```apg
table Person {
    first_name: str;
    last_name:  str;
    full_name:  str = first_name + " " + last_name;
}
```

### Null-safe default

```apg
table Profile {
    display_name: str?;
    username:     str;
    label:        str = display_name ?? username;
}
```

### Boolean flag from numeric

```apg
table Account {
    balance:     decimal = 0.0;
    is_overdrawn: bool = balance < 0;
}
```

## Job queue computed fields

For long-running computations, declare a `computed` field backed by a job:

```apg
table Report {
    raw_data:  json;
    summary:   str = compute_summary(raw_data);   // dispatched to job queue
}
```

The field value is `null` until the job completes, then populated by the in-process job queue (Wave R).

## API behaviour

- `GET /entities/Order/records` — `tax_amount` and `total` appear in every response.
- `POST /entities/Order/records` — `tax_amount` and `total` are ignored in the request body.
- `PATCH /entities/Order/records/<id>` — updating `subtotal` causes `tax_amount` and `total` to be recalculated automatically.

## Limitations

- Computed expressions cannot call external services (use the job queue for that).
- Cross-entity expressions (e.g. aggregating child records) are not supported in the expression syntax; implement them as view queries.
