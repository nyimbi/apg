# Your First App

This tutorial builds a small order-management system step by step, introducing relationships, enums, validation, and computed fields.

## Step 1 — Plain tables

```apg
module orders version 1.0.0 {
    description: "Order management";
}

table Customer {
    name:         str;
    email:        str;
    phone:        str?;
    segment:      str = "standard";
    credit_limit: decimal = 10000.0;
    is_active:    bool = true;
}

table Order {
    order_number: str;
    customer_id:  str;   // foreign key by convention
    order_date:   date;
    status:       str = "draft";
    total:        decimal = 0.0;
    notes:        str | None;
}

app Orders {
    routes: ["/customers", "/orders"];
}
```

Compile and run:

```bash
apg compile orders.apg -o out/
python out/app.py
```

## Step 2 — Add an enum

Replace the plain `str` status with a typed enum so invalid values are rejected at the API boundary:

```apg
enum OrderStatus {
    Draft;
    Confirmed;
    Shipped;
    Delivered;
    Cancelled;
}

table Order {
    order_number: str;
    customer_id:  str;
    order_date:   date;
    status:       OrderStatus = Draft;
    total:        decimal = 0.0;
    notes:        str | None;
}
```

## Step 3 — Add validation

```apg
table Customer {
    name:         str [min_length=2, max_length=120];
    email:        str [email];
    phone:        str?;
    segment:      str = "standard";
    credit_limit: decimal [min=0] = 10000.0;
    is_active:    bool = true;
}
```

Validators are enforced at the REST layer — invalid payloads return `422 Unprocessable Entity`.

## Step 4 — Add relationships

```apg
table Customer {
    name:         str;
    email:        str;
    has_many:     Order;
}

table Order {
    order_number: str;
    belongs_to:   Customer;
    order_date:   date;
    status:       str = "draft";
    total:        decimal = 0.0;
    has_many:     OrderLine;
}

table OrderLine {
    belongs_to:   Order;
    product_code: str;
    quantity:     int = 1;
    unit_price:   decimal;
}
```

This generates nested endpoints:

- `GET /entities/Customer/<id>/orders` — list orders for a customer
- `POST /entities/Order/<id>/lines` — add a line to an order

## Step 5 — Add a computed field

```apg
table Order {
    order_number:  str;
    subtotal:      decimal = 0.0;
    tax_rate:      float = 0.16;
    tax_amount:    decimal = subtotal * tax_rate;   // computed
    total:         decimal = subtotal + tax_amount;  // computed
}
```

Computed fields are re-evaluated on every read. They do not have a settable API endpoint.

## Step 6 — Compile and test

```bash
apg compile orders.apg -o out/ --verify
python out/smoke_test.py
python out/app.py --host 127.0.0.1 --port 8080
```

```bash
# Create a customer
curl -s -X POST http://localhost:8080/entities/Customer/records \
  -H 'Content-Type: application/json' \
  -d '{"record":{"name":"Asha Traders","email":"asha@example.com"}}'

# List customers
curl -s http://localhost:8080/entities/Customer/records
```

## What's next

- [Relationships deep-dive](../reference/relationships.md)
- [Enums & Validation](../reference/enums-validation.md)
- [Adding webhooks](../cookbook/webhooks.md)
- [Multi-tenant SaaS](../cookbook/multi-tenant.md)
