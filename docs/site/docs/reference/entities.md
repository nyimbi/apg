# Entities & Fields

Every APG program is built from **entities** — named blocks that declare structure and behaviour. The most common entity kind for data is `table`.

## Syntax

```apg
table EntityName {
    field_name: type;
    field_name: type = default_value;
    optional_field: type?;
    nullable_field: type | None;
}
```

## Scalar field types

| APG type | Python type | SQL (PostgreSQL) | Notes |
|----------|-------------|-------------------|-------|
| `str` | `str` | `TEXT` | UTF-8 string |
| `int` | `int` | `INTEGER` | 32-bit signed integer |
| `float` | `float` | `DOUBLE PRECISION` | IEEE 754 double |
| `decimal` | `Decimal` | `NUMERIC(18,4)` | Exact fixed-point; prefer for money |
| `bool` | `bool` | `BOOLEAN` | `true` / `false` |
| `date` | `date` | `DATE` | ISO 8601 date |
| `datetime` | `datetime` | `TIMESTAMP WITH TIME ZONE` | Timezone-aware |
| `time` | `time` | `TIME` | Wall-clock time |
| `bytes` | `bytes` | `BYTEA` | Raw binary |
| `uuid` | `str` | `TEXT` (UUID v7) | Auto-generated ID |
| `text` | `str` | `TEXT` | Alias for `str`; signals large text |
| `json` | `Any` | `JSONB` | Arbitrary JSON value |

## Collection types

| APG type | Python type | SQL |
|----------|-------------|-----|
| `List[str]` | `list[str]` | `JSONB` |
| `Dict[str, str]` | `dict[str, str]` | `JSONB` |
| `Dict[str, Any]` | `dict[str, Any]` | `JSONB` |
| `List[int]` | `list[int]` | `JSONB` |

## Optionality

```apg
table User {
    name:       str;          // required — cannot be null
    nickname:   str?;         // optional shorthand — nullable
    bio:        str | None;   // explicit union with None — same effect
    score:      int = 0;      // required with default
    avatar_url: str? = "";    // optional with default
}
```

## Default values

Defaults are evaluated at record-creation time:

```apg
table Order {
    status:     str = "draft";
    created_at: datetime;          // auto-set to NOW() by the app
    amount:     decimal = 0.0;
    is_active:  bool = true;
    flags:      List[str] = [];
    meta:       Dict[str, Any] = {};
}
```

## File fields

Use the `file` type to enable file-upload support for a field:

```apg
table Document {
    title:    str;
    owner_id: str;
    content:  file;    // generates multipart/form-data upload endpoint
}
```

Uploaded files are stored in `APG_UPLOAD_DIR` (default: `./uploads`). The field stores a relative path string. See [File Uploads](../generated/file-uploads.md).

## Auto-generated fields

Every `table` entity automatically receives:

| Field | Type | Behaviour |
|-------|------|-----------|
| `id` | UUID v7 string | Auto-generated on creation |
| `created_at` | datetime | Set on first write |
| `updated_at` | datetime | Updated on every write |
| `deleted_at` | datetime? | Set on soft-delete; `NULL` while alive |

## Module declaration

A module wraps all entities in a namespace:

```apg
module my_app version 1.0.0 {
    description: "Short description";
}
```

The `module` name becomes the Python module name of the generated `app.py`.

## App entity

Every APG file needs exactly one `app` entity to tell the compiler which entities to wire into routes:

```apg
app MyApp {
    description: "My application";
    routes: ["/customers", "/orders"];
}
```

Routes map to entity endpoint prefixes. The entity name is inferred from the path segment.

## Full example

```apg
module crm version 1.0.0 {
    description: "Customer Relationship Manager";
}

table Contact {
    first_name:   str;
    last_name:    str;
    email:        str;
    phone:        str?;
    company:      str | None;
    score:        int = 0;
    revenue:      decimal = 0.0;
    is_qualified: bool = false;
    tags:         List[str];
    meta:         Dict[str, Any];
    joined_at:    date;
    last_seen:    datetime?;
}

app CRM {
    description: "CRM";
    routes: ["/contacts"];
}
```
