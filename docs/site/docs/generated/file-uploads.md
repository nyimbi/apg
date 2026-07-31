# File Uploads

When a `table` entity has a field of type `file`, the generated app adds multipart upload endpoints for that entity.

## Declaring a file field

```apg
table Document {
    title:     str;
    owner_id:  str;
    content:   file;        // generates upload endpoint
    thumbnail: file?;       // optional file
}
```

## Upload endpoint

```
POST /entities/Document/records/<id>/upload/content
Content-Type: multipart/form-data

file=@report.pdf
```

On success the field stores the relative path within `APG_UPLOAD_DIR`:

```json
{
  "id": "01923abc-...",
  "content": "uploads/Document/01923abc-.../report.pdf",
  ...
}
```

## Download endpoint

```
GET /entities/Document/records/<id>/download/content
```

Returns the file with appropriate `Content-Type` and `Content-Disposition` headers.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_UPLOAD_DIR` | `./uploads` | Directory where files are stored |

## Storage layout

```
uploads/
  Document/
    01923abc-.../
      report.pdf
    01923def-.../
      summary.xlsx
```

Each record gets its own subdirectory identified by the record ID.

## Size and type limits

By default there are no size limits beyond Flask's `MAX_CONTENT_LENGTH`. Set it in your environment:

```bash
export MAX_CONTENT_LENGTH=52428800   # 50 MB
```

To restrict MIME types, add validation in a `capability` block (see Capability Contracts).

## Multiple file fields

An entity can have as many `file` fields as needed:

```apg
table Employee {
    name:        str;
    photo:       file?;
    resume:      file?;
    id_document: file?;
}
```

Each field gets its own `/upload/<field_name>` and `/download/<field_name>` endpoints.
