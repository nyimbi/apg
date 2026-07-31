# With Relationships

This cookbook walks through building a blog platform that demonstrates `has_many`, `belongs_to`, `has_one`, and `through` relationships.

## Schema

```apg
module blog version 1.0.0 {
    description: "Blog platform";
}

table Author {
    name:         str;
    email:        str;
    bio:          text?;
    is_active:    bool = true;
    has_many:     Post;
    has_one:      AuthorProfile;
}

table AuthorProfile {
    belongs_to:   Author;
    avatar_url:   str?;
    website:      str?;
    twitter:      str?;
}

table Post {
    belongs_to:   Author;
    title:        str;
    slug:         str;
    body:         text;
    status:       str = "draft";
    published_at: datetime?;
    has_many:     Comment;
    has_many:     Tag through PostTag;
}

table Tag {
    name:    str;
    slug:    str;
    has_many: Post through PostTag;
}

table PostTag {
    belongs_to: Post;
    belongs_to: Tag;
}

table Comment {
    belongs_to:   Post;
    author_name:  str;
    body:         text;
    is_approved:  bool = false;
}

app Blog {
    routes: ["/authors", "/posts", "/tags", "/comments"];
}
```

## Compile and run

```bash
apg compile blog.apg -o out/ --verify
python out/app.py --host 127.0.0.1 --port 8080
```

## Working with nested endpoints

### Create an author

```bash
curl -s -X POST http://localhost:8080/entities/Author/records \
  -H "Content-Type: application/json" \
  -d '{"record": {"name": "Wanjiru Kamau", "email": "wanjiru@example.com"}}'
```

Note the returned `id`, e.g. `01923abc-...`.

### Create a post for that author

```bash
curl -s -X POST http://localhost:8080/entities/Author/01923abc-.../posts \
  -H "Content-Type: application/json" \
  -d '{"record": {"title": "Hello World", "slug": "hello-world", "body": "My first post."}}'
```

The `author_id` field is set automatically from the URL.

### List all posts by an author

```bash
curl http://localhost:8080/entities/Author/01923abc-.../posts
```

### Add a comment to a post

```bash
curl -s -X POST http://localhost:8080/entities/Post/01923def-.../comments \
  -H "Content-Type: application/json" \
  -d '{"record": {"author_name": "Reader", "body": "Great article!"}}'
```

### Tag a post (many-to-many)

```bash
# Create tag
curl -s -X POST http://localhost:8080/entities/Tag/records \
  -H "Content-Type: application/json" \
  -d '{"record": {"name": "Python", "slug": "python"}}'

# Associate tag with post
curl -s -X POST http://localhost:8080/entities/Post/01923def-.../tags \
  -H "Content-Type: application/json" \
  -d '{"record": {"tag_id": "01923fff-..."}}'
```

### List tags on a post

```bash
curl http://localhost:8080/entities/Post/01923def-.../tags
```

### Get the author's profile (has_one)

```bash
curl http://localhost:8080/entities/Author/01923abc-.../profile
```

### Set profile

```bash
curl -s -X PUT http://localhost:8080/entities/Author/01923abc-.../profile \
  -H "Content-Type: application/json" \
  -d '{"record": {"avatar_url": "https://example.com/avatar.png", "website": "https://wanjiru.dev"}}'
```

## Full endpoint map

```
GET  /entities/Author/records
POST /entities/Author/records
GET  /entities/Author/<id>
PUT  /entities/Author/<id>
GET  /entities/Author/<id>/posts           ← has_many
POST /entities/Author/<id>/posts
GET  /entities/Author/<id>/profile         ← has_one
PUT  /entities/Author/<id>/profile
GET  /entities/Post/<id>/comments          ← has_many
POST /entities/Post/<id>/comments
GET  /entities/Post/<id>/tags              ← through
POST /entities/Post/<id>/tags
DELETE /entities/Post/<id>/tags/<tag_id>
```
