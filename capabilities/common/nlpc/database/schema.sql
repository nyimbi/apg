-- NLPC PostgreSQL Schema — Natural Language Processing Core
-- Copyright © 2025 Datacraft
-- Run: psql $DATABASE_URL -f database/schema.sql
--
-- All tables carry: id, tenant_id, created_at, updated_at, created_by,
-- is_deleted, version for audit and multi-tenancy.
--
-- Indexes cover: tenant scoping, FK lookups, GIN on JSONB metadata,
-- common filter columns.

-- ---------------------------------------------------------------------------
-- Extension
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid fallback

-- ---------------------------------------------------------------------------
-- nlpc_documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_documents (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1 CHECK (version >= 1),

    content         TEXT            NOT NULL,
    title           TEXT,
    source          TEXT,
    source_id       TEXT,
    language        TEXT,
    content_type    TEXT            NOT NULL DEFAULT 'text/plain',
    content_hash    TEXT,
    word_count      INTEGER         CHECK (word_count >= 0),
    char_count      INTEGER         CHECK (char_count >= 0),
    is_sensitive    BOOLEAN         NOT NULL DEFAULT FALSE,
    retention_days  INTEGER         CHECK (retention_days >= 1),
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_doc_tenant
    ON nlpc_documents (tenant_id, is_deleted, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nlpc_doc_language
    ON nlpc_documents (tenant_id, language) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_nlpc_doc_source
    ON nlpc_documents (tenant_id, source_id) WHERE source_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nlpc_doc_hash
    ON nlpc_documents (tenant_id, content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nlpc_doc_meta
    ON nlpc_documents USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- nlpc_entities
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_entities (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    text            TEXT            NOT NULL,
    entity_type     TEXT            NOT NULL,
    start_char      INTEGER         NOT NULL CHECK (start_char >= 0),
    end_char        INTEGER         NOT NULL CHECK (end_char >= start_char),
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    canonical       TEXT,
    kb_id           TEXT,
    kb_url          TEXT,
    sentence_idx    INTEGER,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_ent_doc
    ON nlpc_entities (document_id, is_deleted);
CREATE INDEX IF NOT EXISTS idx_nlpc_ent_tenant
    ON nlpc_entities (tenant_id, entity_type) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_nlpc_ent_meta
    ON nlpc_entities USING GIN (metadata);

-- ---------------------------------------------------------------------------
-- nlpc_sentiments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_sentiments (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    label           TEXT            NOT NULL,
    score           REAL            NOT NULL CHECK (score BETWEEN 0 AND 1),
    positive        REAL            NOT NULL DEFAULT 0.0 CHECK (positive BETWEEN 0 AND 1),
    negative        REAL            NOT NULL DEFAULT 0.0 CHECK (negative BETWEEN 0 AND 1),
    neutral         REAL            NOT NULL DEFAULT 0.0 CHECK (neutral BETWEEN 0 AND 1),
    compound        REAL            NOT NULL DEFAULT 0.0 CHECK (compound BETWEEN -1 AND 1),
    emotions        JSONB           NOT NULL DEFAULT '{}',
    model_used      TEXT,
    aspect_scores   JSONB           NOT NULL DEFAULT '[]',
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_sent_doc
    ON nlpc_sentiments (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_sent_tenant_label
    ON nlpc_sentiments (tenant_id, label) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_languages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_languages (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    detected        TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    candidates      JSONB           NOT NULL DEFAULT '[]',
    script          TEXT,
    is_african      BOOLEAN         NOT NULL DEFAULT FALSE,
    model_used      TEXT,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_lang_doc
    ON nlpc_languages (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_lang_detected
    ON nlpc_languages (tenant_id, detected, is_african) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_summaries
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_summaries (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    version             INTEGER     NOT NULL DEFAULT 1,

    document_id         TEXT        NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    summary_text        TEXT        NOT NULL,
    method              TEXT        NOT NULL DEFAULT 'extractive',
    max_words           INTEGER     CHECK (max_words >= 1),
    actual_word_count   INTEGER     NOT NULL DEFAULT 0 CHECK (actual_word_count >= 0),
    compression_ratio   REAL        NOT NULL DEFAULT 0.0 CHECK (compression_ratio BETWEEN 0 AND 1),
    key_sentences       JSONB       NOT NULL DEFAULT '[]',
    model_used          TEXT,
    metadata            JSONB       NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_sum_doc
    ON nlpc_summaries (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_sum_tenant
    ON nlpc_summaries (tenant_id, method) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_translations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_translations (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    source_language TEXT            NOT NULL,
    target_language TEXT            NOT NULL,
    translated_text TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    model_used      TEXT,
    char_count      INTEGER         NOT NULL DEFAULT 0 CHECK (char_count >= 0),
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_trans_doc
    ON nlpc_translations (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_trans_langs
    ON nlpc_translations (tenant_id, source_language, target_language) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_embeddings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_embeddings (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    -- vector stored as JSONB array; use pgvector extension for HNSW search
    vector          JSONB           NOT NULL DEFAULT '[]',
    dimensions      INTEGER         NOT NULL CHECK (dimensions >= 1),
    model_used      TEXT            NOT NULL,
    model_provider  TEXT            NOT NULL DEFAULT 'ollama',
    norm            REAL,
    chunk_index     INTEGER         CHECK (chunk_index >= 0),
    chunk_text      TEXT,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_emb_doc
    ON nlpc_embeddings (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_emb_model
    ON nlpc_embeddings (tenant_id, model_used) WHERE is_deleted = FALSE;

-- pgvector support (optional — apply after installing the extension):
-- ALTER TABLE nlpc_embeddings ADD COLUMN IF NOT EXISTS vec_768 vector(768);
-- CREATE INDEX IF NOT EXISTS idx_nlpc_emb_hnsw ON nlpc_embeddings
--     USING hnsw (vec_768 vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- nlpc_classifications
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_classifications (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    taxonomy        TEXT            NOT NULL,
    label           TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    all_scores      JSONB           NOT NULL DEFAULT '{}',
    hierarchy       JSONB           NOT NULL DEFAULT '[]',
    model_used      TEXT,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_cls_doc
    ON nlpc_classifications (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_cls_taxonomy
    ON nlpc_classifications (tenant_id, taxonomy, label) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_intents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_intents (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    intent_label    TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    all_scores      JSONB           NOT NULL DEFAULT '{}',
    model_used      TEXT,
    utterance       TEXT,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_int_doc
    ON nlpc_intents (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_int_label
    ON nlpc_intents (tenant_id, intent_label) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_relations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_relations (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    subject_id      TEXT            NOT NULL,
    object_id       TEXT            NOT NULL,
    relation        TEXT            NOT NULL,
    confidence      REAL            NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    sentence_idx    INTEGER,
    model_used      TEXT,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_rel_doc
    ON nlpc_relations (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_rel_relation
    ON nlpc_relations (tenant_id, relation) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_coref_chains
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_coref_chains (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    cluster_id      INTEGER         NOT NULL CHECK (cluster_id >= 0),
    mentions        JSONB           NOT NULL DEFAULT '[]',
    representative  TEXT,
    entity_id       TEXT,
    model_used      TEXT,

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_coref_doc
    ON nlpc_coref_chains (document_id);

-- ---------------------------------------------------------------------------
-- nlpc_keyphrases
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_keyphrases (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    version         INTEGER         NOT NULL DEFAULT 1,

    document_id     TEXT            NOT NULL REFERENCES nlpc_documents(id) ON DELETE CASCADE,
    phrase          TEXT            NOT NULL,
    score           REAL            NOT NULL DEFAULT 0.0 CHECK (score BETWEEN 0 AND 1),
    frequency       INTEGER         NOT NULL DEFAULT 1 CHECK (frequency >= 1),
    method          TEXT            NOT NULL DEFAULT 'tfidf',
    metadata        JSONB           NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_kp_doc
    ON nlpc_keyphrases (document_id);
CREATE INDEX IF NOT EXISTS idx_nlpc_kp_tenant
    ON nlpc_keyphrases (tenant_id) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_batch_jobs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_batch_jobs (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    version             INTEGER     NOT NULL DEFAULT 1,

    name                TEXT        NOT NULL,
    document_ids        JSONB       NOT NULL DEFAULT '[]',
    tasks               JSONB       NOT NULL DEFAULT '[]',
    status              TEXT        NOT NULL DEFAULT 'pending',
    priority            TEXT        NOT NULL DEFAULT 'normal',
    progress            REAL        NOT NULL DEFAULT 0.0 CHECK (progress BETWEEN 0 AND 100),
    total_documents     INTEGER     NOT NULL CHECK (total_documents >= 1),
    processed_documents INTEGER     NOT NULL DEFAULT 0 CHECK (processed_documents >= 0),
    failed_documents    INTEGER     NOT NULL DEFAULT 0 CHECK (failed_documents >= 0),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error_summary       TEXT,
    configuration       JSONB       NOT NULL DEFAULT '{}',

    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_batch_tenant_status
    ON nlpc_batch_jobs (tenant_id, status) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- nlpc_model_configs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nlpc_model_configs (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    version             INTEGER     NOT NULL DEFAULT 1,

    name                TEXT        NOT NULL,
    provider            TEXT        NOT NULL,
    model_name          TEXT        NOT NULL,
    supported_tasks     JSONB       NOT NULL DEFAULT '[]',
    supported_languages JSONB       NOT NULL DEFAULT '["en"]',
    max_input_chars     INTEGER     NOT NULL DEFAULT 100000 CHECK (max_input_chars >= 1),
    gpu_required        BOOLEAN     NOT NULL DEFAULT FALSE,
    memory_mb           INTEGER     CHECK (memory_mb >= 0),
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    load_priority       INTEGER     NOT NULL DEFAULT 50 CHECK (load_priority BETWEEN 1 AND 100),
    configuration       JSONB       NOT NULL DEFAULT '{}',
    performance_metrics JSONB       NOT NULL DEFAULT '{}',

    PRIMARY KEY (id),
    UNIQUE (tenant_id, name)
);

CREATE INDEX IF NOT EXISTS idx_nlpc_mc_tenant_active
    ON nlpc_model_configs (tenant_id, is_active) WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- Audit trigger — update updated_at on every row change
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION nlpc_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    NEW.version    := OLD.version + 1;
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'nlpc_documents', 'nlpc_entities', 'nlpc_sentiments',
        'nlpc_languages', 'nlpc_summaries', 'nlpc_translations',
        'nlpc_embeddings', 'nlpc_classifications', 'nlpc_intents',
        'nlpc_relations', 'nlpc_coref_chains', 'nlpc_keyphrases',
        'nlpc_batch_jobs', 'nlpc_model_configs'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%s_updated_at ON %s; '
            'CREATE TRIGGER trg_%s_updated_at '
            'BEFORE UPDATE ON %s FOR EACH ROW '
            'EXECUTE FUNCTION nlpc_set_updated_at();',
            t, t, t, t
        );
    END LOOP;
END;
$$;
