-- Executed once by the postgres image on first start (empty volume).
-- The embedding dimension (768) must match EMBED_DIM / the embedding model.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           bigserial PRIMARY KEY,
    source       text NOT NULL,
    path         text NOT NULL,
    url          text NOT NULL,
    title        text NOT NULL,
    content_hash text NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord          int NOT NULL,
    heading_path text NOT NULL,
    content      text NOT NULL,
    anchor       text,
    embedding    vector(768) NOT NULL,
    -- Keyword index for the lexical half of hybrid search, maintained by Postgres itself
    tsv          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', heading_path), 'A') ||
                     setweight(to_tsvector('english', content), 'B')
                 ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
