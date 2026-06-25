"""Postgres-backed stores (pgvector / VectorChord) behind the VectorStore/DocStore/GraphStore APIs.

Rows are tagged by `namespace` (the bearer/api-key identity), so one database serves many isolated agents.
Tables:
  engram_vectors(namespace, collection, key, embedding vector, payload jsonb)   facts/cold/episodes/summary
  engram_docs(namespace, key, payload jsonb)                                     Episode log
  engram_entities(namespace, id, user_id, name_lc, payload jsonb)               graph nodes
  engram_relations(namespace, id, subject_id, object_id, fact_id, valid_at, invalid_at, payload jsonb)
  engram_meta(namespace, blob bytea)                                             pickled aux state
"""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Optional

from ..types import Episode, Entity, Fact, Relation
from ..util import cosine
from .base import DocStore, GraphStore, Predicate, VectorStore

_VEC_TYPES = {"facts": Fact, "cold": Fact, "episodes": Episode, "summary": Episode}
_VEC_FIELD = {"episodes": "embedding", "summary": "summary_embedding"}

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS engram_vectors (
  namespace text NOT NULL, collection text NOT NULL, key text NOT NULL,
  embedding vector, payload jsonb NOT NULL,
  PRIMARY KEY (namespace, collection, key)
);
CREATE TABLE IF NOT EXISTS engram_docs (
  namespace text NOT NULL, key text NOT NULL, payload jsonb NOT NULL,
  PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS engram_entities (
  namespace text NOT NULL, id text NOT NULL, user_id text NOT NULL, name_lc text NOT NULL,
  payload jsonb NOT NULL,
  PRIMARY KEY (namespace, id), UNIQUE (namespace, user_id, name_lc)
);
CREATE TABLE IF NOT EXISTS engram_relations (
  namespace text NOT NULL, id text NOT NULL, subject_id text NOT NULL, object_id text NOT NULL,
  fact_id text NOT NULL, valid_at double precision NOT NULL, invalid_at double precision,
  payload jsonb NOT NULL,
  PRIMARY KEY (namespace, id)
);
CREATE INDEX IF NOT EXISTS engram_rel_subj ON engram_relations (namespace, subject_id);
CREATE INDEX IF NOT EXISTS engram_rel_obj  ON engram_relations (namespace, object_id);
CREATE INDEX IF NOT EXISTS engram_rel_fact ON engram_relations (namespace, fact_id);
CREATE TABLE IF NOT EXISTS engram_meta (
  namespace text PRIMARY KEY, blob bytea NOT NULL
);
"""

_POOLS: dict[str, Any] = {}
_POOLS_LOCK = threading.Lock()


def _get_pool(dsn: str):
    with _POOLS_LOCK:
        pool = _POOLS.get(dsn)
        if pool is not None:
            return pool
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        # The vector extension + tables must exist before register_vector runs on pooled connections.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(_SCHEMA)

        def _configure(conn) -> None:
            conn.autocommit = True
            register_vector(conn)

        pool = ConnectionPool(dsn, min_size=1, max_size=8, configure=_configure, open=True)
        _POOLS[dsn] = pool
        return pool


def _to_list(vec) -> Optional[list[float]]:
    if vec is None:
        return None
    return [float(x) for x in vec]


class _VecCodec:
    """Serialize a dataclass payload to jsonb without its searched embedding (which lives in the `embedding`
    column) and restore it on read."""

    def __init__(self, collection: str) -> None:
        self.cls = _VEC_TYPES[collection]
        self.field = _VEC_FIELD.get(collection, "embedding")

    def dump(self, payload: Any) -> dict:
        d = asdict(payload)
        d.pop("embedding", None)
        d.pop("summary_embedding", None)
        return d

    def load(self, payload: dict, embedding) -> Any:
        payload[self.field] = _to_list(embedding)
        return self.cls(**payload)


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, namespace: str, collection: str) -> None:
        self._pool = _get_pool(dsn)
        self._ns = namespace
        self._col = collection
        self._codec = _VecCodec(collection)

    def upsert(self, key: str, vector: list[float], payload: Any) -> None:
        from psycopg.types.json import Jsonb

        emb = vector if vector else None
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO engram_vectors (namespace, collection, key, embedding, payload) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (namespace, collection, key) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "payload = EXCLUDED.payload",
                (self._ns, self._col, key, emb, Jsonb(self._codec.dump(payload))),
            )

    def search(self, vector: list[float], top_k: int,
               where: Optional[Predicate] = None) -> list[tuple[float, Any]]:
        overfetch = max(top_k * 8, 256)
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT payload, embedding FROM engram_vectors "
                "WHERE namespace = %s AND collection = %s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (self._ns, self._col, vector, overfetch),
            ).fetchall()
        out: list[tuple[float, Any]] = []
        for payload, emb in rows:
            obj = self._codec.load(payload, emb)
            if where is not None and not where(obj):
                continue
            out.append((cosine(vector, _to_list(emb) or []), obj))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[:top_k]

    def get(self, key: str) -> Any | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT payload, embedding FROM engram_vectors "
                "WHERE namespace = %s AND collection = %s AND key = %s",
                (self._ns, self._col, key),
            ).fetchone()
        return self._codec.load(row[0], row[1]) if row else None

    def delete(self, key: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM engram_vectors WHERE namespace = %s AND collection = %s AND key = %s",
                (self._ns, self._col, key),
            )

    def values(self) -> list[Any]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT payload, embedding FROM engram_vectors WHERE namespace = %s AND collection = %s",
                (self._ns, self._col),
            ).fetchall()
        return [self._codec.load(p, e) for p, e in rows]


class PgDocStore(DocStore):
    def __init__(self, dsn: str, namespace: str) -> None:
        self._pool = _get_pool(dsn)
        self._ns = namespace

    def put(self, key: str, obj: Any) -> None:
        from psycopg.types.json import Jsonb

        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO engram_docs (namespace, key, payload) VALUES (%s, %s, %s) "
                "ON CONFLICT (namespace, key) DO UPDATE SET payload = EXCLUDED.payload",
                (self._ns, key, Jsonb(asdict(obj))),
            )

    def get(self, key: str) -> Any | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT payload FROM engram_docs WHERE namespace = %s AND key = %s", (self._ns, key)
            ).fetchone()
        return Episode(**row[0]) if row else None

    def values(self) -> list[Any]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM engram_docs WHERE namespace = %s", (self._ns,)
            ).fetchall()
        return [Episode(**r[0]) for r in rows]

    def delete(self, key: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM engram_docs WHERE namespace = %s AND key = %s", (self._ns, key))


class PgGraphStore(GraphStore):
    def __init__(self, dsn: str, namespace: str) -> None:
        self._pool = _get_pool(dsn)
        self._ns = namespace

    @property
    def entities(self) -> dict[str, Entity]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM engram_entities WHERE namespace = %s", (self._ns,)
            ).fetchall()
        return {r[0]["id"]: Entity(**r[0]) for r in rows}

    def upsert_entity(self, entity: Entity) -> Entity:
        from psycopg.types.json import Jsonb

        name_lc = entity.name.lower()
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO engram_entities (namespace, id, user_id, name_lc, payload) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (namespace, user_id, name_lc) DO NOTHING RETURNING payload",
                (self._ns, entity.id, entity.user_id, name_lc, Jsonb(asdict(entity))),
            ).fetchone()
            if row is not None:
                return entity
            existing = conn.execute(
                "SELECT payload FROM engram_entities WHERE namespace = %s AND user_id = %s AND name_lc = %s",
                (self._ns, entity.user_id, name_lc),
            ).fetchone()
        return Entity(**existing[0]) if existing else entity

    def get_entity(self, user_id: str, name: str) -> Entity | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT payload FROM engram_entities WHERE namespace = %s AND user_id = %s AND name_lc = %s",
                (self._ns, user_id, name.lower()),
            ).fetchone()
        return Entity(**row[0]) if row else None

    def add_relation(self, relation: Relation) -> None:
        from psycopg.types.json import Jsonb

        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO engram_relations (namespace, id, subject_id, object_id, fact_id, valid_at, "
                "invalid_at, payload) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (namespace, id) DO UPDATE SET invalid_at = EXCLUDED.invalid_at, "
                "payload = EXCLUDED.payload",
                (self._ns, relation.id, relation.subject_id, relation.object_id, relation.fact_id,
                 relation.valid_at, relation.invalid_at, Jsonb(asdict(relation))),
            )

    def neighbors(self, entity_id: str, as_of: Optional[float] = None,
                  direction: str = "out") -> list[Relation]:
        col = "subject_id" if direction == "out" else "object_id"
        sql = f"SELECT payload FROM engram_relations WHERE namespace = %s AND {col} = %s"
        params: list = [self._ns, entity_id]
        if as_of is not None:
            sql += " AND valid_at <= %s AND (invalid_at IS NULL OR invalid_at > %s)"
            params += [as_of, as_of]
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Relation(**r[0]) for r in rows]

    def invalidate_relations_for_fact(self, fact_id: str, t: float) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE engram_relations SET invalid_at = %s, payload = jsonb_set(payload, '{invalid_at}', "
                "to_jsonb(%s::double precision)) "
                "WHERE namespace = %s AND fact_id = %s AND invalid_at IS NULL",
                (t, t, self._ns, fact_id),
            )

    def relations(self) -> list[Relation]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT payload FROM engram_relations WHERE namespace = %s", (self._ns,)
            ).fetchall()
        return [Relation(**r[0]) for r in rows]


class PostgresBackend:
    kind = "postgres"

    def __init__(self, dsn: str, namespace: str) -> None:
        self.dsn = dsn
        self.namespace = namespace
        self._pool = _get_pool(dsn)

    def vector(self, collection: str) -> PgVectorStore:
        return PgVectorStore(self.dsn, self.namespace, collection)

    def doc(self) -> PgDocStore:
        return PgDocStore(self.dsn, self.namespace)

    def graph(self) -> PgGraphStore:
        return PgGraphStore(self.dsn, self.namespace)

    def meta_load(self) -> Optional[bytes]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT blob FROM engram_meta WHERE namespace = %s", (self.namespace,)
            ).fetchone()
        return bytes(row[0]) if row else None

    def meta_save(self, blob: bytes) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO engram_meta (namespace, blob) VALUES (%s, %s) "
                "ON CONFLICT (namespace) DO UPDATE SET blob = EXCLUDED.blob",
                (self.namespace, blob),
            )

    def drop_namespace(self) -> None:
        with self._pool.connection() as conn:
            for tbl in ("engram_vectors", "engram_docs", "engram_entities", "engram_relations",
                        "engram_meta"):
                conn.execute(f"DELETE FROM {tbl} WHERE namespace = %s", (self.namespace,))
