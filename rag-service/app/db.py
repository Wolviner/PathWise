"""Postgres access. The service reads PathWise's own tables (users, accounts,
transactions, budgets) and reads/writes the rag_chunks table that the PathWise
Prisma migration creates."""

from contextlib import contextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg

from .config import DATABASE_URL

# Prisma-only URL params (pgbouncer=true, connection_limit, schema...) are not valid
# for libpq, so keep only the ones libpq understands.
_LIBPQ_PARAMS = {"sslmode", "connect_timeout", "application_name", "options", "sslrootcert"}


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k in _LIBPQ_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query)))


@contextmanager
def connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    # prepare_threshold=None: Supabase's transaction pooler (port 6543) can't use
    # server-side prepared statements.
    try:
        conn = psycopg.connect(clean_url(DATABASE_URL), prepare_threshold=None, connect_timeout=15)
    except psycopg.ProgrammingError:
        # psycopg echoes the connection string (with the password) in this error; don't log it.
        raise RuntimeError("DATABASE_URL is not a valid Postgres connection URL") from None
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def to_vector(values: list[float]) -> str:
    """pgvector text format, used with a ::vector cast."""
    return "[" + ",".join(f"{v:.7f}" for v in values) + "]"
