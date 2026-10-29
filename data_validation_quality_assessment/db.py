"""Read-only database connection for the validation stage (settings come from .env)."""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def get_readonly_connection():
    """Open a PostgreSQL connection that cannot write. The password is never printed."""
    password = os.getenv("DB_PASSWORD")
    if not password or password == "change_me":
        raise SystemExit("DB_PASSWORD is missing in .env. Copy .env.example to .env first.")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "task3_articles"),
        user=os.getenv("DB_USER", "task3_user"),
        password=password,
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
    )
    # read-only: any INSERT/UPDATE/DELETE/DDL fails. REPEATABLE READ: every query of one run
    # sees the same snapshot of the data, so the two duplicate-check passes always agree.
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    return conn
