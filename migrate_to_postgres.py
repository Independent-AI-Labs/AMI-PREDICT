#!/usr/bin/env python
"""
Migrate SQLite data to PostgreSQL for parallel processing.
"""

import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# PostgreSQL connection (configurable via environment)
PG_HOST = os.getenv("PG_HOST", os.getenv("POSTGRES_HOST", os.getenv("AMI_HOST", "127.0.0.1")))
PG_PORT = int(os.getenv("PG_PORT", os.getenv("POSTGRES_PORT", 5432)))
PG_DB = os.getenv("PG_DB", os.getenv("POSTGRES_DB", "postgres"))  # Use default database
PG_USER = os.getenv("PG_USER", os.getenv("POSTGRES_USER", "postgres"))
PG_PASS = os.getenv("PG_PASS", os.getenv("POSTGRES_PASSWORD", "postgres"))


def create_pg_tables():
    """Create PostgreSQL tables with proper indexing."""
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER, password=PG_PASS)
    cursor = conn.cursor()

    # Drop and recreate table
    cursor.execute("DROP TABLE IF EXISTS market_data_1m CASCADE")

    cursor.execute(
        """
        CREATE TABLE market_data_1m (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL
        )
    """
    )

    # Create indexes for fast parallel queries
    cursor.execute("CREATE INDEX idx_symbol ON market_data_1m(symbol)")
    cursor.execute("CREATE INDEX idx_timestamp ON market_data_1m(timestamp)")
    cursor.execute("CREATE INDEX idx_symbol_timestamp ON market_data_1m(symbol, timestamp)")

    conn.commit()
    logger.info("Created PostgreSQL tables with indexes")
    return conn


def migrate_batch(batch_data, conn):
    """Migrate a batch of data to PostgreSQL."""
    cursor = conn.cursor()

    query = """
        INSERT INTO market_data_1m (symbol, timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    execute_batch(cursor, query, batch_data, page_size=10000)
    conn.commit()

    return len(batch_data)


def migrate_data():
    """Migrate data from SQLite to PostgreSQL in parallel."""

    # Connect to SQLite
    sqlite_conn = sqlite3.connect("data/crypto_5years.db")

    # Create PostgreSQL tables
    pg_conn = create_pg_tables()

    # Get total count
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM market_data_1m")
    total_records = cursor.fetchone()[0]
    logger.info(f"Migrating {total_records:,} records to PostgreSQL")

    # Read data in chunks
    chunk_size = 100000
    offset = 0
    migrated = 0
    start_time = time.time()

    while offset < total_records:
        # Read chunk from SQLite
        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM market_data_1m
            LIMIT {chunk_size} OFFSET {offset}
        """

        cursor.execute(query)
        batch_data = cursor.fetchall()

        if not batch_data:
            break

        # Migrate to PostgreSQL
        records = migrate_batch(batch_data, pg_conn)
        migrated += records
        offset += chunk_size

        # Progress
        elapsed = time.time() - start_time
        speed = migrated / elapsed
        eta = (total_records - migrated) / speed

        logger.info(
            f"Migrated {migrated:,}/{total_records:,} records " f"({migrated/total_records*100:.1f}%) - " f"Speed: {speed:.0f} rec/sec - ETA: {eta/60:.1f} min"
        )

    # Analyze tables for query optimization
    cursor = pg_conn.cursor()
    cursor.execute("ANALYZE market_data_1m")
    pg_conn.commit()

    elapsed = time.time() - start_time
    logger.info(f"\nMigration complete in {elapsed/60:.1f} minutes")
    logger.info(f"Average speed: {total_records/elapsed:.0f} records/second")

    sqlite_conn.close()
    pg_conn.close()


def test_parallel_query():
    """Test parallel query performance."""
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, database=PG_DB, user=PG_USER, password=PG_PASS)

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

    def query_symbol(symbol):
        cursor = conn.cursor()
        start = time.time()
        cursor.execute(
            """
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM market_data_1m
            WHERE symbol = %s
        """,
            (symbol,),
        )
        result = cursor.fetchone()
        elapsed = time.time() - start
        return symbol, result, elapsed

    logger.info("\nTesting parallel queries...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(query_symbol, symbol) for symbol in symbols]

        for future in as_completed(futures):
            symbol, result, elapsed = future.result()
            count, min_ts, max_ts = result
            logger.info(f"{symbol}: {count:,} records from {min_ts} to {max_ts} " f"(query time: {elapsed:.2f}s)")

    conn.close()


if __name__ == "__main__":
    migrate_data()
    test_parallel_query()
