"""
conftest.py — Shared pytest fixtures for nsys-ai tests.

All fixtures here are available to every test module without explicit imports.
"""

import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Minimal in-memory SQLite database that mimics an Nsight Systems export.
# Only the tables and columns exercised by the test suite are created; this
# avoids the need for a real .sqlite profile during CI.
# ---------------------------------------------------------------------------

_NSYS_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS StringIds (
    id      INTEGER PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS TARGET_INFO_GPU (
    id              INTEGER PRIMARY KEY,
    name            TEXT,
    busLocation     TEXT DEFAULT '',
    totalMemory     INTEGER DEFAULT 0,
    smCount         INTEGER DEFAULT 0,
    chipName        TEXT DEFAULT '',
    memoryBandwidth INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS TARGET_INFO_CUDA_DEVICE (
    gpuId    INTEGER,
    cudaId   INTEGER,
    pid      INTEGER DEFAULT 0,
    uuid     TEXT DEFAULT '',
    numMultiprocessors INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_KERNEL (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    shortName       INTEGER NOT NULL,
    demangledName   INTEGER DEFAULT 0,
    gridX           INTEGER DEFAULT 1,
    gridY           INTEGER DEFAULT 1,
    gridZ           INTEGER DEFAULT 1,
    blockX          INTEGER DEFAULT 1,
    blockY          INTEGER DEFAULT 1,
    blockZ          INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_MEMCPY (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    copyKind        INTEGER DEFAULT 0,
    bytes           INTEGER DEFAULT 0,
    srcKind         INTEGER DEFAULT 0,
    dstKind         INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_MEMSET (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    bytes           INTEGER DEFAULT 0,
    value           INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_RUNTIME (
    globalTid       INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    nameId          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS NVTX_EVENTS (
    globalTid       INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER DEFAULT -1,
    text            TEXT DEFAULT '',
    eventType       INTEGER DEFAULT 59,
    rangeId         INTEGER DEFAULT 0,
    textId          INTEGER DEFAULT NULL
);
"""

_NSYS_SEED_SQL = """\
INSERT INTO StringIds VALUES
    (1, 'kernel_A'), (2, 'kernel_B'), (10, 'nccl_AllReduce_kernel'),
    -- CUDA API names for anti-pattern detection tests
    (20, 'cudaDeviceSynchronize'),
    (21, 'cudaMemcpy'),
    (22, 'cudaMemcpyAsync'),
    (23, 'cudaMemset'),
    (24, 'cudaLaunchKernel');

INSERT INTO TARGET_INFO_GPU VALUES
    (0, 'NVIDIA Test GPU', '0000:00:00.0', 8589934592, 108, 'TestChip', 0);

INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
    (0, 0, 100, '', 108);

INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    (100, 0, 7, 1, 1000000, 2000000, 1, 1, 32, 1, 1, 256, 1, 1),
    (100, 0, 7, 2, 3000000, 4000000, 2, 2, 16, 1, 1, 128, 1, 1),
    (100, 0, 8, 10, 2500000, 3500000, 10, 10, 1, 1, 1, 512, 1, 1);

INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
    -- Kernel launches (existing)
    (100, 1,   900000,  1000000, 24),   -- cudaLaunchKernel
    (100, 2,  2900000,  3000000, 24),   -- cudaLaunchKernel
    (100, 10, 2400000,  2500000, 24),   -- cudaLaunchKernel
    -- Sync API calls (anti-pattern: Excessive Synchronization)
    (100, 100, 5000000, 15000000, 20),  -- cudaDeviceSynchronize, 10ms
    (100, 101, 16000000, 22000000, 20), -- cudaDeviceSynchronize, 6ms
    -- Sync memcpy (anti-pattern: Synchronous Memcpy)
    (100, 102, 23000000, 23500000, 21), -- cudaMemcpy (sync), 0.5ms
    -- Sync memset (anti-pattern: Synchronous Memset)
    (100, 103, 24000000, 24200000, 23); -- cudaMemset (sync), 0.2ms

-- Sync memcpy correlated data (correlationId=102 matches cudaMemcpy runtime entry)
-- srcKind=1 = Pageable memory (see CUPTI schema docs)
INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
    (100, 0, 7, 102, 1, 1048576, 1, 2, 23100000, 23400000);

-- Sync memset correlated data
INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
    (100, 0, 7, 103, 4096, 0, 24050000, 24150000);

INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) VALUES
    (100, 500000,  4500000, 'train_step', 59, 0),
    (100, 900000,  2100000, 'forward',    59, 1);
"""


@pytest.fixture
def minimal_nsys_conn():
    """Return a SQLite connection pre-populated with minimal Nsight tables.

    The connection uses :memory: and is closed automatically after the test.
    Use this fixture in unit tests that need a sqlite3.Connection object.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_NSYS_SCHEMA_SQL)
    conn.executescript(_NSYS_SEED_SQL)
    yield conn
    conn.close()


@pytest.fixture
def minimal_nsys_db_path(tmp_path):
    """Write the minimal Nsight schema to a temporary .sqlite file and return its path.

    Use this fixture when a test requires an actual file path (e.g. for
    profile.open() or CLI subprocess calls).
    """
    db_path = tmp_path / "test_profile.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_NSYS_SCHEMA_SQL)
    conn.executescript(_NSYS_SEED_SQL)
    conn.close()
    return str(db_path)
