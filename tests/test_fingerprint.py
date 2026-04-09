"""Tests for framework fingerprinting logic."""

import sqlite3

from nsys_ai.fingerprint import get_fingerprint


def build_mock_db(string_ids: list[str], target_info: list[tuple] = None, payload_schemas: list[str] = None):
    conn = sqlite3.connect(":memory:")

    # We must patch PRAGMA table_info natively or just use sqlite3 correctly so adapter understands
    c = conn.cursor()
    c.execute("CREATE TABLE StringIds (value TEXT)")

    # Needs to be a valid schema for adapter
    c.execute("CREATE TABLE NsightSchemaMeta (version INTEGER)")

    for s in string_ids:
        c.execute("INSERT INTO StringIds VALUES (?)", (s,))

    if target_info:
        c.execute("CREATE TABLE TARGET_INFO_NIC_INFO (vendorId TEXT, name TEXT)")
        for v in target_info:
            c.execute("INSERT INTO TARGET_INFO_NIC_INFO VALUES (?, ?)", v)

    if payload_schemas:
        c.execute("CREATE TABLE NVTX_PAYLOAD_SCHEMAS (name TEXT)")
        for p in payload_schemas:
            c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (?)", (p,))

    conn.commit()
    return conn

def test_fingerprint_megatron():
    conn = build_mock_db(["MegatronModule", "flash_attn"])
    fp = get_fingerprint(conn)
    assert fp.framework == "Megatron-LM"
    assert not fp.distributed
    assert not fp.multi_node

def test_fingerprint_vllm():
    conn = build_mock_db(["SamplerOutput", "vllm"])
    fp = get_fingerprint(conn)
    assert fp.framework == "vLLM"

def test_fingerprint_pytorch_generic():
    conn = build_mock_db(["forward", "backward"])
    fp = get_fingerprint(conn)
    assert fp.framework == "PyTorch"

def test_fingerprint_ambiguous():
    # A profile with both PyTorch generic strings and Megatron clusters.
    # To ensure Megatron-LM wins over PyTorch, we add 3 Megatron hits, and 2 PyTorch hits.
    conn = build_mock_db([
        "Megatron_1", "Megatron_2", "Megatron_3",
        "forward", "backward"
    ])
    fp = get_fingerprint(conn)
    assert fp.framework == "Megatron-LM"

def test_topology():
    conn = build_mock_db([], target_info=[("5555", "mlx5_0")], payload_schemas=["NCCL communicator"])
    fp = get_fingerprint(conn)
    assert fp.distributed is True
    assert fp.multi_node is True

def test_legacy_fallback():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE NVTX_EVENTS (text TEXT)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES ('DeepSpeedEngine')")
    conn.commit()

    fp = get_fingerprint(conn)
    assert fp.framework == "DeepSpeed"
