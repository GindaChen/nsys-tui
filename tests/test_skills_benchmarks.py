def test_module_loading_execute(minimal_nsys_conn):
    """Test module_loading skill executes and correctly aggregates JIT events."""
    from nsys_ai.skills.registry import get_skill

    # Insert a synthetic module load runtime event
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (999, 'cuModuleLoadData')")
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (10, 1000000, 2000000, 999)'
    )

    skill = get_skill("module_loading")
    rows = skill.execute(minimal_nsys_conn)
    assert len(rows) > 0
    names = [r["api_name"] for r in rows]
    assert "cuModuleLoadData" in names

    match = next(r for r in rows if r["api_name"] == "cuModuleLoadData")
    assert match["occurrences"] == 1
    assert match["total_ms"] == 1.0


def test_module_loading_duckdb(duckdb_conn):
    """Test module_loading works on DuckDB."""
    from nsys_ai.skills.registry import get_skill

    duckdb_conn.execute("INSERT INTO StringIds VALUES (999, 'cuModuleLoadData')")
    duckdb_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (10, 1000000, 2000000, 999)'
    )

    skill = get_skill("module_loading")
    rows = skill.execute(duckdb_conn)
    names = [r["api_name"] for r in rows]
    assert "cuModuleLoadData" in names


def test_gc_impact_execute(minimal_nsys_conn):
    """Test gc_impact successfully processes both runtime and NVTX branches."""
    from nsys_ai.skills.registry import get_skill

    # Runtime branch: insert cudaFree
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (998, 'cudaFree')")
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (11, 1000000, 3000000, 998)'
    )
    # NVTX textId branch
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (997, 'GC collection generation 2')")
    minimal_nsys_conn.execute(
        'INSERT INTO NVTX_EVENTS (start, "end", text, textId, eventType) VALUES (5000000, 6000000, NULL, 997, 59)'
    )
    # NVTX text branch
    minimal_nsys_conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", text, textId, eventType) VALUES (7000000, 8000000, 'GC phase 1', NULL, 59)"
    )

    skill = get_skill("gc_impact")
    rows = skill.execute(minimal_nsys_conn)

    names = [r["event_name"] for r in rows]
    assert "cudaFree" in names
    assert "GC collection generation 2" in names
    assert "GC phase 1" in names


def test_pipeline_bubble_metrics_sqlite(minimal_nsys_conn):
    """Test pipeline bubble metrics executes on SQLite with kernel, memcpy, and memset."""
    from nsys_ai.skills.registry import get_skill

    # Add a synthetic memset interval that partially overlaps
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET (deviceId, start, "end") VALUES (0, 1500000, 2500000)'
    )

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(minimal_nsys_conn)

    assert len(rows) > 0
    r = rows[0]
    assert "deviceId" in r
    assert "total_span_ms" in r
    assert "active_ms" in r
    assert "bubble_ms" in r
    assert "bubble_pct" in r
    assert r["bubble_pct"] >= 0.0


def test_pipeline_bubble_metrics_duckdb(duckdb_conn):
    """Test pipeline bubble math executes the same logic on DuckDB."""
    from nsys_ai.skills.registry import get_skill

    duckdb_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET (deviceId, start, "end") VALUES (0, 1500000, 2500000)'
    )

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(duckdb_conn)

    assert len(rows) > 0
    assert "bubble_pct" in rows[0]


def test_pipeline_bubble_metrics_no_tables_graceful(minimal_nsys_conn):
    """Test pipeline bubble metrics falls back gracefully if activity tables are omitted."""
    from nsys_ai.skills.registry import get_skill

    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY")
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMSET")

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(minimal_nsys_conn)
    assert rows == []


def test_arithmetic_intensity_execute(minimal_nsys_conn):
    """Test arithmetic_intensity produces a roofline classification."""
    from nsys_ai.skills.registry import get_skill

    # Update the fixture GPU to A100 so we can test chip lookup
    # TARGET_INFO_GPU already has id=0 from the fixture with chipName='TestChip'
    minimal_nsys_conn.execute(
        "UPDATE TARGET_INFO_GPU SET name='NVIDIA A100-SXM4-80GB', "
        "chipName='GA100', memoryBandwidth=2039000000000 WHERE id=0"
    )

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(minimal_nsys_conn, theoretical_flops=1e15)

    assert len(rows) == 1
    r = rows[0]
    assert "error" not in r
    assert "classification" in r
    assert "achieved_tflops" in r
    assert "mfu_pct" in r
    assert r["peak_fp16_tflops"] == 312.0  # GA100 lookup


def test_arithmetic_intensity_no_gpu_table(minimal_nsys_conn):
    """Test arithmetic_intensity falls back when TARGET_INFO_GPU is missing."""
    from nsys_ai.skills.registry import get_skill

    minimal_nsys_conn.execute("DROP TABLE TARGET_INFO_GPU")
    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=1e15,
        peak_tflops=989.4,
        hbm_bw_gbps=3352,
    )

    assert len(rows) == 1
    r = rows[0]
    assert "error" not in r
    assert r["peak_fp16_tflops"] == 989.4  # User override
