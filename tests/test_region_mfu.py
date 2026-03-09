import sqlite3

from nsys_ai.region_mfu import (
    compute_mfu_metrics_for_region,
    compute_region_mfu_from_conn,
    find_nvtx_ranges,
    get_region_kernels,
    select_nvtx_occurrence,
    summarize_region_kernel_times,
)


def _make_min_region_db(path: str):
    """Create a minimal DB with NVTX + RUNTIME + KERNEL for region_mfu tests."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INT, correlationId INT, start INT, [end] INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    # GPU tables for peak TFLOPS lookup
    conn.execute(
        "CREATE TABLE TARGET_INFO_GPU(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT, "
        "totalMemory INTEGER, smCount INTEGER, chipName TEXT, memoryBandwidth INTEGER)"
    )
    conn.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT, numMultiprocessors INTEGER)"
    )
    conn.execute("INSERT INTO TARGET_INFO_GPU(id, name) VALUES (0, 'NVIDIA H100 80GB HBM3')")
    conn.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE(gpuId, cudaId) VALUES (0, 0)")

    conn.execute("INSERT INTO StringIds(id, value) VALUES (1,'k_flash'), (2,'k_flash_dem')")
    # One kernel 1000–2000 ns, correlationId 1 on device 0 / stream 7
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES (1000, 2000, 0, 7, 1, 1, 2)"
    )
    # NVTX range 500–2500 containing the runtime launch
    conn.execute(
        "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES ('FlashAttention', 1, 500, 2500)"
    )
    # Runtime 900–1000 so kernel 1000–2000 is inside the NVTX CPU span
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) "
        "VALUES (1, 1, 900, 1000)"
    )
    conn.commit()
    conn.close()


def test_find_nvtx_ranges_and_select_occurrence(tmp_path):
    db = tmp_path / "nvtx.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        rows = find_nvtx_ranges(conn, "FlashAttention", match_mode="contains")
        assert rows
        chosen = select_nvtx_occurrence(rows, 1)
        assert "error" not in chosen
        assert chosen["text"] == "FlashAttention"
        assert chosen["occurrence_index"] == 1
    finally:
        conn.close()


def test_get_region_kernels_and_summarize(tmp_path):
    db = tmp_path / "kernels.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        nvtx_rows = find_nvtx_ranges(conn, "FlashAttention", match_mode="exact")
        chosen = select_nvtx_occurrence(nvtx_rows, 1)
        assert "error" not in chosen
        kernels = get_region_kernels(
            conn,
            nvtx_start_ns=chosen["start_ns"],
            nvtx_end_ns=chosen["end_ns"],
            global_tid=chosen.get("global_tid"),
            device_id=0,
        )
        assert len(kernels) == 1
        summary = summarize_region_kernel_times(kernels)
        assert summary["kernel_count"] == 1
        assert summary["kernel_sum_ns"] == 1000
        assert summary["kernel_union_ns"] == 1000
        assert summary["device_ids"] == [0]
        assert summary["stream_ids"] == [7]
    finally:
        conn.close()


def test_compute_mfu_metrics_for_region():
    out = compute_mfu_metrics_for_region(
        theoretical_flops=1e18,
        peak_tflops=989.0,
        wall_time_s=10.0,
        kernel_sum_s=10.0,
        kernel_union_s=10.0,
    )
    assert "error" not in out
    assert out["mfu_pct_wall"] == round(100.0 * ((1e18 / 10.0) / 1e12) / 989.0, 2)


def test_compute_region_mfu_from_conn_happy_path(tmp_path):
    db = tmp_path / "region.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        result = compute_region_mfu_from_conn(
            conn,
            str(db),
            "FlashAttention",
            theoretical_flops=1e18,
            peak_tflops=None,
            occurrence_index=1,
            device_id=0,
            match_mode="contains",
        )
        assert "error" not in result
        assert result["nvtx_name"] == "FlashAttention"
        assert result["matched_text"] == "FlashAttention"
        assert result["kernel_count"] == 1
        assert result["wall_time_s"] > 0
        assert "mfu_pct_wall" in result
        assert "mfu_pct_kernel_union" in result
    finally:
        conn.close()

