"""Optional CUDA Graph schema support.

Nsight Systems adds CUDA Graph columns and tables independently of the core
kernel activity schema.  Keep the probes in one small module so SQLite,
DuckDB, and the Parquet cache agree on what "graph-aware" means without
turning graph tracing into a hard requirement for older reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .connection import resolve_table_variant

GRAPH_TABLE_NAMES = (
    "CUDA_GRAPH_EVENTS",
    "CUDA_GRAPH_NODE_EVENTS",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
)

# Canonical output names used by nsys-ai.  The values are the names NVIDIA
# uses in the kernel activity export.  Keeping the mapping explicit also lets
# us tolerate harmless case changes in future exports.
GRAPH_KERNEL_COLUMNS = {
    "graph_node_id": "graphNodeId",
    "graph_id": "graphId",
}


def actual_table(tables: Iterable[str], name: str) -> str | None:
    """Return the table matching *name*, including a ``_V2``/``_V3`` variant.

    An exact comparison dropped the versioned names Nsight actually writes on
    newer exports, so a profile carrying ``CUPTI_ACTIVITY_KIND_GRAPH_TRACE_V3``
    reported no graph metadata at all. Resolution goes through the shared
    ``resolve_table_variant``, which is what the repository's Nsight table-name
    contract requires and which already knows that ``_V10`` beats ``_V3``.

    The case-insensitive exact match is kept as the first step, because the
    shared resolver compares exactly and some callers hand in names whose case
    came from a different export.
    """
    wanted = name.lower()
    exact = next((table for table in tables if str(table).lower() == wanted), None)
    if exact is not None:
        return exact
    return resolve_table_variant({str(t) for t in tables}, name)


def actual_columns(columns: Iterable[str]) -> dict[str, str]:
    """Return lower-case column names mapped to their actual spelling."""
    return {str(column).lower(): str(column) for column in columns}


def graph_tables(tables: Iterable[str]) -> dict[str, str]:
    """Resolve the optional graph tables present in an export."""
    return {
        name: table
        for name in GRAPH_TABLE_NAMES
        if (table := actual_table(tables, name)) is not None
    }


def kernel_graph_columns(columns: Iterable[str]) -> dict[str, str]:
    """Resolve graph columns on a kernel table to canonical output names."""
    by_lower = actual_columns(columns)
    resolved: dict[str, str] = {}
    for canonical, source in GRAPH_KERNEL_COLUMNS.items():
        # Accept both NVIDIA's camelCase export and the normalized names used
        # by the Parquet cache/raw parquetdir adapter.
        for candidate in (source, canonical):
            if candidate.lower() in by_lower:
                resolved[canonical] = by_lower[candidate.lower()]
                break
    return resolved


def graph_capability(
    tables: Iterable[str],
    kernel_columns: Mapping[str, str],
    *,
    kernel_rows_present: bool | None = None,
) -> dict[str, object]:
    """Describe optional graph support without claiming more than the data.

    ``kernel_attribution`` is true only when kernel rows carry a graph ID. A
    graph metadata table alone is useful for diagnostics, but it cannot be
    joined to kernel timing without inventing an attribution.

    The columns alone are not evidence of that, which is what this used to treat
    them as. The Parquet cache deliberately synthesises ``NULL AS graph_node_id``
    for a capture that has none, so downstream SQL does not have to branch — so
    once cached, a pre-graph profile carries the columns and reported attribution
    as available. The same profile answered differently before and after its
    cache was built.

    Nor can the spelling settle it: the cache normalises real graph data to the
    same snake_case names it synthesises, so ``graph_node_id`` means both things.

    ``kernel_rows_present`` carries the only thing that does settle it — whether
    a row actually holds a graph id. ``None`` means it could not be established,
    and attribution is not claimed on a maybe.
    """
    resolved_tables = graph_tables(tables)
    attribution = bool(kernel_columns) and kernel_rows_present is True
    return {
        "kernel_attribution": attribution,
        "kernel_columns": dict(kernel_columns),
        "kernel_rows_present": kernel_rows_present,
        "tables": resolved_tables,
        "available": bool(kernel_columns or resolved_tables),
    }
