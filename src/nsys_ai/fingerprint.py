"""
fingerprint.py — Detects the ML framework and network topology from Nsight SQLite traces.

Extracts a ProfileFingerprint efficiently using O(1) string pool queries.
"""

import typing
from dataclasses import dataclass, field

from .connection import DB_ERRORS, wrap_connection


@dataclass
class ProfileFingerprint:
    framework: str
    distributed: bool
    multi_node: bool
    nic_summary: str = ""
    precision_notes: list[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        lines = [
            f"Framework: {self.framework}",
            f"Distributed training: {'yes' if self.distributed else 'no'}",
            f"Multi-node (RDMA): {'yes' if self.multi_node else 'no'}",
        ]
        if self.nic_summary:
            import re

            safe_nic = re.sub(r"[^\w\s\-.,()]", "", str(self.nic_summary)).strip()[:200]
            lines.append(f"Network: {safe_nic}")
        if self.precision_notes:
            lines.append("Notes: " + "; ".join(self.precision_notes))
        return "\n".join(lines)


# Ranked by priority / specificity.
# An environment matching multiple (e.g. Megatron and PyTorch) resolves to the first match.
FRAMEWORK_PRIORITY = [
    ("vLLM", ["paged_attention", "vllm", "SamplerOutput", "ModelRunner"]),
    ("SGLang", ["sglang", "RadixAttention", "TokenAttention"]),
    ("Megatron-LM", ["Megatron", "p2p_comm", "FlushGroups", "MegatronModule"]),
    ("DeepSpeed", ["DeepSpeed", "ZeRO", "offload", "DeepSpeedEngine"]),
    ("PyTorch", ["forward", "backward", "optimizer_step", "flash_attn"]),
]

_LOWERCASE_FRAMEWORK_PRIORITY = [
    (fw, [kw.lower() for kw in keywords]) for fw, keywords in FRAMEWORK_PRIORITY
]

# Known high-performance interconnect vendors
KNOWN_NIC_VENDORS = {
    5555: "Mellanox / NVIDIA",
    5348: "Broadcom",
    6082: "Cray",
    32902: "Intel",
}


def get_fingerprint(conn: typing.Any) -> ProfileFingerprint:
    adapter = wrap_connection(conn)
    tables = adapter.get_table_names()

    # Step A: O(1) String Search via Chunked Python Sweeps
    # Escapes SQLite AST depth limits (large OR chains) and magic LIMIT caps
    found_frameworks: set[str] = set()

    def _scan_table_strings(query: str):
        try:
            cur = adapter.execute(query)
            while True:
                rows = cur.fetchmany(10000)
                if not rows:
                    break
                for row in rows:
                    val_lower = str(row[0]).lower()
                    for fw, lower_keywords in _LOWERCASE_FRAMEWORK_PRIORITY:
                        if fw not in found_frameworks:
                            if any(kw in val_lower for kw in lower_keywords):
                                found_frameworks.add(fw)
                # If we've found the highest-possible priority framework, short-circuit
                if FRAMEWORK_PRIORITY[0][0] in found_frameworks:
                    break
        except DB_ERRORS:
            pass

    if "StringIds" in tables:
        _scan_table_strings("SELECT value FROM StringIds WHERE value IS NOT NULL")

    if not found_frameworks and "NVTX_EVENTS" in tables:
        _scan_table_strings(
            "SELECT DISTINCT text FROM NVTX_EVENTS WHERE text IS NOT NULL LIMIT 1000"
        )

    # Match framework with strict explicit prioritization
    framework = "Generic CUDA"
    for fw, _ in FRAMEWORK_PRIORITY:
        if fw in found_frameworks:
            framework = fw
            break

    # Step B: Topology Search
    multi_node = False
    nic_summary = ""
    if "TARGET_INFO_NIC_INFO" in tables:
        try:
            vendor_keys = ",".join(map(str, KNOWN_NIC_VENDORS.keys()))
            cur = adapter.execute(
                f"SELECT vendorId, name FROM TARGET_INFO_NIC_INFO "
                f"WHERE CAST(vendorId AS INTEGER) IN ({vendor_keys}) "
                f"OR name LIKE 'mlx5_%' OR name LIKE 'cxi%' LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                multi_node = True
                v_id = -1
                try:
                    v_id = int(row[0])
                except (ValueError, TypeError):
                    pass
                vendor_name = KNOWN_NIC_VENDORS.get(v_id, "NIC")
                nic_summary = (
                    f"{vendor_name} hardware detected (vendorId: {row[0]}, name: {row[1]})"
                )
        except DB_ERRORS:
            pass

    distributed = False
    if "NVTX_PAYLOAD_SCHEMAS" in tables:
        try:
            cur = adapter.execute(
                "SELECT 1 FROM NVTX_PAYLOAD_SCHEMAS WHERE name LIKE '%NCCL%' LIMIT 1"
            )
            if cur.fetchone():
                distributed = True
        except DB_ERRORS:
            pass

    return ProfileFingerprint(
        framework=framework,
        distributed=distributed,
        multi_node=multi_node,
        nic_summary=nic_summary,
        precision_notes=[],
    )
