#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MQSim output adapter: parses an MQSim simulation result XML/log into the
KV telemetry row schema used by stage_II.features.kv.

MQSim writes a workload-statistics XML on exit. This adapter extracts throughput,
latency percentiles, write/read amplification, and GC overhead. Behavioral
KV-FTL signals (inline ratio, CMT dynamics) are NOT emitted by upstream MQSim —
they must come from a KV-SSD-patched MQSim or a companion KVPack trace.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET


def parse_mqsim_xml(path: str | Path) -> Dict[str, Any]:
    """Parse MQSim XML results. Returns a row dict or raises on fatal error."""
    p = Path(path)
    tree = ET.parse(str(p))
    root = tree.getroot()

    row: Dict[str, Any] = {"source": "mqsim", "source_path": str(p)}

    def _first_text(xpath: str) -> Optional[str]:
        node = root.find(xpath)
        return node.text.strip() if node is not None and node.text else None

    def _try_float(xpath: str, key: str):
        v = _first_text(xpath)
        if v is None:
            return
        try:
            row[key] = float(v)
        except ValueError:
            row[key] = v

    # Common MQSim fields (names vary by fork; we try a few).
    _try_float(".//Device_Throughput_KOPS", "throughput_kops")
    _try_float(".//Device_Throughput", "throughput_kops")
    _try_float(".//Average_Read_Latency", "read_latency_p50_us")
    _try_float(".//Read_Latency_P99", "read_latency_p99_us")
    _try_float(".//Read_Latency_P999", "tail_latency_p999_us")
    _try_float(".//Tail_Latency_P999", "tail_latency_p999_us")
    _try_float(".//Write_Amplification", "write_amplification")
    _try_float(".//Read_Amplification", "read_amplification")
    _try_float(".//GC_Overhead", "gc_overhead_pct")

    # Workload config echo (if present)
    _try_float(".//Average_Value_Size", "value_size_bytes")
    _try_float(".//Average_Key_Size", "key_size_bytes")
    _try_float(".//Read_Write_Ratio", "read_write_ratio")
    _try_float(".//Access_Skew", "access_skew")

    # Device identity
    for xpath, key in [
        (".//Device_Vendor", "vendor"),
        (".//Device_Model", "model"),
        (".//Firmware_Version", "firmware"),
        (".//Device_Capacity_GB", "capacity_gb"),
        (".//Interface_Type", "interface"),
    ]:
        v = _first_text(xpath)
        if v is not None:
            row[key] = v

    return row


def parse_mqsim_dir(dir_path: str | Path) -> Dict[str, Any]:
    """Scan a MQSim output directory for a results XML and parse it."""
    d = Path(dir_path)
    candidates = list(d.glob("*.xml")) + list(d.glob("**/workload_statistics*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No MQSim XML results found under {d}")
    return parse_mqsim_xml(candidates[0])
