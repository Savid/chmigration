#!/usr/bin/env python3
"""Generate migration SQL + summary markdown from dumped schema DDL and declarative rules."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import difflib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


@dataclasses.dataclass
class Column:
    name: str
    col_type: str
    suffix: str = ""


@dataclasses.dataclass
class SQLObject:
    kind: str
    name: str
    database: str
    object_name: str
    columns: list[Column]
    engine: str | None = None
    partition: str | None = None
    order: str | None = None
    settings: str | None = None
    comment: str | None = None
    to_table: str | None = None
    as_table: str | None = None
    as_select: str | None = None
    source_path: str | None = None

    def is_local_table(self) -> bool:
        return self.kind == "TABLE" and self.object_name.endswith("_local")

    def is_distributed_table(self) -> bool:
        return self.kind == "TABLE" and not self.object_name.endswith("_local")


MANUAL_TABLE_COMMENTS: dict[str, str] = {
    'admin.cryo': "'Tracks cryo dataset processing state per block'",
    'admin.cryo_local': "'Tracks cryo dataset processing state per block'",
    'admin.execution_block': "'Tracks execution block processing state'",
    'admin.execution_block_local': "'Tracks execution block processing state'",
    'default.beacon_api_eth_v1_beacon_blob': "'Contains beacon API blob metadata derived from block blob_kzg_commitments from each sentry client attached to a beacon node'",
    'default.beacon_api_eth_v1_beacon_blob_local': "'Contains beacon API blob metadata derived from block blob_kzg_commitments from each sentry client attached to a beacon node'",
    'default.beacon_api_eth_v1_events_attestation_local': "'Contains beacon API attestation events from each sentry client attached to a beacon node'",
    'default.beacon_api_slot_local': "'Contains beacon API slot data from each sentry client attached to a beacon node'",
    'default.imported_sources': "'This table contains the list of sources that have been imported into the database'",
    'default.imported_sources_local': "'Contains the list of sources that have been imported into the database'",
    'default.libp2p_identify': "'Contains libp2p identify protocol exchange results including remote peer agent info, supported protocols, and connection metadata'",
    'default.libp2p_identify_local': "'Contains libp2p identify protocol exchange results including remote peer agent info, supported protocols, and connection metadata'",
    'default.libp2p_peer': "'Lookup table mapping seahashed peer_id + network to original peer ID. Collected from deep instrumentation within forked consensus layer clients. Partition: monthly by `event_date_time`'",
    'default.libp2p_peer_local': "'Lookup table mapping seahashed peer_id + network to original peer ID. Collected from deep instrumentation within forked consensus layer clients. Partition: monthly by `event_date_time`'",
    'default.libp2p_recv_rpc': "'Contains RPC messages received from peers. Collected from deep instrumentation within forked consensus layer clients. Control messages are split into separate tables referencing this via rpc_meta_unique_key. Partition: monthly by `event_date_time`'",
    'default.libp2p_recv_rpc_local': "'Contains RPC messages received from peers. Collected from deep instrumentation within forked consensus layer clients. Control messages are split into separate tables referencing this via rpc_meta_unique_key. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_graft': "'Contains GRAFT control messages from gossipsub RPC. Collected from deep instrumentation within forked consensus layer clients. Peers request to join the mesh for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_graft_local': "'Contains GRAFT control messages from gossipsub RPC. Collected from deep instrumentation within forked consensus layer clients. Peers request to join the mesh for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_ihave': "'Contains IHAVE control messages from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Peers advertise message IDs they have available. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_ihave_local': "'Contains IHAVE control messages from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Peers advertise message IDs they have available. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_iwant': "'Contains IWANT control messages from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Peers request specific message IDs. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_iwant_local': "'Contains IWANT control messages from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Peers request specific message IDs. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_prune': "'Contains PRUNE control messages from gossipsub RPC. Collected from deep instrumentation within forked consensus layer clients. Peers are removed from the mesh for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_control_prune_local': "'Contains PRUNE control messages from gossipsub RPC. Collected from deep instrumentation within forked consensus layer clients. Peers are removed from the mesh for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_message': "'Contains RPC message metadata from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Each row represents a message within an RPC with topic and message ID. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_message_local': "'Contains RPC message metadata from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Each row represents a message within an RPC with topic and message ID. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_subscription': "'Contains RPC subscription changes from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Each row represents a subscribe/unsubscribe action for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_rpc_meta_subscription_local': "'Contains RPC subscription changes from gossipsub. Collected from deep instrumentation within forked consensus layer clients. Each row represents a subscribe/unsubscribe action for a topic. Partition: monthly by `event_date_time`'",
    'default.libp2p_send_rpc': "'Contains RPC messages sent to peers. Collected from deep instrumentation within forked consensus layer clients. Control messages are split into separate tables referencing this via rpc_meta_unique_key. Partition: monthly by `event_date_time`'",
    'default.libp2p_send_rpc_local': "'Contains RPC messages sent to peers. Collected from deep instrumentation within forked consensus layer clients. Control messages are split into separate tables referencing this via rpc_meta_unique_key. Partition: monthly by `event_date_time`'",
    'observoor.block_merge': "'Aggregated block device I/O merge metrics from eBPF tracing of Ethereum client processes'",
    'observoor.block_merge_local': "'Aggregated block device I/O merge metrics from eBPF tracing of Ethereum client processes'",
    'observoor.cpu_utilization': "'Aggregated CPU utilization metrics from eBPF tracing of Ethereum client processes'",
    'observoor.cpu_utilization_local': "'Aggregated CPU utilization metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_bytes': "'Aggregated disk I/O byte metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_bytes_local': "'Aggregated disk I/O byte metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_latency': "'Aggregated disk I/O latency metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_latency_local': "'Aggregated disk I/O latency metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_queue_depth': "'Aggregated disk queue depth metrics from eBPF tracing of Ethereum client processes'",
    'observoor.disk_queue_depth_local': "'Aggregated disk queue depth metrics from eBPF tracing of Ethereum client processes'",
    'observoor.fd_close': "'Aggregated file descriptor close metrics from eBPF tracing of Ethereum client processes'",
    'observoor.fd_close_local': "'Aggregated file descriptor close metrics from eBPF tracing of Ethereum client processes'",
    'observoor.fd_open': "'Aggregated file descriptor open metrics from eBPF tracing of Ethereum client processes'",
    'observoor.fd_open_local': "'Aggregated file descriptor open metrics from eBPF tracing of Ethereum client processes'",
    'observoor.host_specs': "'Periodic host hardware specification snapshots including CPU, memory, and disk details'",
    'observoor.host_specs_local': "'Periodic host hardware specification snapshots including CPU, memory, and disk details'",
    'observoor.mem_compaction': "'Aggregated memory compaction metrics from eBPF tracing of Ethereum client processes'",
    'observoor.mem_compaction_local': "'Aggregated memory compaction metrics from eBPF tracing of Ethereum client processes'",
    'observoor.mem_reclaim': "'Aggregated memory reclaim metrics from eBPF tracing of Ethereum client processes'",
    'observoor.mem_reclaim_local': "'Aggregated memory reclaim metrics from eBPF tracing of Ethereum client processes'",
    'observoor.memory_usage': "'Periodic memory usage snapshots of Ethereum client processes from /proc/[pid]/status'",
    'observoor.memory_usage_local': "'Periodic memory usage snapshots of Ethereum client processes from /proc/[pid]/status'",
    'observoor.net_io': "'Aggregated network I/O metrics from eBPF tracing of Ethereum client processes'",
    'observoor.net_io_local': "'Aggregated network I/O metrics from eBPF tracing of Ethereum client processes'",
    'observoor.oom_kill': "'Aggregated OOM kill events from eBPF tracing of Ethereum client processes'",
    'observoor.oom_kill_local': "'Aggregated OOM kill events from eBPF tracing of Ethereum client processes'",
    'observoor.page_fault_major': "'Aggregated major page fault metrics from eBPF tracing of Ethereum client processes'",
    'observoor.page_fault_major_local': "'Aggregated major page fault metrics from eBPF tracing of Ethereum client processes'",
    'observoor.page_fault_minor': "'Aggregated minor page fault metrics from eBPF tracing of Ethereum client processes'",
    'observoor.page_fault_minor_local': "'Aggregated minor page fault metrics from eBPF tracing of Ethereum client processes'",
    'observoor.process_exit': "'Aggregated process exit events from eBPF tracing of Ethereum client processes'",
    'observoor.process_exit_local': "'Aggregated process exit events from eBPF tracing of Ethereum client processes'",
    'observoor.process_fd_usage': "'Periodic file descriptor usage snapshots of Ethereum client processes from /proc/[pid]/fd and /proc/[pid]/limits'",
    'observoor.process_fd_usage_local': "'Periodic file descriptor usage snapshots of Ethereum client processes from /proc/[pid]/fd and /proc/[pid]/limits'",
    'observoor.process_io_usage': "'Periodic I/O usage snapshots of Ethereum client processes from /proc/[pid]/io'",
    'observoor.process_io_usage_local': "'Periodic I/O usage snapshots of Ethereum client processes from /proc/[pid]/io'",
    'observoor.process_sched_usage': "'Periodic scheduler usage snapshots of Ethereum client processes from /proc/[pid]/status and /proc/[pid]/sched'",
    'observoor.process_sched_usage_local': "'Periodic scheduler usage snapshots of Ethereum client processes from /proc/[pid]/status and /proc/[pid]/sched'",
    'observoor.raw_events': "'Raw eBPF events captured from Ethereum client processes, one row per kernel event'",
    'observoor.raw_events_local': "'Raw eBPF events captured from Ethereum client processes, one row per kernel event'",
    'observoor.sched_off_cpu': "'Aggregated scheduler off-CPU metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sched_off_cpu_local': "'Aggregated scheduler off-CPU metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sched_on_cpu': "'Aggregated scheduler on-CPU metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sched_on_cpu_local': "'Aggregated scheduler on-CPU metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sched_runqueue': "'Aggregated scheduler run queue metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sched_runqueue_local': "'Aggregated scheduler run queue metrics from eBPF tracing of Ethereum client processes'",
    'observoor.swap_in': "'Aggregated swap-in metrics from eBPF tracing of Ethereum client processes'",
    'observoor.swap_in_local': "'Aggregated swap-in metrics from eBPF tracing of Ethereum client processes'",
    'observoor.swap_out': "'Aggregated swap-out metrics from eBPF tracing of Ethereum client processes'",
    'observoor.swap_out_local': "'Aggregated swap-out metrics from eBPF tracing of Ethereum client processes'",
    'observoor.sync_state': "'Sync state snapshots for consensus and execution layers'",
    'observoor.sync_state_local': "'Sync state snapshots for consensus and execution layers'",
    'observoor.syscall_epoll_wait': "'Aggregated epoll_wait syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_epoll_wait_local': "'Aggregated epoll_wait syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_fdatasync': "'Aggregated fdatasync syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_fdatasync_local': "'Aggregated fdatasync syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_fsync': "'Aggregated fsync syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_fsync_local': "'Aggregated fsync syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_futex': "'Aggregated futex syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_futex_local': "'Aggregated futex syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_mmap': "'Aggregated mmap syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_mmap_local': "'Aggregated mmap syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_pwrite': "'Aggregated pwrite syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_pwrite_local': "'Aggregated pwrite syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_read': "'Aggregated read syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_read_local': "'Aggregated read syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_write': "'Aggregated write syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.syscall_write_local': "'Aggregated write syscall metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_cwnd': "'Aggregated TCP congestion window metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_cwnd_local': "'Aggregated TCP congestion window metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_retransmit': "'Aggregated TCP retransmit metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_retransmit_local': "'Aggregated TCP retransmit metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_rtt': "'Aggregated TCP round-trip time metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_rtt_local': "'Aggregated TCP round-trip time metrics from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_state_change': "'Aggregated TCP state change events from eBPF tracing of Ethereum client processes'",
    'observoor.tcp_state_change_local': "'Aggregated TCP state change events from eBPF tracing of Ethereum client processes'",
}


def apply_manual_table_comment_fallback(obj: SQLObject) -> None:
    if obj.comment:
        return
    fallback = MANUAL_TABLE_COMMENTS.get(obj.name)
    if fallback:
        obj.comment = fallback


def decode_schema_sql(raw: str) -> str:
    """Schema dumps are stored as one line with escaped newlines."""
    text = raw.replace("\\r\\n", "\\n").replace("\\n", "\n")
    return text.strip() + "\n"


def split_type_suffix(rest: str) -> tuple[str, str]:
    depth = 0
    for idx, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch.isspace() and depth == 0:
            return rest[:idx].strip(), rest[idx + 1 :].strip()
    return rest.strip(), ""


def split_sql_list(expr: str) -> list[str]:
    expr = expr.strip()
    if not expr:
        return []

    if expr.startswith("(") and expr.endswith(")") and is_balanced(expr[1:-1]):
        expr = expr[1:-1].strip()

    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            continue
        buf.append(ch)

    token = "".join(buf).strip()
    if token:
        out.append(token)
    return out


def is_balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def render_key_list(keys: list[str]) -> str:
    keys = [k.strip() for k in keys if k.strip()]
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    return "(" + ", ".join(keys) + ")"


def normalize_partition_expr(expr: str) -> str:
    expr = expr.strip()
    m = re.fullmatch(r"toStartOfMonth\((.+)\)", expr)
    if m:
        return f"toYYYYMM({m.group(1).strip()})"
    return expr


def ensure_partition_prefix(expr: str, prefix: str) -> str:
    expr = expr.strip()
    if expr == prefix:
        return expr
    if expr.startswith(f"({prefix},"):
        return expr
    return f"({prefix}, {expr})"


def extract_clause(text: str, key: str, stops: Iterable[str]) -> str | None:
    stops = list(stops)
    if not stops:
        pattern = re.compile(rf"\n{re.escape(key)}\s*(.*?)(?=\Z)", flags=re.S)
    else:
        stop_pattern = "|".join(re.escape(s) for s in stops)
        pattern = re.compile(
            rf"\n{re.escape(key)}\s*(.*?)(?=\n(?:{stop_pattern})\b|\Z)",
            flags=re.S,
        )
    m = pattern.search("\n" + text.strip())
    if not m:
        return None
    return m.group(1).strip().rstrip(";").strip()


def normalize_comment_literal(expr: str) -> str:
    expr = expr.strip().rstrip(";").strip()
    if expr.startswith("\\'") and expr.endswith("\\'") and len(expr) >= 4:
        inner = expr[2:-2].replace("\\\\", "\\").replace("\\'", "'")
        escaped = inner.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return expr


def extract_last_table_comment(text: str) -> str | None:
    matches = re.findall(r"^\s*COMMENT\s+(.+?)\s*;?\s*$", text, flags=re.M)
    if not matches:
        return None
    return normalize_comment_literal(matches[-1])


def parse_columns_block(block: str) -> list[Column]:
    columns: list[Column] = []
    for raw_line in block.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        m = re.match(r"^`([^`]+)`\s+(.+)$", line)
        if not m:
            continue
        name = m.group(1)
        col_type, suffix = split_type_suffix(m.group(2))
        columns.append(Column(name=name, col_type=col_type, suffix=suffix))
    return columns


def parse_object(text: str, source_path: Path) -> SQLObject:
    text = text.strip()

    table_match = re.match(r"^CREATE TABLE\s+([^\s]+)", text)
    mv_match = re.match(r"^CREATE MATERIALIZED VIEW\s+([^\s]+)\s+TO\s+([^\s]+)", text)

    if table_match:
        kind = "TABLE"
        full_name = table_match.group(1)
        to_table = None
    elif mv_match:
        kind = "MATERIALIZED VIEW"
        full_name = mv_match.group(1)
        to_table = mv_match.group(2)
    else:
        raise ValueError(f"Unsupported CREATE statement in {source_path}")

    if "." not in full_name:
        raise ValueError(f"Expected qualified table name in {source_path}: {full_name}")
    database, object_name = full_name.split(".", 1)

    open_idx = text.find("(")
    if open_idx < 0:
        raise ValueError(f"Missing column block in {source_path}")

    depth = 0
    close_idx = -1
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = idx
                break

    if close_idx < 0:
        raise ValueError(f"Unbalanced column block in {source_path}")

    block = text[open_idx + 1 : close_idx]
    tail = text[close_idx + 1 :].strip()

    columns = parse_columns_block(block)

    if kind == "TABLE":
        engine = extract_clause(tail, "ENGINE =", ["PARTITION BY", "ORDER BY", "SETTINGS", "COMMENT"])  # noqa: E501
        partition = extract_clause(tail, "PARTITION BY", ["ORDER BY", "SETTINGS", "COMMENT"])
        order = extract_clause(tail, "ORDER BY", ["SETTINGS", "COMMENT"])
        settings = extract_clause(tail, "SETTINGS", ["COMMENT"])
        comment = extract_last_table_comment(tail)
        as_select = None
    else:
        engine = None
        partition = None
        order = None
        settings = None
        comment = None
        as_match = re.search(r"\nAS SELECT\b.*$", "\n" + tail, flags=re.S)
        if not as_match:
            raise ValueError(f"Missing AS SELECT for materialized view: {source_path}")
        as_select = as_match.group(0).strip()

    return SQLObject(
        kind=kind,
        name=full_name,
        database=database,
        object_name=object_name,
        columns=columns,
        engine=engine,
        partition=partition,
        order=order,
        settings=settings,
        comment=comment,
        to_table=to_table,
        as_table=None,
        as_select=as_select,
        source_path=str(source_path),
    )


def load_schema_objects(schemas_dir: Path) -> dict[str, SQLObject]:
    objects: dict[str, SQLObject] = {}
    for path in sorted(schemas_dir.glob("*/*.sql")):
        decoded = decode_schema_sql(path.read_text(encoding="utf-8"))
        obj = parse_object(decoded, path)
        objects[obj.name] = obj
    return objects


def column_map(columns: list[Column]) -> dict[str, Column]:
    return {c.name: c for c in columns}


def parse_column_definition(definition: str) -> tuple[str, str]:
    return split_type_suffix(definition.strip())


def load_reference_local_columns(path: Path, tables: set[str]) -> dict[str, dict[str, Column]]:
    if not path.exists() or not tables:
        return {}

    text = path.read_text(encoding="utf-8")
    stmt_re = re.compile(
        r"^CREATE TABLE IF NOT EXISTS\s+([^\s]+)\s+ON CLUSTER\s+'[^']+'\n\((.*?)\n\)\nENGINE\s*=",
        flags=re.M | re.S,
    )

    result: dict[str, dict[str, Column]] = {}
    for match in stmt_re.finditer(text):
        table_name = match.group(1)
        if table_name not in tables:
            continue
        cols = parse_columns_block(match.group(2))
        result[table_name] = {c.name: Column(name=c.name, col_type=c.col_type, suffix=c.suffix) for c in cols}
    return result


def apply_local_definition_parity(
    obj: SQLObject,
    parity_tables: set[str],
    reference_local_columns: dict[str, dict[str, Column]],
) -> None:
    if obj.name not in parity_tables:
        return

    reference_columns = reference_local_columns.get(obj.name)
    if not reference_columns:
        raise ValueError(f"Parity table not found in reference SQL: {obj.name}")

    missing_columns = [c.name for c in obj.columns if c.name not in reference_columns]
    if missing_columns:
        raise ValueError(f"Parity table {obj.name} missing columns in reference SQL: {missing_columns}")

    for col in obj.columns:
        ref = reference_columns[col.name]
        col.col_type = ref.col_type
        col.suffix = ref.suffix


def apply_column_rules(obj: SQLObject, spec: dict) -> None:
    columns_cfg = spec.get("columns", {})

    # Global removals.
    remove_cols = set(columns_cfg.get("remove_globally", []))

    remove_by_table = columns_cfg.get("remove_by_table", {})
    remove_cols.update(remove_by_table.get(obj.name, []))

    # Conditional observoor sampling removals.
    sampling_cfg = columns_cfg.get("remove_sampling_columns", {})
    if (
        sampling_cfg
        and obj.database == sampling_cfg.get("database")
        and obj.object_name not in set(sampling_cfg.get("except_tables", []))
    ):
        remove_cols.update(sampling_cfg.get("columns", []))

    obj.columns = [c for c in obj.columns if c.name not in remove_cols]

    # Renames.
    renames = columns_cfg.get("renames", {}).get(obj.name, {})
    for col in obj.columns:
        if col.name in renames:
            col.name = renames[col.name]

    # Type overrides.
    for col in obj.columns:
        col_type_override = spec.get("types", {}).get("overrides", {}).get(obj.name, {}).get(col.name)
        if col_type_override:
            col.col_type = col_type_override

    # Added/replaced columns.
    add_or_replace = columns_cfg.get("add_or_replace", {})
    for col_name, table_defs in add_or_replace.items():
        if obj.name not in table_defs:
            continue
        col_type, suffix = parse_column_definition(table_defs[obj.name])
        cmap = column_map(obj.columns)
        if col_name in cmap:
            cmap[col_name].col_type = col_type
            cmap[col_name].suffix = suffix
            continue
        obj.columns.insert(0, Column(name=col_name, col_type=col_type, suffix=suffix))

    # Full definition overrides (type + suffix), highest precedence.
    definition_overrides = columns_cfg.get("definition_overrides", {}).get(obj.name, {})
    for col in obj.columns:
        override = definition_overrides.get(col.name)
        if not override:
            continue
        col_type, suffix = parse_column_definition(override)
        col.col_type = col_type
        col.suffix = suffix


def ensure_order_prefix(order_expr: str | None, prefix_key: str) -> str:
    keys = split_sql_list(order_expr or "")
    if prefix_key in keys:
        keys = [k for k in keys if k != prefix_key]
    keys.insert(0, prefix_key)
    return render_key_list(keys)


def transform_local_table(
    obj: SQLObject,
    spec: dict,
    reference_comments: dict[str, str],
    parity_tables: set[str],
    reference_local_columns: dict[str, dict[str, Column]],
) -> None:
    apply_column_rules(obj, spec)
    apply_local_definition_parity(obj, parity_tables, reference_local_columns)

    local_cfg = spec.get("local_rules", {})
    plain_tables = set(spec.get("engine", {}).get("local", {}).get("plain_merge_tree_tables", []))

    path_tmpl = spec.get("engine", {}).get("replicated_path_template", "")
    path = path_tmpl

    if obj.name in plain_tables:
        obj.engine = f"ReplicatedMergeTree('{path}', '{{replica}}')"
    else:
        obj.engine = f"ReplicatedReplacingMergeTree('{path}', '{{replica}}', updated_date_time)"
        if "updated_date_time" not in {c.name for c in obj.columns}:
            raise ValueError(f"{obj.name} uses ReplicatedReplacingMergeTree but has no updated_date_time")

    if not obj.comment and obj.name in reference_comments:
        obj.comment = reference_comments[obj.name]
    apply_manual_table_comment_fallback(obj)

    partition_overrides = local_cfg.get("partition", {}).get("overrides", {})
    if obj.name in partition_overrides:
        obj.partition = partition_overrides[obj.name]
    else:
        base_partition = normalize_partition_expr(obj.partition or "")
        if not base_partition:
            base_partition = "meta_network_name"
        obj.partition = ensure_partition_prefix(base_partition, "meta_network_name")

    order_cfg = local_cfg.get("order", {})
    keep_original = set(order_cfg.get("keep_original_order_tables", []))
    if obj.name not in keep_original:
        first_key_overrides = order_cfg.get("first_key_overrides", {})
        first_key = first_key_overrides.get(obj.name, "meta_network_name")
        obj.order = ensure_order_prefix(obj.order, first_key)


def parse_shard_expr(engine_expr: str | None) -> str | None:
    if not engine_expr:
        return None
    m = re.match(r"Distributed\([^,]+,[^,]+,[^,]+,\s*(.+)\)$", engine_expr.strip(), flags=re.S)
    if not m:
        return None
    return m.group(1).strip()


def load_reference_table_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}

    text = path.read_text(encoding="utf-8")
    statements: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        cur.append(line)
        if line.strip().endswith(";"):
            statements.append("\n".join(cur))
            cur = []
    if cur:
        statements.append("\n".join(cur))

    shard_exprs: dict[str, str] = {}
    table_comments: dict[str, str] = {}

    for stmt in statements:
        m = re.search(r"^CREATE TABLE IF NOT EXISTS\s+([^\s]+)", stmt, flags=re.M)
        if not m:
            continue
        table_name = m.group(1)

        engine = extract_clause(stmt, "ENGINE =", ["PARTITION BY", "ORDER BY", "SETTINGS", "COMMENT"])
        shard_expr = parse_shard_expr(engine)
        if shard_expr:
            shard_exprs[table_name] = shard_expr

        comment = extract_last_table_comment(stmt)
        if comment:
            table_comments[table_name] = comment

    return shard_exprs, table_comments


def build_cityhash(keys: list[str]) -> str:
    return "cityHash64(" + ", ".join(keys) + ")"


def transform_distributed_table(
    obj: SQLObject,
    local_map: dict[str, SQLObject],
    spec: dict,
    reference_shards: dict[str, str],
    reference_comments: dict[str, str],
    distributed_fallback_to_local_comment: bool,
) -> None:
    apply_column_rules(obj, spec)
    local_name = f"{obj.database}.{obj.object_name}_local"

    shard_cfg = spec.get("distributed_rules", {}).get("shard_key", {})
    rand_tables = set(shard_cfg.get("rand_tables", []))
    shard_overrides = shard_cfg.get("overrides", {})

    if obj.name in rand_tables:
        shard_expr = "rand()"
    elif obj.name in shard_overrides:
        shard_expr = shard_overrides[obj.name]
    elif obj.name in reference_shards:
        shard_expr = reference_shards[obj.name]
    else:
        shard_expr = parse_shard_expr(obj.engine)
        if not shard_expr:
            local_table = local_map.get(local_name)
            if not local_table:
                raise ValueError(f"Missing local table for distributed table: {obj.name}")
            shard_keys = split_sql_list(local_table.order or "")
            shard_expr = build_cityhash(shard_keys)

    obj.engine = f"Distributed('{{cluster}}', '{obj.database}', '{obj.object_name}_local', {shard_expr})"
    obj.as_table = local_name
    if not obj.comment and obj.name in reference_comments:
        obj.comment = reference_comments[obj.name]
    if not obj.comment and distributed_fallback_to_local_comment:
        local_comment = local_map.get(local_name).comment if local_name in local_map else None
        if local_comment:
            obj.comment = local_comment
    apply_manual_table_comment_fallback(obj)

    # Distributed tables should only keep engine + optional comment.
    obj.partition = None
    obj.order = None
    obj.settings = None


def transform_objects(schema_objects: dict[str, SQLObject], spec: dict) -> dict[str, SQLObject]:
    excluded = set(spec.get("exclude_objects", []))
    transformed: dict[str, SQLObject] = {}
    shard_ref_path = spec.get("distributed_rules", {}).get("shard_key", {}).get("reference_sql")
    comment_ref_path = spec.get("comments", {}).get("table_comment_fallback_sql")
    distributed_fallback_to_local_comment = bool(spec.get("comments", {}).get("distributed_fallback_to_local", False))
    parity_ref_path = spec.get("parity", {}).get("reference_sql")
    parity_tables = set(spec.get("parity", {}).get("local_definition_tables", []))

    reference_shards: dict[str, str] = {}
    reference_comments: dict[str, str] = {}
    reference_local_columns: dict[str, dict[str, Column]] = {}
    if shard_ref_path:
        ref_shards, _ = load_reference_table_metadata(Path(shard_ref_path))
        reference_shards.update(ref_shards)
    if comment_ref_path:
        _, ref_comments = load_reference_table_metadata(Path(comment_ref_path))
        reference_comments.update(ref_comments)
    if parity_ref_path and parity_tables:
        reference_local_columns = load_reference_local_columns(Path(parity_ref_path), parity_tables)

    local_tables: dict[str, SQLObject] = {}
    for name, obj in sorted(schema_objects.items()):
        if name in excluded:
            continue
        if not obj.is_local_table():
            continue
        new_obj = copy.deepcopy(obj)
        transform_local_table(new_obj, spec, reference_comments, parity_tables, reference_local_columns)
        transformed[name] = new_obj
        local_tables[name] = new_obj

    for name, obj in sorted(schema_objects.items()):
        if name in excluded:
            continue
        if not obj.is_distributed_table():
            continue
        new_obj = copy.deepcopy(obj)
        transform_distributed_table(
            new_obj,
            local_tables,
            spec,
            reference_shards,
            reference_comments,
            distributed_fallback_to_local_comment,
        )
        transformed[name] = new_obj

    for name, obj in sorted(schema_objects.items()):
        if name in excluded:
            continue
        if obj.kind != "MATERIALIZED VIEW":
            continue
        transformed[name] = copy.deepcopy(obj)

    return transformed


def db_sort_key(db: str, db_order: list[str]) -> tuple[int, str]:
    try:
        return (db_order.index(db), db)
    except ValueError:
        return (len(db_order), db)


def render_columns(columns: list[Column]) -> str:
    lines: list[str] = []
    for idx, col in enumerate(columns):
        suffix = f" {col.suffix}" if col.suffix else ""
        trailing = "," if idx < len(columns) - 1 else ""
        lines.append(f"    `{col.name}` {col.col_type}{suffix}{trailing}")
    return "\n".join(lines)


def render_table_sql(obj: SQLObject) -> str:
    assert obj.kind == "TABLE"
    if obj.is_distributed_table() and obj.as_table:
        lines: list[str] = [f"CREATE TABLE IF NOT EXISTS {obj.name} ON CLUSTER '{{cluster}}'"]
        lines.append(f"AS {obj.as_table}")
        if obj.engine:
            lines.append(f"ENGINE = {obj.engine}")
        if obj.comment:
            lines.append(f"COMMENT {obj.comment}")
        lines[-1] = lines[-1] + ";"
        return "\n".join(lines)

    lines: list[str] = [f"CREATE TABLE IF NOT EXISTS {obj.name} ON CLUSTER '{{cluster}}'", "("]
    lines.append(render_columns(obj.columns))
    lines.append(")")
    if obj.engine:
        lines.append(f"ENGINE = {obj.engine}")
    if obj.partition:
        lines.append(f"PARTITION BY {obj.partition}")
    if obj.order:
        lines.append(f"ORDER BY {obj.order}")
    if obj.comment:
        lines.append(f"COMMENT {obj.comment}")
    lines[-1] = lines[-1] + ";"
    return "\n".join(lines)


def render_mv_sql(obj: SQLObject) -> str:
    assert obj.kind == "MATERIALIZED VIEW"
    lines: list[str] = [
        f"CREATE MATERIALIZED VIEW IF NOT EXISTS {obj.name} ON CLUSTER '{{cluster}}' TO {obj.to_table}",
        "(",
        render_columns(obj.columns),
        ")",
        obj.as_select or "AS SELECT 1",
    ]
    lines[-1] = lines[-1] + ";"
    return "\n".join(lines)


def generate_sql(transformed: dict[str, SQLObject], spec: dict) -> str:
    db_order = spec.get("databases", {}).get("order", [])

    header = spec.get("rendering", {}).get("header_comment", "").strip()
    lines: list[str] = []
    if header:
        lines.append(header)
        lines.append("")

    for db in spec.get("databases", {}).get("create_if_missing", []):
        lines.append(f"CREATE DATABASE IF NOT EXISTS {db} ON CLUSTER '{{cluster}}';")
        lines.append("")

    lines.append("-- LOCAL TABLES")
    local_tables = [o for o in transformed.values() if o.is_local_table()]
    local_tables.sort(key=lambda o: (db_sort_key(o.database, db_order), o.object_name))
    grouped_local: dict[str, list[SQLObject]] = defaultdict(list)
    for obj in local_tables:
        grouped_local[obj.database].append(obj)

    for db in sorted(grouped_local, key=lambda d: db_sort_key(d, db_order)):
        lines.append(f"-- {db} database")
        for obj in grouped_local[db]:
            lines.append("")
            lines.append(render_table_sql(obj))
        lines.append("")

    lines.append("-- DISTRIBUTED TABLES")
    dist_tables = [o for o in transformed.values() if o.is_distributed_table()]
    dist_tables.sort(key=lambda o: (db_sort_key(o.database, db_order), o.object_name))
    grouped_dist: dict[str, list[SQLObject]] = defaultdict(list)
    for obj in dist_tables:
        grouped_dist[obj.database].append(obj)

    for db in sorted(grouped_dist, key=lambda d: db_sort_key(d, db_order)):
        lines.append(f"-- {db} database")
        for obj in grouped_dist[db]:
            lines.append("")
            lines.append(render_table_sql(obj))
        lines.append("")

    lines.append("-- MATERIALIZED VIEWS")
    mvs = [o for o in transformed.values() if o.kind == "MATERIALIZED VIEW"]
    mvs.sort(key=lambda o: (db_sort_key(o.database, db_order), o.object_name))
    for obj in mvs:
        lines.append("")
        lines.append(render_mv_sql(obj))

    lines.append("")
    return "\n".join(lines)


def columns_to_dict(obj: SQLObject) -> dict[str, Column]:
    return {c.name: c for c in obj.columns}


def compute_diff_stats(old: dict[str, SQLObject], new: dict[str, SQLObject]) -> dict:
    old_names = set(old.keys())
    new_names = set(new.keys())

    removed_columns: dict[str, list[str]] = defaultdict(list)
    added_columns: dict[str, list[str]] = defaultdict(list)
    type_changes: list[tuple[str, str, str, str]] = []
    order_changes: list[tuple[str, str | None, str | None]] = []

    for name in sorted(old_names & new_names):
        old_obj = old[name]
        new_obj = new[name]
        if old_obj.kind != "TABLE" or new_obj.kind != "TABLE":
            continue

        old_cols = columns_to_dict(old_obj)
        new_cols = columns_to_dict(new_obj)

        for col in sorted(old_cols.keys() - new_cols.keys()):
            removed_columns[col].append(name)
        for col in sorted(new_cols.keys() - old_cols.keys()):
            added_columns[col].append(name)
        for col in sorted(old_cols.keys() & new_cols.keys()):
            if old_cols[col].col_type != new_cols[col].col_type:
                type_changes.append((name, col, old_cols[col].col_type, new_cols[col].col_type))

        if (old_obj.order or "") != (new_obj.order or ""):
            order_changes.append((name, old_obj.order, new_obj.order))

    return {
        "missing_in_migration": sorted(old_names - new_names),
        "extra_in_migration": sorted(new_names - old_names),
        "removed_columns": removed_columns,
        "added_columns": added_columns,
        "type_changes": sorted(type_changes),
        "order_changes": order_changes,
    }


def first_order_key(expr: str | None) -> str:
    keys = split_sql_list(expr or "")
    return keys[0] if keys else ""


def partition_pattern_counts(local_tables: list[SQLObject]) -> Counter:
    counter: Counter = Counter()
    for table in local_tables:
        counter[table.partition or ""] += 1
    return counter


def shard_expr_category(expr: str | None) -> str:
    expr = (expr or "").strip()
    if expr == "rand()":
        return "rand"
    if expr.startswith("cityHash64("):
        return "cityHash64"
    return "other"


def generate_markdown(old: dict[str, SQLObject], new: dict[str, SQLObject], spec: dict) -> str:
    diff = compute_diff_stats(old, new)

    tables = [o for o in new.values() if o.kind == "TABLE"]
    locals_ = [o for o in tables if o.is_local_table()]
    dists = [o for o in tables if o.is_distributed_table()]
    mvs = [o for o in new.values() if o.kind == "MATERIALIZED VIEW"]

    db_counts: Counter = Counter(o.database for o in tables)

    remove_counts = {k: len(v) for k, v in diff["removed_columns"].items()}
    added_counts = {k: len(v) for k, v in diff["added_columns"].items()}

    replacing_local = [o for o in locals_ if (o.engine or "").startswith("ReplicatedReplacingMergeTree(")]
    plain_local = [o for o in locals_ if (o.engine or "").startswith("ReplicatedMergeTree(")]

    partition_counts = partition_pattern_counts(locals_)

    non_meta_partition = [
        o for o in locals_ if not ((o.partition or "") == "meta_network_name" or (o.partition or "").startswith("(meta_network_name,"))
    ]

    moved_meta_order = 0
    other_order_reorders: list[str] = []
    for name, old_order, new_order in diff["order_changes"]:
        if not name.endswith("_local"):
            continue
        old_first = first_order_key(old_order)
        new_first = first_order_key(new_order)
        if new_first == "meta_network_name" and old_first != "meta_network_name":
            moved_meta_order += 1
        elif new_first and new_first != old_first:
            other_order_reorders.append(name)

    old_dist_by_name = {n: o for n, o in old.items() if o.is_distributed_table()}
    rand_to_city = 0

    dist_shard_expr = {o.name: parse_shard_expr(o.engine) for o in dists}
    shard_categories = Counter(shard_expr_category(expr) for expr in dist_shard_expr.values())

    for name, old_obj in old_dist_by_name.items():
        if name not in dist_shard_expr:
            continue
        old_expr = parse_shard_expr(old_obj.engine)
        new_expr = dist_shard_expr[name]
        if shard_expr_category(old_expr) == "rand" and shard_expr_category(new_expr) == "cityHash64":
            rand_to_city += 1

    rand_tables = sorted(name for name, expr in dist_shard_expr.items() if shard_expr_category(expr) == "rand")
    other_shard_tables = sorted(name for name, expr in dist_shard_expr.items() if shard_expr_category(expr) == "other")

    local_type_changes = [
        change for change in diff["type_changes"] if change[0].endswith("_local")
    ]

    removed_version_tables = diff["removed_columns"].get("version", [])

    db_create_count = len(spec.get("databases", {}).get("create_if_missing", []))
    total_statements = len(tables) + len(mvs) + db_create_count

    lines: list[str] = []
    lines.append("# Migration 102: Schema V2")
    lines.append("")
    lines.append(
        "Migration 102 (`102_schema_v2.up.sql`) is generated from `schemas/` and `spec.yaml`. "
        "It uses `CREATE TABLE IF NOT EXISTS`, so it is a no-op on existing clusters and fully bootstraps new clusters."
    )
    lines.append("")
    lines.append(
        f"- **{total_statements} total statements**: {len(locals_)} local tables + {len(dists)} distributed tables + "
        f"{len(mvs)} materialized views + {db_create_count} database creates"
    )
    lines.append(
        "- **3 databases**: "
        f"`default` ({db_counts['default']} tables), `observoor` ({db_counts['observoor']} tables), `admin` ({db_counts['admin']} tables)"
    )
    lines.append("- **Down migration**: intentionally empty")
    lines.append("")

    lines.append("## Column removals")
    lines.append("")
    lines.append("| Column | Occurrences removed |")
    lines.append("|--------|---------------------|")
    for col in ["meta_client_id", "meta_network_id", "meta_labels"]:
        lines.append(f"| `{col}` | {remove_counts.get(col, 0)} tables |")
    lines.append("")
    lines.append("Additional removals and renames:")
    lines.append("")
    lines.append(
        f"- `version` removed from {len(removed_version_tables)} tables "
        "(`libp2p_gossipsub_*` and canonical beacon validator version-column tables)."
    )
    lines.append(
        f"- `sampling_mode` + `sampling_rate` removed from {remove_counts.get('sampling_mode', 0)} observoor tables."
    )
    lines.append(f"- `kzg_commitments` removed from {remove_counts.get('kzg_commitments', 0)} tables.")
    lines.append(f"- `erc20 -> erc721` rename applied in {added_counts.get('erc721', 0)} tables.")
    lines.append("")

    lines.append("## Engine configuration")
    lines.append("")
    lines.append(
        f"- `{len(replacing_local)}` local tables use `ReplicatedReplacingMergeTree(updated_date_time)`"
    )
    lines.append(f"- `{len(plain_local)}` local tables remain `ReplicatedMergeTree`")
    if plain_local:
        for table in sorted(plain_local, key=lambda o: o.name):
            lines.append(f"  - `{table.name}`")
    lines.append("")

    lines.append("## Partition keys")
    lines.append("")
    lines.append("| Pattern | Count |")
    lines.append("|---------|-------|")
    for pattern, count in sorted(partition_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{pattern}` | {count} |")
    lines.append("")
    if non_meta_partition:
        lines.append("Special cases without `meta_network_name` prefix:")
        for table in sorted(non_meta_partition, key=lambda o: o.name):
            lines.append(f"- `{table.name}`: `{table.partition}`")
        lines.append("")

    lines.append("## ORDER BY changes")
    lines.append("")
    lines.append(f"- `{moved_meta_order}` local tables moved `meta_network_name` to the first ORDER BY position.")
    if other_order_reorders:
        lines.append(f"- `{len(other_order_reorders)}` local tables had non-meta key reordering:")
        for name in sorted(other_order_reorders):
            lines.append(f"  - `{name}`")
    lines.append("")

    lines.append("## Distributed sharding")
    lines.append("")
    lines.append(f"- `cityHash64(...)`: {shard_categories.get('cityHash64', 0)} tables")
    lines.append(f"- `rand()`: {shard_categories.get('rand', 0)} tables")
    lines.append(f"- other expressions: {shard_categories.get('other', 0)} tables")
    lines.append(f"- rand() -> cityHash64 transitions from baseline schemas: {rand_to_city}")
    if rand_tables:
        lines.append("- Tables still using `rand()`: " + ", ".join(f"`{n}`" for n in rand_tables))
    if other_shard_tables:
        lines.append("- Tables using non-city/non-rand shard expressions: " + ", ".join(f"`{n}`" for n in other_shard_tables))
    lines.append("")

    lines.append("## Type changes")
    lines.append("")
    lines.append("| Table | Column | Old Type | New Type |")
    lines.append("|-------|--------|----------|----------|")
    for table, column, old_type, new_type in local_type_changes:
        lines.append(f"| `{table.split('.', 1)[1]}` | `{column}` | `{old_type}` | `{new_type}` |")
    lines.append("")
    lines.append("All corresponding distributed tables have the same type changes where applicable.")
    lines.append("")

    lines.append("## Materialized views")
    lines.append("")
    for mv in sorted(mvs, key=lambda o: o.name):
        lines.append(f"- `{mv.name}` -> `{mv.to_table}`")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    missing = sorted(diff["missing_in_migration"])
    excluded = sorted(spec.get("exclude_objects", []))
    if excluded:
        lines.append("Intentionally excluded from generated migration SQL:")
        for name in excluded:
            lines.append(f"- `{name}`")
        lines.append("")
    if missing:
        lines.append("Objects present in `schemas/` but excluded from generated migration:")
        for name in missing:
            lines.append(f"- `{name}`")
    else:
        lines.append("No objects are missing from generated migration output.")

    lines.append("")
    return "\n".join(lines)


def generate_outputs(schemas_dir: Path, spec_path: Path) -> tuple[str, str, dict[str, SQLObject], dict[str, SQLObject]]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    source_objects = load_schema_objects(schemas_dir)
    transformed = transform_objects(source_objects, spec)

    excluded = set(spec.get("exclude_objects", []))
    filtered_source = {k: v for k, v in source_objects.items() if k not in excluded}

    sql_output = generate_sql(transformed, spec)
    md_output = generate_markdown(filtered_source, transformed, spec)

    return sql_output, md_output, filtered_source, transformed


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def check_equal(path: Path, generated: str) -> bool:
    if not path.exists():
        print(f"[check] missing file: {path}", file=sys.stderr)
        return False

    existing = path.read_text(encoding="utf-8")
    if existing == generated:
        return True

    print(f"[check] drift detected: {path}", file=sys.stderr)
    diff = difflib.unified_diff(
        existing.splitlines(),
        generated.splitlines(),
        fromfile=str(path),
        tofile=f"generated:{path}",
        lineterm="",
    )
    for idx, line in enumerate(diff):
        if idx > 200:
            print("... (diff truncated)", file=sys.stderr)
            break
        print(line, file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate migration SQL and markdown from schemas + spec")
    parser.add_argument("--schemas-dir", default="schemas", help="Input schema directory")
    parser.add_argument("--spec", default="spec.yaml", help="Declarative YAML spec")
    parser.add_argument("--out-sql", default="migration.sql", help="Output SQL file")
    parser.add_argument("--out-md", default="migration.md", help="Output markdown file")
    parser.add_argument("--check", action="store_true", help="Verify outputs are up-to-date without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schemas_dir = Path(args.schemas_dir)
    spec_path = Path(args.spec)
    out_sql = Path(args.out_sql)
    out_md = Path(args.out_md)

    sql_output, md_output, _, _ = generate_outputs(schemas_dir, spec_path)

    if args.check:
        sql_ok = check_equal(out_sql, sql_output)
        md_ok = check_equal(out_md, md_output)
        return 0 if sql_ok and md_ok else 1

    write_text(out_sql, sql_output)
    write_text(out_md, md_output)
    print(f"Generated {out_sql}")
    print(f"Generated {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
