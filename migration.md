# Migration 102: Schema V2

Migration 102 (`102_schema_v2.up.sql`) is generated from `schemas/` and `spec.yaml`. It uses `CREATE TABLE IF NOT EXISTS`, so it is a no-op on existing clusters and fully bootstraps new clusters.

- **292 total statements**: 144 local tables + 144 distributed tables + 2 materialized views + 2 database creates
- **3 databases**: `default` (208 tables), `observoor` (76 tables), `admin` (4 tables)
- **Down migration**: intentionally empty

## Column removals

| Column | Occurrences removed |
|--------|---------------------|
| `meta_client_id` | 156 tables |
| `meta_network_id` | 186 tables |
| `meta_labels` | 86 tables |

Additional removals and renames:

- `version` removed from 14 tables (`libp2p_gossipsub_*` and canonical beacon validator version-column tables).
- `sampling_mode` + `sampling_rate` removed from 70 observoor tables.
- `kzg_commitments` removed from 2 tables.
- `erc20 -> erc721` rename applied in 2 tables.

## Engine configuration

- `141` local tables use `ReplicatedReplacingMergeTree(updated_date_time)`
- `3` local tables remain `ReplicatedMergeTree`
  - `default.beacon_api_slot_local`
  - `default.imported_sources_local`
  - `observoor.raw_events_local`

## Partition keys

| Pattern | Count |
|---------|-------|
| `(meta_network_name, toYYYYMM(slot_start_date_time))` | 36 |
| `(meta_network_name, toYYYYMM(window_start))` | 35 |
| `(meta_network_name, toYYYYMM(event_date_time))` | 33 |
| `(meta_network_name, intDiv(block_number, 5000000))` | 20 |
| `meta_network_name` | 7 |
| `(meta_network_name, toYYYYMM(epoch_start_date_time))` | 3 |
| `(meta_network_name, intDiv(block_number, 201600))` | 2 |
| `(meta_network_name, toYYYYMM(event_time))` | 2 |
| `(chain_id, toYYYYMM(timestamp))` | 1 |
| `(meta_network_name, toYYYYMM(contribution_slot_start_date_time))` | 1 |
| `(meta_network_name, toYYYYMM(wallclock_epoch_start_date_time))` | 1 |
| `(meta_network_name, toYYYYMM(wallclock_slot_start_date_time))` | 1 |
| `(network, toYYYYMM(detecttime))` | 1 |
| `toStartOfMonth(create_date_time)` | 1 |

Special cases without `meta_network_name` prefix:
- `default.block_native_mempool_transaction_local`: `(network, toYYYYMM(detecttime))`
- `default.imported_sources_local`: `toStartOfMonth(create_date_time)`
- `default.mempool_dumpster_transaction_local`: `(chain_id, toYYYYMM(timestamp))`

## ORDER BY changes

- `100` local tables moved `meta_network_name` to the first ORDER BY position.
- `2` local tables had non-meta key reordering:
  - `default.block_native_mempool_transaction_local`
  - `default.mempool_dumpster_transaction_local`

## Distributed sharding

- `cityHash64(...)`: 131 tables
- `rand()`: 2 tables
- other expressions: 11 tables
- rand() -> cityHash64 transitions from baseline schemas: 3
- Tables still using `rand()`: `default.imported_sources`, `observoor.raw_events`
- Tables using non-city/non-rand shard expressions: `default.libp2p_drop_rpc`, `default.libp2p_peer`, `default.libp2p_recv_rpc`, `default.libp2p_rpc_meta_control_graft`, `default.libp2p_rpc_meta_control_idontwant`, `default.libp2p_rpc_meta_control_ihave`, `default.libp2p_rpc_meta_control_iwant`, `default.libp2p_rpc_meta_control_prune`, `default.libp2p_rpc_meta_message`, `default.libp2p_rpc_meta_subscription`, `default.libp2p_send_rpc`

## Type changes

| Table | Column | Old Type | New Type |
|-------|--------|----------|----------|
| `beacon_api_eth_v2_beacon_block_local` | `execution_payload_block_hash` | `FixedString(66)` | `Nullable(FixedString(66))` |
| `beacon_api_eth_v2_beacon_block_local` | `execution_payload_block_number` | `UInt32` | `Nullable(UInt32)` |
| `beacon_api_eth_v2_beacon_block_local` | `execution_payload_fee_recipient` | `String` | `Nullable(String)` |
| `canonical_execution_contracts_local` | `block_number` | `UInt32` | `UInt64` |
| `canonical_execution_logs_local` | `block_number` | `UInt32` | `UInt64` |
| `canonical_execution_logs_local` | `transaction_index` | `UInt32` | `UInt64` |
| `canonical_execution_storage_diffs_local` | `block_number` | `UInt32` | `UInt64` |
| `canonical_execution_storage_diffs_local` | `transaction_index` | `UInt32` | `UInt64` |
| `canonical_execution_storage_reads_local` | `block_number` | `UInt32` | `UInt64` |
| `canonical_execution_storage_reads_local` | `transaction_index` | `UInt32` | `UInt64` |
| `canonical_execution_traces_local` | `action_gas` | `UInt32` | `UInt64` |
| `canonical_execution_traces_local` | `action_value` | `String` | `UInt256` |
| `canonical_execution_traces_local` | `block_number` | `UInt32` | `UInt64` |
| `canonical_execution_traces_local` | `result_gas_used` | `UInt32` | `UInt64` |
| `canonical_execution_traces_local` | `transaction_index` | `UInt32` | `UInt64` |
| `canonical_execution_transaction_local` | `gas_price` | `UInt64` | `UInt128` |
| `canonical_execution_transaction_local` | `transaction_type` | `UInt32` | `UInt8` |
| `canonical_execution_transaction_structlog_agg_local` | `transaction_index` | `UInt32` | `UInt64` |
| `canonical_execution_transaction_structlog_local` | `transaction_index` | `UInt32` | `UInt64` |
| `execution_engine_get_blobs_local` | `duration_ms` | `UInt32` | `UInt64` |
| `execution_engine_new_payload_local` | `duration_ms` | `UInt32` | `UInt64` |
| `libp2p_connected_local` | `remote_port` | `UInt16` | `Nullable(UInt16)` |
| `libp2p_disconnected_local` | `remote_port` | `UInt16` | `Nullable(UInt16)` |
| `libp2p_handle_metadata_local` | `direction` | `LowCardinality(Nullable(String))` | `LowCardinality(String)` |
| `libp2p_handle_status_local` | `direction` | `LowCardinality(Nullable(String))` | `LowCardinality(String)` |
| `node_record_consensus_local` | `node_id` | `Nullable(String)` | `String` |

All corresponding distributed tables have the same type changes where applicable.

## Materialized views

- `default.beacon_api_slot_attestation_mv_local` -> `default.beacon_api_slot_local`
- `default.beacon_api_slot_block_mv_local` -> `default.beacon_api_slot_local`

## Coverage

Intentionally excluded from generated migration SQL:
- `default.schema_migrations`
- `default.schema_migrations_local`

No objects are missing from generated migration output.
