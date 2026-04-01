# ORDER BY Comparison: Old vs New

## admin

| Table | Old ORDER BY | New ORDER BY | Changed |
|-------|-------------|-------------|---------|
| cryo | `(dataset, mode, meta_network_name)` | `(dataset, mode, meta_network_name)` |  |
| execution_block | `(block_number, processor, meta_network_name)` | `(block_number, processor, meta_network_name)` |  |

## default

| Table | Old ORDER BY | New ORDER BY | Changed |
|-------|-------------|-------------|---------|
| beacon_api_eth_v1_beacon_blob | `(slot_start_date_time, meta_network_name, meta_client_name, block_root, blob_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_root, blob_index)` | YES |
| beacon_api_eth_v1_beacon_committee | `(slot_start_date_time, meta_network_name, meta_client_name, committee_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, committee_index)` | YES |
| beacon_api_eth_v1_events_attestation | `(slot_start_date_time, meta_network_name, meta_client_name)` | `(meta_network_name, slot_start_date_time, meta_client_name, attesting_validator_index, attesting_validator_committee_index, aggregation_bits, beacon_block_root)` | YES |
| beacon_api_eth_v1_events_blob_sidecar | `(slot_start_date_time, meta_network_name, meta_client_name, block_root, blob_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_root, blob_index)` | YES |
| beacon_api_eth_v1_events_block_gossip | `(slot_start_date_time, meta_network_name, meta_client_name, block)` | `(meta_network_name, slot_start_date_time, meta_client_name, block)` | YES |
| beacon_api_eth_v1_events_block | `(slot_start_date_time, meta_network_name, meta_client_name, block)` | `(meta_network_name, slot_start_date_time, meta_client_name, block)` | YES |
| beacon_api_eth_v1_events_chain_reorg | `(slot_start_date_time, meta_network_name, meta_client_name, old_head_block, new_head_block)` | `(meta_network_name, slot_start_date_time, meta_client_name, old_head_block, new_head_block)` | YES |
| beacon_api_eth_v1_events_contribution_and_proof | `(contribution_slot_start_date_time, meta_network_name, meta_client_name, contribution_beacon_block_root, contribution_subcommittee_index, signature)` | `(meta_network_name, contribution_slot_start_date_time, meta_client_name, contribution_beacon_block_root, contribution_subcommittee_index, signature)` | YES |
| beacon_api_eth_v1_events_data_column_sidecar | `(slot_start_date_time, meta_network_name, meta_client_name, block_root, column_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_root, column_index)` | YES |
| beacon_api_eth_v1_events_finalized_checkpoint | `(epoch_start_date_time, meta_network_name, meta_client_name, block, state)` | `(meta_network_name, epoch_start_date_time, meta_client_name, block, state)` | YES |
| beacon_api_eth_v1_events_head | `(slot_start_date_time, meta_network_name, meta_client_name, block, previous_duty_dependent_root, current_duty_dependent_root)` | `(meta_network_name, slot_start_date_time, meta_client_name, block, previous_duty_dependent_root, current_duty_dependent_root)` | YES |
| beacon_api_eth_v1_events_voluntary_exit | `(wallclock_epoch_start_date_time, meta_network_name, meta_client_name, validator_index)` | `(meta_network_name, wallclock_epoch_start_date_time, meta_client_name, validator_index)` | YES |
| beacon_api_eth_v1_proposer_duty | `(slot_start_date_time, meta_network_name, meta_client_name, proposer_validator_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, proposer_validator_index)` | YES |
| beacon_api_eth_v1_validator_attestation_data | `(slot_start_date_time, meta_network_name, meta_client_name, committee_index, beacon_block_root, source_root, target_root)` | `(meta_network_name, slot_start_date_time, meta_client_name, committee_index, beacon_block_root, source_root, target_root)` | YES |
| beacon_api_eth_v2_beacon_block | `(slot_start_date_time, meta_network_name, meta_client_name, block_root, parent_root, state_root)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_root, parent_root, state_root)` | YES |
| beacon_api_eth_v3_validator_block | `(slot_start_date_time, meta_network_name, meta_client_name, event_date_time)` | `(meta_network_name, slot_start_date_time, meta_client_name, event_date_time)` | YES |
| beacon_api_slot_attestation_mv | `(none)` | `—` |  |
| beacon_api_slot_block_mv | `(none)` | `—` |  |
| beacon_api_slot | `(slot_start_date_time, slot, meta_network_name)` | `(meta_network_name, slot_start_date_time, slot)` | YES |
| beacon_block_classification | `(slot_start_date_time, meta_network_name, meta_client_name, proposer_index)` | `(meta_network_name, slot_start_date_time, meta_client_name, proposer_index)` | YES |
| blob_submitter | `(address, meta_network_name)` | `(meta_network_name, address)` | YES |
| block_native_mempool_transaction | `(detecttime, network, hash, fromaddress, nonce, gas)` | `(network, detecttime, hash, fromaddress, nonce, gas)` | YES |
| canonical_beacon_blob_sidecar | `(slot_start_date_time, meta_network_name, block_root, blob_index)` | `(meta_network_name, slot_start_date_time, block_root, blob_index)` | YES |
| canonical_beacon_block_attester_slashing | `(slot_start_date_time, meta_network_name, block_root, attestation_1_attesting_indices, attestation_2_attesting_indices, attestation_1_data_slot, attestation_2_data_slot, attestation_1_data_beacon_block_root, attestation_2_data_beacon_block_root)` | `(meta_network_name, slot_start_date_time, block_root, attestation_1_attesting_indices, attestation_2_attesting_indices, attestation_1_data_slot, attestation_2_data_slot, attestation_1_data_beacon_block_root, attestation_2_data_beacon_block_root)` | YES |
| canonical_beacon_block_bls_to_execution_change | `(slot_start_date_time, meta_network_name, block_root, exchanging_message_validator_index, exchanging_message_from_bls_pubkey, exchanging_message_to_execution_address)` | `(meta_network_name, slot_start_date_time, block_root, exchanging_message_validator_index, exchanging_message_from_bls_pubkey, exchanging_message_to_execution_address)` | YES |
| canonical_beacon_block_deposit | `(slot_start_date_time, meta_network_name, block_root, deposit_data_pubkey, deposit_proof)` | `(meta_network_name, slot_start_date_time, block_root, deposit_data_pubkey, deposit_proof)` | YES |
| canonical_beacon_block_execution_transaction | `(slot_start_date_time, meta_network_name, block_root, position, hash, nonce)` | `(meta_network_name, slot_start_date_time, block_root, position, hash, nonce)` | YES |
| canonical_beacon_block | `(slot_start_date_time, meta_network_name)` | `(meta_network_name, slot_start_date_time)` | YES |
| canonical_beacon_block_proposer_slashing | `(slot_start_date_time, meta_network_name, block_root, signed_header_1_message_slot, signed_header_2_message_slot, signed_header_1_message_proposer_index, signed_header_2_message_proposer_index, signed_header_1_message_body_root, signed_header_2_message_body_root)` | `(meta_network_name, slot_start_date_time, block_root, signed_header_1_message_slot, signed_header_2_message_slot, signed_header_1_message_proposer_index, signed_header_2_message_proposer_index, signed_header_1_message_body_root, signed_header_2_message_body_root)` | YES |
| canonical_beacon_block_sync_aggregate | `(slot_start_date_time, meta_network_name, slot)` | `(meta_network_name, slot_start_date_time, slot)` | YES |
| canonical_beacon_block_voluntary_exit | `(slot_start_date_time, meta_network_name, block_root, voluntary_exit_message_epoch, voluntary_exit_message_validator_index)` | `(meta_network_name, slot_start_date_time, block_root, voluntary_exit_message_epoch, voluntary_exit_message_validator_index)` | YES |
| canonical_beacon_block_withdrawal | `(slot_start_date_time, meta_network_name, block_root, withdrawal_index, withdrawal_validator_index)` | `(meta_network_name, slot_start_date_time, block_root, withdrawal_index, withdrawal_validator_index)` | YES |
| canonical_beacon_committee | `(slot_start_date_time, meta_network_name, committee_index)` | `(meta_network_name, slot_start_date_time, committee_index)` | YES |
| canonical_beacon_elaborated_attestation | `(slot_start_date_time, meta_network_name, block_root, block_slot, position_in_block, beacon_block_root, slot, committee_index, source_root, target_root)` | `(meta_network_name, slot_start_date_time, block_root, block_slot, position_in_block, beacon_block_root, slot, committee_index, source_root, target_root)` | YES |
| canonical_beacon_proposer_duty | `(slot_start_date_time, meta_network_name, proposer_validator_index, proposer_pubkey)` | `(meta_network_name, slot_start_date_time, proposer_validator_index, proposer_pubkey)` | YES |
| canonical_beacon_sync_committee | `(epoch_start_date_time, meta_network_name, sync_committee_period)` | `(meta_network_name, epoch_start_date_time, sync_committee_period)` | YES |
| canonical_beacon_validators | `(epoch_start_date_time, meta_network_name, index, status)` | `(meta_network_name, epoch_start_date_time, index, status)` | YES |
| canonical_beacon_validators_pubkeys | `(index, pubkey, meta_network_name)` | `(meta_network_name, index, pubkey)` | YES |
| canonical_beacon_validators_withdrawal_credentials | `(index, withdrawal_credentials, meta_network_name)` | `(meta_network_name, index, withdrawal_credentials)` | YES |
| canonical_execution_address_appearances | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_balance_diffs | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_balance_reads | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_block | `(block_number, meta_network_name)` | `(meta_network_name, block_number)` | YES |
| canonical_execution_contracts | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_erc20_transfers | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_erc721_transfers | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_four_byte_counts | `(block_number, meta_network_name, transaction_hash)` | `(meta_network_name, block_number, transaction_hash)` | YES |
| canonical_execution_logs | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_native_transfers | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_nonce_diffs | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_nonce_reads | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_storage_diffs | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_storage_reads | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_traces | `(block_number, meta_network_name, transaction_hash, internal_index)` | `(meta_network_name, block_number, transaction_hash, internal_index)` | YES |
| canonical_execution_transaction | `(block_number, meta_network_name, transaction_hash)` | `(meta_network_name, block_number, transaction_hash)` | YES |
| canonical_execution_transaction_structlog_agg | `(block_number, meta_network_name, transaction_hash, call_frame_id, operation)` | `(meta_network_name, block_number, transaction_hash, call_frame_id, operation)` | YES |
| canonical_execution_transaction_structlog | `(block_number, meta_network_name, transaction_hash, index)` | `(meta_network_name, block_number, transaction_hash, index)` | YES |
| consensus_engine_api_get_blobs | `(slot_start_date_time, meta_network_name, meta_client_name, block_root, event_date_time)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_root, event_date_time)` | YES |
| consensus_engine_api_new_payload | `(slot_start_date_time, meta_network_name, meta_client_name, block_hash, event_date_time)` | `(meta_network_name, slot_start_date_time, meta_client_name, block_hash, event_date_time)` | YES |
| ethseer_validator_entity | `(index, pubkey, meta_network_name)` | `(meta_network_name, index, pubkey)` | YES |
| execution_block_metrics | `(block_number, meta_network_name, meta_client_name, event_date_time)` | `(meta_network_name, block_number, meta_client_name, event_date_time)` | YES |
| execution_engine_get_blobs | `(event_date_time, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, meta_client_name)` | YES |
| execution_engine_new_payload | `(block_number, meta_network_name, meta_client_name, block_hash, event_date_time)` | `(meta_network_name, block_number, meta_client_name, block_hash, event_date_time)` | YES |
| execution_state_size | `(block_number, meta_network_name, meta_client_name, state_root, event_date_time)` | `(meta_network_name, block_number, meta_client_name, state_root, event_date_time)` | YES |
| execution_transaction | `(block_number, meta_network_name, block_hash, position)` | `(meta_network_name, block_number, block_hash, position)` | YES |
| imported_sources | `(create_date_time, source)` | `(create_date_time, source)` |  |
| libp2p_add_peer | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key)` | YES |
| libp2p_connected | `(event_date_time, meta_network_name, meta_client_name, remote_peer_id_unique_key, direction, opened)` | `(meta_network_name, event_date_time, meta_client_name, remote_peer_id_unique_key, direction, opened)` | YES |
| libp2p_deliver_message | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | YES |
| libp2p_disconnected | `(event_date_time, meta_network_name, meta_client_name, remote_peer_id_unique_key, direction, opened)` | `(meta_network_name, event_date_time, meta_client_name, remote_peer_id_unique_key, direction, opened)` | YES |
| libp2p_drop_rpc | `(event_date_time, unique_key, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, meta_client_name)` | YES |
| libp2p_duplicate_message | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | YES |
| libp2p_gossipsub_aggregate_and_proof | `(slot_start_date_time, meta_network_name, meta_client_name, peer_id_unique_key, message_id)` | `(meta_network_name, slot_start_date_time, meta_client_name, peer_id_unique_key, message_id)` | YES |
| libp2p_gossipsub_beacon_attestation | `(slot_start_date_time, meta_network_name, meta_client_name, peer_id_unique_key, message_id)` | `(meta_network_name, slot_start_date_time, meta_client_name, peer_id_unique_key, message_id)` | YES |
| libp2p_gossipsub_beacon_block | `(slot_start_date_time, meta_network_name, meta_client_name, peer_id_unique_key, message_id)` | `(meta_network_name, slot_start_date_time, meta_client_name, peer_id_unique_key, message_id)` | YES |
| libp2p_gossipsub_blob_sidecar | `(slot_start_date_time, meta_network_name, meta_client_name, peer_id_unique_key, message_id)` | `(meta_network_name, slot_start_date_time, meta_client_name, peer_id_unique_key, message_id)` | YES |
| libp2p_gossipsub_data_column_sidecar | `(slot_start_date_time, meta_network_name, meta_client_name, peer_id_unique_key, message_id)` | `(meta_network_name, slot_start_date_time, meta_client_name, peer_id_unique_key, message_id)` | YES |
| libp2p_graft | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | YES |
| libp2p_handle_metadata | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, attnets, seq_number, syncnets, latency_milliseconds)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, attnets, seq_number, syncnets, latency_milliseconds)` | YES |
| libp2p_handle_status | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, latency_milliseconds)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, latency_milliseconds)` | YES |
| libp2p_identify | `(event_date_time, meta_network_name, meta_client_name, remote_peer_id_unique_key, direction)` | `(meta_network_name, event_date_time, meta_client_name, remote_peer_id_unique_key, direction)` | YES |
| libp2p_join | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | YES |
| libp2p_leave | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | YES |
| libp2p_peer | `unique_key` | `unique_key` |  |
| libp2p_prune | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name)` | YES |
| libp2p_publish_message | `(event_date_time, meta_network_name, meta_client_name, topic_fork_digest_value, topic_name, message_id)` | `(meta_network_name, event_date_time, meta_client_name, topic_fork_digest_value, topic_name, message_id)` | YES |
| libp2p_recv_rpc | `(event_date_time, unique_key, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, meta_client_name)` | YES |
| libp2p_reject_message | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, topic_fork_digest_value, topic_name, message_id, seq_number)` | YES |
| libp2p_remove_peer | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key)` | YES |
| libp2p_rpc_data_column_custody_probe | `(event_date_time, meta_network_name, meta_client_name, peer_id_unique_key, slot, column_index)` | `(meta_network_name, event_date_time, meta_client_name, peer_id_unique_key, slot, column_index)` | YES |
| libp2p_rpc_meta_control_graft | `(event_date_time, unique_key, control_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, meta_client_name)` | YES |
| libp2p_rpc_meta_control_idontwant | `(event_date_time, unique_key, control_index, peer_id_unique_key, message_id, message_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, peer_id_unique_key, message_id, message_index, meta_client_name)` | YES |
| libp2p_rpc_meta_control_ihave | `(event_date_time, unique_key, control_index, message_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, message_index, meta_client_name)` | YES |
| libp2p_rpc_meta_control_iwant | `(event_date_time, unique_key, control_index, message_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, message_index, meta_client_name)` | YES |
| libp2p_rpc_meta_control_prune | `(event_date_time, unique_key, control_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, meta_client_name)` | YES |
| libp2p_rpc_meta_message | `(event_date_time, unique_key, control_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, meta_client_name)` | YES |
| libp2p_rpc_meta_subscription | `(event_date_time, unique_key, control_index, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, control_index, meta_client_name)` | YES |
| libp2p_send_rpc | `(event_date_time, unique_key, meta_network_name, meta_client_name)` | `(meta_network_name, event_date_time, unique_key, meta_client_name)` | YES |
| libp2p_synthetic_heartbeat | `(event_date_time, meta_network_name, meta_client_name, remote_peer_id_unique_key, updated_date_time)` | `(meta_network_name, event_date_time, meta_client_name, remote_peer_id_unique_key, updated_date_time)` | YES |
| mempool_dumpster_transaction | `(timestamp, chain_id, hash, from, nonce, gas)` | `(chain_id, timestamp, hash, from, nonce, gas)` | YES |
| mempool_transaction | `(event_date_time, meta_network_name, meta_client_name, hash, from, nonce, gas)` | `(meta_network_name, event_date_time, meta_client_name, hash, from, nonce, gas)` | YES |
| mev_relay_bid_trace | `(slot_start_date_time, meta_network_name, relay_name, block_hash, meta_client_name, builder_pubkey, proposer_pubkey)` | `(meta_network_name, slot_start_date_time, relay_name, block_hash, meta_client_name, builder_pubkey, proposer_pubkey)` | YES |
| mev_relay_proposer_payload_delivered | `(slot_start_date_time, meta_network_name, relay_name, block_hash, meta_client_name, builder_pubkey, proposer_pubkey)` | `(meta_network_name, slot_start_date_time, relay_name, block_hash, meta_client_name, builder_pubkey, proposer_pubkey)` | YES |
| mev_relay_validator_registration | `(event_date_time, meta_network_name, meta_client_name, relay_name, validator_index, timestamp)` | `(meta_network_name, event_date_time, meta_client_name, relay_name, validator_index, timestamp)` | YES |
| node_record_consensus | `(event_date_time, meta_network_name, enr, meta_client_name)` | `(meta_network_name, event_date_time, enr, meta_client_name)` | YES |
| node_record_execution | `(event_date_time, meta_network_name, node_id, meta_client_name)` | `(meta_network_name, event_date_time, node_id, meta_client_name)` | YES |
| schema_migrations | `sequence` | `—` |  |

## observoor

| Table | Old ORDER BY | New ORDER BY | Changed |
|-------|-------------|-------------|---------|
| block_merge | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` |  |
| cpu_utilization | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| disk_bytes | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` |  |
| disk_latency | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` |  |
| disk_queue_depth | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, device_id, rw)` |  |
| fd_close | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| fd_open | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| host_specs | `(meta_network_name, event_time, host_id, meta_client_name)` | `(meta_network_name, event_time, host_id, meta_client_name)` |  |
| mem_compaction | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| mem_reclaim | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| memory_usage | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| net_io | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label, direction)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label, direction)` |  |
| oom_kill | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| page_fault_major | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| page_fault_minor | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| process_exit | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| process_fd_usage | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| process_io_usage | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| process_sched_usage | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| raw_events | `(meta_network_name, wallclock_slot_start_date_time, client_type, event_type, pid)` | `(meta_network_name, wallclock_slot_start_date_time, client_type, event_type, pid)` |  |
| sched_off_cpu | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| sched_on_cpu | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| sched_runqueue | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| swap_in | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| swap_out | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| sync_state | `(meta_network_name, event_time, meta_client_name)` | `(meta_network_name, event_time, meta_client_name)` |  |
| syscall_epoll_wait | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_fdatasync | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_fsync | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_futex | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_mmap | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_pwrite | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_read | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| syscall_write | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |
| tcp_cwnd | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label)` |  |
| tcp_retransmit | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label, direction)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label, direction)` |  |
| tcp_rtt | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label)` | `(meta_network_name, window_start, meta_client_name, pid, client_type, port_label)` |  |
| tcp_state_change | `(meta_network_name, window_start, meta_client_name, pid, client_type)` | `(meta_network_name, window_start, meta_client_name, pid, client_type)` |  |

