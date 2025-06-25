"""Utility script to compute dominant batches with and without CrypTen SMPC."""
import __init__
import argparse
import anndata
import crypten
import numpy as np
import copy
import torch
from fedscgen.utils import (
    remove_cell_types,
    combine_cell_types,
    set_seed,
    testset_combination,
)


def create_clients(adata, n_clients, batch_key, train_batches):
    """Return a list of ``AnnData`` objects, one per batch."""
    clients = []
    for client in range(1, n_clients + 1):
        clients.append(adata[adata.obs[batch_key].isin([train_batches.pop()])].copy())
    return clients


def _count_cells(clients, cell_types, cell_key):
    counts = []
    for client in clients:
        counts.append([
            client[client.obs[cell_key] == ct].shape[0] if ct in client.obs[cell_key].unique() else 0
            for ct in cell_types
        ])
    return counts


def dominant_plain(clients, cell_types, cell_key):
    """Return dominant batches without using SMPC."""
    counts = _count_cells(clients, cell_types, cell_key)
    max_idx = np.argmax(np.array(counts), axis=0)
    dominant = {}
    for ct, idx in zip(cell_types, max_idx.tolist()):
        dominant.setdefault(f"client_{int(idx)}", []).append(ct)
    return dominant

def dominant_smpc(clients, cell_types, cell_key):
    """Return dominant batches using CrypTen for secure aggregation, preserving privacy."""

    print("\n🔐 SMPC: Encrypting counts...")
    counts = _count_cells(clients, cell_types, cell_key)
    encrypted_counts = [crypten.cryptensor(c) for c in counts]
    import pdb; pdb.set_trace()

    # Stack encrypted counts across clients
    stacked = crypten.stack(encrypted_counts)  # Shape: [n_clients, n_cell_types]

    print("🔐 SMPC: Performing argmax across clients (dim=0)...")
    max_idx_encrypted = stacked.argmax(dim=0)  # Encrypted result

    # Decrypt only the final result (client assignment per cell type)
    max_idx_plain = max_idx_encrypted.get_plain_text().flatten().tolist()
    max_idx = [int(i) for i in max_idx_plain]

    # Sanity check
    if len(max_idx) != len(cell_types):
        raise ValueError("Mismatch between argmax output and number of cell types.")

    print("\n📊 Dominant batch assignments (SMPC result):")
    dominant = {}
    for ct, idx in zip(cell_types, max_idx):
        print(f"  → {ct} assigned to client_{idx}")
        dominant.setdefault(f"client_{idx}", []).append(ct)

    return dominant




def normalize_result(d):
    """Sort keys and values to ensure consistent comparison."""
    return {k: sorted(v) for k, v in sorted(d.items())}


def main(args):
    adata = anndata.read_h5ad(args.adata)
    for test_batches in testset_combination(args.batches, args.batch_out):
        print(f"{test_batches} as the test batches")
        train_batches = copy.deepcopy(args.batches)
        for batch in test_batches:
            train_batches.remove(batch)
        clients = create_clients(adata, args.n_clients, args.batch_key, train_batches)
        cell_types = sorted(adata.obs[args.cell_key].unique().tolist())
        check_consistency(cell_types, clients, args.cell_key)


def check_consistency(cell_types, clients, cell_key):
    plain_result = dominant_plain(clients, cell_types, cell_key)
    smpc_result = dominant_smpc(clients, cell_types, cell_key)

    print("Dominant batches without SMPC:")
    for client, types in plain_result.items():
        print(f"{client}: {types}")

    print("\nDominant batches with SMPC:")
    for client, types in smpc_result.items():
        print(f"{client}: {types}")

    if normalize_result(plain_result) != normalize_result(smpc_result):
        print("\n❌ Mismatch detected between plain and SMPC results.")
        print("Plain:", normalize_result(plain_result))
        print("SMPC: ", normalize_result(smpc_result))
        raise AssertionError("Results from plain and SMPC methods should match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dominant batches using SMPC")
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--batches", type=str, required=True)
    parser.add_argument("--batch_key", type=str, default="batch")
    parser.add_argument("--cell_key", type=str, default="cell_type")
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--batch_out", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.batches = [b.strip() for b in args.batches.split(",")]
    main(args)
