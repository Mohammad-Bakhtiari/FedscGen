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
    dict_counts = {ct: [sum(c.obs[cell_key] == ct) for c in clients] for ct in cell_types}
    print(dict_counts)
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
    dominant = {f"client_{i}": [] for i in range(len(clients))}
    counts = _count_cells(clients, cell_types, cell_key)
    encrypted_counts = [crypten.cryptensor(c) for c in counts]
    # Stack encrypted counts across clients
    stacked = crypten.stack(encrypted_counts)  # Shape: [n_clients, n_cell_types]
    max_idx_plain = stacked.argmax(dim=0, one_hot=False).get_plain_text().numpy()
    for client_idx, ct in zip(max_idx_plain, cell_types):
        dominant[f"client_{client_idx}"].append(ct)

    print(dominant)
    # tie breaking
    ones = crypten.cryptensor(torch.ones(len(cell_types)))
    one = crypten.cryptensor(torch.tensor(1))
    maxx = stacked.max(dim=0)[0]
    max_count = (maxx == stacked).sum(dim=0)

    ties = (max_count != ones).argmax(dim=0, one_hot=False).get_plain_text().tolist()
    print(ties)
    if type(ties) is not list:
        ties = [ties]
    tied_celltypes = [cell_types[tie] for tie in ties]
    print("Tied cell types:", tied_celltypes)
    for c in dominant.keys():
        dominant[c] = list(set(dominant[c]) - set(tied_celltypes))
    print(dominant)
    for c in range(len(clients)):
        for tie in ties:
            occurrence = (stacked[:c + 1, tie].sum() == maxx[tie]).get_plain_text().item()
            tied_cell_type = cell_types[tie]
            print(f"Client {c}, Cell Type {cell_types[tie]}, Occurrence: {occurrence}")
            if occurrence:
                dominant[f"client_{c}"].append(tied_cell_type)
                break
    print(dominant)
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
