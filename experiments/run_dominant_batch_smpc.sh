#!/bin/bash

# === Load Dataset Config ===
source ./config.sh

root_dir="$(dirname "$PWD")"

# === Loop Through Datasets from config.sh ===
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    n_clients="2"
    batches="0,1"
    if [ "$dataset" == "HumanPancreas" ]; then
      n_clients="5"
      batches="0,1,2,3,4"
    elif [ "$dataset" == "CellLine" ]; then
      n_clients="3"
      batches="0,1,2"
    fi
    raw="${root_dir}/data/datasets/${DATASET}.h5ad"

    python "${root_dir}/scripts/dominant_batch_smpc.py" \
        --adata "$raw" \
        --batches "$batches" \
        --n_clients "$n_clients" \
        --batch_key "batch" \
        --cell_key "cell_type"


    echo ""
done
