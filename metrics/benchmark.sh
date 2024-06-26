#!/bin/bash

scenario=$1
plot_only=$2

plot_only_flag=""
if [ "$plot_only" = "true" ]; then
  plot_only_flag="--plot_only"
fi

scenarios=("datasets" "snapshots" "batch-out" "tuning")
root_dir="$(dirname "$PWD")"

DATASETS=("CellLine" "PBMC" "HumanPancreas" "MouseRetina" "MouseBrain" "MouseHematopoieticStemProgenitorCells" "HumanDendriticCells" "MouseCellAtlas")
#DATASETS=("MouseBrain")

# datasets
if [ "$scenario" = "datasets" ]; then
  for inclusion in "all" "dropped" "combined"; do
    python benchmark.py --data_dir "${root_dir}/results/scgen/centralized" \
    --fed_data_dir "${root_dir}/results/scgen/federated" \
    --scenarios "datasets" \
    --inclusion "${inclusion}" &
  done
  wait
fi

# snapshot
if [ "$scenario" = "snapshots" ]; then
  python benchmark.py --data_dir "${root_dir}/results/scgen/batchout/snapshot" \
  --scenarios "snapshots" \
  --inclusion "all" \
  --cell_key "cell_type" \
  --batch_key "batch" \
  --n_components 20 \
  --n_rounds 10 \
  $plot_only_flag
fi

# batch_out
if [ "$scenario" = "batch-out" ]; then
  python benchmark.py --data_dir "${root_dir}/results/scgen/centralized/HumanPancreas/all" \
  --fed_data_dir "${root_dir}/results/scgen/federated/HumanPancreas/all/BO1-C4" \
  --scenarios "batch-out" \
  --inclusion all \
  --cell_key "cell_type" \
  --batch_key "batch" \
  --n_batches 5 \
  --n_components 20
fi

# Hyperparameter tuning rounds & epochs
if [ "$scenario" = "tuning" ]; then
  for dataset in "${!DATASETS[@]}"; do
    if [ "${DATASETS[$dataset]}" = "MouseBrain" ]; then
      continue
    fi
    target_dir="${root_dir}//results/scgen/federated/param-tuning/${DATASETS[$dataset]}"
    echo "Running benchmark for param tuning of FedScGen for ${DATASETS[$dataset]}"
    # Copy corrected files by ScGen
    cp "${root_dir}/results/scgen/centralized/${DATASETS[$dataset]}/all/corrected.h5ad" "${target_dir}/scGen.h5ad"

    python benchmark.py --data_dir "${target_dir}" \
    --scenarios "${scenario}" \
    --inclusion "all" \
    --cell_key "cell_type" \
    --n_components 20 &
  done
  wait
fi