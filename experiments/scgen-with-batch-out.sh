#!/bin/bash


# Accepting arguments
H5AD_FILE="$1"
REMOVE_CELL_TYPES="$2"
COMBINE="$3"
DROP="$4"
BATCH_OUT_VALUES=($5)
GPU="${6:-1}"
BATCH_SIZE="${7:-50}"
Z_DIM="${8:-10}"


# DATASET is H5AD_FILE without the extension
DATASET=$(echo "$H5AD_FILE" | cut -f 1 -d '.')

# Setting up other variables based on the flags
if [ "$COMBINE" = "true" ]; then
  TARGET_FOLDER="combined"
elif [ "$DROP" = "true" ]; then
  TARGET_FOLDER="dropped"
else
  TARGET_FOLDER="all"
fi

root_dir="$(dirname "$PWD")"
raw="${root_dir}/data/datasets/${H5AD_FILE}"
output_path="${root_dir}/results/scgen/centralized/${DATASET}/${TARGET_FOLDER}"

combine_flag=""
if [ "$COMBINE" = "true" ]; then
    combine_flag="--combine"
fi
echo $BATCH_OUT_VALUES
for i in "${!BATCH_OUT_VALUES[@]}"; do
  batch_out=${BATCH_OUT_VALUES[$i]}
  output_path="${output_path}/BO${i}"
  mkdir -p "${output_path}"
  echo "$batch_out"
  # Running python scripts
  python "${root_dir}/code/scgen_with_batch_out.py" \
  --model_path "${root_dir}/models/centralized/${DATASET}" \
  --data_path "$raw" \
  --output_path "$output_path" \
  --epoch 100 \
  --batch_key "batch" \
  --cell_key "cell_type" \
  --z_dim "$Z_DIM" \
  --hidden_layers_sizes "800,800" \
  --batch_size "$BATCH_SIZE" \
  --batch_out "$batch_out" \
  --remove_cell_types "$REMOVE_CELL_TYPES" \
  --early_stopping_kwargs "{'early_stopping_metric': 'val_loss', 'patience': 20, 'threshold': 0, 'reduce_lr': True, 'lr_patience': 13, 'lr_factor': 0.1}" \
  --gpu "$GPU" \
  $combine_flag
done