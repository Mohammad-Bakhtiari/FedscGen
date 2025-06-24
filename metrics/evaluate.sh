#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh" || {
    echo "❌ Failed to source conda environment script."
    exit 1
}

# Activate fedscgen environment
conda activate r_eval || {
    echo "❌ Failed to activate r_eval environment."
    exit 1
}

root_dir="$(dirname "$PWD")"

chmod +x benchmark.sh
echo "Running benchmark for param tuning (rounds and epochs) of FedscGen in the 'all' inclusion scenario"
./benchmark.sh "tuning"
echo "Running benchmark for all approaches"
./benchmark.sh "approach"

cd "${root_dir}/metrics"
python plot.py --scenario "tuning" --data_dir "${root_dir}/results"
python plot.py --scenario "wmw" --data_dir "${root_dir}/results"


cd "${root_dir}/metrics/kbet"
chmod +x kbet.sh
echo "Running kbet for corrected data by scGen and FedscGen on all datasets across different inclusion scenarios"
./kbet.sh "approach"

echo "Running kbet for corrected data by scGen and FedscGen on HumanPancreas batchout"
./kbet.sh "batch-out"


cd "${root_dir}/metrics"
python plot.py --scenario "batchout" --data_dir "${root_dir}/results"

python plot.py --scenario "scenarios" --data_dir "${root_dir}/results"
python plot.py --scenarios "datasets" --data_dir "${root_dir}/results"
python plot.py --scenario "kbet-diff" --data_dir "${root_dir}/results/fedscgen"

cd "${root_dir}/metrics/lisi"
chmod +x evaluate.sh
echo "Running lisi for raw data, and corrected data by scGen and FedscGen"
./evaluate.sh "all"
./evaluate.sh "dropped"
./evaluate.sh "combined"

cd ..
echo "plotting LISI results for raw, scGen, and FedscGen on all datasets"
python plot.py --scenario "lisi" --data_dir "${root_dir}/results/fedscgen"

echo "Plot accuracy differences error bars"
python plot.py --scenario "classification
" --data_dir "${root_dir}/results"

echo "plotting umaps for corrected data by scGen and FedscGen on HumanPancreas batchout"
python plot_umaps.py --scenario "batchout" --data_dir "${root_dir}/results" \
  --output_dir "${root_dir}/results/umap/batch-out" --n_batches 5

echo "plotting umaps for corrected data by scGen and FedscGen on all datasets"
python plot_umaps.py --scenario "datasets" --data_dir "${root_dir}/results" \
 --raw_data_dir "${root_dir}/data/datasets" --output_dir "${root_dir}/results/umap/datasets"

