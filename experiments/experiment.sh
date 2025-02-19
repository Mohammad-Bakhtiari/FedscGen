#!/bin/bash
NUM_GPUS="${1:-3}"

chmod +x run-scgen.sh
echo "Running scgen"
./run-scgen.sh "${NUM_GPUS}"

chmod +x run-tuning.sh
echo "Running tuning"
./run-tuning.sh "${NUM_GPUS}"

chmod +x run-fedscgen.sh
echo "Running fedscgen without SMPC"
./run-fedscgen.sh  false "${NUM_GPUS}"

chmod +x run-fedscgen.sh
echo "Running fedscgen with SMPC"
./run-fedscgen.sh true "${NUM_GPUS}"


chmod +x scgen-with-batch-out.sh
echo "Running scgen with batch out"
./scgen-with-batch-out.sh 'HumanPancreas.h5ad' '' false false "0,1,2,3,4"

chmod +x run-classification.sh
echo "Running centralized classification using corrected data by scGen and FedscGen"
./run-classification.sh "${NUM_GPUS}"