set -xeuo pipefail

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    python -m venv venv
    source venv/bin/activate
fi

pip install -r requirements.txt

TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPU=8

model_name_or_path=$1
dataset=$2
do_eval=$3
torchrun \
    --nproc-per-node=$NGPU \
    --local-ranks-filter=0 \
    -m train \
    $model_name_or_path \
    $dataset \
    $do_eval
