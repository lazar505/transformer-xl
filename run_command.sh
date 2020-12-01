#!/bin/bash

# Data
RAW_DATA=data/cases.zip
DATA_ROOT=data/cases_data_50k
MODEL_ROOT=model_transformer

# Model
DIV_VAL=1
N_LAYER=12
# D_MODEL == D_EMBED and D_MODEL == N_HEAD x D_HEAD
D_MODEL=512
D_EMBED=512
N_HEAD=16
D_HEAD=32
D_INNER=2048

# Training
TGT_LEN=128
MEM_LEN=128

BSZ=16
NUM_CORE=1

# Testing
TEST_TGT_LEN=64
TEST_MEM_LEN=640
TEST_CLAMP_LEN=400

TEST_BSZ=1
TEST_NUM_CORE=1

echo "Sourcing virtual environment..."
source venv/bin/activate

if [[ $1 == 'make_data' ]]; then
    # Make train.txt valid.txt train.txt
    echo "Making data..."
    python3 make_data.py \
        --raw_data_zip=${RAW_DATA} \
        --data_dir=${DATA_ROOT}/ \
        ${@:2}
elif [[ $1 == 'train_data' ]]; then
    # Make tfrecords for training
    echo "Making tfrecords for training..."
    python3 data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=cdata \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    # Make tfrecords for testing
    echo "Making tfrecords for testing..."
    python3 data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=cdata \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        ${@:2}
elif [[ $1 == 'vocab' ]]; then
    # Print vocabulary to a file
    echo "Printing vocabulary..."
    python3 print_vocabulary.py \
        --data_dir=${DATA_ROOT}/tfrecords/ \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    # Train transformer
    echo 'Running training...'
    python3 run_transformer.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=${MODEL_ROOT} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=1 \
        --save_steps=10 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    # Evaluate transformer
    echo 'Running evaluation...'
    python3 run_transformer.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=${MODEL_ROOT} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
elif [[ $1 == 'gen' ]]; then
    # Generate text
    echo "Starting interactive transformer..."
    python3 run_transformer.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=${MODEL_ROOT} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=1 \
        --do_train=False \
        --do_eval=False \
        --eval_split=test \
        --do_generate=True \
        --token_number=20 \
        ${@:2}
elif [[ $1 == 'web' ]]; then
    # Start web transformer
    echo "Starting web transformer..."
    python3 run_web_transformer.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=${MODEL_ROOT} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=1 \
        ${@:2}
elif [[ $1 == 'make_experiment_data' ]]; then
    # Make data for experiments
    echo "Making data for experiments..."
    python3 make_experiment_data.py
        ${@:2}
elif [[ $1 == 'experiment' ]]; then
    # Do experiments
    echo "Doing experiments..."
    python3 run_transformer.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=${MODEL_ROOT} \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=1 \
        --do_train=False \
        --do_eval=False \
        --eval_split=test \
        --do_generate=False \
        --do_experiment=True \
        --token_number=100 \
        ${@:2}
else
    echo 'Unknown argment!'
fi

