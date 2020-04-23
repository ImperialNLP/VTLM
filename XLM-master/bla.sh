#!/bin/bash

if [ -d "/home/menekse" ]; then
  source /home/menekse/virtualenvs/torch_env/bin/activate
fi

# Allow default values to be defined if vars not defined
export NGPU=${NGPU:-3}
export DATA_PATH=${DATA_PATH:-/data/shared/ConceptualCaptions/XLM_data/50k}
export DUMP_PATH=${DUMP_PATH:-/data/menekse/dumped}

# Reloading
RELOAD_FILE="${DUMP_PATH}/xlm_en_de_tlm/3094/best-valid_en_de_mlm_ppl.pth"
RELOAD_MODEL_ARGS=""
if [ -f $RELOAD_FILE ]; then
  RELOAD_MODEL_ARGS="--reload_model ${RELOAD_FILE}"
fi

if [ $NGPU == "1" ]; then
    ipython -i train.py -- --exp_name xlm_en_de_tlm \
    --dump_path $DUMP_PATH --data_path $DATA_PATH \
    --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 32 --bptt 256 \
    --optimizer adam,lr=0.0001 --epoch_size 300000 --max_epoch 100000 \
    --validation_metrics valid_en_de_mlm_ppl --stopping_criterion valid_en_de_mlm_ppl,25 \
    --fp16 false --save_periodic 2 ${RELOAD_MODEL_ARGS} $@
else
  python -m torch.distributed.launch --nproc_per_node=$NGPU --nnodes=1 --node_rank=0 \
    --master_addr="193.140.236.52" --master_port=8088 train.py --exp_name xlm_en_de_tlm \
    --dump_path $DUMP_PATH --data_path $DATA_PATH \
    --lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --emb_dim 512 --n_layers 6 --n_heads 8 \
    --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 32 --bptt 256 \
    --optimizer adam,lr=0.0001 --epoch_size 300000 --max_epoch 100000 \
    --validation_metrics valid_en_de_mlm_ppl --stopping_criterion valid_en_de_mlm_ppl,25 \
    --fp16 false --save_periodic 2 ${RELOAD_MODEL_ARGS} $@
fi
