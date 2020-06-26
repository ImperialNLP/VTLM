#!/bin/bash

# Allow default values to be defined if vars not defined
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DUMP_PATH=${DUMP_PATH:-/data/ozan/experiments/mmvc/mmvc_code/pretraining}
export DATA_PATH=${DATA_PATH:-/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe}

mkdir -p $DUMP_PATH

PAIR=`basename $DATA_PATH`
EMB=${EMB:-512}
NL=${NL:-6}
NH=${NH:-8}
BS=${BS:-64}
LR=${LR:-0.0001}
NAME="mlm_emb${EMB}_${NL}l_${NH}h_bs${BS}_lr${LR}"
echo $NAME

ipython -i train.py -- --exp_name $NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
--lgs 'en-de' --clm_steps '' --mlm_steps 'en,de' --emb_dim ${EMB} --n_layers ${NL} --n_heads ${NH} \
--dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size ${BS} \
--optimizer "adam,lr=${LR}" --epoch_size 400000 --max_epoch 100000 \
--validation_metrics _valid_mlm_ppl --stopping_criterion _valid_mlm_ppl,50 \
--fp16 false --save_periodic 10 $@
