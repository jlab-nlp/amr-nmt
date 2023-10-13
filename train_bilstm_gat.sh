# DATA_DIR="${1:-data/amr2015/amr2015}"
# MODEL_DIR="${2:-model/amr2015/multiview_cat}"
# GPU="${3:-1}"
# LOG="${4}"
# FUSION="${5-None}"

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
echo   train.py -data data/full_nmt_amr/nmt_amr -save_model model/full_bilstm_gat_small \
        -model_kind bilstm_gat -enc_layers 1 -dec_layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type rnn -position_encoding \
        -train_steps 120000  -max_generator_batches 100 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  \
        -optim adam -adam_beta2 0.998 -decay_method noam \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 \
        -world_size 1 --report_every 100 -gpu_ranks 0 \
        --share_decoder_embeddings --share_embeddings -aggregation jump \
        -log_file log_en_de_trn -accum_count 2 -warmup_steps 16000 -learning_rate 2 -edges 2
        #-train_from model/amr2017/multiview_cat.bpe10.jump5.lr2.both/ADAM_step95000_lr0.00029.pt

python3  train.py -data data/full_nmt_amr/nmt_amr -save_model model/full_bilstm_gat \
        -model_kind bilstm_gat -enc_layers 1 -dec_layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type rnn -position_encoding \
        -train_steps 300000  -max_generator_batches 100 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  \
        -optim adam -adam_beta2 0.998 -decay_method noam \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 \
        -world_size 1 --report_every 100 -gpu_ranks 0 \
        --share_decoder_embeddings --share_embeddings -aggregation jump \
        -log_file full_bilstm_gat.log -accum_count 2 -warmup_steps 16000 -learning_rate 2 -edges 2
        #-train_from model/amr2017/multiview_cat.bpe10.jump5.lr2.both/ADAM_step95000_lr0.00029.pt

