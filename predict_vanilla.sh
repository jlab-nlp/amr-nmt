





CUDA_VISIBLE_DEVICES=0
prefix=$1
python3 translate.py -model model/full_nmt_amr_mha_concat_single_decoder_acc63.18_ppl1.00_lr0.00055_step130000.pt -src data/nc_v11_nmt_amr/test.amr -grh data/nc_v11_nmt_amr/test.grh \
          -eval_mt_src  wmt16_de_en/newstest2016.tok.en -eval_mt_tgt  wmt16_de_en/newstest2016.tok.de \
          -output ${prefix}.de.pred -replace_unk -share_vocab -max_length 50 -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
          -tgt_lang de_DE -bart_config config.json

perl tools/tokenizer.perl -q -l de -threads 8 < ${prefix}.gold.txt > ${prefix}.gold.tok.txt
perl tools/tokenizer.perl -q -l de -threads 8 < ${prefix}.de.pred > ${prefix}.de.tok.pred
perl tools/multi-bleu.perl ${prefix}.gold.tok.txt  < ${prefix}.de.tok.pred


echo "-------------------"
echo "test done!"
echo "-------------------"
