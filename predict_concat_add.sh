





CUDA_VISIBLE_DEVICES=0

python3 translate.py -model model/nmt_amr_full_mha_add_concat_acc62.62_ppl1.00_lr0.00039_step45000.pt -src data/full_nmt_amr/test.amr -grh data/full_nmt_amr/test.grh \
          -eval_mt_src wmt16_de_en/newstest2016.tok.en -eval_mt_tgt wmt16_de_en/newstest2016.tok.de \
          -output mha_add_concat.full.pred -replace_unk -share_vocab -max_length 50 -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
          -tgt_lang de_DE -bart_config config.json

perl tools/tokenizer.perl -q -l de -threads 8 < mha_add_concat.gold.txt > mha_add_concat.gold.tok.txt
perl tools/tokenizer.perl -q -l de -threads 8 < mha_add_concat.full.pred > mha_add_concat.full.tok.pred
perl tools/multi-bleu.perl mha_add_concat.gold.tok.txt  < mha_add_concat.full.tok.pred


echo "-------------------"
echo "test done!"
echo "-------------------"
