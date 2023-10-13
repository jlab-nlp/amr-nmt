





CUDA_VISIBLE_DEVICES=0

python3 translate.py -model model/nmt_amr/nmt_amr_2020.pt -src data/fixed_nmt_amr/debug.amr -grh data/fixed_nmt_amr/debug.grh \
          -eval_mt_src data/amr_nmt/newstest2016.en -eval_mt_tgt data/amr_nmt/newstest2016.de \
          -output test.pred -replace_unk -share_vocab -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
          -tgt_lang de_DE -bart_config config.json

perl tools/multi-bleu.perl gold.txt  < test.pred


echo "-------------------"
echo "test done!"
echo "-------------------"