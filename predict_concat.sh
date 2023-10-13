





CUDA_VISIBLE_DEVICES=0

python3 translate.py -model model/en_vi_concat_tiny_acc60.84_ppl1.00_lr0.00079_step31000.pt -src data/en_vi_nmt_amr/test1.amr -grh data/en_vi_nmt_amr/test1.grh \
          -eval_mt_src en-vi-source/tst2013.tok.en -eval_mt_tgt en-vi-source/tst2013.tok.norm.vi \
          -output concat.vi.pred -replace_unk -share_vocab -max_length 50 -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
          -tgt_lang vi_VN -bart_config config.json

perl tools/tokenizer.perl -q -l vi -threads 8 < concat.gold.txt > concat.gold.tok.txt
perl tools/tokenizer.perl -q -l vi -threads 8 < concat.vi.pred > concat.vi.tok.pred
perl tools/multi-bleu.perl concat.gold.tok.txt  < concat.vi.tok.pred


echo "-------------------"
echo "test done!"
echo "-------------------"
