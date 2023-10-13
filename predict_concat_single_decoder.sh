





CUDA_VISIBLE_DEVICES=0

python3 translate.py -model model/full_nmt_amr_concat_single_decoder_acc66.44_ppl1.00_lr0.00053_step140000.pt -src data/full_nmt_amr/test.amr -grh data/full_nmt_amr/test.grh \
          -eval_mt_src wmt16_de_en/newstest2016.tok.en -eval_mt_tgt wmt16_de_en/newstest2016.tok.de \
          -output concat-single-decoder.de.pred -replace_unk -share_vocab -max_length 50 -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
          -tgt_lang de_DE -bart_config config.json

perl tools/tokenizer.perl -q -l de -threads 8 < concat-single-decoder.gold.txt > concat-single-decoder.gold.tok.txt
perl tools/tokenizer.perl -q -l de -threads 8 < concat-single-decoder.de.pred > concat-single-decoder.de.tok.pred
perl tools/multi-bleu.perl concat-single-decoder.gold.tok.txt  < concat-single-decoder.gold.tok.txt


echo "-------------------"
echo "test done!"
echo "-------------------"
