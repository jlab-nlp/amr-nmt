





CUDA_VISIBLE_DEVICES=0
for MODEL in `find model -name 'en_mg_mha_concat_tiny_acc*'`
do
	echo $MODEL
	python3 translate.py -model $MODEL -src data/en_mg_nmt_amr/test.amr -grh data/en_mg_nmt_amr/test.grh -eval_mt_src lowresource/test.tok.en -eval_mt_tgt lowresource/test.tok.mg \
		-output mha_concat.mg.pred -replace_unk -share_vocab -max_length 50 -beam_size 10 -gpu 0 --batch_size 16 -log_file result -src_lang en_XX \
		-tgt_lang de_DE -bart_config config.json
        perl tools/tokenizer.perl -q -l mg -threads 8 < mha_concat.gold.txt > mha_concat.gold.tok.txt
	perl tools/tokenizer.perl -q -l mg -threads 8 < mha_concat.mg.pred > mha_concat.mg.tok.pred
	perl tools/multi-bleu.perl mha_concat.gold.tok.txt  < mha_concat.mg.tok.pred
done
#perl tools/tokenizer.perl -q -l vi -threads 8 < concat.gold.txt > concat.gold.tok.txt
#perl tools/tokenizer.perl -q -l vi -threads 8 < concat.vi.pred > concat.vi.tok.pred
#perl tools/multi-bleu.perl concat.gold.tok.txt  < concat.vi.tok.pred


echo "-------------------"
echo "test done!"
echo "-------------------"
