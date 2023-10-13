./multeval.sh eval --refs ../pred_outs_full/gold.tok.txt \
	           --hyps-baseline ../pred_outs_full/vanilla.full.tok.pred \
		   --hyps-sys1 ../pred_outs_full/concat.full.tok.pred \
		   --hyps-sys2 ../pred_outs_full/mha_concat.full.tok.pred \
		   --hyps-sys3 ../pred_outs_full/concat-single-decoder.de.tok.pred \
		   --hyps-sys4 ../pred_outs_full/mha_concat_single_decoder.de.tok.pred \
 		   --hyps-sys5 ../pred_outs_full/bilstm_gat.de.tok.pred \
		   --hyps-sys6 ../pred_outs_full/seperate.de.tok.pred \
		   --meteor.language de
