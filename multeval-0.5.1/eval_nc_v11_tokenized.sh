./multeval.sh eval --refs ../pred_outs/gold.tok.txt \
		   --hyps-baseline ../pred_outs/test_vanilla.tok.pred \
		   --hyps-sys1 ../pred_outs/test.concat.tok.pred \
		   --hyps-sys2 ../pred_outs/mha-concat.tok.pred \
		   --hyps-sys3 ../pred_outs/concat-single-decoder.de.tok.pred \
		   --hyps-sys4 ../pred_outs/mha-concat-single-decoder.de.tok.pred \
 		   --hyps-sys5 ../pred_outs/bilstm_gat.de.tok.pred \
		   --hyps-sys6 ../pred_outs/seperate.de.tok.pred \
		   --meteor.language de 
