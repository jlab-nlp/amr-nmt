./multeval.sh eval --refs ../pred_outs_en_mg_tiny/gold.tok.txt \
		   --hyps-baseline ../pred_outs_en_mg_tiny/vanilla.mg.tok.pred \
		   --hyps-sys1 ../pred_outs_en_mg_tiny/bilstm_gat.mg.tok.pred \
		   --hyps-sys2 ../pred_outs_en_mg_tiny/concat.mg.tok.pred \
		   --hyps-sys3 ../pred_outs_en_mg_tiny/mha_concat.mg.tok.pred \
                   --meteor.language xx \
		   --meteor.task li
