./multeval.sh eval --refs ../pred_outs_en_mg_small/gold.tok.txt \
		   --hyps-baseline ../pred_outs_en_mg_small/vanilla.mg.tok.pred \
		   --hyps-sys1 ../pred_outs_en_mg_small/concat.mg.tok.pred \
		   --hyps-sys2 ../pred_outs_en_mg_small/mha_concat.mg.tok.pred \
                   --meteor.language xx \
		   --meteor.task li
