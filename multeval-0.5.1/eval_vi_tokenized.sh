./multeval.sh eval --refs ../pred_outs_en_vi_tiny/gold.tok.txt1 \
		   --hyps-baseline ../pred_outs_en_vi_tiny/vanilla.vi.tok.pred1 \
		   --hyps-sys1 ../pred_outs_en_vi_tiny/bilstm_gat.vi.tok.pred1 \
		   --hyps-sys2 ../pred_outs_en_vi_tiny/concat.vi.tok.pred1 \
		   --hyps-sys3 ../pred_outs_en_vi_tiny/mha_concat.vi.tok.pred1 \
                   --meteor.language xx \
		   --meteor.task li
