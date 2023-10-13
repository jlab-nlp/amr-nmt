./multeval.sh eval --refs ../pred_outs/gold.txt \
                   --hyps-baseline ../pred_outs/test_vanilla.pred \
		   --hyps-sys1 ../pred_outs/test.concat.pred \
		   --hyps-sys2 ../pred_outs/mha-concat.pred \
		   --hyps-sys3 ../pred_outs/mha-concat-add.pred \
                   --meteor.language de \
		   --rankDir rank \
                   --sentLevelDir sentLevel
