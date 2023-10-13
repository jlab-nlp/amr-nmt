DATA_DIR=data
HEAD=$1
PREPROC_DIR=${DATA_DIR}/${HEAD}_tmp
ORIG_AMR_DIR=${DATA_DIR}/abstract_meaning_representation_amr_2.0/data/alignments/split
FINAL_AMR_DIR=${DATA_DIR}/${HEAD}

#for SPLIT in train dev test1 test2; do
#	 python3 preprocess/amr_preprocess/split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
#	 python3 preprocess/amr_preprocess/preproc_amr.py --input_amr ${PREPROC_DIR}/${SPLIT}/graphs.txt --input_surface ${PREPROC_DIR}/${SPLIT}/surface.txt --output ${FINAL_AMR_DIR}/${SPLIT}.amr --output_surface ${FINAL_AMR_DIR}/${SPLIT}_surface.pp.txt --mode LINE_GRAPH --triples-output ${FINAL_AMR_DIR}/${SPLIT}.grh --anon --map-output ${FINAL_AMR_DIR}/${SPLIT}_map.pp.txt --anon-surface ${FINAL_AMR_DIR}/${SPLIT}.snt --nodes-scope ${FINAL_AMR_DIR}/${SPLIT}_nodes.scope.pp.txt --scope
#	 paste ${FINAL_AMR_DIR}/${SPLIT}.amr ${FINAL_AMR_DIR}/${SPLIT}.grh > ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
#done

#echo '{"d": 1, "r": 2, "s": 3}' > ${FINAL_AMR_DIR}/edge_vocab.json

python3 preprocess.py -train_src ${FINAL_AMR_DIR}/train.amr -train_tgt ${FINAL_AMR_DIR}/train_surface.pp.txt -train_graph ${FINAL_AMR_DIR}/train.grh -valid_src ${FINAL_AMR_DIR}/dev.amr -valid_tgt ${FINAL_AMR_DIR}/dev_surface.pp.txt -valid_graph ${FINAL_AMR_DIR}/dev.grh -edges_vocab ${FINAL_AMR_DIR}/edge_vocab.json -save_data ${FINAL_AMR_DIR}/nmt_amr -src_vocab_size 30000 -tgt_vocab_size 30000 -shared_vocab_size 8000 -src_seq_length 200 -tgt_seq_length 200 -share_vocab -train_mt_src en-vi-source/train.clean.en -valid_mt_src en-vi-source/tst2012.tok.en -train_mt_tgt en-vi-source/train.clean.norm.vi -valid_mt_tgt en-vi-source/tst2012.tok.norm.vi -src_lang en_XX -tgt_lang vi_VN -bart_config config.json
