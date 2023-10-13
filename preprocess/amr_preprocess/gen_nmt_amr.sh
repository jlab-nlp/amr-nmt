#!/bin/bash

# CHANGE THIS
REPO_DIR=/Users/changmaoli/Research/amrintomt/GraphToSequence/HetGT-master

# CONSTANTS
DATA_DIR=${REPO_DIR}/data
PREPROC_DIR=${DATA_DIR}/nc_v11_nmt_amr_tmp
ORIG_AMR_DIR=${DATA_DIR}/abstract_meaning_representation_amr_2.0/data/alignments/split
FINAL_AMR_DIR=${DATA_DIR}/nc_v11_nmt_amr

#####
# CREATE FOLDER STRUCTURE

# mkdir -p ${PREPROC_DIR}/train
# mkdir -p ${PREPROC_DIR}/dev
# mkdir -p ${PREPROC_DIR}/test

# mkdir -p ${FINAL_AMR_DIR}/
# mkdir -p ${FINAL_AMR_DIR}/train
# mkdir -p ${FINAL_AMR_DIR}/dev
# mkdir -p ${FINAL_AMR_DIR}/test

#####
# CONCAT ALL SEMBANKS INTO A SINGLE ONE
# cat ${ORIG_AMR_DIR}/training/* > ${PREPROC_DIR}/train/raw_amrs.txt
# cat ${ORIG_AMR_DIR}/dev/* > ${PREPROC_DIR}/dev/raw_amrs.txt
# cat ${ORIG_AMR_DIR}/test/* > ${PREPROC_DIR}/test/raw_amrs.txt

#####
# CONVERT ORIGINAL AMR SEMBANK TO ONELINE FORMAT
for SPLIT in train dev test; do
    python3 /Users/changmaoli/Research/amrintomt/GraphToSequence/HetGT-master/preprocess/amr_preprocess/split_amr.py ${PREPROC_DIR}/${SPLIT}/raw_amrs.txt ${PREPROC_DIR}/${SPLIT}/surface.txt ${PREPROC_DIR}/${SPLIT}/graphs.txt
    python3 /Users/changmaoli/Research/amrintomt/GraphToSequence/HetGT-master/preprocess/amr_preprocess/preproc_amr.py --input_amr ${PREPROC_DIR}/${SPLIT}/graphs.txt --input_surface ${PREPROC_DIR}/${SPLIT}/surface.txt --output ${FINAL_AMR_DIR}/${SPLIT}.amr --output_surface ${FINAL_AMR_DIR}/${SPLIT}_surface.pp.txt --mode LINE_GRAPH --triples-output ${FINAL_AMR_DIR}/${SPLIT}.grh --anon --map-output ${FINAL_AMR_DIR}/${SPLIT}_map.pp.txt --anon-surface ${FINAL_AMR_DIR}/${SPLIT}.snt --nodes-scope ${FINAL_AMR_DIR}/${SPLIT}_nodes.scope.pp.txt --scope
    paste ${FINAL_AMR_DIR}/${SPLIT}.amr ${FINAL_AMR_DIR}/${SPLIT}.grh > ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
done




#python3 global_node.py --input_dir ${FINAL_AMR_DIR}/

#for SPLIT in train dev test; do
#	cp ${FINAL_AMR_DIR}/${SPLIT}.amr_g  ${FINAL_AMR_DIR}/${SPLIT}.amr
#	cp ${FINAL_AMR_DIR}/${SPLIT}.grh_g  ${FINAL_AMR_DIR}/${SPLIT}.grh
#	cp ${FINAL_AMR_DIR}/${SPLIT}.amrgrh_g  ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
#done

# rm ${FINAL_AMR_DIR}/*_g

echo '{"d": 1, "r": 2, "s": 3}' > ${FINAL_AMR_DIR}/edge_vocab.json




