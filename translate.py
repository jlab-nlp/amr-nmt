#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
import reduce_embeding_size
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import codecs
import onmt.model_builder
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.transformers import MBartTokenizer
import torch


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    logger.info(opt)
    #print(opt.output)
    out_file = codecs.open(opt.output, 'w+', 'utf-8')
    gold_file = codecs.open(opt.output.split(".")[0]+"."+"gold.txt", 'w+', 'utf-8')
    #print(out_file)
    fields, model, model_opt = onmt.model_builder.load_test_model(opt)
    if int(opt.gpu) != -1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)
    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    grh_reader = inputters.str2reader["grh"].from_opt(opt)
    bart_text_reader = inputters.str2reader["bart_text"].from_opt(opt)
    res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
    res.set_training(False)
    model.config.vocab_size = len(res.new_ids_to_org_ids)
    # translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    grh_shards = split_corpus(opt.grh, opt.shard_size)
    mt_src_shards = split_corpus(opt.eval_mt_src[0], opt.shard_size)
    mt_tgt_shards = split_corpus(opt.eval_mt_tgt[0], opt.shard_size)
    shard_pairs = zip(src_shards, grh_shards, mt_src_shards, mt_tgt_shards)
    trans_output = []
    for i, (src_shard, grh_shard, mt_src_shard, mt_tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        bart_text_shard = {"src": mt_src_shard, "tgt": mt_tgt_shard}
        dataset = inputters.Dataset(
            fields,
            readers=([src_reader, grh_reader, bart_text_reader]),
            data=([("src", src_shard), ('grh', grh_shard), ('bart_text', bart_text_shard)]),
            dirs=([opt.src_dir, None, None]),
            sort_key=inputters.str2sortkey[opt.data_type],
        )
        data_iter = inputters.OrderedIterator(
            dataset=dataset,
            device=device,
            batch_size=opt.batch_size,
            batch_size_fn=None,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False
        )
        mbart_tokenizer = MBartTokenizer(vocab_file="sentencepiece.bpe.model")
        for batch_id, batch in enumerate(data_iter):
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            grh = batch.grh
            input_ids = batch.bart_text["input_ids"]
            # input_ids = reduce_embeding_size.original_ids_to_new_ids(input_ids, res.org_ids_to_new_ids)
            decoder_input_ids = batch.bart_text["decoder_input_ids"]
            # decoder_input_ids = reduce_embeding_size.original_ids_to_new_ids(decoder_input_ids, res.org_ids_to_new_ids)
            labels = batch.bart_text["labels"]
            # labels = reduce_embeding_size.original_ids_to_new_ids(labels, res.org_ids_to_new_ids)
            #print(input_ids.size())
            #print(input_ids)
            #print(res.bos_token_id)
            #print(res.pad_token_id)
            #print(res.eos_token_id)
            translated_tokens = model.generate(input_ids=input_ids.to(device),
                                               min_length=opt.min_length,
                                               max_length=opt.max_length,
                                               bos_token_id=res.bos_token_id,
                                               pad_token_id=res.pad_token_id,
                                               eos_token_id=res.eos_token_id,
                                               decoder_start_token_id=res.de_DE_id,
                                               num_beams=opt.beam_size,
                                               beam_size=opt.beam_size,
                                               vocab_size=len(res.new_ids_to_org_ids),
                                               device=device,
                                               input_src=input_ids.to(device),
                                               src=src.to(device),
                                               grh=grh.to(device),
                                               lengths=src_lengths.to(device))

            # print(translated_tokens)
            # outputs = model(input_ids=input_ids,
            #                 decoder_input_ids=decoder_input_ids,
            #                 labels=labels,
            #                 src=src,
            #                 lengths=src_lengths,
            #                 grh=grh,
            #                 bptt=False,
            #                 return_dict=True)
            # lm_logits = outputs[1]
            # translated_tokens = torch.argmax(lm_logits, dim=-1)

            org_translated_tokens = []
            for translated_token in translated_tokens:
                org_translated_token = reduce_embeding_size.new_ids_to_original_ids(translated_token,
                                                                                    res.new_ids_to_org_ids,
                                                                                    False)
                org_translated_tokens.append(org_translated_token)
            converted_labels = []
            for label in labels:
                org_label = reduce_embeding_size.new_ids_to_original_ids(label,
                                                                         res.new_ids_to_org_ids,
                                                                         False)
                converted_labels.append(org_label)
            # print(org_translated_tokens)
            translations = mbart_tokenizer.batch_decode(org_translated_tokens, skip_special_tokens=True)
            gold_translations = mbart_tokenizer.batch_decode(converted_labels, skip_special_tokens=True)
            # print(translations)
            trans_output.extend(translations)
            for translation in translations:
                # print("translation:", translation)
                out_file.write(translation + "\n")
            for gold_translation in gold_translations:
                gold_file.write(gold_translation+"\n")
            #msg = "batch:"+str(batch_id) + " processed!"
            #logger.info(msg)
    out_file.close()
    gold_file.close()


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
