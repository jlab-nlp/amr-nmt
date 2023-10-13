from onmt.transformers.tokenization_mbart import MBartTokenizer
import torch
import json
from onmt.utils.logging import init_logger, logger
tokenizer = MBartTokenizer(vocab_file="sentencepiece.bpe.model")


class ReduceEmbeddingSize:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ReduceEmbeddingSize.__instance is None:
            ReduceEmbeddingSize()
        return ReduceEmbeddingSize.__instance

    def __init__(self):
        if ReduceEmbeddingSize.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            logger.info("Creating ReduceEmbeddingSize indexes...")
            train_en_file = "wmt16_de_en_full/train.tok.clean.en"
            valid_en_file = "wmt16_de_en/newstest2013.tok.en"
            test_en_file = "wmt16_de_en/newstest2016.tok.en"
            train_de_file = "wmt16_de_en_full/train.tok.clean.de"
            valid_de_file = "wmt16_de_en/newstest2013.tok.de"
            test_de_file = "wmt16_de_en/newstest2016.tok.de"
            train_en_used_pieces_ids = get_used_pieces_ids_by_corpus(train_en_file)
            logger.info("train_en_used_pieces_ids:", len(train_en_used_pieces_ids))
            valid_en_used_pieces_ids = get_used_pieces_ids_by_corpus(valid_en_file)
            logger.info("valid_en_used_pieces_ids:", len(valid_en_used_pieces_ids))
            test_en_used_pieces_ids = get_used_pieces_ids_by_corpus(test_en_file)
            logger.info("test_en_used_pieces_ids:", len(test_en_used_pieces_ids))
            en_used_pieces_ids = list(
                set(train_en_used_pieces_ids + valid_en_used_pieces_ids + test_en_used_pieces_ids))
            logger.info("en_used_pieces_ids:", len(en_used_pieces_ids))
            train_de_used_pieces_ids = get_used_pieces_ids_by_corpus(train_de_file)
            logger.info("train_de_used_pieces_ids:", len(train_de_used_pieces_ids))
            valid_de_used_pieces_ids = get_used_pieces_ids_by_corpus(valid_de_file)
            logger.info("valid_de_used_pieces_ids:", len(valid_de_used_pieces_ids))
            test_de_used_pieces_ids = get_used_pieces_ids_by_corpus(test_de_file)
            logger.info("test_de_used_pieces_ids:", len(test_de_used_pieces_ids))
            de_used_pieces_ids = list(
                set(train_de_used_pieces_ids + valid_de_used_pieces_ids + test_de_used_pieces_ids))
            logger.info("de_used_pieces_ids:", len(de_used_pieces_ids))
            train_used_pieces_ids = list(set(train_en_used_pieces_ids + train_de_used_pieces_ids))
            logger.info("train_used_pieces_ids:", len(train_used_pieces_ids))
            all_used_pieces_ids = list(set(en_used_pieces_ids + de_used_pieces_ids))
            logger.info("all_used_pieces_ids:", len(all_used_pieces_ids))
            all_en_train_de_used_pieces_ids = list(set(en_used_pieces_ids + train_de_used_pieces_ids))
            logger.info("all_en_train_de_used_pieces_ids:", len(all_en_train_de_used_pieces_ids))
            org_ids_to_new_ids = build_org_ids_to_new_ids(all_en_train_de_used_pieces_ids)
            new_ids_to_org_ids = build_new_ids_to_org_ids(all_en_train_de_used_pieces_ids)
            self.org_ids_to_new_ids = org_ids_to_new_ids
            self.new_ids_to_org_ids = new_ids_to_org_ids
            self.unk_token_id = self.org_ids_to_new_ids[tokenizer.unk_token_id]
            self.pad_token_id = self.org_ids_to_new_ids[tokenizer.pad_token_id]
            self.bos_token_id = self.org_ids_to_new_ids[tokenizer.bos_token_id]
            self.eos_token_id = self.org_ids_to_new_ids[tokenizer.eos_token_id]
            en_and_de_ids = tokenizer.convert_tokens_to_ids(["en_XX", "de_DE"])
            if en_and_de_ids != [16004, 16003]:
                raise Exception("incorrect en and de ids")
                exit(0)
            self.en_XX_id = self.org_ids_to_new_ids[en_and_de_ids[0]]
            self.de_DE_id = self.org_ids_to_new_ids[en_and_de_ids[1]]
            self.is_training = True
            ReduceEmbeddingSize.__instance = self

    def set_training(self, is_training):
        self.is_training = is_training


def get_used_pieces_ids_by_corpus(corpus_file_path):
    #print(len(tokenizer.get_added_vocab()))
    #print(tokenizer.all_special_tokens)
    #print(tokenizer.all_special_ids)
    #print(tokenizer.vocab_size+len(tokenizer.all_special_tokens))
    #exit(0)
    used_pieces_ids = [i for i in range(16027)]
    #used_pieces_ids = [0, 1, 2, 3] + tokenizer.convert_tokens_to_ids(["en_XX", "de_DE"])
    #with open(corpus_file_path, "r", encoding="utf-8") as corpus_file:
    #    lines = corpus_file.readlines()
    #    for line in lines:
    #        line = line.strip("\n")
    #        used_pieces_ids += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
    return list(set(used_pieces_ids))


def build_org_ids_to_new_ids(used_pieces_ids):
    used_pieces_ids = sorted(used_pieces_ids)
    org_ids_to_new_ids = {}
    for new_id, old_id in enumerate(used_pieces_ids):
        assert old_id not in org_ids_to_new_ids
        org_ids_to_new_ids[old_id] = new_id
    return org_ids_to_new_ids


def build_new_ids_to_org_ids(used_pieces_ids):
    used_pieces_ids = sorted(used_pieces_ids)
    return used_pieces_ids


def original_ids_to_new_ids(original_input_ids, org_ids_to_new_ids, return_tensor=True):
    new_input_ids = []
    if not isinstance(original_input_ids, list):
        original_input_ids = original_input_ids.tolist()
    for old_input in original_input_ids:
        new_input = []
        for old_id in old_input:
            if old_id in org_ids_to_new_ids:
                new_input.append(org_ids_to_new_ids[old_id])
            else:
                new_input.append(3)
        new_input_ids.append(new_input)
    if return_tensor is True:
        new_input_ids = torch.LongTensor(new_input_ids)
    return new_input_ids


def new_ids_to_original_ids(new_input_ids, new_ids_to_org_ids, return_tensor=True):
    org_input_ids = []
    if not isinstance(new_input_ids, list):
        new_input_ids = new_input_ids.tolist()
    for new_id in new_input_ids:
        org_id = new_ids_to_org_ids[new_id]
        if org_id == tokenizer.eos_token_id:
            break
        org_input_ids.append(org_id)
    if return_tensor is True:
        org_input_ids = torch.LongTensor(org_input_ids)
    return org_input_ids


def main():
    train_en_file = "wmt16_de_en/train.tok.clean.en"
    valid_en_file = "wmt16_de_en/newstest2013.tok.en"
    test_en_file = "wmt16_de_en/newstest2016.tok.en"
    train_de_file = "wmt16_de_en/train.tok.clean.de"
    valid_de_file = "wmt16_de_en/newstest2013.tok.de"
    test_de_file = "wmt16_de_en/newstest2016.tok.de"
    train_en_used_pieces_ids = get_used_pieces_ids_by_corpus(train_en_file)
    print("train_en_used_pieces_ids:", len(train_en_used_pieces_ids))
    valid_en_used_pieces_ids = get_used_pieces_ids_by_corpus(valid_en_file)
    print("valid_en_used_pieces_ids:", len(valid_en_used_pieces_ids))
    test_en_used_pieces_ids = get_used_pieces_ids_by_corpus(test_en_file)
    print("test_en_used_pieces_ids:", len(test_en_used_pieces_ids))
    en_used_pieces_ids = list(set(train_en_used_pieces_ids + valid_en_used_pieces_ids + test_en_used_pieces_ids))
    print("en_used_pieces_ids:", len(en_used_pieces_ids))
    train_de_used_pieces_ids = get_used_pieces_ids_by_corpus(train_de_file)
    print("train_de_used_pieces_ids:", len(train_de_used_pieces_ids))
    valid_de_used_pieces_ids = get_used_pieces_ids_by_corpus(valid_de_file)
    print("valid_de_used_pieces_ids:", len(valid_de_used_pieces_ids))
    test_de_used_pieces_ids = get_used_pieces_ids_by_corpus(test_de_file)
    print("test_de_used_pieces_ids:", len(test_de_used_pieces_ids))
    de_used_pieces_ids = list(set(train_de_used_pieces_ids + valid_de_used_pieces_ids + test_de_used_pieces_ids))
    print("de_used_pieces_ids:", len(de_used_pieces_ids))
    train_used_pieces_ids = list(set(train_en_used_pieces_ids + train_de_used_pieces_ids))
    print("train_used_pieces_ids:", len(train_used_pieces_ids))
    all_used_pieces_ids = list(set(en_used_pieces_ids + de_used_pieces_ids))
    print("all_used_pieces_ids:", len(all_used_pieces_ids))
    org_ids_to_new_ids = build_org_ids_to_new_ids(all_used_pieces_ids)
    new_ids_to_org_ids = build_new_ids_to_org_ids(all_used_pieces_ids)
    return org_ids_to_new_ids, new_ids_to_org_ids


if __name__ == "__main__":
    org_ids_to_new_ids, new_ids_to_org_ids = main()
    with open("./org_ids_to_new_ids.json", "w") as f:
        json.dump(org_ids_to_new_ids, f)
    from onmt.transformers.modeling_mbart import MBartForConditionalGeneration

    model = MBartForConditionalGeneration.from_pretrained("config.json")
    original_wights = model.base_model.shared.weight.data
    extract_indexes = torch.LongTensor(new_ids_to_org_ids)
    new_wights = original_wights[extract_indexes]
    emb_layer = torch.nn.Embedding(len(extract_indexes), 1024, padding_idx=1)
    emb_layer.weights = torch.nn.Parameter(new_wights)
    model.base_model.shared = emb_layer
    model.base_model.encoder.embed_tokens = emb_layer
    model.base_model.decoder.embed_tokens = emb_layer
    print(model)
