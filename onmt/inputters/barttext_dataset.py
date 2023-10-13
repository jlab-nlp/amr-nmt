import six
import json
import torch
import numpy as np
from functools import partial
from torchtext.data import RawField, Pipeline
from torchtext.data.utils import get_tokenizer
from onmt.transformers.tokenization_mbart import MBartTokenizer
from onmt.transformers.modeling_bart import shift_tokens_right
from onmt.transformers.configuration_mbart import MBartConfig
from onmt.inputters.datareader_base import DataReaderBase
from torchsummaryX import summary
import reduce_embeding_size

class BartTextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read bart data from disk.

        Args:
            sequences (dict):
                path to src and tgt.
            side (str): Prefix used in return dict. Usually
                ``"src"`` , ``"tgt" or "grh``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        # assert _dir is not None or _dir != "", \
        #     "Must use _dir with GrhDataReader (provide edges vocab)."
        src_sequences = sequences["src"]
        tgt_sequences = sequences["tgt"]
        if isinstance(src_sequences, str):
            src_sequences = DataReaderBase._read_file(src_sequences)
        if isinstance(tgt_sequences, str):
            src_sequences = DataReaderBase._read_file(tgt_sequences)
        # vocab = json.load(_dir)
        for i, (src_seq, tgt_seq) in enumerate(zip(src_sequences, tgt_sequences)):
            if isinstance(src_seq, six.binary_type):
                src_seq = src_seq.decode("utf-8")
            if isinstance(tgt_seq, six.binary_type):
                tgt_seq = tgt_seq.decode("utf-8")
            yield {side: {"src": src_seq, "tgt": tgt_seq}, "indices": i}


class BartTextField(RawField):
    """ custom field.

    We need custom field for BART input features.

    Notice that here we dont implement multi-shards.
    """

    def __init__(self, bart_config, src_lang="en_XX", tgt_lang="ro_RO"):
        super(BartTextField, self).__init__()
        self.bart_config = bart_config
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = None

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        return x

    def process(self, batch_examples, device=None):
        """ Process a list of examples to create a torch.Tensor.
            Graph information is in the form of an adjacency list.
            We convert this to an adjacency matrix in NumPy format.
            The matrix contains label ids.
            IMPORTANT: we add one to the label id stored in the matrix.
            This is because 0 is a valid vocab id but we want to use 0's
            to represent lack of edges instead. This means that the GCN code
            should account for this.
            But it is better to be defined in postprocessing
        """

        def original_ids_to_new_ids(original_input_ids, res):
            new_input_ids = []
            if not isinstance(original_input_ids, list):
                original_input_ids = original_input_ids.tolist()
            for old_input in original_input_ids:
                new_input = []
                for old_id in old_input:
                    if res.is_training:
                        new_input.append(res.org_ids_to_new_ids[old_id])
                    else:
                        if old_id in res.org_ids_to_new_ids:
                            new_input.append(res.org_ids_to_new_ids[old_id])
                        else:
                            new_input.append(res.unk_token_id)
                new_input_ids.append(new_input)
            new_input_ids = torch.LongTensor(new_input_ids)
            return new_input_ids
        src_batch = [example["src"] for example in batch_examples]
        tgt_batch = [example["tgt"] for example in batch_examples]
        if self.tokenizer is None:
            self.tokenizer = MBartTokenizer(vocab_file="sentencepiece.bpe.model")
        output_arrs = self.tokenizer.prepare_seq2seq_batch(src_batch, src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                                                           tgt_texts=tgt_batch, return_attention_mask=True,
                                                           max_length=50, max_target_length=50)
        batch_input_ids = output_arrs["input_ids"]
        batch_labels = output_arrs["labels"]
        batch_decoder_input_ids = shift_tokens_right(batch_labels, self.tokenizer.pad_token_id)
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        if res.org_ids_to_new_ids:
            batch_input_ids = original_ids_to_new_ids(batch_input_ids, res)
            batch_labels = original_ids_to_new_ids(batch_labels, res)
            batch_decoder_input_ids = original_ids_to_new_ids(batch_decoder_input_ids, res)
        return {"input_ids": batch_input_ids.to(device), "labels": batch_labels.to(device),
                "decoder_input_ids": batch_decoder_input_ids.to(device), "input_src": src_batch}


def barttext_fields(**kwargs):
    """Create bart text fields."""
    bart_tokenizer = kwargs.get("bart_config")
    src_lang = kwargs.get("src_lang")
    tgt_lang = kwargs.get("tgt_lang")
    feat = BartTextField(bart_config=bart_tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
    return feat


if __name__ == "__main__":
    from onmt.transformers.tokenization_mbart import MBartTokenizer
    from onmt.transformers.modeling_mbart import MBartForConditionalGeneration
    import torch

    tokenizer = MBartTokenizer(vocab_file="sentencepiece.bpe.model")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
    # from onmt.transformers.modeling_mbart import MBartForConditionalGeneration
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    tokens = tokenizer.tokenize("Wenn man bedenkt, dass die meisten asiatischen Staats- und Regierungschefs der Meinung sind, dass vor allem die Uneinigkeit Asiens den Westen übermächtig werden ließ, ist es nicht verwunderlich, dass die meisten dieser Einigkeitsappelle mit verständlicher Zurückhaltung aufgenommen werden.".lower())
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    vocabs = tokenizer.get_vocab()
    print(tokens)
    print(input_ids)
    print(tokenizer.convert_tokens_to_ids(["en_XX", "de_DE"]))
    # vocab = tokenizer.get_vocab()
    # tokenizer.save_vocabulary("/Users/changmaoli/Research/amrintomt/GraphToSequence/HetGT-master/")

    print(tokenizer.all_special_ids)
    print(tokenizer.all_special_ids)
    print(tokenizer.convert_ids_to_tokens(0))
    print(tokenizer.convert_ids_to_tokens(1))
    print(tokenizer.convert_ids_to_tokens(2))
    print(tokenizer.convert_ids_to_tokens(3))

    # example_english_phrase = ["UN Chief Says There Is No Military Solution in Syria"]
    # extract_indexes = torch.LongTensor([i for i in range(6, 250000, 2)])
    # original_wights = model.base_model.shared.weight.data
    # new_wights = original_wights[extract_indexes]
    # expected_translation_romanian = ["Şeful ONU declară că nu există o soluţie militară în Siria"]
    # emb_layer = torch.nn.Embedding(len(extract_indexes), 1024, padding_idx=1)
    # emb_layer.weights = torch.nn.Parameter(new_wights)
    # model.base_model.shared = emb_layer
    # model.base_model.encoder.embed_tokens = emb_layer
    # model.base_model.decoder.embed_tokens = emb_layer
    example_english_phrase = ["That partnership has five components: wider opportunities for education in order to produce a workforce with cutting-edge skills; investment in infrastructure – roads, power plants, and ports – that supports commerce; funds for research and development to expand the frontiers of knowledge in ways that generate new products; an immigration policy that attracts and retains talented people from beyond America’s borders; and business regulations strong enough to prevent disasters such as the near-meltdown of the financial system in 2008 but not so stringent as to stifle the risk-taking and innovation that produce growth."]
    example_german_phrase = ["Diese Partnerschaft umfasst fünf Komponenten: umfassendere Ausbildungsmöglichkeiten, um eine Erwerbsbevölkerung mit Spitzenfertigkeiten hervorzubringen; Investitionen in Infrastruktur – Straßen, Kraftwerke und Häfen –, die den Handel unterstützt; Mittel für Forschung und Entwicklung, um die Grenzen des Wissens auf Weisen auszuweiten, die neue Produkte hervorbringen; eine Einwanderungspolitik, die talentierte Menschen von außerhalb der US-Grenzen anlockt und im Lande hält; und eine wirtschaftliche Regulierung, die ausreichend stark ist, um Katastrophen wie den Beinahe-GAU des Finanzsystems 2008 zu verhindern, aber nicht so stringent, dass sie jene Risikobereitschaft und Innovation erstickt, die Wachstum hervorbringt."]
    batch = tokenizer.prepare_seq2seq_batch(example_english_phrase, src_lang="en_XX", tgt_lang="de_DE",
                                            tgt_texts=example_german_phrase,
                                            return_attention_mask=True)
    batch_labels = batch["labels"]
    decoder_input_ids = shift_tokens_right(batch_labels, tokenizer.pad_token_id)
    print(decoder_input_ids.size())
    print(decoder_input_ids)
    # print(batch["input_ids"])
    # print(batch["input_ids"].size())
    # print(batch["input_ids"].dtype)
    # print(batch)
    translated = [[5533, 198, 3840, 4, 1225, 198, 14450, 72799, 5, 2, 250003, 5533, 32, 32, 32, 32, 32, 5, 32, 32, 32, 32, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 32, 32, 32, 5, 32, 32, 32, 32, 5, 5, 5, 5, 2, 2, 2, 5]]
    # print(batch)
    # input_ids = batch["input_ids"]
    # labels = batch["labels"]
    # decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id)
    # # model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
    # # print(model)
    # print(input_ids)
    # print(decoder_input_ids)
    # print(labels)
