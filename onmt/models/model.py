""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
import copy
import math
from onmt.transformers.modeling_mbart import MBartForConditionalGeneration, MBartConfig
from onmt.transformers.modeling_bart import (
    shift_tokens_right,
    BART_INPUTS_DOCSTRING,
    Seq2SeqLMOutput,
    _CONFIG_FOR_DOC,
    BART_GENERATION_EXAMPLE
)
from torch.nn import CrossEntropyLoss, CTCLoss
import torch.nn.functional as F
from graph4nlp.pytorch.modules.graph_construction.embedding_construction import WordEmbedding
from graph4nlp.pytorch.modules.prediction.generation.decoder_strategy import DecoderStrategy
from graph4nlp.pytorch.modules.prediction.generation.StdRNNDecoder import StdRNNDecoder

from graph4nlp.pytorch.models.base import Graph2XBase
import warnings
from onmt.transformers.file_utils import (
    add_end_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)

import torch.utils.checkpoint as checkpoint
import reduce_embeding_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, grh=None, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        # now tgt is (tgt_len, batch) so the last tgt maybe </end>
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)


class TransformerModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig):
        super().__init__(config)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }


class AMRBARTModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        new_memory_bank = torch.cat((bart_encoder_last_hidden_state, memory_bank), 1)
        #
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape((decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        #final_decode_output = bart_decoder_last_hidden_state + dec_out[:,-1,:].unsqueeze(dim=1)
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }




class AMRBARTSingleDecoderModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        new_memory_bank = torch.cat((bart_encoder_last_hidden_state, memory_bank), 1)
        #
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape((decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        # bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }




class AMRBARTSeperateModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        if bptt is False and step is None:
            self.decoder.init_state(src, memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape((decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }



class AMRBARTLearnedEmbeddingModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.type_embedding = torch.nn.Embedding(2, 512)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        new_memory_bank = torch.cat((bart_encoder_last_hidden_state, memory_bank), 1)
        #
        bart_encoder_last_hidden_state_encoding_ids = torch.zeros((bart_encoder_last_hidden_state.size(0), bart_encoder_last_hidden_state.size(1)), dtype=torch.long, device=memory_bank.get_device())
        memory_bank_encoding_ids = torch.ones((memory_bank.size(0), memory_bank.size(1)), dtype=torch.long, device=memory_bank.get_device())
        new_memory_bank_encoding_ids = torch.cat((bart_encoder_last_hidden_state_encoding_ids, memory_bank_encoding_ids), 1)
        new_memory_bank_encoding = self.type_embedding(new_memory_bank_encoding_ids)
        new_memory_bank = new_memory_bank + new_memory_bank_encoding
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape(
            (decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }


class AMRBARTAttentionModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None, attention_drop=0.1):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.mha = MultiHeadAttention(256, head_num=2, drop=attention_drop)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        g_to_s_att = self.mha(memory_bank, bart_encoder_last_hidden_state, bart_encoder_last_hidden_state)
        s_to_g_att = self.mha(bart_encoder_last_hidden_state, memory_bank, memory_bank)
        new_memory_bank = torch.cat((g_to_s_att, s_to_g_att), 1)
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape(
            (decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }



class AMRBARTAttentionSingleDecoderModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None, attention_drop=0.1):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.mha = MultiHeadAttention(512, head_num=2, drop=attention_drop)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        g_to_s_att = self.mha(memory_bank, bart_encoder_last_hidden_state, bart_encoder_last_hidden_state)
        s_to_g_att = self.mha(bart_encoder_last_hidden_state, memory_bank, memory_bank)
        new_memory_bank = torch.cat((g_to_s_att, s_to_g_att), 1)
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape(
            (decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        # bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }


class AMRBARTAttentionAddModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.mha = MultiHeadAttention(256, head_num=2)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")

        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        g_to_s_att = self.mha(memory_bank, bart_encoder_last_hidden_state, bart_encoder_last_hidden_state)
        s_to_g_att = self.mha(bart_encoder_last_hidden_state, memory_bank, memory_bank)
        new_memory_bank = torch.cat((g_to_s_att + memory_bank, s_to_g_att + bart_encoder_last_hidden_state), 1)
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        # lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
        #                                size=(bart_encoder_last_hidden_state.size()[0],),
        #                                device=memory_bank.get_device())
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],))
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape(
            (decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out = self.decoder(tgt, new_memory_bank, lengths, step=step)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu,
                 drop=0.1):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        y = self.dropout(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class AMRBARTNONAUTOModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, encoder=None, decoder=None):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        is_training = res.is_training
        bart_encoder_last_hidden_state = outputs.encoder_last_hidden_state
        # if past_key_values is not None:
        #     print("past_key_values:", len(past_key_values))
        #     print("past_key_values[0]:", past_key_values[0])
        # else:
        #     print("past_key_values: None")
        bart_decoder_hidden_states = list(outputs.decoder_hidden_states)
        # print("bart_encoder_last_hidden_state:", bart_encoder_last_hidden_state.size())
        bart_decoder_last_hidden_state = outputs.last_hidden_state
        # print("bart_decoder_last_hidden_state:", bart_decoder_last_hidden_state.size())
        # print(src.size())
        # print(grh.size())
        # print(lengths.size())
        enc_state, memory_bank, lengths = self.encoder(src, grh, lengths)
        # print("memory_bank:", memory_bank.size())
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        new_memory_bank = torch.cat((bart_encoder_last_hidden_state, memory_bank), 1)
        new_memory_bank = new_memory_bank.transpose(0, 1).contiguous()
        lengths = lengths + torch.full(dtype=torch.int64, fill_value=bart_encoder_last_hidden_state.size()[1],
                                       size=(bart_encoder_last_hidden_state.size()[0],),
                                       device=memory_bank.get_device())
        # print("new_memory_bank:", new_memory_bank.size())
        if bptt is False and step is None:
            self.decoder.init_state(src, new_memory_bank, enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape((decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:
        dec_out, dec_hidden_states = self.decoder(tgt, new_memory_bank, lengths, step=step, output_hidden_states=True)
        # print("dec_out:", dec_out.size())
        dec_out = dec_out.transpose(0, 1).contiguous()
        bart_decoder_last_hidden_state = bart_decoder_last_hidden_state.contiguous()
        final_decode_output = bart_decoder_last_hidden_state + dec_out
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.model.shared.weight, self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            # Original CrossEntropy Loss
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # Intermediate layer Alignment Loss
            for bart_decoder_hidden_state, dec_hidden_state in zip(bart_decoder_hidden_states, dec_hidden_states):
                dec_hidden_state = dec_hidden_state.transpose(0, 1).contiguous()
                bart_decoder_hidden_state = bart_decoder_hidden_state.contiguous()
                alignment_output = bart_decoder_hidden_state + dec_hidden_state
                alignment_logits = F.linear(alignment_output, self.model.shared.weight, self.final_logits_bias)
                masked_lm_loss = masked_lm_loss + loss_fct(alignment_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # CTC Loss
            loss_fct = CTCLoss(blank=res.pad_token_id)
            lm_logits_lengths = input_ids.ne(res.pad_token_id).long().sum(1)
            labels_lengths = labels.ne(res.pad_token_id).long().sum(1)
            log_logits = lm_logits.log_softmax(-1).transpose(0, 1)
            labels = labels.long()
            masked_lm_loss = masked_lm_loss + loss_fct(log_logits, labels, lm_logits_lengths, labels_lengths)

        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }


class LSTMAMRModel(MBartForConditionalGeneration):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, config: MBartConfig, rnn_encoder=None, gnn_encoder=None, rnn_decoder=None, gnn_decoder=None):
        super().__init__(config)
        self.rnn_encoder = rnn_encoder
        self.gnn_encoder = gnn_encoder
        self.rnn_decoder = rnn_decoder
        self.gnn_decoder = gnn_decoder


    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None,
            tgt=None,
            lengths=None,
            grh=None,
            bptt=False,
            step=None,
            **unused,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
                with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

                # Mask filling only works for bart-large
                from transformers import BartTokenizer, BartForConditionalGeneration
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
                TXT = "My friends are <mask> but they eat too many carbs."

                model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
                input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
                logits = model(input_ids).logits

                masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
                probs = logits[0, masked_index].softmax(dim=0)
                values, predictions = probs.topk(5)

                tokenizer.decode(predictions).split()
                # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO: change input_ids into src type
        encoder_input_ids = torch.transpose(input_ids, 0, 1).reshape((input_ids.size()[1], input_ids.size()[0], 1))
        text_enc_state, text_memory_bank, text_lengths = self.rnn_encoder(encoder_input_ids)
        graph_enc_state, graph_memory_bank, graph_lengths = self.gnn_encoder(src, grh, lengths)
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)
        res = reduce_embeding_size.ReduceEmbeddingSize.get_instance()
        #bptt = False
        #if bptt is False and step is None:
        self.rnn_decoder.init_state(src, text_memory_bank, text_enc_state)
        self.gnn_decoder.init_state(src, text_memory_bank, text_enc_state)
        tgt = torch.transpose(decoder_input_ids, 0, 1).reshape((decoder_input_ids.size()[1], decoder_input_ids.size()[0], 1))
        # print("tgt:", tgt.size())
        # if is_training:
        #     dec_out = checkpoint.checkpoint(self.decoder, tgt, new_memory_bank, lengths)
        # else:

        tgt_dec_out = self.rnn_decoder(tgt, text_memory_bank, text_lengths, step=step)
        grn_dec_out = self.gnn_decoder(tgt, graph_memory_bank, graph_lengths, step=step)
        # print("dec_out:", dec_out.size())
        final_decode_output = tgt_dec_out[0].transpose(0, 1).contiguous() + grn_dec_out[0].transpose(0, 1).contiguous()
        # if is_training:
        #   lm_logits = checkpoint.checkpoint(F.linear, final_decode_output, self.model.shared.weight,
        #                                      self.final_logits_bias)
        # else:
        #
        lm_logits = F.linear(final_decode_output, self.rnn_encoder.embeddings.make_embedding.emb_luts.embedding.weight,
                             torch.zeros((1, self.rnn_encoder.embeddings.make_embedding.emb_luts.embedding.num_embeddings), device=final_decode_output.get_device()))
        # seq_logits = F.linear(bart_decoder_last_hidden_state, self.model.shared.weight, self.final_logits_bias)
        # grh_logits = F.linear(dec_out, self.model.shared.weight, self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=res.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(seq_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))
            # masked_lm_loss += loss_fct(grh_logits.view(-1, len(res.new_ids_to_org_ids)), labels.view(-1))

            # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits
        )

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        beam_size = kwargs["beam_size"]
        device = kwargs["device"]
        src = kwargs["src"]
        grh = kwargs["grh"]
        src_size_zero = src.size(0)
        src_size_dim = src.size(-1)
        lengths = kwargs["lengths"]
        batch_size = grh.size(0)
        grh_row = grh.size(-2)
        grh_column = grh.size(-1)
        input_src = kwargs["input_src"]
        input_length = input_src.size(-1)
        assert batch_size == src.size(1)
        if beam_size > 1:
            input_src = input_src.unsqueeze(1).expand(batch_size, beam_size, input_length)
            input_src = input_src.contiguous().view(batch_size*beam_size, input_length)
            src = src.unsqueeze(2).expand(src_size_zero, batch_size, beam_size, src_size_dim)
            grh = grh.unsqueeze(1).expand(batch_size, beam_size, grh_row, grh_column)
            lengths = lengths.unsqueeze(1).expand(batch_size, beam_size)
            src = src.contiguous().view(src_size_zero, batch_size*beam_size, src_size_dim)
            grh = grh.contiguous().view(batch_size*beam_size, grh_row, grh_column)
            lengths = lengths.contiguous().view(batch_size*beam_size)

        return {
            "input_ids": input_src,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "src": src.to(device),
            "grh": grh.to(device),
            "lengths": lengths.to(device)
        }



if __name__ == "__main__":
    model = AMRBARTModel
    pass
    # TODO: Fake Input data: 1. Fake AMR graph input data, 2. Fake text input data

    # TODO: Processed input data: 1. process AMR graph input data, 2. process text input data

    # TODO: Build Model

    # TODO: Input data into model

    # TODO: Output model details
