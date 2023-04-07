import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CustomGraformer(nn.Module):
    def __init__(self, masked_encoder:nn.Module|str, causal_decoder: nn.Module|str, 
                 d_model:int=512, n_heads=8, dff=2048, n_encoder_layers=6, n_decoder_layers=6,
                 layer_norm=1e-5, dropout=.1, activation=F.gelu, encoder_tokenizer=None, decoder_tokenizer=None,
                 *args, **kwargs) -> None:
        """
        Implementation of Graformer, as per the article
        "Multilingual Translation via Grafting Pre-trained Language Models", Zewei et al.,
        Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2735–2747, November 7–11, 2021

        Parameters:
        :param masked_encoder: the masked LM encoder, mBERT in the original paper. A string for the name of the model
            as per Hugging Face's naming rules, or the model inherited by the Hugging Face's PreTrainedModel.
        :param causal_decoder: the causal decoder, mGPT in the original paper. A string for the name of the model
            as per Hugging Face's naming rules, or the model inherited by the Hugging Face's PreTrainedModel.

        For the additional K-layer encoder-decoder stack:
        :param d_model: dimension of the model, an int
        :param n_heads: number of attention heads
        :param dff: dimension of the feed-forward network
        :param n_encoder_layers: number of encoder layers
        :param n_decoder_layers: number of decoder layers

        General parameters for the additional stacks:
        :param layer_norm: Layer normalization hyperparameter
        :param dropout: Rate of dropout
        :param activation: Activation function used, default to gelu

        General parameters for the entire model:
        :param encoder_tokenizer: Tokenizer used for the encoder. If `None`, `masked_encoder` must be `str`.
        :param decoder_tokenizer: Tokenizer used for the decoder. If `None`, `causal_decoder` must be `str`.
        """

        super().__init__(*args, **kwargs)

        if isinstance(masked_encoder, str):
            self.masked_encoder = AutoModel.from_pretrained(masked_encoder)
        else:
            self.masked_encoder = masked_encoder

        if isinstance(causal_decoder, str):
            self.causal_decoder = AutoModel.from_pretrained(causal_decoder)  # Model head included.
        else:
            self.causal_decoder = causal_decoder

        # Freeze the causal decoder
        self.causal_decoder.requires_grad_(False)

        self.k_layer_encoder_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dff, dropout, activation, layer_norm_eps=layer_norm),
            # norm = nn.LayerNorm(self.masked_encoder.config.vocab_size, layer_norm),
            num_layers=6#n_encoder_layers
        )

        self.k_layer_decoder_stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dff, dropout, activation, layer_norm_eps=layer_norm),
            norm=nn.LayerNorm(self.causal_decoder.config.vocab_size, layer_norm),
            num_layers=n_decoder_layers
        )

        # HuggingFace will raise errors itself if both tokenizer and model are None.
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(masked_encoder) \
            if encoder_tokenizer is None else encoder_tokenizer
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(causal_decoder) \
            if decoder_tokenizer is None else decoder_tokenizer

    def forward(self, source):
        encoder_tokens = self.encoder_tokenizer(source, return_tensors='pt', padding='max_length', max_length=128)
        encoder_input_ids, encoder_attention_mask = encoder_tokens['input_ids'], encoder_tokens['attention_mask']
        masked_encoder_output = self.masked_encoder(encoder_input_ids, encoder_attention_mask).pooler_output

        decoder_tokens = self.decoder_tokenizer(source, return_tensors='pt', padding='max_length', max_length=128)
        decoder_input_ids, decoder_attention_mask = decoder_tokens['input_ids'], decoder_tokens['attention_mask']
        causal_decoder_output = self.causal_decoder(decoder_input_ids, decoder_attention_mask).last_hidden_state





    def freeze_decoder(self):
        for param in self.decoder.parameters(): param.requires_grad=False
    
    def freeze_encoder(self):
        for param in self.encoder.parameters(): param.requires_grad=False
    
    # def generate_square_subsequent_mask(self, sz, device='cpu'):
    #     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask


    # def create_mask(self, src, tgt, device='cpu'):
    #     src_seq_len = src.shape[0]
    #     tgt_seq_len = tgt.shape[0]

    #     tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
    #     src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    #     src_padding_mask = (src == self.encoder_tokenizer.pad_token_id).transpose(0, 1)
    #     tgt_padding_mask = (tgt == self.decoder_tokenizer.pad_token_id).transpose(0, 1)
    #     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def greedy_decode(self, src, src_mask, max_len, start_symbol, device='cpu'):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.decoder_tokenizer.eos_token_id:
                break
        return ys
    
